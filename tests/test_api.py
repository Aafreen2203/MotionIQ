import numpy as np
from fastapi.testclient import TestClient

import motioniq.api as api_module
from motioniq.video import create_synthetic_video

client = TestClient(api_module.app)


def test_cors_preflight_allows_both_5175_origins():
    for origin in ("http://localhost:5175", "http://127.0.0.1:5175"):
        response = client.options(
            "/api/videos",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert "POST" in response.headers["access-control-allow-methods"]
        assert "GET" in response.headers["access-control-allow-methods"]
        assert "authorization" in response.headers["access-control-allow-headers"].lower()
        assert "access-control-allow-credentials" not in response.headers


def test_videos_endpoint_handles_missing_and_empty_library(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(api_module, "UPLOAD_DIR", missing)
    response = client.get("/api/videos", headers={"Origin": "http://localhost:5175"})
    assert response.status_code == 200
    assert response.json() == {"videos": [], "count": 0}
    assert response.headers["access-control-allow-origin"] == "http://localhost:5175"

    missing.mkdir()
    response = client.get("/api/videos")
    assert response.status_code == 200
    assert response.json() == {"videos": [], "count": 0}


def test_videos_endpoint_lists_metadata_without_local_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path)
    create_synthetic_video(tmp_path / "sample.mp4", frames=8, fps=8)

    response = client.get("/api/videos")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["videos"][0]["video_id"] == "sample"
    assert body["videos"][0]["video"]["frame_count"] == 8
    assert "path" not in body["videos"][0]["video"]


def test_cors_header_is_present_on_unhandled_server_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path)

    def fail_analysis(_path):
        raise RuntimeError("test-only failure")

    monkeypatch.setattr(api_module, "analyze_video", fail_analysis)
    test_client = TestClient(api_module.app, raise_server_exceptions=False)
    response = test_client.post(
        "/api/videos",
        files={"file": ("sample.mp4", b"test bytes", "video/mp4")},
        headers={"Origin": "http://localhost:5175"},
    )

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "http://localhost:5175"


def test_upload_and_result_retrieval_keeps_analysis_statuses(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path)

    def analyze(path):
        return {
            "video": {
                "path": str(path),
                "width": 320,
                "height": 240,
                "fps": 12,
                "frame_count": 12,
                "duration_seconds": 1.0,
                "size_bytes": 1,
            },
            "segmentation_masks": {"3": np.zeros((2, 2), dtype=np.uint8)},
            "pose": {"status": "not_configured", "reason": "model missing", "frames": []},
            "segmentation": {"status": "not_configured", "reason": "model missing"},
            "overlay": {"status": "not_configured", "url": None},
            "processed_video": {"status": "not_configured", "url": None},
            "segmentation_video": {"status": "not_configured", "url": None},
            "movement_measurements": {"status": "not_configured"},
            "prediction": {"status": "not_configured"},
            "evaluation": {"status": "not_evaluated"},
        }

    monkeypatch.setattr(api_module, "analyze_video", analyze)
    response = client.post("/api/videos", files={"file": ("clip.mp4", b"video", "video/mp4")})
    assert response.status_code == 200
    body = response.json()
    assert "segmentation_masks" not in body
    assert "3" in api_module.SEGMENTATION_MASKS[body["video_id"]]
    result = client.get(f"/api/videos/{body['video_id']}")
    assert result.status_code == 200
    assert result.json()["pose"]["status"] == "not_configured"
    assert result.json()["evaluation"]["status"] == "not_evaluated"


def test_pose_evaluation_requires_and_scores_ground_truth():
    video_id = "pose-eval-test"
    landmarks = [{"x": 0.25, "y": 0.5} for _ in range(17)]
    api_module.RESULTS[video_id] = {
        "pose": {"frames": [{"frame": 3, "landmarks": landmarks}]},
        "evaluation": {"status": "not_evaluated"},
    }
    try:
        response = client.post(
            f"/api/videos/{video_id}/evaluation/pose",
            json={"frames": [{"frame": 3, "landmarks": [[0.25, 0.5] for _ in range(17)]}]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["evaluated_frames"] == 1
        assert response.json()["mpjpe"] == 0
        assert response.json()["pck"] == 1
    finally:
        api_module.RESULTS.pop(video_id, None)


def test_segmentation_evaluation_requires_and_scores_ground_truth():
    video_id = "seg-eval-test"
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    api_module.RESULTS[video_id] = {"video": {"height": 2, "width": 2}}
    api_module.SEGMENTATION_MASKS[video_id] = {"1": mask}
    try:
        response = client.post(
            f"/api/videos/{video_id}/evaluation/segmentation",
            json={"frames": [{"frame": 1, "mask": mask.tolist()}]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["mean_iou"] == 1
        assert response.json()["mean_dice"] == 1
    finally:
        api_module.RESULTS.pop(video_id, None)
        api_module.SEGMENTATION_MASKS.pop(video_id, None)


def test_upload_rejects_unsupported_and_corrupt_files(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path)
    unsupported = client.post("/api/videos", files={"file": ("clip.txt", b"x", "text/plain")})
    assert unsupported.status_code == 415
    corrupt = client.post("/api/videos", files={"file": ("clip.mp4", b"not a video", "video/mp4")})
    assert corrupt.status_code == 422
    assert list(tmp_path.iterdir()) == []
