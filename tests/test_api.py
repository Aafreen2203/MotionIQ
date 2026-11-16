import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import motioniq.api as api_module
from motioniq.models import make_random_forest
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
    monkeypatch.setattr(api_module, "MODEL_CHECKPOINT", tmp_path / "missing.joblib")
    monkeypatch.setattr(api_module, "CLASSIFIER_BUNDLE", None)
    monkeypatch.setattr(api_module, "CLASSIFIER_LOAD_ERROR", None)
    monkeypatch.setattr(api_module, "CLASSIFIER_MTIME", None)

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


def test_processed_video_endpoint_serves_h264_with_byte_ranges(tmp_path):
    source = tmp_path / "range-source.mp4"
    artifact = tmp_path / "range-source.pose.mp4"
    writer = cv2.VideoWriter(str(artifact), cv2.VideoWriter_fourcc(*"avc1"), 5, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV H.264 encoder unavailable in this environment")
    for value in range(5):
        writer.write(np.full((48, 64, 3), value * 30, dtype=np.uint8))
    writer.release()
    video_id = "range-video-test"
    api_module.RESULTS[video_id] = {
        "video": {"path": str(source)},
        "processed_video": {"status": "completed", "url": f"/api/videos/{video_id}/processed"},
    }
    try:
        response = client.get(f"/api/videos/{video_id}/processed")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("video/mp4")
        assert len(response.content) == artifact.stat().st_size
        capture = cv2.VideoCapture(str(artifact))
        assert capture.isOpened()
        assert capture.read()[0]
        capture.release()
        ranged = client.get(f"/api/videos/{video_id}/processed", headers={"Range": "bytes=0-1023"})
        assert ranged.status_code == 206
        assert ranged.headers["content-range"].startswith("bytes 0-1023/")
        assert len(ranged.content) == 1024
    finally:
        api_module.RESULTS.pop(video_id, None)


def test_upload_does_not_advertise_missing_processed_video(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path)

    def analysis_result(path):
        return {
            "video": {"path": str(path), "width": 64, "height": 48, "fps": 5},
            "segmentation_masks": {},
            "pose": {"status": "completed", "frames": [], "detected_frames": 0},
            "segmentation": {"status": "completed"},
            "overlay": {"status": "available", "url": None},
            "processed_video": {"status": "completed", "url": None},
            "segmentation_video": {"status": "completed", "url": None},
            "movement_measurements": {"status": "not_configured"},
            "prediction": {"status": "not_configured"},
            "evaluation": {"status": "not_evaluated"},
        }

    monkeypatch.setattr(api_module, "analyze_video", analysis_result)
    response = client.post("/api/videos", files={"file": ("clip.mp4", b"video", "video/mp4")})
    assert response.status_code == 200
    body = response.json()
    assert body["processed_video"]["status"] == "failed"
    assert body["processed_video"]["url"] is None
    assert body["segmentation_video"]["status"] == "failed"
    assert body["segmentation_video"]["url"] is None


def test_model_status_is_honest_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "MODEL_CHECKPOINT", tmp_path / "missing.joblib")
    monkeypatch.setattr(api_module, "CLASSIFIER_BUNDLE", None)
    monkeypatch.setattr(api_module, "CLASSIFIER_LOAD_ERROR", None)
    monkeypatch.setattr(api_module, "CLASSIFIER_MTIME", None)
    response = client.get("/api/model/status")
    assert response.status_code == 200
    assert response.json()["status"] == "not_configured"
    assert "No trained checkpoint" in response.json()["reason"]


def test_upload_uses_configured_classifier_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "UPLOAD_DIR", tmp_path)
    model = make_random_forest(seed=9, n_estimators=50, min_samples_leaf=1)
    model.fit([[0.0], [2.0], [98.0], [100.0]], ["squat", "squat", "lunge", "lunge"])
    monkeypatch.setattr(
        api_module,
        "CLASSIFIER_BUNDLE",
        {
            "model": model,
            "feature_names": ["left_knee_flexion_mean"],
            "class_labels": ["lunge", "squat"],
            "feature_schema_version": "motioniq-temporal-v1",
            "model_version": "test-rf",
            "metadata": {"confidence_threshold": 0.5, "held_out_test_metrics": None},
        },
    )

    def analyze(path):
        return {
            "video": {"path": str(path), "width": 320, "height": 240, "fps": 2},
            "segmentation_masks": {},
            "pose": {"status": "completed", "frames": [], "detected_frames": 0},
            "segmentation": {"status": "not_configured"},
            "overlay": {"status": "not_configured", "url": None},
            "processed_video": {"status": "not_configured", "url": None},
            "segmentation_video": {"status": "not_configured", "url": None},
            "movement_measurements": {
                "status": "completed",
                "frames": [
                    {"timestamp_seconds": 0.0, "left_knee_flexion_deg": 0.0},
                    {"timestamp_seconds": 1.0, "left_knee_flexion_deg": 2.0},
                ],
            },
            "prediction": {"status": "not_configured", "reason": "No model"},
            "evaluation": {"status": "not_evaluated"},
        }

    monkeypatch.setattr(api_module, "analyze_video", analyze)
    response = client.post("/api/videos", files={"file": ("clip.mp4", b"video", "video/mp4")})
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction["status"] == "completed"
    assert prediction["predicted_class"] == "squat"
    assert prediction["model_version"] == "test-rf"
    assert set(prediction["probabilities"]) == {"squat", "lunge"}
