import builtins

import cv2
import numpy as np
import pytest

from motioniq import analysis
from motioniq.analysis import _measure
from motioniq.video import create_synthetic_video
from motioniq.vision import HumanSegmenter, PoseEstimator


def _landmarks():
    points = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0, "pixel_x": 160.0, "pixel_y": 120.0}
        for _ in range(17)
    ]
    coords = {
        5: (0.4, 0.2),
        6: (0.6, 0.2),
        11: (0.43, 0.5),
        12: (0.57, 0.5),
        13: (0.4, 0.65),
        14: (0.6, 0.65),
        15: (0.4, 0.8),
        16: (0.6, 0.8),
    }
    for index, (x, y) in coords.items():
        points[index].update(x=x, y=y, pixel_x=x * 320, pixel_y=y * 240)
    return points


def _known_measurement_frame():
    landmarks = [
        {"x": 0.5, "y": 0.5, "pixel_x": 500, "pixel_y": 250, "visibility": 1.0} for _ in range(17)
    ]
    coordinates = {
        5: (150, 50),
        6: (250, 50),
        11: (100, 100),
        12: (300, 100),
        13: (100, 200),
        14: (300, 200),
        15: (100, 300),
        16: (400, 200),
    }
    for index, (pixel_x, pixel_y) in coordinates.items():
        landmarks[index].update(x=pixel_x / 1000, y=pixel_y / 500, pixel_x=pixel_x, pixel_y=pixel_y)
    return {"frame": 0, "timestamp_seconds": 0.0, "landmarks": landmarks}


class FakePose:
    status = "completed"
    unavailable_reason = None
    model = object()

    def process(self, _frame, _timestamp):
        return {"status": "completed", "detection": "person", "landmarks": _landmarks()}

    def close(self):
        pass


class FakeSegmenter:
    status = "completed"
    reason = None
    model = object()

    def process(self, frame):
        return {"status": "completed", "mask": np.ones(frame.shape[:2], dtype=np.float32)}

    def close(self):
        pass


def test_analysis_samples_timestamps_artifacts_and_measurements(tmp_path, monkeypatch):
    video = create_synthetic_video(tmp_path / "source.mp4", frames=24, fps=12)
    monkeypatch.setattr(analysis, "PoseEstimator", FakePose)
    monkeypatch.setattr(analysis, "HumanSegmenter", FakeSegmenter)
    result = analysis.analyze_video(video, sample_fps=2)

    assert result["video"]["width"] == 320
    assert result["video"]["height"] == 240
    assert result["video"]["fps"] == 12
    assert result["sampling"]["sampled_frames"] == 4
    assert result["sampling"]["timestamps_seconds"] == [0.0, 0.5, 1.0, 1.5]
    assert result["pose"]["status"] == "completed"
    assert result["pose"]["frames"][0]["landmarks"][13]["pixel_x"] == 128
    assert result["segmentation"]["status"] == "completed"
    assert result["movement_measurements"]["status"] == "completed"
    assert result["movement_measurements"]["frames"][0]["left_knee_internal_angle_deg"] > 0
    assert result["prediction"]["status"] == "not_configured"
    assert result["evaluation"]["status"] == "not_evaluated"
    assert (tmp_path / "source.pose.mp4").stat().st_size > 0
    assert (tmp_path / "source.segmentation.mp4").stat().st_size > 0
    for name in ("pose", "segmentation"):
        artifact = tmp_path / f"source.{name}.mp4"
        capture = cv2.VideoCapture(str(artifact))
        assert capture.isOpened()
        assert (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ) == (320, 240)
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        assert "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).lower() in {
            "avc1",
            "h264",
        }
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
        capture.release()
        assert len(decoded) == 4
    source_frame = cv2.VideoCapture(str(video))
    _, first_source_frame = source_frame.read()
    source_frame.release()
    pose_capture = cv2.VideoCapture(str(tmp_path / "source.pose.mp4"))
    _, first_pose_frame = pose_capture.read()
    pose_capture.release()
    seg_capture = cv2.VideoCapture(str(tmp_path / "source.segmentation.mp4"))
    _, first_seg_frame = seg_capture.read()
    seg_capture.release()
    assert np.mean(cv2.absdiff(first_pose_frame, first_source_frame)) > 1.0
    assert np.mean(cv2.absdiff(first_seg_frame, first_source_frame)) > 1.0
    green_edges = (first_pose_frame[:, :, 1] > first_pose_frame[:, :, 0] + 25) & (
        first_pose_frame[:, :, 1] > first_pose_frame[:, :, 2] + 10
    )
    assert green_edges.sum() > 25


def test_missing_model_assets_report_not_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pose = PoseEstimator(model_path="missing.pt")
    segmenter = HumanSegmenter(model_path="missing.pt")
    assert pose.process(np.zeros((16, 16, 3), dtype=np.uint8))["status"] == "not_configured"
    assert "missing.pt" in pose.unavailable_reason
    assert segmenter.process(np.zeros((16, 16, 3), dtype=np.uint8))["status"] == "not_configured"
    assert "missing.pt" in segmenter.reason


def test_missing_inference_dependency_is_not_configured(tmp_path, monkeypatch):
    pose_model = tmp_path / "pose.pt"
    segmentation_model = tmp_path / "segment.pt"
    pose_model.write_bytes(b"placeholder")
    segmentation_model.write_bytes(b"placeholder")
    original_import = builtins.__import__

    def import_without_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("test simulates absent dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ultralytics)
    pose = PoseEstimator(model_path=pose_model)
    segmenter = HumanSegmenter(model_path=segmentation_model)
    assert pose.status == "not_configured"
    assert segmenter.status == "not_configured"
    assert "Install pose and segmentation dependencies" in pose.unavailable_reason
    assert "Install pose and segmentation dependencies" in segmenter.reason


def test_left_and_right_knee_measurements_are_independent_and_pixel_based():
    result = _measure([_known_measurement_frame()], duration=1.0, width=1000, height=500)
    row = result["frames"][0]
    assert row["left_knee_internal_angle_deg"] == 180
    assert row["left_knee_flexion_deg"] == 0
    assert row["right_knee_internal_angle_deg"] == 90
    assert row["right_knee_flexion_deg"] == 90
    assert row["trunk_inclination_deg"] == 0
    assert result["valid_measurement_counts"]["left_knee_internal_angle"] == 1
    assert result["valid_measurement_counts"]["right_knee_flexion"] == 1


def test_measurements_reject_missing_low_confidence_and_invalid_landmarks():
    frame = _known_measurement_frame()
    frame["landmarks"][15]["visibility"] = 0.1
    frame["landmarks"][16]["x"] = 1.1
    frame["landmarks"][6]["visibility"] = 0.1
    frame["landmarks"][12]["x"] = float("nan")
    result = _measure([frame], duration=1.0, width=1000, height=500)
    row = result["frames"][0]
    assert row["left_knee_internal_angle_deg"] is None
    assert row["right_knee_internal_angle_deg"] is None
    assert row["left_hip_internal_angle_deg"] is not None
    assert row["right_hip_internal_angle_deg"] is None
    assert row["trunk_inclination_deg"] is None
    assert result["left_right_symmetry"] is None
    assert result["symmetry_valid_frame_count"] == 0


def test_measurements_tolerate_absent_landmark_slot():
    frame = _known_measurement_frame()
    frame["landmarks"][15] = None
    result = _measure([frame], duration=1.0, width=1000, height=500)
    row = result["frames"][0]
    assert row["left_knee_internal_angle_deg"] is None
    assert row["right_knee_internal_angle_deg"] == 90


def test_trunk_measurement_preserves_known_image_plane_lean():
    frame = _known_measurement_frame()
    # Set shoulder midpoint 15 degrees toward image-right from a vertical body axis.
    lean_px = 100 * np.tan(np.deg2rad(15))
    for index in (5, 6):
        frame["landmarks"][index].update(
            x=(200 + lean_px) / 1000,
            y=0.0,
            pixel_x=200 + lean_px,
            pixel_y=0,
        )
    result = _measure([frame], duration=1.0, width=1000, height=500)
    assert result["frames"][0]["trunk_inclination_deg"] == pytest.approx(15)


def test_symmetry_reports_paired_valid_frame_and_interval_counts():
    first = _known_measurement_frame()
    second = _known_measurement_frame()
    second["frame"] = 1
    second["timestamp_seconds"] = 0.5
    second["landmarks"][15]["pixel_y"] += 50
    second["landmarks"][15]["y"] += 0.1
    second["landmarks"][16]["pixel_y"] += 25
    second["landmarks"][16]["y"] += 0.05
    result = _measure([first, second], duration=1.0, width=1000, height=500)
    assert result["left_right_symmetry"] == pytest.approx(2 / 3)
    assert result["symmetry_valid_interval_count"] == 1
    assert result["symmetry_valid_frame_count"] == 2


def test_no_person_is_not_reported_as_measurements(tmp_path, monkeypatch):
    class NoPerson(FakePose):
        def process(self, _frame, _timestamp):
            return {"status": "completed", "detection": "no_person", "landmarks": []}

    video = create_synthetic_video(tmp_path / "empty.mp4", frames=12, fps=12)
    monkeypatch.setattr(analysis, "PoseEstimator", NoPerson)
    monkeypatch.setattr(analysis, "HumanSegmenter", FakeSegmenter)
    result = analysis.analyze_video(video, sample_fps=2)
    assert result["pose"]["status"] == "completed"
    assert result["pose"]["detected_frames"] == 0
    assert result["movement_measurements"]["status"] == "not_configured"


def test_real_cpu_models_handle_blank_frame_when_assets_are_installed():
    pytest.importorskip("ultralytics")
    from pathlib import Path

    if not Path("models/yolo11n-pose.pt").is_file() or not Path("models/yolo11n-seg.pt").is_file():
        pytest.skip("pretrained model assets are not installed")
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    with PoseEstimator() as pose:
        pose_result = pose.process(frame, 0)
    segmenter = HumanSegmenter()
    segmentation = segmenter.process(frame)
    segmenter.close()
    assert pose_result["status"] == "completed"
    assert pose_result["detection"] == "no_person"
    assert segmentation["status"] == "completed"
    assert segmentation["mask"].shape == frame.shape[:2]
