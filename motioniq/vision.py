"""Optional MediaPipe pose and Ultralytics segmentation adapters."""

from __future__ import annotations

import time

import cv2
import numpy as np


class PoseEstimator:
    """Reusable MediaPipe model instance for sampled-frame inference."""

    def __init__(self, confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError:
            self.pose = None
            self.unavailable_reason = "Install requirements-vision.txt for MediaPipe."
        else:
            try:
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=confidence,
                    min_tracking_confidence=confidence,
                )
                self.unavailable_reason = None
            except Exception as error:
                self.pose = None
                self.unavailable_reason = f"MediaPipe could not initialize: {error}"

    def process(self, frame: np.ndarray) -> dict[str, object]:
        if self.pose is None:
            return {
                "status": "unavailable",
                "reason": self.unavailable_reason,
            }
        started = time.perf_counter()
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return {
                "status": "no_pose",
                "landmarks": [],
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        landmarks = [
            {
                "x": float(point.x),
                "y": float(point.y),
                "z": float(point.z),
                "visibility": float(point.visibility),
            }
            for point in result.pose_landmarks.landmark
        ]
        return {
            "status": "ok",
            "landmarks": landmarks,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "validated": False,
        }

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def pose_frame(frame: np.ndarray, confidence: float = 0.5) -> dict[str, object]:
    """One-frame API; video callers should reuse a PoseEstimator instance."""
    with PoseEstimator(confidence) as estimator:
        return estimator.process(frame)


def segment_frame(
    frame: np.ndarray, model_path: str = "models/yolo11n-seg.pt"
) -> dict[str, object]:
    """Run optional YOLO segmentation with a locally provisioned weights file."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "unavailable", "reason": "Install requirements-vision.txt for YOLO."}
    from pathlib import Path

    if not Path(model_path).is_file():
        return {
            "status": "unavailable",
            "reason": f"YOLO weights were not found at {model_path}.",
        }
    model = YOLO(model_path)
    started = time.perf_counter()
    result = model.predict(frame, verbose=False)[0]
    masks = result.masks.data.cpu().numpy() if result.masks is not None else np.empty((0, 0, 0))
    overlay = result.plot()
    return {
        "status": "ok",
        "mask_count": int(len(masks)),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "validated": False,
        "overlay_frame": overlay,
    }


def draw_pose(frame: np.ndarray, landmarks: list[dict[str, float]]) -> np.ndarray:
    """Draw MediaPipe's 2D pose skeleton on a copy of the source frame."""
    try:
        import mediapipe as mp
    except ImportError:
        return frame.copy()
    output = frame.copy()
    height, width = output.shape[:2]
    points = [(int(point["x"] * width), int(point["y"] * height)) for point in landmarks]
    for first, second in mp.solutions.pose.POSE_CONNECTIONS:
        if first < len(points) and second < len(points):
            cv2.line(output, points[first], points[second], (45, 210, 150), 2, cv2.LINE_AA)
    for point, landmark in zip(points, landmarks):
        if landmark["visibility"] >= 0.5:
            cv2.circle(output, point, 3, (245, 190, 80), -1, cv2.LINE_AA)
    return output
