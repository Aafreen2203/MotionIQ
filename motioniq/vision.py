"""Optional CPU inference adapters for pretrained Ultralytics YOLO11 models."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# COCO-17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
POSE_CONNECTIONS = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def _load_yolo(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Install pose and segmentation dependencies: python -m pip install -e '.[vision]'"
        ) from error
    return YOLO(str(path))


class PoseEstimator:
    """Reusable pretrained YOLO11n pose model running on CPU."""

    def __init__(self, confidence: float = 0.25, model_path: str | Path = "models/yolo11n-pose.pt"):
        self.model = None
        self.status = "not_configured"
        self.unavailable_reason = None
        self.confidence = confidence
        path = Path(model_path)
        if not path.is_file():
            self.unavailable_reason = (
                f"Pose weights are missing: {path}. Run python -m motioniq.download_models."
            )
            return
        try:
            self.model = _load_yolo(path)
            self.status = "completed"
        except Exception as error:
            self.status = (
                "not_configured"
                if isinstance(error, RuntimeError) and str(error).startswith("Install ")
                else "failed"
            )
            self.unavailable_reason = f"Pose model could not initialize: {error}"

    def process(self, frame: np.ndarray, timestamp_ms: int = 0) -> dict[str, object]:
        del timestamp_ms  # YOLO pose is frame-based; timestamps are attached by the video pipeline.
        if self.model is None:
            return {"status": self.status, "reason": self.unavailable_reason, "landmarks": []}
        started = time.perf_counter()
        result = self.model.predict(frame, device="cpu", conf=self.confidence, verbose=False)[0]
        keypoints = result.keypoints
        if (
            keypoints is None
            or keypoints.xy.shape[0] == 0
            or result.boxes is None
            or len(result.boxes) == 0
        ):
            return {
                "status": "completed",
                "detection": "no_person",
                "landmarks": [],
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        person_index = int(result.boxes.conf.argmax().item()) if result.boxes is not None else 0
        coordinates = keypoints.xy[person_index].cpu().numpy()
        confidences = (
            keypoints.conf[person_index].cpu().numpy()
            if keypoints.conf is not None
            else np.ones(len(coordinates))
        )
        height, width = frame.shape[:2]
        landmarks = []
        for (pixel_x, pixel_y), confidence in zip(coordinates, confidences):
            landmarks.append(
                {
                    "x": float(pixel_x / width),
                    "y": float(pixel_y / height),
                    "z": 0.0,
                    "visibility": float(confidence),
                    "pixel_x": float(pixel_x),
                    "pixel_y": float(pixel_y),
                }
            )
        return {
            "status": "completed",
            "detection": "person",
            "landmarks": landmarks,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "validated": False,
        }

    def close(self) -> None:
        # Ultralytics models hold no explicit session resource requiring close.
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HumanSegmenter:
    """YOLO11n instance segmentation; only COCO person masks are included."""

    def __init__(self, model_path: str | Path = "models/yolo11n-seg.pt"):
        self.model = None
        self.status = "not_configured"
        self.reason = None
        path = Path(model_path)
        if not path.is_file():
            self.reason = (
                f"Segmentation weights are missing: {path}. Run python -m motioniq.download_models."
            )
            return
        try:
            self.model = _load_yolo(path)
            self.status = "completed"
        except Exception as error:
            self.status = (
                "not_configured"
                if isinstance(error, RuntimeError) and str(error).startswith("Install ")
                else "failed"
            )
            self.reason = f"Segmentation model could not initialize: {error}"

    def process(self, frame: np.ndarray) -> dict[str, object]:
        if self.model is None:
            return {"status": self.status, "reason": self.reason}
        started = time.perf_counter()
        result = self.model.predict(frame, device="cpu", classes=[0], verbose=False)[0]
        height, width = frame.shape[:2]
        masks = result.masks
        if masks is None or len(masks.data) == 0:
            empty_mask = np.zeros((160, 160), np.uint8)
            return {
                "status": "completed",
                "mask": np.zeros((height, width), np.float32),
                "evaluation_mask": empty_mask,
                "mask_count": 0,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "model": "Ultralytics YOLO11n-seg COCO person",
                "validated": False,
            }
        model_mask = masks.data.max(dim=0).values.cpu().numpy().astype(np.float32)
        combined = cv2.resize(model_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        return {
            "status": "completed",
            "mask": combined,
            "evaluation_mask": (model_mask >= 0.5).astype(np.uint8),
            "mask_count": int(len(masks.data)),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "model": "Ultralytics YOLO11n-seg COCO person",
            "validated": False,
        }

    def close(self) -> None:
        return None


def blend_mask(frame: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    output = frame.copy()
    foreground = resized >= threshold
    tint = np.zeros_like(frame)
    tint[:] = (50, 170, 245)
    output[foreground] = (
        frame[foreground].astype(np.float32) * 0.55 + tint[foreground].astype(np.float32) * 0.45
    ).astype(np.uint8)
    return output


def draw_pose(frame: np.ndarray, landmarks: list[dict[str, float]]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    points = [(int(point["x"] * width), int(point["y"] * height)) for point in landmarks]
    for first, second in POSE_CONNECTIONS:
        if first < len(points) and second < len(points):
            cv2.line(output, points[first], points[second], (45, 210, 150), 2, cv2.LINE_AA)
    for point, landmark in zip(points, landmarks):
        if landmark.get("visibility", 0) >= 0.25:
            cv2.circle(output, point, 3, (245, 190, 80), -1, cv2.LINE_AA)
    return output
