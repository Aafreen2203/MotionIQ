"""Video analysis orchestration with honest reporting of optional model status."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2

from motioniq.video import inspect_video
from motioniq.vision import PoseEstimator, draw_pose, segment_frame


def analyze_video(path: str | Path, *, max_frames: int = 60) -> dict[str, object]:
    metadata = inspect_video(path)
    cap = cv2.VideoCapture(str(path))
    stride = max(1, int(metadata["frame_count"]) // max_frames)
    poses, segmentations, latencies = [], [], []
    overlay_path = Path(path).with_name(f"{Path(path).stem}.overlay.jpg")
    pose_estimator = PoseEstimator()
    index = 0
    started = time.perf_counter()
    while len(poses) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            pose = pose_estimator.process(frame)
            segmentation = (
                segment_frame(frame) if not segmentations else {"status": "not_run_sampled"}
            )
            if not poses:
                overlay_frame = segmentation.pop("overlay_frame", None)
                overlay_frame = (
                    overlay_frame
                    if overlay_frame is not None
                    else draw_pose(frame, pose.get("landmarks", []))
                )
                if pose.get("landmarks") or segmentation.get("status") == "ok":
                    cv2.imwrite(str(overlay_path), overlay_frame)
            poses.append({"frame": index, **pose})
            segmentations.append(segmentation)
            if "latency_ms" in pose:
                latencies.append(float(pose["latency_ms"]))
        index += 1
    cap.release()
    pose_estimator.close()
    result = {
        "video": metadata,
        "pose": {
            "status": poses[0]["status"] if poses else "unavailable",
            "frames": poses,
            "validated": False,
        },
        "segmentation": segmentations[0] if segmentations else {"status": "unavailable"},
        "overlay": {
            "status": "available" if overlay_path.exists() else "unavailable",
            "url": None,
            "validated": False,
        },
        "movement_measurements": {
            "status": "unavailable",
            "reason": "A detected pose and calibrated view are required.",
        },
        "prediction": {
            "status": "unavailable",
            "reason": "No validated model weights or task label mapping are configured.",
        },
        "evaluation": {"status": "unavailable", "reason": "Ground-truth annotations are required."},
        "performance": {
            "end_to_end_ms": (time.perf_counter() - started) * 1000,
            "mean_pose_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        },
    }
    return result


def export_measurements_csv(analysis: dict[str, object], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "status", "landmark_index", "x", "y", "z", "visibility"])
        for frame in analysis.get("pose", {}).get("frames", []):
            for index, landmark in enumerate(frame.get("landmarks", [])):
                writer.writerow(
                    [
                        frame.get("frame"),
                        frame.get("status"),
                        index,
                        landmark.get("x"),
                        landmark.get("y"),
                        landmark.get("z"),
                        landmark.get("visibility"),
                    ]
                )
    return output
