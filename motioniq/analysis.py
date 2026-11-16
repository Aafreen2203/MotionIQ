"""Timestamped sampled-frame analysis with explicit model and evaluation status."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np

from motioniq.geometry import joint_angle, knee_flexion, movement_symmetry, trunk_inclination
from motioniq.video import inspect_video
from motioniq.vision import HumanSegmenter, PoseEstimator, blend_mask, draw_pose

_L = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
MIN_LANDMARK_CONFIDENCE = 0.25
VIDEO_CODEC = "avc1"


def _validate_video_artifact(
    path: Path, expected_frames: int, width: int, height: int
) -> tuple[bool, str | None]:
    """Decode the finished MP4 before exposing it as a completed artifact."""
    if not path.is_file() or path.stat().st_size == 0:
        return False, "Video encoder produced no output file."
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return False, "Generated video cannot be opened for decoding."
    decoded = 0
    try:
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded += 1
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    if (actual_width, actual_height) != (width, height):
        return (
            False,
            "Generated video resolution is "
            f"{actual_width}x{actual_height}, expected {width}x{height}.",
        )
    if decoded != expected_frames:
        return False, f"Generated video decoded {decoded}/{expected_frames} frames."
    if codec.lower() not in {"avc1", "h264"}:
        return (
            False,
            f"Generated video codec is {codec!r}; "
            f"expected browser-compatible H.264 ({VIDEO_CODEC}).",
        )
    return True, None


def _measure(
    frames: list[dict[str, object]], duration: float, width: int, height: int
) -> dict[str, object]:
    def point(frame: dict[str, object], name: str):
        landmarks = frame.get("landmarks", [])
        index = _L[name]
        if not isinstance(landmarks, list) or index >= len(landmarks):
            return None
        landmark = landmarks[index]
        try:
            confidence = float(landmark["visibility"])
            x, y = float(landmark["x"]), float(landmark["y"])
            pixel_x = float(landmark.get("pixel_x", x * width))
            pixel_y = float(landmark.get("pixel_y", y * height))
        except (KeyError, TypeError, ValueError):
            return None
        values = np.array([x, y, pixel_x, pixel_y, confidence], dtype=float)
        if (
            not np.isfinite(values).all()
            or confidence < MIN_LANDMARK_CONFIDENCE
            or not (0 <= x <= 1 and 0 <= y <= 1)
            or not (0 <= pixel_x <= width and 0 <= pixel_y <= height)
        ):
            return None
        return np.array([pixel_x, pixel_y]), np.array([x, y])

    rows = []
    feature_names = [
        f"{side}_{feature}"
        for side in ("left", "right")
        for feature in ("knee_internal_angle", "knee_flexion", "hip_internal_angle")
    ] + ["trunk_inclination"]
    valid_counts = {name: 0 for name in feature_names}
    valid_counts["symmetry_intervals"] = 0
    row_points = []
    for frame in frames:
        row: dict[str, object] = {
            "frame": frame.get("frame"),
            "timestamp_seconds": frame.get("timestamp_seconds"),
        }
        points = {name: point(frame, name) for name in _L}
        for side in ("left", "right"):

            def coords(name):
                value = points[f"{side}_{name}"]
                return value[0] if value is not None else None

            hip, knee, ankle = coords("hip"), coords("knee"), coords("ankle")
            shoulder = coords("shoulder")
            internal = (
                joint_angle(hip, knee, ankle)
                if all(p is not None for p in (hip, knee, ankle))
                else float("nan")
            )
            flexion = knee_flexion(internal)
            hip_internal = (
                joint_angle(shoulder, hip, knee)
                if all(p is not None for p in (shoulder, hip, knee))
                else float("nan")
            )
            for suffix, value in (
                ("knee_internal_angle_deg", internal),
                ("knee_flexion_deg", flexion),
                ("hip_internal_angle_deg", hip_internal),
            ):
                row[f"{side}_{suffix}"] = float(value) if np.isfinite(value) else None
                if np.isfinite(value):
                    count_name = f"{side}_{suffix.removesuffix('_deg')}"
                    valid_counts[count_name] += 1
            # COCO-17 has no foot landmarks, so ankle angles cannot be observed.
            row[f"{side}_ankle_angle_deg"] = None
        shoulder_points = [points["left_shoulder"], points["right_shoulder"]]
        hip_points = [points["left_hip"], points["right_hip"]]
        if all(value is not None for value in shoulder_points + hip_points):
            shoulder_mid = np.mean([value[0] for value in shoulder_points], axis=0)
            hip_mid = np.mean([value[0] for value in hip_points], axis=0)
            trunk = trunk_inclination(shoulder_mid, hip_mid)
        else:
            trunk = float("nan")
        row["trunk_inclination_deg"] = float(trunk) if np.isfinite(trunk) else None
        if np.isfinite(trunk):
            valid_counts["trunk_inclination"] += 1
        for side in ("left", "right"):
            hip_point = points[f"{side}_hip"]
            row[f"{side}_hip_pixel_x"] = float(hip_point[0][0]) if hip_point else None
            row[f"{side}_hip_pixel_y"] = float(hip_point[0][1]) if hip_point else None
        for side in ("left", "right"):
            key = f"{side}_hip"
            previous_hip = row_points[-1].get(key) if rows else None
            current_hip = points[key]
            if previous_hip is not None and current_hip is not None:
                delta = current_hip[0] - previous_hip[0]
                row[f"{side}_hip_displacement_pixels"] = float(np.linalg.norm(delta))
                row[f"{side}_hip_displacement_normalized"] = float(
                    np.linalg.norm([delta[0] / width, delta[1] / height])
                )
            else:
                row[f"{side}_hip_displacement_pixels"] = None
                row[f"{side}_hip_displacement_normalized"] = None
        rows.append(row)
        row_points.append(points)

    # Compare paired changes in normalized vertical hip-to-ankle separation.
    left_changes, right_changes, symmetry_frame_indices = [], [], set()
    for index in range(1, len(row_points)):
        previous, current = row_points[index - 1], row_points[index]
        required = ("left_hip", "left_ankle", "right_hip", "right_ankle")
        if not all(previous[name] is not None and current[name] is not None for name in required):
            continue
        left_before = previous["left_ankle"][1][1] - previous["left_hip"][1][1]
        left_after = current["left_ankle"][1][1] - current["left_hip"][1][1]
        right_before = previous["right_ankle"][1][1] - previous["right_hip"][1][1]
        right_after = current["right_ankle"][1][1] - current["right_hip"][1][1]
        left_changes.append(left_after - left_before)
        right_changes.append(right_after - right_before)
        symmetry_frame_indices.update((index - 1, index))
    valid_counts["symmetry_intervals"] = len(left_changes)
    symmetry_value = (
        movement_symmetry(np.asarray(left_changes), np.asarray(right_changes))
        if left_changes
        else float("nan")
    )
    symmetry = float(symmetry_value) if np.isfinite(symmetry_value) else None
    valid_measurement_count = sum(valid_counts[name] for name in feature_names)
    valid_measurement_count += valid_counts["symmetry_intervals"]
    status = "completed" if valid_measurement_count else "not_configured"
    return {
        "status": status,
        "reason": None
        if status == "completed"
        else "No valid, sufficiently confident measurement landmarks were found.",
        "coordinate_space": (
            "Pixel coordinates in the 2D image plane for angles; "
            "normalized coordinates for symmetry."
        ),
        "landmark_confidence_threshold": MIN_LANDMARK_CONFIDENCE,
        "angle_definition": (
            "Internal angle at the named joint; knee flexion is 180 degrees minus the "
            "internal knee angle."
        ),
        "valid_measurement_counts": valid_counts,
        "symmetry_valid_frame_count": len(symmetry_frame_indices),
        "symmetry_valid_interval_count": len(left_changes),
        "physical_calibration": (
            "Not calibrated; angles and displacement are not clinical or physical measurements."
        ),
        "movement_duration_seconds": duration,
        "left_right_symmetry": symmetry,
        "symmetry_reason": (
            None
            if symmetry is not None
            else "No valid paired movement, or both sides had zero measured change."
        ),
        "left_right_symmetry_definition": (
            "1 - mean(abs(left_change - right_change)) / "
            "mean(abs(left_change) + abs(right_change)); "
            "changes are paired samples of normalized vertical ankle-minus-hip separation."
        ),
        "symmetry_definition": (
            "1 - mean(abs(left_change - right_change)) / "
            "mean(abs(left_change) + abs(right_change)); "
            "changes are paired samples of normalized vertical ankle-minus-hip separation."
        ),
        "repetition_count": {
            "status": "not_configured",
            "reason": "No task-specific repetition signal/threshold has been configured.",
        },
        "frames": rows,
        "limitations": (
            "Single-view 2D landmarks are affected by camera angle, occlusion, "
            "and pose-model error."
        ),
    }


def analyze_video(
    path: str | Path,
    *,
    sample_fps: float | None = None,
    max_frames: int = 300,
    artifact_dir: str | Path | None = None,
    write_artifacts: bool = True,
) -> dict[str, object]:
    if sample_fps is None:
        sample_fps = float(os.getenv("MOTIONIQ_SAMPLE_FPS", "3"))
    if sample_fps <= 0 or max_frames < 1:
        raise ValueError("sample_fps and max_frames must be positive")
    path = Path(path)
    metadata = inspect_video(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    effective_sample_fps = min(sample_fps, float(metadata["fps"]))
    interval = 1.0 / effective_sample_fps
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else path.parent
    if write_artifacts:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    pose_path = artifact_dir / f"{path.stem}.pose.mp4"
    segmentation_path = artifact_dir / f"{path.stem}.segmentation.mp4"
    pose_writer = segmentation_writer = None
    pose_encoder_open = segmentation_encoder_open = False
    poses: list[dict[str, object]] = []
    segmentation_masks: dict[str, np.ndarray] = {}
    pose = PoseEstimator()
    segmenter = HumanSegmenter()
    pose_status = pose.status
    pose_reason = pose.unavailable_reason
    segment_status = segmenter.status
    segment_reason = segmenter.reason
    segmentation_detected_frames = 0
    started = time.perf_counter()
    next_sample = 0.0
    index = 0
    try:
        while len(poses) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp <= 0 and index:
                timestamp = index / float(metadata["fps"])
            if timestamp + 1e-6 >= next_sample:
                h, w = frame.shape[:2]
                if write_artifacts and pose_writer is None:
                    pose_writer = cv2.VideoWriter(
                        str(pose_path),
                        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                        effective_sample_fps,
                        (w, h),
                    )
                    segmentation_writer = cv2.VideoWriter(
                        str(segmentation_path),
                        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                        effective_sample_fps,
                        (w, h),
                    )
                    pose_encoder_open = pose_writer.isOpened()
                    segmentation_encoder_open = segmentation_writer.isOpened()
                if pose_status == "completed":
                    try:
                        pose_result = pose.process(frame, int(timestamp * 1000))
                        if pose_result["status"] != "completed":
                            pose_status = pose_result["status"]
                            pose_reason = pose_result.get("reason")
                    except Exception as error:
                        pose_status, pose_reason = "failed", str(error)
                        pose_result = {"status": "failed", "reason": pose_reason, "landmarks": []}
                else:
                    pose_result = {
                        "status": pose_status,
                        "reason": pose_reason,
                        "landmarks": [],
                    }
                pose_result.update({"frame": index, "timestamp_seconds": timestamp})
                poses.append(pose_result)
                if pose_writer is not None and pose_writer.isOpened():
                    pose_writer.write(draw_pose(frame, pose_result.get("landmarks", [])))
                if segment_status != "failed":
                    try:
                        seg_result = segmenter.process(frame)
                        if seg_result["status"] != "completed":
                            segment_status = seg_result["status"]
                            segment_reason = seg_result.get("reason")
                        elif segment_status == "completed":
                            if not write_artifacts:
                                pass
                            elif segmentation_writer is not None and segmentation_writer.isOpened():
                                segmentation_writer.write(blend_mask(frame, seg_result["mask"]))
                            else:
                                segment_status = "failed"
                                segment_reason = "H.264 segmentation video encoder could not open."
                            segmentation_detected_frames += int(seg_result.get("mask_count", 0) > 0)
                            segmentation_masks[str(index)] = seg_result.get(
                                "evaluation_mask", (seg_result["mask"] >= 0.5).astype(np.uint8)
                            )
                    except Exception as error:
                        segment_status, segment_reason = "failed", str(error)
                next_sample += interval
            index += 1
    finally:
        cap.release()
        pose.close()
        segmenter.close()
        if pose_writer is not None:
            pose_writer.release()
        if segmentation_writer is not None:
            segmentation_writer.release()
    video_width, video_height = int(metadata["width"]), int(metadata["height"])
    pose_artifact_ok, pose_artifact_reason = (
        _validate_video_artifact(pose_path, len(poses), video_width, video_height)
        if write_artifacts and poses and pose_encoder_open
        else (False, "H.264 pose video encoder could not open.")
    )
    seg_artifact_ok, seg_artifact_reason = (
        _validate_video_artifact(segmentation_path, len(poses), video_width, video_height)
        if write_artifacts and poses and segmentation_encoder_open
        else (False, "H.264 segmentation video encoder could not open.")
    )
    pose_available = write_artifacts and pose_status == "completed" and pose_artifact_ok
    seg_available = write_artifacts and segment_status == "completed" and seg_artifact_ok
    if write_artifacts and pose_status == "completed" and not pose_available:
        pose_reason = pose_artifact_reason
    if write_artifacts and segment_status == "completed" and not seg_available:
        segment_reason = seg_artifact_reason
    measurements = _measure(
        poses, float(metadata["duration_seconds"]), int(metadata["width"]), int(metadata["height"])
    )
    return {
        "video": metadata,
        "sampling": {
            "status": "completed",
            "sample_fps": sample_fps,
            "effective_sample_fps": effective_sample_fps,
            "sampled_frames": len(poses),
            "truncated": len(poses) >= max_frames and index < int(metadata["frame_count"]),
            "timestamps_seconds": [f["timestamp_seconds"] for f in poses],
        },
        "pose": {
            "status": pose_status,
            "reason": pose_reason,
            "frames": poses,
            "detected_frames": sum(bool(f.get("landmarks")) for f in poses),
            "model": "Ultralytics YOLO11n-pose COCO-17" if pose_status == "completed" else None,
            "validated": False,
        },
        "segmentation": {
            "status": segment_status if segment_status != "completed" else "completed",
            "reason": segment_reason or segmenter.reason,
            "model": "Ultralytics YOLO11n-seg COCO person"
            if segment_status == "completed"
            else None,
            "detected_frames": segmentation_detected_frames,
            "overlay_url": None,
            "validated": False,
        },
        "segmentation_masks": segmentation_masks,
        "overlay": {
            "status": "available"
            if pose_available
            else ("not_generated" if not write_artifacts else pose_status),
            "url": None,
            "validated": False,
        },
        "processed_video": {
            "status": "completed"
            if pose_available
            else (
                "not_generated"
                if not write_artifacts and pose_status == "completed"
                else pose_status
            ),
            "url": None,
            "reason": pose_reason,
            "codec": VIDEO_CODEC if pose_available else None,
        },
        "segmentation_video": {
            "status": "completed"
            if seg_available
            else (
                "not_generated"
                if not write_artifacts and segment_status == "completed"
                else segment_status
            ),
            "url": None,
            "reason": segment_reason,
            "codec": VIDEO_CODEC if seg_available else None,
        },
        "movement_measurements": measurements,
        "prediction": {
            "status": "not_configured",
            "reason": (
                "No trained checkpoint and class label map are configured; no prediction was made."
            ),
        },
        "evaluation": {
            "status": "not_evaluated",
            "reason": (
                "Ground-truth pose, segmentation, and classification annotations were not supplied."
            ),
        },
        "performance": {"end_to_end_ms": (time.perf_counter() - started) * 1000},
    }


def export_measurements_csv(analysis: dict[str, object], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame",
                "timestamp_seconds",
                "status",
                "landmark_index",
                "x",
                "y",
                "z",
                "pixel_x",
                "pixel_y",
                "visibility",
            ]
        )
        for frame in analysis.get("pose", {}).get("frames", []):
            for index, landmark in enumerate(frame.get("landmarks", [])):
                writer.writerow(
                    [
                        frame.get("frame"),
                        frame.get("timestamp_seconds"),
                        frame.get("status"),
                        index,
                        landmark.get("x"),
                        landmark.get("y"),
                        landmark.get("z"),
                        landmark.get("pixel_x"),
                        landmark.get("pixel_y"),
                        landmark.get("visibility"),
                    ]
                )
    return output
