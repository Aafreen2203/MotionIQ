"""Geometry and temporal features computed from normalized 2D landmarks."""

from __future__ import annotations

import numpy as np


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return internal angle ABC in degrees, or NaN for invalid/degenerate 2D points."""
    try:
        a, b, c = (np.asarray(point, dtype=float) for point in (a, b, c))
    except (TypeError, ValueError):
        return float("nan")
    if any(point.shape != (2,) or not np.isfinite(point).all() for point in (a, b, c)):
        return float("nan")
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= np.finfo(float).eps:
        return float("nan")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def knee_flexion(internal_angle: float) -> float:
    """Return flexion magnitude as 180° minus the internal knee angle."""
    try:
        angle = float(internal_angle)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(angle) or not 0 <= angle <= 180:
        return float("nan")
    return 180.0 - angle


def trunk_inclination(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Image-plane trunk angle from vertical; positive toward image-right.

    Points use pixel coordinates (x right, y down). This sign describes the image,
    not anatomical forward/backward lean, which requires a known sagittal camera view.
    """
    try:
        shoulder_mid = np.asarray(shoulder_mid, dtype=float)
        hip_mid = np.asarray(hip_mid, dtype=float)
    except (TypeError, ValueError):
        return float("nan")
    if any(
        point.shape != (2,) or not np.isfinite(point).all() for point in (shoulder_mid, hip_mid)
    ):
        return float("nan")
    delta = shoulder_mid - hip_mid
    if np.linalg.norm(delta) <= np.finfo(float).eps:
        return float("nan")
    return float(np.degrees(np.arctan2(delta[0], -delta[1])))


def landmark_displacement(points: np.ndarray) -> np.ndarray:
    """Frame-to-frame Euclidean displacement for an (frames, joints, xy) array."""
    points = np.asarray(points, dtype=float)
    if points.ndim < 2 or points.shape[0] < 2:
        return np.empty((0, *points.shape[1:-1]), dtype=float)
    return np.linalg.norm(np.diff(points, axis=0), axis=-1)


def movement_symmetry(left: np.ndarray, right: np.ndarray) -> float:
    """Bounded paired-motion similarity score in [0, 1].

    Score = 1 - mean(|left-right|) / mean(|left|+|right|). Inputs must be
    corresponding finite signed movement values. This is a coefficient, not a percent.
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        left.shape != right.shape
        or left.size == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        return float("nan")
    denominator = np.mean(np.abs(left) + np.abs(right))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.clip(1.0 - np.mean(np.abs(left - right)) / denominator, 0.0, 1.0))


def detect_repetitions(signal: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """Detect peaks in a 1D movement signal using SciPy."""
    from scipy.signal import find_peaks

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    peaks, _ = find_peaks(values, distance=min_distance)
    return peaks


def cycle_durations(peak_frames: np.ndarray, fps: float) -> np.ndarray:
    """Return seconds between repetition peaks."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return np.diff(np.asarray(peak_frames, dtype=float)) / fps


def temporal_features(points: np.ndarray, fps: float) -> dict[str, float]:
    """Compact motion features; units are normalized-frame coordinates."""
    points = np.asarray(points, dtype=float)
    movement = landmark_displacement(points)
    durations = cycle_durations(detect_repetitions(np.nanmean(points[..., 1], axis=1)), fps)
    return {
        "mean_speed_per_second": float(np.nanmean(movement) * fps) if movement.size else 0.0,
        "measurement_consistency_std": float(np.nanstd(movement)) if movement.size else 0.0,
        "detected_repetitions": float(
            len(durations)
            + (1 if len(detect_repetitions(np.nanmean(points[..., 1], axis=1))) else 0)
        ),
        "mean_cycle_seconds": float(np.mean(durations)) if durations.size else float("nan"),
    }
