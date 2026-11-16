"""Evaluation metrics for pose, segmentation and classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean per-joint position error; inputs must use matching coordinates/units."""
    return float(np.mean(np.linalg.norm(np.asarray(predicted) - np.asarray(target), axis=-1)))


def pck(predicted: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """Percentage of keypoints within a coordinate-distance threshold."""
    errors = np.linalg.norm(np.asarray(predicted) - np.asarray(target), axis=-1)
    return float(np.mean(errors <= threshold))


def tracking_stability(tracked: np.ndarray) -> float:
    """Fraction of adjacent frame pairs where each landmark is present in both."""
    tracked = np.asarray(tracked, dtype=bool)
    if tracked.ndim < 1:
        raise ValueError("tracked must include a frame dimension")
    if tracked.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.logical_and(tracked[1:], tracked[:-1])))


def segmentation_scores(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted, target = np.asarray(predicted, bool), np.asarray(target, bool)
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    total = predicted.sum() + target.sum()
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / total) if total else 1.0,
    }


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
