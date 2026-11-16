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


def classification_scores(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> dict[str, object]:
    """Return held-out classification metrics with stable class ordering."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays with matching shapes")
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "labels": labels,
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "sample_count": int(y_true.size),
    }
