"""Small reproducible movement classifiers for feature sequences."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURE_SCHEMA_VERSION = "motioniq-temporal-v1"
MOVEMENT_SERIES = (
    "left_knee_flexion_deg",
    "right_knee_flexion_deg",
    "left_hip_internal_angle_deg",
    "right_hip_internal_angle_deg",
    "trunk_inclination_deg",
    "left_hip_displacement_normalized",
    "right_hip_displacement_normalized",
)


def sequence_features(frames: list[dict[str, object]], width: int, height: int) -> dict[str, float]:
    """Summarize timestamped image-space movement series and temporal velocities."""
    if not frames or width <= 0 or height <= 0:
        raise ValueError("A non-empty frame sequence and positive image dimensions are required")
    timestamps = np.asarray(
        [frame.get("timestamp_seconds", np.nan) for frame in frames], dtype=float
    )
    result: dict[str, float] = {
        "duration_seconds": float(np.nanmax(timestamps) - np.nanmin(timestamps))
    }
    for name in MOVEMENT_SERIES:
        values = np.asarray([frame.get(name, np.nan) for frame in frames], dtype=float)
        valid = np.isfinite(values)
        prefix = name.removesuffix("_deg").removesuffix("_normalized")
        result[f"{prefix}_valid_fraction"] = float(valid.mean())
        if not valid.any():
            for statistic in (
                "mean",
                "std",
                "min",
                "max",
                "range",
                "mean_abs_velocity",
                "max_abs_velocity",
            ):
                result[f"{prefix}_{statistic}"] = float("nan")
            continue
        observed = values[valid]
        result[f"{prefix}_mean"] = float(np.mean(observed))
        result[f"{prefix}_std"] = float(np.std(observed))
        result[f"{prefix}_min"] = float(np.min(observed))
        result[f"{prefix}_max"] = float(np.max(observed))
        result[f"{prefix}_range"] = float(np.max(observed) - np.min(observed))
        velocities = []
        for i in range(1, len(frames)):
            dt = timestamps[i] - timestamps[i - 1]
            if valid[i] and valid[i - 1] and np.isfinite(dt) and dt > 0:
                velocities.append(abs(values[i] - values[i - 1]) / dt)
        result[f"{prefix}_mean_abs_velocity"] = (
            float(np.mean(velocities)) if velocities else float("nan")
        )
        result[f"{prefix}_max_abs_velocity"] = (
            float(np.max(velocities)) if velocities else float("nan")
        )
    return result


def make_random_forest(seed: int = 42, **parameters: object) -> Pipeline:
    """Median-imputed random forest. No scaling is needed for tree splits."""
    defaults: dict[str, object] = {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": seed,
        "n_jobs": -1,
    }
    defaults.update(parameters)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", RandomForestClassifier(**defaults)),
        ]
    )


def save_classifier_bundle(
    model: object,
    path: str | Path,
    *,
    feature_names: list[str],
    class_labels: list[str],
    metadata: dict[str, object],
) -> None:
    """Persist a trusted local sklearn pipeline and its complete inference contract."""
    import joblib

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "class_labels": class_labels,
        "class_to_index": {label: index for index, label in enumerate(class_labels)},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": "motioniq-random-forest-v1",
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".joblib", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(bundle, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_classifier_bundle(path: str | Path) -> dict[str, object]:
    """Load a MotionIQ bundle. Only load trusted files because joblib uses pickle."""
    import joblib

    bundle = joblib.load(path)
    required = {
        "model",
        "feature_names",
        "class_labels",
        "class_to_index",
        "feature_schema_version",
        "model_version",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Classifier checkpoint is missing required MotionIQ metadata")
    if bundle["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Classifier feature schema does not match this application")
    return bundle


def predict_classifier_bundle(
    bundle: dict[str, object], features: dict[str, float], *, confidence_threshold: float = 0.55
) -> dict[str, object]:
    names = list(bundle["feature_names"])
    vector = np.asarray([[features.get(name, np.nan) for name in names]], dtype=float)
    finite_count = int(np.isfinite(vector).sum())
    observed_coverage = [
        value
        for name, value in features.items()
        if name.endswith("_valid_fraction") and np.isfinite(value)
    ]
    if finite_count == 0 or (observed_coverage and max(observed_coverage) <= 0):
        return {
            "status": "failed",
            "reason": "No valid movement features were detected in this video.",
            "model_version": bundle["model_version"],
        }
    model = bundle["model"]
    probabilities = model.predict_proba(vector)[0]
    model_classes = [str(value) for value in model.named_steps["classifier"].classes_]
    probability_map = {label: float(probabilities[i]) for i, label in enumerate(model_classes)}
    predicted = max(probability_map, key=probability_map.get)
    confidence = probability_map[predicted]
    if confidence < confidence_threshold:
        return {
            "status": "completed",
            "decision": "low_confidence",
            "predicted_class": None,
            "top_candidate": predicted,
            "confidence": confidence,
            "probabilities": probability_map,
            "reason": (
                "Top Random Forest vote share is below threshold " f"{confidence_threshold:.2f}."
            ),
            "model_version": bundle["model_version"],
            "probability_interpretation": "Uncalibrated Random Forest class vote fractions.",
        }
    return {
        "status": "completed",
        "decision": "predicted",
        "predicted_class": predicted,
        "confidence": confidence,
        "probabilities": probability_map,
        "reason": None,
        "model_version": bundle["model_version"],
        "probability_interpretation": "Uncalibrated Random Forest class vote fractions.",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_sklearn_baseline(
    features: np.ndarray, labels: np.ndarray, seed: int = 42
) -> ExtraTreesClassifier:
    set_seed(seed)
    model = ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
    model.fit(np.asarray(features), np.asarray(labels))
    return model


def save_model(model: object, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> object:
    import joblib

    return joblib.load(path)


def make_temporal_model(input_features: int, classes: int):
    """Create an experimental bidirectional LSTM; PyTorch is an optional dependency."""
    from torch import nn

    class TemporalClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_features, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, classes))

        def forward(self, sequence):
            encoded, _ = self.lstm(sequence)
            return self.head(encoded.mean(dim=1))

    return TemporalClassifier()


def train_temporal_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
):
    """Experimental LSTM training for (samples, frames, features); RF remains default."""
    import torch
    from torch import nn

    set_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    classes, y = np.unique(labels, return_inverse=True)
    target = torch.as_tensor(y, dtype=torch.long, device=device)
    model = make_temporal_model(x.shape[-1], len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_fn(model(x), target).backward()
        optimizer.step()
    return model, classes.tolist()


def predict_temporal(model, sequence: np.ndarray, classes: list[str], device: str = "cpu") -> str:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0))
    return str(classes[int(logits.argmax(dim=1).item())])
