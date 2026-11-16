"""Small reproducible movement classifiers for feature sequences."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


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
    """Create a small PyTorch GRU classifier; PyTorch is an optional dependency."""
    from torch import nn

    class TemporalClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_features, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, classes))

        def forward(self, sequence):
            encoded, _ = self.gru(sequence)
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
    """Train a compact GRU using sequences shaped (samples, frames, features)."""
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
