"""Video inspection, deterministic frame extraction and synthetic fixtures."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


def inspect_video(path: str | Path) -> dict[str, object]:
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or count <= 0 or fps <= 0:
        raise ValueError("Video has invalid dimensions, frame count, or frame rate")
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": count,
        "duration_seconds": count / fps,
        "size_bytes": path.stat().st_size,
    }


def extract_frames(path: str | Path, output_dir: str | Path, every_n_frames: int = 1) -> list[Path]:
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be at least 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    results, index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % every_n_frames == 0:
            target = output_dir / f"frame_{index:06d}.jpg"
            if not cv2.imwrite(str(target), frame):
                cap.release()
                raise OSError(f"Failed to write {target}")
            results.append(target)
        index += 1
    cap.release()
    return results


def build_manifest(root: str | Path, output: str | Path) -> list[dict[str, object]]:
    root, output = Path(root), Path(output)
    rows = []
    for path in sorted(
        p for p in root.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    ):
        try:
            rows.append(inspect_video(path))
        except ValueError as error:
            rows.append({"path": str(path), "valid": False, "error": str(error)})
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            if writer.fieldnames:
                writer.writeheader()
                writer.writerows(rows)
    return rows


def split_by_participant(
    records: list[dict[str, object]],
    *,
    seed: int = 42,
    train: float = 0.7,
    validation: float = 0.15,
) -> dict[str, list[dict[str, object]]]:
    """Split recording metadata by participant ID to avoid subject leakage."""
    if train <= 0 or validation < 0 or train + validation >= 1:
        raise ValueError("Require train > 0, validation >= 0, and train + validation < 1")
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        participant = record.get("participant_id")
        if participant is None:
            raise ValueError("Every record needs participant_id for a leakage-safe split")
        grouped.setdefault(str(participant), []).append(record)
    participants = sorted(grouped)
    random.Random(seed).shuffle(participants)
    train_end = int(len(participants) * train)
    validation_end = train_end + int(len(participants) * validation)
    assignments = {
        "train": participants[:train_end],
        "validation": participants[train_end:validation_end],
        "test": participants[validation_end:],
    }
    return {
        name: [record for participant in ids for record in grouped[participant]]
        for name, ids in assignments.items()
    }


def create_synthetic_video(path: str | Path, frames: int = 48, fps: int = 24) -> Path:
    """Create a labelled-for-testing stick-person clip; it is not research data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    if not writer.isOpened():
        raise OSError("OpenCV could not initialize an mp4v writer")
    for frame_index in range(frames):
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        phase = 2 * np.pi * frame_index / max(frames, 1)
        y = int(150 + 30 * np.sin(phase))
        cv2.circle(frame, (160, y - 65), 14, (30, 30, 30), -1)
        cv2.line(frame, (160, y - 50), (160, y), (30, 30, 30), 5)
        cv2.line(frame, (160, y - 40), (125, y - 15), (30, 30, 30), 4)
        cv2.line(frame, (160, y - 40), (195, y - 15), (30, 30, 30), 4)
        cv2.line(frame, (160, y), (140, y + 42), (30, 30, 30), 5)
        cv2.line(frame, (160, y), (180, y + 42), (30, 30, 30), 5)
        writer.write(frame)
    writer.release()
    return path
