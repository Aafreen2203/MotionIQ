"""Reproducible REHAB24-6 feature preparation and subject-held-out training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from motioniq.analysis import analyze_video
from motioniq.metrics import classification_scores
from motioniq.models import (
    FEATURE_SCHEMA_VERSION,
    make_random_forest,
    save_classifier_bundle,
    sequence_features,
    set_seed,
)

REHAB24_URL = "https://zenodo.org/records/13305826"
REHAB24_LICENSE = "Academic/non-profit noncommercial research only; see source record"
EXERCISES = {
    1: "arm_abduction",
    2: "arm_vw",
    3: "push_up",
    4: "leg_abduction",
    5: "leg_lunge",
    6: "squat",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _find_rehab_video(root: Path, video_id: str, camera: str) -> Path:
    patterns = (
        f"**/{video_id}*{camera}*.mp4",
        f"**/{video_id}*{camera}*.mov",
    )
    matches = sorted(
        {path for pattern in patterns for path in root.glob(pattern) if path.is_file()}
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {camera} video for {video_id} under {root}; "
            f"found {len(matches)}."
        )
    return matches[0]


def prepare_rehab24_features(
    annotation_csv: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    *,
    camera: str = "Camera18",
    sample_fps: float = 3.0,
    max_frames: int = 300,
) -> Path:
    """Run MotionIQ pose/measurement extraction on actual REHAB24 RGB repetitions."""
    if camera not in {"Camera17", "Camera18"}:
        raise ValueError("camera must be Camera17 or Camera18")
    annotations_path, video_root, output_dir = (
        Path(annotation_csv),
        Path(video_root),
        Path(output_dir),
    )
    if not annotations_path.is_file() or not video_root.is_dir():
        raise FileNotFoundError(
            "Provide the official Segmentation.csv and extracted videos directory"
        )
    with annotations_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    required = {
        "video_id",
        "repetition_number",
        "exercise_id",
        "person_id",
        "first_frame",
        "last_frame",
        "correctness",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Segmentation.csv must include columns: {sorted(required)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, object]] = []
    frame_output = output_dir / "pose_measurement_frames.jsonl"
    analysis_cache: dict[Path, dict[str, object]] = {}
    video_paths: dict[str, Path] = {}
    remaining_repetitions = Counter(row["video_id"].strip() for row in rows)
    source_info: dict[str, dict[str, object]] = {}

    def release_video_reference(video_id: str, source: Path) -> None:
        remaining_repetitions[video_id] -= 1
        if remaining_repetitions[video_id] == 0:
            analysis_cache.pop(source, None)

    with frame_output.open("w", encoding="utf-8") as frame_stream:
        for annotation in rows:
            video_id = annotation["video_id"].strip()
            exercise_id = int(annotation["exercise_id"])
            if exercise_id not in EXERCISES:
                raise ValueError(f"Unsupported REHAB24 exercise_id: {exercise_id}")
            if video_id not in video_paths:
                video_paths[video_id] = _find_rehab_video(video_root, video_id, camera)
            source = video_paths[video_id]
            if source not in analysis_cache:
                result = analyze_video(
                    source,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                    write_artifacts=False,
                )
                analysis_cache[source] = result
                source_info[str(source)] = {
                    "sha256": _sha256(source),
                    "bytes": source.stat().st_size,
                    "sampled_frames": result["sampling"]["sampled_frames"],
                    "video_metadata": result["video"],
                }
                for pose_frame, measurement_frame in zip(
                    result["pose"]["frames"], result["movement_measurements"]["frames"]
                ):
                    frame_stream.write(
                        json.dumps(
                            {
                                "video_id": video_id,
                                "subject_id": str(annotation["person_id"]),
                                "source_frame": pose_frame["frame"],
                                "timestamp_seconds": pose_frame["timestamp_seconds"],
                                "landmarks": pose_frame.get("landmarks", []),
                                "measurements": measurement_frame,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
            result = analysis_cache[source]
            first_frame, last_frame = int(annotation["first_frame"]), int(annotation["last_frame"])
            measurement_frames = [
                frame
                for frame in result["movement_measurements"]["frames"]
                if first_frame <= int(frame["frame"]) <= last_frame
            ]
            if len(measurement_frames) < 2:
                release_video_reference(video_id, source)
                continue
            features = sequence_features(
                measurement_frames, int(result["video"]["width"]), int(result["video"]["height"])
            )
            valid_coverage = [
                value for key, value in features.items() if key.endswith("_valid_fraction")
            ]
            if not valid_coverage or max(valid_coverage) <= 0:
                release_video_reference(video_id, source)
                continue
            sample_id = f"{video_id}-{camera}-rep{annotation['repetition_number']}"
            feature_rows.append(
                {
                    **features,
                    "sample_id": sample_id,
                    "subject_id": str(annotation["person_id"]),
                    "video_id": video_id,
                    "camera": camera,
                    "repetition_number": int(annotation["repetition_number"]),
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "start_seconds": first_frame / 30.0,
                    "end_seconds": last_frame / 30.0,
                    "exercise_id": exercise_id,
                    "label": EXERCISES[exercise_id],
                    "correctness": int(annotation["correctness"]),
                    "cam17_orientation": annotation.get("cam17_orientation", ""),
                    "mocap_erroneous": int(annotation.get("mocap_erroneous", 0)),
                    "exercise_subtype": annotation.get("exercise_subtype", ""),
                    "lights_on": int(annotation.get("lights_on", 0)),
                    "extra_person_in_cam17": int(annotation.get("extra_person_in_cam17", 0)),
                    "extra_person_in_cam18": int(annotation.get("extra_person_in_cam18", 0)),
                    "frame_count": len(measurement_frames),
                    "source_path": str(source),
                }
            )
            release_video_reference(video_id, source)
    if not feature_rows:
        raise ValueError("No annotated repetition had enough valid detected movement frames")
    features_path = output_dir / "features.csv"
    pd.DataFrame(feature_rows).to_csv(features_path, index=False)
    provenance = {
        "dataset": "REHAB24-6",
        "source": REHAB24_URL,
        "license": REHAB24_LICENSE,
        "camera": camera,
        "sample_fps": sample_fps,
        "max_frames_per_video": max_frames,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_files": source_info,
        "annotation_sha256": _sha256(annotations_path),
        "annotation_rows": len(rows),
        "usable_repetitions": len(feature_rows),
        "excluded_repetitions": len(rows) - len(feature_rows),
        "subjects": sorted({row["subject_id"] for row in feature_rows}),
        "classes": sorted({row["label"] for row in feature_rows}),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "opencv": _installed_version("opencv-python"),
            "ultralytics": _installed_version("ultralytics"),
            "torch": _installed_version("torch"),
            "numpy": _installed_version("numpy"),
        },
        "inference_weights": {
            str(path): _sha256(path)
            for path in (Path("models/yolo11n-pose.pt"), Path("models/yolo11n-seg.pt"))
            if path.is_file()
        },
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return features_path


def grouped_split(
    subjects: np.ndarray,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.16,
) -> np.ndarray:
    """Return train/validation/test tags with no subject shared across partitions."""
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must be positive and sum to less than one")
    indices = np.arange(len(subjects))
    test_split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    remaining, test = next(test_split.split(indices, groups=subjects))
    relative_validation = validation_fraction / (1 - test_fraction)
    validation_split = GroupShuffleSplit(
        n_splits=1, test_size=relative_validation, random_state=seed + 1
    )
    train_local, validation_local = next(
        validation_split.split(remaining, groups=subjects[remaining])
    )
    partitions = np.full(len(subjects), "test", dtype=object)
    partitions[remaining[train_local]] = "train"
    partitions[remaining[validation_local]] = "validation"
    return partitions


def train_from_features(
    features_csv: str | Path,
    checkpoint: str | Path,
    *,
    seed: int = 42,
    confidence_threshold: float = 0.55,
) -> dict[str, object]:
    """Tune only against validation subjects; calculate final metrics on held-out subjects."""
    if not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in (0, 1]")
    data = pd.read_csv(features_csv)
    required = {"subject_id", "label", "sample_id"}
    if not required.issubset(data.columns):
        raise ValueError(f"Feature file missing required metadata columns: {sorted(required)}")
    excluded = required | {
        "video_id",
        "camera",
        "cam17_orientation",
        "mocap_erroneous",
        "exercise_subtype",
        "lights_on",
        "extra_person_in_cam17",
        "extra_person_in_cam18",
        "repetition_number",
        "first_frame",
        "last_frame",
        "start_seconds",
        "end_seconds",
        "exercise_id",
        "correctness",
        "frame_count",
        "source_path",
    }
    feature_names = [name for name in data.columns if name not in excluded]
    if not feature_names or data["label"].nunique() < 2 or data["subject_id"].nunique() < 5:
        raise ValueError(
            "Training requires features, at least two classes, and at least five subjects"
        )
    if data["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    data["split"] = grouped_split(data["subject_id"].astype(str).to_numpy(), seed=seed)
    train, validation, test = (data[data.split == name] for name in ("train", "validation", "test"))
    if train["label"].nunique() != data["label"].nunique() or validation.empty or test.empty:
        raise ValueError(
            "Subject split did not produce train classes plus validation and test samples"
        )
    x_train = train[feature_names].to_numpy(dtype=float)
    y_train = train["label"].astype(str).to_numpy()
    x_validation = validation[feature_names].to_numpy(dtype=float)
    y_validation = validation["label"].astype(str).to_numpy()
    classes = sorted(data["label"].astype(str).unique())
    best_score, best_parameters = -1.0, None
    for n_estimators, min_samples_leaf, max_features in (
        (300, 1, "sqrt"),
        (500, 2, "sqrt"),
        (500, 3, 0.7),
    ):
        model = make_random_forest(
            seed=seed,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        model.fit(x_train, y_train)
        score = float(
            f1_score(
                y_validation,
                model.predict(x_validation),
                labels=classes,
                average="macro",
                zero_division=0,
            )
        )
        if score > best_score:
            best_score = score
            best_parameters = {
                "n_estimators": n_estimators,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
    final_model = make_random_forest(seed=seed, **best_parameters)
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model.fit(
        train_validation[feature_names].to_numpy(dtype=float), train_validation["label"].astype(str)
    )
    y_test = test["label"].astype(str).to_numpy()
    predicted = final_model.predict(test[feature_names].to_numpy(dtype=float)).astype(str)
    metrics = classification_scores(y_test, predicted, labels=classes)
    split_subjects = {
        key: sorted(data.loc[data.split == key, "subject_id"].astype(str).unique().tolist())
        for key in ("train", "validation", "test")
    }
    if any(
        set(split_subjects[a]) & set(split_subjects[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("Subject leakage detected")
    provenance_path = Path(features_csv).parent / "provenance.json"
    dataset_provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {"feature_table": str(Path(features_csv).resolve())}
    )
    metadata = {
        "seed": seed,
        "confidence_threshold": confidence_threshold,
        "training_algorithm": "scikit-learn RandomForestClassifier",
        "preprocessing": "median imputation with missingness indicators; no scaling for tree model",
        "imbalance": "balanced_subsample class weights",
        "tuning": "macro-F1 on validation subjects only; refit on train+validation",
        "validation_macro_f1": best_score,
        "best_parameters": best_parameters,
        "held_out_test_metrics": metrics,
        "held_out_test_sample_ids": test["sample_id"].tolist(),
        "split_subjects": split_subjects,
        "split_counts": {
            key: int((data.split == key).sum()) for key in ("train", "validation", "test")
        },
        "source_features_sha256": _sha256(Path(features_csv)),
        "dataset_provenance": dataset_provenance,
    }
    save_classifier_bundle(
        final_model,
        checkpoint,
        feature_names=feature_names,
        class_labels=classes,
        metadata=metadata,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-rehab24")
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--videos", required=True)
    prepare.add_argument("--output", default="data/processed/rehab24")
    prepare.add_argument("--camera", choices=("Camera17", "Camera18"), default="Camera18")
    prepare.add_argument("--sample-fps", type=float, default=3.0)
    prepare.add_argument("--max-frames", type=int, default=300)
    train = commands.add_parser("train")
    train.add_argument("--features", required=True)
    train.add_argument("--checkpoint", default="models/motioniq-rf.joblib")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--confidence-threshold", type=float, default=0.55)
    args = parser.parse_args()
    set_seed(getattr(args, "seed", 42))
    if args.command == "prepare-rehab24":
        path = prepare_rehab24_features(
            args.annotations,
            args.videos,
            args.output,
            camera=args.camera,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
        )
        print(path)
    else:
        checkpoint = Path(args.checkpoint)
        status_path = checkpoint.with_suffix(".status.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"status": "training", "started_at_utc": datetime.now(UTC).isoformat()}),
            encoding="utf-8",
        )
        try:
            metadata = train_from_features(
                args.features,
                args.checkpoint,
                seed=args.seed,
                confidence_threshold=args.confidence_threshold,
            )
            status_path.write_text(
                json.dumps({"status": "ready", "metadata": metadata}, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(metadata, indent=2))
        except Exception as error:
            status_path.write_text(
                json.dumps({"status": "failed", "reason": str(error)}, indent=2),
                encoding="utf-8",
            )
            raise


if __name__ == "__main__":
    main()
