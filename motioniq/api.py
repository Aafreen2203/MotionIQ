"""FastAPI upload and analysis service."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from motioniq.analysis import analyze_video, export_measurements_csv
from motioniq.metrics import mpjpe, pck, segmentation_scores
from motioniq.models import load_classifier_bundle, predict_classifier_bundle, sequence_features
from motioniq.video import inspect_video

api = FastAPI(title="MotionIQ", version="0.1.0")
UPLOAD_DIR = Path("uploads")
RESULTS: dict[str, dict[str, object]] = {}
SEGMENTATION_MASKS: dict[str, dict[str, np.ndarray]] = {}
ALLOWED = {".mp4", ".mov", ".avi", ".mkv"}
MODEL_CHECKPOINT = Path(os.getenv("MOTIONIQ_MODEL_PATH", "models/motioniq-rf.joblib"))
CLASSIFIER_BUNDLE: dict[str, object] | None = None
CLASSIFIER_LOAD_ERROR: str | None = None
if MODEL_CHECKPOINT.is_file():
    try:
        CLASSIFIER_BUNDLE = load_classifier_bundle(MODEL_CHECKPOINT)
    except Exception as error:  # A corrupt or incompatible checkpoint must be visible in status.
        CLASSIFIER_LOAD_ERROR = str(error)
CLASSIFIER_MTIME = MODEL_CHECKPOINT.stat().st_mtime if CLASSIFIER_BUNDLE is not None else None


@api.get("/health")
def health():
    return {"status": "ok", "service": "motioniq"}


def classifier_status() -> dict[str, object]:
    global CLASSIFIER_BUNDLE, CLASSIFIER_LOAD_ERROR, CLASSIFIER_MTIME
    status_path = MODEL_CHECKPOINT.with_suffix(".status.json")
    training_status = {}
    if status_path.is_file():
        try:
            training_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            training_status = {}
    if training_status.get("status") == "training":
        return {"status": "training", "checkpoint": str(MODEL_CHECKPOINT)}
    checkpoint_mtime = MODEL_CHECKPOINT.stat().st_mtime if MODEL_CHECKPOINT.is_file() else None
    if checkpoint_mtime is not None and checkpoint_mtime != CLASSIFIER_MTIME:
        try:
            CLASSIFIER_BUNDLE = load_classifier_bundle(MODEL_CHECKPOINT)
            CLASSIFIER_LOAD_ERROR = None
            CLASSIFIER_MTIME = checkpoint_mtime
        except Exception as error:
            CLASSIFIER_BUNDLE = None
            CLASSIFIER_LOAD_ERROR = str(error)
            CLASSIFIER_MTIME = checkpoint_mtime
    if CLASSIFIER_BUNDLE is None and training_status.get("status") == "failed":
        CLASSIFIER_LOAD_ERROR = str(training_status.get("reason", "Training failed"))
    if CLASSIFIER_BUNDLE is None:
        return {
            "status": "failed" if CLASSIFIER_LOAD_ERROR else "not_configured",
            "reason": CLASSIFIER_LOAD_ERROR or f"No trained checkpoint at {MODEL_CHECKPOINT}.",
            "checkpoint": str(MODEL_CHECKPOINT),
        }
    metadata = CLASSIFIER_BUNDLE.get("metadata", {})
    return {
        "status": "ready",
        "model_version": CLASSIFIER_BUNDLE["model_version"],
        "class_labels": CLASSIFIER_BUNDLE["class_labels"],
        "feature_schema_version": CLASSIFIER_BUNDLE["feature_schema_version"],
        "checkpoint": str(MODEL_CHECKPOINT),
        "evaluation": metadata.get("held_out_test_metrics"),
        "training_metadata": metadata,
        "last_training_failure": (
            training_status.get("reason") if training_status.get("status") == "failed" else None
        ),
    }


@api.get("/api/model/status")
def get_model_status():
    return classifier_status()


@api.get("/api/videos")
def list_videos():
    """List valid and invalid video files found in the local upload library."""
    if not UPLOAD_DIR.is_dir():
        return {"videos": [], "count": 0}

    videos = []
    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        try:
            metadata = inspect_video(path)
            metadata.pop("path", None)
            videos.append(
                {
                    "video_id": path.stem,
                    "filename": path.name,
                    "status": "available",
                    "video": metadata,
                    "analysis_available": path.stem in RESULTS,
                }
            )
        except (OSError, ValueError) as error:
            videos.append(
                {
                    "video_id": path.stem,
                    "filename": path.name,
                    "status": "invalid",
                    "error": str(error),
                    "analysis_available": path.stem in RESULTS,
                }
            )
    return {"videos": videos, "count": len(videos)}


@api.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, "Upload an MP4, MOV, AVI, or MKV video")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{video_id}{suffix}"
    with path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)
    try:
        results = analyze_video(path)
    except (ValueError, OSError) as error:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"Invalid video: {error}") from error
    results["video_id"] = video_id
    if CLASSIFIER_BUNDLE is not None:
        measurements = results.get("movement_measurements", {})
        rows = measurements.get("frames", []) if isinstance(measurements, dict) else []
        if rows:
            features = sequence_features(
                rows, int(results["video"]["width"]), int(results["video"]["height"])
            )
            threshold = float(
                CLASSIFIER_BUNDLE.get("metadata", {}).get("confidence_threshold", 0.55)
            )
            results["prediction"] = predict_classifier_bundle(
                CLASSIFIER_BUNDLE, features, confidence_threshold=threshold
            )
        else:
            results["prediction"] = {
                "status": "failed",
                "reason": "No movement measurement frames are available for inference.",
                "model_version": CLASSIFIER_BUNDLE["model_version"],
            }
    metrics = classifier_status().get("evaluation")
    if isinstance(metrics, dict):
        results["evaluation"] = {
            "status": "completed",
            "tasks": {"classification": metrics},
            "reason": None,
        }
    SEGMENTATION_MASKS[video_id] = results.pop("segmentation_masks", {})
    results["video_url"] = f"/api/videos/{video_id}/file"
    source_path = Path(results["video"]["path"])
    pose_artifact = source_path.with_name(f"{source_path.stem}.pose.mp4")
    segmentation_artifact = source_path.with_name(f"{source_path.stem}.segmentation.mp4")
    if (
        results["processed_video"]["status"] == "completed"
        and pose_artifact.is_file()
        and pose_artifact.stat().st_size > 0
    ):
        results["overlay"]["url"] = f"/api/videos/{video_id}/processed"
        results["processed_video"]["url"] = f"/api/videos/{video_id}/processed"
    elif results["processed_video"]["status"] == "completed":
        results["processed_video"].update(
            status="failed", reason="Validated pose video artifact is missing or empty."
        )
        results["overlay"].update(status="failed", url=None)
    if (
        results["segmentation_video"]["status"] == "completed"
        and segmentation_artifact.is_file()
        and segmentation_artifact.stat().st_size > 0
    ):
        results["segmentation_video"]["url"] = f"/api/videos/{video_id}/segmentation"
        results["segmentation"]["overlay_url"] = results["segmentation_video"]["url"]
    elif results["segmentation_video"]["status"] == "completed":
        results["segmentation_video"].update(
            status="failed", reason="Validated segmentation video artifact is missing or empty."
        )
    RESULTS[video_id] = results
    return results


@api.get("/api/videos/{video_id}")
def get_results(video_id: str):
    if video_id not in RESULTS:
        raise HTTPException(404, "Analysis not found")
    return RESULTS[video_id]


@api.post("/api/videos/{video_id}/evaluation/pose")
def evaluate_pose(video_id: str, annotation: dict[str, object]):
    """Evaluate normalized COCO-17 landmarks against supplied per-frame ground truth."""
    result = RESULTS.get(video_id)
    if result is None:
        raise HTTPException(404, "Analysis not found")
    annotated = annotation.get("frames")
    if not isinstance(annotated, list) or not annotated:
        raise HTTPException(422, "Provide a non-empty frames array with frame and landmarks fields")
    predictions = {frame["frame"]: frame for frame in result.get("pose", {}).get("frames", [])}
    predicted_points, target_points, evaluated_frames = [], [], 0
    try:
        for frame in annotated:
            detected = predictions.get(int(frame["frame"]))
            target = np.asarray(frame["landmarks"], dtype=float)
            if detected is None or not detected.get("landmarks"):
                continue
            predicted = np.asarray(
                [[point["x"], point["y"]] for point in detected["landmarks"]], dtype=float
            )
            if target.shape != predicted.shape or target.ndim != 2 or target.shape[1] != 2:
                raise ValueError(
                    "Each target frame must match the predicted number of [x, y] landmarks."
                )
            if not np.isfinite(target).all():
                raise ValueError(
                    "Ground-truth landmarks must contain finite normalized coordinates"
                )
            if np.any(target < 0) or np.any(target > 1):
                raise ValueError(
                    "Ground-truth coordinates must be normalized to the [0, 1] interval"
                )
            predicted_points.append(predicted)
            target_points.append(target)
            evaluated_frames += 1
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, f"Invalid pose annotations: {error}") from error
    if not evaluated_frames:
        raise HTTPException(422, "No annotated frame had a detected pose to compare")
    predicted_array, target_array = np.concatenate(predicted_points), np.concatenate(target_points)
    evaluation = {
        "status": "completed",
        "task": "pose",
        "evaluated_frames": evaluated_frames,
        "landmark_count": int(predicted_array.shape[0]),
        "coordinate_space": "normalized image coordinates",
        "mpjpe": mpjpe(predicted_array, target_array),
        "pck_threshold": 0.05,
        "pck": pck(predicted_array, target_array, threshold=0.05),
        "ground_truth_supplied": True,
    }
    current = result.get("evaluation", {})
    tasks = current.get("tasks", {}) if isinstance(current, dict) else {}
    tasks["pose"] = evaluation
    result["evaluation"] = {"status": "completed", "tasks": tasks}
    return evaluation


@api.post("/api/videos/{video_id}/evaluation/segmentation")
def evaluate_segmentation(video_id: str, annotation: dict[str, object]):
    """Evaluate sampled binary person masks against supplied pixel masks."""
    result = RESULTS.get(video_id)
    if result is None:
        raise HTTPException(404, "Analysis not found")
    annotated = annotation.get("frames")
    if not isinstance(annotated, list) or not annotated:
        raise HTTPException(422, "Provide a non-empty frames array with frame and mask fields")
    predicted_masks = SEGMENTATION_MASKS.get(video_id, {})
    scores = []
    try:
        expected_shape = (int(result["video"]["height"]), int(result["video"]["width"]))
        for frame in annotated:
            predicted = predicted_masks.get(str(int(frame["frame"])))
            if predicted is None:
                continue
            target = np.asarray(frame["mask"], dtype=bool)
            if target.shape != expected_shape:
                raise ValueError(f"Each ground-truth mask must have shape {expected_shape}")
            target_at_model_resolution = cv2.resize(
                target.astype(np.uint8),
                (predicted.shape[1], predicted.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            scores.append(segmentation_scores(predicted, target_at_model_resolution))
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, f"Invalid segmentation annotations: {error}") from error
    if not scores:
        raise HTTPException(422, "No annotated frame had a stored segmentation prediction")
    evaluation = {
        "status": "completed",
        "task": "segmentation",
        "evaluated_frames": len(scores),
        "mean_iou": float(np.mean([score["iou"] for score in scores])),
        "mean_dice": float(np.mean([score["dice"] for score in scores])),
        "metric_resolution": "model mask resolution (ground truth resized with nearest-neighbor)",
        "ground_truth_supplied": True,
    }
    current = result.get("evaluation", {})
    tasks = current.get("tasks", {}) if isinstance(current, dict) else {}
    tasks["segmentation"] = evaluation
    result["evaluation"] = {"status": "completed", "tasks": tasks}
    return evaluation


@api.get("/api/videos/{video_id}/file")
def get_video(video_id: str):
    result = RESULTS.get(video_id)
    if result is None:
        raise HTTPException(404, "Analysis not found")
    path = Path(result["video"]["path"])
    return FileResponse(path)


@api.get("/api/videos/{video_id}/overlay")
def get_overlay(video_id: str):
    return get_processed(video_id)


def _analysis_artifact(video_id: str, key: str, suffix: str):
    result = RESULTS.get(video_id)
    if result is None or result[key]["status"] != "completed":
        raise HTTPException(404, "Processed artifact is unavailable")
    video_path = Path(result["video"]["path"])
    path = video_path.with_name(f"{video_path.stem}.{suffix}.mp4")
    if not path.is_file():
        raise HTTPException(404, "Processed artifact file is missing")
    return FileResponse(path, media_type="video/mp4")


@api.get("/api/videos/{video_id}/processed")
def get_processed(video_id: str):
    return _analysis_artifact(video_id, "processed_video", "pose")


@api.get("/api/videos/{video_id}/segmentation")
def get_segmentation(video_id: str):
    return _analysis_artifact(video_id, "segmentation_video", "segmentation")


@api.get("/api/videos/{video_id}/report.csv")
def get_csv(video_id: str):
    if video_id not in RESULTS:
        raise HTTPException(404, "Analysis not found")
    path = export_measurements_csv(RESULTS[video_id], UPLOAD_DIR / f"{video_id}.csv")
    return FileResponse(path, media_type="text/csv", filename=f"{video_id}-measurements.csv")


_default_origins = (
    "http://localhost:5173,http://127.0.0.1:5173," "http://localhost:5175,http://127.0.0.1:5175"
)
origins = os.getenv("MOTIONIQ_CORS_ORIGINS", _default_origins)
app = CORSMiddleware(
    app=api,
    allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)
