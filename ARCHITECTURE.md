# Architecture

## Processing flow

1. The React dashboard uploads a local MP4, MOV, AVI, or MKV to `POST /api/videos`.
2. FastAPI stores the file under ignored `uploads/`, validates it with OpenCV, and captures dimensions, FPS, frame count, duration and file size.
3. The analyzer samples at `MOTIONIQ_SAMPLE_FPS` (default 3 FPS, capped at 300 samples). It runs the pretrained Ultralytics YOLO11n COCO-17 pose model and COCO person segmentation model on CPU when their checksum-verified weights and optional inference dependencies are present.
4. Each sampled frame records source frame index/time and normalized/pixel coordinates or a clear no-person result. The service writes sampled MP4 pose and segmentation overlay artifacts. It also retains low-resolution binary masks in memory for optional segmentation evaluation.
5. Biomechanical descriptors use COCO-17 indices (shoulders 5/6, hips 11/12, knees 13/14, ankles 15/16) and reject nonfinite, out-of-frame, or confidence-below-0.25 points. Internal knee/hip angles are computed from pixel coordinates as `acos(((A-B)·(C-B))/(|A-B||C-B|))`; knee flexion is `180° - internal angle`. Pixel coordinates avoid the aspect-ratio distortion caused by directly measuring in normalized x/y coordinates. Trunk angle is measured from image vertical, positive toward image-right. It cannot identify anatomical forward/backward lean without known sagittal camera orientation. Symmetry is `1 - mean(|ΔL-ΔR|)/mean(|ΔL|+|ΔR|)` for paired changes in normalized vertical ankle-minus-hip separation; it is a 0–1 coefficient, not a percentage. If the denominator is zero, no score is reported. Responses include valid counts per feature and the number of valid symmetry frames/intervals. Repetition counts and ankle angles remain unconfigured because no task-specific thresholds or foot keypoints are available.
6. Prediction is `not_configured` without a trained MotionIQ checkpoint and class map. Evaluation is `not_evaluated` until annotations are submitted. Pose landmarks can be evaluated with MPJPE/PCK; person masks with IoU/Dice. Classification metrics are available in `metrics.py` when matching labels and predictions exist.
7. `GET /api/videos/{id}/report.csv` exports timestamped pose landmarks and pixel/normalized coordinates. The dashboard presents source and processed videos, overlays, landmarks, measurements and honest status/reason text.

The service keeps result metadata and low-resolution segmentation masks in process memory. Restarting it clears lookups and evaluation data; uploaded and processed files are not automatically expired. No authentication, queue, database, upload-size cap, or multi-worker coordination is implemented. CORS defaults to the local Vite development origins and is configurable with `MOTIONIQ_CORS_ORIGINS`.

## Analysis and models

`vision.py` adapts CPU YOLO11n pose and segmentation models. `geometry.py` contains 2D joint angles, trunk inclination, displacement, symmetry and peak/cycle helpers. Coordinates and thresholds must use consistent units. Single-camera 2D output is not a calibrated physical measurement.

`models.py` supplies a trainable ExtraTrees baseline and optional bidirectional GRU for feature sequences. No trained classifier checkpoint or task-specific class map is included. Training needs labeled recordings and participant-disjoint splits.

`metrics.py` implements MPJPE, PCK, IoU, Dice and weighted classification scores. The API exposes pose/segmentation evaluation routes and computes metrics only against supplied annotations for matching sampled frames.

## Reproducibility and performance

Inference versions are pinned in `pyproject.toml`; `python -m motioniq.download_models` fetches fixed official model assets and verifies SHA-256. Seed Python, NumPy and PyTorch via `set_seed`. Use `split_by_participant` before fitting to avoid participant leakage. Record model version, dependency lock, dataset version, split, coordinate convention, thresholds and device in experiments. The service reports processing time but does not promise real-time inference.
