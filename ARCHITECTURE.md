# Architecture

## Processing flow

1. The dashboard uploads an allowed video container to `POST /api/videos`.
2. FastAPI stores it under ignored `uploads/`, validates dimensions/FPS/frame count with OpenCV, then samples up to 60 frames.
3. Optional MediaPipe produces frame-wise normalized landmarks with per-landmark visibility. Optional Ultralytics YOLO segmentation runs on the first sampled frame when a local weights file exists. Each adapter reports `unavailable` if its package/model cannot initialize. When a model returns output, the first sampled frame is rendered with a pose or segmentation overlay preview.
4. Analysis returns model outputs, explicit unavailable states for unconfigured measurements/prediction/evaluation, and observed processing latency.
5. `GET /api/videos/{id}/report.csv` exports sampled landmarks.

The service keeps result metadata in process memory and is for local research demos. Restarting it clears result lookup; files are not automatically expired. No authentication, queue, persistence database, or multi-worker coordination is implemented.

## Analysis and models

`geometry.py` contains 2D joint angle, trunk inclination, displacement, symmetry, peak detection, cycle duration, and temporal features. Coordinates must have consistent units. Inference from one camera is 2D and does not establish physical joint angles in 3D.

`models.py` supplies a balanced ExtraTrees baseline for engineered feature vectors and an optional bidirectional GRU for sequences shaped `(samples, frames, features)`. Training functions are deliberately small. User-supplied participant splits and labels are needed; there is no model trained on real research data in this repository.

`metrics.py` implements MPJPE, PCK, IoU, Dice, and weighted classification scores. Metrics require correctly aligned ground truth and predictions and are not computed from synthetic clips.

## Reproducibility and performance

Seed Python, NumPy and PyTorch via `set_seed`. Use `split_by_participant` before fitting to avoid participant leakage. Record model version, dependency lock, dataset version, participant split, coordinate convention, thresholds, and device with every experiment. The current service samples frames and reports wall-clock timing; it does not guarantee real-time inference.
