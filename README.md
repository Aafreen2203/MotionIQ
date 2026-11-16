# MotionIQ

MotionIQ is a research software prototype for video-based human movement analysis. It provides local video ingestion, optional Ultralytics YOLO11 pose and person-segmentation inference, image-space movement features, trainable classification baselines, a FastAPI service, and a React dashboard.

**Research status:** pose and segmentation use pretrained COCO YOLO11n models when installed and downloaded. These outputs have not been validated against a MotionIQ dataset. Prediction remains unconfigured until a trained checkpoint and class map are supplied; evaluation remains not evaluated until ground-truth annotations are supplied. Reported joint angles and normalized image displacements are uncalibrated 2D projections, not physical or clinical measurements.

## Quick start

Requires Python 3.11 or 3.12 and Node.js 20 or newer. See [SETUP.md](SETUP.md) for platform details.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn motioniq.api:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The dashboard is at http://localhost:5175 and API docs at http://localhost:8000/docs. The frontend calls the configurable FastAPI origin directly; CORS allows the local development origins. Install CPU inference dependencies with `python -m pip install -e '.[vision]'`, then download checksum-verified YOLO11n COCO pose and segmentation weights with `python -m motioniq.download_models`.

Set `MOTIONIQ_SAMPLE_FPS` to change analysis sampling (default `3`). After upload, ground-truth COCO-17 pose landmarks can be submitted to `POST /api/videos/{video_id}/evaluation/pose` as JSON `{"frames":[{"frame":0,"landmarks":[[x,y], ...]}]}`. Coordinates must be normalized to `[0,1]`, match the predicted 17-keypoint order, and correspond to sampled source frame indices. That route computes MPJPE and PCK@0.05 only on annotated frames with a detected pose. Binary person masks can be posted to `/api/videos/{video_id}/evaluation/segmentation` as `{"frames":[{"frame":0,"mask":[[0,1,...], ...]}]}` using one row per image pixel; it reports mean IoU and Dice. Classification precision/recall/F1 and confusion matrix are available in the metrics module but cannot be surfaced until a trained checkpoint and class predictions exist.

## CLI

```bash
python -m motioniq.cli inspect path/to/video.mp4
python -m motioniq.cli extract-frames path/to/video.mp4 data/frames --every 5
python -m motioniq.cli manifest data/raw data/manifest.json
```

Software-only sample clip (never research data):

```bash
python -c 'from motioniq.video import create_synthetic_video; create_synthetic_video("work/synthetic.mp4")'
```

## Layout

- `motioniq/`: ingestion, geometry, metrics, model and service modules
- `frontend/`: React + TypeScript dashboard
- `tests/`: geometry, sampled analysis/artifacts, API/CORS, annotation metrics, and baseline coverage
- `data/`: ignored location for user-downloaded datasets
- `models/`: ignored location for trained weights

See [ARCHITECTURE.md](ARCHITECTURE.md), [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md), [EXPERIMENTS.md](EXPERIMENTS.md), and [LIMITATIONS.md](LIMITATIONS.md).
