# MotionIQ

MotionIQ is a research software prototype for video-based human movement analysis. It provides reproducible video ingestion, optional MediaPipe pose estimation and YOLO segmentation adapters, geometric movement features, lightweight classification baselines, a FastAPI service, and a React dashboard.

**Research status:** outputs depend on optional model packages and weights. Pose estimates are not validated against a dataset in this repository. Movement measurements requiring calibration, predictions, and evaluation remain explicitly unavailable until configured. This is not a clinical tool.

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

The dashboard is at http://localhost:5173 and API docs at http://localhost:8000/docs. Install optional vision dependencies with `python -m pip install -e '.[vision]'`. Place compatible YOLO segmentation weights at `models/yolo11n-seg.pt` to enable segmentation; the application does not download model weights automatically.

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
- `tests/`: geometry, video and baseline smoke coverage
- `data/`: ignored location for user-downloaded datasets
- `models/`: ignored location for trained weights

See [ARCHITECTURE.md](ARCHITECTURE.md), [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md), [EXPERIMENTS.md](EXPERIMENTS.md), and [LIMITATIONS.md](LIMITATIONS.md).
