# MotionIQ

MotionIQ is a research software prototype for video-based human movement analysis. It provides local video ingestion, optional Ultralytics YOLO11 pose and person-segmentation inference, image-space movement features, trainable classification baselines, a FastAPI service, and a React dashboard.

**Research status:** this local research workspace has a trained REHAB24-6 Random Forest checkpoint and reports its held-out classification metrics. A fresh clone will not include the ignored dataset files or model checkpoint, and will report prediction as not configured until a checkpoint is trained or supplied. The YOLO11n pose and segmentation models are pretrained COCO models and have not been evaluated against manual MotionIQ ground truth. Pose/segmentation evaluation therefore remains not evaluated. Reported joint angles and normalized image displacements are uncalibrated 2D projections, not physical or clinical measurements.

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

Set `MOTIONIQ_SAMPLE_FPS` to change analysis sampling (default `3`). After upload, ground-truth COCO-17 pose landmarks can be submitted to `POST /api/videos/{video_id}/evaluation/pose` as JSON `{"frames":[{"frame":0,"landmarks":[[x,y], ...]}]}`. Coordinates must be normalized to `[0,1]`, match the predicted 17-keypoint order, and correspond to sampled source frame indices. That route computes MPJPE and PCK@0.05 only on annotated frames with a detected pose. Binary person masks can be posted to `/api/videos/{video_id}/evaluation/segmentation` as `{"frames":[{"frame":0,"mask":[[0,1,...], ...]}]}` using one row per image pixel; it reports mean IoU and Dice. Held-out classification precision/recall/F1 and confusion matrix are shown only from the trained classifier bundle's recorded test report.

## Reproducible exercise classifier

The first supported real-video adapter is REHAB24-6, under the dataset's academic/nonprofit noncommercial terms. Obtain the videos and official `Segmentation.csv` from [the Zenodo record](https://zenodo.org/records/13305826) only if your use fits those terms. Extract the videos without changing their names, then run:

```bash
python -m pip install -e '.[vision]'
python -m motioniq.training prepare-rehab24 \
  --annotations data/raw/rehab24/Segmentation.csv \
  --videos data/raw/rehab24/videos \
  --output data/processed/rehab24 \
  --camera Camera18
python -m motioniq.training train \
  --features data/processed/rehab24/features.csv \
  --checkpoint models/motioniq-rf.joblib \
  --seed 42
```

Preparation runs the existing YOLO pose/measurement pipeline on actual RGB frames, writes timestamped per-frame measurements and repetition-level angle/displacement/velocity features, and records subject IDs plus dataset/file hashes. Training makes participant-disjoint train/validation/test partitions, chooses tree settings from validation macro-F1, refits on train+validation, then calculates final metrics once on held-out test subjects. The bundle stores class mapping, feature schema, seed, package versions, split IDs and test report. Restart FastAPI after training; `GET /api/model/status` reports the loaded checkpoint. The React result shows the actual classifier output and test metrics only when this bundle exists. Random Forest class vote fractions are uncalibrated and the confidence threshold is a rejection heuristic, not a guaranteed out-of-distribution detector.

The files under `data/` and `models/motioniq-rf.joblib` are local, ignored artifacts and are not distributed by this repository. In this research workspace, 1,045 of 1,072 annotated repetitions were usable after actual YOLO feature extraction; the local checkpoint is Ready. Its held-out test scores and subject split are recorded in `models/motioniq-rf.status.json`. Do not train on the synthetic clips under `work/`; those are software fixtures only. See [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md) before acquiring or using research data.

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
