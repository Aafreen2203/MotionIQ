# Setup

## Supported local environment

Use Python 3.11 or 3.12, Node.js 20+, and a working OpenCV video codec backend. The project intentionally pins Python dependencies and separates optional vision/training extras. On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install -e '.[vision]' # optional; PyTorch CPU and Ultralytics inference runtime
python -m pip install -e '.[training]' # optional; PyTorch model-training tools
python -m motioniq.download_models # downloads checksum-verified YOLO11n pose/seg weights (~12 MB)
```

For controlled reproducibility, generate and commit a platform-specific lock file after resolving packages on the target platform. Downloaded model artifacts stay under ignored `models/`. Inference uses pinned Ultralytics 8.3.40, PyTorch 2.5.1 and the official Ultralytics assets release v8.3.0 `yolo11n-pose.pt` (COCO 17 keypoints) and `yolo11n-seg.pt` (COCO person class), on CPU by default. The small/nano models favor local execution speed; they are not MotionIQ-trained or validated. Google Colab GPU training remains available through the optional `training` extra and `motioniq.models` training functions; no model is trained automatically.

The pose model provides 17 COCO landmarks. Therefore ankle angles that need foot/toe landmarks are reported as unavailable. Knee and hip angles are the 2D angle ABC at knee/hip using normalized image-plane coordinates. Trunk inclination is the signed shoulder-midpoint-to-hip-midpoint angle from image vertical. Symmetry compares left/right normalized vertical hip-to-ankle changes between detected samples. These are image-space descriptors; physical distances and clinical biomechanics require calibration and validation. Repetition counting remains unconfigured because no task-specific signal and thresholds are defined.

## Start services

Backend: `uvicorn motioniq.api:app --reload` (default port 8000).

Frontend: `cd frontend && npm install && npm run dev` (port 5175). By default it calls `http://localhost:8000` directly. For a remote API set `VITE_API_URL` before starting Vite and add the dashboard origin to `MOTIONIQ_CORS_ORIGINS`.

## Quality checks

```bash
ruff check motioniq tests
pytest
cd frontend && npm run build
```

The checked-in tests use a generated synthetic clip and tiny in-memory arrays; they do not download a dataset or train a large model.
