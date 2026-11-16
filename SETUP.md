# Setup

## Supported local environment

Use Python 3.11 or 3.12, Node.js 20+, and a working OpenCV video codec backend. The project intentionally pins Python dependencies and separates optional vision/training extras. On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install -e '.[vision]' # optional; MediaPipe and Ultralytics
python -m pip install -e '.[training]' # optional; PyTorch
```

For controlled reproducibility, generate and commit a platform-specific lock file after resolving packages on the target platform. Obtain YOLO segmentation weights from an authorized source, place them at `models/yolo11n-seg.pt`, and keep model artifacts outside Git. The vision extra is optional because MediaPipe wheel/runtime support varies by platform and display/GPU environment.

## Start services

Backend: `uvicorn motioniq.api:app --reload` (default port 8000).

Frontend: `cd frontend && npm install && npm run dev` (port 5173). For a remote API set `VITE_API_URL` before starting Vite and configure API CORS for the dashboard origin.

## Quality checks

```bash
ruff check motioniq tests
pytest
cd frontend && npm run build
```

The checked-in tests use a generated synthetic clip and tiny in-memory arrays; they do not download a dataset or train a large model.
