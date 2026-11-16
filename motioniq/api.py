"""FastAPI upload and analysis service."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from motioniq.analysis import analyze_video, export_measurements_csv

app = FastAPI(title="MotionIQ", version="0.1.0")
origins = os.getenv("MOTIONIQ_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path("uploads")
RESULTS: dict[str, dict[str, object]] = {}
ALLOWED = {".mp4", ".mov", ".avi", ".mkv"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "motioniq"}


@app.post("/api/videos")
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
    results["video_url"] = f"/api/videos/{video_id}/file"
    if results["overlay"]["status"] == "available":
        results["overlay"]["url"] = f"/api/videos/{video_id}/overlay"
    RESULTS[video_id] = results
    return results


@app.get("/api/videos/{video_id}")
def get_results(video_id: str):
    if video_id not in RESULTS:
        raise HTTPException(404, "Analysis not found")
    return RESULTS[video_id]


@app.get("/api/videos/{video_id}/file")
def get_video(video_id: str):
    result = RESULTS.get(video_id)
    if result is None:
        raise HTTPException(404, "Analysis not found")
    path = Path(result["video"]["path"])
    return FileResponse(path)


@app.get("/api/videos/{video_id}/overlay")
def get_overlay(video_id: str):
    result = RESULTS.get(video_id)
    if result is None or result["overlay"]["status"] != "available":
        raise HTTPException(404, "No pose or segmentation overlay is available")
    video_path = Path(result["video"]["path"])
    path = video_path.with_name(f"{video_path.stem}.overlay.jpg")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/videos/{video_id}/report.csv")
def get_csv(video_id: str):
    if video_id not in RESULTS:
        raise HTTPException(404, "Analysis not found")
    path = export_measurements_csv(RESULTS[video_id], UPLOAD_DIR / f"{video_id}.csv")
    return FileResponse(path, media_type="text/csv", filename=f"{video_id}-measurements.csv")
