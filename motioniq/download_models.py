"""Fetch version-pinned Ultralytics pretrained pose and segmentation weights."""

from __future__ import annotations

import hashlib
import shutil
import ssl
import urllib.request
from pathlib import Path

MODELS = {
    "yolo11n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
        "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
    ),
    "yolo11n-seg.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    ),
}


def download_models(directory: str | Path = "models") -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for filename, (url, expected_hash) in MODELS.items():
        target = directory / filename
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                import certifi

                context = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(url, context=context) as response:
                    with temporary.open("wb") as destination:
                        shutil.copyfileobj(response, destination)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {target}; remove it and download again")
        results.append(target)
    return results


if __name__ == "__main__":
    for model in download_models():
        print(f"Verified {model}")
