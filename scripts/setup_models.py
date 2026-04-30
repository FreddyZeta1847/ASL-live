"""Download external model files needed at runtime.

Currently the only model fetched here is MediaPipe's
``hand_landmarker.task`` (~13 MB). It's separate from the trained MLP
because (a) it's a third-party asset, not something we produce, and
(b) the new MediaPipe Tasks API requires the model file to exist on
disk before the detector can be constructed.

The trained MLP (``models/mlp.onnx``) is produced by phase 2 and is
*not* fetched here.

Usage::

    python scripts/setup_models.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from asl_live.config import HAND_LANDMARKER_MODEL, MODELS_DIR

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def download_if_missing(url: str, target: Path) -> None:
    """Fetch ``url`` to ``target`` unless ``target`` already exists."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        size_kb = target.stat().st_size // 1024
        print(f"[skip] already present: {target} ({size_kb} KB)")
        return
    print(f"[get ] {url}")
    print(f"   -> {target}")
    urllib.request.urlretrieve(url, target)
    size_kb = target.stat().st_size // 1024
    print(f"   ok ({size_kb} KB)")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_if_missing(HAND_LANDMARKER_URL, HAND_LANDMARKER_MODEL)
    print("Done.")


if __name__ == "__main__":
    main()
