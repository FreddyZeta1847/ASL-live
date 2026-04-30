"""Project-wide constants and paths.

Pydantic-backed config with file persistence is deferred to phase 6
(when the language menu needs to round-trip a chosen language to disk).
Until then, this module exposes typed constants that every other
module imports directly.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""The repository root — the directory containing pyproject.toml."""

DATA_DIR: Path = REPO_ROOT / "data"
"""Collected and ingested landmark .npy files live here. Gitignored."""

MODELS_DIR: Path = REPO_ROOT / "models"
"""Trained model artifacts (mlp.onnx, label_map.json, training_report.json). Gitignored."""

LANDMARKS_DIR: Path = DATA_DIR / "landmarks"
"""Per-class landmark vectors: data/landmarks/<class>/<source>_<id>.npy."""


# ---------------------------------------------------------------------------
# Class set (feature 1, feature 2)
# ---------------------------------------------------------------------------

LETTER_CLASSES: tuple[str, ...] = tuple(
    c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in {"J", "Z"}
)
"""24 ASL alphabet letters (A-Y minus J and Z, both motion signs)."""

CONTROL_CLASSES: tuple[str, ...] = ("SPACE", "DELETE")
"""Custom control gestures: open palm = SPACE, thumb-down = DELETE."""

CLASSES: tuple[str, ...] = LETTER_CLASSES + CONTROL_CLASSES
"""All 26 classes the classifier outputs."""

NUM_CLASSES: int = len(CLASSES)


# ---------------------------------------------------------------------------
# Landmark feature shape (feature 1)
# ---------------------------------------------------------------------------

NUM_LANDMARKS: int = 21
"""MediaPipe Hands returns 21 keypoints per hand."""

LANDMARK_DIMS: int = 3
"""Each keypoint is (x, y, z)."""

LANDMARK_FEATURES: int = NUM_LANDMARKS * LANDMARK_DIMS  # 63
"""Flattened feature-vector length fed to the classifier."""


# ---------------------------------------------------------------------------
# Camera (feature 1)
# ---------------------------------------------------------------------------

CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480


# ---------------------------------------------------------------------------
# Debounce thresholds (feature 4)
# ---------------------------------------------------------------------------

STABLE_FRAMES: int = 5
"""Consecutive same-class frames required before a commit. ~167 ms at 30 fps."""

GAP_FRAMES: int = 3
"""Blind cooldown frames after a commit. ~100 ms at 30 fps."""

MIN_CONF: float = 0.85
"""Per-frame confidence threshold below which a prediction is treated as no-signal."""


# ---------------------------------------------------------------------------
# GPIO pin assignments (feature 8) — referenced when the [pi] profile is installed
# ---------------------------------------------------------------------------

GPIO_BUTTON_B1: int = 17
"""Main button — header pin 11."""

GPIO_BUTTON_B2: int = 27
"""Auxiliary button — header pin 13."""
