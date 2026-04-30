"""Preprocess Kaggle ASL Alphabet images into normalized landmark .npy files.

Per feature 2 (`.claude/docs/features/feature-2-data-collection.md`):
- Walks the Kaggle dataset's class folders.
- Runs the same `LandmarkExtractor.extract()` used at runtime.
- Drops images where MediaPipe finds no hand.
- Saves the resulting 63-dim vector and a mirror-augmented copy under
  ``data/landmarks/<class>/``.
- Discards classes J, Z (motion signs, out of scope) and the dataset's
  own `del`, `nothing`, `space` classes (their hand shapes don't match
  our chosen control gestures).

Usage::

    # Full run
    python scripts/ingest_public.py --src ~/datasets/asl_alphabet_train

    # Dry run on the first 10 images per class
    python scripts/ingest_public.py --src <path> --limit 10

Acceptance per the phase-1 plan: a full run produces >= 2,000 .npy
files per kept class (after no-hand drops, before mirror augmentation
that doubles each).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from asl_live.config import LANDMARK_DIMS, LANDMARKS_DIR, LETTER_CLASSES
from asl_live.recognition.landmarks import LandmarkExtractor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Stats container
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    """Per-run counters. Mutated in place by the processing loop."""

    images_seen: int = 0
    hands_detected: int = 0
    samples_saved: int = 0
    per_class: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Kaggle ASL Alphabet -> normalized landmark .npy files",
    )
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the Kaggle asl_alphabet_train directory (containing A/, B/, ... folders)",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=LANDMARKS_DIR,
        help="Output root directory (default: data/landmarks/)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per class — useful for a dry run",
    )
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirror augmentation (default: enabled)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def discover_class_folders(src: Path) -> dict[str, Path]:
    """Return `{class_name: folder_path}` for every kept class present under ``src``.

    Kaggle uses uppercase letter folders (A, B, ..., Z). We keep only
    the 24 letters in :data:`LETTER_CLASSES` (A-Y minus J, Z).
    """
    found: dict[str, Path] = {}
    for cls in LETTER_CLASSES:
        folder = src / cls
        if folder.is_dir():
            found[cls] = folder
    return found


def list_images(folder: Path, limit: Optional[int]) -> list[Path]:
    """Sorted list of image paths in `folder`, optionally truncated to ``limit``."""
    images = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        images = images[:limit]
    return images


def load_image_bgr(path: Path) -> Optional[np.ndarray]:
    """Read a BGR image from disk. Returns ``None`` on read failure."""
    import cv2  # local import — keeps the module lean to import in tests

    return cv2.imread(str(path))


# ---------------------------------------------------------------------------
# Sample I/O
# ---------------------------------------------------------------------------


def mirror_normalized(vec: np.ndarray) -> np.ndarray:
    """Mirror an already-normalized 63-dim vector (x -> -x in normalized space).

    Equivalent to running the runtime image-space mirror (x -> 1 - x)
    *before* normalization: in both cases the wrist stays at origin
    after translation, and mirroring flips the hand horizontally.
    """
    coords = vec.reshape(-1, LANDMARK_DIMS).copy()
    coords[:, 0] = -coords[:, 0]
    return coords.flatten().astype(np.float32)


def save_sample(
    vec: np.ndarray,
    class_dst: Path,
    source: str,
    idx: int,
    suffix: str = "",
) -> Path:
    """Save a single landmark vector as ``<class_dst>/<source>_<idx><suffix>.npy``."""
    class_dst.mkdir(parents=True, exist_ok=True)
    fname = f"{source}_{idx:06d}{suffix}.npy"
    out_path = class_dst / fname
    np.save(out_path, vec)
    return out_path


# ---------------------------------------------------------------------------
# Per-image / per-class processing
# ---------------------------------------------------------------------------


def process_image(
    extractor: LandmarkExtractor,
    img_path: Path,
    class_dst: Path,
    idx: int,
    save_mirror: bool,
    stats: IngestStats,
) -> None:
    """Load, extract, and save (with optional mirror) — update ``stats`` in place."""
    stats.images_seen += 1

    frame = load_image_bgr(img_path)
    if frame is None:
        return

    vec = extractor.extract(frame)
    if vec is None:
        return
    stats.hands_detected += 1

    cls = class_dst.name
    save_sample(vec, class_dst, source="kaggle", idx=idx)
    stats.samples_saved += 1
    stats.per_class[cls] = stats.per_class.get(cls, 0) + 1

    if save_mirror:
        save_sample(
            mirror_normalized(vec),
            class_dst,
            source="kaggle",
            idx=idx,
            suffix="_m",
        )
        stats.samples_saved += 1
        stats.per_class[cls] += 1


def process_class(
    extractor: LandmarkExtractor,
    cls: str,
    folder: Path,
    dst_root: Path,
    limit: Optional[int],
    save_mirror: bool,
    stats: IngestStats,
) -> None:
    """Run ``process_image`` over every image in one class folder."""
    images = list_images(folder, limit)
    class_dst = dst_root / cls
    print(f"[{cls}] processing {len(images)} images...")
    for idx, img_path in enumerate(images):
        process_image(extractor, img_path, class_dst, idx, save_mirror, stats)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(stats: IngestStats) -> None:
    print()
    print("=" * 50)
    print(f"Images seen:    {stats.images_seen}")
    print(f"Hands detected: {stats.hands_detected}")
    print(f"Samples saved:  {stats.samples_saved}")
    print()
    print("Per class (count includes mirrored copies):")
    for cls in sorted(stats.per_class):
        print(f"  {cls:6} {stats.per_class[cls]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"--src is not a directory: {args.src}")

    folders = discover_class_folders(args.src)
    if not folders:
        raise SystemExit(
            f"No expected class folders (A, B, ...) found under {args.src}"
        )

    stats = IngestStats()
    save_mirror = not args.no_mirror

    # static_image_mode=True because each Kaggle image is unrelated to
    # the next — MediaPipe should re-detect from scratch every time.
    with LandmarkExtractor(static_image_mode=True) as extractor:
        for cls, folder in folders.items():
            process_class(
                extractor=extractor,
                cls=cls,
                folder=folder,
                dst_root=args.dst,
                limit=args.limit,
                save_mirror=save_mirror,
                stats=stats,
            )

    print_summary(stats)


if __name__ == "__main__":
    main()
