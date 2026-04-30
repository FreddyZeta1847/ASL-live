"""Interactive landmark collector for SPACE / DELETE custom gestures.

Per feature 2 (`.claude/docs/features/feature-2-data-collection.md`):
- Auto-captures while a hand is detected and stable for 5 frames.
- 10-frame cooldown between captures.
- Saves a 63-dim normalized vector and a mirror-augmented copy under
  ``data/landmarks/<class>/``.
- OpenCV preview shows class label, sample counter, and the 21-landmark
  overlay so the user can see MediaPipe locking on cleanly.
- ``q`` to quit before reaching the count.

Usage::

    python -m asl_live.capture.collect --class SPACE --count 500
    python -m asl_live.capture.collect --class DELETE --count 500

Multiple sessions append (the script picks the next unused index by
scanning existing files), so collecting in batches is fine.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from asl_live.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROL_CLASSES,
    LANDMARK_DIMS,
    LANDMARKS_DIR,
    LETTER_CLASSES,
)
from asl_live.recognition.landmarks import LandmarkExtractor


# ---------------------------------------------------------------------------
# Capture cadence (per feature 2 decision 3)
# ---------------------------------------------------------------------------

STABLE_FRAMES_FOR_CAPTURE = 5
COOLDOWN_FRAMES_AFTER_CAPTURE = 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive landmark collector for ASL-live",
    )
    p.add_argument(
        "--class",
        dest="cls",
        required=True,
        choices=list(LETTER_CLASSES) + list(CONTROL_CLASSES),
        help="Class to collect (24 letters + SPACE + DELETE)",
    )
    p.add_argument("--count", type=int, default=500, help="Target sample count")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument(
        "--dst",
        type=Path,
        default=LANDMARKS_DIR,
        help="Output root directory (default: data/landmarks/)",
    )
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirror augmentation (default: enabled)",
    )
    p.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save raw frames for debugging (under <class>/_frames/)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Capture state machine
# ---------------------------------------------------------------------------


@dataclass
class CaptureState:
    streak: int = 0
    cooldown: int = 0


def update_capture_state(state: CaptureState, hand_present: bool) -> bool:
    """Advance the state machine. Returns True iff a capture should fire this frame."""
    if state.cooldown > 0:
        state.cooldown -= 1
        return False
    if not hand_present:
        state.streak = 0
        return False
    state.streak += 1
    if state.streak >= STABLE_FRAMES_FOR_CAPTURE:
        state.streak = 0
        state.cooldown = COOLDOWN_FRAMES_AFTER_CAPTURE
        return True
    return False


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------


def open_camera(camera_index: int):
    """Open the camera at the configured resolution. Returns ``None`` on failure."""
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # Windows often needs the DirectShow backend explicitly.
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap


# ---------------------------------------------------------------------------
# Sample I/O
# ---------------------------------------------------------------------------


def mirror_normalized(vec: np.ndarray) -> np.ndarray:
    """Mirror an already-normalized 63-dim vector (x -> -x)."""
    coords = vec.reshape(-1, LANDMARK_DIMS).copy()
    coords[:, 0] = -coords[:, 0]
    return coords.flatten().astype(np.float32)


def save_sample(
    vec: np.ndarray,
    class_dst: Path,
    idx: int,
    suffix: str = "",
) -> Path:
    """Write a single landmark vector to ``<class_dst>/custom_<idx><suffix>.npy``."""
    class_dst.mkdir(parents=True, exist_ok=True)
    fname = f"custom_{idx:06d}{suffix}.npy"
    out_path = class_dst / fname
    np.save(out_path, vec)
    return out_path


def save_with_mirror(
    vec: np.ndarray, class_dst: Path, idx: int, save_mirror: bool
) -> None:
    save_sample(vec, class_dst, idx)
    if save_mirror:
        save_sample(mirror_normalized(vec), class_dst, idx, suffix="_m")


def save_frame(frame: np.ndarray, class_dst: Path, idx: int) -> None:
    """Optional debug aid: write the raw camera frame next to the .npy."""
    import cv2

    frames_dir = class_dst / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frames_dir / f"{idx:06d}.png"), frame)


def starting_index(class_dst: Path) -> int:
    """Find the next unused sample index so multiple sessions don't collide.

    Filenames look like ``custom_000123.npy`` or ``custom_000123_m.npy``;
    we scan the integer between the underscores and return ``max + 1``.
    """
    if not class_dst.is_dir():
        return 0
    indices = []
    for path in class_dst.glob("custom_*.npy"):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            indices.append(int(parts[1]))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_overlay(
    frame: np.ndarray,
    raw_coords: Optional[np.ndarray],
    cls: str,
    saved: int,
    target: int,
    state: CaptureState,
) -> None:
    """Draw class label, counter, status, and 21-landmark overlay on `frame`."""
    import cv2

    h, w = frame.shape[:2]

    # Header bar — class + count
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{cls}    {saved}/{target}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Right-side status
    status, color = _status_for(state, raw_coords is not None, saved >= target)
    cv2.putText(
        frame,
        status,
        (w - 200, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    if raw_coords is not None:
        _draw_landmarks(frame, raw_coords)

    # Footer — q to quit
    cv2.putText(
        frame,
        "q to quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )


def _status_for(
    state: CaptureState, hand_present: bool, done: bool
) -> tuple[str, tuple[int, int, int]]:
    """Pick a status string and BGR color based on the current state."""
    if done:
        return "DONE", (0, 200, 255)
    if state.cooldown > 0:
        return f"COOLDOWN {state.cooldown}", (100, 100, 255)
    if not hand_present:
        return "NO HAND", (0, 0, 200)
    return f"HOLD {state.streak}/{STABLE_FRAMES_FOR_CAPTURE}", (0, 200, 0)


def _draw_landmarks(frame: np.ndarray, raw_coords: np.ndarray) -> None:
    """Draw 21 small dots at the landmark positions in `frame`."""
    import cv2

    h, w = frame.shape[:2]
    for x, y, _z in raw_coords:
        px = int(x * w)
        py = int(y * h)
        cv2.circle(frame, (px, py), 4, (0, 200, 255), -1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def collect_loop(
    cap,
    extractor: LandmarkExtractor,
    cls: str,
    target: int,
    class_dst: Path,
    save_mirror: bool,
    save_frames_flag: bool,
) -> int:
    """Run the camera + capture loop until ``target`` samples are saved or `q`.

    Returns the count of original (non-mirror) samples saved this session.
    """
    import cv2

    state = CaptureState()
    next_idx = starting_index(class_dst)
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera; exiting.")
            break

        result = extractor.extract_with_raw(frame)
        raw_coords = result[0] if result is not None else None
        normalized = result[1] if result is not None else None

        should_capture = (
            update_capture_state(state, normalized is not None)
            and saved < target
            and normalized is not None
        )
        if should_capture:
            save_with_mirror(normalized, class_dst, next_idx, save_mirror)
            if save_frames_flag:
                save_frame(frame, class_dst, next_idx)
            next_idx += 1
            saved += 1

        draw_overlay(frame, raw_coords, cls, saved, target, state)
        cv2.imshow("ASL-live collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or saved >= target:
            break

    return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit(f"Failed to open camera {args.camera}")

    class_dst = args.dst / args.cls
    save_mirror = not args.no_mirror

    print(f"Collecting up to {args.count} samples for class {args.cls}")
    print(f"Output:  {class_dst}")
    print(f"Camera:  {args.camera}")
    print("Auto-capture starts when the hand is stable for 5 frames.")
    print("Press q in the preview window to quit early.\n")

    try:
        with LandmarkExtractor() as extractor:
            saved = collect_loop(
                cap=cap,
                extractor=extractor,
                cls=args.cls,
                target=args.count,
                class_dst=class_dst,
                save_mirror=save_mirror,
                save_frames_flag=args.save_frames,
            )
    finally:
        cap.release()
        import cv2

        cv2.destroyAllWindows()

    multiplier = 2 if save_mirror else 1
    print(
        f"\nSaved {saved} samples "
        f"({saved * multiplier} .npy files including {'mirrors' if save_mirror else 'no mirrors'})",
    )


if __name__ == "__main__":
    main()
