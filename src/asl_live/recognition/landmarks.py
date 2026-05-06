"""MediaPipe HandLandmarker wrapper returning canonical 63-dim landmark vectors.

Per feature 1 decisions (`.claude/docs/features/feature-1-hand-landmarks.md`):
- Single hand: pick the most-confident detection.
- Left-hand detections mirrored to canonical right-handed form.
- ``extract()`` returns ``None`` when no hand is detected; the
  classifier is not invoked on no-signal frames.
- Normalization: wrist to origin, scale so max wrist-to-landmark
  distance = 1. **No rotation normalization.**

Implementation note
-------------------
This module uses the **new MediaPipe Tasks API**
(``mediapipe.tasks.python.vision.HandLandmarker``) — the older
``mp.solutions.hands.Hands`` namespace was removed in recent
MediaPipe releases. The Tasks API requires a separate model file
(`hand_landmarker.task`); run ``scripts/setup_models.py`` once to
download it before constructing a ``LandmarkExtractor``.

The pure normalization helpers (``_mirror``, ``_translate_to_wrist_origin``,
``_scale_to_unit_max``, ``_normalize``) are intentionally importable
without pulling MediaPipe — that is what ``tests/test_landmarks.py``
relies on. MediaPipe and OpenCV are imported lazily inside
``LandmarkExtractor.__init__`` so the test suite stays light.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from asl_live.config import HAND_LANDMARKER_MODEL


# ---------------------------------------------------------------------------
# Pure normalization helpers (no MediaPipe / OpenCV — unit-testable)
# ---------------------------------------------------------------------------


def _mirror(coords: np.ndarray) -> np.ndarray:
    """Flip x-coordinates around 0.5 (image-space horizontal mirror).

    Applied only to left-hand detections so the classifier always sees
    a canonical right-handed sign. MediaPipe returns x in [0, 1], so
    the mirror is ``x -> 1 - x``. Does not mutate the input.
    """
    out = coords.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def _translate_to_wrist_origin(coords: np.ndarray) -> np.ndarray:
    """Subtract the wrist (landmark 0) from every keypoint."""
    return coords - coords[0]


def _scale_to_unit_max(coords: np.ndarray) -> Optional[np.ndarray]:
    """Divide so the largest distance from origin is exactly 1.

    Expects ``coords`` to be already translated so the wrist sits at
    the origin. Returns ``None`` on a degenerate frame where every
    landmark coincides with the wrist (max distance = 0) — this avoids
    a division by zero downstream.
    """
    distances = np.linalg.norm(coords, axis=1)
    max_dist = float(distances.max())
    if max_dist < 1e-6:
        return None
    return coords / max_dist


def _normalize(coords: np.ndarray) -> Optional[np.ndarray]:
    """Two-step normalization: translate, scale, then flatten.

    Input: ``(21, 3)`` MediaPipe landmark array in image space.
    Output: ``(63,)`` float32 vector, or ``None`` on degenerate input.
    """
    translated = _translate_to_wrist_origin(coords)
    scaled = _scale_to_unit_max(translated)
    if scaled is None:
        return None
    return scaled.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# MediaPipe wrapper (new Tasks API)
# ---------------------------------------------------------------------------


class LandmarkExtractor:
    """Wraps MediaPipe ``HandLandmarker``. One instance per process.

    ``extract(frame_bgr)`` returns a normalized 63-dim vector for the
    most-confident hand, or ``None`` if no hand is detected. Left-hand
    detections are mirrored to canonical right-handed form *before*
    normalization.

    Use as a context manager so MediaPipe's resources are released::

        with LandmarkExtractor() as ext:
            v = ext.extract(frame_bgr)

    The constructor expects ``hand_landmarker.task`` to exist at
    ``models/hand_landmarker.task`` (configurable via ``model_path``).
    Run ``scripts/setup_models.py`` once to download it.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: Optional[Path] = None,
    ) -> None:
        # Lazy-import heavy deps so the pure helpers above can be
        # imported (and unit-tested) without pulling MediaPipe / cv2.
        import cv2  # noqa: WPS433
        import mediapipe as mp  # noqa: WPS433
        from mediapipe.tasks import python as mp_python  # noqa: WPS433
        from mediapipe.tasks.python import vision as mp_vision  # noqa: WPS433

        self._cv2 = cv2
        self._mp = mp
        self._mp_vision = mp_vision

        path = Path(model_path) if model_path is not None else HAND_LANDMARKER_MODEL
        if not path.is_file():
            raise FileNotFoundError(
                f"HandLandmarker model not found at {path}.\n"
                "Run `python scripts/setup_models.py` to download it.",
            )

        # static_image_mode=False -> VIDEO mode with cross-frame tracking,
        # for camera streams (collect.py, runtime).
        # static_image_mode=True  -> IMAGE mode with fresh detection per
        # call, for batches of unrelated stills (ingest_public.py).
        self._is_video_mode = not static_image_mode
        running_mode = (
            mp_vision.RunningMode.VIDEO
            if self._is_video_mode
            else mp_vision.RunningMode.IMAGE
        )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            num_hands=2,  # detect up to 2 so we can pick the most confident
            running_mode=running_mode,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        # VIDEO mode requires a monotonically-increasing millisecond
        # timestamp on every detect_for_video call. We use a wall-clock
        # offset so timestamps look reasonable but only monotonicity
        # matters.
        self._start_ns = time.monotonic_ns()

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Process one BGR camera frame, return canonical 63-dim vector or None."""
        result = self._run_mediapipe(frame_bgr)
        if result is None:
            return None
        coords, is_left = result
        if is_left:
            coords = _mirror(coords)
        return _normalize(coords)

    def extract_with_raw(
        self, frame_bgr: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Like ``extract`` but also returns the image-space landmarks.

        Returns ``(image_coords, normalized_vector)`` where
        ``image_coords`` is the original (21, 3) MediaPipe output in
        normalized image coordinates ([0, 1] in x and y, actual depth
        in z). Useful for drawing landmark overlays at their real
        camera positions during interactive collection.
        Returns ``None`` if no hand is detected or the frame is
        degenerate.

        Why this is a separate method (and not always returned by
        ``extract``): the live recognizer never draws on screen, so it
        calls ``extract`` to skip the extra raw-coords copy on every
        frame. The collector needs both raw (to draw the 21 dots) and
        normalized (to save), so it calls this variant.
        """
        result = self._run_mediapipe(frame_bgr)
        if result is None:
            return None
        coords, is_left = result
        raw = coords.copy()
        if is_left:
            coords = _mirror(coords)
        normalized = _normalize(coords)
        if normalized is None:
            return None
        return raw, normalized

    def _run_mediapipe(
        self, frame_bgr: np.ndarray
    ) -> Optional[tuple[np.ndarray, bool]]:
        """Run MediaPipe; return (coords, is_left) or None on no-hand."""
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        if self._is_video_mode:
            timestamp_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        idx = _best_hand_index(result)
        landmarks = result.hand_landmarks[idx]
        is_left = result.handedness[idx][0].category_name == "Left"
        return _landmarks_to_array(landmarks), is_left

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# MediaPipe-result helpers (private, used by LandmarkExtractor)
# ---------------------------------------------------------------------------


def _best_hand_index(result) -> int:
    """Index of the most-confident hand in a HandLandmarker result."""
    scores = [hd[0].score for hd in result.handedness]
    return int(np.argmax(scores))


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a list of NormalizedLandmark to a (21, 3) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )
