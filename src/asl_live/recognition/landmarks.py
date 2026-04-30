"""MediaPipe Hands wrapper returning canonical 63-dim landmark vectors.

Per feature 1 decisions (`.claude/docs/features/feature-1-hand-landmarks.md`):
- Single hand: pick MediaPipe's most-confident detection.
- Left-hand detections mirrored to canonical right-handed form.
- ``extract()`` returns ``None`` when no hand is detected; the
  classifier is not invoked on no-signal frames.
- Normalization: wrist to origin, scale so max wrist-to-landmark
  distance = 1. **No rotation normalization.**
- ``model_complexity=1`` by default; fall back to ``0`` only if the Pi
  cannot sustain >=15 fps in phase 3 testing.

Module layout
-------------
The pure normalization helpers (``_mirror``, ``_translate_to_wrist_origin``,
``_scale_to_unit_max``, ``_normalize``) are intentionally importable
without pulling MediaPipe — that is what ``tests/test_landmarks.py``
relies on. MediaPipe and OpenCV are imported lazily inside
``LandmarkExtractor.__init__`` so the test suite stays light.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


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
# MediaPipe wrapper
# ---------------------------------------------------------------------------


class LandmarkExtractor:
    """Wraps MediaPipe Hands. One instance per process.

    ``extract(frame_bgr)`` returns a normalized 63-dim vector for the
    most-confident hand, or ``None`` if no hand is detected. Left-hand
    detections are mirrored to canonical right-handed form *before*
    normalization.

    Use as a context manager so MediaPipe's resources are released::

        with LandmarkExtractor() as ext:
            v = ext.extract(frame_bgr)
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        # Lazy-import heavy deps so the pure helpers above can be
        # imported (and unit-tested) without pulling MediaPipe / cv2.
        import cv2  # noqa: WPS433
        import mediapipe as mp  # noqa: WPS433

        self._cv2 = cv2
        # static_image_mode=False (default) is for video streams — uses
        # cross-frame tracking for efficiency. Set True for processing
        # batches of unrelated stills (e.g., the Kaggle ingest), where
        # detection should run from scratch on every image.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,  # detect up to 2 so we can pick the most confident
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None
        idx = _best_hand_index(results)
        landmarks = results.multi_hand_landmarks[idx]
        is_left = results.multi_handedness[idx].classification[0].label == "Left"
        return _landmarks_to_array(landmarks), is_left

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# MediaPipe-result helpers (private, used by LandmarkExtractor)
# ---------------------------------------------------------------------------


def _best_hand_index(results) -> int:
    """Index of the most-confident hand in a MediaPipe ``Hands`` result."""
    scores = [hd.classification[0].score for hd in results.multi_handedness]
    return int(np.argmax(scores))


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmarkList to a (21, 3) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
        dtype=np.float32,
    )
