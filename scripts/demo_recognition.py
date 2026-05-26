"""Live PC recognition demo — landmarks → classifier → debounce → terminal.

End-to-end smoke test of the phase-2 model + the phase-3 debouncer.
Opens the camera, runs each frame through the same path the production
recognizer worker will use, and prints commit events to the terminal.

Output format
-------------
- Each LETTER commit prints the letter on its own line: ``H``.
- A SPACE commit prints the assembled word with a leading arrow:
  ``→ HELLO``, and clears the buffer. Silent no-op if the buffer is empty.
- A DELETE commit pops the last char and prints ``←``. Silent no-op if
  the buffer is empty.

Usage::

    python scripts/demo_recognition.py
    python scripts/demo_recognition.py --camera 1

``q`` in the OpenCV preview window quits.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from asl_live.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    GAP_FRAMES,
    MIN_CONF,
    STABLE_FRAMES,
)
from asl_live.recognition.classifier import Classifier
from asl_live.recognition.debounce import CommitEvent, Debouncer
from asl_live.recognition.landmarks import LandmarkExtractor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live recognition demo for ASL-live (PC only).",
    )
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Word buffer + commit handler
# ---------------------------------------------------------------------------


@dataclass
class WordBuffer:
    """Running word state the demo prints from."""

    chars: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.chars)

    def append(self, letter: str) -> None:
        self.chars.append(letter)

    def pop(self) -> bool:
        """Pop the last char. Returns False if the buffer was empty."""
        if not self.chars:
            return False
        self.chars.pop()
        return True

    def flush(self) -> Optional[str]:
        """Return the current word and clear. None if buffer was empty."""
        if not self.chars:
            return None
        word = self.text
        self.chars.clear()
        return word


def handle_event(event: CommitEvent, buffer: WordBuffer) -> None:
    """Apply buffer logic for a single commit event and print to terminal."""
    if event.kind == "LETTER":
        assert event.letter is not None
        buffer.append(event.letter)
        print(event.letter, flush=True)
    elif event.kind == "SPACE":
        word = buffer.flush()
        if word is not None:
            print(f"→ {word}", flush=True)  # "→ HELLO"
    elif event.kind == "DELETE":
        if buffer.pop():
            print("←", flush=True)  # "←"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def open_camera(camera_index: int):
    """Open the camera at the configured resolution. Returns None on failure.

    Mirrors ``scripts/collect.py``'s helper — Windows tends to need the
    explicit DirectShow backend if the default open fails.
    """
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_landmarks(frame: np.ndarray, raw_coords: np.ndarray) -> None:
    """Draw 21 small dots at the landmark positions in ``frame``."""
    import cv2

    h, w = frame.shape[:2]
    for x, y, _z in raw_coords:
        px = int(x * w)
        py = int(y * h)
        cv2.circle(frame, (px, py), 4, (0, 200, 255), -1)


def _draw_header(
    frame: np.ndarray,
    label: Optional[str],
    confidence: Optional[float],
    debouncer: Debouncer,
    fps: float,
) -> None:
    """Header bar: top-1 class + confidence, debounce status, fps."""
    import cv2

    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

    if label is not None and confidence is not None:
        left = f"{label}  {confidence * 100:5.1f}%"
        # Dim the text under MIN_CONF so the user sees what's being thrown out.
        color = (0, 255, 0) if confidence >= MIN_CONF else (120, 120, 120)
    else:
        left = "NO HAND"
        color = (0, 0, 200)
    cv2.putText(frame, left, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    middle = _debounce_status_string(debouncer)
    cv2.putText(
        frame, middle, (w // 2 - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (200, 200, 200), 2,
    )

    right = f"{fps:4.1f} fps"
    cv2.putText(
        frame, right, (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2,
    )


def _debounce_status_string(debouncer: Debouncer) -> str:
    """Compact one-liner: cooldown countdown, hold counter, or idle."""
    if debouncer.cooldown > 0:
        return f"COOLDOWN {debouncer.cooldown}"
    if debouncer.streak > 0:
        return f"HOLD {debouncer.streak}/{STABLE_FRAMES}"
    return "IDLE"


def _draw_buffer(frame: np.ndarray, buffer: WordBuffer) -> None:
    """Footer bar: live word buffer + quit hint."""
    import cv2

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, f"buffer: {buffer.text}", (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )
    cv2.putText(
        frame, "q to quit", (w - 110, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )


# ---------------------------------------------------------------------------
# FPS smoother
# ---------------------------------------------------------------------------


class FpsMeter:
    """EMA over recent inter-frame intervals so the displayed fps is readable."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._last_t: Optional[float] = None
        self._fps: float = 0.0

    def tick(self) -> float:
        now = time.monotonic()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                instant = 1.0 / dt
                self._fps = (
                    instant if self._fps == 0.0
                    else (1 - self._alpha) * self._fps + self._alpha * instant
                )
        self._last_t = now
        return self._fps


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_demo(
    cap,
    extractor: LandmarkExtractor,
    classifier: Classifier,
    debouncer: Debouncer,
) -> None:
    """Camera → landmarks → classifier → debounce → print + draw, until q."""
    import cv2

    buffer = WordBuffer()
    fps_meter = FpsMeter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera; exiting.")
            return

        prediction, raw_coords = _classify_frame(frame, extractor, classifier)
        event = debouncer.step(prediction)
        if event is not None:
            handle_event(event, buffer)

        # Draw after the step so the overlay reflects the post-commit state
        # (e.g. cooldown countdown starts on the same frame the commit fired).
        _draw_overlay(frame, raw_coords, prediction, debouncer, buffer, fps_meter.tick())
        cv2.imshow("ASL-live demo", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return


def _classify_frame(
    frame: np.ndarray,
    extractor: LandmarkExtractor,
    classifier: Classifier,
) -> tuple[Optional[tuple[str, float]], Optional[np.ndarray]]:
    """Run one frame through extractor + classifier.

    Returns ``(prediction, raw_coords)`` where either may be ``None``.
    Splitting this out keeps ``run_demo`` readable and isolates the only
    line that touches the model session.
    """
    result = extractor.extract_with_raw(frame)
    if result is None:
        return None, None
    raw_coords, normalized = result
    prediction = classifier.predict(normalized)
    return prediction, raw_coords


def _draw_overlay(
    frame: np.ndarray,
    raw_coords: Optional[np.ndarray],
    prediction: Optional[tuple[str, float]],
    debouncer: Debouncer,
    buffer: WordBuffer,
    fps: float,
) -> None:
    """Bundle the per-frame drawing calls so ``run_demo`` stays linear."""
    if raw_coords is not None:
        _draw_landmarks(frame, raw_coords)
    label = prediction[0] if prediction is not None else None
    confidence = prediction[1] if prediction is not None else None
    _draw_header(frame, label, confidence, debouncer, fps)
    _draw_buffer(frame, buffer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit(f"Failed to open camera {args.camera}")

    print("ASL-live demo")
    print(f"  STABLE_FRAMES={STABLE_FRAMES}  GAP_FRAMES={GAP_FRAMES}  MIN_CONF={MIN_CONF}")
    print("  Press q in the preview window to quit.\n")

    classifier = Classifier()
    debouncer = Debouncer()

    try:
        with LandmarkExtractor() as extractor:
            run_demo(cap, extractor, classifier, debouncer)
    finally:
        cap.release()
        import cv2

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
