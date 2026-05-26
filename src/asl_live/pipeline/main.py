"""3-process orchestrator — camera → translation → speech.

Spawns three worker processes connected by two ``multiprocessing.Queue``s:

    recognizer ──(word, lang)──▶ translator ──(text, lang)──▶ speaker

Each worker constructs its own heavy objects (camera, ONNX session,
Argos packs, Piper voices) **inside** the worker, not in main —
``spawn`` (required on Windows) pickles all args, and Argos/Piper
sessions aren't picklable.

Shutdown is signalled two ways:

- A ``multiprocessing.Event`` (``shutdown_event``) — polled by the
  recognizer at the top of each frame, since it doesn't read from a
  queue and has no other way to learn that it should stop.
- A ``None`` sentinel pushed into both queues — unblocks the queue
  consumers (translator + speaker) immediately, so they don't have to
  wake on a timeout.

Main process catches SIGINT, sets the event, pushes both sentinels,
joins all three children with a 5 s budget each, then falls back to
``terminate()`` if a child is hung.

Usage::

    python -m asl_live.pipeline.main
    python -m asl_live.pipeline.main --lang fr
    python -m asl_live.pipeline.main --camera 1
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import queue
import signal
import sys
import time
from typing import Optional

import numpy as np

from asl_live.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    GAP_FRAMES,
    MIN_CONF,
    STABLE_FRAMES,
    TRANSLATION_QUEUE_MAX,
    TTS_LANGS,
    TTS_QUEUE_MAX,
)
from asl_live.pipeline.buffer import WordBuffer
from asl_live.recognition.classifier import Classifier
from asl_live.recognition.debounce import CommitEvent, Debouncer
from asl_live.recognition.landmarks import LandmarkExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="asl_live.pipeline.main",
        description="ASL-live full pipeline: camera → translation → speech.",
    )
    p.add_argument(
        "--lang", default="it", choices=TTS_LANGS,
        help="Target language for translation + TTS. Default: it.",
    )
    p.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index. Default: 0.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Drop-oldest helper for the bounded TTS queue (feature-6 §4)
# ---------------------------------------------------------------------------


def _put_drop_oldest(q: mp.Queue, item: object) -> None:
    """Push ``item``; if the queue is full, drop the oldest then re-push.

    Producer-side implementation since ``multiprocessing.Queue`` has no
    native drop-oldest mode. Tiny race where the consumer might pick up
    the about-to-be-dropped item; acceptable per feature-6 §4.
    """
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()  # discard oldest
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            # Still full — the consumer is in deep trouble; drop this one too.
            logger.warning("tts_in queue still full after drop; dropping current item")


# ---------------------------------------------------------------------------
# Recognizer worker
# ---------------------------------------------------------------------------


def recognizer_loop(
    lang: str,
    camera_index: int,
    translation_in_q: mp.Queue,
    shutdown_event,
) -> None:
    """Camera → landmarks → classifier → debounce → buffer → emit on SPACE."""
    import cv2

    cap = _open_camera(camera_index)
    if cap is None:
        logger.error("recognizer: failed to open camera %d", camera_index)
        return

    classifier = Classifier()
    debouncer = Debouncer()
    buffer = WordBuffer()

    try:
        with LandmarkExtractor() as extractor:
            while not shutdown_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    logger.error("recognizer: cap.read() failed; exiting")
                    return

                prediction, raw_coords = _classify_frame(frame, extractor, classifier)
                event = debouncer.step(prediction)
                if event is not None:
                    _apply_commit_event(event, buffer, lang, translation_in_q)

                _draw_overlay(frame, raw_coords, prediction, debouncer, buffer)
                cv2.imshow("ASL-live pipeline", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    shutdown_event.set()
                    return
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _open_camera(camera_index: int):
    """Open the camera with the standard Windows DSHOW fallback."""
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap


def _classify_frame(
    frame: np.ndarray,
    extractor: LandmarkExtractor,
    classifier: Classifier,
) -> tuple[Optional[tuple[str, float]], Optional[np.ndarray]]:
    result = extractor.extract_with_raw(frame)
    if result is None:
        return None, None
    raw_coords, normalized = result
    return classifier.predict(normalized), raw_coords


def _apply_commit_event(
    event: CommitEvent,
    buffer: WordBuffer,
    lang: str,
    translation_in_q: mp.Queue,
) -> None:
    """LETTER appends; SPACE flushes and dispatches; DELETE pops; all silent on empty."""
    if event.kind == "LETTER":
        assert event.letter is not None
        buffer.append(event.letter)
    elif event.kind == "SPACE":
        word = buffer.flush()
        if word is not None:
            try:
                translation_in_q.put_nowait((word, lang))
            except queue.Full:
                logger.warning("translation queue full; dropping word %r", word)
    elif event.kind == "DELETE":
        buffer.pop()


# ---------------------------------------------------------------------------
# Drawing — kept inline; will factor out if a third consumer needs it
# ---------------------------------------------------------------------------


def _draw_overlay(
    frame: np.ndarray,
    raw_coords: Optional[np.ndarray],
    prediction: Optional[tuple[str, float]],
    debouncer: Debouncer,
    buffer: WordBuffer,
) -> None:
    import cv2

    if raw_coords is not None:
        h, w = frame.shape[:2]
        for x, y, _z in raw_coords:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 200, 255), -1)

    h, w = frame.shape[:2]
    # Top bar — top-1 class + confidence + debounce status.
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    if prediction is not None:
        label, conf = prediction
        color = (0, 255, 0) if conf >= MIN_CONF else (120, 120, 120)
        cv2.putText(
            frame, f"{label}  {conf * 100:5.1f}%",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
    else:
        cv2.putText(
            frame, "NO HAND",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2,
        )
    cv2.putText(
        frame, _debounce_status(debouncer),
        (w // 2 - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
    )

    # Bottom bar — running buffer + quit hint.
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, f"buffer: {buffer.text}",
        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )
    cv2.putText(
        frame, "q to quit",
        (w - 110, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )


def _debounce_status(debouncer: Debouncer) -> str:
    if debouncer.cooldown > 0:
        return f"COOLDOWN {debouncer.cooldown}"
    if debouncer.streak > 0:
        return f"HOLD {debouncer.streak}/{STABLE_FRAMES}"
    return "IDLE"


# ---------------------------------------------------------------------------
# Translator worker
# ---------------------------------------------------------------------------


def translator_loop(
    translation_in_q: mp.Queue,
    tts_in_q: mp.Queue,
    shutdown_event,
) -> None:
    """Blocking get from translation_in_q; translate; drop-oldest push to tts_in_q."""
    from asl_live.translation.translator import Translator

    translator = Translator()
    while True:
        item = translation_in_q.get()  # blocks; sentinel will unblock it
        if item is None or shutdown_event.is_set():
            _put_drop_oldest(tts_in_q, None)  # propagate sentinel downstream
            return
        word, lang = item
        translated = translator.translate(word, lang)
        _put_drop_oldest(tts_in_q, (translated, lang))


# ---------------------------------------------------------------------------
# Speaker worker
# ---------------------------------------------------------------------------


def speaker_loop(tts_in_q: mp.Queue, shutdown_event) -> None:
    """Blocking get from tts_in_q; synthesize + play."""
    from asl_live.tts.speaker import Speaker

    speaker = Speaker()
    while True:
        item = tts_in_q.get()  # blocks; sentinel will unblock it
        if item is None or shutdown_event.is_set():
            return
        text, lang = item
        speaker.speak(text, lang)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    ctx = mp.get_context("spawn")  # required on Windows; safe on Linux
    translation_in_q: mp.Queue = ctx.Queue(maxsize=TRANSLATION_QUEUE_MAX)
    tts_in_q: mp.Queue = ctx.Queue(maxsize=TTS_QUEUE_MAX)
    shutdown_event = ctx.Event()

    procs = [
        ctx.Process(
            target=recognizer_loop, name="recognizer",
            args=(args.lang, args.camera, translation_in_q, shutdown_event),
        ),
        ctx.Process(
            target=translator_loop, name="translator",
            args=(translation_in_q, tts_in_q, shutdown_event),
        ),
        ctx.Process(
            target=speaker_loop, name="speaker",
            args=(tts_in_q, shutdown_event),
        ),
    ]

    def _request_shutdown(*_args: object) -> None:
        if shutdown_event.is_set():
            return
        print("\nshutdown requested ...", flush=True)
        shutdown_event.set()
        # Unblock anything sitting in a get() right now.
        try:
            translation_in_q.put_nowait(None)
        except queue.Full:
            pass
        try:
            tts_in_q.put_nowait(None)
        except queue.Full:
            pass

    signal.signal(signal.SIGINT, _request_shutdown)

    for p in procs:
        p.start()

    # Babysit: if any child dies on its own, request shutdown of the rest.
    while not shutdown_event.is_set():
        if any(not p.is_alive() for p in procs):
            _request_shutdown()
            break
        time.sleep(0.2)

    for p in procs:
        p.join(timeout=5.0)
        if p.is_alive():
            logger.warning("%s did not exit within 5 s; terminating", p.name)
            p.terminate()
            p.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
