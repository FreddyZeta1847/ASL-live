"""Tests for the pipeline package.

Covers:
- ``WordBuffer`` semantics (LETTER/SPACE/DELETE + empty-buffer no-op,
  per feature-4 §4.7).
- Graceful shutdown of the worker pattern: a queue-sentinel stub
  worker exits when fed ``None``, and an event-driven stub worker
  exits when a ``multiprocessing.Event`` is set.

Full 3-process boot-up with real camera/Argos/Piper is **not** tested
here — multiprocessing-on-Windows is flaky in pytest and live audio
isn't unit-testable. Those are verified manually per the phase-5
acceptance criteria.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from asl_live.pipeline.buffer import WordBuffer


# ---------------------------------------------------------------------------
# WordBuffer — pure state, no processes
# ---------------------------------------------------------------------------


def test_buffer_starts_empty():
    b = WordBuffer()
    assert b.text == ""
    assert b.chars == []


def test_append_adds_letters_in_order():
    b = WordBuffer()
    b.append("H")
    b.append("I")
    assert b.text == "HI"


def test_pop_removes_last_letter_and_returns_true():
    b = WordBuffer()
    b.append("H")
    b.append("I")
    assert b.pop() is True
    assert b.text == "H"


def test_pop_on_empty_returns_false_and_stays_empty():
    """feature-4 §4.7: DELETE on empty buffer is a silent no-op."""
    b = WordBuffer()
    assert b.pop() is False
    assert b.text == ""


def test_flush_returns_word_and_clears():
    b = WordBuffer()
    b.append("H")
    b.append("I")
    assert b.flush() == "HI"
    assert b.text == ""


def test_flush_on_empty_returns_none():
    """feature-4 §4.7: SPACE on empty buffer is a silent no-op."""
    b = WordBuffer()
    assert b.flush() is None


def test_flush_then_append_starts_a_new_word():
    b = WordBuffer()
    b.append("H")
    b.flush()
    b.append("E")
    assert b.text == "E"


# ---------------------------------------------------------------------------
# Shutdown stubs — must be top-level functions so spawn can pickle them
# ---------------------------------------------------------------------------


def _queue_sentinel_worker(q: mp.Queue, done: mp.Queue) -> None:
    """Mirror of the translator/speaker shutdown path."""
    while True:
        item = q.get()
        if item is None:
            break
    done.put("OK")


def _event_polling_worker(evt, done: mp.Queue) -> None:
    """Mirror of the recognizer shutdown path."""
    while not evt.is_set():
        time.sleep(0.05)
    done.put("OK")


# ---------------------------------------------------------------------------
# Shutdown integration tests
# ---------------------------------------------------------------------------


def test_queue_sentinel_unblocks_worker():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    done = ctx.Queue()
    p = ctx.Process(target=_queue_sentinel_worker, args=(q, done))
    p.start()
    q.put(None)  # sentinel
    p.join(timeout=3)
    assert not p.is_alive()
    assert done.get_nowait() == "OK"


def test_event_set_unblocks_worker():
    ctx = mp.get_context("spawn")
    evt = ctx.Event()
    done = ctx.Queue()
    p = ctx.Process(target=_event_polling_worker, args=(evt, done))
    p.start()
    # Let the worker enter its polling loop, then signal.
    time.sleep(0.2)
    evt.set()
    p.join(timeout=3)
    assert not p.is_alive()
    assert done.get_nowait() == "OK"
