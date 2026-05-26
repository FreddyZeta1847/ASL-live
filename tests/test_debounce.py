"""Unit tests for the Debouncer state machine.

Covers the six required cases listed in `feature-4-debounce.md` §4.8 plus
the commit-event kind mapping (LETTER / SPACE / DELETE) and a few of the
locked edge cases from §4.7. Pure tests — no camera, no model, just
canned (label, confidence) streams.
"""
from __future__ import annotations

from typing import Optional

import pytest

from asl_live.config import GAP_FRAMES, MIN_CONF, STABLE_FRAMES
from asl_live.recognition.debounce import CommitEvent, Debouncer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


Prediction = Optional[tuple[str, float]]


def feed(
    debouncer: Debouncer, stream: list[Prediction]
) -> list[CommitEvent]:
    """Step ``debouncer`` over ``stream``; collect every emitted event."""
    events: list[CommitEvent] = []
    for prediction in stream:
        event = debouncer.step(prediction)
        if event is not None:
            events.append(event)
    return events


def confident(label: str, n: int, conf: float = 0.95) -> list[Prediction]:
    """``n`` copies of a single high-confidence prediction."""
    return [(label, conf)] * n


# ---------------------------------------------------------------------------
# Required cases (feature-4 §4.8)
# ---------------------------------------------------------------------------


def test_case1_five_a_frames_emit_exactly_one_letter():
    """1. 5 × ('A', 0.95) → one LETTER('A')."""
    events = feed(Debouncer(), confident("A", 5))
    assert len(events) == 1
    assert events[0] == CommitEvent(kind="LETTER", letter="A", confidence=0.95)


def test_case2_commit_then_no_hand_does_not_double_emit():
    """2. 5 × A, then 8 × None → still exactly one LETTER('A')."""
    stream = confident("A", 5) + [None] * 8
    events = feed(Debouncer(), stream)
    assert len(events) == 1
    assert events[0].letter == "A"


def test_case3_two_streaks_separated_by_no_hand_emit_twice():
    """3. 5 × A, 3 × None, 5 × A → two LETTER('A') commits.

    The 3 None frames cover the entire cooldown, so the second streak
    starts with the debouncer back in WATCHING with an empty streak.
    """
    stream = confident("A", 5) + [None] * 3 + confident("A", 5)
    events = feed(Debouncer(), stream)
    assert len(events) == 2
    assert all(e.letter == "A" for e in events)


def test_case4_single_b_resets_the_streak():
    """4. 4 × A, 1 × B, 5 × A → one LETTER('A'), no LETTER('B').

    The B resets the streak to 1 (on B). The next five A frames build
    a fresh streak A=1..5 and commit on the fifth.
    """
    stream = confident("A", 4) + confident("B", 1) + confident("A", 5)
    events = feed(Debouncer(), stream)
    assert len(events) == 1
    assert events[0].letter == "A"


def test_case5_low_confidence_never_commits():
    """5. 5 × ('A', 0.5) → no commits (every frame below MIN_CONF)."""
    assert MIN_CONF > 0.5, "test assumes MIN_CONF is above 0.5"
    events = feed(Debouncer(), [("A", 0.5)] * 5)
    assert events == []


def test_case6_long_hold_recommits_after_cooldown():
    """6. 35 × A with no transition → STABLE_FRAMES + GAP_FRAMES cycle repeats.

    Documented behavior: a held sign past the cooldown re-streaks and
    re-commits. With STABLE=5, GAP=3, 35 frames covers
    floor(35 / (5+3)) = 4 commits (frames 5, 13, 21, 29).
    """
    events = feed(Debouncer(), confident("A", 35))
    cycle = STABLE_FRAMES + GAP_FRAMES
    expected = 35 // cycle
    assert len(events) == expected
    assert all(e.letter == "A" for e in events)


# ---------------------------------------------------------------------------
# Commit event mapping (LETTER / SPACE / DELETE)
# ---------------------------------------------------------------------------


def test_space_class_emits_space_event_with_no_letter():
    events = feed(Debouncer(), confident("SPACE", STABLE_FRAMES))
    assert len(events) == 1
    assert events[0].kind == "SPACE"
    assert events[0].letter is None


def test_delete_class_emits_delete_event_with_no_letter():
    events = feed(Debouncer(), confident("DELETE", STABLE_FRAMES))
    assert len(events) == 1
    assert events[0].kind == "DELETE"
    assert events[0].letter is None


def test_letter_event_carries_class_name():
    events = feed(Debouncer(), confident("H", STABLE_FRAMES))
    assert events[0].kind == "LETTER"
    assert events[0].letter == "H"


def test_commit_carries_last_frame_confidence():
    """The CommitEvent.confidence equals the confidence of the triggering frame."""
    stream: list[Prediction] = [
        ("A", 0.90),
        ("A", 0.91),
        ("A", 0.92),
        ("A", 0.93),
        ("A", 0.99),  # triggering frame
    ]
    events = feed(Debouncer(), stream)
    assert events[0].confidence == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Cooldown semantics — "blind" means truly blind
# ---------------------------------------------------------------------------


def test_cooldown_ignores_no_hand_frames():
    """During cooldown, None frames just tick the timer; no reset side-effects."""
    deb = Debouncer()
    feed(deb, confident("A", STABLE_FRAMES))
    assert deb.cooldown == GAP_FRAMES
    feed(deb, [None] * GAP_FRAMES)
    assert deb.cooldown == 0
    assert deb.streak == 0


def test_cooldown_ignores_low_confidence_frames():
    deb = Debouncer()
    feed(deb, confident("A", STABLE_FRAMES))
    events = feed(deb, [("X", 0.1)] * GAP_FRAMES)
    assert events == []
    assert deb.cooldown == 0


def test_cooldown_ignores_other_class_frames():
    """A different high-confidence class during cooldown does not start a new streak."""
    deb = Debouncer()
    feed(deb, confident("A", STABLE_FRAMES))
    feed(deb, confident("B", GAP_FRAMES))
    assert deb.current_class is None
    assert deb.streak == 0


def test_cooldown_decrements_by_one_per_frame():
    deb = Debouncer()
    feed(deb, confident("A", STABLE_FRAMES))
    for expected in range(GAP_FRAMES - 1, -1, -1):
        deb.step(None)
        assert deb.cooldown == expected


# ---------------------------------------------------------------------------
# Edge cases from §4.7
# ---------------------------------------------------------------------------


def test_held_low_confidence_never_commits_and_never_times_out():
    """No 'give up' timer: ambiguous frames just wait, indefinitely."""
    events = feed(Debouncer(), [("A", MIN_CONF - 0.01)] * 1000)
    assert events == []


def test_no_hand_stream_emits_nothing():
    events = feed(Debouncer(), [None] * 50)
    assert events == []


def test_streak_at_min_conf_exactly_commits():
    """A confidence of exactly MIN_CONF is treated as a signal (>= threshold)."""
    events = feed(Debouncer(), confident("A", STABLE_FRAMES, conf=MIN_CONF))
    assert len(events) == 1


def test_streak_just_below_min_conf_does_not_commit():
    """One epsilon below MIN_CONF and we drop the frame as no-signal."""
    events = feed(Debouncer(), confident("A", STABLE_FRAMES, conf=MIN_CONF - 1e-6))
    assert events == []


# ---------------------------------------------------------------------------
# Streak boundary — STABLE_FRAMES exactly is the commit edge
# ---------------------------------------------------------------------------


def test_streak_of_stable_minus_one_does_not_commit():
    events = feed(Debouncer(), confident("A", STABLE_FRAMES - 1))
    assert events == []


def test_streak_resets_to_one_on_class_change_not_zero():
    """B after a partial A streak starts a fresh B=1 streak, not B=0."""
    deb = Debouncer()
    feed(deb, confident("A", STABLE_FRAMES - 1))
    deb.step(("B", 0.95))
    assert deb.current_class == "B"
    assert deb.streak == 1


# ---------------------------------------------------------------------------
# Behavior trace from §4.6 — signing "HELLO"
# ---------------------------------------------------------------------------


def test_hello_trace_emits_h_e_l_l_o():
    """End-to-end shape of the §4.6 worked example.

    Each letter held for STABLE_FRAMES frames, then a GAP_FRAMES gap of
    no-hand frames (simulating the transition between distinct letters).
    The double L just keeps holding L through the cooldown.
    """
    stream: list[Prediction] = []
    stream += confident("H", STABLE_FRAMES) + [None] * GAP_FRAMES
    stream += confident("E", STABLE_FRAMES) + [None] * GAP_FRAMES
    stream += confident("L", STABLE_FRAMES) + confident("L", GAP_FRAMES)
    stream += confident("L", STABLE_FRAMES) + [None] * GAP_FRAMES
    stream += confident("O", STABLE_FRAMES)

    events = feed(Debouncer(), stream)
    letters = [e.letter for e in events]
    assert letters == ["H", "E", "L", "L", "O"]


# ---------------------------------------------------------------------------
# Custom thresholds — the debouncer honors constructor overrides
# ---------------------------------------------------------------------------


def test_custom_stable_frames_changes_commit_threshold():
    deb = Debouncer(stable_frames=3, gap_frames=GAP_FRAMES, min_conf=MIN_CONF)
    events = feed(deb, confident("A", 3))
    assert len(events) == 1


def test_custom_gap_frames_changes_cooldown_length():
    deb = Debouncer(stable_frames=STABLE_FRAMES, gap_frames=10, min_conf=MIN_CONF)
    feed(deb, confident("A", STABLE_FRAMES))
    assert deb.cooldown == 10


def test_custom_min_conf_changes_signal_threshold():
    deb = Debouncer(stable_frames=STABLE_FRAMES, gap_frames=GAP_FRAMES, min_conf=0.5)
    events = feed(deb, [("A", 0.6)] * STABLE_FRAMES)
    assert len(events) == 1
