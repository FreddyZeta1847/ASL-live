"""Debounce state machine — per-frame predictions to commit events.

Implements [`feature-4-debounce.md`](../../../.claude/docs/features/feature-4-debounce.md).
Single counter + blind cooldown. Pure function over the prediction
stream: no I/O, no globals, no clock, fully unit-testable by feeding
canned streams.

Public API
----------
- ``CommitEvent`` — frozen dataclass returned on the frame that triggers
  a commit.
- ``Debouncer`` — state machine with one method, ``step(prediction)``.

Wiring
------
The live recognizer feeds one frame's prediction per call::

    debouncer = Debouncer()
    event = debouncer.step(prediction)   # None | CommitEvent
    if event is not None:
        ...apply buffer logic, print, etc.

``prediction`` is ``None`` for no-hand frames or a ``(label, conf)``
tuple otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from asl_live.config import GAP_FRAMES, MIN_CONF, STABLE_FRAMES


# ---------------------------------------------------------------------------
# Commit event
# ---------------------------------------------------------------------------


CommitKind = Literal["LETTER", "SPACE", "DELETE"]


@dataclass(frozen=True)
class CommitEvent:
    """A single commit emitted by the debouncer.

    ``letter`` is set only when ``kind == "LETTER"``; for SPACE / DELETE
    it is ``None``. ``confidence`` carries the predicted probability of
    the frame that triggered the commit (the last frame of the streak),
    useful for diagnostic overlays.
    """

    kind: CommitKind
    letter: Optional[str]
    confidence: float


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class Debouncer:
    """Converts a noisy per-frame prediction stream into discrete commits.

    Three pieces of state:

    - ``current_class`` — class of the running streak (None if broken).
    - ``streak`` — consecutive frames matching ``current_class``.
    - ``cooldown`` — frames remaining in the blind window after a commit.

    During cooldown every frame is ignored regardless of content; this is
    what lets a held sign emit *one* letter rather than many, and lets a
    repeated letter (e.g. the two L's in HELLO) commit twice without
    requiring a hand-lift.
    """

    def __init__(
        self,
        stable_frames: int = STABLE_FRAMES,
        gap_frames: int = GAP_FRAMES,
        min_conf: float = MIN_CONF,
    ) -> None:
        self._stable_frames = stable_frames
        self._gap_frames = gap_frames
        self._min_conf = min_conf
        self._current_class: Optional[str] = None
        self._streak: int = 0
        self._cooldown: int = 0
        self._last_confidence: float = 0.0

    # -- public accessors (overlay-friendly) ---------------------------------

    @property
    def streak(self) -> int:
        """Current consecutive-match counter, for the diagnostic overlay."""
        return self._streak

    @property
    def cooldown(self) -> int:
        """Remaining cooldown frames, for the diagnostic overlay."""
        return self._cooldown

    @property
    def current_class(self) -> Optional[str]:
        """The class of the running streak, or ``None`` if no streak is active."""
        return self._current_class

    # -- step -----------------------------------------------------------------

    def step(
        self, prediction: Optional[tuple[str, float]]
    ) -> Optional[CommitEvent]:
        """Feed one frame's prediction; return a CommitEvent on commit frames.

        ``prediction`` is ``None`` when the landmark extractor returned
        no hand; otherwise ``(label, confidence)``.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if not _is_signal(prediction, self._min_conf):
            self._reset_streak()
            return None

        # Past the guards: prediction is a real (label, conf) above MIN_CONF.
        assert prediction is not None
        label, confidence = prediction
        self._advance_streak(label, confidence)

        if self._streak >= self._stable_frames:
            return self._emit_commit()
        return None

    # -- internals ------------------------------------------------------------

    def _reset_streak(self) -> None:
        self._current_class = None
        self._streak = 0

    def _advance_streak(self, label: str, confidence: float) -> None:
        """Bump or restart the streak based on whether the class still matches."""
        if label == self._current_class:
            self._streak += 1
        else:
            self._current_class = label
            self._streak = 1
        self._last_confidence = confidence

    def _emit_commit(self) -> CommitEvent:
        """Build the CommitEvent, arm cooldown, clear the streak."""
        assert self._current_class is not None
        event = _make_commit_event(self._current_class, self._last_confidence)
        self._cooldown = self._gap_frames
        self._reset_streak()
        return event


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _is_signal(
    prediction: Optional[tuple[str, float]], min_conf: float
) -> bool:
    """Return True iff the prediction is present and confident enough."""
    if prediction is None:
        return False
    _label, confidence = prediction
    return confidence >= min_conf


def _make_commit_event(label: str, confidence: float) -> CommitEvent:
    """Build a CommitEvent from a class label + confidence.

    Maps the three control classes to their event kinds; everything else
    is a LETTER. Letter-event ``letter`` carries the class name; SPACE /
    DELETE leave it ``None`` so callers can't misuse it.
    """
    if label == "SPACE":
        return CommitEvent(kind="SPACE", letter=None, confidence=confidence)
    if label == "DELETE":
        return CommitEvent(kind="DELETE", letter=None, confidence=confidence)
    return CommitEvent(kind="LETTER", letter=label, confidence=confidence)
