# Feature 4 — Debounce / commit logic

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/recognition/debounce.py`

**Public API:**
```python
@dataclass
class CommitEvent:
    kind: Literal["LETTER", "SPACE", "DELETE"]
    letter: Optional[str]   # only set when kind == "LETTER"
    confidence: float

class Debouncer:
    def step(self, prediction: Optional[tuple[str, float]]) -> Optional[CommitEvent]:
        """
        Feed one frame's classifier output. Returns a CommitEvent on the
        frame that triggers a commit, otherwise None. Pass `None` for
        frames where the landmark extractor returned no hand.
        """
```

## 4.1 — Problem

The classifier produces a `(class, confidence)` tuple per frame at
15–30 fps. A single held sign produces dozens of nearly-identical
predictions. Frames during gesture transitions (when the hand is
mid-shape) produce noisy, low-confidence, or simply wrong predictions.

We need to convert this messy stream of per-frame predictions into
discrete *commit events* — exactly one event per intentional sign —
without:
- committing 30 copies of "A" while the user is just holding A,
- committing spurious letters from transition frames,
- missing the user's intended next letter.

## 4.2 — Design: single counter + blind cooldown

The locked design is the simplest one that does the job. Three pieces of
internal state:
- `current_class` — the class of the running streak (None if broken).
- `streak` — consecutive frames matching `current_class`.
- `cooldown` — frames remaining in a blind cooldown after the last
  commit.

Per-frame logic:

1. **If cooldown > 0**: decrement and return None. We ignore every input
   during cooldown — it doesn't matter whether the classifier sees the
   same class, a different class, or nothing.
2. **Otherwise, if input is None or below `MIN_CONF`**: reset streak,
   return None.
3. **Otherwise**: if class matches the running streak, increment streak;
   if not, restart streak at 1 with the new class.
4. **If streak reaches `STABLE_FRAMES`**: commit the class, set
   `cooldown = GAP_FRAMES`, reset streak, return the `CommitEvent`.

That's the entire algorithm. ~25 lines including the dataclass.

## 4.3 — Why this is sufficient (and the rejected alternatives)

**Rejected: explicit two-state machine (WATCHING / COOLDOWN).** Earlier
draft. Adds nothing — the "states" are just whether `cooldown == 0` or
not. Removed.

**Rejected: track a per-frame "gap" condition (no-hand or different
class).** Earlier draft. Required defining "what counts as a gap frame"
and led to subtle edge cases when the user holds the same letter and
then signs it again. The blind cooldown is simpler and behaves the same
in normal use.

**Rejected: majority-within-sliding-window for stability detection.**
Tolerates classifier flicker (e.g., A,A,A,B,A,A,A) more gracefully than
strict consecutive matching, but adds a window length parameter and
hides bugs in the classifier behind tolerant logic. We use **strict
consecutive** for v1: any non-matching frame resets the streak. If
phase-3 testing reveals real flicker problems we'll switch — but most
likely the classifier is good enough that strict works.

## 4.4 — Trade-off introduced by the blind cooldown

Because the cooldown is blind, after a commit the user must **move the
hand within `STABLE_FRAMES + GAP_FRAMES` frames** or the same letter
will commit a second time. With defaults (5 + 3 = 8 frames at 30 fps =
~270 ms) this is tight but reasonable for ASL fingerspelling, which
naturally transitions between shapes.

This is also what makes **repeated letters** (the two L's in "HELLO")
work cleanly without requiring the user to lift their hand — they just
need to make any small transition during the cooldown. With the
previous "gap requires no-hand or different class" design, repeated
letters required a hand-lift, which is unnatural for ASL.

The realistic tuning is probably `GAP_FRAMES = 10–15` (~330–500 ms) once
we test live. Default stays at 3 until phase 7.

## 4.5 — Parameters

| Parameter | Default | Source / role |
|---|---|---|
| `STABLE_FRAMES` | 5 | How many consecutive same-class frames before commit. ~167 ms at 30 fps. |
| `GAP_FRAMES` | 3 | Blind cooldown after commit. ~100 ms at 30 fps. |
| `MIN_CONF` | 0.85 | Locked from feature 3 §3.7. Frames below this are treated as "no signal." |

All three live in `config.py` and are tuned empirically in phase 7. The
debounce module reads them at construction time.

## 4.6 — Behavior trace: signing "HELLO" (STABLE=5, GAP=3)

| Frames | Input | State after | Result |
|---|---|---|---|
| 1–5 | H, H, H, H, H | streak=5 → commit | **emits LETTER('H')**, cooldown=3 |
| 6–8 | (anything) | cooldown ticks down | None |
| 9–13 | E, E, E, E, E | streak=5 → commit | **emits LETTER('E')**, cooldown=3 |
| 14–16 | (transition / hand still moving) | cooldown ticks down | None |
| 17–21 | L, L, L, L, L | streak=5 → commit | **emits LETTER('L')**, cooldown=3 |
| 22–24 | L, L, L (user still holding L) | cooldown ticks down | None |
| 25–29 | L, L, L, L, L | streak=5 → commit | **emits LETTER('L')** (second L), cooldown=3 |
| 30+ | … O eventually | … | **emits LETTER('O')** when streak hits 5 |

Note that the second L commits *without* the user lifting their hand —
they just need to keep holding L past the cooldown.

## 4.7 — Edge-case behavior (locked)

**Empty buffer + SPACE / DELETE.** The debounce just emits the event;
the *recognizer worker* is responsible for handling these:
- SPACE on empty buffer → silent no-op (no translation triggered).
- DELETE on empty buffer → silent no-op.
- Rationale: don't punish accidental signs with audible feedback.

**Held ambiguous sign with no clear class.** Confidence stays below
`MIN_CONF`, streak never grows, no commit ever fires. The Debouncer
simply waits. No timeout, no give-up logic — the user makes a clearer
sign and it commits.

**Classifier flicker (A,A,A,B,A,A,A).** Strict consecutive matching: the
single B resets the streak to 1. The next A starts a fresh streak. If
the classifier is well-trained (per feature 3 acceptance bar) this is
rare.

## 4.8 — Testability

`Debouncer.step()` is a pure function over the prediction stream — no
I/O, no globals, no clock. It can be unit-tested by feeding canned
streams and asserting the emitted events. Tests live in
`tests/test_debounce.py` and run in milliseconds without camera or
model.

Required test cases:
1. 5 × ("A", 0.95) → emits LETTER('A') exactly once.
2. 5 × ("A", 0.95) followed by 8 × None → still exactly one
   LETTER('A') (cooldown then idle).
3. 5 × ("A", 0.95), 3 × None, 5 × ("A", 0.95) → emits LETTER('A')
   twice (cooldown elapsed, fresh streak).
4. 4 × ("A", 0.95), 1 × ("B", 0.95), 5 × ("A", 0.95) → emits
   LETTER('A') exactly once (the B reset the streak).
5. 5 × ("A", 0.5) → no emit (below MIN_CONF).
6. 5 × ("A", 0.95) followed by 30 × ("A", 0.95) without any
   transition → emits LETTER('A') ~3 times (cooldown elapses,
   streak rebuilds, recommits — documented behavior).

The tests are written *before* the implementation, per phase 3 of
PLAN.md.

## Out of scope for this feature

- Multi-frame motion features (would be needed for J/Z signs).
- Adaptive thresholds based on signing speed.
- Confidence-weighted commits (e.g., averaging confidence across the
  streak).
- "Soft commit" UI that previews the imminent commit before it happens.
