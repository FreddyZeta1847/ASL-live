# Phase 3 — Live recognition demo (PC-only)

Implementation plan for [PLAN.md §2 / Phase 3](../PLAN.md). Approved 2026-05-26. Authoritative reference until phase 3 ships.

---

## Context

Phase 2 shipped on 2026-05-25: trained MLP at macro-F1 0.9884, exported to `models/mlp.onnx`, runtime wrapper `recognition/classifier.py` already in place. The acceptance bar was relaxed from 2 % to 3 % per-cell after evidence that 2 % was below the noise floor for the test-set size; documented locally in `current-task.md`.

Phase 3 wires the classifier into a live camera loop on the **PC only**. Debounce algorithm is locked in [`feature-4-debounce.md`](../../docs/features/feature-4-debounce.md) — no new design.

---

## Scope decisions for this phase

| Decision | Choice |
|---|---|
| Demo script scope | **Minimal**: print each committed letter alone; print SPACE-flushed word with a distinguishing prefix (`→ HELLO`). No CSV instrumentation. |
| `MIN_CONF` calibration | Accept the locked 0.85 default. Revisit only if HELLO acceptance fails. |
| Pi port / fps gate | **Deferred to phase 4** (phase 4 needs the Pi for LCD work; bundling avoids one trip through Pi-setup pain). |
| Unit tests | Required by PLAN.md, but **deferred to next session** at user request. Implementation ships first; tests added when user can run them. |

---

## Pipeline at a glance

```
camera → BGR frame
   │
   ▼
LandmarkExtractor.extract_with_raw(frame)        ← feature-1 (existing)
   │
   ├── None  ──▶ debouncer.step(None)
   └── (raw, normalized) ──▶ classifier.predict(normalized)   ← feature-3 (existing)
                                  │
                                  ▼
                            (label, conf)
                                  │
                                  ▼
                       debouncer.step((label, conf))   ← feature-4 (NEW)
                                  │
                                  ├── None         → nothing
                                  └── CommitEvent  → buffer logic + print
                                          │
                                          ├── LETTER 'H'  → buffer += 'H', print "H"
                                          ├── DELETE      → buffer = buffer[:-1], print "←"
                                          └── SPACE       → print f"→ {buffer}", buffer = ""
```

---

## Modules

### `src/asl_live/recognition/debounce.py` — NEW

Pure state machine. ~50 lines.

- **`CommitEvent`** — frozen dataclass: `kind: Literal["LETTER", "SPACE", "DELETE"]`, `letter: Optional[str]`, `confidence: float`.
- **`Debouncer`** — three pieces of state (`current_class`, `streak`, `cooldown`), one method `step(prediction)`.
  - `prediction` is `None` for no-hand frames OR a `(label, conf)` tuple.
  - Returns `None` on most frames; returns a `CommitEvent` on the frame that triggers a commit.
- Reads `STABLE_FRAMES`, `GAP_FRAMES`, `MIN_CONF` from `config.py` at construction time.
- Pure: no I/O, no globals, no clock. Unit-testable by feeding canned streams.

Algorithm (locked in feature-4 §4.2):
1. If `cooldown > 0`: decrement, return None.
2. If input is None or `conf < MIN_CONF`: reset streak, return None.
3. If class matches running streak: increment; else restart streak at 1.
4. If `streak == STABLE_FRAMES`: emit `CommitEvent`, set `cooldown = GAP_FRAMES`, reset streak.

### `scripts/demo_recognition.py` — NEW

Live camera demo. ~120 lines.

- CLI: `--camera <int>` (default 0). No other flags in v1.
- Reuses `open_camera()` pattern from `scripts/collect.py` (handles Windows DSHOW fallback).
- Per-frame loop:
  1. `cap.read()` → BGR frame.
  2. `LandmarkExtractor.extract_with_raw(frame)` → `(raw, normalized)` or `None`.
  3. If a vector exists, `Classifier.predict(normalized)` → `(label, conf)`.
  4. `Debouncer.step(...)` with the prediction or `None`.
  5. On `CommitEvent`, apply buffer logic + print.
  6. Draw OpenCV overlay (top-1 class, confidence, streak counter, buffer, fps).
  7. `cv2.imshow`; quit on `q`.

Output formatting:
- `LETTER` commit → print the letter (e.g. `H`).
- `SPACE` commit → print `→ {buffer}` (only if buffer non-empty; silent no-op otherwise per feature-4 §4.7).
- `DELETE` commit → print `←` (only if buffer non-empty; silent no-op otherwise).

### `src/asl_live/config.py` — UNCHANGED

`STABLE_FRAMES`, `GAP_FRAMES`, `MIN_CONF` already added in a prior commit (lines 77-84). Phase 3 imports them directly.

### `tests/test_debounce.py` — DEFERRED

User requested skip for this session. Will add in next session per the six locked cases in feature-4 §4.8.

---

## Commits

| # | Commit | Files |
|---|---|---|
| 1 | `feat(recognition): debounce state machine (feature-4)` | `debounce.py` |
| 2 | `feat(scripts): demo_recognition — live PC recognition loop` | `demo_recognition.py` |
| 3 | `chore: regenerate tree.md after phase-3 additions` | `tree.md` |

Each pushed after commit per git-autopush.

---

## Acceptance criteria (PLAN.md §2 phase 3, minus Pi-fps)

1. Manual: signing **H E L L O + SPACE** produces:
   ```
   H
   E
   L
   L
   O
   → HELLO
   ```
2. Manual: signing a letter then **DELETE** removes it from the buffer.
3. Manual: held single sign produces *one* letter, not many.
4. Demo overlay shows ≥25 fps on PC.

Pi-fps gate (≥15 fps) moved to phase 4.

Unit tests (`tests/test_debounce.py`) carried over to next session.

---

## Out of scope for this phase

- Pi port, LCD output, translation, TTS, buttons (phases 4-6).
- CSV instrumentation / `--verbose` flag (phase 7).
- `MIN_CONF` calibration tooling (deferred until/unless HELLO test fails).
- Multi-process pipeline / `pipeline/main.py` (phase 5).
- Adaptive thresholds, soft-commit UI, confidence averaging (feature-4 §4.8 out-of-scope).
