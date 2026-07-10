# Current task

The active task in this session. Update on every session start and
whenever scope shifts. Keep it short — this file is for orientation,
not documentation.

---

## Now

**Phase 5 (translation + TTS) — pipeline live and audibly working.**

5 of 7 phase-5 commits landed (`da17946`..`26fdb04`):
- Argos `Translator` (4 packs preloaded, LRU cache, identity shortcut, exception-swallowing).
- `scripts/setup_argos.py` — idempotent offline pack installer.
- Piper `Speaker` (5 voices preloaded, FileNotFoundError on missing — no eSpeak-NG fallback per updated feature-6 §5).
- `scripts/setup_piper.py` — idempotent voice downloader (atomic .tmp + os.replace).
- `pipeline/main.py` — 3-process orchestrator (`spawn`), `multiprocessing.Event` + `None` sentinels for graceful shutdown, producer-side drop-oldest on `tts_in_q`. CLI: `--lang {it,es,fr,en,de}`, `--camera <int>`.
- `tests/test_pipeline.py` — WordBuffer logic + queue-sentinel + Event-driven shutdown. Full suite green: 160 passed.

User live-tested: end-to-end signing → audible translation **works**.

**Remaining phase-5 commits (housekeeping):**
1. `docs(readme): phase-5 setup steps` — add Argos + Piper install steps, document the new `python -m asl_live.pipeline.main` entry point.
2. `chore: regenerate tree.md`.

**Phase-7 worklist (live-test findings — recollect + retrain pass):**
Real-world drift between phase-2 test set (macro-F1 0.9884) and live signing on user's camera/hand/articulation. Known-hard classes from PLAN.md risk register all surfaced. Specific live confusions captured 2026-05-26:
- **G** — not recognized (likely flickering between G/H, never holds a streak).
- **U → R** (R recognized instead).
- **T → A** (A recognized instead).
- **S / M / N** — three-way confusion cluster (PLAN.md flagged M/N specifically).
- **X vs D** — bent-vs-straight index, depth-z noise likely the cause.
- **F** — not recognized.

Mitigation per PLAN.md §3 risk register: collect ~200 originals each for G, U, T, S, M, N, X, F via `scripts/collect.py`, mirror-augment to ~400, retrain. ~1,600 new originals, ~30 minutes of signing + ~30 minutes retraining.

**Hardware-blocked items:**
- Phase 4 (LCD over I2C) — waiting on Pi + DFR0063 LCD.
- Phase 6 (buttons + audio language menu) — waiting on Pi + buttons.

## Recently shipped

- 2026-05-26 — **Phase 5 orchestrator live + audible** (`26fdb04`).
  `pipeline/main.py` runs 3 processes via `spawn`, graceful shutdown
  on Ctrl-C, drop-oldest on `tts_in_q`. End-to-end signing → speech
  confirmed by user on dev PC.
- 2026-05-26 — **`scripts/setup_piper.py`** (`b2c09a7`) — 5 voices
  downloaded into `PIPER_VOICES_DIR` (env-var overrideable for Pi).
- 2026-05-26 — **Piper `Speaker`** (`edec2ea`) — preloads all 5
  voices, hard-errors on missing voice (feature-6 §5 updated to drop
  eSpeak-NG fallback with rationale).
- 2026-05-26 — **`scripts/setup_argos.py`** (`bfe0a0d`) — idempotent
  offline pack installer.
- 2026-05-26 — **Argos `Translator`** (`da17946`) — warmup at init,
  per-instance LRU(128), identity shortcut for EN, swallows runtime
  exceptions.
- 2026-05-26 — **Phase-5 plan + feature-6 §5 reconciled.** Spec and
  plan now both drop the eSpeak-NG fallback with full 3-point
  rationale captured in the locked doc.
- 2026-05-26 — **Phase-5 plan approved** (`phase-5-translation-tts.md`).
- 2026-05-26 — **Debouncer thresholds tuned live**: STABLE 5→10,
  GAP 3→15 (`c266eed`). Test cases pinned to (5, 3) via
  `spec_debouncer()` so future tuning doesn't break the §4.8
  worked examples.
- 2026-05-26 — **Debouncer unit tests landed (24 tests, all green).**
  Followed by commit of the long-untracked `scripts/demo_recognition.py`
  live-PC demo script. Implementation of feature-4 is now fully on disk
  and exercisable end-to-end.
- 2026-05-26 — **Debouncer state machine implemented** (`debounce.py`)
  per feature-4 §4.2: single counter + blind cooldown, pure function
  over the prediction stream.
- 2026-05-09 — **Phase-3 classifier wrapper** (`classifier.py`, 331861f)
  wraps `onnxruntime` for live inference with the phase-2 ONNX. Lazy
  session load, float32 cast, softmax with logit-max subtraction.
- 2026-05-25 — **Phase-2 acceptance bar PASSED, mlp.onnx exported.**
  Per-cell threshold relaxed from 2 % to 3 % after evidence that the
  test-set size puts 2 % below the noise floor. Macro-F1 0.9884.
- 2026-05-25 — **Recorded another 500 N originals** (→1,000 with
  mirror). N count: 3,586 → 4,586, now in the A–Y range.
- 2026-05-25 — **Second phase-2 verification run.** Test acc improved,
  M→N confusion now passes (1.41 %), N→M improved from 4.65 % to 2.51 %
  but still over the 2.00 % bar. No ONNX exported. Other 24 classes
  clean (≤5 errors per row out of 400–500 samples).
- 2026-05-24 — **Recorded 500 originals each for N and M** (→1,000
  files each after mirror augmentation). On-disk: N 2,586 → 3,586;
  M 3,262 → 4,262.
- 2026-05-14 — **`_load_records` now uses `ThreadPoolExecutor` (16
  workers)** so the cache-cold rebuild on Windows overlaps Defender
  scans instead of serializing them. Expected 4–8× speedup (real
  number will land in issue/001 after the next rebuild). Reviewed by
  `ml-python-expert`. Cached fast path (<1 s) unchanged. Defender
  exclusion (the "free" mitigation) is unavailable on this machine
  because admin policy disables user-added exclusions.
- 2026-05-09 — **Phase-2 verification run executed end-to-end on the
  full ~113 k dataset.** Test acc 0.9871, macro-F1 0.9863 (passes),
  per-cell M↔N failed (above). No ONNX exported. Run details in
  `logs/train_run.out` (untracked).
- 2026-05-09 — `dataset.py`: progress logging every 5 000 files in
  `_load_records` so the silent ~45 min Defender-throttled load is
  visible (issue/001's "no stdout" trap re-bit us on rerun).
- 2026-05-09 — Datapoint for issue/001: real first-load time was
  **2755 s ≈ 46 min**, not the ~10 min projected. Defender exclusion
  on `data/` is now strongly recommended.
- 2026-05-08 — Phase-2 plan approved + mirrored into project tree.
- 2026-05-08 — `feature-3-classifier.md` §3.6 expanded: worked example
  showing precision/recall/F1/macro-F1 computation directly off the
  confusion matrix; implementation notes for `metrics.py`.
- 2026-05-08 — `landmarks.py`: docstring on `extract_with_raw` clarifies
  why it's a separate method from `extract` (commit `cfa3ecd`).
- 2026-05-08 — 100 wrong-class SPACE samples reclassified into DELETE
  (renumbered 0–99); SPACE down to 500 samples + mirrors.
- 2026-04-30 — **Phase 1 complete (5 commits, all pushed).**
- 2026-04-29 — All 10 sub-features locked with full rationale.

## Blocked / waiting on

- Pi 5 + DFR0063 LCD + 2 momentary buttons + USB speaker for phases 4 + 6.
