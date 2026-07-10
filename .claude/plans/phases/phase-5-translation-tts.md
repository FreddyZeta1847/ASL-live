# Phase 5 — Translation + TTS (PC orchestrator)

Implementation plan for [PLAN.md §2 / Phase 5](../PLAN.md). Drafted 2026-05-26. Authoritative reference until phase 5 ships.

---

## Context

Phase 3 shipped on 2026-05-26: live recognition demo on PC with debouncer + buffer + terminal printout. `STABLE_FRAMES=10`, `GAP_FRAMES=15` tuned live from real signing.

Phase 5 wires the recognized words into **Argos** (offline translation) and **Piper** (offline TTS) so the device actually *speaks* the translated word out the PC speakers. Module designs are locked in [`feature-5-translation.md`](../../docs/features/feature-5-translation.md) and [`feature-6-tts.md`](../../docs/features/feature-6-tts.md) — no new design.

PLAN.md's phase 4 (LCD over I2C) requires the Pi and is **swapped after phase 5** in our ordering, since the LCD is the only thing blocking on hardware. Argos + Piper run identically on Windows and Pi.

---

## Scope decisions for this phase

| Decision | Choice |
|---|---|
| Process model | **Full 3-process orchestrator now** — `recognizer`, `translator`, `speaker` connected by 2 `multiprocessing.Queue`s. Matches the production architecture from PLAN.md and feature-5/6. |
| Active language | **CLI-fixed for phase 5** (`--lang it` default). Button-driven runtime switching deferred to phase 6 (lang_menu). Each `(word, lang)` queue message carries its own lang, so phase 6 won't need to restart any workers. |
| Word buffer location | **Inside the recognizer process** — buffer state is per-frame and tightly coupled to the debouncer. SPACE flushes the assembled word into the translation queue; LETTER commits stay process-local. |
| Boot announcement | **Deferred to phase 6** — feature-6 §7 spec'd it as a `lang_menu` responsibility, and lang_menu doesn't exist yet. Phase 5 boots silently. |
| Piper voice install path | New config constant `PIPER_VOICES_DIR`. Default `models/piper/voices/` (gitignored, sibling to `mlp.onnx`). Overridable by env var `ASL_PIPER_VOICES_DIR`. Phase-7 Pi systemd unit will export `ASL_PIPER_VOICES_DIR=/opt/piper/voices/` per feature-6 §3 — no code change between PC and Pi. |
| Missing Piper voice | **Hard error at `Speaker.__init__`** — `FileNotFoundError`. Matches feature-6 §5 (locked doc updated alongside this plan to drop the eSpeak-NG fallback for full rationale). |
| Demo script (`scripts/demo_recognition.py`) | **Untouched.** Keeps working as the no-audio debouncer-only smoke test. The audio path lives in `pipeline/main.py`. |
| Pi-side latency target | **Deferred** — feature-5/6 spec'd ≤500 ms SPACE→audio on Pi 5. Phase 5 on PC has the softer target ≤1.5 s. Pi-port + measurement folded into phase 4 (LCD) when the hardware arrives. |
| Unit tests | Required this time — translator and speaker mock cleanly. The 3-process orchestrator is verified manually, not in pytest (multiprocessing on Windows is flaky in CI). |

---

## Pipeline at a glance

```
                        ┌─── main process ──────────────────┐
                        │  parse CLI, spawn 3 children,     │
                        │  trap SIGINT, send sentinels,     │
                        │  join, exit.                      │
                        └──────────────┬────────────────────┘
                                       │ spawns
       ┌───────────────────────────────┼───────────────────────────────┐
       ▼                               ▼                               ▼
┌──────────────┐ translation_in ┌──────────────┐  tts_in       ┌──────────────┐
│  recognizer  │  (word, lang)  │  translator  │ (text, lang)  │    speaker   │
│              │ ──────────────▶│              │──────────────▶│              │
│  camera      │   maxsize=8    │ Argos        │  maxsize=3    │ Piper voice  │
│  landmarks   │                │ (4 packs     │  drop-oldest  │ (5 voices    │
│  classifier  │                │  preloaded)  │               │  preloaded)  │
│  debouncer   │                │              │               │              │
│  buffer      │                │              │               │              │
└──────────────┘                └──────────────┘               └──────────────┘
```

- **`recognizer`** owns the camera and emits one `(word, lang)` per SPACE commit. LETTER commits append to its in-process buffer; DELETE pops; empty-buffer SPACE/DELETE = silent (feature-4 §4.7).
- **`translator`** is a blocking `queue.get()` loop. Reads `(word, lang)`, calls `Translator.translate(word, lang)`, pushes `(translated, lang)` to `tts_in_queue`.
- **`speaker`** is a blocking `queue.get()` loop. Reads `(text, lang)`, calls `Speaker.speak(text, lang)` which blocks until ALSA/WASAPI finishes playing.
- Each queue is `multiprocessing.Queue`. `translation_in` is size 8 (soft cap — a SPACE wave shouldn't ever queue that many). `tts_in` is size 3 with **drop-oldest** semantics per feature-6 §4.

---

## Modules

### `src/asl_live/translation/translator.py` — NEW

~60 lines.

- **`Translator`** class. `__init__` runs one dummy translation per language pair (warmup, feature-5 §3).
- **`translate(word: str, target: str) -> str`** — lowercases input, returns lowercased translation. Identity shortcut when `target == "en"`. Empty input → empty output. Wrapped with `functools.lru_cache(maxsize=128)`. Any `argostranslate` exception caught → logs warning, returns the lowercased input unchanged.
- Reads installed packs at init; raises `RuntimeError` if any of {it, es, fr, de} is missing (setup must have been run).

### `src/asl_live/tts/speaker.py` — NEW

~60 lines.

- **`Speaker`** class. `__init__` preloads all 5 Piper voices from `PIPER_VOICES_DIR` (feature-6 §2). Missing voice → `FileNotFoundError` (no eSpeak-NG fallback per scope decision).
- **`speak(text: str, lang: str) -> None`** — synthesizes via Piper → 16-bit PCM, plays via `sounddevice.play(...)` and blocks on `sounddevice.wait()`. Empty input → no-op. Any synthesis exception caught → logs warning, returns.

### `src/asl_live/pipeline/main.py` — NEW

~150 lines.

- CLI: `--lang {it,es,fr,en,de}` (default `it`), `--camera <int>` (default 0).
- Spawns three `multiprocessing.Process`-es (`spawn` start method — required on Windows, also fine on Linux).
- Two `multiprocessing.Queue`s + one `multiprocessing.Event` (`shutdown_event`) shared with all workers.
- Each worker function is a module-level function (picklable for `spawn`). They construct their respective heavy objects (camera, Translator, Speaker) **inside** the worker, not in main — `argostranslate`'s installed-packs cache and Piper's voice sessions don't survive pickling.
- Main process:
  1. Parse args.
  2. Construct queues + `shutdown_event`.
  3. Spawn workers with `(lang, camera_index, queues..., shutdown_event)` args.
  4. Install SIGINT handler: `shutdown_event.set()` AND push `None` sentinel into both queues so any blocked `get()` returns immediately.
  5. `join()` all three with a 5 s timeout each. After timeout, `terminate()` then `join()`.
- Workers:
  - `recognizer_loop(lang, camera_index, translation_in_q, shutdown_event)`: same loop as `demo_recognition.py` but on SPACE commit pushes `(word, lang)` instead of printing. **Polls `shutdown_event.is_set()` at the top of each frame** — the recognizer doesn't read from a queue, so the Event is its only shutdown signal. Releases camera + extractor in `finally:`.
  - `translator_loop(translation_in_q, tts_in_q, shutdown_event)`: blocking `get()`. On sentinel `None` or `shutdown_event.is_set()`, push `None` downstream then exit.
  - `speaker_loop(tts_in_q, shutdown_event)`: blocking `get()`. On sentinel `None` or `shutdown_event.is_set()`, exit. On real message, call `Speaker.speak`. Drop-oldest implemented via `Queue(maxsize=3)` + producer-side `put_nowait` with `except queue.Full: drop one` — see implementation note below.

**Drop-oldest implementation note.** Python's `multiprocessing.Queue` lacks a native drop-oldest mode. The cleanest implementation is on the **producer side** (translator worker): on `put_nowait` raising `queue.Full`, drain one message with `get_nowait` then re-put. Tiny risk window where the consumer just got the would-be-dropped message; that's acceptable per feature-6 §4 ("occasional dropped words").

### `src/asl_live/config.py` — MODIFIED

Add:
```python
PIPER_VOICES_DIR: Path = Path(
    os.environ.get("ASL_PIPER_VOICES_DIR", str(MODELS_DIR / "piper" / "voices"))
)
"""Where Piper voice .onnx + .json files live. Override with env var
ASL_PIPER_VOICES_DIR; the Pi systemd unit (phase 7) sets this to
/opt/piper/voices/ per feature-6 §3."""

ARGOS_LANGS: tuple[str, ...] = ("it", "es", "fr", "de")
"""Argos pack target languages, EN→x. Plus 'en' is identity-shortcut."""

TTS_QUEUE_MAX: int = 3
"""Bounded TTS queue; oldest dropped when full. Per feature-6 §4."""

TRANSLATION_QUEUE_MAX: int = 8
"""Bounded translation queue. Soft cap — SPACE bursts shouldn't fill this."""
```

`STABLE_FRAMES`, `GAP_FRAMES`, `MIN_CONF` unchanged.

### `scripts/setup_argos.py` — NEW

~50 lines. Cross-platform.

- For each lang in `ARGOS_LANGS`: check if `en→<lang>` is already installed via `argostranslate.translate.get_installed_languages()`; skip if present.
- Otherwise download the `.argosmodel` from the pinned Argos package URL (URLs pinned in the script — Argos's mirror at argosopentech.com/argospm/v1/).
- Call `argostranslate.package.install_from_path(...)` on the downloaded file.
- Idempotent. Re-runnable safely.

### `scripts/setup_piper.py` — NEW

~60 lines. Cross-platform.

- For each lang in `("it", "es", "fr", "en", "de")`: check if `<PIPER_VOICES_DIR>/<lang>.onnx` exists; skip if present.
- Otherwise download the `.onnx` + matching `.json` from the pinned Piper voices mirror (rhasspy/piper-voices on Hugging Face).
- Voice filenames pinned in the script (medium quality per feature-6 §1):
  - `it_IT-riccardo-x_low` (waiting for a medium-quality IT to land; x_low is the placeholder)
  - `es_ES-davefx-medium`
  - `fr_FR-siwis-medium`
  - `en_US-amy-medium`
  - `de_DE-thorsten-medium`
- Rename downloaded files to `<lang>.onnx` / `<lang>.json` inside `PIPER_VOICES_DIR` so the Speaker class has a flat lookup.
- Idempotent.

### `pyproject.toml` — MODIFIED

Add to `[dev]` (and mirror into `[pi]`):
- `argostranslate` (pin version once we pick it)
- `piper-tts`
- `sounddevice`

### `tests/test_translator.py` — NEW

`argostranslate.translate` is monkey-patched. Cases:

1. Identity shortcut: `translate("HELLO", "en")` returns `"hello"` and the mock is never called.
2. Lowercase normalization: `translate("HELLO", "it")` calls the mock with `"hello"` (not `"HELLO"`).
3. LRU cache: two identical calls invoke the mock once.
4. Empty string: `translate("", "it")` returns `""`, mock not called.
5. Exception swallowed: mock raises → return value is the lowercased input, no exception escapes.
6. Constructor warms up all 4 pairs (the mock is invoked 4 times during `__init__`).

### `tests/test_speaker.py` — NEW

`piper` and `sounddevice.play` are monkey-patched. Cases:

1. `speak("ciao", "it")` invokes the Italian voice's `synthesize`.
2. Missing voice file → `Speaker.__init__` raises `FileNotFoundError` (no fallback).
3. Synthesis exception swallowed during `speak()`, no crash.
4. Empty input → no-op (synthesize not called).
5. Constructor preloads all 5 voices.

### `tests/test_pipeline.py` — NEW

Two test sets:

**Word-buffer logic** (pure, no processes). The buffer state machine is extracted as a small class `WordBuffer` (similar to the one already in `demo_recognition.py`) so the SPACE/DELETE semantics from feature-4 §4.7 are unit-testable without spawning anything:
- LETTER appends.
- DELETE on non-empty pops; on empty does nothing and returns `None`.
- SPACE on non-empty flushes and returns the word; on empty returns `None`.

**Graceful shutdown** (real `multiprocessing.Queue` + `Event`, real `Process`). Two stubs:
- A queue-driven stub worker: send `None`, assert it exits within 2 s.
- An Event-driven stub worker (mirrors the recognizer's poll loop): set `shutdown_event`, assert it exits within 2 s.

**Not tested in pytest:** full 3-process boot-up with real camera/Argos/Piper. Verified manually per the acceptance criteria below.

---

## Setup scripts — execution order

First-time setup (once per dev machine, once at Pi provisioning):
1. `pip install -e .[dev]` (or `[pi]`) — installs argostranslate, piper-tts, sounddevice.
2. `python scripts/setup_argos.py` — downloads + installs 4 Argos packs.
3. `python scripts/setup_piper.py` — downloads 5 Piper voices into `PIPER_VOICES_DIR`.

These steps are documented in README's setup section as part of phase 5's deliverables.

---

## Commits

| # | Commit | Files |
|---|---|---|
| 1 | `feat(translation): Argos wrapper with warmup + LRU cache (feature-5)` | `translator.py`, `tests/test_translator.py`, `config.py` (ARGOS_LANGS) |
| 2 | `feat(scripts): setup_argos — offline pack installer` | `scripts/setup_argos.py` |
| 3 | `feat(tts): Piper wrapper with eSpeak-NG fallback (feature-6)` | `speaker.py`, `tests/test_speaker.py`, `config.py` (PIPER_VOICES_DIR) |
| 4 | `feat(scripts): setup_piper — voice downloader` | `scripts/setup_piper.py` |
| 5 | `feat(pipeline): 3-process orchestrator with multiprocessing queues` | `pipeline/main.py`, `tests/test_pipeline.py`, `config.py` (queue sizes) |
| 6 | `chore: add argostranslate + piper-tts + sounddevice to pyproject` | `pyproject.toml` |
| 7 | `docs(readme): phase-5 setup steps (Argos packs, Piper voices)` | `README.md` |
| 8 | `chore: regenerate tree.md after phase-5 additions` | `tree.md` |

Each pushed after commit per git-autopush.

---

## Acceptance criteria (PLAN.md §2 phase 5, PC-only — Pi latency deferred)

1. `python scripts/setup_argos.py` then `python scripts/setup_piper.py` both run idempotently — second run is a no-op.
2. `python -m asl_live.pipeline.main --lang it` boots cleanly. No console errors. All 4 Argos packs + 5 Piper voices loaded inside ~5 s on dev PC.
3. Manual: sign **H E L L O + SPACE** → speaker says "ciao". SPACE→audio latency ≤ 1.5 s on dev PC.
4. Manual: with `--lang fr`, `--lang es`, `--lang de` — same word "HELLO" plays "bonjour", "hola", "hallo" respectively.
5. Manual: with `--lang en` — identity shortcut, plays "hello" with no Argos call (verifiable by adding a temporary log line during validation).
6. Manual: while speaker is mid-utterance, sign a second word → it lands in queue and plays after the first finishes. Queueing more than 3 words drops the oldest (verifiable by counting expected vs spoken outputs).
7. Manual: camera FPS in overlay stays ≥ 25 fps during translation+synthesis — recognizer process not blocked by speaker.
8. Manual: Ctrl-C → all 3 processes exit within 2 s, camera released (re-running the command immediately works without "device busy").
9. All new unit tests pass; full suite stays green.

---

## Out of scope for this phase

- LCD output (phase 4, Pi).
- Boot announcement / language-name speak on startup (phase 6, lang_menu).
- Button-driven language switching at runtime (phase 6).
- Pi-side latency target ≤ 500 ms SPACE→audio (deferred to phase 4 when the Pi is in hand).
- Voice quality / per-language voice ear-tuning (phase 7).
- CSV verbose logging of every prediction (phase 7).
- Sentence-level translation context (feature-5 out-of-scope).
- Multi-hand or two-handed signs (feature-1 out-of-scope).
- eSpeak-NG fallback for missing voices — dropped from the locked spec (feature-6 §5) during phase-5 planning; missing voice is a `FileNotFoundError` at `Speaker.__init__`. Revisit only if Piper ever drops one of our 5 languages with no neural replacement.
