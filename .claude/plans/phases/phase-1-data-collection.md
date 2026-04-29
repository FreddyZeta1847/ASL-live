# Phase 1 — Data collection pipeline

Implementation plan for [PLAN.md §2 / Phase 1](PLAN.md). Approved
2026-04-29. Authoritative reference until phase 1 ships.

## Goal

Stand up the data-collection pipeline so we can produce the labeled
landmark dataset that phase 2 trains on. By end of phase 1 we should
have:
- A working `LandmarkExtractor` that turns a camera frame into a
  normalized 63-dim vector or `None`.
- A one-shot script that converts the Kaggle ASL Alphabet dataset into
  our `.npy` landmark format.
- An interactive collector for the two custom gestures (SPACE, DELETE).
- Unit tests for the pure helpers and the collection protocol
  documented.

## Locked decisions

| Decision | Choice |
|---|---|
| Build tool | `setuptools` + `pip`, PEP 621 in `pyproject.toml` |
| Python version | ≥ 3.11 |
| Kaggle dataset acquisition | Manual download; document in README |
| `collect.py` visual debug | Draw 21 landmarks on the live preview |
| Class set | 26 (A–Y minus J, Z + SPACE + DELETE) |
| Pydantic config persistence | Deferred to phase 6 — phase 1's `config.py` is constants only |

## Deliverables (in implementation order)

### Commit 1 — `chore: bootstrap pyproject + package skeleton`

| File | Purpose |
|---|---|
| `pyproject.toml` | PEP 621 metadata; deps split into base / `[dev]` / `[pi]` profiles per PLAN §1; pytest config |
| `.gitignore` | venv, caches, `data/`, `models/`, IDE folders |
| `src/asl_live/__init__.py` | empty package marker |
| `src/asl_live/recognition/__init__.py` | empty package marker |
| `src/asl_live/capture/__init__.py` | empty package marker |
| `tests/__init__.py` | empty package marker |
| `tests/conftest.py` | `sys.path` fixup for `src/` layout |
| `scripts/__init__.py` | empty package marker |
| `src/asl_live/config.py` | constants: `DATA_DIR`, `MODELS_DIR`, the 26-class list, debounce thresholds, GPIO pins (referenced later) |

Acceptance: `pip install -e ".[dev]"` succeeds on the dev PC; `pytest`
runs (zero tests yet, but no errors).

### Commit 2 — `feat(recognition): landmark extractor with unit tests`

| File | Purpose |
|---|---|
| `src/asl_live/recognition/landmarks.py` | Pure helpers `_normalize(coords)` and `_mirror(coords)`; `LandmarkExtractor` class wrapping MediaPipe Hands per sub-feature #1 |
| `tests/test_landmarks.py` | Unit tests on synthetic 21×3 arrays — wrist-to-origin, scale-to-unit, degenerate input → None, mirror flips x-coordinate |

Acceptance: `pytest tests/test_landmarks.py` passes ≥ 5 tests in < 1 s,
no MediaPipe import required at test time.

### Commit 3 — `feat(scripts): Kaggle ingest preprocessor`

| File | Purpose |
|---|---|
| `scripts/ingest_public.py` | CLI: walks Kaggle dir, runs `extract`, drops no-hand frames, mirror-augments, saves `.npy` to `data/landmarks/<class>/`. Discards J, Z, del, nothing, space classes. Supports `--limit N` for dry runs and `--src PATH` for the Kaggle root |

Acceptance: `python scripts/ingest_public.py --src ~/datasets/asl_alphabet_train --limit 10`
prints stats and exits cleanly. Full run produces ≥ 2,000 `.npy` files
per kept class (after no-hand drops).

### Commit 4 — `feat(capture): interactive SPACE/DELETE collector`

| File | Purpose |
|---|---|
| `src/asl_live/capture/collect.py` | CLI: `python -m asl_live.capture.collect --class SPACE --count 500`. OpenCV preview with class label, sample counter, 21-landmark overlay. Auto-capture on stable hand for 5 frames + 10-frame cooldown. Saves `.npy` + mirrored copy. Q to quit |

Acceptance: collecting 10 SPACE samples saves 20 `.npy` files (10
original + 10 mirrored), each loads back to shape `(63,)`.

### Commit 5 — `docs: collection protocol in README`

| File | Purpose |
|---|---|
| `README.md` | Replace stub with project overview + "Collecting data" section |
| `tree.md` | Refresh to reflect added files |

Acceptance: README reads end-to-end as a guide for someone who just
cloned the repo to reproduce the dataset.

## Data and hardware required (user-side, before commit 3)

- **Kaggle ASL Alphabet** (grassknoted): manual download, ~1 GB
  zipped / ~3 GB extracted. Suggested local path:
  `~/datasets/asl_alphabet_train/`.
- **USB or laptop camera** for `collect.py`. Default
  `cv2.VideoCapture(0)`; `--camera N` flag for alternates.
- **Disk space**: ~5 MB for the resulting `.npy` files.
- **Environment**: Python ≥ 3.11, `pip`.

## Subagents

- **Main thread** implements all 5 commits directly. ~600 lines total,
  design fully specified in `../../docs/features/`.
- **`ml-python-expert`** as an optional single review pass on
  `landmarks.py` + `ingest_public.py` after commit 3 lands. Skip if no
  surprises.
- No subagents for skeleton, tests, or README.

## Testing strategy

- **`pytest`** for `_normalize` and `_mirror`. Synthetic input, no
  MediaPipe.
- **Manual smoke test** of `LandmarkExtractor.extract` against a still
  image — one-liner, only on demand.
- **`ingest_public.py --limit 10`** dry-run before each full run.
- **`collect.py`** tested by collecting 10 samples and inspecting the
  saved `.npy` files.
- No MediaPipe in unit tests — too slow, too fragile.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| MediaPipe install on Windows fails | Verify `import mediapipe` in a smoke test before writing landmarks.py against it |
| OpenCV `VideoCapture(0)` fails on Windows | Add `cv2.CAP_DSHOW` backend fallback in `collect.py` |
| Kaggle dataset directory layout differs from assumption | Quick `ls` of the user's dataset before parsing; document expected layout in README |
| Class folder names in Kaggle dataset don't match our class enum | `ingest_public.py` uses an explicit name-mapping dict instead of trusting folder casing |

## Out of scope for phase 1

- Training the MLP (phase 2).
- Live recognition demo (phase 3).
- Any Pi-side code (phase 4+).
- Argos/Piper setup (phase 5+).
- Pydantic config with file persistence (phase 6).
- Two-handed signs, J/Z motion signs — out of scope for v1 entirely.
