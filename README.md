<p align="center">
  <h1 align="center">ASL-live</h1>
  <p align="center"><b>Offline ASL → Speech on Raspberry Pi</b></p>
  <p align="center">Sign ASL letters into a USB camera, hear the translated word spoken back in your chosen language — fully phone-free and internet-free.</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-Hand_Landmarks-0097A7?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Video-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/ONNX_Runtime-Inference-005CED?style=for-the-badge&logo=onnx&logoColor=white" />
  <img src="https://img.shields.io/badge/Argos-Translate-22B573?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Piper-TTS-9B59B6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Raspberry_Pi-5-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white" />
</p>

The full design lives under [`.claude/docs/`](.claude/docs/) (architecture, tech stack, per-feature decisions). The implementation roadmap and per-phase plans live under [`.claude/plans/`](.claude/plans/).

## Status

Phases 1–5 complete on the dev PC. The full pipeline runs end-to-end
on a webcam: sign letters → see them debounced into words → hear the
translated word spoken through the PC speakers.

| Phase | Scope | Status |
|---|---|---|
| 1 | Data collection | ✅ done |
| 2 | Train MLP classifier | ✅ done (macro-F1 0.9884) |
| 3 | Live recognition + debounce | ✅ done |
| 5 | Translation (Argos) + TTS (Piper) | ✅ done (PC) |
| 4 | LCD over I2C | ⏳ blocked on Pi hardware |
| 6 | Buttons + audio language menu | ⏳ blocked on Pi hardware |
| 7 | Polish, tuning, systemd autostart | ⏳ pending |

See [`.claude/plans/PLAN.md`](.claude/plans/PLAN.md) for the full
7-phase roadmap.

## Quick install

Two install profiles are defined in `pyproject.toml`:

- **`[dev]`** — development PC: training stack (PyTorch, scikit-learn,
  XGBoost) and tests.
- **`[pi]`** — Raspberry Pi: GPIO and I2C peripheral libraries
  (`RPLCD`, `gpiozero`, `smbus2`).

```bash
git clone https://github.com/FreddyZeta1847/ASL-live.git
cd ASL-live

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # development PC
# pip install -e ".[pi]"           # on the Raspberry Pi

python scripts/setup_models.py     # download the MediaPipe hand model (~13 MB, one-time)

pytest                              # unit tests should pass
```

Python ≥ 3.11 required.

`setup_models.py` fetches `hand_landmarker.task` from Google's CDN
into `models/`. The new MediaPipe Tasks API requires this file on
disk; the script is idempotent (skips if already present).

## Collecting data

The classifier is trained on a mix of two sources, both producing the
same `.npy` files under `data/landmarks/<class>/`:

1. **Kaggle ASL Alphabet** (grassknoted) for the 24 letter classes
   (A–Y minus J and Z).
2. **Self-collected** SPACE and DELETE gestures via the interactive
   collector.

### 1. Letters — Kaggle ingest

Download the dataset manually from
<https://www.kaggle.com/datasets/grassknoted/asl-alphabet> (Kaggle
account required, ~1 GB zipped). Unpack so the layout is:

```
~/datasets/asl_alphabet_train/
├── A/  A1.jpg  A2.jpg  …
├── B/
├── …
├── del/        ← discarded
├── nothing/    ← discarded
└── space/      ← discarded
```

Then run the ingest script:

```bash
# Dry run on the first 10 images per class — verifies plumbing, ~1 minute
python scripts/ingest_public.py --src ~/datasets/asl_alphabet_train --limit 10

# Full run — ~1–2 hours of CPU
python scripts/ingest_public.py --src ~/datasets/asl_alphabet_train
```

The script discards J, Z, `del`, `nothing`, `space` (per
[feature 2 decisions](.claude/docs/features/feature-2-data-collection.md)).
For every kept image it runs MediaPipe Hands, drops images where no
hand is detected, and saves the normalized 63-dim vector plus a
mirror-augmented copy (suffix `_m`). Expect roughly 5,000–6,000 `.npy`
files per kept class once the run finishes (≈ 3,000 images × 2 from
mirroring, minus the ~5 % that MediaPipe couldn't lock on to).

### 2. Custom gestures — interactive collector

For SPACE (open palm with fingers spread) and DELETE (closed fist with
thumb pointing down), collect ~500 samples each from your own camera:

```bash
python scripts/collect.py --class SPACE  --count 500
python scripts/collect.py --class DELETE --count 500
```

The preview window shows:
- The class name and saved/target counter.
- A status indicator: `NO HAND` / `HOLD k/5` / `COOLDOWN n` / `DONE`.
- The 21 hand keypoints overlaid as yellow dots so you can see
  MediaPipe locking on cleanly.

Auto-capture fires every time MediaPipe sees a stable hand for 5
consecutive frames, then waits 10 frames before the next capture.
Press `q` in the preview window to stop early. Sessions append to the
same `<class>/` folder, so you can split the 500 samples across
several short sittings.

### Diversity tips

For best generalization on the small SPACE / DELETE classes, vary your
captures within each session:

- Move the hand to each corner of the frame, not just the center.
- Try a close distance (~30 cm) and a far one (~80 cm).
- Slightly tilt the hand between captures.
- If possible, run a second session under different lighting (lamp
  vs daylight).

You do not have to be perfect — the training pipeline applies
Gaussian-noise, scale, and translation augmentation to compensate for
natural variation.

## Translation + speech (phase 5)

The full pipeline ships English-letter recognition → word assembly →
offline translation → offline speech, all wired through three OS
processes connected by `multiprocessing.Queue`s. Engines:

- **Translation:** [Argos Translate](https://github.com/argosopentech/argos-translate) —
  offline neural MT, one `.argosmodel` pack per language pair.
- **Speech:** [Piper](https://github.com/rhasspy/piper) — offline
  neural TTS, one `.onnx` voice per language.

### One-time setup

After `pip install -e .[dev]`, run the two installers. Both are
idempotent — safe to re-run after a dropped download:

```bash
python scripts/setup_argos.py       # ~800 MB total (4 packs)
python scripts/setup_piper.py       # ~250–500 MB total (5 voices)
```

Packs install to argostranslate's platform home dir
(`%LOCALAPPDATA%\argos-translate\` on Windows,
`~/.local/share/argos-translate/` on Linux/Pi). Voices install to
`models/piper/voices/` by default; override with
`ASL_PIPER_VOICES_DIR` (the Pi systemd unit will set this to
`/opt/piper/voices/`).

### Running the pipeline

```bash
python -m asl_live.pipeline.main --lang it    # default
python -m asl_live.pipeline.main --lang fr    # any of it, es, fr, en, de
python -m asl_live.pipeline.main --camera 1   # second webcam
```

The OpenCV window shows live landmarks, top-1 class + confidence,
debouncer status, and the running word buffer. Sign letters; on a
SPACE-sign commit the assembled word is translated and spoken
through the default audio device. DELETE removes the last letter.
`q` (in the OpenCV window) or `Ctrl-C` shuts down all three workers.

Target language is fixed for the run in phase 5 — phase 6 will add
button-driven language switching at runtime.

### Lightweight smoke test (no audio)

`scripts/demo_recognition.py` exercises the camera → landmarks →
classifier → debouncer path and prints commit events to the
terminal. No Argos, no Piper, no audio device required. Useful when
debugging the recognition stack in isolation:

```bash
python scripts/demo_recognition.py
```

## Project layout

```
src/asl_live/
├── recognition/     Hand landmarks, classifier, debouncer
├── train/           MLP training stack (PC-only)
├── translation/     Argos wrapper (4 packs preloaded)
├── tts/             Piper wrapper (5 voices preloaded)
└── pipeline/        3-process orchestrator + word buffer
tests/               Unit tests (pytest, no hardware required)
scripts/             One-shot CLIs (ingest, collect, setup_argos, setup_piper, demos)
data/                Generated datasets (gitignored)
models/              Trained model artifacts + Piper voices (gitignored)
.claude/             Project knowledge — design docs, plans, agents (gitignored)
```

## License

GPL-2.0-or-later — matches the Kaggle ASL Alphabet dataset's licence,
which is propagated to the trained model artifacts.
