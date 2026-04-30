# ASL-live

Offline ASL alphabet → spoken translation device on Raspberry Pi 5.
A user signs ASL letters into a USB camera; the Pi recognizes them, builds words,
translates each completed word into the chosen language (IT / ES / FR / EN / DE),
and speaks it through a USB speaker. Fully phone-free and internet-free.

The full design lives under [`.claude/docs/`](.claude/docs/) (architecture,
tech stack, per-feature decisions). The implementation roadmap and per-phase
plans live under [`.claude/plans/`](.claude/plans/).

## Status

Phase 1 — data collection pipeline. See
[`.claude/plans/PLAN.md`](.claude/plans/PLAN.md) for the full 7-phase
roadmap.

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
python -m asl_live.capture.collect --class SPACE  --count 500
python -m asl_live.capture.collect --class DELETE --count 500
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

## Project layout

```
src/asl_live/        Application source (recognition, capture, …)
tests/               Unit tests (pytest)
scripts/             One-shot CLIs (Kaggle ingest, future setup helpers)
data/                Generated datasets (gitignored)
models/              Trained model artifacts (gitignored)
.claude/             Project knowledge — design docs, plans, agents
```

## License

GPL-2.0-or-later — matches the Kaggle ASL Alphabet dataset's licence,
which is propagated to the trained model artifacts.
