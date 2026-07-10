# Tech stack

Every external technology used by ASL-live, with the rationale behind
the choice. Companion to [`architecture.md`](architecture.md) (which
covers *what* the system does) and the
[`features/`](features/) folder (which covers per-feature locked
decisions in depth).

---

## 1. Hardware

| Component | Choice | Rationale |
|---|---|---|
| Compute | **Raspberry Pi 5, 8 GB** | Comfortably runs MediaPipe + MLP + Argos + Piper concurrently. Pi 4 8 GB also viable but tighter on memory headroom. |
| Camera | **USB UVC webcam, 720p @ 30 fps** | Cheap Logitech-class is fine; landmarks don't need high resolution and MediaPipe internally resizes anyway. |
| Speaker | **USB speaker (or 3.5 mm)** | ALSA-default device for Piper to write to. |
| Display | **DFRobot DFR0063 — 16×2 character LCD with PCF8574 I2C backpack** | Plain HD44780 LCD; connects directly to Pi I2C (5V, GND, SDA = GPIO 2, SCL = GPIO 3). No microcontroller, no serial protocol. |
| Buttons | **2× momentary tactile pushbuttons** | One leg → GPIO, one leg → GND. Internal pull-up resistors handle the un-pressed state. |
| Misc | Breadboard, jumpers, microSD ≥ 32 GB | — |

The earlier idea of an **ESP32 with display** was rejected — the DFR0063
is a plain LCD with I2C backpack and connects directly to the Pi.

---

## 2. Recognition stack

### MediaPipe Hands — landmark extraction
- 21 keypoints × 3 coords = 63-dim vector per frame.
- Runs ~30 fps on Pi 5 at `model_complexity = 1`.
- Robust to lighting, skin tone, and background out of the box.
- Replaces the obvious-but-wrong choice of a CNN on raw images: 100×
  smaller model, more robust, trivial dataset.
- Per-feature decisions:
  [`features/feature-1-hand-landmarks.md`](features/feature-1-hand-landmarks.md).

### Small MLP on landmarks — classification
- Architecture: `63 → 128 → 64 → 26` with ReLU + dropout 0.2 + softmax.
- ~30 k parameters / ~120 KB at FP32. Inference < 5 µs per frame.
- Trained on PC in PyTorch, exported to ONNX, loaded on Pi via
  `onnxruntime`.
- **XGBoost** is trained alongside as a sanity-check baseline only —
  not deployed.
- Per-feature decisions:
  [`features/feature-3-classifier.md`](features/feature-3-classifier.md).

### Debounce on prediction stream
- Single counter + blind cooldown turns per-frame predictions into
  discrete commit events (one per intentional sign).
- Per-feature decisions:
  [`features/feature-4-debounce.md`](features/feature-4-debounce.md).

---

## 3. Translation: Argos Translate

- **Offline OpenNMT-based MT**, designed for exactly this case.
- ~100 MB per language pair (EN↔IT, EN↔ES, EN↔FR, EN↔DE).
- Source language is always EN (fingerspelling produces English letters).
- Inference: ~100–500 ms per word on Pi 5.
- Word-by-word translation by user choice (speed > sentence context).

**Why not a small LLM:** a 1–3 B quantized LLM is 10× larger, slower,
and worse-quality than a dedicated MT model for translation. LLMs would
only help if we wanted sentence-level context, which we explicitly
don't.

Per-feature decisions:
[`features/feature-5-translation.md`](features/feature-5-translation.md).

---

## 4. Text-to-speech: Piper

- **Neural TTS, real-time on Pi 5 CPU.**
- Voice models 50–100 MB each; one per language.
- Output via ALSA → USB speaker.
- All 5 voices preloaded at startup so language switches are instant.
- **Fallback:** eSpeak-NG if any language voice is missing.

Per-feature decisions:
[`features/feature-6-tts.md`](features/feature-6-tts.md).

---

## 5. Pi-side peripheral libraries

| Library | Purpose |
|---|---|
| `RPLCD` | HD44780 + PCF8574 I2C wrapper for the LCD |
| `gpiozero` | Button input with edge detection and software debounce |
| `smbus2` | Underlying I2C bus access |
| `sounddevice` | Audio playback to ALSA from Piper PCM output |

---

## 6. Python ecosystem

| Package | Purpose | Profile |
|---|---|---|
| `mediapipe` | Hand landmark extraction | dev + pi |
| `opencv-python` | Camera capture, frame ops | dev + pi |
| `numpy` | Array math | dev + pi |
| `onnxruntime` | MLP inference at runtime | dev + pi |
| `torch` | MLP training only | dev |
| `argostranslate` | Offline MT | dev + pi |
| `piper-tts` | Offline TTS | dev + pi |
| `sounddevice` | Audio playback | dev + pi |
| `RPLCD` | HD44780 over I2C | pi |
| `smbus2` | I2C bus | pi |
| `gpiozero` | Button input | pi |
| `pydantic` | Config validation | dev + pi |
| `pytest` | Tests | dev |

Two install profiles in `pyproject.toml`:
- `[dev]` — PC: trains the model, runs the demo without GPIO.
- `[pi]` — Pi: runs the device; no `torch`, adds GPIO/LCD libs.

---

## 7. System-level dependencies (Pi)

- `i2c-tools`, kernel I2C enabled via `raspi-config`.
- ALSA configured to default to USB speaker (`/etc/asound.conf`).
- Argos language packs preloaded by `scripts/setup_argos.py`.
- Piper voice files preinstalled at `/opt/piper/voices/`.

---

## 8. Build and packaging

- **`setuptools` + `pip`** (PEP 621 in `pyproject.toml`). No `poetry`,
  no `uv` — keeps the setup minimal.
- Python ≥ 3.11 (modern type syntax, MediaPipe support).
- Auto-start on Pi via a `systemd` unit (`asl-live.service`),
  installed in phase 7.

---

## 9. Export format: ONNX (opset 17)

ONNX is the **file format** in which the trained MLP travels from PC to
Pi. It is *not* a model — it's a portable container, like JPEG for
images. The same trained MLP could be saved as `.pt` (PyTorch) or
`.tflite` (TensorFlow Lite); we pick `.onnx` because the runtime
(`onnxruntime`) is a tiny Python package with no other dependencies.

Picked over TFLite because:
- Better Python ecosystem (numpy interop, debugging tools).
- TFLite is tuned for mobile / accelerator hardware (Coral NPU, etc.) —
  irrelevant on plain Pi 5 CPU.

Quantization (FP32 → INT8) is **skipped for v1**: the model is already
~120 KB, the Pi has 8 GB RAM, and inference latency is dwarfed by
MediaPipe. Revisit only if profiling demands it.
