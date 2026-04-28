# ASL-live — Design Discussion

Offline ASL alphabet → spoken translation device. Phone-free, internet-free.
A mute user signs letters into a USB camera; the Pi recognizes them, builds words,
translates each completed word into the user's chosen language, and speaks it through
a USB speaker.

This document captures the agreed design from the initial discussion and is the
source of truth that will drive implementation in `.claude/plans/`.

---

## 1. Goals & non-goals

**Goals**
- Fully offline: no internet, no phone, no cloud services.
- Real-time letter recognition (≥15 fps) on a Raspberry Pi.
- Top-5 European languages: IT, ES, FR, EN, DE.
- Word-by-word translation (speed > sentence-level context, by user choice).
- Visual confirmation of the current word before sending (LCD).
- Inline error correction (delete-last-letter sign).

**Non-goals (v1)**
- Motion-based ASL letters (J, Z) — skipped.
- Continuous-sentence ASL — alphabet fingerspelling only.
- Sentence-level context for translation.
- Battery-management UX (prototype runs from any USB power bank).
- Touchscreen UI.

---

## 2. Hardware

| Component | Choice | Notes |
|---|---|---|
| Compute | Raspberry Pi 5, 8 GB | Pi 4 8 GB also viable but tighter on Piper + classifier. |
| Camera | USB UVC webcam, 720p @ 30 fps | Cheap Logitech-class is fine; landmarks don't need high res. |
| Speaker | USB speaker (or 3.5 mm) | Whatever Piper can output to via ALSA. |
| Display | DFRobot DFR0063 — I2C 16×2 character LCD | Connects to Pi I2C: 5V, GND, SDA (GPIO 2), SCL (GPIO 3). |
| Buttons | 2× momentary pushbuttons | GPIO with internal pull-ups. |
| Misc | Breadboard, jumpers, microSD ≥ 32 GB | — |

The earlier idea of an **ESP32 with display** is dropped — the DFR0063 is a plain
HD44780 LCD with PCF8574 I2C backpack and connects directly to the Pi. No second
microcontroller, no serial protocol.

---

## 3. Gesture vocabulary

24 ASL alphabet letters (A–Y excluding J, Z) + 2 control gestures = **26 classes**.

| Gesture | Meaning |
|---|---|
| A–Y (no J, Z) | Letter |
| Open palm, 5 fingers spread | **SPACE** — commit current word, translate, speak |
| Thumb-down (fist with thumb pointing down) | **DELETE** — remove last letter from buffer |

Selection rationale: both control gestures are visually distinct from any letter
shape so the landmark classifier won't confuse them. If thumb-down turns out to
collide with letter "A" (closed fist with thumb on side), fall back to a pinch
gesture (thumb tip + index tip touching).

---

## 4. Recognition pipeline

**Stage 1 — Hand landmarks: MediaPipe Hands**
- 21 keypoints × 3 coords = 63-dim vector per frame.
- Runs ~30 fps on Pi 5.
- Robust to lighting, skin tone, background.

**Stage 2 — Classifier: small MLP on landmarks**
- Input: normalized 63-dim vector (translate + scale-normalize relative to wrist).
- Architecture: 63 → 128 → 64 → 26 (dense, ReLU, softmax). Few hundred KB.
- Trained on PC, exported to ONNX or kept as PyTorch/TFLite.
- Inference: <5 ms per frame.
- Trained on a self-collected dataset (~200 samples per class) plus public ASL
  alphabet datasets where applicable.

**Stage 3 — Debounce / commit logic**
- A class only "commits" when the same prediction is stable for ~5 frames.
- Then a no-hand or different-class gap is required before the next commit, so
  a single held sign produces exactly one letter.

**Why landmarks instead of CNN-on-images:** 100× smaller model, more robust,
trivial dataset collection, and runs in real time on Pi without an accelerator.

---

## 5. Translation: Argos Translate (not an LLM)

- Offline OpenNMT-based MT, designed exactly for this case.
- ~100 MB per language pair (EN↔IT, EN↔ES, EN↔FR, EN↔DE).
- Source language is always EN (since fingerspelling produces English letters).
- Inference: ~100–500 ms per word on Pi 5.
- Word-by-word translation as agreed (speed prioritized over sentence context).

**Why not a small LLM:** a 1–3 B quantized LLM is 10× larger, slower, and
worse-quality than a dedicated MT model for translation. LLMs would only help if
we wanted sentence-level context, which we explicitly don't.

---

## 6. TTS: Piper

- Real-time neural TTS, runs comfortably on Pi 5 CPU.
- Voice models 50–100 MB each; one per language.
- Output via ALSA → USB speaker.

**Fallback:** eSpeak-NG if any language voice is missing or quality is acceptable.

---

## 7. Process architecture

Single Python application, three workers connected by `multiprocessing.Queue`s,
so recognition stays at full frame rate while translation/TTS happen in
parallel.

```
[Camera capture]
       │
       ▼
[Recognizer worker]                     ┌──── current_word ───▶ [LCD writer]
  MediaPipe → MLP → debounce            │
  on letter:  buffer += letter ─────────┤
  on DELETE:  buffer = buffer[:-1] ─────┤
  on SPACE:   send buffer to ───────────┴───▶ [Translator worker]
               translation_queue                 Argos EN→target
               buffer = ""                        │
                                                  ▼
                                          [TTS worker]
                                            Piper → ALSA
```

A 4th lightweight thread services button events and the audio language menu.

---

## 8. UX: 2 buttons + audio menu

| State | Button 1 (main, large) | Button 2 (aux, small) |
|---|---|---|
| **Idle** | Start capture session | Enter / cycle language menu |
| **Capturing** | Stop, return to idle | Force-send current word (manual SPACE) |
| **Lang menu** | Cancel, keep previous selection | Next language |

**Language menu flow**
1. B2 in idle → enter menu, Piper says current language ("Italiano").
2. Each B2 press → next language: IT → ES → FR → EN → DE → IT.
3. 3 seconds without a press → confirm, Piper says "OK [language]", save to
   `~/.aslive/config.json`, return to idle.
4. B1 at any time → cancel, restore previous language, return to idle.

On boot, Piper announces the persisted language so the user knows the state
without a screen.

**Why force-send on B2 in capturing:** if the SPACE gesture mis-fires or the
classifier gets stuck, the user has a deterministic out without breaking flow.

---

## 9. LCD layout (16×2)

```
┌────────────────┐
│HELLO_          │   ← line 1: current word buffer (last 16 chars)
│IT|REC          │   ← line 2: lang code | status
└────────────────┘
```

Statuses on line 2:
- `IDLE` — device on, not capturing
- `REC`  — capturing letters
- `TX`   — translating
- `TTS`  — speaking
- `LANG` — in language menu (line 1 shows current candidate)

If the word exceeds 16 chars, show the last 16 (most useful while typing).
Updates are written via `RPLCD` (Python lib for HD44780 + PCF8574).

---

## 10. Power & deployment

- Prototype: powered by any USB-C power bank (5 V / 3 A recommended for Pi 5).
- Pi 5 averages ~5 W under this workload; a 10 000 mAh pack gives roughly
  3–4 hours of operation. Sufficient for prototype demos.
- No power-management UX (no battery-level UI, no sleep) in v1.

---

## 11. Implementation roadmap

Each phase ends with a working, demoable artifact. Phases 1–3 happen mostly on
the development PC; phase 4 onward involves the Pi hardware.

1. **Data collection script** (PC + USB camera)
   - Capture landmark vectors per class with on-screen prompts.
   - Output: CSV/Numpy dataset, ~200 samples × 26 classes.

2. **Train MLP classifier** (PC)
   - Train, validate, export to ONNX/TFLite.
   - Confusion-matrix sanity check (especially A vs DELETE-thumb-down).

3. **Live recognition demo** (PC, then port to Pi)
   - Camera → MediaPipe → classifier → debounced letter stream printed to
     terminal.
   - This validates that v1 recognition is good enough before any peripherals.

4. **Pi integration**
   - Run the demo on the Pi, confirm frame rate.
   - Wire the LCD on I2C, push current word to it.

5. **Translation + TTS workers**
   - Argos Translate (one pair at a time loaded from config).
   - Piper TTS playback over USB speaker.
   - Wire the worker queues.

6. **Buttons + language menu**
   - GPIO debounce, state machine, audio prompts.
   - Persist selection to `~/.aslive/config.json`.

7. **Polish**
   - Tune debounce thresholds.
   - Final BOM + wiring diagram.

---

## 12. Open / deferred items

- **Camera mounting / framing.** The user must hold their hand in a roughly
  consistent location. Could add a visual frame on the LCD or a target
  outline — defer until we see how recognition feels live.
- **Self-collected dataset only vs. mixing public datasets.** Decide after
  phase 2 evaluation; user-only data tends to overfit one hand.
- **DELETE = thumb-down vs. pinch.** Pick after seeing classifier confusion
  matrix.
- **J / Z support.** Out of scope for v1; revisit if users miss them.
- **Sentence-level mode.** Could add a "period" sign that buffers multiple
  words and translates with full context. Out of scope by user decision (speed
  preferred over context).
