# Architecture

System-level design for ASL-live. Companion to
[`tech-stack.md`](tech-stack.md) (which covers *which* libraries and
hardware we picked) and the
[`features/`](features/) folder (which covers the locked decisions per
sub-feature). Implementation is sequenced by
[`../plans/PLAN.md`](../plans/PLAN.md).

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

## 2. Gesture vocabulary

24 ASL alphabet letters (A–Y excluding J, Z) + 2 control gestures = **26 classes**.

| Gesture | Meaning |
|---|---|
| A–Y (no J, Z) | Letter |
| Open palm, 5 fingers spread | **SPACE** — commit current word, translate, speak |
| Thumb-down (fist with thumb pointing down) | **DELETE** — remove last letter from buffer |

Selection rationale: both control gestures are visually distinct from any
letter shape so the landmark classifier won't confuse them. If thumb-down
turns out to collide with letter "A" (closed fist with thumb on side),
fall back to a pinch gesture (thumb tip + index tip touching).

---

## 3. Process architecture

Single Python application, three workers connected by
`multiprocessing.Queue`s, so recognition stays at full frame rate while
translation and TTS happen in parallel.

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

A 4th lightweight thread services button events and the audio language
menu. Detailed queue shapes, bounds, and overflow policies live in
[`features/feature-10-orchestration.md`](features/feature-10-orchestration.md).

---

## 4. UX: 2 buttons + audio menu

| State | Button 1 (main, large) | Button 2 (aux, small) |
|---|---|---|
| **Idle** | Start capture session | Enter / cycle language menu |
| **Capturing** | Stop, return to idle | Force-send current word (manual SPACE) |
| **Lang menu** | Cancel, keep previous selection | Next language |

**Language menu flow**
1. B2 in idle → enter menu, Piper says current language ("Italiano").
2. Each B2 press → next language: IT → ES → FR → EN → DE → IT.
3. 3 seconds without a press → confirm, Piper says "OK [language]",
   save to `~/.aslive/config.json`, return to idle.
4. B1 at any time → cancel, restore previous language, return to idle.

On boot, Piper announces the persisted language so the user knows the
state without a screen.

**Why force-send on B2 in capturing:** if the SPACE gesture mis-fires
or the classifier gets stuck, the user has a deterministic out without
breaking flow.

---

## 5. LCD layout (16×2)

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

Words longer than 16 chars show the last 16 (most useful while typing).
Implementation details (cell-diffing render, status formatting, failure
handling) live in
[`features/feature-7-lcd.md`](features/feature-7-lcd.md).

---

## 6. Power & deployment

- Prototype: powered by any USB-C power bank (5 V / 3 A recommended for
  Pi 5).
- Pi 5 averages ~5 W under this workload; a 10 000 mAh pack gives
  roughly 3–4 hours of operation. Sufficient for prototype demos.
- No power-management UX (no battery-level UI, no sleep) in v1.

---

## 7. Implementation roadmap (summary)

Each phase ends with a working, demoable artifact. Detailed phase plans
live in [`../plans/phases/`](../plans/phases/).

1. **Data collection** (PC) — landmark dataset from public ingest +
   custom SPACE/DELETE collection.
2. **Train MLP classifier** (PC) — train, validate, export to ONNX.
3. **Live recognition demo** (PC, then port to Pi).
4. **Pi peripherals: LCD on I2C.**
5. **Translation + TTS workers.**
6. **Buttons + audio language menu.**
7. **Polish** — auto-start, BOM, wiring diagram, threshold tuning.

---

## 8. Open / deferred items

- **Camera mounting / framing.** The user must hold their hand in a
  roughly consistent location. Could add a visual frame on the LCD or a
  target outline — defer until we see how recognition feels live.
- **Self-collected dataset only vs. mixing public datasets.** Decide
  after phase 2 evaluation; user-only data tends to overfit one hand.
  (Default per `feature-2-data-collection.md`: mix public + custom.)
- **DELETE = thumb-down vs. pinch.** Pick after seeing the classifier's
  confusion matrix in phase 2.
- **J / Z support.** Out of scope for v1; revisit if users miss them.
- **Sentence-level mode.** Could add a "period" sign that buffers
  multiple words and translates with full context. Out of scope by
  user decision (speed preferred over context).
