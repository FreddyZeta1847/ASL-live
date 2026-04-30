# ASL-live — Implementation Plan

Detailed, phase-by-phase plan to build the offline ASL → speech device described
in [`.claude/docs/architecture.md`](../docs/architecture.md) and
[`.claude/docs/tech-stack.md`](../docs/tech-stack.md). Each phase has concrete
deliverables, acceptance criteria, and a clear definition of done so we can
demo at every checkpoint.

---

## 0. Repository structure (target, annotated by phase)

This is the *end-state* layout. ✅ marks files that already exist;
otherwise the marker shows the phase that creates the file.
Subpackage comments describe the role each module plays in the system.
For the *current* on-disk layout (which lags behind this target), see
[`tree.md`](../../tree.md) at the repo root.

```
ASL-live/
├── src/asl_live/
│   ├── __init__.py                      ✅ phase 1
│   ├── config.py                        ✅ phase 1   paths, classes, thresholds, GPIO pins
│   │
│   ├── recognition/                     "what does this gesture look like?"
│   │   ├── landmarks.py                 ✅ phase 1   MediaPipe wrapper + normalization helpers
│   │   ├── classifier.py                   phase 2   loads mlp.onnx, predict(landmarks) -> (label, conf)
│   │   └── debounce.py                     phase 3   prediction stream -> commit events
│   │
│   ├── capture/                         interactive data collection
│   │   └── collect.py                   ✅ phase 1   webcam UI, SPACE/DELETE samples
│   │
│   ├── train/                              phase 2   PC-only training stack
│   │   └── train_mlp.py                              loads .npy -> MLP + XGBoost baseline -> mlp.onnx
│   │
│   ├── translation/                        phase 5   offline MT
│   │   └── translator.py                             Argos wrapper with LRU cache + identity short-circuit
│   │
│   ├── tts/                                phase 5   offline speech synthesis
│   │   └── speaker.py                                Piper wrapper with bounded drop-oldest queue
│   │
│   ├── ui/                              peripheral drivers
│   │   ├── lcd.py                          phase 4   DFR0063 16x2 LCD over I2C, cell-diff render
│   │   ├── buttons.py                      phase 6   gpiozero buttons -> event queue
│   │   └── lang_menu.py                    phase 6   audio-only language menu + persistence
│   │
│   └── pipeline/                           phase 5+  full orchestrator
│       └── main.py                                   workers, queues, lifecycle FSM, signal handlers
│
├── scripts/                             one-shot CLIs (not part of the runtime)
│   ├── ingest_public.py                 ✅ phase 1   Kaggle ASL Alphabet -> landmark .npy
│   ├── demo_recognition.py                 phase 3   PC live demo, no peripherals
│   ├── setup_argos.py                      phase 5   install Argos packs offline at provisioning time
│   └── asl-live.service                    phase 7   systemd unit for boot auto-start
│
├── tests/                               pytest, PC-only, no hardware
│   ├── test_landmarks.py                ✅ phase 1   normalization helpers
│   ├── test_debounce.py                    phase 3   debounce state machine
│   └── test_lang_menu.py                   phase 6   menu state machine + atomic persistence
│
├── data/                                runtime datasets (gitignored)
│   └── landmarks/<class>/*.npy                       e.g. kaggle_000123.npy, custom_000045_m.npy
│
├── models/                              training artifacts (gitignored)
│   ├── mlp.onnx                            phase 2   the deployable model
│   ├── label_map.json                      phase 2   {0: "A", ..., 25: "DELETE"}
│   └── training_report.json                phase 2   hyperparams, metrics, confusion matrix, git SHA
│
├── .claude/                             project knowledge (see CLAUDE.md)
│   ├── CLAUDE.md                                     project entry point
│   ├── agents/                                       project-local subagent definitions
│   ├── docs/
│   │   ├── architecture.md                           system design (this file's companion)
│   │   ├── tech-stack.md                             chosen technologies + rationale
│   │   ├── features/feature-N-<name>.md              per-feature locked decisions
│   │   └── decisions/                                ADRs (cross-cutting choices)
│   └── plans/
│       ├── PLAN.md                                   this file — phase roadmap
│       ├── plan_zip.md                               condensed index of plan files
│       ├── current-task.md                           pointer at the active task
│       └── phases/phase-N-<name>.md                  per-phase implementation plan
│
├── pyproject.toml                       ✅ phase 1   PEP 621 + [dev] / [pi] install profiles
├── .gitignore                           ✅ phase 1
├── README.md                            ✅ phase 1   project overview + collection protocol
└── tree.md                              ✅ phase 1   current-state filesystem map (regenerated on changes)
```

**Two install profiles** in `pyproject.toml`:
- `[dev]` — PC: mediapipe, opencv, torch, onnxruntime, argos-translate,
  piper-tts, scikit-learn, xgboost, pytest.
- `[pi]` — Pi: same base minus torch (training is PC-only) plus RPLCD,
  gpiozero, smbus2.

---

## 1. Dependencies

### System (Pi)
- `i2c-tools`, kernel I2C enabled via `raspi-config`
- ALSA configured to default to USB speaker
- Argos language packs preloaded (`en→it`, `en→es`, `en→fr`, `en→de`)
- Piper voices in `/opt/piper/voices/` (one per language)

### Python (single `pyproject.toml`, optional groups)
| Package | Purpose | Profile |
|---|---|---|
| `mediapipe` | Hand landmark extraction | dev + pi |
| `opencv-python` | Camera capture, frame ops | dev + pi |
| `numpy` | Array math | dev + pi |
| `onnxruntime` | Inference at runtime | dev + pi |
| `torch` | Training only | dev |
| `argostranslate` | Offline MT | dev + pi |
| `piper-tts` | Offline TTS | dev + pi |
| `sounddevice` | Audio playback for menu prompts | dev + pi |
| `RPLCD` | HD44780 over I2C | pi |
| `smbus2` | I2C bus | pi |
| `gpiozero` | Button input | pi |
| `pydantic` | Config validation | dev + pi |
| `pytest` | Tests | dev |

---

## 2. Phase-by-phase plan

### Phase 1 — Data collection script (PC)

**Goal:** capture a clean labeled dataset of landmark vectors.

**Tasks**
1. Implement `src/asl_live/recognition/landmarks.py`:
   - Wrap MediaPipe Hands (single hand, model_complexity=1).
   - `extract(frame) -> Optional[np.ndarray]` returns 63-dim vector (21×3).
   - Normalize: subtract wrist (landmark 0), scale so max distance from wrist = 1.
2. Implement `src/asl_live/capture/collect.py`:
   - CLI: `python -m asl_live.capture.collect --class A --count 200`.
   - Show camera feed, overlay current class + sample counter.
   - Capture only when a hand is detected and stable for 3 frames.
   - Save per-class to `data/landmarks/<class>/<timestamp>.npy`.
   - Mirror-augment (flip left↔right hand) at save time = free 2× data.
3. Document collection protocol in `README.md` (lighting, distance, angle variation).

**Deliverables**
- `landmarks.py`, `collect.py`, dataset directory.

**Acceptance**
- 200 samples collected for at least 3 classes (A, B, SPACE) in a quick smoke test.
- `.npy` files load back to shape `(63,)` and reproduce visually via a debug viewer.

---

### Phase 2 — Train MLP classifier (PC)

**Goal:** trained, exported model with measured accuracy.

**Tasks**
1. Implement `src/asl_live/train/train_mlp.py`:
   - Load all `data/landmarks/**/*.npy`, infer class from folder name.
   - Stratified 80/10/10 train/val/test split.
   - Architecture: `Linear(63,128) → ReLU → Dropout(0.2) → Linear(128,64) → ReLU → Linear(64,26)`.
   - Adam, cross-entropy, early stopping on val loss.
   - Export to `models/mlp.onnx` (opset 17) + `models/label_map.json`.
   - Print confusion matrix (focus on A vs DELETE).
2. Implement `src/asl_live/recognition/classifier.py`:
   - Load ONNX, single-frame predict returning `(label, confidence)`.

**Deliverables**
- `train_mlp.py`, `classifier.py`, `mlp.onnx`, `label_map.json`, training report (printed).

**Acceptance**
- Test-set accuracy ≥ 95 % on collected data.
- DELETE not confused with A more than 2 % of the time. If it is, switch DELETE
  to the pinch gesture (decision deferred to here per architecture.md §8).
- ONNX inference < 5 ms per frame on dev machine.

---

### Phase 3 — Live recognition demo (PC, then port to Pi)

**Goal:** see letters appear in a terminal in real time, with debounce.

**Tasks**
1. Implement `src/asl_live/recognition/debounce.py`:
   - State machine: only commit a class after `STABLE_FRAMES=5` consecutive
     identical predictions with confidence ≥ `MIN_CONF=0.85`.
   - Require a `GAP_FRAMES=3` window of "no-hand" or different class before
     the next commit, so a held sign produces exactly one letter.
   - Pure function over a frame-prediction stream — fully unit-testable.
2. Implement `scripts/demo_recognition.py`:
   - Camera → landmarks → classifier → debounce → print.
   - Show overlay: current top-1 class + confidence + buffer string.
   - Q to quit.
3. Port + run on the Pi to confirm frame rate ≥ 15 fps.

**Deliverables**
- `debounce.py`, `demo_recognition.py`, `tests/test_debounce.py`.

**Acceptance**
- Signing "HELLO" produces exactly the buffer `HELLO` on first try.
- DELETE removes the last letter, SPACE prints the buffer and clears it.
- ≥15 fps on Pi 5, ≥25 fps on dev PC.
- Unit tests cover: single sign held → 1 commit; sign change with gap → 2 commits;
  low-confidence frames → 0 commits.

---

### Phase 4 — Pi peripherals: LCD on I2C

**Goal:** current word visible on the DFR0063 LCD in real time.

**Tasks**
1. Enable I2C on Pi, confirm `i2cdetect -y 1` shows the LCD address (0x27 or 0x3F).
2. Implement `src/asl_live/ui/lcd.py`:
   - `LCDWriter` class wrapping `RPLCD.i2c.CharLCD`.
   - `update(word: str, lang: str, status: str)` — formats both lines; writes
     only changed cells (avoid flicker).
   - Truncate word to last 16 chars when longer.
3. Wire LCD updates into the demo: every debounced commit → `lcd.update(...)`.

**Deliverables**
- `lcd.py`, updated `demo_recognition.py` accepting `--lcd` flag.

**Acceptance**
- Live signing on Pi: word appears character-by-character on LCD line 1.
- Status on line 2 cycles `IDLE` ↔ `REC` correctly.
- No visible flicker.

---

### Phase 5 — Translation + TTS workers

**Goal:** signing a word + SPACE produces spoken output in the chosen language.

**Tasks**
1. Implement `src/asl_live/translation/translator.py`:
   - Wrap `argostranslate.translate.translate(text, "en", target)`.
   - Lazy-load only the active pair on language switch.
2. Implement `src/asl_live/tts/speaker.py`:
   - Wrap Piper: load voice for active language, synthesize → 16-bit PCM.
   - Play through `sounddevice` (blocking is fine; runs in TTS worker process).
3. Implement `src/asl_live/pipeline/main.py`:
   - Three processes: `recognizer`, `translator`, `speaker`.
   - Two `multiprocessing.Queue`s: word→translate, translation→speak.
   - LCD updates from main process via a thread reading recognizer events.

**Deliverables**
- `translator.py`, `speaker.py`, `main.py`.

**Acceptance**
- Sign `H E L L O` + SPACE → LCD shows `TX` then `TTS`, speaker says "Ciao" (in IT).
- Latency from SPACE to first audio < 1.5 s on Pi 5.
- Recognition keeps committing letters during translation/TTS (no UI freeze).

---

### Phase 6 — Buttons + audio language menu

**Goal:** full UX with no PC tether — boot, switch language, sign, listen, repeat.

**Tasks**
1. Wire 2 momentary buttons to GPIO with internal pull-ups. Document the pin
   choice in `config.py` (e.g., GPIO 17 main, GPIO 27 aux).
2. Implement `src/asl_live/ui/buttons.py`:
   - `gpiozero.Button` with `when_pressed` callbacks.
   - 30 ms software debounce.
3. Implement `src/asl_live/ui/lang_menu.py`:
   - State machine: idle → menu → confirm-or-cancel.
   - Each B2 press cycles `[IT, ES, FR, EN, DE]`, Piper announces the name.
   - 3 s no-press → save selection to `~/.aslive/config.json`, Piper says
     "OK <language>".
   - B1 → cancel, restore previous.
4. Hook buttons into `pipeline/main.py`:
   - B1 idle → start capture; B1 capturing → stop.
   - B2 idle → enter language menu; B2 capturing → force-send current word.
5. On boot: load config, Piper announces current language.

**Deliverables**
- `buttons.py`, `lang_menu.py`, updated `main.py`, `config.py`,
  `tests/test_lang_menu.py`.

**Acceptance**
- Cold boot → Piper announces current language → press B1 → sign word + SPACE
  → translation spoken. No keyboard or screen interaction.
- Language menu cycles all 5 languages, confirms on timeout, persists across
  reboot.
- B1 cancel from menu correctly restores previous language.

---

### Phase 7 — Polish

**Goal:** make it feel like a product (within prototype scope).

**Tasks**
1. Tune debounce thresholds with real-world usage data (capture frame logs,
   adjust `STABLE_FRAMES`, `MIN_CONF`).
2. Add a `--verbose` mode that logs every classifier prediction to a CSV for
   later analysis.
3. Final BOM with part numbers and prices.
4. Wiring diagram (PNG or Fritzing) committed to repo.
5. Auto-start on Pi boot via systemd unit `asl-live.service`.

**Deliverables**
- `scripts/asl-live.service`, `docs/wiring.png`, updated `README.md`.

**Acceptance**
- Unplug Pi, plug back in, no keyboard/SSH needed → device announces language
  and is ready to capture.

---

## 3. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| MediaPipe Hands fps drops on Pi 5 | Med | Reduce camera res to 480p; lower `model_complexity` to 0. |
| Letter classes confuse easily (M/N/T) | Med | Collect more samples for confused pairs; consider relative-distance features in addition to raw landmarks. |
| Argos pair quality is poor for short single words | Med | Acceptable per user decision (speed > context); document limitation. |
| Piper voice for one language is weak | Low | eSpeak-NG fallback for that one language. |
| I2C address conflicts (LCD vs other peripherals) | Low | Only LCD on I2C in v1; address documented. |
| GPIO debounce causes double-presses | Low | gpiozero debounce + 30 ms software guard. |
| User's hand frame is inconsistent in real use | Med | Phase 7: optional alignment guides on LCD; collect more "in the wild" samples. |

---

## 4. Milestones (demoable)

| # | Milestone | What's demoable |
|---|---|---|
| M1 | End of Phase 2 | Trained model + confusion matrix; can predict letters from saved frames. |
| M2 | End of Phase 3 | Live signing on PC, terminal shows correctly debounced word. |
| M3 | End of Phase 4 | Same as M2 but on Pi with LCD output. |
| M4 | End of Phase 5 | Full PC-tethered demo: sign → translate → speak. |
| M5 | End of Phase 6 | Standalone Pi: boot, button, sign, speak — no PC. |
| M6 | End of Phase 7 | Auto-start, polished, BOM + wiring diagram complete. |

---

## 5. Out of scope (explicit)

- J / Z motion-based letters.
- Sentence-level translation context.
- Multi-hand or two-handed signs.
- Battery management UX (level, sleep, wake).
- Touchscreen UI of any kind.
- Any networked feature (sync, OTA updates, telemetry).
