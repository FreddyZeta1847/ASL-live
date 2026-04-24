# ASL-live — Design Discussion

## How this file works
- One section per chapter. Each has a status: `DECIDED`, `IN PROGRESS`, or `NOT STARTED`.
- Decisions inside a chapter are bulleted, each one standalone.
- Open questions sit under "Still to decide" inside their chapter.
- Cross-chapter constraints go at the top under "Standing constraints" with a date and a list of chapters they apply to.
- Update this file at the end of every design discussion so nothing is lost between sessions.

---

## Standing constraints

### 2026-04-24 — Streaming word-by-word, single command gesture
- No phrase buffer. Each word translates and speaks as soon as SPACE fires.
- Only SPACE is a command gesture. No SEND, no DELETE.
- TTS plays from a FIFO queue — words never interrupt each other; queue can build if the user signs faster than speech plays.
- Accepted tradeoff: single-word translation loses context, agreement, and fluency. Goal is speed, not fluency.
- Applies to: Ch 1 (scope), Ch 4 (classifier classes = 26 letters + SPACE), Ch 6 (state reduces to `selected_language` / `word_buffer` / `is_running`, no phrase buffer), Ch 7 (word-level input), Ch 8 (FIFO queue, no interrupt).

---

## Chapters

### 1. Product scope & framing — NOT STARTED
Scope: what this is, what it is not, success criteria, who uses it, v1 boundaries.

### 2. Hardware & physical setup — NOT STARTED
Scope: Pi model + RAM, accelerator yes/no, camera choice, audio-out path, physical controls (GPIO buttons yes/no), enclosure considerations.

### 3. Software architecture — NOT STARTED
Scope: process model, threading vs asyncio, control-plane pluggability, module layout, concurrency/locking, error-propagation policy.

### 4. Detection pipeline — NOT STARTED
Scope: camera → hand landmarks → normalized features → static classifier → commit gate → optional motion head for J/Z. Classifier classes constrained by Apr 24: 26 letters + SPACE.

### 5. Training & dataset strategy — NOT STARTED
Scope: public datasets vs self-recorded, landmark normalization, classifier family (MLP on landmarks vs CNN on crops), evaluation harness, ONNX export, licensing.

### 6. Command semantics & application state — NOT STARTED
Scope: state machine. Constrained by Apr 24: state is `selected_language`, `word_buffer`, `is_running`. SPACE commits `word_buffer` → translate queue → TTS queue, then clears the buffer.

### 7. Offline translation — NOT STARTED
Scope: Opus-MT vs NLLB-200-distilled, quantization, Pi memory budget, failure modes (unknown word, empty input), caching, licensing. Constrained by Apr 24: word-level input.

### 8. Offline TTS & audio output — NOT STARTED
Scope: Piper voices per language, sample rate, ALSA pipeline, volume control, licensing. Constrained by Apr 24: FIFO queue, no interrupt.

### 9. Phone UI & API contract — NOT STARTED
Scope: REST endpoints and payloads, WebSocket message schema, single HTML page layout, live state feedback, error surfacing.

### 10. Networking & hotspot — NOT STARTED
Scope: Pi AP mode via NetworkManager, SSID/password, DHCP, mDNS (`aslive.local`), captive portal yes/no, phone-disconnect behavior, proof of no internet leakage.

### 11. Dev environment, testing, deployment, ops — NOT STARTED
Scope: PC simulation approach, fake camera/speaker, test pyramid (unit → MP4-replay → live), systemd unit, logs, model hot-swap, failure recovery, update path.

---

## Parking lot
Items raised in one chapter that belong to another. Move each entry to the correct chapter when it's reopened.

- _(empty)_
