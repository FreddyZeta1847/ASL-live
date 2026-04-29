# Feature 6 — Text-to-speech (Piper)

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/tts/speaker.py`
**Voice install path:** `/opt/piper/voices/<lang>.onnx` (+ `.json` config)

**Public API:**
```python
class Speaker:
    def __init__(self):
        """Preload all 5 voices."""
    def speak(self, text: str, lang: str) -> None:
        """Synthesize and play. Blocks until playback finishes."""
```

## Decisions

1. **Library: `piper-tts` (Python package).** Version pinned in
   `pyproject.toml`. One voice per language: 5 voices total in
   `/opt/piper/voices/` (IT, ES, FR, EN, DE). Specific voice models
   chosen by ear during phase-7 provisioning, defaulting to each
   language's medium-quality option.

2. **Preloading.** All 5 voices loaded into memory at `Speaker.__init__()`
   so language switches are instant. Each voice ~50–100 MB; total
   resident is fine on Pi 5 8 GB.

3. **Audio output: ALSA → USB speaker.** ALSA's default device is
   pointed at the USB speaker via `/etc/asound.conf` during phase-7
   provisioning. Piper outputs raw PCM (typically 16-bit, 22 050 Hz);
   `sounddevice.play()` hands that to ALSA.

4. **Backpressure: bounded queue, drop-oldest.** The TTS input queue is
   size 3. If a 4th word arrives while we're still speaking, the oldest
   queued word is dropped. Rationale: keeps audio close to real-time
   when the user signs faster than speech; unbounded queueing would
   produce a growing lag the user can't recover from. The trade-off
   (occasional dropped words) is documented in user-facing docs.

5. **Failure handling.**
   - Missing voice file → log a warning, fall back to **eSpeak-NG** for
     that language only (eSpeak-NG is tiny, ships with Linux, much
     lower quality but always works).
   - Synthesis exception → log, skip that word, do not crash the
     worker.
   - Empty input → no-op.

6. **Worker structure.** Mirror of the Translator worker: blocking loop
   reading `tts_in_queue: (text, lang)`. Each message carries its own
   `lang`, so language changes propagate without restarting the worker
   or sharing state. `speak()` blocks until ALSA finishes playing.

7. **Boot announcement.** On startup, `lang_menu.py` (feature 9) calls
   `speak(<language_name>, current_lang)` so the user hears the active
   language and knows the device is ready. Only "device ready" signal
   in the absence of a screen.

8. **Performance target.** Piper synthesizes ~5–10× real-time on Pi 5
   (~100–200 ms for a 1-second utterance). End-to-end latency from
   SPACE-sign commit to first audio: translation ~300 ms + synthesis
   ~200 ms ≈ 500 ms. Comfortable for the UX.

## Out of scope for this feature

- Voice cloning, custom voices, prosody control.
- Audio file output (no save-to-WAV mode).
- Bluetooth speakers (USB / 3.5 mm only in v1).
- Speech-rate or volume controls (system ALSA mixer if needed).
- Mid-utterance interruption (a new message can't cut off the current
  one — it lands in the bounded queue or gets dropped).
