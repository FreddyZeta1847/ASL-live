# Feature 7 — LCD display

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/ui/lcd.py`
**Hardware:** DFRobot DFR0063 — 16×2 character LCD with PCF8574 I2C
backpack, on Pi I2C bus 1 (GPIO 2 = SDA, GPIO 3 = SCL).

**Public API:**
```python
class LCDWriter:
    def update(self, word: str, lang: str, status: str) -> None:
        """Render line 1 = word (last 16 chars), line 2 = `<lang>|<status>`.
        Diffs against the previous content and only writes changed cells."""
```

## Decisions

1. **Library: `RPLCD`** (HD44780 + PCF8574 wrapper). Pinned in
   `pyproject.toml` under the `[pi]` install profile — does not install
   on the dev PC.

2. **I2C address auto-detect.** DFR0063 typically replies on `0x27`,
   some clones on `0x3F`. On startup probe both; use whichever
   responds. If neither responds, log an error and continue running
   without a screen (degraded mode, not a crash).

3. **Cell-diffing update strategy.** `LCDWriter` keeps a 32-character
   shadow of the previous render and only writes cells whose content
   has changed. Eliminates the flicker that comes from clear-and-rewrite.

4. **Single update entrypoint.** `update(word, lang, status)` is the
   only public method. No separate line-writes, no flush/commit calls
   — the diff lives inside the writer.

5. **Word truncation.** Words longer than 16 chars: show the **last**
   16 (most useful while typing). When the word commits and the buffer
   clears, line 1 clears.

6. **Line-2 format:** `<LANG>|<STATUS>` left-justified, padded to 16
   chars. Status codes:
   - `IDLE` — device on, not capturing.
   - `REC ` — capturing letters.
   - `TX  ` — translating.
   - `TTS ` — speaking.
   - `LANG` — language menu open (line 1 shows the candidate language
     name in this state).

7. **Backlight always on.** ~0.05 W draw is negligible; a dark display
   reads as "device broken." Toggleable via `RPLCD` if a future power
   mode demands it.

8. **Update source.** A small thread in the main process subscribes to
   recognizer commit events and lifecycle state changes and calls
   `update()`. Not run from the recognizer worker (kept pure / fast).
   Updates fire on state change only, never per-frame.

9. **Boot splash.** First render after boot: line 1 = `ASL-live`,
   line 2 = `loading...`, while Piper / Argos / camera initialize.
   Replaced by `<lang>|IDLE` once startup is complete.

10. **Failure handling.** Mid-session I2C write failures (loose wire,
    flaky cable) are logged and swallowed. The next `update()` call
    re-attempts. The system keeps running without the screen; the user
    just loses visual confirmation.

11. **Testability split.** Pure formatter
    `format_lines(word, lang, status) -> tuple[str, str]` is
    unit-tested without hardware (truncation, padding, status-code
    formatting). The I2C writer has only a smoke test on the Pi.

## Out of scope for this feature

- Custom 5×8 glyphs / icons (RPLCD supports them — not needed in v1).
- Scrolling long words across line 1 (we show the last 16 chars
  instead).
- Brightness control / dimming (no PWM on the backlight pin via
  PCF8574).
- Multi-line wrap of the word across lines 1 and 2 (line 2 is
  status-only).
