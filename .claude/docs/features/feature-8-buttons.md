# Feature 8 — Buttons

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/ui/buttons.py`
**Hardware:** 2 momentary tactile pushbuttons, each wired GPIO pin → GND.

**Public API:**
```python
class Buttons:
    def __init__(self, on_b1: Callable[[], None], on_b2: Callable[[], None]):
        """Bind callbacks to the two physical buttons. Callbacks fire on
        the gpiozero callback thread — handlers should be quick and
        thread-safe (typically just enqueue an event)."""
```

## Decisions

1. **Library: `gpiozero`.** Pinned under the `[pi]` install profile.
   Provides edge detection, debounce, and callback threading.

2. **Pin assignment** (locked, drives wiring docs):
   - **B1 (main)** → GPIO 17 (header pin 11)
   - **B2 (aux)** → GPIO 27 (header pin 13)
   Adjacent on the header for easy breadboard wiring. Clear of I2C
   (GPIO 2/3) and UART (GPIO 14/15).

3. **Topology.** Each button has one leg on its GPIO pin and one leg on
   GND. `gpiozero.Button(pin, pull_up=True)` enables the Pi's internal
   pull-up resistor — pressed reads LOW. No external resistors.

4. **Hardware debounce.** `bounce_time=0.03` (30 ms) on each Button.
   Filters mechanical bounce typical of cheap tactile switches.

5. **Short-press only.** No long-press detection. The 2-button + audio
   menu UX (feature 9) doesn't need it — B2 short-press is itself the
   language-menu trigger when idle. Simpler, fewer accidental triggers.

6. **Callback thread → event queue.** `gpiozero` invokes callbacks on
   its internal thread. The bound callbacks do nothing more than push
   `ButtonEvent("B1")` or `ButtonEvent("B2")` onto the orchestrator's
   event queue and return. All state-machine logic runs on the main
   thread.

7. **Failure handling.** GPIO init exception (permission, hardware) is
   logged and swallowed; the application continues running without
   button input. Useful when developing on the PC where there is no
   GPIO at all.

8. **PC keyboard fallback** (`--keyboard-buttons` dev flag). On the
   development PC, keys `1` and `2` invoke the same callbacks as B1/B2.
   Used by the phase-3 PC demo. Not enabled in production Pi runs.

9. **Testability.** This module is intentionally thin — the
   interesting logic lives in the orchestrator state machine
   (feature 10) and the audio language menu (feature 9), both of which
   are tested by enqueueing `ButtonEvent` objects directly.

## Out of scope for this feature

- Long-press, double-press, or chorded press detection.
- Hardware debounce circuitry (gpiozero's software debounce is enough).
- Hot-pluggable buttons / dynamic re-binding.
- More than 2 buttons (UX is locked at 2 by `architecture.md` §4).
