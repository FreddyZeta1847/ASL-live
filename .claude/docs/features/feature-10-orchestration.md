# Feature 10 — Process orchestration

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/pipeline/main.py`
**Entry point:** `python -m asl_live.pipeline.main` (dev) or
`asl-live.service` systemd unit (production).

## Decisions

1. **4 processes total.**
   - **Main** — orchestrator state machine, LCD update thread, menu
     `tick()` thread, signal handlers.
   - **Recognizer** — camera capture, MediaPipe, classifier, debouncer.
   - **Translator** — Argos.
   - **TTS** — Piper synthesis + ALSA playback.

   `multiprocessing.Process` for the three workers. CPU-heavy work
   isolated from the GIL.

2. **Queues** (all `multiprocessing.Queue` except where noted):

   | Queue | Direction | Payload | Bound | Overflow |
   |---|---|---|---|---|
   | `recognizer_out_queue` | recognizer → main | `CommitEvent` | 16 | block (events are rare) |
   | `translation_in_queue` | main → translator | `(word, target_lang)` | 16 | block |
   | `tts_in_queue` | translator → tts | `(text, lang)` | **3** | **drop-oldest** (per feature 6) |
   | `button_event_queue` | gpiozero thread → main | `ButtonEvent` | 16 | block |

   The button queue is `queue.Queue` (in-process) since gpiozero runs
   in the same process as main.

3. **Lifecycle state machine** (main thread):
   - **IDLE** — device on, not capturing. B1 → CAPTURING. B2 → open
     language menu (state becomes MENU until menu closes).
   - **CAPTURING** — recognizer active. B1 → IDLE (cancels current
     word). B2 → force-send current word, remain CAPTURING.
   - **MENU** — `LanguageMenu.is_open`. Button events routed to the
     menu; recognizer events ignored.

4. **Recognizer start/stop.** A `multiprocessing.Event` named
   `capture_enabled` gates the recognizer's main loop. Main sets it on
   IDLE→CAPTURING and clears it on CAPTURING→IDLE. While cleared, the
   recognizer waits on the Event — no MediaPipe work, no CPU burn.

5. **Word buffer lives in main, not the recognizer.**
   - On `LETTER` event → append to buffer, refresh LCD.
   - On `DELETE` event → pop last char (no-op on empty), refresh LCD.
   - On `SPACE` event → if buffer non-empty, push to
     `translation_in_queue`, clear buffer, set LCD status to `TX`;
     empty buffer → silent no-op.
   The recognizer just emits committed gestures; UX semantics are main's
   concern.

6. **Logging.** `logging` to a rotating file:
   - `/var/log/asl-live/app.log` when running under systemd.
   - `~/.aslive/logs/app.log` otherwise.
   Level INFO by default, DEBUG with `--verbose`. Each worker tags log
   records with its process name.

7. **Crash policy: log, don't auto-restart (v1).** Main checks each
   worker's `is_alive()` once per second. On a worker exit it logs the
   failure and continues running with degraded functionality. Restart
   with state recovery is a phase-7 item if real-world failures
   warrant it.

8. **Graceful shutdown** on SIGINT (Ctrl+C) and SIGTERM (systemd stop):
   - Set `shutdown_event = multiprocessing.Event()`.
   - Push sentinel `None` to each worker input queue.
   - Workers detect either signal, drain in-flight work, exit.
   - Main joins each worker with a 3 s timeout, then `terminate()` if
     still alive.
   - LCD shows `shutdown...`, backlight off on final exit.

9. **Config loading at startup.** `load_config()` reads
   `~/.aslive/config.json` (creates with defaults if missing) and
   exposes a typed `Config` object (pydantic) used by all components.
   Includes `target_lang`, debounce thresholds (`STABLE_FRAMES`,
   `GAP_FRAMES`, `MIN_CONF`), GPIO pin assignments, and log level.

10. **Testability.** Each worker is testable in isolation by feeding
    its input queue directly. The main state machine is testable by
    injecting a fake `button_event_queue` and a fake
    `recognizer_out_queue`, then asserting on calls to mocked
    `LCDWriter`, `Translator`, `Speaker`. No camera, no GPIO, no
    audio device required for unit tests.

## Out of scope for this feature

- Multi-tenant operation (one device, one user, one session at a time).
- Hot-reload of config (changes require restart).
- Auto-restart of crashed workers (deferred to phase 7 if needed).
- IPC mechanisms other than stdlib queues (no ZeroMQ, no shared
  memory).
- Web / network admin interface (offline device, no exposed services).
