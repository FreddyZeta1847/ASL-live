# Feature 9 — Audio language menu

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/ui/lang_menu.py`
**Persistence:** `~/.aslive/config.json`

**Public API:**
```python
class LanguageMenu:
    def __init__(self, speaker: Speaker, lcd: LCDWriter, config_path: Path): ...
    def open(self) -> None:           """Enter menu, announce current candidate."""
    def on_b2_press(self) -> None:    """Cycle to next candidate."""
    def on_b1_press(self) -> None:    """Cancel, restore previous selection."""
    def tick(self) -> None:           """Called periodically; auto-confirms on 3 s timeout."""
    @property
    def is_open(self) -> bool: ...
    @property
    def current_lang(self) -> str: ...
```

## Decisions

1. **Persistence: `~/.aslive/config.json`.** Schema:
   ```json
   {"target_lang": "it", "version": 1}
   ```
   Loaded on boot, written atomically (write-to-temp + rename) on
   confirm. `version` field is forward-compatibility insurance.

2. **First-boot default: English (`"en"`).** Neutral choice when no
   config file exists. The user changes it via the menu on first use
   and the file is created.

3. **Cycle order:** IT → ES → FR → EN → DE → IT.

4. **Announcement strategy: each language speaks its own name in its
   own voice.**
   - IT candidate → Piper Italian voice says "Italiano".
   - ES → Spanish voice says "Español".
   - FR → French voice says "Français".
   - EN → English voice says "English".
   - DE → German voice says "Deutsch".

   On confirm, "OK <lang>" is spoken in the *newly chosen* language's
   voice. On cancel, the *previous (preserved)* language's name is
   spoken in its voice — the user always hears something when leaving
   the menu, removing ambiguity about which selection ended up active.

5. **Boot announcement.** After all workers warm up, the persisted
   language's name is spoken in its own voice. Only "device ready" cue
   in the absence of a screen.

6. **Confirm timeout: 3 s.** Each candidate announcement resets the
   timer. After 3 s of no B2 press, the current candidate is confirmed
   and persisted. Walk-away behavior is acceptable: if the user
   accidentally cycled before walking off, they re-open and fix it.

7. **State transitions:**
   - **closed** + B2 → enter menu, announce current candidate, start
     timer.
   - **open** + B2 → advance candidate, re-announce, restart timer.
   - **open** + B1 → cancel, restore previous, announce previous,
     close.
   - **open** + 3 s elapsed → confirm, persist, announce "OK <lang>",
     close.

8. **LCD during MENU_OPEN.**
   - Line 1: candidate language name (e.g., `Italiano`).
   - Line 2: `LANG|<code>` (e.g., `LANG|IT  `).
   On exit, the LCD reverts to whatever idle/capture render the
   orchestrator drives next.

9. **Routing.** When `is_open` is True, the orchestrator delivers
   button events to `on_b1_press` / `on_b2_press`; capture-mode
   handlers see nothing. When `is_open` is False, the orchestrator
   handles capture-mode logic and only `open()` is callable from
   outside.

10. **Testability.** Unit tests mock `Speaker.speak()` and inject a
    fake clock for the timeout. No GPIO required. Cover: cycle order,
    confirm-on-timeout, cancel-restores-previous,
    persistence-write-atomic-on-confirm, persistence-not-written-on-cancel.

## Out of scope for this feature

- Visual menu on the LCD beyond the candidate name (no scrolling list,
  no arrow indicators).
- Voice prompts for menu help / instructions.
- Adding / removing languages at runtime (set is fixed at IT/ES/FR/EN/DE).
- Per-user profiles.
