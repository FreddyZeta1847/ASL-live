# Feature 5 — Translation (Argos)

**Status:** ✅ LOCKED
**Companion docs:** [`../architecture.md`](../architecture.md) · [`../tech-stack.md`](../tech-stack.md)

**Module:** `src/asl_live/translation/translator.py`
**Setup script:** `scripts/setup_argos.py`

**Public API:**
```python
class Translator:
    def __init__(self):
        """Warm up all 4 language pairs (en→it, es, fr, de) on construction."""
    def translate(self, word: str, target: str) -> str:
        """Translate an uppercase English word to target language code
        (one of: it, es, fr, en, de). Returns the translated string."""
```

## Decisions

1. **Library: `argostranslate` (Python package).** Used directly, version
   pinned in `pyproject.toml`. No GUI / CLI tooling pulled in.

2. **Offline pack installation.** All 4 EN→target packs (`.argosmodel`
   files for IT, ES, FR, DE) are pre-installed at provisioning time by
   `scripts/setup_argos.py`. The runtime never attempts to fetch a pack,
   so missing-pack equals broken setup, not a recoverable runtime error.

3. **Startup warmup.** `Translator.__init__()` triggers one dummy
   translation per pair. Argos lazy-loads on first use; warmup moves
   that latency from "the user's first word after boot" to startup.
   Memory cost (~few hundred MB resident) is acceptable on Pi 5 8 GB.

4. **Casing.** Recognizer emits uppercase ("HELLO"). MT models translate
   lowercase more accurately. `translate()` lowercases before sending to
   Argos and returns the lowercase result. The TTS pronounces the same
   either way.

5. **LRU cache.** `translate()` is wrapped with
   `functools.lru_cache(maxsize=128)`. Repeated words within a session
   translate once. Tiny memory cost.

6. **Identity short-circuit.** When `target == "en"` we return the
   lowercased input unchanged. No Argos call.

7. **Failure handling.** Any exception from Argos (missing pack,
   internal error) is caught: log a warning, return the lowercased
   original word. The worker never crashes. Empty input → return empty
   string silently.

8. **Worker / queue structure.** A separate process reads from
   `translation_in_queue: (word, target_lang)` and writes to
   `tts_in_queue: translated_string`. Each input message carries its
   own `target_lang`, so language changes propagate without restarting
   the worker or sharing state.

9. **Performance target.** 100–500 ms per word on Pi 5. If profiling
   shows worse, swap that pair's model for a smaller Argos variant.

## Out of scope for this feature

- Sentence-level translation (PLAN.md keeps v1 word-by-word).
- Other source languages (we always source EN — fingerspelling produces
  English letters).
- Re-translation on language change (per #8, language is per-message).
- Custom MT models (we trust the upstream Argos packs).
