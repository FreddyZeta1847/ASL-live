"""Argos translation wrapper — per [`feature-5-translation.md`](../../../.claude/docs/features/feature-5-translation.md).

Verifies all four EN→{IT,ES,FR,DE} packs are installed at construction
time, warms each one with a dummy translation so the runtime first-word
latency is paid at startup instead of on the user's first SPACE commit,
and exposes ``translate()`` with LRU caching, identity shortcut for EN,
lowercase normalization, and exception swallowing.

Locked decisions
----------------
- Missing pack → ``RuntimeError`` (feature-5 §2: broken setup, not a
  recoverable runtime error). The recognizer/translator worker should
  not be started if setup hasn't been run.
- Runtime exceptions during ``translate()`` → caught, logged, return the
  lowercased input unchanged (feature-5 §7: worker must never crash).
- Cache lives per-instance so it dies with the Translator (avoids the
  ``functools.lru_cache``-on-bound-method footgun where the cache holds
  ``self`` forever).
"""
from __future__ import annotations

import functools
import logging

import argostranslate.package as _argos_pkg
import argostranslate.translate as _argos_translate

from asl_live.config import ARGOS_LANGS

logger = logging.getLogger(__name__)

_WARMUP_WORD = "hello"
"""Dummy word fed through every pack during ``__init__`` to force Argos
to load weights eagerly. ASCII, single word, common enough to be in
every pack's vocab."""


class Translator:
    """Translates lowercase English words to one of the supported targets.

    Construction does all the heavy lifting (pack verification + warmup)
    so the runtime hot path is fast and predictable.
    """

    def __init__(self) -> None:
        self._verify_packs_installed()
        self._cached_translate = functools.lru_cache(maxsize=128)(
            self._translate_uncached
        )
        self._warm_up()

    def translate(self, word: str, target: str) -> str:
        """Translate ``word`` into ``target``.

        Returns lowercased translation. Empty input → empty output.
        ``target == "en"`` is an identity shortcut — no Argos call,
        no cache entry. Argos exceptions are swallowed and the
        lowercased input is returned unchanged.
        """
        if not word:
            return ""
        word_lower = word.lower()
        if target == "en":
            return word_lower
        return self._cached_translate(word_lower, target)

    # -- internals -----------------------------------------------------------

    def _translate_uncached(self, word_lower: str, target: str) -> str:
        try:
            return _argos_translate.translate(word_lower, "en", target).lower()
        except Exception as exc:
            logger.warning(
                "Argos translate(%r, en→%s) failed: %s; returning original",
                word_lower, target, exc,
            )
            return word_lower

    def _verify_packs_installed(self) -> None:
        """Raise if any required EN→target pack is missing."""
        installed = {
            (p.from_code, p.to_code) for p in _argos_pkg.get_installed_packages()
        }
        missing = [t for t in ARGOS_LANGS if ("en", t) not in installed]
        if missing:
            pairs = ", ".join(f"en→{t}" for t in missing)
            raise RuntimeError(
                f"Missing Argos packs: {pairs}. "
                "Run `python scripts/setup_argos.py` to install them."
            )

    def _warm_up(self) -> None:
        """One dummy translation per target so model weights are paged in."""
        for target in ARGOS_LANGS:
            # Bypass the cache during warmup — we don't want the dummy
            # 'hello' translation occupying a cache slot at runtime.
            self._translate_uncached(_WARMUP_WORD, target)
