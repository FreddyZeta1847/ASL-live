"""Piper text-to-speech wrapper — per [`feature-6-tts.md`](../../../.claude/docs/features/feature-6-tts.md).

Preloads all 5 voices at construction so language switches are instant.
Synthesizes via Piper and plays through whatever audio device
``sounddevice`` defaults to (ALSA on Pi, WASAPI on Windows).

Locked decisions
----------------
- Missing voice file at ``__init__`` → ``FileNotFoundError``. No
  eSpeak-NG fallback (feature-6 §5; rationale captured there).
- Synthesis exceptions during ``speak()`` are caught, logged, and
  swallowed — the worker must never crash on a bad word.
- Empty input → no-op (skip both synthesis and playback).
- ``speak()`` blocks until playback finishes (feature-6 §3).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sounddevice
from piper import PiperVoice

from asl_live.config import PIPER_VOICES_DIR, TTS_LANGS

logger = logging.getLogger(__name__)


def _voice_paths(lang: str) -> tuple[Path, Path]:
    """Return ``(onnx_path, json_path)`` for a given language code."""
    return (
        PIPER_VOICES_DIR / f"{lang}.onnx",
        PIPER_VOICES_DIR / f"{lang}.json",
    )


class Speaker:
    """Synthesizes English/translated text and plays it on the default audio device.

    All voices are loaded in ``__init__``; subsequent ``speak()`` calls
    are pure synthesis + playback.
    """

    def __init__(self) -> None:
        self._voices: dict[str, PiperVoice] = {}
        for lang in TTS_LANGS:
            self._voices[lang] = self._load_voice(lang)

    def speak(self, text: str, lang: str) -> None:
        """Synthesize ``text`` in ``lang`` and play it. Blocks until done.

        Empty text is a silent no-op. Any synthesis or playback exception
        is caught and logged; the call still returns normally.
        """
        if not text:
            return
        voice = self._voices.get(lang)
        if voice is None:
            logger.warning("No Piper voice loaded for lang=%r; skipping speak", lang)
            return
        try:
            self._synthesize_and_play(voice, text)
        except Exception as exc:
            logger.warning("Piper speak(%r, %s) failed: %s; skipping", text, lang, exc)

    # -- internals -----------------------------------------------------------

    def _load_voice(self, lang: str) -> PiperVoice:
        onnx_path, json_path = _voice_paths(lang)
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"Piper voice missing for {lang!r}: {onnx_path}. "
                "Run `python scripts/setup_piper.py` to install."
            )
        if not json_path.is_file():
            raise FileNotFoundError(
                f"Piper config missing for {lang!r}: {json_path}. "
                "Run `python scripts/setup_piper.py` to install."
            )
        return PiperVoice.load(str(onnx_path), config_path=str(json_path))

    def _synthesize_and_play(self, voice: PiperVoice, text: str) -> None:
        """Run synthesis to completion, concatenate chunks, play, block on done."""
        chunks = list(voice.synthesize(text))
        if not chunks:
            return
        sample_rate = chunks[0].sample_rate
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        sounddevice.play(audio, sample_rate)
        sounddevice.wait()
