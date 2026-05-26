"""Unit tests for the Piper Speaker wrapper.

``piper.PiperVoice`` and ``sounddevice`` are monkey-patched so tests run
without any real voice files on disk and without playing audio. Each
test rebuilds the Speaker against a temp voices dir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from asl_live.config import TTS_LANGS
from asl_live.tts import speaker as speaker_mod


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeAudioChunk:
    sample_rate: int = 22050
    audio_int16_array: np.ndarray = field(
        default_factory=lambda: np.zeros(100, dtype=np.int16)
    )


class _FakeVoice:
    """Minimal stand-in for piper.PiperVoice. Records synthesize calls."""

    def __init__(self, lang_tag: str) -> None:
        self.lang_tag = lang_tag
        self.synthesize_calls: list[str] = []

    def synthesize(self, text: str):
        self.synthesize_calls.append(text)
        yield _FakeAudioChunk()


class _PlayRecorder:
    """Captures sounddevice.play(audio, rate) calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []  # (num_samples, rate)

    def __call__(self, audio: np.ndarray, rate: int) -> None:
        self.calls.append((len(audio), rate))


def _write_voice_files(voices_dir: Path, langs: tuple[str, ...]) -> None:
    """Create empty placeholder .onnx + .json files so the existence check passes."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        (voices_dir / f"{lang}.onnx").write_bytes(b"")
        (voices_dir / f"{lang}.json").write_text("{}")


@pytest.fixture
def voices_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "voices"
    monkeypatch.setattr(speaker_mod, "PIPER_VOICES_DIR", d)
    return d


@pytest.fixture
def fake_piper(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeVoice]:
    """Replace PiperVoice.load with a factory that returns a tagged fake."""
    voices: dict[str, _FakeVoice] = {}

    def fake_load(model_path: str, config_path: str | None = None, **_: object) -> _FakeVoice:
        # Tag the fake voice with the filename stem so tests can identify it.
        lang = Path(model_path).stem
        voice = _FakeVoice(lang_tag=lang)
        voices[lang] = voice
        return voice

    monkeypatch.setattr(speaker_mod.PiperVoice, "load", staticmethod(fake_load))
    return voices


@pytest.fixture
def play_recorder(monkeypatch: pytest.MonkeyPatch) -> _PlayRecorder:
    rec = _PlayRecorder()
    monkeypatch.setattr(speaker_mod.sounddevice, "play", rec)
    monkeypatch.setattr(speaker_mod.sounddevice, "wait", lambda: None)
    return rec


# ---------------------------------------------------------------------------
# Construction — preload all 5 voices; raise on any missing
# ---------------------------------------------------------------------------


def test_constructor_preloads_all_five_voices(
    voices_dir: Path, fake_piper: dict[str, _FakeVoice]
) -> None:
    _write_voice_files(voices_dir, TTS_LANGS)
    speaker_mod.Speaker()
    assert set(fake_piper) == set(TTS_LANGS)


def test_missing_onnx_raises_file_not_found(
    voices_dir: Path, fake_piper: dict[str, _FakeVoice]
) -> None:
    """No fallback — feature-6 §5 (locked decision)."""
    _write_voice_files(voices_dir, TTS_LANGS)
    (voices_dir / "fr.onnx").unlink()  # remove just the French voice
    with pytest.raises(FileNotFoundError, match=r"'fr'"):
        speaker_mod.Speaker()


def test_missing_json_raises_file_not_found(
    voices_dir: Path, fake_piper: dict[str, _FakeVoice]
) -> None:
    _write_voice_files(voices_dir, TTS_LANGS)
    (voices_dir / "es.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"'es'"):
        speaker_mod.Speaker()


# ---------------------------------------------------------------------------
# speak() — voice selection, empty input, exception swallowing
# ---------------------------------------------------------------------------


def test_speak_routes_to_correct_voice(
    voices_dir: Path,
    fake_piper: dict[str, _FakeVoice],
    play_recorder: _PlayRecorder,
) -> None:
    _write_voice_files(voices_dir, TTS_LANGS)
    s = speaker_mod.Speaker()
    s.speak("ciao", "it")
    assert fake_piper["it"].synthesize_calls == ["ciao"]
    # Other voices untouched.
    for lang in TTS_LANGS:
        if lang != "it":
            assert fake_piper[lang].synthesize_calls == []


def test_speak_plays_synthesized_audio(
    voices_dir: Path,
    fake_piper: dict[str, _FakeVoice],
    play_recorder: _PlayRecorder,
) -> None:
    _write_voice_files(voices_dir, TTS_LANGS)
    s = speaker_mod.Speaker()
    s.speak("hello", "en")
    assert len(play_recorder.calls) == 1
    num_samples, rate = play_recorder.calls[0]
    assert num_samples == 100  # one fake chunk of 100 samples
    assert rate == 22050


def test_speak_with_empty_text_is_silent_noop(
    voices_dir: Path,
    fake_piper: dict[str, _FakeVoice],
    play_recorder: _PlayRecorder,
) -> None:
    _write_voice_files(voices_dir, TTS_LANGS)
    s = speaker_mod.Speaker()
    s.speak("", "it")
    assert fake_piper["it"].synthesize_calls == []
    assert play_recorder.calls == []


def test_synthesis_exception_is_swallowed(
    voices_dir: Path,
    fake_piper: dict[str, _FakeVoice],
    play_recorder: _PlayRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_voice_files(voices_dir, TTS_LANGS)
    s = speaker_mod.Speaker()

    def explode(*_args, **_kwargs):
        raise RuntimeError("piper exploded")
        yield  # pragma: no cover — make this a generator function

    monkeypatch.setattr(fake_piper["it"], "synthesize", explode)
    # Must not raise.
    s.speak("ciao", "it")
    assert play_recorder.calls == []


def test_unknown_lang_is_silent_noop(
    voices_dir: Path,
    fake_piper: dict[str, _FakeVoice],
    play_recorder: _PlayRecorder,
) -> None:
    _write_voice_files(voices_dir, TTS_LANGS)
    s = speaker_mod.Speaker()
    s.speak("hello", "ja")  # not in TTS_LANGS
    assert play_recorder.calls == []
