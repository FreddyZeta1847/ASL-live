"""Unit tests for the Argos Translator wrapper.

``argostranslate`` is monkey-patched so tests run without installed packs
and without invoking the real translation model. Each test rebuilds the
Translator (no cache leakage between cases) and asserts on what the mock
saw.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from asl_live.config import ARGOS_LANGS
from asl_live.translation import translator as translator_mod


# ---------------------------------------------------------------------------
# Mock argostranslate
# ---------------------------------------------------------------------------


@dataclass
class _FakePackage:
    from_code: str
    to_code: str


def _all_packs_installed() -> list[_FakePackage]:
    """Mock for argostranslate.package.get_installed_packages — full set."""
    return [_FakePackage("en", t) for t in ARGOS_LANGS]


def _no_packs_installed() -> list[_FakePackage]:
    return []


class _TranslateRecorder:
    """Stand-in for argostranslate.translate.translate. Records each call."""

    def __init__(self, side_effect: Callable[[str, str, str], str] | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._side_effect = side_effect

    def __call__(self, text: str, from_code: str, to_code: str) -> str:
        self.calls.append((text, from_code, to_code))
        if self._side_effect is not None:
            return self._side_effect(text, from_code, to_code)
        # Default: prefix the text so the test can tell input from output.
        return f"<{to_code}>{text}"

    def clear(self) -> None:
        """Forget all calls so far. Tests call this after constructing the
        Translator so the warmup translations don't pollute later assertions
        (warmup uses 'hello', which also happens to be a popular test input)."""
        self.calls.clear()


@pytest.fixture
def packs_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        translator_mod._argos_pkg, "get_installed_packages", _all_packs_installed
    )


@pytest.fixture
def packs_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        translator_mod._argos_pkg, "get_installed_packages", _no_packs_installed
    )


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _TranslateRecorder:
    rec = _TranslateRecorder()
    monkeypatch.setattr(translator_mod._argos_translate, "translate", rec)
    return rec


# ---------------------------------------------------------------------------
# Constructor — warmup + pack verification
# ---------------------------------------------------------------------------


def test_constructor_warms_up_all_four_pairs(
    packs_installed: None, recorder: _TranslateRecorder
) -> None:
    translator_mod.Translator()
    warmup_calls = [c for c in recorder.calls if c[0] == translator_mod._WARMUP_WORD]
    assert len(warmup_calls) == len(ARGOS_LANGS)
    assert {c[2] for c in warmup_calls} == set(ARGOS_LANGS)
    # Source is always 'en'.
    assert all(c[1] == "en" for c in warmup_calls)


def test_constructor_raises_when_any_pack_missing(packs_missing: None) -> None:
    with pytest.raises(RuntimeError, match="Missing Argos packs"):
        translator_mod.Translator()


def test_constructor_raises_lists_only_missing_packs(
    monkeypatch: pytest.MonkeyPatch, recorder: _TranslateRecorder
) -> None:
    """If 3 of 4 packs are installed, the error names only the missing one."""
    partial = [_FakePackage("en", t) for t in ARGOS_LANGS if t != "fr"]
    monkeypatch.setattr(
        translator_mod._argos_pkg, "get_installed_packages", lambda: partial
    )
    with pytest.raises(RuntimeError, match=r"en→fr"):
        translator_mod.Translator()


# ---------------------------------------------------------------------------
# translate() — identity, lowercase, empty, exception swallowing
# ---------------------------------------------------------------------------


def test_identity_shortcut_for_english_target(
    packs_installed: None, recorder: _TranslateRecorder
) -> None:
    t = translator_mod.Translator()
    recorder.clear()
    result = t.translate("HELLO", "en")
    assert result == "hello"
    assert recorder.calls == []  # no Argos call


def test_input_is_lowercased_before_argos(
    packs_installed: None, recorder: _TranslateRecorder
) -> None:
    t = translator_mod.Translator()
    recorder.clear()
    t.translate("HELLO", "it")
    assert recorder.calls == [("hello", "en", "it")]


def test_translation_result_is_lowercased(
    monkeypatch: pytest.MonkeyPatch, packs_installed: None
) -> None:
    rec = _TranslateRecorder(side_effect=lambda *_: "CIAO")
    monkeypatch.setattr(translator_mod._argos_translate, "translate", rec)
    t = translator_mod.Translator()
    assert t.translate("HELLO", "it") == "ciao"


def test_empty_input_returns_empty_and_skips_argos(
    packs_installed: None, recorder: _TranslateRecorder
) -> None:
    t = translator_mod.Translator()
    recorder.clear()
    assert t.translate("", "it") == ""
    assert recorder.calls == []


def test_argos_exception_is_swallowed_and_input_returned(
    monkeypatch: pytest.MonkeyPatch, packs_installed: None
) -> None:
    def _raise(*_args: object) -> str:
        raise RuntimeError("argos exploded")

    rec = _TranslateRecorder(side_effect=_raise)
    monkeypatch.setattr(translator_mod._argos_translate, "translate", rec)
    t = translator_mod.Translator()
    assert t.translate("HELLO", "it") == "hello"


# ---------------------------------------------------------------------------
# LRU cache — same input twice means one underlying call
# ---------------------------------------------------------------------------


def test_lru_cache_collapses_repeat_calls(
    packs_installed: None, recorder: _TranslateRecorder
) -> None:
    t = translator_mod.Translator()
    recorder.clear()
    t.translate("HELLO", "it")
    t.translate("HELLO", "it")
    t.translate("hello", "it")  # same after lowercasing
    assert recorder.calls == [("hello", "en", "it")]


def test_lru_cache_separates_targets(
    packs_installed: None, recorder: _TranslateRecorder
) -> None:
    t = translator_mod.Translator()
    recorder.clear()
    t.translate("HELLO", "it")
    t.translate("HELLO", "fr")
    assert ("hello", "en", "it") in recorder.calls
    assert ("hello", "en", "fr") in recorder.calls


def test_lru_cache_is_per_instance(
    packs_installed: None, recorder: _TranslateRecorder
) -> None:
    """Two Translator instances must not share cache state."""
    t1 = translator_mod.Translator()
    t1.translate("HELLO", "it")
    t2 = translator_mod.Translator()
    recorder.clear()
    t2.translate("HELLO", "it")
    # t2 must invoke Argos even though t1 already translated the same input.
    assert recorder.calls == [("hello", "en", "it")]
