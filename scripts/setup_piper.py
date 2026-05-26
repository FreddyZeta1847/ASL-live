"""Offline Piper voice downloader — one-shot, idempotent.

Downloads the 5 voice files the Speaker needs into ``PIPER_VOICES_DIR``
and renames them to the flat ``<lang>.onnx`` / ``<lang>.json`` layout
the Speaker class expects. Upstream voices live in the
``rhasspy/piper-voices`` Hugging Face repo.

Voice selection
---------------
One voice per language, defaulting to a *medium* model where available
(feature-6 §1). The Italian voice is ``x_low`` quality until a medium
option is published. Files are downloaded to a ``.tmp`` path and
``os.replace``-d into place so an interrupted download never leaves
a half-written file that looks complete on the re-run.

Usage::

    python scripts/setup_piper.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

from asl_live.config import PIPER_VOICES_DIR, TTS_LANGS

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# lang -> (upstream voice name, upstream directory path)
VOICES: dict[str, tuple[str, str]] = {
    "it": ("it_IT-riccardo-x_low", "it/it_IT/riccardo/x_low"),
    "es": ("es_ES-davefx-medium", "es/es_ES/davefx/medium"),
    "fr": ("fr_FR-siwis-medium", "fr/fr_FR/siwis/medium"),
    "en": ("en_US-amy-medium", "en/en_US/amy/medium"),
    "de": ("de_DE-thorsten-medium", "de/de_DE/thorsten/medium"),
}


def is_installed(lang: str) -> bool:
    """True iff both ``<lang>.onnx`` and ``<lang>.json`` are on disk."""
    return (
        (PIPER_VOICES_DIR / f"{lang}.onnx").is_file()
        and (PIPER_VOICES_DIR / f"{lang}.json").is_file()
    )


def upstream_urls(lang: str) -> tuple[str, str]:
    """Return ``(onnx_url, json_url)`` for the pinned voice of ``lang``."""
    name, path = VOICES[lang]
    return (
        f"{BASE_URL}/{path}/{name}.onnx",
        f"{BASE_URL}/{path}/{name}.onnx.json",
    )


def download_atomic(url: str, final_path: Path) -> None:
    """Download to ``final_path.tmp`` then ``os.replace`` into place."""
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    urlretrieve(url, tmp_path)
    os.replace(tmp_path, final_path)


def install_voice(lang: str) -> None:
    onnx_url, json_url = upstream_urls(lang)
    onnx_path = PIPER_VOICES_DIR / f"{lang}.onnx"
    json_path = PIPER_VOICES_DIR / f"{lang}.json"

    print(f"  {lang}: downloading {onnx_url}", flush=True)
    download_atomic(onnx_url, onnx_path)
    print(f"  {lang}: downloading {json_url}", flush=True)
    download_atomic(json_url, json_path)


def main() -> int:
    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"setup_piper: voices dir = {PIPER_VOICES_DIR}")
    print(f"setup_piper: targets = {','.join(TTS_LANGS)}")

    todo = [lang for lang in TTS_LANGS if not is_installed(lang)]
    skipped = [lang for lang in TTS_LANGS if lang not in todo]

    for lang in skipped:
        print(f"  {lang}: already installed, skipping")
    for lang in todo:
        install_voice(lang)

    print()
    print(f"Done. Installed: {len(todo)}; already present: {len(skipped)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
