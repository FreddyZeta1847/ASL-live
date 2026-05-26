"""Offline Argos pack installer — one-shot, idempotent.

Installs the 4 EN→{IT,ES,FR,DE} packs the Translator needs (per
``config.ARGOS_LANGS``). Idempotent: re-runs detect already-installed
packs via ``argostranslate.package.get_installed_packages`` and skip
them — useful when only one download succeeded and the rest need a
retry, or when the script is re-run as part of provisioning.

The catalog is fetched from the upstream Argos package index
(argosopentech.com). Run this on every machine that will *use* the
Translator — both the dev PC and the Pi.

Usage::

    python scripts/setup_argos.py
"""
from __future__ import annotations

import sys

import argostranslate.package as argos_pkg

from asl_live.config import ARGOS_LANGS

SOURCE_LANG = "en"


def is_installed(target: str) -> bool:
    """Return True iff the EN→target pack is already on disk."""
    return any(
        p.from_code == SOURCE_LANG and p.to_code == target
        for p in argos_pkg.get_installed_packages()
    )


def find_available_pack(target: str):
    """Return the upstream AvailablePackage for EN→target, or None."""
    return next(
        (
            p for p in argos_pkg.get_available_packages()
            if p.from_code == SOURCE_LANG and p.to_code == target
        ),
        None,
    )


def install_pack(target: str) -> None:
    """Download + install the EN→target pack. Raises if not in the catalog."""
    pack = find_available_pack(target)
    if pack is None:
        raise RuntimeError(
            f"No upstream Argos pack found for {SOURCE_LANG}→{target}. "
            "Check the catalog at argosopentech.com."
        )
    print(f"  {SOURCE_LANG}→{target}: downloading ...", flush=True)
    download_path = pack.download()
    print(f"  {SOURCE_LANG}→{target}: installing ...", flush=True)
    argos_pkg.install_from_path(download_path)


def main() -> int:
    print(f"setup_argos: target packs {SOURCE_LANG}→{','.join(ARGOS_LANGS)}")
    print("Updating Argos package index ...", flush=True)
    argos_pkg.update_package_index()

    todo = [t for t in ARGOS_LANGS if not is_installed(t)]
    skipped = [t for t in ARGOS_LANGS if t not in todo]

    for t in skipped:
        print(f"  {SOURCE_LANG}→{t}: already installed, skipping")
    for t in todo:
        install_pack(t)

    print()
    print(f"Done. Installed: {len(todo)}; already present: {len(skipped)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
