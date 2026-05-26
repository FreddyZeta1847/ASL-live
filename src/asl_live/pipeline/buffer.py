"""Word buffer — turns Debouncer commit events into emitted words.

Lives in the recognizer worker process. LETTER commits append, DELETE
commits pop, SPACE commits flush and emit. Empty-buffer SPACE / DELETE
are silent no-ops per [`feature-4-debounce.md`](../../../.claude/docs/features/feature-4-debounce.md) §4.7
(don't punish accidental signs with audible feedback).

Pure state — no I/O, no clock. Tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordBuffer:
    """Running word state in the recognizer worker."""

    chars: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.chars)

    def append(self, letter: str) -> None:
        self.chars.append(letter)

    def pop(self) -> bool:
        """Pop the last char. Returns False if the buffer was empty."""
        if not self.chars:
            return False
        self.chars.pop()
        return True

    def flush(self) -> Optional[str]:
        """Return the assembled word and clear; None if buffer was empty."""
        if not self.chars:
            return None
        word = self.text
        self.chars.clear()
        return word
