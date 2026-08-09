"""
Reintroduce NM-0001: put a legacy Windows console back in front of a UTF-8-only CLI.

The shipped fix has TWO halves, and this reverts both, because that is the state the
defect was actually reported from:

  * `_ascii_only()` -- the signal `_glyph()` and `_spinner_column()` consult to pick
    decoration the console can encode. Forcing it False restores the Braille spinner and
    the U+2022 / U+2713 literals.
  * `_soften_encoding_errors()` -- the backstop for characters we do not choose, which is
    what covers Rich's own U+2026 truncation of a `no_wrap` column. Making it a no-op
    restores the strict error handler.

Anchored on the two function SIGNATURES via regex, not on their bodies. The bodies are
prose-heavy and have already been reworded once; a replay pinned to their text would rot
into a hole that still reads as coverage. Each half is reverted by inserting an early
return, which is why the docstring below it becomes unreachable rather than being deleted
-- less text to match means fewer ways to rot.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/presentation/cli/main.py"

# `-> bool` / `-> None` are part of the anchor deliberately: they are the two things that
# would have to change for the mutation to stop being meaningful.
ASCII_ONLY_RE = re.compile(r"^def _ascii_only\([^)]*\) -> bool:\n", re.M)
SOFTEN_RE = re.compile(r"^def _soften_encoding_errors\([^)]*\) -> None:\n", re.M)

ASCII_ONLY_BODY = "    return False  # NM-0001 replay: pretend every console speaks UTF-8\n"
SOFTEN_BODY = "    return None  # NM-0001 replay: leave the stream on strict encoding\n"


def _insert_after(text: str, pattern: re.Pattern[str], body: str, what: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise LookupError(f"NM-0001 replay: {what} not found in {TARGET}")
    return text[: match.end()] + body + text[match.end() :]


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    text = _insert_after(text, ASCII_ONLY_RE, ASCII_ONLY_BODY, "_ascii_only signature")
    text = _insert_after(text, SOFTEN_RE, SOFTEN_BODY, "_soften_encoding_errors signature")
    path.write_text(text, encoding="utf-8")
