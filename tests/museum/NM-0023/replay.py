"""
Reintroduce NM-0023 by narrowing the lint command back to a subset of the code tree.

This entry exists to prove the museum handles defects that are NOT wrong lines in src/.
A unified diff against src/ cannot express "the CI job forgot a directory", and pretending
otherwise would let the museum claim coverage it does not have.
"""

from __future__ import annotations

import pathlib
import re

TARGET = ".github/workflows/ci.yml"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    narrowed = re.sub(
        r"(ruff (?:check|format --check)) [^\n]+",
        r"\1 src/nexus_matcher",
        text,
    )
    if narrowed == text:
        raise SystemExit(f"NM-0023 replay: no ruff invocation found in {TARGET}")
    path.write_text(narrowed, encoding="utf-8")
