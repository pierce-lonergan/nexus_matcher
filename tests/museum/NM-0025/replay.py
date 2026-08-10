"""
Reintroduce NM-0025: strip the governance payload back out of the JSON output.

Two one-line deletions, because the shipped defect was two facts and not one broken
function: the classification was absent from the dictionary entry, and the weights that
produced the confidence were absent from the record. Restoring the whole pre-fix
`_format_json` verbatim would be a larger anchor with no more evidence in it, and would
rot the first time the writer is reformatted.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/presentation/cli/main.py"

# The classification the entire stated use case depends on.
PROTECTION_RE = re.compile(r'^ *"protection_level": entry\.protection_level\.value,\r?\n', re.M)

# Without these, the five components are five numbers with no stated way to combine them.
WEIGHTS_RE = re.compile(r'^ *"weights": dict\(weights\),\r?\n', re.M)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")

    for name, pattern in (("protection_level", PROTECTION_RE), ("weights", WEIGHTS_RE)):
        match = pattern.search(text)
        if match is None:
            raise LookupError(f"NM-0025 replay: {name} line not found in {TARGET}")
        text = text[: match.start()] + text[match.end() :]

    path.write_text(text, encoding="utf-8")
