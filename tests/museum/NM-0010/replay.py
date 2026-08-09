"""
Reintroduce NM-0010: feed data straight into Rich's markup parser again.

Both data-bearing cells of the results table are unescaped -- the field path and the
business name. They are one defect, not two: the mistake is treating a value that came
from the user's schema or their glossary as though it were formatting we wrote.

The style cells (`[green]87.00%[/green]` and the decision colour) are deliberately left
alone. Those brackets ARE markup we wrote, and escaping them would break the colouring
rather than reproduce the defect.

Anchored on the expressions being escaped, with `[^)]*` covering the truncation slice, so
tuning `business_name[:40]` to another width does not turn this entry into a hole.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/presentation/cli/main.py"

# Each pattern keeps the escaped expression and drops the call around it.
UNESCAPES = (
    (re.compile(r"escape\((field_path)\)"), "the field-path cell"),
    (re.compile(r"escape\((match\.dictionary_entry\.business_name[^)]*)\)"), "the match cell"),
)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")

    for pattern, what in UNESCAPES:
        text, count = pattern.subn(r"\1", text, count=1)
        if count != 1:
            raise LookupError(f"NM-0010 replay: {what} is not escaped in {TARGET}")

    path.write_text(text, encoding="utf-8")
