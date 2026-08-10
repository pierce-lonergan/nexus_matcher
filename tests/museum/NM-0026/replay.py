"""
Reintroduce NM-0026: put the human status output back on stdout.

One line. `_status_console` is the whole fix -- before it, Progress and every `rich.print`
in `match` went to the default console, which is stdout, and the JSON payload went to the
same place.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/presentation/cli/main.py"
ANCHOR_RE = re.compile(
    r"^    return Console\(stderr=True\) if payload_on_stdout else console\r?\n", re.M
)
BROKEN = "    return console\n"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = ANCHOR_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0026 replay: the status-console return was not found in {TARGET}")
    path.write_text(text[: match.start()] + BROKEN + text[match.end() :], encoding="utf-8")
