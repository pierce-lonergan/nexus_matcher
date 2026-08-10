"""Reintroduce NM-0024: hash a hand-listed subset instead of the embedded text."""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/application/ingest.py"
ANCHOR_RE = re.compile(r"^    payload = entry\.to_searchable_text\(\)\n", re.M)
BROKEN = '    payload = "\x1f".join((entry.business_name, entry.logical_name, entry.definition))\n'


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = ANCHOR_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0024 replay: content_hash payload line not found in {TARGET}")
    path.write_text(text[: match.start()] + BROKEN + text[match.end() :], encoding="utf-8")
