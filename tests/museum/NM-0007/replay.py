"""
Reintroduce NM-0007: put an extras-only name back into __all__.

Nothing else changes -- `create_app` still resolves lazily, `_OPTIONAL_EXPORTS` still
lists it, `__dir__` still advertises it. The whole defect is one string in one list, which
is why it was so easy to ship: the name is genuinely importable, on every machine that has
the `api` extra, including all of them in CI.

Anchored on the `__all__ = [` opening line rather than on any neighbouring entry. The list
is alphabetically grouped with section comments and gains names regularly; anchoring next
to a sibling would make this replay rot the next time somebody exports something.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/__init__.py"

ANCHOR_RE = re.compile(r"^__all__ = \[\n", re.M)
ENTRY = '    "create_app",  # NM-0007 replay: needs the api extra, promised unconditionally\n'


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = ANCHOR_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0007 replay: no `__all__ = [` list in {TARGET}")
    if '"create_app"' in text[match.end() :].split("\n]", 1)[0]:
        raise LookupError(f"NM-0007 replay: create_app is already in __all__ in {TARGET}")
    path.write_text(text[: match.end()] + ENTRY + text[match.end() :], encoding="utf-8")
