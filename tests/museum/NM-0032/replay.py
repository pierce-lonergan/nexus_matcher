"""Reintroduce NM-0032: let an unrecognised option travel on to the reader unremarked.

The check's BODY is emptied rather than the function being deleted, so the call site in
`load_entries` still resolves and the only thing that changes is whether an unknown
keyword is noticed. That is exactly the historical state: every keyword `load_entries` did
not name for itself went into `**kwargs`, straight through `read_source`, into a reader
that pops the options it knows and never looks at the rest.

Anchored on the statements themselves -- the assignment, the guard, and the closing paren
of the refusal -- rather than on whatever function happens to follow. A neighbour-anchored
match would rot the moment a helper is added next to this one, and rot in a museum replay
is not a failure: it is a HOLE that reports PASS. The refusal's message text is
deliberately not matched on, because rewording it is the likeliest edit this function will
ever see, and NM-0016 was quietly turned into a hole exactly that way.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/application/ingest.py"

# From the unsupported-option assignment through the closing paren of the refusal it
# guards. The docstring above it is left alone -- what is under test is the behaviour.
BLOCK_RE = re.compile(
    r"^    unsupported = sorted\(set\(kwargs\) - _READER_OPTIONS\)\n"
    r"    if unsupported:\n"
    r"        raise ValueError\(\n"
    r"(?:.*?\n)*?"
    r"        \)\n",
    re.M,
)

BROKEN = "    return  # NM-0032 replay: an unknown option goes to the reader, which drops it.\n"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = BLOCK_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0032 replay: the unsupported-option check was not found in {TARGET}")
    path.write_text(text[: match.start()] + BROKEN + text[match.end() :], encoding="utf-8")
