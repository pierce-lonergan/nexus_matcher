"""
Reintroduce NM-0031: let the last declaration of an alias quietly win.

The conflict check is DELETED, not inverted. Inverting it would refuse catalogs that were
never ambiguous -- a loud, different defect that every governance test in the repo would
notice. The historical shape is the permissive one: `resolved[key] = target_key` simply
overwrites whatever the previous declaration put there, so the catalog resolves the
contested spelling to whichever class was written second and nothing anywhere reports it.

Anchored from the `if key in resolved` line up to the assignment that follows it, and the
assignment is what the lookahead pins. Matching the condition line alone would leave a
dangling `raise ValueError(...)` behind, and matching the comment inside the block would
rot the first time somebody rewords it -- NM-0016 was silently turned into a hole exactly
that way, by anchoring on a default argument that was later tuned. A plain assignment is
the most reformat-stable thing in this function; if the check is ever restructured out of
this loop, this stops matching and the runner reports a HOLE rather than a replay that
no-ops into a false PASS.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/domain/governance.py"

# From the conflict check up to (not including) the write it guards.
BLOCK_RE = re.compile(
    r"^            if key in resolved[^\n]*\n(?:.*?\n)*?(?=^            resolved\[key\] = target_key)",
    re.M,
)

BROKEN = "            # NM-0031 replay: the last declaration of a token wins, silently.\n"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = BLOCK_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0031 replay: the alias conflict check was not found in {TARGET}")
    path.write_text(text[: match.start()] + BROKEN + text[match.end() :], encoding="utf-8")
