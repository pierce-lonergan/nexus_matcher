"""
Reintroduce NM-0029: put a confidential term back into a file that ships in the wheel.

The planted string is the PUBLIC CANARY, not the term that actually leaked. This is a
deliberate choice, not a weakening of the replay:

  * the code path is identical -- a blocked digest matching a line under src/, which is
    the entire mechanism the fix installs. Nothing about the real term is special to the
    scanner; it is one digest among many in the same list.
  * storing the real term here to prove the real term cannot be stored would defeat the
    fix, and would put it in git history permanently, where it is far harder to remove
    than from a working tree.

The comment is planted immediately above DEFAULT_HIERARCHY_DATA, which is where the
original sat -- the attribution of a default data structure to the organisation it was
sketched from is exactly the shape this defect takes when it recurs.

Anchored on the assignment statement rather than on any comment above it. A comment is
the most-edited line in any file; anchoring there is how NM-0003 quietly became a hole
when somebody added an explanatory sentence next to it.
"""

from __future__ import annotations

import pathlib

TARGET = "src/nexus_matcher/domain/services/domain_hierarchy.py"
ANCHOR = "DEFAULT_HIERARCHY_DATA: dict = {"

# Matches tests/meta/test_no_confidential_terms.py::CANARY. Public and meaningless.
CANARY = "zzqx-canary-confidential-gate"
PLANTED = f"# {CANARY} domain hierarchy\n"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    if ANCHOR not in text:
        raise LookupError(f"NM-0029 replay: {ANCHOR!r} not found in {TARGET}")
    path.write_text(text.replace(ANCHOR, PLANTED + ANCHOR, 1), encoding="utf-8")
