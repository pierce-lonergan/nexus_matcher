"""
Reintroduce NM-0006: key results by the derived dotted path instead of the caller's name.

Note this is NOT the same mutation as NM-0005, which reverts the CALL SITE
(`results[...] = ...`) and so reintroduces the collision. This one reverts the identity
FUNCTION, which is the half about addressability: with it, no two fields collide, every
count check still passes, and the keys are simply names the caller's schema never used.

Anchored on the function signature by regex. The body is a long docstring explaining the
+19.3 P@1 reason the parser rewrites the name at all -- exactly the sort of prose that
gets reworded, and exactly what a replay must not depend on.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/application/use_cases/match_schema.py"

ANCHOR_RE = re.compile(r"^def field_result_key\([^)]*\) -> str:\n", re.M)

BODY = (
    "    # NM-0006 replay: hand back the path we derived, not the name we were given.\n"
    "    return field.full_path\n"
)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = ANCHOR_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0006 replay: field_result_key signature not found in {TARGET}")
    path.write_text(text[: match.end()] + BODY + text[match.end() :], encoding="utf-8")
