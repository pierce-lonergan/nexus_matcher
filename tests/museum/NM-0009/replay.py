"""
Reintroduce NM-0009: let the summary line divide by a field count of zero again.

Switching the guard off rather than deleting it is deliberate. The percentage expression
below it is what actually raises, and it stays exactly as shipped -- so this replay
reproduces the defect through the real arithmetic on the real code path, instead of
substituting a hand-written `1 / 0` that would prove nothing about this CLI.

Anchored on the guard's condition with the indentation captured, so the block may be
re-nested without the replay rotting.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/presentation/cli/main.py"

GUARD_RE = re.compile(r"^(?P<indent>[ \t]+)if not total_fields:[ \t]*$", re.M)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")

    rewritten, count = GUARD_RE.subn(
        r"\g<indent>if False:  # NM-0009 replay: no empty-schema guard",
        text,
        count=1,
    )
    if count != 1:
        raise LookupError(f"NM-0009 replay: no `if not total_fields:` guard in {TARGET}")

    path.write_text(rewritten, encoding="utf-8")
