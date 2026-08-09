"""
Reintroduce NM-0004: disconnect --output-dir from everything again.

The declaration, the help text and the docstring example all stay exactly as they are --
that is the whole character of this defect. Only the one branch that consumes the option
is switched off, so `sync dict.csv -o ./index` goes back to reporting success and writing
nothing.

Anchored on the guard line by regex with the indentation captured, so the block can move
between nesting levels without the replay rotting. The branch BODY is untouched: leaving
it in place (dead) rather than deleting it keeps this replay small, and a small replay is
one with fewer ways to stop matching.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/presentation/cli/main.py"

GUARD_RE = re.compile(r"^(?P<indent>[ \t]+)if output_dir is not None:[ \t]*$", re.M)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")

    rewritten, count = GUARD_RE.subn(
        # `if False:` rather than deleting the block, so nothing below it has to be
        # re-indented -- the mutation is one token wide and cannot corrupt the file.
        r"\g<indent>if False:  # NM-0004 replay: --output-dir consumed by nobody",
        text,
        count=1,
    )
    if count != 1:
        raise LookupError(f"NM-0004 replay: no `if output_dir is not None:` branch in {TARGET}")

    path.write_text(rewritten, encoding="utf-8")
