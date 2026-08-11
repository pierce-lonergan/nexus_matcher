"""
Reintroduce NM-0028: let the loader accept a row whose tier contradicts its own code.

The derivation check is DELETED rather than inverted. Inverting it would produce a
different, louder defect -- correct rows suddenly rejected -- and every test in the repo
would notice. The historical shape is the permissive one: `problems_with()` simply has
nothing to say about the tier, so the row sails through `load_entries()` and is indexed
with a classification its own code disowns. Nothing errors, nothing warns.

Anchored on the whole `if columns.classification is not None:` block. Matching only the
comparison line would leave a dangling `problems.append(...)`; matching the whole block
means that if the code is ever restructured the anchor stops matching and the runner
reports a HOLE, rather than a replay that silently no-ops into a false PASS.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/domain/governance.py"

# From the `if columns.classification` line up to (not including) the flag loop that
# follows it.
BLOCK_RE = re.compile(
    r"^        if columns\.classification is not None:\n(?:.*?\n)*?(?=^        for column,)",
    re.M,
)

BROKEN = """        if columns.classification is not None:
            # The tier is whatever the row says it is.
            pass

"""


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = BLOCK_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0028 replay: the derivation check was not found in {TARGET}")
    path.write_text(text[: match.start()] + BROKEN + text[match.end() :], encoding="utf-8")
