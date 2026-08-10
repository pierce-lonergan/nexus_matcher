"""
Reintroduce NM-0003: make --output unreachable again at the default format.

Two mutations, because the defect had two causes and either one alone still loses the
user's file:

  * `_resolve_format` stops inferring anything from the output path, so an unset
    `--format` means "table" no matter what `-o results.json` said;
  * the table branch stops writing, so resolving to "table" means writing nothing.

Anchored on the `_resolve_format` SIGNATURE and on the `_write_output(...)` CALL rather
than on any surrounding prose. The docstrings in this file carry the explanation of the
defect and are the most likely thing in it to be reworded.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/presentation/cli/main.py"

RESOLVE_RE = re.compile(r"^def _resolve_format\([^)]*\) -> str:\n", re.M)
RESOLVE_BODY = (
    "    # NM-0003 replay: --format defaults to table and never consults --output.\n"
    '    return requested if requested is not None else "table"\n'
)

# The table branch's write-through: the CALL ITSELF, and nothing around it.
#
# This used to also require `if output:` on the immediately preceding line. Someone added
# an explanatory comment between the two, the anchor stopped matching, and the entry
# reported itself as a hole -- while the docstring above claimed it was anchored on the
# call rather than on surrounding prose. Now that claim is true.
#
# Replacing the write with a console print reproduces the defect exactly -- a `-o` path
# that renders to the terminal and produces no file -- and leaves valid syntax, which
# deleting the line would not.
TABLE_WRITE_RE = re.compile(r"_write_output\(output, _render_table_text\([^\n]*\)\)")
TABLE_WRITE_BROKEN = "console.print(table)  # NM-0003 replay: writes nothing"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")

    match = RESOLVE_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0003 replay: _resolve_format signature not found in {TARGET}")
    text = text[: match.end()] + RESOLVE_BODY + text[match.end() :]

    text, removed = TABLE_WRITE_RE.subn(TABLE_WRITE_BROKEN, text, count=1)
    if removed != 1:
        raise LookupError(
            f"NM-0003 replay: the table branch no longer writes to --output in {TARGET}; "
            f"nothing left to take away"
        )

    path.write_text(text, encoding="utf-8")
