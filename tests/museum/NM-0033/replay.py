"""Reintroduce NM-0033: let the load path index whatever the loader handed it.

The historical state is exactly this. `load_dictionary` called the loader, took the
entries, and indexed them; the loader's port has no field for a protection code, so every
entry arrived with `governance_code=None` and the vocabulary the matcher was configured
with was never consulted at load time at all. Emptying `_attach_governance` down to
`return list(entries)` restores that, and restores all three of its consequences at once
-- no codes, no derivation check on this path, and no refusal for a glossary that carries
codes nobody is configured to read.

Anchored on the METHOD SIGNATURE, deliberately, and on none of the following:

  * the call site in `load_dictionary`. It is one statement under 100 characters today
    and the formatter will wrap it the moment a parameter name grows, which is the class
    of rot that turned NM-0016 into a hole (it pinned `batch_size: int = 64`, someone
    tuned it to 512, and the entry reported PASS while catching nothing).
  * any comment. The most-edited line in any file; NM-0003 became a hole that way.
  * the docstring, which states the measured cost of the second read and is expected to
    be re-measured.

The signature carries no default argument, so there is no tunable in it to rot. Renaming
the method breaks the anchor LOUDLY -- `museum_replay.py` reports a hole rather than a
pass -- which is the correct failure for an anchor that no longer describes the code.

Inserting the body immediately after the signature leaves the real docstring in place as
the first statement after the injected `return`, so nothing below it runs. That is the
same shape NM-0016 uses and it keeps the diff to one inserted block.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/application/use_cases/match_schema.py"

SIGNATURE_RE = re.compile(
    r"^    def _attach_governance\(\n(?:.*\n)*?    \) -> list\[DictionaryEntry\]:\n",
    re.M,
)

BODY = (
    "        # NM-0033 replay: index what the loader produced. The loader has no\n"
    "        # protection code to give, so nothing is attached and nothing is checked.\n"
    "        return list(entries)\n"
)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    matches = list(SIGNATURE_RE.finditer(text))
    if len(matches) != 1:
        raise LookupError(
            f"NM-0033 replay: expected exactly one _attach_governance signature in "
            f"{TARGET}, found {len(matches)}"
        )
    end = matches[0].end()
    path.write_text(text[:end] + BODY + text[end:], encoding="utf-8")
