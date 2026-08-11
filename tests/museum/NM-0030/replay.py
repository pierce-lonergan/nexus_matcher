"""
Reintroduce NM-0030: let `sync` forget how the index was built.

The historical shape is exactly one expression. `GlossaryIndex` held the provider and
nothing else, so `sync(index, source)` -- the call the module's own docstring shows --
re-read the glossary with no vocabulary, no column mapping and no sheet name. The refresh
loop that follows then replaced every entry object with the uncoded one, which is what made
the loss total rather than partial.

Anchored on the `load_entries` call itself, not on the comment above it or on the
`load_options` field. A comment is the most-edited line in any file (NM-0003 became a hole
that way), and a replay that deleted the field would break the constructor rather than the
behaviour, which is a different defect wearing this one's name.

## What the replay cannot reproduce, and why that is fine

Two things landed with the fix that the original defect predates: the report now carries
`governance_changed`, and `load_entries` now refuses a glossary that carries protection
codes with no vocabulary to read them. So a replayed sync no longer strips 30,000 codes in
silence -- against a source that still has its code column it fails loudly, with the wrong
error.

That is the second gate catching the first gate's defect, which is the net working. The
test therefore asserts the refusal a self-contradicting row must produce -- the same one
`load_entries` gives -- rather than "some ValueError", because "some ValueError" is
satisfied by a bypass that fails for an unrelated reason.
"""

from __future__ import annotations

import pathlib

TARGET = "src/nexus_matcher/application/ingest.py"

ANCHOR = "    entries = load_entries(source, **{**index.load_options, **kwargs})"
BROKEN = "    entries = load_entries(source, **kwargs)"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1:
        raise LookupError(
            f"NM-0030 replay: expected exactly one {ANCHOR!r} in {TARGET}, "
            f"found {text.count(ANCHOR)}"
        )
    path.write_text(text.replace(ANCHOR, BROKEN, 1), encoding="utf-8")
