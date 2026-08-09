"""
H-006 -- cross-lane work leaves dead or half-wired code.

Parallel work partitioned by file produces changes whose halves land in different lanes.
Three occurrences so far: a version bump that fell between two lanes, a requirements.txt
still carrying defects already fixed in pyproject.toml, and `search_batch()` shipping 2.5x
faster with NO CALLER AT ALL because its call site belonged to another lane's file.

The third is the dangerous shape, because everything looks healthy: the code is written,
tested, reviewed and committed. It simply never runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "nexus_matcher"

# Public methods that are reached other than by a literal call in src/: protocol
# implementations invoked through an interface, and names re-exported for library users.
# Each needs a REASON, so the list cannot quietly become a dumping ground.
KNOWN_EXTERNAL = {
    "search_batch": "called by match_schema._search_dense_batch via getattr probe",
}


def _defined_public_methods() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                        found.setdefault(item.name, path)
    return found


def _all_source_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))


def test_search_batch_specifically_has_a_caller():
    """
    The concrete instance, pinned. It shipped dead once; deleting the call site must fail
    a test rather than silently reverting a 2.5x optimisation back to unreachable.
    """
    text = _all_source_text()
    assert "search_batch" in text
    callers = [
        p
        for p in SRC.rglob("*.py")
        if "search_batch" in p.read_text(encoding="utf-8") and "vector_stores" not in str(p)
    ]
    assert callers, (
        "search_batch is defined but referenced nowhere outside the vector store that "
        "defines it -- it is dead code again."
    )


# Measured 2026-08-06. A RATCHET, not a target: this number may fall and may never rise.
#
# The broad sweep below finds 85 public methods defined in src/ and mentioned
# nowhere else. Most are almost certainly fine -- protocol implementations reached through
# an interface, properties, adapters for backends nobody instantiates in-tree -- and
# triaging them properly needs a call graph, not a name count.
#
# Setting the bar at today's count is the honest middle: it cannot retroactively condemn
# code nobody has reviewed, and it makes the NEXT dead symbol fail immediately, which is
# the case that actually bit (search_batch, 2.5x faster and unreachable). Drive it down;
# do not raise it. The list is in docs/DEFENSIBILITY.md.
ORPHAN_BUDGET = 85


def test_orphan_count_does_not_grow():
    """
    Coarse by construction: a name count cannot prove reachability. It catches the shape
    that actually happened -- a whole method written, tested, and wired to nothing.
    """
    text = _all_source_text()
    orphans = sorted(
        name
        for name in _defined_public_methods()
        if name not in KNOWN_EXTERNAL and text.count(name) <= 1
    )
    assert len(orphans) <= ORPHAN_BUDGET, (
        f"unreferenced public methods rose from {ORPHAN_BUDGET} to {len(orphans)}.\n"
        "New since the budget was set:\n  "
        + "\n  ".join(orphans[ORPHAN_BUDGET:])
        + "\nWire them up, delete them, or justify them in KNOWN_EXTERNAL."
    )


def test_every_known_external_entry_still_exists():
    """An allowlist entry for a method that no longer exists is silent rot."""
    defined = _defined_public_methods()
    stale = sorted(set(KNOWN_EXTERNAL) - set(defined))
    assert not stale, f"KNOWN_EXTERNAL names methods that no longer exist: {stale}"


def test_version_agrees_between_the_two_places_that_declare_it():
    """
    H-006's first occurrence: __version__ and the changelog disagreed because they sat in
    different lanes. pyproject reads the version from __init__, so those two cannot drift
    -- the changelog can.
    """
    import re

    init = (SRC / "__init__.py").read_text(encoding="utf-8")
    version = re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^##\s*\[?([0-9]+\.[0-9]+\.[0-9]+)\]?", changelog, re.M)
    assert headings, "no version headings found in CHANGELOG.md"
    assert version == headings[0], (
        f"__version__ is {version} but the newest CHANGELOG heading is {headings[0]}. "
        f"The tree would build an artifact the changelog does not describe."
    )
