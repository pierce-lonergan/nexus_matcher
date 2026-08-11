"""
NM-0029 | Confidential vocabulary reached the published wheel.

This library was written inside a bank, and a comment above DEFAULT_HIERARCHY_DATA named
the employer and the business unit whose data model that default hierarchy was sketched
from. It was in the repository for two years, in the module documentation, in an
integration test, and -- the part that mattered -- inside the wheels uploaded to PyPI for
2.0.0 and 2.0.1. A wheel on PyPI cannot be edited. Yanking hides it from resolvers and
leaves the file downloadable.

What made it survive was not that it was hard to see. It was that the rule against it was
never executable. It lived in task descriptions and in a regex somebody typed at a shell
when they happened to think of it -- scoped, each time, to whichever words were on their
mind that day. A two-year-old comment was never on anyone's mind.

This test guards the scope that made it CRITICAL rather than merely embarrassing: files
under src/, which are what gets built into the distribution. The tree-wide gate is
tests/meta/test_no_confidential_terms.py; the check on the built artifact, which is the
last chokepoint before upload, is in scripts/release_preflight.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

sys.path.insert(0, str(REPO / "tests" / "meta"))
try:
    from test_no_confidential_terms import _blocklist, scan_text
finally:
    sys.path.pop(0)

# Model artefacts vendored from HuggingFace. The BERT-uncased wordpiece vocabulary
# contains ordinary English that collides with blocked terms; it references nothing.
SKIP_NAMES = {"tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".toml", ".rst"}


def _shipped_text_files() -> list[Path]:
    return [
        p
        for p in SRC.rglob("*")
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and p.name not in SKIP_NAMES
    ]


def test_no_confidential_terms_in_shipped_source():
    """Nothing under src/ may carry a blocked term -- src/ is what becomes the wheel."""
    blocked = _blocklist()
    assert blocked, "blocklist is empty -- this test would pass vacuously"

    findings = []
    for path in _shipped_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, digest in scan_text(text, blocked):
            findings.append(f"  {path.relative_to(REPO).as_posix()}:{line_no} (digest {digest})")

    assert not findings, (
        "confidential term(s) in shipped source -- these would go into the wheel:\n"
        + "\n".join(findings)
        + "\n\nThe term is deliberately not printed. Open the line to see it."
    )


def test_default_hierarchy_is_documented_as_illustrative():
    """The hierarchy must not present itself as anyone's real taxonomy.

    Removing the attribution is only half the fix. A bare tree of banking domains with
    no framing invites the next reader to treat it as authoritative -- and invites the
    next author to 'correct' it back towards whatever internal model they know. Saying
    plainly that it is an illustrative default, and that callers should load their own,
    removes the reason anyone would reach for a real one.
    """
    source = (SRC / "nexus_matcher" / "domain" / "services" / "domain_hierarchy.py").read_text(
        encoding="utf-8"
    )
    head = source.split("DEFAULT_HIERARCHY_DATA")[0][-900:].lower()

    assert "illustrative" in head or "example" in head, (
        "DEFAULT_HIERARCHY_DATA is not framed as illustrative"
    )
    assert "from_dict" in head, "the comment does not tell callers how to supply their own"


@pytest.mark.parametrize("doc", ["docs/modules/domain_hierarchy.md"])
def test_docs_do_not_attribute_the_hierarchy(doc):
    """The same attribution was in the module docs, twice."""
    text = (REPO / doc).read_text(encoding="utf-8", errors="ignore")
    assert not scan_text(text, _blocklist()), f"{doc} carries a blocked term"
