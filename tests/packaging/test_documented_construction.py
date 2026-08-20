"""
tests.packaging.test_documented_construction | Layer: GATE
The constructor signatures three documents write down, checked against the real ones.

Why this exists
---------------
`docs/API_REFERENCE.md` prints `NexusMatcher(...)` and `NexusMatcher.from_config(...)` as
literal signatures, and `docs/guides/governed_abbreviations.md` builds its whole argument on
a NEGATIVE claim about one of them: *"`from_config` does not take `abbreviation_expander`,
so supplying your own approved-abbreviation catalog requires the full constructor."*

A negative claim about an interface is the exact shape `test_documented_routes.py` was
written for -- there, a document saying "there is no HTTP matching endpoint" while the
endpoint existed. The failure is asymmetric and quiet in both directions:

  * if `abbreviation_expander` is later ADDED to `from_config`, the guide keeps sending
    readers down a twenty-line hand-assembly they no longer need, and nothing goes red;
  * if `governance` is REMOVED from `__init__`, API_REFERENCE's construction block keeps
    advertising a parameter that raises `TypeError`, and nothing goes red either.

Both were live risks when this file was written: `governance` had already been added to
`__init__` and `from_config` without API_REFERENCE noticing, which is how the stale block
that prompted this test got there.

What this cannot do
-------------------
It checks PARAMETER NAMES, not types, defaults, order or semantics. A parameter that keeps
its name and changes meaning passes. It also does not read the prose around the signature --
only that the sentence carrying the negative claim is still present, so deleting the claim
silently is caught but rewording it in a way this file does not recognise is not.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from nexus_matcher.application.use_cases.match_schema import NexusMatcher

REPO = Path(__file__).resolve().parents[2]
API_REFERENCE = REPO / "docs" / "API_REFERENCE.md"
GUIDE = REPO / "docs" / "guides" / "governed_abbreviations.md"


def _parameters(func) -> set[str]:
    return {name for name in inspect.signature(func).parameters if name != "self"}


# =============================================================================
# THE SIGNATURES
# =============================================================================


@pytest.mark.parametrize(
    "name",
    [
        "embedding_provider",
        "vector_store",
        "sparse_retriever",
        "reranker",
        "schema_parser_registry",
        "dictionary_loader_registry",
        "abbreviation_expander",
        "context_enricher",
        "domain_matcher",
        "config",
        "governance",
    ],
)
def test_api_reference_construction_block_names_a_real_parameter(name):
    """Every parameter API_REFERENCE prints in the `NexusMatcher(...)` block exists."""
    assert name in _parameters(NexusMatcher.__init__), (
        f"docs/API_REFERENCE.md advertises `{name}` in the NexusMatcher(...) construction "
        f"block, but __init__ does not take it. Passing it raises TypeError."
    )


def test_api_reference_construction_block_is_not_missing_a_parameter():
    """
    The other direction. `governance` was added to `__init__` and the block was not
    updated; that is the defect this half catches.
    """
    documented = set()
    text = API_REFERENCE.read_text(encoding="utf-8")
    block = text.split("NexusMatcher(", 1)[1].split(")\n```", 1)[0]
    for line in block.splitlines():
        candidate = line.strip().split("#", 1)[0].strip().rstrip(",").split("=", 1)[0].strip()
        if candidate:
            documented.add(candidate)
    missing = _parameters(NexusMatcher.__init__) - documented
    assert not missing, (
        f"NexusMatcher.__init__ takes {sorted(missing)}, which docs/API_REFERENCE.md's "
        f"construction block does not list."
    )


def test_from_config_still_cannot_take_an_abbreviation_expander():
    """
    The negative claim `docs/guides/governed_abbreviations.md` is built on.

    If this fails because `from_config` grew the parameter, that is GOOD NEWS and the fix
    is to rewrite the guide's 'from_config cannot reach this feature' section and
    API_REFERENCE's callout -- not to revert the code.
    """
    assert "abbreviation_expander" not in _parameters(NexusMatcher.from_config), (
        "from_config now accepts `abbreviation_expander`. "
        "docs/guides/governed_abbreviations.md and docs/API_REFERENCE.md both state that "
        "it does not, and send the reader to the full constructor because of it. "
        "Update both documents."
    )


def test_from_config_takes_exactly_what_api_reference_prints():
    assert _parameters(NexusMatcher.from_config) == {"config", "governance"}, (
        "docs/API_REFERENCE.md prints from_config(config, governance); the real signature "
        f"is {sorted(_parameters(NexusMatcher.from_config))}."
    )


# =============================================================================
# THE SENTENCE
# =============================================================================
# Deleting a claim is a way of making a gate green that is indistinguishable from fixing
# it, unless the gate also requires the claim to be there. Same call
# `test_documented_behaviour.py` makes for the REJECT rule in docs/GOVERNANCE.md.


def test_the_guide_still_states_the_from_config_limitation():
    text = GUIDE.read_text(encoding="utf-8")
    assert "from_config" in text and "abbreviation_expander" in text, (
        "docs/guides/governed_abbreviations.md no longer mentions the from_config "
        "limitation. If from_config can now take an expander, say so; do not drop it."
    )


def test_the_guide_documents_the_wiring_it_claims_is_plumbed():
    """
    The guide's headline is that the capability already ships. The two public names that
    make it true must appear in the page a reader is sent to.
    """
    text = GUIDE.read_text(encoding="utf-8")
    for symbol in ("AbbreviationExpander", "AbbreviationDictionary.from_dict"):
        assert symbol in text, f"the guide no longer shows `{symbol}`"
