"""
NM-0028 -- a field inherited a classification its own protection code disowns.

A protection code IMPLIES its tier. That is the whole reason a controlled vocabulary
exists: "USAGE" is not a label somebody writes next to a tier, it is the thing the tier is
DERIVED from. So a glossary row carrying a code and a tier that the code does not derive is
not a preference to be honoured -- it is a data defect, and the two most likely causes are
both bad:

  * the row was reclassified by hand and the code was never updated, so the code is stale
  * the code was updated and the tier was never refreshed, so the tier is stale

Either way somebody has to look. Indexing the row lets a matched field inherit a tier its
own code disowns, and the tier is the thing every downstream control keys off. Nothing
errors. Nothing warns. The index looks healthy and hands back a wrong classification --
the same silent-governance shape as NM-0005, arriving from the data side instead of the
code side.

The loader therefore REFUSES the load. Not "skips the row": skipping returns the good rows
and a caller who does not count them indexes a glossary that is quietly missing exactly the
entries whose governance was wrong.

## What this asserts

The OBSERVABLE SYMPTOM at the boundary a caller uses -- `load_entries()` raises, and no
entry comes back carrying the contradicted tier -- rather than the internals of
`problems_with()`. If the validation is restructured, this test should still be right.

## The vocabulary is FICTIONAL

Thornbury Water Authority does not exist. This library ships no taxonomy: the vocabulary
is supplied by the caller, and this file supplies one exactly as a caller would. What is
under test is the STRUCTURE -- a closed set of codes, each deriving a tier and two flags --
which is what every real catalog has.
"""

from __future__ import annotations

import pytest

from nexus_matcher.application.ingest import load_entries
from nexus_matcher.domain.governance import GovernanceVocabulary

# USAGE derives "Guarded". Pinned as a literal here, and asserted below, so the expectation
# does not come from the code under test (H-004): deriving it with `classification_for`
# would agree with an implementation that read the tier straight off the glossary row.
VOCABULARY = {
    "open_classification": "Open",
    "classes": [
        {
            "code": "USAGE",
            "name": "Metered Consumption Reading",
            "classification": "Guarded",
            "personal_information": True,
            "direct_identifier": False,
        },
        {
            "code": "PUBMAP",
            "name": "Published Network Map Reference",
            "classification": "Open",
            "personal_information": False,
            "direct_identifier": False,
        },
    ],
}

GOOD_ROW = {
    "Term": "Trunk Main Reference",
    "Definition": "Identifier of a trunk main on the published network map.",
    "Protection Class": "PUBMAP",
    "Classification": "Open",
}

# The defect, in one row: the code says USAGE, which derives "Guarded"; the row says the
# field is at the open tier. Honouring the row publishes household consumption data.
CONTRADICTING_ROW = {
    "Term": "Quarterly Reading",
    "Definition": "Volume of water drawn at a supply point over a billing quarter.",
    "Protection Class": "USAGE",
    "Classification": "Open",
}


def test_the_premise_the_code_and_the_row_really_do_disagree():
    """
    Guards this file against passing vacuously.

    If the fixture were ever edited so the row agreed with its code, every assertion below
    would be about a row that is simply valid, and the entry would become decoration that
    reports coverage it does not have.
    """
    vocabulary = GovernanceVocabulary.from_json(VOCABULARY)

    assert vocabulary.get("USAGE").classification == "Guarded"
    assert CONTRADICTING_ROW["Classification"] == "Open"
    assert CONTRADICTING_ROW["Classification"] != vocabulary.get("USAGE").classification


def test_a_row_whose_tier_contradicts_its_code_is_refused():
    """THE INVARIANT. The load raises rather than indexing a self-contradicting row."""
    with pytest.raises(ValueError) as excinfo:
        load_entries([GOOD_ROW, CONTRADICTING_ROW], governance=VOCABULARY)

    message = str(excinfo.value)
    assert "'Open'" in message and "'Guarded'" in message, (
        f"the refusal must name the tier the row stated AND the tier its code derives, "
        f"or the reader cannot tell which of the two files is wrong: {message!r}"
    )


def test_no_entry_survives_carrying_the_contradicted_tier():
    """
    The consequence, stated as the thing that must not exist.

    A loader that logged a warning and carried on would satisfy nothing here: the point is
    that no DictionaryEntry exists for a field to inherit from.
    """
    with pytest.raises(ValueError):
        load_entries([GOOD_ROW, CONTRADICTING_ROW], governance=VOCABULARY)


def test_the_softer_mode_still_refuses_to_honour_the_row():
    """
    `governance_strict=False` is an escape hatch for loading a messy glossary. It is NOT
    an instruction to believe the row. The catalog wins, so the entry carries USAGE --
    which derives "Guarded" -- and the contradiction rides along as evidence.

    Without this, "turn strict off" would become the documented way to reintroduce the
    defect, which is how a gate ends up switched off in production.
    """
    entries = load_entries(
        [GOOD_ROW, CONTRADICTING_ROW], governance=VOCABULARY, governance_strict=False
    )
    reading = next(e for e in entries if e.business_name == "Quarterly Reading")

    assert reading.governance_code == "USAGE"
    assert (
        GovernanceVocabulary.from_json(VOCABULARY).classification_for(reading.governance_code)
        == "Guarded"
    )
    assert reading.source_metadata["governance_problems"], (
        "the row was loaded with no record that anything was wrong with it"
    )


def test_a_glossary_that_agrees_with_its_own_codes_loads_normally():
    """
    The control. A loader that refused every coded row would pass all three tests above
    while making the feature unusable, and nobody would notice until a real glossary was
    pointed at it.
    """
    entries = load_entries([GOOD_ROW], governance=VOCABULARY)

    assert len(entries) == 1
    assert entries[0].governance_code == "PUBMAP"
    assert "governance_problems" not in entries[0].source_metadata
