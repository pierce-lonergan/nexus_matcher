"""
NM-0031 -- one alias claimed by two classes silently downgraded a restricted field to the
open tier.

A catalog can map a legacy spelling onto a class, so a glossary that still says
`PAX_NM_OLD` keeps resolving after the code was renamed. Duplicate *codes* have always
raised. Duplicate *aliases* were written into a dict and the last declaration won,
positionally -- so a catalog where the restricted class and the open class both claim one
legacy spelling resolved it to whichever happened to be written second.

Measured: the passenger-name row loaded carrying `TIDE_TABLE` / `OPEN_DECK`,
`problems_with()` returned nothing, the strict load was not refused, and the classification
the caller applied came back as the open tier for a direct identifier. That is NM-0005's
harm -- a field losing the class it should have inherited -- reached through the CATALOG
rather than through the row.

## Why an `__init__` check alone was not the fix

It was implemented and measured, and it missed the likeliest duplicate of all: two
byte-identical spellings. `from_json` accumulated aliases into a dict keyed by the raw
token, so `"PAX_NM_OLD"` written twice collapsed to one entry before any check downstream
could see two. A test written to "duplicate aliases raise" would have passed while the
defect it was written for survived, which is why the byte-identical case is asserted here
by name rather than left to the punctuation-differing one.

## The direction that is easy to miss

The same silence runs the other way. A top-level `"n/a": null` says "this token means no
class, drop it"; a class also claiming `n/a` promoted that token into a real class. The
module's own docstring promises that "we quietly dropped something" cannot happen
unnoticed, and this was the shape that made it happen.

## The vocabulary is FICTIONAL

The Gravel Bay Ferry Authority does not exist and neither does this taxonomy; it is the
example pack this repository ships. This library holds no opinion about anyone's tiers --
the vocabulary is supplied by the caller, and this file supplies one exactly as a caller
would. What is under test is the STRUCTURE: a closed set of codes, each deriving a tier.
"""

from __future__ import annotations

import copy

import pytest

from nexus_matcher.application.ingest import load_entries
from nexus_matcher.domain.governance import GovernanceVocabulary

# MANIFEST_NAME is a direct identifier at the most restricted tier; TIDE_TABLE is published
# sailing data at the open one. Pinned as literals and asserted below, so the expectation
# does not come from the code under test.
RESTRICTED = {
    "code": "MANIFEST_NAME",
    "name": "Passenger manifest identity",
    "classification": "SEALED_RESTRICTED",
    "personal_information": True,
    "direct_identifier": True,
}
OPEN = {
    "code": "TIDE_TABLE",
    "name": "Published sailing and tide data",
    "classification": "OPEN_DECK",
    "personal_information": False,
    "direct_identifier": False,
}

# The legacy spelling a renamed code leaves behind in a glossary nobody has re-exported.
LEGACY = "PAX_NM_OLD"

PASSENGER_ROW = {
    "Term": "Passenger Legal Name",
    "Definition": "Name of the passenger as printed on the sailing manifest.",
    "Protection Class": LEGACY,
}


def vocabulary(*, restricted_aliases=(), open_aliases=(), top_level=None) -> dict:
    """The two-class catalog, with aliases placed where a given flavour needs them."""
    restricted = {**copy.deepcopy(RESTRICTED), "aliases": list(restricted_aliases)}
    open_class = {**copy.deepcopy(OPEN), "aliases": list(open_aliases)}
    document: dict = {
        "open_classification": "OPEN_DECK",
        "classes": [restricted, open_class],
    }
    if top_level is not None:
        document["aliases"] = top_level
    return document


def test_the_premise_one_class_claiming_the_legacy_spelling_resolves_to_it():
    """
    Guards this file against passing vacuously.

    Every assertion below is about what happens when TWO classes claim one alias. If
    aliases stopped resolving at all, "the wrong class did not win" would be true of
    nothing, and this entry would report coverage it does not have.
    """
    catalog = GovernanceVocabulary.from_json(vocabulary(restricted_aliases=[LEGACY]))

    assert catalog.classification_for(LEGACY) == "SEALED_RESTRICTED"
    assert catalog.get(LEGACY).direct_identifier is True


def test_a_row_carrying_the_contested_alias_never_loads_at_the_open_tier():
    """
    THE DEFECT, at the boundary a caller actually uses.

    Measured before the fix: this exact row loaded with `governance_code == 'TIDE_TABLE'`,
    `source_metadata` carrying only the raw code, and no `governance_problems` at all --
    under `governance_strict`, which is the setting whose entire job is to refuse a
    glossary whose governance is wrong. The refusal is the observable behaviour, so the
    assertion is that no entry survives carrying the open tier, not merely that something
    raised.
    """
    contested = vocabulary(restricted_aliases=[LEGACY], open_aliases=["PAX-NM-OLD"])

    with pytest.raises(ValueError) as refusal:
        load_entries([PASSENGER_ROW], governance=contested)

    # Both spellings as the caller wrote them: a normalised key is not searchable in their
    # own file, and "declared twice" is useless advice if it does not say where.
    message = str(refusal.value)
    assert LEGACY in message and "PAX-NM-OLD" in message
    assert "MANIFEST_NAME" in message and "TIDE_TABLE" in message


def test_the_byte_identical_duplicate_is_refused_too():
    """
    The case an `__init__`-only fix misses, and the likeliest one to be written by hand.

    Two classes claiming the SAME string, not two spellings of it. `from_json` used to key
    aliases by the raw token, so these collapsed into one entry before anything downstream
    could count two -- a check placed after that point sees a clean catalog and a test
    written to it goes green over a live defect.
    """
    with pytest.raises(ValueError, match="declared twice"):
        GovernanceVocabulary.from_json(
            vocabulary(restricted_aliases=[LEGACY], open_aliases=[LEGACY])
        )


def test_a_token_declared_as_drop_this_cannot_be_promoted_into_a_class():
    """
    The other direction, and the one that contradicts the module's own promise.

    `{"n/a": null}` at the top level says the token means no class. A class also claiming
    `n/a` overwrote that, so a row that stated it had no protection code came back
    carrying a restricted class -- the inverse harm, and reached the same way.
    """
    with pytest.raises(ValueError, match="declared twice"):
        GovernanceVocabulary.from_json(
            vocabulary(restricted_aliases=["n/a"], top_level={"n/a": None})
        )


def test_restating_the_same_mapping_still_loads():
    """
    The control. A rule that could not tell a restatement from a conflict would be worked
    around by deleting the restatement -- which is the line documenting the intent -- and
    it would refuse catalogs that were never ambiguous, including a pack that names one
    legacy spelling at the top level and again on the class it belongs to.
    """
    catalog = GovernanceVocabulary.from_json(
        vocabulary(restricted_aliases=[LEGACY], top_level={LEGACY: "MANIFEST_NAME"})
    )

    assert catalog.classification_for(LEGACY) == "SEALED_RESTRICTED"

    entries = load_entries([PASSENGER_ROW], governance=catalog)
    assert [entry.governance_code for entry in entries] == ["MANIFEST_NAME"]
