"""
tests.unit.presentation.api.test_governance_contract | Layer: TEST
The coupling between this HTTP layer and the domain's governance types.

## Relationships
# TESTS → presentation/api/matching :: _governance_id and _governance_payload
# TESTS → domain/governance :: ProtectionClass, read by name
# TESTS → domain/models/entities :: MatchResult.governance / .governance_id

This layer reads `MatchResult.governance_id`, `MatchResult.governance`, and six
attributes on `ProtectionClass`, BY NAME. Names are exactly what a parallel lane renames,
and H-006 is the ledger entry for changes whose two halves land in different files --
three occurrences so far, the most expensive of which shipped 2.5x faster and unreachable.

So the names are pinned here as literals rather than read back off the objects. An
expectation derived from the thing it is checking is an identity, and an identity holds
just as well when both sides are wrong.

## Why the boundary re-checks what the domain already guarantees

`MatchResult.__post_init__` now fills a blank `governance_id` from the entry and refuses a
contradicting one, so the refusals below cannot be triggered by a real result any more.
They are kept, and tested through a deliberately malformed stand-in, because the promise
"`governanceId` is always populated" is made by THIS RESPONSE to an external caller. A
promise that holds only because of an invariant one layer down is a promise that a future
refactor of that layer can retract without anything on this side noticing.
"""

from __future__ import annotations

import dataclasses

import pytest

from nexus_matcher.domain.governance import ProtectionClass
from nexus_matcher.domain.models.entities import SchemaField
from nexus_matcher.presentation.api.errors import ContractDriftError
from nexus_matcher.presentation.api.matching import _governance_id, _governance_payload
from nexus_matcher.shared.types.base import DataType, MatchDecision, PerformanceMetrics
from tests.unit.presentation.api._support import (
    FICTIONAL_VOCABULARY,
    GLOSSARY,
    MalformedMatch,
    governed_match,
)

# The attribute names this layer reads off a protection class, and the response key each
# one becomes. Literals, so a rename in the domain lane fails HERE rather than turning
# into a silently missing member of a governance payload.
CONTRACT: tuple[tuple[str, str], ...] = (
    ("code", "code"),
    ("name", "name"),
    ("classification", "classification"),
    ("personal_information", "personalInformation"),
    ("direct_identifier", "directIdentifier"),
    # Read by name like the five above, but NOT required like them -- see
    # `test_an_absent_enhancement_is_a_null_and_not_a_refusal`. It is here because a rename
    # in the domain lane must fail this file, and because losing it silently is the same
    # class of defect as losing any other member: the caller stops being told how to
    # protect the field and nothing says so.
    ("enhancement", "enhancement"),
)

FIELD = SchemaField(name="resident_nm", data_type=DataType.STRING, full_path="a.resident_nm")


def test_protection_class_still_declares_every_attribute_this_layer_reads():
    """
    The whole coupling, in one assertion.

    Dropping or renaming any of these five would not break the endpoint -- `getattr` would
    simply stop finding it -- so without this test the failure mode is a governance payload
    that quietly loses a member, which is the single thing this response must never do.
    """
    declared = {f.name for f in dataclasses.fields(ProtectionClass)}
    missing = sorted({attribute for attribute, _key in CONTRACT} - declared)
    assert not missing, (
        f"domain.governance.ProtectionClass no longer declares {missing}. The HTTP "
        f"governance payload reads those names and would lose a member of the "
        f"classification it exists to carry."
    )


def test_the_match_result_still_carries_the_two_fields_the_endpoint_promotes():
    """
    `governanceId` and `governance` are the two things a caller matches a field IN ORDER
    to get. If either stopped being a field on MatchResult, this layer would fall back to
    the dictionary entry (for the id) or to null (for the class) and keep answering 200 --
    a correct-looking response that has stopped carrying the answer.
    """
    declared = {f.name for f in dataclasses.fields(GLOSSARY[0].__class__)}
    assert "governance_code" in declared

    match = governed_match(FIELD, GLOSSARY[0])
    assert match.governance_id == GLOSSARY[0].id
    assert match.governance is not None


def test_the_classification_tier_is_carried_through_untouched():
    """
    The tier is DERIVED from the code by whoever owns the vocabulary. This layer must pass
    it through verbatim and must not normalise, title-case or map it: a response that
    rewrote a caller's own tier string would be wrong in the most confusing way available,
    because it would still look like their taxonomy.
    """
    for code, expected in FICTIONAL_VOCABULARY.items():
        entry = next(e for e in GLOSSARY if e.governance_code == code)
        payload = _governance_payload(governed_match(FIELD, entry))

        assert payload is not None
        assert tuple(payload) == tuple(key for _attribute, key in CONTRACT)
        assert payload["code"] == code
        assert payload["classification"] == expected.classification
        assert payload["personalInformation"] is expected.personal_information
        assert payload["directIdentifier"] is expected.direct_identifier
        assert payload["enhancement"] == expected.enhancement


def test_the_enhancement_crosses_the_wire_in_both_of_its_declared_states():
    """
    A declared instruction and a declared null, from the same fixture.

    `enhancement` is the only member of a protection class that says what to DO with the
    field rather than what the field is, and it reached this layer on every result and was
    then dropped -- so the caller deciding whether to mask or tokenise had to read a file on
    the server. Both states are pinned because the null one is what makes it optional: four
    of the fixture's classes declare an instruction and one declares null, exactly as five
    of the nine classes in the shipped example pack do.
    """
    instructed = next(e for e in GLOSSARY if e.governance_code == "RESIDENT")
    declared_none = next(e for e in GLOSSARY if e.governance_code == "OUTAGENOTE")

    assert _governance_payload(governed_match(FIELD, instructed))["enhancement"] == "MASK_IN_LOGS"
    assert _governance_payload(governed_match(FIELD, declared_none))["enhancement"] is None


def test_an_entry_with_no_code_yields_no_class():
    uncoded = next(e for e in GLOSSARY if e.governance_code is None)
    assert _governance_payload(governed_match(FIELD, uncoded)) is None


def test_the_governance_id_is_the_dictionary_entry_id():
    """The contract's own wording: the entry id IS the governance id."""
    assert _governance_id(governed_match(FIELD, GLOSSARY[0])) == GLOSSARY[0].id


def _malformed(**overrides) -> MalformedMatch:
    base = governed_match(FIELD, GLOSSARY[0])
    fields = {
        "schema_field": base.schema_field,
        "dictionary_entry": base.dictionary_entry,
        "rank": base.rank,
        "final_confidence": base.final_confidence,
        "score_breakdown": base.score_breakdown,
        "decision": MatchDecision.REVIEW,
        "performance": PerformanceMetrics(latency_ms=1.0),
        "governance_id": base.governance_id,
        "governance": base.governance,
    }
    fields.update(overrides)
    return MalformedMatch(**fields)


def test_an_empty_governance_id_is_refused_rather_than_emitted():
    """
    Present-but-blank is NM-0005 wearing a value instead of a missing key: the caller is
    told the field's governance id is the empty string, applies nothing, and nothing
    raises.
    """
    with pytest.raises(ContractDriftError, match="governance_id"):
        _governance_id(_malformed(governance_id=""))


def test_a_partially_populated_protection_class_is_refused():
    """
    Defaulting a missing member would answer `personalInformation: false` to the one
    question the caller asked. Refusing the whole response is the cheaper mistake.
    """

    class _HalfBuilt:
        code = "RESIDENT"
        name = "Resident Name"
        classification = "LUMENPORT_GUARDED"
        personal_information = True
        # direct_identifier deliberately absent

    with pytest.raises(ContractDriftError, match="direct_identifier"):
        _governance_payload(_malformed(governance=_HalfBuilt()))


def test_an_absent_enhancement_is_a_null_and_not_a_refusal():
    """
    The asymmetry with the five required members, stated as a test.

    A class object that never declares `enhancement` at all -- a caller's own
    ProtectionClass-shaped adapter, or a domain lane that has not added the attribute --
    must produce an explicit null, not the `ContractDriftError` that a missing
    `direct_identifier` produces one test up. Null is a DOCUMENTED value for this member
    and a defect for the other five, so `getattr(..., None)` is the right read here and the
    `drift()` path is not: routing it through `drift()` would turn the five classes in the
    shipped example pack that declare `"enhancement": null` into a 500.
    """

    class _NoEnhancement:
        code = "RESIDENT"
        name = "Resident Name"
        classification = "LUMENPORT_GUARDED"
        personal_information = True
        direct_identifier = True

    payload = _governance_payload(_malformed(governance=_NoEnhancement()))
    assert payload is not None
    assert payload["enhancement"] is None


def test_a_result_with_no_governance_attributes_at_all_still_yields_an_id():
    """
    The degradation path, kept live.

    An object that is MatchResult-shaped but predates the governance fields -- a caller's
    own adapter, a mock in somebody else's test suite -- must still produce a populated
    `governanceId` from the entry, and an explicit null class. Silently emitting an empty
    id would be worse than either.
    """

    class _NoGovernanceFields:
        dictionary_entry = GLOSSARY[0]

    assert _governance_id(_NoGovernanceFields()) == GLOSSARY[0].id
    assert _governance_payload(_NoGovernanceFields()) is None
