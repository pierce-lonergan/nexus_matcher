"""
tests.unit.domain.test_field_decision | Layer: TEST
The field-level verdict: one answer per column, and a way to say "nothing matched".

## Relationships
# TESTS → domain/models/entities :: FieldDecision, derive_field_decision, MatchingSession

## What is being pinned, and why each half matters

`MatchDecision` is a per-CANDIDATE verdict and has no member meaning "the dictionary holds
nothing for this column". That is structural rather than a tuning gap: rank-1
`final_confidence` cannot fall below `semantic_weight * fusion_alpha` (0.63 shipped) while
`review_threshold` is 0.50, so rank 1 can never be REJECT on score alone and every field
comes back at least REVIEW. `tests/unit/application/test_absolute_score_floor.py` measures
that arithmetic against the shipped config rather than restating it; this file pins what
the domain does about it.

Two mistakes a fix aimed only at the symptom would make, and the tests that fail them:

  * "ship a default floor" -- a number invented for a corpus this library has never seen,
    which would start emitting NO_MATCH on somebody's glossary the day they upgrade.
    `test_no_floor_configured_never_emits_no_match_from_a_score` is that case.
  * "any REJECT is a no-match" -- REJECT is per candidate and runner-ups are routinely
    REJECT on a field whose top match is excellent.
    `test_a_rejected_rank_one_is_reject_not_no_match` is that case.

Sessions and matches here are built by hand, so each case states the confidences,
decisions and absolute scores it depends on instead of hoping a fixture produces them.
"""

from __future__ import annotations

import pytest

from nexus_matcher.domain.models.entities import (
    DictionaryEntry,
    FieldDecision,
    MatchingSession,
    MatchResult,
    Schema,
    SchemaField,
    derive_field_decision,
)
from nexus_matcher.shared.types.base import (
    DataType,
    EntityId,
    MatchDecision,
    PerformanceMetrics,
    ProtectionLevel,
    ScoreBreakdown,
)

_ENTRY = DictionaryEntry(
    id="d1",
    business_name="Ferry Departure Time",
    logical_name="ferry_departure_time",
    definition="Scheduled departure of a sailing.",
    data_type=DataType.TIMESTAMP,
    protection_level=ProtectionLevel.INTERNAL,
)

_FIELD = SchemaField(name="dep_ts", data_type=DataType.TIMESTAMP, full_path="sailing.dep_ts")


def _match(
    decision: MatchDecision,
    *,
    rank: int = 1,
    confidence: float = 0.71,
    absolute: float | None = 0.42,
) -> MatchResult:
    """One candidate with the three numbers this file's rules actually read."""
    return MatchResult(
        schema_field=_FIELD,
        dictionary_entry=_ENTRY,
        rank=rank,
        final_confidence=confidence,
        score_breakdown=ScoreBreakdown(fused_retrieval_score=0.9, absolute_cosine=absolute),
        decision=decision,
        performance=PerformanceMetrics(latency_ms=0.0),
    )


# =============================================================================
# THE VOCABULARY
# =============================================================================


class TestVocabulary:
    """`FieldDecision` is `MatchDecision` plus exactly one member."""

    def test_it_carries_every_match_decision_spelled_identically(self):
        """
        The three shared members are rank 1's own verdict passed through, so a client that
        already branches on `AUTO_APPROVE` keeps working when it reads the field-level
        value instead. Two enums whose shared members drifted apart would turn that
        passthrough into a silent relabelling, and `derive_field_decision` looks the value
        up BY VALUE, so a drifted spelling is a ValueError in production.

        Pinned name-for-name and value-for-value rather than as a set, because
        `MatchDecision.REVIEW` mapping onto a `FieldDecision.REVIEW` whose value read
        `"review"` would pass a name-only check and put a lowercase verdict on the wire.
        """
        for member in MatchDecision:
            assert member.name in FieldDecision.__members__, (
                f"MatchDecision.{member.name} has no FieldDecision counterpart, so "
                f"derive_field_decision() would raise on a field that matched"
            )
            assert FieldDecision[member.name].value == member.value

    def test_it_adds_exactly_one_member_and_that_member_is_no_match(self):
        """
        The extra member is the whole point, and it is the ONLY extra member. A second one
        would be a field-level state this file's rules never produce, published to a
        generated client as though it might arrive.
        """
        extra = set(FieldDecision.__members__) - set(MatchDecision.__members__)
        assert extra == {"NO_MATCH"}
        assert FieldDecision.NO_MATCH.value == "NO_MATCH"

    def test_match_decision_itself_still_cannot_say_no_match(self):
        """
        The reason this enum exists at all. If `MatchDecision` ever gains the member, the
        per-candidate `decision` on the wire gains a value a Java client generated last
        week cannot deserialise -- so that change is a deliberate one, and this test is
        where it gets noticed rather than in a consumer's exception handler.
        """
        assert "NO_MATCH" not in MatchDecision.__members__


# =============================================================================
# THE DERIVATION
# =============================================================================


class TestDerivation:
    """`derive_field_decision` -- the one roll-up rule, so nobody writes a second one."""

    def test_a_field_with_no_candidates_is_no_match(self):
        """
        Needs no floor and no calibration: the result list is empty, so there is nothing
        to inherit from and saying so invents nothing. This is the only NO_MATCH a caller
        who configured nothing can ever see.
        """
        assert derive_field_decision(()) is FieldDecision.NO_MATCH
        assert derive_field_decision((), absolute_score_floor=None) is FieldDecision.NO_MATCH

    @pytest.mark.parametrize(
        "decision",
        [MatchDecision.AUTO_APPROVE, MatchDecision.REVIEW, MatchDecision.REJECT],
    )
    def test_rank_one_decides_and_is_passed_through_unchanged(self, decision):
        matches = (_match(decision), _match(MatchDecision.REJECT, rank=2, confidence=0.2))
        assert derive_field_decision(matches).value == decision.value

    def test_runner_ups_do_not_get_a_vote(self):
        """
        REJECT is per candidate: on the shipped fixtures most runner-ups are REJECT while
        the top match is fine. A roll-up that looked at the whole list -- "any REJECT means
        trouble" -- would flag nearly every field on nearly every schema, which is the same
        non-answer as flagging none.
        """
        matches = (
            _match(MatchDecision.AUTO_APPROVE, confidence=0.93),
            _match(MatchDecision.REJECT, rank=2, confidence=0.31),
            _match(MatchDecision.REJECT, rank=3, confidence=0.22),
        )
        assert derive_field_decision(matches) is FieldDecision.AUTO_APPROVE

    def test_a_rejected_rank_one_is_reject_not_no_match(self):
        """
        The two are different claims. REJECT says the top candidate did not clear the
        review bar; NO_MATCH says nothing in the dictionary describes this column. A
        consumer routes them differently -- REJECT to a reviewer with a candidate to look
        at, NO_MATCH to whoever writes new glossary terms.
        """
        assert derive_field_decision((_match(MatchDecision.REJECT),)) is FieldDecision.REJECT

    def test_no_floor_configured_never_emits_no_match_from_a_score(self):
        """
        THE DEFAULT MUST NOT INVENT A FLOOR. This candidate's absolute score is 0.02 --
        near-orthogonal, the shape of a field nothing describes -- and with no floor
        configured the verdict is still rank 1's own REVIEW. Emitting NO_MATCH here would
        mean the library had picked a cut point for a corpus it has never seen, and every
        deployment that upgraded would start unclassifying fields on a number nobody chose.
        """
        matches = (_match(MatchDecision.REVIEW, confidence=0.71, absolute=0.02),)
        assert derive_field_decision(matches) is FieldDecision.REVIEW
        assert derive_field_decision(matches, absolute_score_floor=None) is FieldDecision.REVIEW

    def test_a_configured_floor_turns_a_high_confidence_field_into_no_match(self):
        """
        The case the whole feature exists for, in one assertion.

        Confidence 0.71 is above the 0.63 structural floor AND above the 0.50 review
        threshold, so nothing in the per-candidate vocabulary can express doubt about it.
        The absolute score is 0.02. With a floor of 0.30 the field reads NO_MATCH, and the
        SAME candidates read REVIEW without one -- so the verdict moved because the caller
        made a calibration decision, not because the library did.
        """
        matches = (_match(MatchDecision.REVIEW, confidence=0.71, absolute=0.02),)
        assert derive_field_decision(matches, absolute_score_floor=0.30) is FieldDecision.NO_MATCH

    def test_a_candidate_that_clears_the_floor_keeps_its_own_verdict(self):
        matches = (_match(MatchDecision.AUTO_APPROVE, confidence=0.93, absolute=0.87),)
        assert (
            derive_field_decision(matches, absolute_score_floor=0.30) is FieldDecision.AUTO_APPROVE
        )

    def test_the_floor_is_inclusive_at_its_own_value(self):
        """
        `>= floor` clears. A floor is "do not trust anything BELOW this", and a candidate
        sitting exactly on it is not below it. Pinned because the boundary is the one place
        a floor silently means one candidate more or less than the caller measured for.
        """
        matches = (_match(MatchDecision.REVIEW, absolute=0.30),)
        assert derive_field_decision(matches, absolute_score_floor=0.30) is FieldDecision.REVIEW

    def test_a_floor_of_zero_is_a_floor_and_not_off(self):
        """
        `None` means off; `0.0` means a floor at zero. A truthiness test would collapse the
        two and silently disable a caller's deliberate configuration -- and 0.0 is a
        defensible floor, because a candidate at or below zero similarity is not a match
        under any metric that returns one.
        """
        assert (
            derive_field_decision((_match(MatchDecision.REVIEW, absolute=-0.01),), 0.0)
            is FieldDecision.NO_MATCH
        )
        assert (
            derive_field_decision((_match(MatchDecision.REVIEW, absolute=-0.01),), None)
            is FieldDecision.REVIEW
        )

    def test_a_candidate_with_no_absolute_score_cannot_clear_a_floor(self):
        """
        `absolute_cosine` is None when the dense arm never returned this candidate -- it
        reached the shortlist through the lexical arm alone. That is NOT zero, and it is
        not evidence of similarity either.

        A caller who sets a floor is saying "I do not trust a match below this". Clearing
        the floor on a candidate whose absolute similarity was never measured would grant
        exactly the trust they withheld, and the cost of the two errors is not symmetric:
        a wrong NO_MATCH sends a field to a human, a wrong pass ships a classification.
        """
        matches = (_match(MatchDecision.AUTO_APPROVE, confidence=0.93, absolute=None),)
        assert derive_field_decision(matches, absolute_score_floor=0.30) is FieldDecision.NO_MATCH
        # ...and with no floor configured the same candidate is untouched, so the rule
        # above cannot leak into the default configuration.
        assert derive_field_decision(matches) is FieldDecision.AUTO_APPROVE


# =============================================================================
# THE SESSION SURFACE
# =============================================================================


def _session(
    results: dict[str, tuple[MatchResult, ...]],
    floor: float | None = None,
) -> MatchingSession:
    return MatchingSession(
        session_id=EntityId(),
        schema=Schema(name="sailings", fields=(_FIELD,)),
        results=results,
        total_duration_ms=1.0,
        absolute_score_floor=floor,
    )


class TestSessionFieldDecisions:
    """A library caller gets the same verdict the HTTP caller does, from the same rule."""

    def test_one_verdict_per_result_key_in_schema_order(self):
        session = _session(
            {
                "a": (_match(MatchDecision.AUTO_APPROVE),),
                "b": (_match(MatchDecision.REVIEW),),
                "c": (),
            }
        )
        assert list(session.field_decisions()) == ["a", "b", "c"]

    def test_a_field_nothing_matched_is_present_and_no_match(self):
        """
        Present, not dropped. A field missing from this map inherits nothing while nothing
        says so, which is the conservation failure this package keeps a museum entry for.
        """
        decisions = _session({"a": (_match(MatchDecision.REVIEW),), "b": ()}).field_decisions()
        assert decisions == {"a": FieldDecision.REVIEW, "b": FieldDecision.NO_MATCH}

    def test_the_sessions_own_floor_is_the_one_applied(self):
        """
        The session carries the floor it was matched under, so two sessions from two
        differently configured matchers do not have to be told apart by their caller.
        """
        results = {"a": (_match(MatchDecision.REVIEW, confidence=0.71, absolute=0.02),)}
        assert _session(results).field_decisions() == {"a": FieldDecision.REVIEW}
        assert _session(results, floor=0.30).field_decisions() == {"a": FieldDecision.NO_MATCH}

    def test_the_default_session_has_no_floor(self):
        """A session built by hand -- every existing caller and test -- keeps no floor."""
        assert _session({}).absolute_score_floor is None
