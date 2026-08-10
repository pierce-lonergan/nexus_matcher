"""
tests.unit.domain.test_low_confidence_fields | Layer: TEST
What `MatchingSession.get_low_confidence_fields()` is allowed to mean.

The museum entry (NM-0027) pins the escaped symptom: the 0.6 default could not select
anything, so the review queue was always empty. This file pins the REPLACEMENT contract,
and specifically the two mistakes that a fix aimed only at the symptom would make.

  * "just raise the default to 0.87" -- a number frozen in the domain layer, which
    silently stops agreeing with `auto_approve_threshold` the moment anyone tunes it, and
    which cannot see `min_confidence_gap` at all. `test_a_confident_near_tie_is_flagged`
    is that case: a top match ABOVE the auto-approve bar that the matcher nonetheless
    refused to auto-approve, because it was a near-tie. A numeric comparison clears it.
    The decision does not.
  * "flag everything, it is safer" -- a review queue containing every field is the same
    non-answer as an empty one, and `test_auto_approved_fields_are_not_flagged` fails it.

Sessions here are built by hand rather than matched, so each case states its own
confidences and decisions instead of hoping a fixture produces them.
"""

from __future__ import annotations

import pytest

from nexus_matcher.domain.models.entities import (
    DictionaryEntry,
    MatchingSession,
    MatchResult,
    Schema,
    SchemaField,
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
    business_name="Customer Email Address",
    logical_name="cust_email",
    definition="Email address of a customer",
    data_type=DataType.STRING,
    protection_level=ProtectionLevel.PII,
)


def _match(confidence: float, decision: MatchDecision, rank: int = 1) -> MatchResult:
    return MatchResult(
        schema_field=SchemaField(name="f", data_type=DataType.STRING, full_path="f"),
        dictionary_entry=_ENTRY,
        rank=rank,
        final_confidence=confidence,
        score_breakdown=ScoreBreakdown(fused_retrieval_score=0.9),
        decision=decision,
        performance=PerformanceMetrics(latency_ms=0.0),
    )


def _session(results: dict, floor: float | None = 0.63) -> MatchingSession:
    return MatchingSession(
        session_id=EntityId(),
        schema=Schema(name="S", fields=()),
        results=results,
        total_duration_ms=0.0,
        minimum_achievable_confidence=floor,
    )


class TestTheDefault:
    """No argument means "the matcher did not auto-approve this"."""

    def test_review_and_reject_are_flagged_and_auto_approve_is_not(self):
        session = _session(
            {
                "approved": (_match(0.95, MatchDecision.AUTO_APPROVE),),
                "reviewed": (_match(0.75, MatchDecision.REVIEW),),
                "rejected": (_match(0.40, MatchDecision.REJECT),),
            }
        )
        assert session.get_low_confidence_fields() == ["reviewed", "rejected"]

    def test_auto_approved_fields_are_not_flagged(self):
        """
        Guards the opposite failure. A queue containing every field is as useless as an
        empty one -- it just fails in the direction that looks careful.
        """
        session = _session(
            {
                "a": (_match(0.91, MatchDecision.AUTO_APPROVE),),
                "b": (_match(0.99, MatchDecision.AUTO_APPROVE),),
            }
        )
        assert session.get_low_confidence_fields() == []

    def test_a_confident_near_tie_is_flagged(self):
        """
        The case that decides the design: confidence 0.8708, ABOVE the 0.87 auto-approve
        bar, and the matcher still sent it to review because its margin over the runner-up
        was under `min_confidence_gap` -- two dictionary entries scoring almost
        identically, which is precisely when a human should adjudicate.

        `get_low_confidence_fields(0.87)` returns nothing for this field. Reading the
        decision returns it. Any default expressed as a number is blind here, whatever
        number it is. (Measured, not invented: this is `pay_val` from the 6-field
        reproduction in the NM-0027 report.)
        """
        session = _session({"pay_val": (_match(0.8708, MatchDecision.REVIEW),)})

        assert session.get_low_confidence_fields() == ["pay_val"]
        assert session.get_low_confidence_fields(0.87) == [], (
            "premise check: a numeric threshold at the auto-approve bar clears this field"
        )

    def test_a_field_that_matched_nothing_is_flagged(self):
        """Nothing matched it. That is the least trustworthy outcome, not an absence."""
        session = _session({"orphan": (), "ok": (_match(0.95, MatchDecision.AUTO_APPROVE),)})
        assert session.get_low_confidence_fields() == ["orphan"]

    def test_only_the_top_match_decides(self):
        """A weak rank-2 alternative does not make a confidently-approved field suspect."""
        session = _session(
            {
                "f": (
                    _match(0.95, MatchDecision.AUTO_APPROVE),
                    _match(0.20, MatchDecision.REJECT, rank=2),
                )
            }
        )
        assert session.get_low_confidence_fields() == []

    def test_results_come_back_in_schema_order(self):
        """
        The order a reviewer works through. `results` is insertion-ordered by
        `_match_fields`, which walks the parsed fields in schema order, so the queue
        matches the document the reviewer has open.
        """
        session = _session({name: (_match(0.7, MatchDecision.REVIEW),) for name in ("c", "a", "b")})
        assert session.get_low_confidence_fields() == ["c", "a", "b"]


class TestAnExplicitThreshold:
    """A float still asks the numeric question -- unless it cannot have an answer."""

    def test_a_workable_threshold_selects_numerically(self):
        session = _session(
            {
                "low": (_match(0.70, MatchDecision.REVIEW),),
                "high": (_match(0.86, MatchDecision.REVIEW),),
            }
        )
        assert session.get_low_confidence_fields(0.80) == ["low"]

    def test_a_threshold_below_the_floor_is_refused_and_names_the_floor(self):
        """
        The defect's own default, passed explicitly. Returning [] here would be the same
        silent lie in a new place; the caller has to be told that no answer was possible,
        and told the number that makes it impossible.
        """
        session = _session({"f": (_match(0.75, MatchDecision.REVIEW),)})

        with pytest.raises(ValueError) as excinfo:
            session.get_low_confidence_fields(0.6)

        message = str(excinfo.value)
        assert "0.6300" in message, "the error must name the floor; the floor is the news"
        assert "get_low_confidence_fields()" in message, "and must say what to do instead"

    def test_the_floor_itself_is_refused(self):
        """
        `<=`, not `<`. A threshold exactly AT the floor selects nothing either: the
        comparison is `confidence < threshold` and confidence cannot go below the floor.
        """
        session = _session({"f": (_match(0.75, MatchDecision.REVIEW),)})
        with pytest.raises(ValueError):
            session.get_low_confidence_fields(0.63)

    def test_just_above_the_floor_is_allowed(self):
        """The refusal must not creep upward into thresholds that can genuinely select."""
        session = _session({"f": (_match(0.6301, MatchDecision.REVIEW),)})
        assert session.get_low_confidence_fields(0.6302) == ["f"]

    def test_an_unknown_floor_refuses_nothing(self):
        """
        A session built by hand, or produced with a reranker wired, does not know its
        floor. Guessing one would be worse than not checking: it would reject thresholds
        that are perfectly answerable. None means unknown, and unknown means allow.
        """
        session = _session({"f": (_match(0.75, MatchDecision.REVIEW),)}, floor=None)
        assert session.get_low_confidence_fields(0.6) == []
