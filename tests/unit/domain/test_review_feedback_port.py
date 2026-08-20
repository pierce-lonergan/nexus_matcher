"""
tests.unit.domain.test_review_feedback_port | Layer: TEST
Tests: ReviewVerdict, ReviewedVerdict, approval_binding, FeedbackConsumer, NullFeedbackConsumer
Target: domain/ports/review_feedback.py

The port that lets a deployment consume the reviewer-verdict trail (AR-7), and the two
domain facts it rests on.

WHAT IS LOAD-BEARING HERE

  THE FOURTH VERDICT IS NOT A VERDICT. `UNSPECIFIED` exists so that a record written
  before the vocabulary could express `MANUAL_OVERRIDE` is COUNTED as unreadable rather
  than quietly folded into one of the three real values. A test that let it behave like
  `REJECTED` or like `APPROVED` would erase exactly the distinction WC-11 is about.

  THE BINDING COVERS BOTH HALVES OF A TERM. `DictionaryEntry.content_hash` deliberately
  excludes `governance_code`, because governance is metadata about a term rather than a
  description of it and folding it in would turn every re-classification into a full
  re-embed. That exclusion is right for embedding and exactly wrong for an approval: a
  reviewer approving a field against a term is approving the CLASS the field inherits.
  `test_a_reclassification_moves_the_binding_although_the_content_hash_does_not` is the
  one that would go green under a binding built on the content hash alone, so it asserts
  BOTH sides of that comparison in one test rather than trusting the reader to notice.

  THE DEFAULT CONSUMES NOTHING, STRUCTURALLY. `NullFeedbackConsumer` is not a test double
  living in a test file; it ships, so "an append-only audit trail is a legitimate default"
  is a position the package states rather than a gap it happens to have.
"""

from __future__ import annotations

import dataclasses

import pytest

from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports import (
    ApprovedPair,
    BaseFeedbackConsumer,
    FeedbackConsumer,
    NullFeedbackConsumer,
    ReviewedVerdict,
    ReviewVerdict,
    approval_binding,
)
from nexus_matcher.shared.types.base import DataType, DocumentId

ENTRY = DictionaryEntry(
    id=DocumentId("GBF-0001"),
    business_name="Resident Full Name",
    logical_name="resident_full_nm",
    definition="The full name of the resident recorded against the account.",
    data_type=DataType.STRING,
    governance_code="PC-3",
)

FIELD = SchemaField(
    name="resident_nm",
    data_type=DataType.STRING,
    full_path="account.resident_nm",
    parent_path="account",
)


# =============================================================================
# THE VOCABULARY
# =============================================================================


class TestVerdictVocabulary:
    @pytest.mark.parametrize(
        ("verdict", "stands_behind_a_term"),
        [
            (ReviewVerdict.APPROVED, True),
            (ReviewVerdict.MANUAL_OVERRIDE, True),
            (ReviewVerdict.UNSPECIFIED, True),
            (ReviewVerdict.REJECTED, False),
        ],
    )
    def test_only_rejection_names_no_term(self, verdict, stands_behind_a_term):
        """
        `REJECTED` is the only value that means "nothing in this glossary governs this".

        `UNSPECIFIED` is on the True side and that is the whole point of it existing: the
        reviewer's chosen id IS on the record, so the choice is usable; what is missing is
        whether that choice had been proposed. Putting it on the False side would throw
        away a usable answer, and merging it into APPROVED would claim an observation
        nobody made.
        """
        assert verdict.is_a_choice is stands_behind_a_term

    def test_the_wire_values_are_the_names(self):
        """A stored trail spells these; a rename is a file-format break, not a refactor."""
        assert [v.value for v in ReviewVerdict] == [
            "APPROVED",
            "REJECTED",
            "MANUAL_OVERRIDE",
            "UNSPECIFIED",
        ]


class TestVerdictRecordInvariants:
    def test_a_choice_must_name_the_term_it_chose(self):
        with pytest.raises(ValueError, match="names no chosen entry"):
            ReviewedVerdict(
                field_key="account.resident_nm",
                verdict=ReviewVerdict.MANUAL_OVERRIDE,
                chosen_entry_id="",
            )

    def test_a_rejection_must_not_also_name_one(self):
        """Two answers to one question. A consumer would have to guess which was meant."""
        with pytest.raises(ValueError, match="two different answers"):
            ReviewedVerdict(
                field_key="account.resident_nm",
                verdict=ReviewVerdict.REJECTED,
                chosen_entry_id="GBF-0001",
            )

    def test_an_unkeyed_verdict_is_refused(self):
        with pytest.raises(ValueError, match="field_key cannot be empty"):
            ReviewedVerdict(field_key="", verdict=ReviewVerdict.APPROVED, chosen_entry_id="X")


# =============================================================================
# WHAT INVALIDATES AN APPROVAL
# =============================================================================


class TestApprovalBinding:
    def test_the_same_entry_binds_the_same_way_twice(self):
        assert approval_binding(ENTRY) == approval_binding(ENTRY)

    def test_a_redefinition_moves_the_binding(self):
        redefined = dataclasses.replace(ENTRY, definition="Now means the account holder.")
        assert approval_binding(redefined) != approval_binding(ENTRY)

    def test_a_reclassification_moves_the_binding_although_the_content_hash_does_not(self):
        """
        THE TEST A CONTENT-HASH-ONLY BINDING FAILS.

        `content_hash` covers business name, logical name, definition and type, and
        deliberately not the protection code. So a glossary that moves this term from one
        protection class to another leaves the content hash byte-identical -- asserted
        here, not assumed -- while changing the exact thing a reviewer was approving.
        """
        reclassified = dataclasses.replace(ENTRY, governance_code="PC-9")

        assert reclassified.content_hash == ENTRY.content_hash, (
            "the premise of this test has changed: content_hash now covers governance, so "
            "the two-part binding may no longer be necessary"
        )
        assert approval_binding(reclassified) != approval_binding(ENTRY)

    def test_losing_a_code_entirely_moves_the_binding(self):
        """`None` and a code are different bindings; `None` and `''` are the same one."""
        declassified = dataclasses.replace(ENTRY, governance_code=None)
        assert approval_binding(declassified) != approval_binding(ENTRY)
        assert approval_binding(declassified) == approval_binding(
            dataclasses.replace(ENTRY, governance_code="")
        )

    def test_the_two_halves_cannot_be_confused_by_where_the_separator_falls(self):
        """
        A readable `hash:code` composite would be ambiguous exactly where two codes differ
        only by a separator. Two entries whose (content, code) pairs differ must bind
        differently even when a naive concatenation of them would not.
        """
        a = dataclasses.replace(ENTRY, governance_code="PC|3")
        b = dataclasses.replace(ENTRY, governance_code="PC", definition=ENTRY.definition)
        assert approval_binding(a) != approval_binding(b)


# =============================================================================
# THE PORT
# =============================================================================


class TestThePort:
    def test_the_shipped_null_consumer_never_has_an_opinion(self):
        assert NullFeedbackConsumer().approved_pair(FIELD) is None

    def test_binding_a_null_consumer_is_a_no_op_it_survives(self):
        """
        `BaseFeedbackConsumer.bind` doing nothing is correct for a consumer holding no ids
        and wrong for every other one. Calling it with an obviously wrong argument proves
        the base really does ignore it rather than happening to work.
        """
        consumer = NullFeedbackConsumer()
        consumer.bind(None)  # type: ignore[arg-type]
        assert consumer.approved_pair(FIELD) is None

    def test_the_null_consumer_satisfies_the_protocol(self):
        assert isinstance(NullFeedbackConsumer(), FeedbackConsumer)
        assert issubclass(NullFeedbackConsumer, FeedbackConsumer)

    def test_an_object_that_answers_both_questions_is_recognised_without_inheriting(self):
        """
        `runtime_checkable` is what lets a deployment hand over what it already has. If
        this stops holding, every adopter has to import and subclass a base class from a
        library whose whole point here is to stay out of the way.
        """

        class Elsewhere:
            def bind(self, entries):
                return None

            def approved_pair(self, field):
                return None

        assert isinstance(Elsewhere(), FeedbackConsumer)

    def test_a_subclass_that_forgets_the_answer_cannot_be_built(self):
        class Forgetful(BaseFeedbackConsumer):
            pass

        with pytest.raises(TypeError, match="approved_pair"):
            Forgetful()  # type: ignore[abstract]

    def test_an_approved_pair_carries_who_decided_and_when(self):
        """
        The provenance a surface needs in order to say "a human decided this" instead of
        implying a retriever did. Carried on the answer, not looked up afterwards.
        """
        pair = ApprovedPair(
            field_key="account.resident_nm",
            entry=ENTRY,
            verdict=ReviewVerdict.MANUAL_OVERRIDE,
            binding=approval_binding(ENTRY),
            reviewer="steward-a",
            decided_at="2026-08-01T09:00:00+00:00",
        )
        assert pair.reviewer == "steward-a"
        assert pair.decided_at == "2026-08-01T09:00:00+00:00"
        assert pair.binding == approval_binding(pair.entry)
