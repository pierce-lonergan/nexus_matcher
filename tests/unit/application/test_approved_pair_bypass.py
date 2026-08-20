"""
tests.unit.application.test_approved_pair_bypass | Layer: TEST
Tests: NexusMatcher(feedback_consumer=...), _bypassed_fields, _approved_pair_results
Target: application/use_cases/match_schema.py

AR-7 where it meets the matcher: a field a human has already decided skips retrieval, and
every field that has not is matched exactly as it was before this seam existed.

THE TWO HALVES, AND WHY BOTH ARE HERE

  THE DEFAULT CONSUMES NOTHING. `TestTheShippedDefault` compares a matcher with no
  consumer against the same matcher with `NullFeedbackConsumer` attached, candidate for
  candidate and confidence for confidence, and separately asserts that with no consumer the
  seam is not even consulted. The first alone would pass if the null consumer happened to
  be equivalent to a broken one; the second alone would pass if the code path were dead.
  The full-corpus paired proof against the previous build lives in the report accompanying
  this change -- 688/688, zero discordant on id and on confidence -- because a unit fixture
  cannot make that claim.

  A BYPASSED RESULT DOES NOT PRETEND TO BE A RETRIEVED ONE. `TestWhatABypassedResultSays`
  pins every number on it, including the two that are `None` and `0` on purpose. What SAYS
  a human decided it is `performance.retrieval_stage`, read through `provenance_of` -- not
  the confidence.

  THE CONFIDENCE IS NOT A SENTINEL, AND THIS FILE USED TO SAY IT WAS. It claimed 1.0 was
  "outside the range the scorer can produce" and backed that with two tests that matched
  ONE fixture and reported the highest confidence it happened to reach. That is an
  observation dressed as a structural property, and the property is false: the five default
  weights sum to exactly 1.0. `TestTheConfidenceIsNotASentinel` replaces both -- it proves
  the arithmetic with no corpus at all, then SEARCHES for a reaching case by constructing
  the maximal signals deliberately, and finds one.

WHAT IS NOT MEASURED HERE: whether bypassing improves accuracy. It cannot: the answer is a
human's and the matcher was never asked. Precision on a seen pair is 100% by construction,
which is a tautology, and a test asserting it would be measuring the fixture.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
    QuerySignals,
    _weighted_confidence,
    field_result_key,
)
from nexus_matcher.domain.governance import GovernanceVocabulary
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports import (
    ApprovedPair,
    BaseFeedbackConsumer,
    NullFeedbackConsumer,
    ReviewVerdict,
    approval_binding,
)
from nexus_matcher.domain.ports.review_feedback import (
    APPROVED_PAIR_STAGE,
    MatchProvenance,
    provenance_of,
)
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import FlattenedAvroParser
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, DocumentId, MatchDecision
from tests.properties._support import BagOfTokensProvider

# =============================================================================
# FIXTURES
# =============================================================================

VOCABULARY_JSON = {
    "tiers_most_open_first": ["OPEN", "SENSITIVE"],
    "open_classification": "OPEN",
    "classes": [
        {
            "code": "PC-3",
            "name": "Named Individual",
            "classification": "SENSITIVE",
            "personal_information": True,
            "direct_identifier": True,
            "enhancement": "MASK_ON_EXPORT",
        }
    ],
}


def _entries() -> list[DictionaryEntry]:
    rows = [
        ("GBF-0001", "Customer Email Address", "The electronic mail address", "PC-3"),
        ("GBF-0002", "Account Balance Amount", "The money held in the account", None),
        ("GBF-0003", "Merchant Settlement Status", "The settlement status", None),
        ("GBF-0004", "Customer Phone Number", "The telephone number", "PC-3"),
    ]
    return [
        DictionaryEntry(
            id=DocumentId(entry_id),
            business_name=name,
            logical_name=name.lower().replace(" ", "_"),
            definition=definition,
            data_type=DataType.STRING,
            governance_code=code,
        )
        for entry_id, name, definition, code in rows
    ]


class CountingProvider(BagOfTokensProvider):
    """The shipped encoder shape, plus a tally of what it was actually asked to encode."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, texts: Any) -> Any:
        self.batches.append(list(texts))
        return super().embed(texts)

    @property
    def query_texts(self) -> list[str]:
        """Every text encoded AFTER indexing -- the query side only."""
        return [text for batch in self.batches[1:] for text in batch]


class SpyConsumer(BaseFeedbackConsumer):
    """Answers for the field keys it was given, and records that it was asked at all."""

    def __init__(self, answers: dict[str, DictionaryEntry] | None = None) -> None:
        self.answers = answers or {}
        self.asked: list[str] = []
        self.binds = 0

    def bind(self, entries: Any) -> None:
        self.binds += 1
        self.lookup = entries

    def approved_pair(self, field: SchemaField) -> ApprovedPair | None:
        key = field_result_key(field)
        self.asked.append(key)
        entry = self.answers.get(key)
        if entry is None:
            return None
        return ApprovedPair(
            field_key=key,
            entry=entry,
            verdict=ReviewVerdict.MANUAL_OVERRIDE,
            binding=approval_binding(entry),
            reviewer="steward-a",
            decided_at="2026-08-10T09:15:02+00:00",
        )


def _matcher(consumer: Any = None, entries: list[DictionaryEntry] | None = None) -> NexusMatcher:
    provider = CountingProvider()
    matcher = NexusMatcher(
        embedding_provider=provider,
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=provider.dimension)
        ),
        sparse_retriever=BM25Retriever(),
        schema_parser_registry={"flattened_avro": FlattenedAvroParser()},
        config=MatchingConfig(results_per_field=3),
        governance=GovernanceVocabulary.from_json(VOCABULARY_JSON),
        feedback_consumer=consumer,
    )
    matcher._index_dictionary(entries or _entries())
    return matcher


COLUMNS = [
    {"flattenedName": "cust__email_addr", "dataType": "string"},
    {"flattenedName": "acct__balance_amt", "dataType": "string"},
    {"flattenedName": "mrch__settle_status", "dataType": "string"},
]


def _fields(columns: list[dict[str, Any]] | None = None) -> list[SchemaField]:
    parsed = FlattenedAvroParser().parse(columns or COLUMNS)
    return list(parsed.unwrap().fields)


def _shape(results: dict[str, tuple[Any, ...]]) -> list[tuple[str, tuple[tuple[str, float], ...]]]:
    """Every key, every candidate id and every confidence -- the comparable surface."""
    return [
        (key, tuple((m.dictionary_entry.id, round(m.final_confidence, 12)) for m in matches))
        for key, matches in results.items()
    ]


# =============================================================================
# THE SHIPPED DEFAULT
# =============================================================================


class TestTheShippedDefault:
    def test_no_consumer_means_the_seam_is_never_consulted(self):
        matcher = _matcher()
        assert matcher._feedback_consumer is None
        assert matcher._bypassed_fields(_fields()) == {}

    def test_a_null_consumer_produces_the_same_answers_as_no_consumer(self):
        """
        Candidate for candidate, confidence to twelve decimals. A ranking change small
        enough to move only the sixth decimal is still a ranking change, and the shipped
        default's whole claim is that this seam moves nothing.
        """
        baseline = _shape(_matcher()._match_fields(_fields()))
        attached = _shape(_matcher(NullFeedbackConsumer())._match_fields(_fields()))
        assert attached == baseline

    def test_a_consumer_with_no_answers_produces_the_same_answers_too(self):
        """
        Separate from the test above because a null consumer can be skipped by a shortcut
        that a real one holding zero pairs would not take, and then the two paths would not
        be the same path at all.
        """
        spy = SpyConsumer()
        baseline = _shape(_matcher()._match_fields(_fields()))
        attached = _shape(_matcher(spy)._match_fields(_fields()))

        assert attached == baseline
        assert len(spy.asked) == len(COLUMNS), "every field must be offered to the consumer"

    def test_indexing_binds_the_consumer_to_the_glossary(self):
        """
        The invalidation hook. It hangs off `_index_dictionary` rather than
        `load_dictionary`, so the incremental sync and the direct-index path used by the
        benchmarks get it too -- a hook on the public loader alone would leave those two
        applying verdicts about a dictionary that is no longer loaded.
        """
        spy = SpyConsumer()
        matcher = _matcher(spy)
        assert spy.binds == 1

        matcher._index_dictionary(_entries())
        assert spy.binds == 2

    def test_the_consumer_is_handed_a_lookup_port_not_the_private_entry_map(self):
        """
        Hexagonal boundary. A caller-supplied object must not receive a mutable reference
        to the matcher's own state, and exact resolution is precisely what AR-5's port is.
        """
        from nexus_matcher.domain.ports import EntryLookup

        spy = SpyConsumer()
        matcher = _matcher(spy)

        assert isinstance(spy.lookup, EntryLookup)
        assert spy.lookup is not matcher._dictionary_entries
        assert spy.lookup.lookup("GBF-0001").business_name == "Customer Email Address"


# =============================================================================
# WHAT A BYPASSED RESULT SAYS
# =============================================================================


class TestWhatABypassedResultSays:
    @staticmethod
    def _bypassed():
        entries = {e.id: e for e in _entries()}
        matcher = _matcher(SpyConsumer({"cust__email_addr": entries["GBF-0004"]}))
        results = matcher._match_fields(_fields())
        return matcher, results["cust__email_addr"]

    def test_one_candidate_and_it_is_the_humans(self):
        """
        Ranks 2..N would have to come from retrieval, and retrieval did not run. Inventing
        runner-ups for a human's answer would present a shortlist nobody produced.

        `GBF-0004` is deliberately NOT what retrieval returns first for this column, so a
        bypass that quietly fell through to matching would be visible here.
        """
        matches = self._bypassed()[1]
        assert len(matches) == 1
        assert matches[0].dictionary_entry.id == "GBF-0004"
        assert matches[0].rank == 1

    def test_retrieval_would_have_answered_differently(self):
        """Non-vacuity for the test above."""
        retrieved = _matcher()._match_fields(_fields())
        assert retrieved["cust__email_addr"][0].dictionary_entry.id != "GBF-0004"

    def test_the_confidence_is_one_and_it_is_not_load_bearing(self):
        """
        Pinned as the emitted value, and NOT as an identifier -- see
        `TestTheConfidenceIsNotASentinel` for why it cannot be one. What identifies a
        bypassed candidate is `performance.retrieval_stage`, asserted below.
        """
        matches = self._bypassed()[1]
        assert matches[0].final_confidence == 1.0

    def test_the_decision_is_the_humans_and_the_governance_survives_it(self):
        matches = self._bypassed()[1]
        assert matches[0].decision is MatchDecision.AUTO_APPROVE
        assert matches[0].governance is not None
        assert matches[0].governance.code == "PC-3"
        assert matches[0].governance_id == "GBF-0004"

    def test_the_absolute_score_is_absent_rather_than_zero(self):
        """
        `None` already means "the dense retriever never returned this candidate" everywhere
        else in this library, and that is exactly true here. Zero would be a similarity
        somebody measured.
        """
        matches = self._bypassed()[1]
        assert matches[0].score_breakdown.absolute_cosine is None

    def test_every_scored_component_is_zero_because_nothing_was_scored(self):
        matches = self._bypassed()[1]
        breakdown = matches[0].score_breakdown
        assert breakdown.fused_retrieval_score == 0.0
        assert breakdown.lexical_score == 0.0
        assert breakdown.edit_distance_score == 0.0
        assert breakdown.type_compatibility_score == 0.0
        assert breakdown.domain_score == 0.0

    def test_the_provenance_is_stated_not_implied(self):
        matches = self._bypassed()[1]
        performance = matches[0].performance
        assert performance.retrieval_stage == "approved_pair"
        assert performance.candidates_evaluated == 0
        assert performance.reranking_applied is False

    def test_a_retrieved_candidate_never_carries_that_stage(self):
        """Non-vacuity: `retrieval_stage` has to distinguish something."""
        results = _matcher()._match_fields(_fields())
        stages = {m.performance.retrieval_stage for ms in results.values() for m in ms}
        assert APPROVED_PAIR_STAGE not in stages

    def test_the_provenance_reads_back_as_a_value_rather_than_as_an_inference(self):
        """
        The member a client is supposed to read. `provenance_of` is one function over the
        stage the matcher stamped, so the wire projection and a library caller cannot come
        to different conclusions about the same candidate.
        """
        matches = self._bypassed()[1]
        assert provenance_of(matches[0]) is MatchProvenance.APPROVED_PAIR

        retrieved = _matcher()._match_fields(_fields())
        assert {provenance_of(m) for ms in retrieved.values() for m in ms} == {
            MatchProvenance.RETRIEVAL
        }


# =============================================================================
# THE CONFIDENCE IS NOT A SENTINEL
# =============================================================================

# A second, deliberately MAXIMAL glossary. Everything here is invented for this file --
# a fictional utility's billing vocabulary -- and it is shaped for one purpose: to drive
# all five scoring signals to their maximum simultaneously for one column.
#
#   fusedRetrieval  1.0  the same entry is rank 1 in BOTH arms, so min-max maps it to 1.0
#                        in each and the fused score is alpha + (1 - alpha)
#   lexical         1.0  the column's tokens are a subset of the entry's
#   editDistance    1.0  the column's normalised tokens are the entry's, character for
#                        character -- which needs a SINGLE-token column name, because the
#                        flattened parser reduces `a_b_c` to leaf `c`
#   type            1.0  string against string
#   domain          1.0  the request's `domain` signal contains the entry's own domain
#
# `_MAXIMAL_ROWS` puts `tariff` in exactly TWO definitions on purpose. In one, BM25's IDF
# drops the term and the lexical arm returns nothing, so min-max over a single-element map
# yields 0.0 and the fused score stops at alpha; in three of six it goes negative and the
# arm returns nothing at all. Two is the window where the lexical arm has a strict winner.
_MAXIMAL_ROWS = (
    ("MX-1", "Tariff", "the tariff the premises is billed on"),
    ("MX-2", "Meter", "the meter installed at the premises under a tariff"),
    ("MX-3", "Premises", "the premises the supply serves"),
    ("MX-4", "Invoice", "the invoice raised for one billing period"),
    ("MX-5", "Arrears", "the arrears carried on the account"),
    ("MX-6", "Settlement", "the settlement run a charge belongs to"),
)

_MAXIMAL_COLUMN = "tariff"
_MAXIMAL_ANSWER = "MX-1"
_MAXIMAL_SIGNALS = {"domain": "billing"}


def _maximal_entries() -> list[DictionaryEntry]:
    return [
        DictionaryEntry(
            id=DocumentId(entry_id),
            business_name=name,
            logical_name=name.lower(),
            definition=f"The governed element recording {definition}.",
            data_type=DataType.STRING,
            domain="billing",
        )
        for entry_id, name, definition in _MAXIMAL_ROWS
    ]


def _maximal_matcher(consumer: Any = None) -> NexusMatcher:
    return _matcher(consumer=consumer, entries=_maximal_entries())


def _maximal_fields(extra: list[dict[str, Any]] | None = None) -> list[SchemaField]:
    columns = [{"flattenedName": _MAXIMAL_COLUMN, "dataType": "string"}, *(extra or [])]
    return _fields(columns)


class TestTheConfidenceIsNotASentinel:
    """
    THE CLAIM THAT WAS FALSE, and the shape of test that could not see it.

    Shipped source said `final_confidence = 1.0` was "chosen because THE SCORER CANNOT
    REACH IT ... a value outside the range the model can produce", and two tests "proved"
    it by matching ONE fixture and asserting the maximum confidence that fixture happened
    to produce was below 1.0. That is an observation written up as a structural property.
    The property is false: the five default weights sum to exactly 1.0 and every signal is
    attainable at 1.0, so the scorer's range INCLUDES 1.0 by arithmetic.

    So this class does not sample. It proves the arithmetic with no corpus at all, and then
    SEARCHES for a reaching case by constructing the maximal signals deliberately -- which
    is the difference between "we did not see it" and "it cannot happen".
    """

    def test_the_default_weights_sum_to_exactly_one_so_the_maximum_confidence_is_one(self):
        """
        THE STRUCTURAL PROOF, with no fixture in it. `_weighted_confidence` is the one
        definition of the confidence and it clamps to [0, 1]; each signal is a score in
        [0, 1]. The maximum is therefore `sum(weights)` clamped, and the shipped weights
        sum to exactly 1.0 -- so 1.0 is inside the range the scorer can produce, and would
        remain inside it for any weight set summing to 1.0 or more.
        """
        config = MatchingConfig()
        weights = (
            config.semantic_weight,
            config.lexical_weight,
            config.edit_distance_weight,
            config.type_weight,
            config.domain_weight,
        )
        assert math.fsum(weights) == 1.0
        assert _weighted_confidence((1.0, 1.0, 1.0, 1.0, 1.0), weights) == 1.0

    def test_ordinary_retrieval_reaches_it_on_a_deliberately_maximal_fixture(self):
        """
        THE SEARCH, resolved. The fixture is constructed to put all five signals at their
        maximum for one column at once, and every signal is asserted individually -- so a
        future change that caps one of them below 1.0 shows up here as the signal that
        moved, not as an opaque confidence that drifted.

        `stage` is asserted too: this candidate came out of `fused` retrieval. No consumer
        is attached, so nothing here could have been bypassed.
        """
        results = _maximal_matcher()._match_fields(
            _maximal_fields(), signals=QuerySignals.coerce(_MAXIMAL_SIGNALS)
        )
        top = results[_MAXIMAL_COLUMN][0]
        breakdown = top.score_breakdown

        assert (
            breakdown.fused_retrieval_score,
            breakdown.lexical_score,
            breakdown.edit_distance_score,
            breakdown.type_compatibility_score,
            breakdown.domain_score,
        ) == (1.0, 1.0, 1.0, 1.0, 1.0)
        assert top.final_confidence == 1.0
        assert top.decision is MatchDecision.AUTO_APPROVE
        assert top.performance.retrieval_stage == "fused"
        assert top.dictionary_entry.id == _MAXIMAL_ANSWER

    def test_the_pair_a_client_was_told_to_read_no_longer_separates_the_two(self):
        """
        THE CONSEQUENCE, in one batch. A retrieved candidate and a human's answer, both
        carrying `(confidence 1.0, decision AUTO_APPROVE)`. That pair was documented as the
        way a client identifies a bypass; here it identifies nothing.

        This is the test the two it replaces were trying to be. Theirs asked "did anything
        in this fixture reach 1.0?"; this one puts a reaching candidate and a bypassed one
        side by side and shows the two members are equal across them.
        """
        entries = {e.id: e for e in _maximal_entries()}
        extra = [{"flattenedName": "arrears", "dataType": "string"}]
        matcher = _maximal_matcher(SpyConsumer({"arrears": entries[DocumentId("MX-4")]}))
        results = matcher._match_fields(
            _maximal_fields(extra), signals=QuerySignals.coerce(_MAXIMAL_SIGNALS)
        )

        retrieved = results[_MAXIMAL_COLUMN][0]
        bypassed = results["arrears"][0]

        assert (
            (retrieved.final_confidence, retrieved.decision)
            == (
                bypassed.final_confidence,
                bypassed.decision,
            )
            == (1.0, MatchDecision.AUTO_APPROVE)
        )

    def test_provenance_separates_them_where_the_two_numbers_cannot(self):
        """
        THE FIX, on the same batch. One member, one value, no inference: a number's meaning
        must never have to be worked out from a conjunction with a second number.
        """
        entries = {e.id: e for e in _maximal_entries()}
        extra = [{"flattenedName": "arrears", "dataType": "string"}]
        matcher = _maximal_matcher(SpyConsumer({"arrears": entries[DocumentId("MX-4")]}))
        results = matcher._match_fields(
            _maximal_fields(extra), signals=QuerySignals.coerce(_MAXIMAL_SIGNALS)
        )

        assert provenance_of(results[_MAXIMAL_COLUMN][0]) is MatchProvenance.RETRIEVAL
        assert provenance_of(results["arrears"][0]) is MatchProvenance.APPROVED_PAIR

    def test_the_absolute_score_conjunction_is_not_the_answer_either(self):
        """
        The undocumented fallback a client could have reached for -- `confidence == 1.0`
        AND `absoluteScore is null` -- pinned as insufficient rather than left to be
        discovered. The retrieved candidate that reaches 1.0 HAS an absolute score, so the
        conjunction happens to separate this pair; but `absoluteScore is null` independently
        means "the dense arm never returned this candidate", so it is a different question
        that agrees by accident. Asserted so nobody rebuilds the inference on it.
        """
        results = _maximal_matcher()._match_fields(
            _maximal_fields(), signals=QuerySignals.coerce(_MAXIMAL_SIGNALS)
        )
        top = results[_MAXIMAL_COLUMN][0]
        assert top.final_confidence == 1.0
        assert top.score_breakdown.absolute_cosine is not None


# =============================================================================
# THE WORK THAT IS SKIPPED
# =============================================================================


class TestRetrievalIsActuallySkipped:
    def test_a_bypassed_field_is_never_encoded(self):
        """
        The speed half, and the half that makes the precision half true by construction
        rather than by a threshold: the encoder is not asked about a field a human has
        already answered.
        """
        entries = {e.id: e for e in _entries()}
        matcher = _matcher(SpyConsumer({"cust__email_addr": entries["GBF-0004"]}))
        matcher._match_fields(_fields())

        encoded = matcher._embedding_provider.query_texts
        assert len(encoded) == len(COLUMNS) - 1
        assert not any("email" in text for text in encoded)

    def test_a_fully_bypassed_batch_calls_the_encoder_not_at_all(self):
        """
        The empty-batch edge. An encoder handed an empty list is an adapter-specific
        question this library must not ask, so the call is skipped rather than made with
        nothing in it.
        """
        entries = {e.id: e for e in _entries()}
        answers = {c["flattenedName"]: entries["GBF-0004"] for c in COLUMNS}
        matcher = _matcher(SpyConsumer(answers))
        results = matcher._match_fields(_fields())

        assert matcher._embedding_provider.query_texts == []
        assert len(results) == len(COLUMNS)
        assert all(m[0].final_confidence == 1.0 for m in results.values())


# =============================================================================
# THE REST OF THE BATCH
# =============================================================================


class TestTheRestOfTheBatchIsUnaffected:
    def test_the_fields_that_were_matched_are_matched_identically(self):
        """
        The SLOT ALIGNMENT, and only that. This is the test that would go red if the
        shortened batch handed a field another field's query text, embedding or dense
        candidates -- the defect class `_unique_result_key` was written for, one layer up.

        IT IS NOT A CLAIM ABOUT PRODUCTION, and the difference is worth naming so nobody
        cites it as one. The fixture encoder is batch-independent by construction, so
        removing a field here cannot move the others through the encoder. The SHIPPED int8
        ONNX encoder is not: measured over the committed 688-field corpus, the same text
        encoded in a batch of 688 and in a batch of 344 has cosine 0.9932 against itself at
        the median and 0.9398 at the worst, and 0 of 344 come back identical -- which moves
        rank 1 on 17 of those 344. That is pre-existing and reproduces exactly on the build
        before this seam existed; see `_match_fields`. A unit fixture cannot assert it away,
        so this test asserts the thing it CAN prove and says what it cannot.
        """
        entries = {e.id: e for e in _entries()}
        baseline = _matcher()._match_fields(_fields())
        mixed = _matcher(SpyConsumer({"cust__email_addr": entries["GBF-0004"]}))._match_fields(
            _fields()
        )

        untouched = [c["flattenedName"] for c in COLUMNS[1:]]
        assert [_shape({k: mixed[k]}) for k in untouched] == [
            _shape({k: baseline[k]}) for k in untouched
        ]

    def test_the_result_keys_and_their_order_are_unchanged(self):
        entries = {e.id: e for e in _entries()}
        baseline = _matcher()._match_fields(_fields())
        mixed = _matcher(SpyConsumer({"acct__balance_amt": entries["GBF-0001"]}))._match_fields(
            _fields()
        )
        assert list(mixed) == list(baseline)

    def test_a_duplicated_column_still_gets_its_suffixed_key(self):
        """
        The conservation law meets the bypass. A genuine duplicate takes a `#2` key, and it
        has to keep doing so when one of the two is bypassed -- keying the bypass map by
        result key instead of by position is exactly how that would break.
        """
        columns = [*COLUMNS, {"flattenedName": "cust__email_addr", "dataType": "string"}]
        entries = {e.id: e for e in _entries()}
        results = _matcher(SpyConsumer({"cust__email_addr": entries["GBF-0004"]}))._match_fields(
            _fields(columns)
        )

        assert "cust__email_addr" in results
        assert "cust__email_addr#2" in results
        assert len(results) == len(columns)
        assert results["cust__email_addr"][0].dictionary_entry.id == "GBF-0004"
        assert results["cust__email_addr#2"][0].dictionary_entry.id == "GBF-0004", (
            "both occurrences of one column name are the same question, so both bypass"
        )

    def test_every_field_gets_exactly_one_entry_however_it_was_answered(self):
        """NM-0005's shape: the count IS the contract, bypass or no bypass."""
        entries = {e.id: e for e in _entries()}
        fields = _fields()
        results = _matcher(SpyConsumer({"mrch__settle_status": entries["GBF-0001"]}))._match_fields(
            fields
        )
        assert len(results) == len(fields)


# =============================================================================
# ONE LEAF NAME, MANY PARENTS, END TO END
# =============================================================================


class TestTheRepeatedLeafHazardThroughTheMatcher:
    def test_a_verdict_about_one_parent_does_not_answer_its_siblings(self):
        """
        The same hazard `test_feedback_loop` pins on the consumer, asserted through the
        whole matcher so a key composed correctly in one place and discarded in another
        cannot pass.
        """
        entries = {e.id: e for e in _entries()}
        columns = [
            {"flattenedName": "billing_addr__line_1", "dataType": "string"},
            {"flattenedName": "shipping_addr__line_1", "dataType": "string"},
        ]
        matcher = _matcher(SpyConsumer({"billing_addr__line_1": entries["GBF-0004"]}))
        results = matcher._match_fields(_fields(columns))

        assert results["billing_addr__line_1"][0].dictionary_entry.id == "GBF-0004"
        assert results["billing_addr__line_1"][0].final_confidence == 1.0
        sibling = results["shipping_addr__line_1"][0]
        assert sibling.final_confidence < 1.0, (
            "the sibling column, which shares every token of its leaf name and differs only "
            "in its parent, was answered from a verdict given about a different parent"
        )
        assert sibling.performance.retrieval_stage != "approved_pair"


# =============================================================================
# FAILURE POSTURE
# =============================================================================


class TestAConsumerThatMisbehaves:
    def test_a_raising_consumer_takes_the_match_down_rather_than_going_quiet(self):
        """
        Deliberate, and the same posture as every other injected port here. Swallowing this
        into "no opinion" would turn a broken bypass into a slow ordinary run that a
        deployment believes is a fast bypassed one -- a silent loss of the only feature
        with guaranteed precision.
        """

        class Broken(BaseFeedbackConsumer):
            def approved_pair(self, field: SchemaField) -> ApprovedPair | None:
                raise RuntimeError("the approval store is unreachable")

        matcher = _matcher(Broken())
        with pytest.raises(RuntimeError, match="approval store is unreachable"):
            matcher._match_fields(_fields())
