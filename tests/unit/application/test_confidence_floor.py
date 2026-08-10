"""
tests.unit.application.test_confidence_floor | Layer: TEST
The structural floor of `final_confidence`, and the score that creates it.

Two claims used to be folklore in this repository, believed by everyone and computed by
nothing:

  1. `final_confidence` cannot go below about 0.63, so any threshold under that selects
     nothing. Nothing computed it, so nothing could contradict a default of 0.6 -- which
     is how NM-0027 shipped.
  2. `semantic_score` is not a semantic score. It is the fused RETRIEVAL score,
     min-max-normalised per field, so the recurring 0.9 is `fusion_alpha` rather than a
     similarity.

Both are now arithmetic with a name, and this file is what keeps them honest. The
decisive test is `test_only_one_of_the_two_scores_moves_with_fusion_alpha`: with the
embeddings, the corpus and the query held fixed, moving `fusion_alpha` moves
`fused_retrieval_score` and leaves `absolute_cosine` untouched. That is the difference
between a rank-relative score and a similarity, stated as an experiment rather than as a
claim in a docstring.

No model is downloaded: the encoder is a stub with hand-chosen vectors, so every number
below is reproducible by hand from four-dimensional dot products.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, ProtectionLevel, Result, ScoreBreakdown

# Query at [0.8, 0.6, 0, 0]: cosine 0.8 to the first entry, 0.6 to the second, 0 to the
# third. Distinct, so min-max normalisation has a real span.
_VECTORS = {
    "Customer Email Address": (1.0, 0.0, 0.0, 0.0),
    "Merchant Postal Code": (0.0, 1.0, 0.0, 0.0),
    "Transaction Amount": (0.0, 0.0, 1.0, 0.0),
}
_QUERY = (0.8, 0.6, 0.0, 0.0)


class _StubProvider:
    dimension = 4
    model_name = "stub"

    def _vector(self, text: str) -> tuple[float, ...]:
        for name, vector in _VECTORS.items():
            if name in text:
                return vector
        return _QUERY

    def embed(self, texts):
        rows = np.array([self._vector(t) for t in texts], dtype=np.float32)

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        return Result.success(np.array(self._vector(text), dtype=np.float32))


def _entry(eid: str, business: str, logical: str) -> DictionaryEntry:
    return DictionaryEntry(
        id=eid,
        business_name=business,
        logical_name=logical,
        definition=f"Definition of {business}",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.INTERNAL,
        domain="CONTACT",
    )


_ENTRIES = [
    _entry("d1", "Customer Email Address", "cust_email"),
    _entry("d2", "Merchant Postal Code", "merch_zip"),
    _entry("d3", "Transaction Amount", "txn_amt"),
]

_SCHEMA = {
    "type": "record",
    "name": "Payload",
    "namespace": "com.example",
    "fields": [{"name": "zzz_one", "type": "string"}],
}


def _matcher(config: MatchingConfig, entries=_ENTRIES, **kwargs) -> NexusMatcher:
    matcher = NexusMatcher(
        embedding_provider=_StubProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=4)
        ),
        schema_parser_registry={"avro": AvroSchemaParser()},
        config=config,
        **kwargs,
    )
    matcher._index_dictionary(entries)
    return matcher


class TestTheFloorIsComputed:
    """The number itself, before anything is matched with it."""

    def test_the_shipped_floor_is_zero_point_six_three(self):
        """Absolute, because 0.6 < 0.63 is the entire defect."""
        assert MatchingConfig().minimum_achievable_confidence == pytest.approx(0.63)

    def test_it_is_the_product_of_the_two_values_it_claims(self):
        config = MatchingConfig(semantic_weight=0.5, fusion_alpha=0.4)
        assert config.minimum_achievable_confidence == pytest.approx(0.20)

    def test_it_is_clamped_like_the_confidence_it_bounds(self):
        """
        `_weighted_confidence` clamps to [0, 1], so with a weight set that oversums every
        confidence piles up at 1.0 -- and so does the floor. An unclamped floor above 1.0
        would be a bound no confidence could satisfy, which reads as a broken matcher.
        """
        assert MatchingConfig(semantic_weight=3.0).minimum_achievable_confidence == 1.0
        assert MatchingConfig(semantic_weight=-1.0).minimum_achievable_confidence == 0.0

    def test_a_reranker_makes_the_matcher_report_no_floor(self):
        """
        A reranker REPLACES the fused score with its own squashed output, which has no
        floor at all. Reporting 0.63 there would be worse than reporting nothing: a
        caller would set a threshold on a bound that does not hold.
        """

        class _Reranker:
            def rerank(self, query, candidates, top_k=None):
                return Result.success([])

        assert _matcher(MatchingConfig()).minimum_achievable_confidence == pytest.approx(0.63)
        assert (
            _matcher(MatchingConfig(), reranker=_Reranker()).minimum_achievable_confidence is None
        )


class TestTheFloorIsReal:
    """A bound nobody has watched hold is a hypothesis. This runs the matcher."""

    @pytest.mark.parametrize("alpha", [0.5, 0.7, 0.9, 1.0])
    def test_the_rank_one_fused_score_is_exactly_fusion_alpha(self, alpha):
        """
        TIGHT, not merely satisfied. With no sparse retriever the fused score of the top
        candidate is `fusion_alpha * 1.0` exactly, because min-max maps the best dense
        score to 1.0 and `fuse_linear_ids` renormalises the arm weights to sum to 1.

        Asserting only `>= floor` would pass for a score of 1.0 and would therefore
        survive any change that made the floor a vacuous lower bound.
        """
        session = _matcher(MatchingConfig(fusion_alpha=alpha)).match_schema_session(_SCHEMA)
        top = session.get_top_matches()["zzz_one"]

        assert top.score_breakdown.fused_retrieval_score == pytest.approx(alpha)
        assert top.final_confidence >= session.minimum_achievable_confidence

    def test_the_session_carries_the_floor_that_produced_it(self):
        session = _matcher(MatchingConfig(fusion_alpha=0.5)).match_schema_session(_SCHEMA)
        assert session.minimum_achievable_confidence == pytest.approx(0.35)

    def test_the_floor_documents_its_own_precondition(self):
        """
        The caveat in the docstring, made executable: with only ONE dense candidate,
        min-max has no span, maps it to 0.0, and there is no floor at all.

        If min-max is ever changed to map a constant to 1.0, this test fails and the
        docstring's second precondition needs rewriting -- which is the point. A stated
        precondition nothing exercises is a comment.
        """
        session = _matcher(MatchingConfig(), entries=_ENTRIES[:1]).match_schema_session(_SCHEMA)
        top = session.get_top_matches()["zzz_one"]

        assert top.score_breakdown.fused_retrieval_score == pytest.approx(0.0)
        # The SESSION reports no floor, because its own data violates the bound. This
        # assertion used to be `top.final_confidence < session.minimum_achievable_confidence`
        # -- it documented the session carrying a floor the session itself disproved, and
        # stopped one line short of the consequence.
        assert session.minimum_achievable_confidence is None

    def test_a_session_that_violates_the_bound_still_answers_the_review_question(self):
        """
        The link the previous test stopped one line short of, and the reason it mattered.

        The config-level floor is a BOUND with preconditions. When they do not hold -- a
        one-entry dictionary, `dense_top_k=1`, a perfect tie -- confidences sit around 0.13
        while the config still says 0.63. Handing that number to the session made
        `get_low_confidence_fields` REFUSE every threshold in the only range where the
        fields actually were, with a message asserting no match could fall below it, on
        precisely the sessions where they all did.

        That is NM-0027's own failure -- an API telling a reviewer there is nothing to see
        -- coming back as an exception instead of an empty list. The fix that caused it was
        mine.
        """
        session = _matcher(MatchingConfig(), entries=_ENTRIES[:1]).match_schema_session(_SCHEMA)
        top = session.get_top_matches()["zzz_one"]
        assert top.final_confidence < 0.63, "fixture no longer exercises the violated bound"

        # Every one of these raised ValueError before the session-level check.
        for threshold in (0.6, 0.5, 0.3):
            assert session.get_low_confidence_fields(threshold) == ["zzz_one"]
        assert session.get_low_confidence_fields() == ["zzz_one"]

    def test_a_session_that_satisfies_the_bound_still_refuses_an_impossible_threshold(self):
        """
        The other half: dropping the floor where it does not hold must not drop it where it
        does. A threshold below a floor that genuinely applies is still a request that can
        only ever return [], and [] reads as "nothing to review".
        """
        session = _matcher(MatchingConfig()).match_schema_session(_SCHEMA)
        assert session.minimum_achievable_confidence == pytest.approx(0.63)
        with pytest.raises(ValueError, match="0.63"):
            session.get_low_confidence_fields(0.6)


class TestTheTwoScoresAreDifferentThings:
    """DX-003, as an experiment rather than an assertion about naming."""

    def test_only_one_of_the_two_scores_moves_with_fusion_alpha(self):
        """
        Identical embeddings, identical corpus, identical query. Only `fusion_alpha`
        changes.

        `fused_retrieval_score` tracks it exactly -- it is rank-relative, and 0.9 was
        never "90% similar", it was `fusion_alpha`. `absolute_cosine` does not move at
        all, because it is the similarity the model actually produced. An auditor asking
        "how similar were they really?" needs the second number, and until it was
        surfaced the API had no answer.
        """
        fused, cosines = [], []
        for alpha in (0.5, 0.9):
            top = (
                _matcher(MatchingConfig(fusion_alpha=alpha))
                .match_schema_session(_SCHEMA)
                .get_top_matches()["zzz_one"]
            )
            fused.append(top.score_breakdown.fused_retrieval_score)
            cosines.append(top.score_breakdown.absolute_cosine)

        assert fused == [pytest.approx(0.5), pytest.approx(0.9)]
        assert cosines[0] == pytest.approx(cosines[1]), (
            "the absolute cosine moved with a fusion weight, so it is not an absolute "
            "cosine -- it is another rank-relative number wearing an honest name"
        )
        assert cosines[0] == pytest.approx(0.8, abs=1e-6), "hand-computed: [0.8,0.6,0,0].[1,0,0,0]"

    def test_the_absolute_cosine_is_reported_for_every_returned_match(self):
        """Rank 2 and 3 carry theirs too: 0.6 and 0.0, the other two hand-computed dots."""
        matches = _matcher(MatchingConfig()).match_schema_session(_SCHEMA).results["zzz_one"]

        assert [m.dictionary_entry.id for m in matches] == ["d1", "d2", "d3"]
        assert [m.score_breakdown.absolute_cosine for m in matches] == [
            pytest.approx(0.8, abs=1e-6),
            pytest.approx(0.6, abs=1e-6),
            pytest.approx(0.0, abs=1e-6),
        ]

    def test_the_absolute_cosine_does_not_feed_the_confidence(self):
        """
        Reported, never scored. If it ever became an input, every calibrated threshold in
        `MatchingConfig` would be measuring a different quantity than the one it was
        calibrated on -- a scoring change smuggled in as a diagnostic.
        """
        matcher = _matcher(MatchingConfig())
        top = matcher.match_schema_session(_SCHEMA).get_top_matches()["zzz_one"]

        assert matcher._calculate_final_confidence(top.score_breakdown) == pytest.approx(
            top.final_confidence
        )
        moved = ScoreBreakdown(
            fused_retrieval_score=top.score_breakdown.fused_retrieval_score,
            lexical_score=top.score_breakdown.lexical_score,
            edit_distance_score=top.score_breakdown.edit_distance_score,
            type_compatibility_score=top.score_breakdown.type_compatibility_score,
            domain_score=top.score_breakdown.domain_score,
            absolute_cosine=0.0123,
        )
        assert matcher._calculate_final_confidence(moved) == pytest.approx(top.final_confidence)


class TestTheDeprecatedAlias:
    """`semantic_score` keeps working for one more major version, and says so."""

    def test_reading_it_warns_and_returns_the_new_field(self):
        breakdown = ScoreBreakdown(fused_retrieval_score=0.77)
        with pytest.warns(DeprecationWarning, match="fused_retrieval_score"):
            assert breakdown.semantic_score == 0.77

    def test_constructing_with_it_warns_and_still_builds_the_same_object(self):
        with pytest.warns(DeprecationWarning):
            old = ScoreBreakdown(semantic_score=0.77, lexical_score=0.5)
        assert old == ScoreBreakdown(fused_retrieval_score=0.77, lexical_score=0.5)

    def test_supplying_both_names_is_an_error(self):
        """
        Silently preferring one would let a caller mid-migration report a number they did
        not intend, in the field an auditor reads.
        """
        with pytest.raises(TypeError, match="both"):
            ScoreBreakdown(fused_retrieval_score=0.1, semantic_score=0.9)

    def test_the_warning_says_what_the_number_actually_is(self):
        """
        A deprecation that only says "renamed" teaches nothing: the reader migrates the
        name and keeps the wrong mental model. The text has to carry the correction.
        """
        with pytest.warns(DeprecationWarning) as record:
            _ = ScoreBreakdown(fused_retrieval_score=0.5).semantic_score
        message = str(record[0].message)
        assert "fused_retrieval_score" in message
        assert "absolute_cosine" in message
        assert "3.0" in message, "a deprecation without a removal version never ends"
