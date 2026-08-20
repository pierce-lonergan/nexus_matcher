"""
tests.unit.application.test_absolute_score_floor | Layer: TEST
The absolute-score floor, and the arithmetic that makes it necessary.

## Relationships
# TESTS → application/use_cases/match_schema :: MatchingConfig.absolute_score_floor,
#         NexusMatcher.absolute_score_floor / .absolute_score_metric, session plumbing
# TESTS → domain/models/entities :: the verdict the floor produces

## The claim this file exists to verify rather than repeat

"Rank 1 can never be REJECT on score alone, so every field comes back at least REVIEW,
so there is no way to say nothing matched." That is an arithmetic claim about the SHIPPED
configuration, and a feature built on a claim nobody re-derived is a feature built on
folklore -- which is how NM-0027 shipped a review queue that was always empty.

So it is derived twice here. `TestRankOneCannotBeRejected` does it from the constants:
`semantic_weight * fusion_alpha` = 0.63 against `review_threshold` = 0.50. Then it does it
by running the matcher on a field chosen so that NOTHING in the dictionary describes it,
and watching the verdict come back REVIEW at confidence 0.76 while the actual similarity
is 0.06.

That gap -- a confident-looking verdict over a near-orthogonal match -- is the whole
argument for an absolute score and a floor over it.

No model is downloaded: the encoder is a stub with hand-chosen four-dimensional vectors,
so every number below is reproducible by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry, FieldDecision
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import (
    DataType,
    MatchDecision,
    ProtectionLevel,
    Result,
)

# Three entries on three orthogonal axes, and a query that is NEARLY orthogonal to all of
# them: cosine ~0.060, ~0.030, 0.0. Distinct, so min-max normalisation still has a span
# and the fused score of the winner is exactly `fusion_alpha` -- which is precisely the
# trap. The best of three irrelevant entries is still normalised to the top of its field.
_VECTORS = {
    "Customer Email Address": (1.0, 0.0, 0.0, 0.0),
    "Merchant Postal Code": (0.0, 1.0, 0.0, 0.0),
    "Transaction Amount": (0.0, 0.0, 1.0, 0.0),
}
_QUERY = (0.06, 0.03, 0.0, 1.0)


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

# A column no glossary entry describes. Named so, because that is the case the whole
# feature is about: a field a human would look at and say "we have no term for this".
_SCHEMA = {
    "type": "record",
    "name": "Payload",
    "namespace": "com.example",
    "fields": [{"name": "zzz_undescribed", "type": "string"}],
}


def _matcher(config: MatchingConfig | None = None, entries=_ENTRIES, **kwargs) -> NexusMatcher:
    matcher = NexusMatcher(
        embedding_provider=_StubProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=4)
        ),
        schema_parser_registry={"avro": AvroSchemaParser()},
        config=config or MatchingConfig(),
        **kwargs,
    )
    matcher._index_dictionary(entries)
    return matcher


# =============================================================================
# THE ARITHMETIC THE FEATURE RESTS ON
# =============================================================================


class TestRankOneCannotBeRejected:
    """Verified from the shipped constants, then watched happen."""

    def test_the_floor_sits_above_the_review_threshold(self):
        """
        The claim, in one line of arithmetic on the values that actually ship.

        0.70 * 0.90 = 0.63, and `review_threshold` is 0.50. A rank-1 confidence cannot
        fall below 0.63, and `_determine_decision` returns REJECT only below 0.50, so no
        setting of `review_threshold` alone recovers "nothing matched": the floor moves
        with the WEIGHTS and the threshold does not move with it.
        """
        config = MatchingConfig()
        assert config.semantic_weight == 0.70
        assert config.fusion_alpha == 0.90
        assert config.minimum_achievable_confidence == pytest.approx(0.63)
        assert config.review_threshold == 0.50
        assert config.minimum_achievable_confidence > config.review_threshold

    def test_a_field_nothing_describes_still_comes_back_reviewable(self):
        """
        The arithmetic above, run.

        `zzz_undescribed` shares no token with any entry and its query vector is nearly
        orthogonal to all three -- the best available similarity is about 0.06. The verdict
        is REVIEW at a confidence in the mid-0.7s, because min-max normalisation puts the
        best of a bad set at the top of its own field and the fused score therefore lands
        at `fusion_alpha` regardless.

        Both numbers are asserted. The confidence alone would look like a mediocre-but-real
        match; the absolute score is what says the set was bad, and the two of them
        together are the argument for promoting it to a first-class field.
        """
        session = _matcher().match_schema_session(_SCHEMA)
        top = session.get_top_matches()["zzz_undescribed"]

        assert top.decision is MatchDecision.REVIEW
        assert top.final_confidence > 0.63
        assert top.score_breakdown.fused_retrieval_score == pytest.approx(0.90)
        assert top.score_breakdown.absolute_cosine == pytest.approx(0.0599, abs=5e-3)

    def test_no_threshold_setting_recovers_the_missing_state(self):
        """
        The obvious workaround, refuted. Raising `review_threshold` to the top of its
        useful range still cannot reject rank 1, because 0.63 is a floor on the
        CONFIDENCE and the threshold is compared against it.

        Raising it ABOVE 0.63 would start rejecting rank 1 on real matches too -- the
        threshold has no way to distinguish "confident because it is good" from "confident
        because it is the best of a bad set". That is the distinction only an absolute
        score carries.
        """
        for review_threshold in (0.50, 0.60, 0.63):
            session = _matcher(
                MatchingConfig(review_threshold=review_threshold)
            ).match_schema_session(_SCHEMA)
            top = session.get_top_matches()["zzz_undescribed"]
            assert top.decision is not MatchDecision.REJECT, (
                f"review_threshold={review_threshold} rejected rank 1, so the structural "
                f"floor this feature is built on no longer holds"
            )


# =============================================================================
# THE CONFIGURATION
# =============================================================================


class TestTheFloorIsCallerSupplied:
    """Off by default, and off means off."""

    def test_the_shipped_default_is_no_floor(self):
        """
        This library ships no floor for the same reason it ships no taxonomy: a floor is a
        statement about a score distribution, and the distribution belongs to a dictionary
        and a set of field names this library has never seen. A default here would start
        unclassifying somebody's fields on a number nobody chose.
        """
        assert MatchingConfig().absolute_score_floor is None
        assert _matcher().absolute_score_floor is None

    def test_with_no_floor_a_hopeless_field_keeps_its_reviewable_verdict(self):
        """
        The default configuration must be unchanged by this feature. The field from
        `test_a_field_nothing_describes_still_comes_back_reviewable` is the worst case
        available, and it still reads REVIEW.
        """
        session = _matcher().match_schema_session(_SCHEMA)
        assert session.field_decisions() == {"zzz_undescribed": FieldDecision.REVIEW}

    def test_a_configured_floor_reaches_the_verdict(self):
        """
        End to end: the number goes into `MatchingConfig`, through the matcher, onto the
        session, and comes out as the verdict. 0.30 is above the 0.06 this field can
        actually reach and below anything a real match would score.
        """
        matcher = _matcher(MatchingConfig(absolute_score_floor=0.30))
        assert matcher.absolute_score_floor == 0.30

        session = matcher.match_schema_session(_SCHEMA)
        assert session.absolute_score_floor == 0.30
        assert session.field_decisions() == {"zzz_undescribed": FieldDecision.NO_MATCH}

    def test_a_floor_below_the_reachable_similarity_changes_nothing(self):
        """
        The other direction, so the previous test is not passing on the mere presence of a
        floor. At 0.01 the same field clears it and reads REVIEW again.
        """
        session = _matcher(MatchingConfig(absolute_score_floor=0.01)).match_schema_session(_SCHEMA)
        assert session.field_decisions() == {"zzz_undescribed": FieldDecision.REVIEW}

    def test_it_is_loadable_from_a_config_file(self, tmp_path):
        """
        Deployment data, not a code change. `_load_matching_config` refuses unknown keys,
        so a field it does not know about cannot be set from a file at all -- which is
        exactly the silent-drop failure that gate exists to prevent.
        """
        path = tmp_path / "matching.json"
        path.write_text('{"absolute_score_floor": 0.42}', encoding="utf-8")
        assert NexusMatcher.from_config(path)._config.absolute_score_floor == 0.42


# =============================================================================
# SAYING WHAT THE NUMBER IS
# =============================================================================


class TestTheMetricIsReported:
    """`absolute_cosine` is only a cosine while the store says it is."""

    def test_the_shipped_wiring_reports_cosine(self):
        assert _matcher().absolute_score_metric == "cosine"

    def test_a_store_configured_for_another_metric_says_so(self):
        """
        The number is whatever the store returned. A caller who wires a dot-product store
        gets something monotone in similarity that is neither bounded to [-1, 1] nor a
        cosine, and a floor measured under one metric means nothing under the other.
        Naming the metric is what turns that from a trap into a fact.
        """
        matcher = NexusMatcher(
            embedding_provider=_StubProvider(),
            vector_store=InMemoryVectorStore(
                VectorStoreConfig(collection_name="dictionary", dimension=4, distance_metric="dot")
            ),
            schema_parser_registry={"avro": AvroSchemaParser()},
        )
        matcher._index_dictionary(_ENTRIES)
        assert matcher.absolute_score_metric == "dot"

    def test_a_store_that_declares_nothing_reports_unknown(self):
        """
        `unknown` is NOT a synonym for cosine. A caller supplying their own store gets an
        answer that admits the library cannot state what the number is, rather than one
        that guesses the flattering case.
        """

        class _OpaqueStore:
            store_type = "opaque"

        matcher = NexusMatcher(
            embedding_provider=_StubProvider(),
            vector_store=_OpaqueStore(),  # type: ignore[arg-type]
        )
        assert matcher.absolute_score_metric == "unknown"
