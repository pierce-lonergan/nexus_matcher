"""
NM-0027 -- the review queue was empty because the bar was under the floor.

`MatchingSession.get_low_confidence_fields(threshold=0.6)` is the one API whose NAME
answers "which of these mappings should I not trust?". It returned an empty list on
every schema this library has ever matched, and it did so silently.

Nothing could fall below 0.6. `final_confidence` is a weighted sum whose largest term is
the fused RETRIEVAL score, and that score is min-max normalised over the candidates
retrieved for one field -- so the best candidate's dense score normalises to exactly 1.0
and, with the two fusion arm weights renormalised to sum to 1, its fused score is at
least `fusion_alpha`. At the shipped weights that puts a structural floor under every
rank-1 match:

    floor = semantic_weight (0.70) * fusion_alpha (0.90) = 0.63

Measured on a real 6-field schema before the fix: default threshold 0 fields flagged,
threshold 0.87 6 fields flagged, actual top-1 confidences 0.730-0.755, six of six below
the auto-approve bar. A governance lead who called this method and trusted its default
was told there was nothing to review, on a schema where nothing was trustworthy. Same
class as NM-0005: a silent governance failure, not an ergonomics complaint.

Why it escaped: no test called the method, and the floor was FOLKLORE. Nothing in the
codebase computed it, so no gate, review or docstring could contradict a threshold that
sat below it. The fix therefore does two things -- it changes the default to "was not
auto-approved", which is the one definition that cannot drift away from the
configuration, and it EXPOSES the floor as
`MatchingConfig.minimum_achievable_confidence`, so the next person to pick a threshold
can check it against a number instead of a hunch.

Absolute values are pinned below rather than derived from the code under test (H-004):
this fixture has no sparse retriever, so the top fused score is exactly `fusion_alpha`,
and the query/entry vectors are chosen so the cosine is exactly 0.8. Both numbers are
checkable by hand, and their disagreement -- fused 0.9, real similarity 0.8 -- is DX-003
in one line.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry, MatchingSession, Schema
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import (
    DataType,
    EntityId,
    Metadata,
    ProtectionLevel,
    Result,
)

# Hand-chosen unit vectors in 4 dimensions. The query sits at [0.8, 0.6, 0, 0], so its
# cosine to the first entry is exactly 0.8 and to the second exactly 0.6 -- distinct, so
# min-max normalisation has a real span and the top candidate normalises to 1.0.
_VECTORS = {
    "Customer Email Address": (1.0, 0.0, 0.0, 0.0),
    "Merchant Postal Code": (0.0, 1.0, 0.0, 0.0),
    "Transaction Amount": (0.0, 0.0, 1.0, 0.0),
}
_QUERY = (0.8, 0.6, 0.0, 0.0)


class _StubProvider:
    """Model-free encoder with DISTINCT vectors: no download, no network, no ties."""

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


def _entry(eid: str, business: str, logical: str, domain: str) -> DictionaryEntry:
    return DictionaryEntry(
        id=eid,
        business_name=business,
        logical_name=logical,
        definition=f"Definition of {business}",
        data_type=DataType.STRING,
        protection_level=ProtectionLevel.INTERNAL,
        domain=domain,
    )


_ENTRIES = [
    _entry("d1", "Customer Email Address", "cust_email", "CONTACT"),
    _entry("d2", "Merchant Postal Code", "merch_zip", "ADDRESS"),
    _entry("d3", "Transaction Amount", "txn_amt", "TRANSACTIONS"),
]

# Field names deliberately share no tokens with any entry, so these are BAD matches --
# the case the method exists to surface.
_SCHEMA = {
    "type": "record",
    "name": "Payload",
    "namespace": "com.example",
    "fields": [
        {"name": "zzz_one", "type": "string"},
        {"name": "zzz_two", "type": "string"},
    ],
}


@pytest.fixture
def session() -> MatchingSession:
    matcher = NexusMatcher(
        embedding_provider=_StubProvider(),
        # No sparse retriever: the fused score is then exactly fusion_alpha * 1.0 for the
        # rank-1 candidate, which is what makes the floor arithmetic checkable by hand.
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=4)
        ),
        schema_parser_registry={"avro": AvroSchemaParser()},
        config=MatchingConfig(),
    )
    matcher._index_dictionary(_ENTRIES)
    return matcher.match_schema_session(_SCHEMA)


def test_the_premise_every_top_match_is_above_the_old_default(session):
    """
    Guards this file against passing vacuously.

    If the fixture ever produced a confidence below 0.6, the old default would have
    flagged it and the test below would pass without proving anything. Pinning the
    premise means a fixture that stops demonstrating the defect fails loudly instead of
    quietly turning the entry into decoration.
    """
    tops = session.get_top_matches()
    assert len(tops) == 2, "fixture should match both fields"
    for key, match in tops.items():
        assert match.final_confidence >= 0.63, (
            f"{key} scored {match.final_confidence}, below the 0.63 structural floor -- "
            f"the fixture no longer demonstrates the defect"
        )
        assert not match.is_auto_approved, f"{key} was auto-approved; it is a bad match"


def test_the_default_flags_every_field_that_was_not_auto_approved(session):
    """
    THE SYMPTOM. Two fields matched entries they share not one token with, both were sent
    to review by the matcher itself, and the method that answers "what should I not
    trust?" used to return [].
    """
    assert session.get_low_confidence_fields() == ["zzz_one", "zzz_two"], (
        "the review queue is missing fields the matcher itself refused to auto-approve"
    )


def test_a_field_that_matched_nothing_is_flagged():
    """
    The second silent skip in the same method: `if matches and ...` dropped a field with
    NO matches instead of flagging it. Nothing matched it, which is the least trustworthy
    outcome there is, and it vanished from the one list a reviewer reads.
    """
    empty = MatchingSession(
        session_id=EntityId(),
        schema=Schema(name="Payload", fields=()),
        results={"orphan": ()},
        total_duration_ms=0.0,
        metadata=Metadata(),
        minimum_achievable_confidence=0.63,
    )
    assert empty.get_low_confidence_fields() == ["orphan"]


def test_a_threshold_below_the_floor_is_refused_by_name(session):
    """
    The old default, passed explicitly, must now be an error that NAMES the floor.

    Returning [] for an impossible threshold is the defect wearing a different hat: the
    caller reads "nothing to review" and has no way to discover that no answer was
    possible. The message has to carry the number, because the number is the thing
    nobody knew.
    """
    with pytest.raises(ValueError, match=r"0\.6300"):
        session.get_low_confidence_fields(0.6)


def test_the_floor_is_computed_and_matches_the_fixture(session):
    """
    The folklore, made checkable. 0.63 is semantic_weight 0.70 x fusion_alpha 0.90, and
    the session carries it so a caller never has to guess again.
    """
    assert MatchingConfig().minimum_achievable_confidence == pytest.approx(0.63)
    assert session.minimum_achievable_confidence == pytest.approx(0.63)


def test_the_reported_scores_are_the_hand_computed_ones(session):
    """
    Absolute values on a hand-verified fixture (H-004), and DX-003 stated as arithmetic.

    `fused_retrieval_score` is 0.9 -- which is `fusion_alpha`, not a similarity. The
    actual cosine between the query and the entry it matched is 0.8. Reading the first
    number as "90% similar" is wrong by construction, and until `absolute_cosine` was
    surfaced the second number existed nowhere in the API.
    """
    top = session.get_top_matches()["zzz_one"]

    assert top.score_breakdown.fused_retrieval_score == pytest.approx(0.9)
    assert top.score_breakdown.fused_retrieval_score == pytest.approx(
        MatchingConfig().fusion_alpha
    ), "the recurring 0.9 IS fusion_alpha; if these ever differ, re-derive the floor"
    assert top.score_breakdown.absolute_cosine == pytest.approx(0.8, abs=1e-6)
    # 0.70*0.9 + 0.05*lexical(0.0) + 0.05*edit(0.1) + 0.05*type(1.0) + 0.15*domain(0.5)
    assert top.final_confidence == pytest.approx(0.76, abs=1e-6)
