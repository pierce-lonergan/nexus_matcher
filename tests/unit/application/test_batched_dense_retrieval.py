"""
tests.unit.application.test_batched_dense_retrieval | Layer: TEST
Batching dense retrieval must not change a single ranking or decision.

`_match_fields` retrieves dense candidates for a whole schema in one `search_batch` call
instead of one `search` per field. That turns N matrix-vector products into one
matrix-matrix product per chunk, which reads the corpus matrix once instead of N times.
Measured end to end: 1.04x at 4.6k entries, 1.35x at 30k, 2.58x at 100k -- the gain grows
with corpus size because the mechanism is memory bandwidth, not arithmetic.

The risk this file exists for
-----------------------------
An optimisation in this exact area was rejected during the same round for claiming
"bit-identical" output while silently reordering rankings. A change that reorders results
here does not raise; it just quietly maps a field to a different glossary entry, and that
entry's protection level is what the field inherits. So equivalence is asserted, not
argued.

Note that the two paths are NOT bit-identical in the scores, and cannot be: a GEMM
accumulates in a different order from a matrix-vector product, so cosine scores differ in
the last bits (measured max delta 6.3e-06 over 400 real FHIR queries). What must hold is
that no ranking, no entry id, and no DECISION moves. Asserting bit-identical floats here
would be a gate that fails for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType

SUBJECTS = ("customer", "order", "invoice", "patient", "shipment", "account")
ATTRS = ("email address", "total amount", "status code", "birth date", "street line")


def _entries(n: int) -> list[DictionaryEntry]:
    out = []
    for i in range(n):
        subject = SUBJECTS[i % len(SUBJECTS)]
        attr = ATTRS[(i // len(SUBJECTS)) % len(ATTRS)]
        out.append(
            DictionaryEntry(
                id=f"d-{i:05d}",
                business_name=f"{subject} {attr} {i}".title(),
                logical_name=f"{subject}_{attr.replace(' ', '_')}",
                definition=f"The {attr} of the {subject} record, used for reporting.",
                data_type=DataType.STRING,
                domain=subject,
            )
        )
    return out


def _fields(n: int) -> list[SchemaField]:
    out = []
    for i in range(n):
        subject = SUBJECTS[i % len(SUBJECTS)]
        attr = ATTRS[(i // len(SUBJECTS)) % len(ATTRS)]
        leaf = attr.replace(" ", "_")
        out.append(
            SchemaField(
                name=f"{leaf}_{i}",
                data_type=DataType.STRING,
                full_path=f"{subject}.{leaf}_{i}",
                parent_path=subject,
                description=f"The {attr} of the {subject}." if i % 2 == 0 else "",
            )
        )
    return out


@pytest.fixture(scope="module")
def matched() -> dict:
    """Run the same schema down both paths against one indexed dictionary."""
    matcher = NexusMatcher.from_config(MatchingConfig())
    matcher._index_dictionary(_entries(400))
    fields = _fields(60)

    store_type = type(matcher._vector_store)
    real = store_type.search_batch

    # Removing the attribute is how the per-field path is reached: _search_dense_batch
    # probes for it with getattr, exactly as it does for a store that never had one.
    delattr(store_type, "search_batch")
    try:
        per_field = matcher._match_fields(fields)
    finally:
        store_type.search_batch = real

    return {"per_field": per_field, "batched": matcher._match_fields(fields)}


class TestBatchedEqualsPerField:
    def test_same_fields_come_back(self, matched):
        assert list(matched["per_field"]) == list(matched["batched"])

    def test_every_ranking_is_identical(self, matched):
        """
        The assertion that matters. Entry ids, in order, for every field. If batching ever
        hands a field another field's candidates -- an off-by-one in the scatter back to
        input order -- this is what catches it.
        """
        for key, expected in matched["per_field"].items():
            got = matched["batched"][key]
            assert [r.dictionary_entry.id for r in expected] == [
                r.dictionary_entry.id for r in got
            ], f"ranking changed for {key!r}"

    def test_no_decision_changes(self, matched):
        """
        The governance-critical one. A field's decision is what determines whether its
        inherited protection level is applied without a human ever seeing it, so a single
        flip is a real-world consequence even if the ranking held.
        """
        for key, expected in matched["per_field"].items():
            for a, b in zip(expected, matched["batched"][key], strict=True):
                assert a.decision == b.decision, f"decision changed for {key!r}"

    def test_confidences_agree_to_float_accumulation_error(self, matched):
        """
        Deliberately a tolerance, not equality. A GEMM sums in a different order from a
        matrix-vector product, so the last bits legitimately differ; measured max delta on
        400 real FHIR queries was 6.3e-06. Demanding equality would make this gate fire on
        correct code, and a gate that cries wolf gets weakened -- which is how tolerances
        rot.
        """
        worst = 0.0
        for key, expected in matched["per_field"].items():
            for a, b in zip(expected, matched["batched"][key], strict=True):
                worst = max(worst, abs(a.final_confidence - b.final_confidence))
        assert worst < 1e-4, f"confidences diverged by {worst:.2e}, far beyond accumulation order"


class TestFallbackPaths:
    """A store without search_batch must still work. Only InMemoryVectorStore has one."""

    def test_store_without_search_batch_still_matches(self):
        matcher = NexusMatcher.from_config(MatchingConfig())
        matcher._index_dictionary(_entries(120))

        class NoBatch:
            """Wraps the real store, hiding search_batch the way Qdrant/HNSW lack it."""

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                if name == "search_batch":
                    raise AttributeError(name)
                return getattr(self._inner, name)

        matcher._vector_store = NoBatch(matcher._vector_store)
        results = matcher._match_fields(_fields(10))
        assert len(results) == 10
        assert all(v for v in results.values())

    def test_a_store_returning_the_wrong_row_count_is_ignored(self):
        """
        Zipping mismatched lists would hand a field ANOTHER field's candidates -- the same
        class of silent misattribution as the result-key collision. The row count is
        checked and the batch discarded rather than trusted.
        """
        matcher = NexusMatcher.from_config(MatchingConfig())
        matcher._index_dictionary(_entries(120))
        fields = _fields(10)
        expected = matcher._match_fields(fields)

        store = matcher._vector_store

        class ShortBatch:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def search_batch(self, embeddings, **kw):
                result = self._inner.search_batch(embeddings, **kw)
                rows = result.unwrap()
                return type(result).success(rows[:-1])  # one row short

        matcher._vector_store = ShortBatch(store)
        got = matcher._match_fields(fields)
        for key in expected:
            assert [r.dictionary_entry.id for r in expected[key]] == [
                r.dictionary_entry.id for r in got[key]
            ], "a short batch was trusted instead of being discarded"

    def test_empty_field_list(self):
        matcher = NexusMatcher.from_config(MatchingConfig())
        matcher._index_dictionary(_entries(50))
        assert matcher._match_fields([]) == {}


class TestSearchBatchIsReachable:
    """
    search_batch shipped once with NO caller at all -- 2.5x faster and dead. This asserts
    the wiring exists, so deleting the call site fails a test instead of silently
    reverting the optimisation.
    """

    def test_match_fields_actually_calls_search_batch(self):
        matcher = NexusMatcher.from_config(MatchingConfig())
        matcher._index_dictionary(_entries(80))

        calls = {"n": 0}
        inner = matcher._vector_store

        class Counting:
            def __getattr__(self, name):
                return getattr(inner, name)

            def search_batch(self, embeddings, **kw):
                calls["n"] += 1
                return inner.search_batch(embeddings, **kw)

        matcher._vector_store = Counting()
        matcher._match_fields(_fields(12))
        assert calls["n"] == 1, "the whole schema should be retrieved in ONE batched call"

    def test_the_store_advertises_it(self):
        store = InMemoryVectorStore(VectorStoreConfig(collection_name="c", dimension=8))
        assert callable(store.search_batch)
        vectors = [np.zeros(8, dtype=np.float32) for _ in range(3)]
        assert len(store.search_batch(vectors, top_k=2).unwrap()) == 3
