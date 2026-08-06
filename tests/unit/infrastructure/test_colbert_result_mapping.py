"""
tests.unit.infrastructure.test_colbert_result_mapping | Layer: TEST
Regression guards for ColBERTMaxSimReranker result attribution.

The production reranker used to join RAGatouille's output back onto the input
candidates using the candidate TEXT as the dict key:

    text_to_candidate = {c.text: (c, idx + 1) for idx, c in enumerate(candidates)}

Candidate texts are not unique in this product -- the same column description
is reused across tables, enum labels repeat, and many columns have an empty
description. A text-keyed dict collapses those duplicates, so every duplicate
but the last was silently DROPPED from the reranked output and the survivor was
attributed the wrong candidate id / metadata / original_rank.

These tests pin the index-based join. They fail against the text-keyed version.
"""

from __future__ import annotations

import pytest

from nexus_matcher.domain.ports.retrieval import RerankCandidate
from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
    ColBERTMaxSimReranker,
)


class _FakeRagatouilleModel:
    """
    Stand-in for RAGatouille's RAGPretrainedModel.

    Mimics the real contract: returns one dict per document, each with
    'content', 'score' and 'result_index' (the offset into the documents list),
    sorted by score descending.
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.seen_documents: list[str] | None = None

    def rerank(self, query: str, documents: list[str], k: int) -> list[dict]:
        self.seen_documents = list(documents)
        items = [
            {"content": doc, "score": self._scores[i], "result_index": i}
            for i, doc in enumerate(documents)
        ]
        items.sort(key=lambda d: d["score"], reverse=True)
        return items[:k]


def _reranker_with(model: _FakeRagatouilleModel) -> ColBERTMaxSimReranker:
    reranker = ColBERTMaxSimReranker()
    # Bypass lazy loading -- _load_model() returns early when _model is set.
    reranker._model = model
    return reranker


class TestDuplicateCandidateTexts:
    """Duplicate candidate texts must not be collapsed or mis-attributed."""

    def test_duplicate_texts_all_survive_reranking(self):
        """
        Three candidates share the identical text. All three must come back.

        Under the text-keyed join only ONE of them survived, because the dict
        had a single entry for that text.
        """
        candidates = [
            RerankCandidate(id="a", text="date of birth"),
            RerankCandidate(id="b", text="date of birth"),
            RerankCandidate(id="c", text="date of birth"),
            RerankCandidate(id="d", text="account balance"),
        ]
        model = _FakeRagatouilleModel(scores=[0.9, 0.8, 0.7, 0.1])

        result = _reranker_with(model).rerank("birth date", candidates)

        assert result.is_success, result.error
        results = result.unwrap()
        assert len(results) == 4, (
            f"expected all 4 candidates back, got {len(results)} -- duplicate texts were collapsed"
        )
        assert {r.id for r in results} == {"a", "b", "c", "d"}

    def test_duplicate_texts_keep_their_own_ids_and_scores(self):
        """Each duplicate keeps the score the reranker gave IT, not a sibling's."""
        candidates = [
            RerankCandidate(id="a", text="same text"),
            RerankCandidate(id="b", text="same text"),
        ]
        model = _FakeRagatouilleModel(scores=[0.95, 0.10])

        results = _reranker_with(model).rerank("q", candidates).unwrap()

        by_id = {r.id: r for r in results}
        assert by_id["a"].score == pytest.approx(0.95)
        assert by_id["b"].score == pytest.approx(0.10)

    def test_duplicate_texts_keep_their_own_metadata(self):
        """
        Metadata must follow the candidate, not whichever duplicate happened to
        win the dict slot. This is the defect that silently corrupts mappings.
        """
        candidates = [
            RerankCandidate(id="a", text="dup", metadata={"table": "orders"}),
            RerankCandidate(id="b", text="dup", metadata={"table": "customers"}),
        ]
        model = _FakeRagatouilleModel(scores=[0.9, 0.8])

        results = _reranker_with(model).rerank("q", candidates).unwrap()

        by_id = {r.id: r for r in results}
        assert by_id["a"].metadata == {"table": "orders"}
        assert by_id["b"].metadata == {"table": "customers"}

    def test_empty_descriptions_do_not_collapse(self):
        """Empty text is the most common duplicate in real dictionaries."""
        candidates = [RerankCandidate(id=f"f{i}", text="") for i in range(5)]
        model = _FakeRagatouilleModel(scores=[0.5, 0.4, 0.3, 0.2, 0.1])

        results = _reranker_with(model).rerank("q", candidates).unwrap()

        assert len(results) == 5
        assert {r.id for r in results} == {"f0", "f1", "f2", "f3", "f4"}


class TestOriginalRankAttribution:
    """original_rank must be the candidate's true input position."""

    def test_original_rank_matches_input_position(self):
        candidates = [
            RerankCandidate(id="first", text="alpha"),
            RerankCandidate(id="second", text="beta"),
            RerankCandidate(id="third", text="gamma"),
        ]
        # Reverse the order so new_rank != original_rank.
        model = _FakeRagatouilleModel(scores=[0.1, 0.5, 0.9])

        results = _reranker_with(model).rerank("q", candidates).unwrap()

        by_id = {r.id: r for r in results}
        assert by_id["first"].original_rank == 1
        assert by_id["second"].original_rank == 2
        assert by_id["third"].original_rank == 3
        # And the new ranks reflect the reranker's ordering.
        assert by_id["third"].rank == 1
        assert by_id["second"].rank == 2
        assert by_id["first"].rank == 3


class TestMappingFallbackAndFailure:
    """Behaviour when result_index is absent or bogus."""

    def test_missing_result_index_falls_back_to_positional_text_join(self):
        """
        Older/other reranker builds omit result_index. The fallback consumes
        one position per duplicate rather than collapsing them.
        """

        class _NoIndexModel:
            def rerank(self, query, documents, k):
                return [
                    {"content": doc, "score": 1.0 - i * 0.1} for i, doc in enumerate(documents)
                ][:k]

        candidates = [
            RerankCandidate(id="a", text="dup"),
            RerankCandidate(id="b", text="dup"),
        ]

        results = _reranker_with(_NoIndexModel()).rerank("q", candidates).unwrap()

        assert len(results) == 2
        assert {r.id for r in results} == {"a", "b"}

    def test_out_of_range_result_index_fails_loudly(self):
        """A bogus index must surface as a failure, not a wrong mapping."""

        class _BadIndexModel:
            def rerank(self, query, documents, k):
                return [{"content": documents[0], "score": 1.0, "result_index": 99}]

        candidates = [RerankCandidate(id="a", text="alpha")]

        result = _reranker_with(_BadIndexModel()).rerank("q", candidates)

        assert result.is_failure
        assert "99" in (result.error or "")

    def test_unmappable_document_fails_loudly(self):
        """Text that matches no candidate must not be silently dropped."""

        class _HallucinatingModel:
            def rerank(self, query, documents, k):
                return [{"content": "a document nobody submitted", "score": 1.0}]

        candidates = [RerankCandidate(id="a", text="alpha")]

        result = _reranker_with(_HallucinatingModel()).rerank("q", candidates)

        assert result.is_failure
