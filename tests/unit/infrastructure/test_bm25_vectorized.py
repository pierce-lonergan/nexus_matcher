"""
tests.unit.infrastructure.test_bm25_vectorized | Layer: TEST
The vectorised BM25 must rank exactly as rank_bm25 did, only faster.

Why this file exists
--------------------
BM25Retriever used to delegate to rank_bm25's BM25Okapi, which scores every document for
every query term -- profiling put 84% of match time inside it, spent almost entirely on
adding zeros for documents that do not contain the term. It now uses an inverted index
with the weights precomputed at index time.

That is a rewrite of the scoring function, so the risk is not that it breaks loudly; it
is that it ranks *slightly* differently and quietly costs accuracy. These tests pin the
scores against rank_bm25 itself, so any drift fails here rather than showing up as an
unexplained P@1 regression months later.

rank_bm25 is a TEST dependency now, not a runtime one. If it is absent the comparison
tests skip and the behavioural ones still run.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from nexus_matcher.domain.ports.retrieval import SparseDocument
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever

VOCAB = [
    "customer",
    "order",
    "email",
    "address",
    "amount",
    "total",
    "status",
    "code",
    "date",
    "identifier",
    "the",
    "of",
    "a",
    "record",
    "payment",
    "invoice",
    "shipping",
    "tax",
    "account",
    "balance",
]


def _corpus(n: int, seed: int = 11) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(VOCAB) for _ in range(rng.randint(3, 14))) for _ in range(n)]


def _indexed(texts: list[str]) -> BM25Retriever:
    r = BM25Retriever()
    r.index([SparseDocument(id=f"d{i}", text=t) for i, t in enumerate(texts)])
    return r


class TestEquivalenceWithRankBm25:
    """The scores are the contract. Everything else is an implementation detail."""

    @staticmethod
    def _reference(retriever: BM25Retriever, texts: list[str]):
        rank_bm25 = pytest.importorskip("rank_bm25")
        return rank_bm25.BM25Okapi([retriever._tokenize(t) for t in texts], k1=1.5, b=0.75)

    @pytest.mark.parametrize("n", [1, 2, 3, 50, 500])
    def test_scores_match_at_several_corpus_sizes(self, n):
        texts = _corpus(n)
        r = _indexed(texts)
        ref = self._reference(r, texts)
        rng = random.Random(5)

        for _ in range(25):
            q = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 6)))
            tokens = r._tokenize(q)
            mine = r._score(tokens)
            theirs = np.asarray(ref.get_scores(tokens))
            # float32 weights, so compare at float32 precision rather than exactly.
            assert np.allclose(mine, theirs, rtol=1e-4, atol=1e-4), q

    def test_repeated_query_terms_count_once_per_occurrence(self):
        """
        BM25Okapi loops `for q in query`, so a repeated term contributes twice. The
        vectorised version folds that into a multiplier instead of walking the postings
        twice -- an easy place to silently diverge.
        """
        texts = _corpus(200)
        r = _indexed(texts)
        ref = self._reference(r, texts)
        for q in ("email email email", "email zzz email", "the the of of a"):
            tokens = r._tokenize(q)
            assert np.allclose(r._score(tokens), np.asarray(ref.get_scores(tokens)), atol=1e-4)

    def test_negative_idf_terms_are_floored_identically(self):
        """
        A term in more than half the corpus gets a negative raw IDF, which rank_bm25
        floors to epsilon * average_idf. Without the floor such a term would push
        documents DOWN for containing it.
        """
        # "common" appears in every document, so its raw IDF is negative.
        texts = [f"common token{i % 7}" for i in range(40)]
        r = _indexed(texts)
        ref = self._reference(r, texts)
        tokens = r._tokenize("common")
        mine, theirs = r._score(tokens), np.asarray(ref.get_scores(tokens))
        assert np.allclose(mine, theirs, atol=1e-4)
        n_docs = len(texts)
        raw = math.log(n_docs - n_docs + 0.5) - math.log(n_docs + 0.5)
        assert raw < 0, "test corpus no longer exercises the negative-IDF branch"
        assert (mine > 0).all(), "floored IDF should keep scores positive"


class TestSearchBehaviour:
    def test_ranks_the_obviously_relevant_document_first(self):
        r = _indexed(
            [
                "customer email address for contact",
                "order total amount including tax",
                "shipping street line and postal code",
            ]
        )
        top = r.search("email address", top_k=3).unwrap()
        assert top[0].id == "d0"

    def test_unknown_terms_score_nothing_rather_than_raising(self):
        r = _indexed(_corpus(20))
        assert r.search("zzzz qqqq", top_k=5).unwrap() == []

    def test_empty_index_returns_no_results(self):
        r = BM25Retriever()
        r.index([])
        assert r.search("anything", top_k=5).unwrap() == []

    def test_empty_query_returns_no_results(self):
        r = _indexed(_corpus(20))
        assert r.search("   ", top_k=5).unwrap() == []

    def test_top_k_larger_than_corpus(self):
        r = _indexed(["customer email", "order amount"])
        assert len(r.search("customer", top_k=50).unwrap()) <= 2

    def test_results_are_sorted_by_descending_score(self):
        r = _indexed(_corpus(200))
        scores = [x.score for x in r.search("customer order email", top_k=20).unwrap()]
        assert scores == sorted(scores, reverse=True)

    def test_add_then_search_finds_the_new_document(self):
        # A real corpus, not two documents: a term present in exactly half the corpus has
        # an IDF of precisely zero (log(n-df+0.5) - log(df+0.5) with n=2, df=1), so it is
        # filtered as unscored. rank_bm25 returns 0.0 there too -- that is BM25, not a
        # regression -- but it makes a two-document fixture useless for this assertion.
        r = _indexed(_corpus(50))
        r.add([SparseDocument(id="new", text="unmistakable zebra quasar")])
        assert r.search("zebra quasar", top_k=3).unwrap()[0].id == "new"

    def test_remove_drops_the_document_from_results(self):
        texts = _corpus(50)
        r = _indexed(texts)
        r.add([SparseDocument(id="doomed", text="unmistakable zebra quasar")])
        assert r.search("zebra quasar", top_k=5).unwrap()[0].id == "doomed"
        r.remove(["doomed"])
        assert all(x.id != "doomed" for x in r.search("zebra quasar", top_k=5).unwrap())

    def test_reindex_replaces_rather_than_appends(self):
        r = _indexed(_corpus(50))
        r.index([SparseDocument(id="only", text="unmistakable zebra quasar")])
        assert r.search("customer order email", top_k=5).unwrap() == []
        assert [x.id for x in r.search("zebra", top_k=5).unwrap()] == []  # sole doc, idf 0

    def test_a_document_repeated_term_does_not_double_count_postings(self):
        """
        Postings hold one entry per (term, document). If a document's repeated term ever
        produced two postings, the `scores[docs] += w` accumulation would silently apply
        only the last write instead of summing -- so assert uniqueness directly.
        """
        r = _indexed(["email email email address", "address"])
        term_id = r._vocab["email"]
        lo, hi = r._term_offsets[term_id], r._term_offsets[term_id + 1]
        docs = r._postings_docs[lo:hi]
        assert len(docs) == len(set(docs.tolist()))
