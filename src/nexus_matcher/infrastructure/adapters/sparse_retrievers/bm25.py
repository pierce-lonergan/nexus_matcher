"""
nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 | Layer: INFRASTRUCTURE
BM25 sparse retriever implementation.

## Relationships
# IMPLEMENTS → domain/ports/retrieval :: SparseRetriever protocol
# DEPENDS_ON → numpy :: vectorised inverted index (rank_bm25 is now test-only)
# USED_BY    → domain/services/search_service :: hybrid retrieval

## Attributes
# Security: No external calls, all processing local
# Performance: O(n) index build; search is O(sum of df over query terms), not O(n)
# Reliability: In-memory only, must be rebuilt on restart
"""

from __future__ import annotations

import math
import pickle
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nexus_matcher.domain.ports.retrieval import (
    BaseSparseRetriever,
    SparseDocument,
    SparseSearchResult,
)
from nexus_matcher.shared.types.base import DocumentId, Result


class BM25Retriever(BaseSparseRetriever):
    """
    BM25 sparse retriever over a vectorised inverted index.

    BM25 (Best Matching 25) is a probabilistic retrieval function that
    ranks documents based on term frequency and inverse document frequency.

    Parameters:
    - k1: Term saturation parameter (default 1.5)
    - b: Length normalization parameter (default 0.75)

    Example:
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index([
            SparseDocument(id="1", text="customer email address"),
            SparseDocument(id="2", text="transaction amount value"),
        ])
        results = retriever.search("email", top_k=10)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Initialize BM25 retriever.

        Args:
            k1: Term saturation parameter
            b: Length normalization parameter
        """
        self._k1 = k1
        self._b = b
        # BM25Okapi's default. Floors negative IDF for terms appearing in more than half
        # the corpus; kept identical so scores stay comparable with rank_bm25.
        self._epsilon = 0.25
        self._documents: dict[DocumentId, SparseDocument] = {}
        self._id_to_index: dict[DocumentId, int] = {}
        self._index_to_id: list[DocumentId] = []
        self._tokenized_corpus: list[list[str]] = []

        # Inverted index, filled by _rebuild_index.
        self._vocab: dict[str, int] = {}
        self._postings_docs: np.ndarray = np.zeros(0, dtype=np.int32)
        self._postings_weights: np.ndarray = np.zeros(0, dtype=np.float32)
        self._term_offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self._n_docs = 0
        self._indexed = False

    @property
    def retriever_type(self) -> str:
        """Retriever type identifier."""
        return "bm25"

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Simple whitespace tokenization with lowercasing and
        basic punctuation removal.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and replace non-alphanumeric with space
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Split on whitespace and filter empty
        tokens = [t.strip() for t in text.split() if t.strip()]

        return tokens

    def _rebuild_index(self) -> None:
        """
        Build an inverted index with the BM25 weights precomputed.

        Replaces rank_bm25's BM25Okapi, which scores EVERY document for EVERY query term:

            for q in query:
                q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])

        That inner list comprehension is one Python-level dict lookup per (term, document)
        pair, so a search costs O(|query| x N) interpreted operations regardless of how
        rare the terms are. Profiling the match path at 10k entries put 84% of total time
        inside it -- 15.9 million dict.get calls for 300 queries.

        A document that does not contain a term contributes exactly zero to the score, so
        all of that work is spent adding zeros. This builds the standard inverted index
        instead: for each term, the documents that actually contain it. A query then costs
        O(sum of document frequencies over its terms), and the accumulation runs inside
        numpy rather than the interpreter.

        Every factor of the BM25 weight is known at index time:

            w(t, d) = idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b*|d|/avgdl))

        so the weights are computed once here rather than per query. Scoring is then a
        gather-and-add over each term's postings.

        Layout: postings are stored as three flat numpy arrays (doc ids, weights, and
        per-term offsets into both) rather than a dict of per-term arrays. One contiguous
        allocation per array keeps a term's postings adjacent in memory, so the gather is
        a sequential read; a dict of small arrays would scatter them across the heap and
        add a dict lookup per term.

        Scores are bit-comparable with BM25Okapi, including its epsilon flooring of
        negative IDF; tests/unit/infrastructure/test_bm25_vectorized.py asserts that
        against rank_bm25 directly.
        """
        n_docs = len(self._tokenized_corpus)
        if not n_docs:
            self._vocab = {}
            self._postings_docs = np.zeros(0, dtype=np.int32)
            self._postings_weights = np.zeros(0, dtype=np.float32)
            self._term_offsets = np.zeros(1, dtype=np.int64)
            self._indexed = False
            return

        # Pass 1: per-document term frequencies, document lengths, document frequencies.
        doc_freqs: list[dict[str, int]] = []
        doc_len = np.empty(n_docs, dtype=np.float64)
        df: dict[str, int] = {}
        for i, tokens in enumerate(self._tokenized_corpus):
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            doc_freqs.append(counts)
            doc_len[i] = len(tokens)
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1

        avgdl = float(doc_len.mean()) if n_docs else 0.0

        # IDF exactly as BM25Okapi computes it, negative values floored to
        # epsilon * average_idf. Terms in more than half the corpus get a negative raw
        # IDF; without the floor they would actively push documents DOWN for containing
        # a query term, which is why rank_bm25 clamps them.
        idf: dict[str, float] = {}
        idf_sum = 0.0
        negative: list[str] = []
        for term, freq in df.items():
            value = math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)
            idf[term] = value
            idf_sum += value
            if value < 0:
                negative.append(term)
        if idf:
            floor = self._epsilon * (idf_sum / len(idf))
            for term in negative:
                idf[term] = floor

        # Length normalisation denominator component, one value per document.
        norm = self._k1 * (1.0 - self._b + self._b * doc_len / avgdl) if avgdl else None

        # Pass 2: fill the flat postings arrays. Sized exactly, so no list growth.
        vocab: dict[str, int] = {term: i for i, term in enumerate(df)}
        total_postings = sum(len(counts) for counts in doc_freqs)
        offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
        for term, freq in df.items():
            offsets[vocab[term] + 1] = freq
        np.cumsum(offsets, out=offsets)

        docs_arr = np.empty(total_postings, dtype=np.int32)
        weights_arr = np.empty(total_postings, dtype=np.float32)
        cursor = offsets[:-1].copy()

        for doc_index, counts in enumerate(doc_freqs):
            denom_norm = norm[doc_index] if norm is not None else self._k1
            for term, freq in counts.items():
                term_id = vocab[term]
                pos = cursor[term_id]
                docs_arr[pos] = doc_index
                weights_arr[pos] = idf[term] * freq * (self._k1 + 1.0) / (freq + denom_norm)
                cursor[term_id] = pos + 1

        self._vocab = vocab
        self._postings_docs = docs_arr
        self._postings_weights = weights_arr
        self._term_offsets = offsets
        self._n_docs = n_docs
        self._indexed = True

    def index(self, documents: Sequence[SparseDocument]) -> Result[int]:
        """Build index from documents (replaces existing index)."""
        try:
            self._documents.clear()
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._tokenized_corpus.clear()

            for doc in documents:
                idx = len(self._index_to_id)
                self._documents[doc.id] = doc
                self._id_to_index[doc.id] = idx
                self._index_to_id.append(doc.id)
                self._tokenized_corpus.append(self._tokenize(doc.text))

            self._rebuild_index()

            return Result.success(len(documents))

        except Exception as e:
            return Result.failure(f"Index build failed: {e}")

    def add(self, documents: Sequence[SparseDocument]) -> Result[int]:
        """Add documents to existing index."""
        try:
            added = 0

            for doc in documents:
                if doc.id in self._documents:
                    # Update existing
                    old_idx = self._id_to_index[doc.id]
                    self._tokenized_corpus[old_idx] = self._tokenize(doc.text)
                else:
                    # Add new
                    idx = len(self._index_to_id)
                    self._id_to_index[doc.id] = idx
                    self._index_to_id.append(doc.id)
                    self._tokenized_corpus.append(self._tokenize(doc.text))
                    added += 1

                self._documents[doc.id] = doc

            self._rebuild_index()

            return Result.success(added)

        except Exception as e:
            return Result.failure(f"Add failed: {e}")

    def remove(self, ids: Sequence[DocumentId]) -> Result[int]:
        """Remove documents from index."""
        try:
            removed = 0
            ids_to_remove = set(ids)

            # Filter out removed documents
            new_corpus: list[list[str]] = []
            new_index_to_id: list[DocumentId] = []

            for doc_id in self._index_to_id:
                if doc_id in ids_to_remove:
                    del self._documents[doc_id]
                    del self._id_to_index[doc_id]
                    removed += 1
                else:
                    # Read the token list at the OLD index before reassigning the
                    # mapping. The previous order overwrote _id_to_index first and then
                    # used the new value to index the old corpus, so every surviving
                    # document silently inherited another document's tokens.
                    old_idx = self._id_to_index[doc_id]
                    new_corpus.append(self._tokenized_corpus[old_idx])
                    self._id_to_index[doc_id] = len(new_index_to_id)
                    new_index_to_id.append(doc_id)

            self._index_to_id = new_index_to_id
            self._tokenized_corpus = new_corpus
            self._rebuild_index()

            return Result.success(removed)

        except Exception as e:
            return Result.failure(f"Remove failed: {e}")

    def _score(self, query_tokens: Sequence[str]) -> np.ndarray:
        """
        BM25 score for every document, via the inverted index.

        Only documents containing a query term are touched. A term absent from the
        vocabulary is skipped outright rather than contributing a row of zeros.

        A term repeated in the query contributes once per occurrence, matching
        BM25Okapi's `for q in query` loop, so the counts are folded into the multiplier
        instead of iterating the same postings twice.
        """
        scores = np.zeros(self._n_docs, dtype=np.float32)

        counts: dict[int, int] = {}
        vocab = self._vocab
        for token in query_tokens:
            term_id = vocab.get(token)
            if term_id is not None:
                counts[term_id] = counts.get(term_id, 0) + 1

        offsets = self._term_offsets
        docs = self._postings_docs
        weights = self._postings_weights
        for term_id, qtf in counts.items():
            lo = offsets[term_id]
            hi = offsets[term_id + 1]
            postings = docs[lo:hi]
            # Document ids within one term's postings are unique, so `+=` through fancy
            # indexing accumulates correctly here. It would NOT if a doc could repeat --
            # numpy would apply only the last write rather than summing.
            if qtf == 1:
                scores[postings] += weights[lo:hi]
            else:
                scores[postings] += weights[lo:hi] * qtf

        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Result[list[SparseSearchResult]]:
        """Search the index."""
        try:
            if not self._indexed or not self._index_to_id:
                return Result.success([])

            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return Result.success([])

            scores = self._score(query_tokens)

            # Select top-k with argpartition (O(N)) then sort only those k, rather than
            # sorting the entire corpus (O(N log N)) to keep a handful of results.
            k = min(top_k, len(scores))
            if k <= 0:
                return Result.success([])
            if k < len(scores):
                candidates = np.argpartition(-scores, k - 1)[:k]
            else:
                candidates = np.arange(len(scores))
            order = candidates[np.argsort(-scores[candidates])]

            # Build results
            results: list[SparseSearchResult] = []
            for idx in order:
                score = float(scores[idx])
                if score <= 0:
                    continue

                doc_id = self._index_to_id[idx]
                doc = self._documents[doc_id]

                results.append(
                    SparseSearchResult(
                        id=doc_id,
                        score=float(score),
                        metadata=doc.metadata,
                    )
                )

            return Result.success(results)

        except Exception as e:
            return Result.failure(f"Search failed: {e}")

    def save(self, path: str) -> Result[bool]:
        """Save index to disk."""
        try:
            data = {
                "k1": self._k1,
                "b": self._b,
                "documents": self._documents,
                "id_to_index": self._id_to_index,
                "index_to_id": self._index_to_id,
                "tokenized_corpus": self._tokenized_corpus,
            }

            with Path(path).open("wb") as f:
                pickle.dump(data, f)

            return Result.success(True)

        except Exception as e:
            return Result.failure(f"Save failed: {e}")

    def load(self, path: str) -> Result[bool]:
        """Load index from disk."""
        try:
            if not Path(path).exists():
                return Result.failure(f"File not found: {path}")

            with Path(path).open("rb") as f:
                data = pickle.load(f)

            self._k1 = data["k1"]
            self._b = data["b"]
            self._documents = data["documents"]
            self._id_to_index = data["id_to_index"]
            self._index_to_id = data["index_to_id"]
            self._tokenized_corpus = data["tokenized_corpus"]

            self._rebuild_index()

            return Result.success(True)

        except Exception as e:
            return Result.failure(f"Load failed: {e}")
