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

import pickle
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nexus_matcher.domain.ports.retrieval import (
    BaseSparseRetriever,
    SparseDocument,
    SparseSearchResult,
)
from nexus_matcher.shared.types.base import DocumentId, Result

# Compiled once at import. _tokenize runs per document per index build AND per query, so
# at 100k entries this was 100k trips through re's internal pattern cache per build.
_NON_TOKEN = re.compile(r"[^a-z0-9\s]")


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
        # str.split() with no argument already splits on runs of whitespace and never
        # yields an empty or whitespace-padded token, so the strip-and-filter this used to
        # do could not change any result -- it just charged two extra str.strip() calls per
        # token. At 100k entries that was 4.7 million calls per index build.
        return _NON_TOKEN.sub(" ", text.lower()).split()

    def _count_terms(self) -> tuple[list[str], list[int], list[int]]:
        """
        Flatten the tokenised corpus into one (term, frequency) entry per posting.

        Returns the terms, their in-document frequencies, and how many distinct terms each
        document contributed -- enough to rebuild the document id of every posting without
        storing one per posting.

        Counter does the counting in C (_count_elements) and `list += dict_view` copies a
        whole view inside CPython, so the interpreter runs once per DOCUMENT here rather
        than once per token or once per posting.

        Deliberately does NOT maintain a running document-frequency map: df falls out of
        the flattened postings for free via bincount, whereas updating it here cost a
        Counter.update call per document, each paying an isinstance() check against the
        Mapping ABC. At 100k entries that was 100k abstract-base-class lookups per build,
        which cProfile showed costing more than the counting itself.
        """
        flat_terms: list[str] = []
        flat_freqs: list[int] = []
        terms_per_doc: list[int] = []
        for tokens in self._tokenized_corpus:
            counts = Counter(tokens)
            flat_terms += counts.keys()
            flat_freqs += counts.values()
            terms_per_doc.append(len(counts))
        return flat_terms, flat_freqs, terms_per_doc

    def _term_idf(self, df: np.ndarray, n_docs: int) -> np.ndarray:
        """
        IDF per term id, exactly as BM25Okapi computes it.

        Negative values are floored to epsilon * average_idf: a term in more than half the
        corpus has a negative raw IDF and would otherwise push documents DOWN for
        containing a query term, which is why rank_bm25 clamps them.
        """
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        if idf.size:
            floor = self._epsilon * (idf.sum() / idf.size)
            np.copyto(idf, floor, where=idf < 0)
        return idf

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

        flat_terms, flat_freqs, terms_per_doc = self._count_terms()
        total_postings = len(flat_terms)
        doc_len = np.fromiter(map(len, self._tokenized_corpus), dtype=np.float64, count=n_docs)
        avgdl = float(doc_len.mean())

        # Term ids in first-seen order. dict.fromkeys dedupes in C while preserving that
        # order, which is the order the previous document-frequency dict happened to have,
        # so term ids -- and every offset derived from them -- are unchanged.
        vocab: dict[str, int] = {term: i for i, term in enumerate(dict.fromkeys(flat_terms))}

        # `map(dict.__getitem__, ...)` inside fromiter keeps the lookups at C speed; the
        # equivalent list comprehension costs a bytecode dispatch per posting.
        #
        # Each flat list is dropped as soon as its array exists. They hold one pointer per
        # posting -- 13 MB apiece at 100k entries -- and holding them alongside the float64
        # weight arrays below took peak build memory 21% above the scalar version, past the
        # 10% the optimization ledger allows a speedup to spend. Freeing them here instead
        # puts it 7% BELOW.
        term_ids = np.fromiter(
            map(vocab.__getitem__, flat_terms), dtype=np.int32, count=total_postings
        )
        del flat_terms
        freqs = np.fromiter(flat_freqs, dtype=np.int32, count=total_postings)
        del flat_freqs
        doc_ids = np.repeat(np.arange(n_docs, dtype=np.int32), terms_per_doc)
        del terms_per_doc

        # Document frequency, without ever having built a df map: there is exactly one
        # posting per (term, document), so the number of postings carrying a term IS the
        # number of documents containing it.
        df = np.bincount(term_ids, minlength=len(vocab))

        idf = self._term_idf(df, n_docs)

        # Length normalisation denominator component, one value per document.
        norm = self._k1 * (1.0 - self._b + self._b * doc_len / avgdl) if avgdl else None

        # Pass 2: every weight at once, instead of a Python loop over each (term, doc)
        # pair. The old loop ran once per POSTING -- 1.7 million iterations at 100k
        # entries -- and each one paid a dict lookup, two numpy scalar element assignments
        # and a numpy scalar read of a cursor. cProfile put it at 51% of the whole index
        # build, more than tokenisation and counting combined.
        #
        # Kept in float64 to the very last step, then rounded once on store, exactly as
        # the scalar version did -- it computed in Python floats and only narrowed when
        # assigning into the float32 array. Computing in float32 here would round twice
        # and drift from rank_bm25.
        #
        # Written as in-place updates rather than one expression: `a * b * c / d` would
        # allocate a separate full-length float64 temporary for every operator, four of
        # them at 13 MB each at 100k entries, all live at once.
        if total_postings:
            weights = idf[term_ids]
            weights *= freqs
            weights *= self._k1 + 1.0
            # Addition is commutative in IEEE-754, so accumulating into the length-norm
            # term gives bit-identical results to the old `freq + denom_norm`.
            denom = norm[doc_ids] if norm is not None else np.full(total_postings, self._k1)
            denom += freqs
            weights /= denom
            del denom
        else:
            weights = np.zeros(0, dtype=np.float64)

        # Group by term. A stable sort on integer keys is a radix sort in numpy, and
        # stability is what makes the result byte-identical to the old cursor fill:
        # postings were flattened in document order, so within each term the document ids
        # stay ascending. That also keeps the search-time gather a sequential read.
        order = np.argsort(term_ids, kind="stable")
        docs_arr = doc_ids[order]
        del doc_ids
        # Narrow to float32 BEFORE the reorder, so the gather moves half as many bytes and
        # no second float64 copy of the postings ever exists.
        weights = weights.astype(np.float32)
        weights_arr = weights[order]

        offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(df, out=offsets[1:])

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
