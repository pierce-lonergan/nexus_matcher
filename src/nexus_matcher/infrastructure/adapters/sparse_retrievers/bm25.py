"""
nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 | Layer: INFRASTRUCTURE
BM25 sparse retriever implementation.

## Relationships
# IMPLEMENTS → domain/ports/retrieval :: SparseRetriever protocol
# DEPENDS_ON → rank_bm25 :: BM25Okapi implementation
# USED_BY    → domain/services/search_service :: hybrid retrieval

## Attributes
# Security: No external calls, all processing local
# Performance: O(n) index build, O(log n + k) search
# Reliability: In-memory only, must be rebuilt on restart
"""

from __future__ import annotations

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
    BM25 sparse retriever using rank_bm25 library.

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
        self._bm25 = None
        self._documents: dict[DocumentId, SparseDocument] = {}
        self._id_to_index: dict[DocumentId, int] = {}
        self._index_to_id: list[DocumentId] = []
        self._tokenized_corpus: list[list[str]] = []

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
        """Rebuild BM25 index from documents."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required. Install with: pip install nexus-matcher[sparse]"
            ) from None

        if not self._tokenized_corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self._k1,
            b=self._b,
        )

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

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Result[list[SparseSearchResult]]:
        """Search the index."""
        try:
            if self._bm25 is None or not self._index_to_id:
                return Result.success([])

            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return Result.success([])

            # Get BM25 scores
            scores = np.asarray(self._bm25.get_scores(query_tokens))

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
