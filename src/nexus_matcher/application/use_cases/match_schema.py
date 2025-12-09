"""
nexus_matcher.application.use_cases.match_schema | Layer: APPLICATION
Core schema matching use case - the main orchestrator.

## Relationships
# DEPENDS_ON → domain/ports/* :: all port interfaces
# DEPENDS_ON → domain/models/* :: domain models
# DEPENDS_ON → infrastructure/config :: configuration
# USED_BY    → presentation/api :: REST endpoints
# USED_BY    → presentation/cli :: CLI commands
# USED_BY    → external :: library API

## Attributes
# Security: No direct access to sensitive data, uses ports
# Performance: Caches embeddings, batch processing
# Reliability: Graceful degradation if components unavailable
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nexus_matcher.domain.models.entities import (
    DictionaryEntry,
    MatchingSession,
    MatchResult,
    Schema,
    SchemaField,
)
from nexus_matcher.domain.ports import (
    DictionaryLoader,
    EmbeddingProvider,
    Reranker,
    SchemaParser,
    SparseRetriever,
    VectorStore,
)
from nexus_matcher.domain.ports.dictionary_loader import ColumnMapping, LoadStatistics
from nexus_matcher.domain.ports.retrieval import RerankCandidate, SparseDocument
from nexus_matcher.domain.ports.vector_store import SearchResult, VectorDocument, VectorStoreConfig
from nexus_matcher.domain.services.abbreviation import AbbreviationExpander
from nexus_matcher.domain.services.context_enricher import ContextEnricher
from nexus_matcher.domain.services.domain_hierarchy import DomainMatcher
from nexus_matcher.shared.types.base import (
    DataType,
    EntityId,
    MatchDecision,
    PerformanceMetrics,
    Result,
    Score,
    ScoreBreakdown,
)


# =============================================================================
# MATCHING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for the matching process."""

    # Retrieval
    dense_top_k: int = 100
    sparse_top_k: int = 100
    fusion_alpha: float = 0.65  # Weight for dense scores

    # Reranking
    colbert_top_k: int = 50
    cross_encoder_top_k: int = 20

    # Scoring weights
    semantic_weight: float = 0.70
    lexical_weight: float = 0.05
    edit_distance_weight: float = 0.05
    type_weight: float = 0.05
    domain_weight: float = 0.15

    # Thresholds
    auto_approve_threshold: float = 0.75
    review_threshold: float = 0.50
    min_confidence_gap: float = 0.10

    # Results
    results_per_field: int = 5


# =============================================================================
# NEXUS MATCHER - Main class
# =============================================================================


class NexusMatcher:
    """
    Main schema matching orchestrator.

    Coordinates all components to match schema fields to dictionary entries.
    Can be used as a library, backend service, or CLI tool.

    Example (Library Mode):
        ```python
        from nexus_matcher import NexusMatcher

        matcher = NexusMatcher.from_config()
        matcher.load_dictionary("data/dictionary.xlsx")

        results = matcher.match_schema("schemas/customer.avsc")
        for field_path, matches in results.items():
            print(f"{field_path} -> {matches[0].dictionary_entry.business_name}")
        ```

    Example (With custom components):
        ```python
        from nexus_matcher import NexusMatcher
        from nexus_matcher.domain.ports import EmbeddingProvider

        matcher = NexusMatcher(
            embedding_provider=my_custom_provider,
            vector_store=my_vector_store,
        )
        ```
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        sparse_retriever: SparseRetriever | None = None,
        reranker: Reranker | None = None,
        schema_parser_registry: dict[str, SchemaParser] | None = None,
        dictionary_loader_registry: dict[str, DictionaryLoader] | None = None,
        abbreviation_expander: AbbreviationExpander | None = None,
        context_enricher: ContextEnricher | None = None,
        domain_matcher: DomainMatcher | None = None,
        config: MatchingConfig | None = None,
    ) -> None:
        """
        Initialize the matcher.

        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Store for vector similarity search
            sparse_retriever: Optional BM25/sparse retriever for hybrid search
            reranker: Optional neural reranker
            schema_parser_registry: Schema parsers by format name
            dictionary_loader_registry: Dictionary loaders by source type
            abbreviation_expander: Expander for abbreviations (uses default if None)
            context_enricher: Enricher for nested schema context (uses default if None)
            domain_matcher: Matcher for domain hierarchy scoring (uses default if None)
            config: Matching configuration
        """
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._sparse_retriever = sparse_retriever
        self._reranker = reranker
        self._schema_parsers = schema_parser_registry or {}
        self._dictionary_loaders = dictionary_loader_registry or {}
        self._abbreviation_expander = abbreviation_expander or AbbreviationExpander.default()
        self._context_enricher = context_enricher or ContextEnricher()
        self._domain_matcher = domain_matcher or DomainMatcher.default()
        self._config = config or MatchingConfig()

        # State
        self._dictionary_entries: dict[str, DictionaryEntry] = {}
        self._is_initialized = False

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> NexusMatcher:
        """
        Create matcher from configuration file.

        Args:
            config_path: Path to config file (uses defaults if None)

        Returns:
            Configured NexusMatcher instance
        """
        # Import infrastructure components
        from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
            SentenceTransformersProvider,
        )
        from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
        from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
        from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
        from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import (
            ExcelDictionaryLoader,
            CsvDictionaryLoader,
        )

        # Create components with defaults
        embedding_provider = SentenceTransformersProvider()
        vector_store = InMemoryVectorStore(VectorStoreConfig(
            collection_name="dictionary",
            dimension=embedding_provider.dimension,
        ))
        sparse_retriever = BM25Retriever()

        # Register parsers and loaders
        schema_parsers = {"avro": AvroSchemaParser()}
        dictionary_loaders = {
            "excel": ExcelDictionaryLoader(),
            "csv": CsvDictionaryLoader(),
        }

        return cls(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            sparse_retriever=sparse_retriever,
            schema_parser_registry=schema_parsers,
            dictionary_loader_registry=dictionary_loaders,
        )

    def load_dictionary(
        self,
        source: str | Path,
        column_mapping: ColumnMapping | None = None,
        source_type: str | None = None,
    ) -> LoadStatistics:
        """
        Load a data dictionary into the matcher.

        Args:
            source: Path to dictionary file
            column_mapping: Custom column mapping
            source_type: Force specific loader (auto-detect if None)

        Returns:
            Loading statistics

        Raises:
            ValueError: If no suitable loader found
        """
        path = Path(source)

        # Auto-detect loader
        if source_type is None:
            for loader_type, loader in self._dictionary_loaders.items():
                if path.suffix.lower() in loader.supported_extensions:
                    source_type = loader_type
                    break

        if source_type is None:
            raise ValueError(f"No loader found for extension {path.suffix}")

        loader = self._dictionary_loaders.get(source_type)
        if loader is None:
            raise ValueError(f"Unknown dictionary source type: {source_type}")

        # Load entries
        result = loader.load(path, column_mapping)
        if result.is_failure:
            raise ValueError(result.error)

        entries, stats = result.unwrap()

        # Index entries
        self._index_dictionary(entries)

        return stats

    def _index_dictionary(self, entries: Sequence[DictionaryEntry]) -> None:
        """Index dictionary entries for search."""
        # Store entries
        self._dictionary_entries = {e.id: e for e in entries}

        # Generate embeddings
        texts = [e.to_searchable_text() for e in entries]
        embed_result = self._embedding_provider.embed(texts)
        if embed_result.is_failure:
            raise RuntimeError(f"Embedding failed: {embed_result.error}")

        embeddings = embed_result.unwrap()

        # Index in vector store
        docs = [
            VectorDocument(
                id=entry.id,
                embedding=embeddings.embeddings[i],
                payload={
                    "business_name": entry.business_name,
                    "logical_name": entry.logical_name,
                    "data_type": entry.data_type.value,
                    "domain": entry.domain,
                },
            )
            for i, entry in enumerate(entries)
        ]

        upsert_result = self._vector_store.upsert(docs)
        if upsert_result.is_failure:
            raise RuntimeError(f"Vector indexing failed: {upsert_result.error}")

        # Index in sparse retriever
        if self._sparse_retriever:
            sparse_docs = [
                SparseDocument(
                    id=entry.id,
                    text=entry.to_searchable_text(),
                    metadata={"domain": entry.domain},
                )
                for entry in entries
            ]
            self._sparse_retriever.index(sparse_docs)

        self._is_initialized = True

    def match_schema(
        self,
        schema_source: str | Path | dict[str, Any],
        schema_format: str | None = None,
    ) -> dict[str, tuple[MatchResult, ...]]:
        """
        Match a schema against the loaded dictionary.

        Args:
            schema_source: Schema file path or content
            schema_format: Force specific parser (auto-detect if None)

        Returns:
            Dictionary mapping field paths to match results
        """
        if not self._is_initialized:
            raise RuntimeError("Dictionary not loaded. Call load_dictionary() first.")

        # Parse schema
        schema = self._parse_schema(schema_source, schema_format)

        # Match each field
        results: dict[str, tuple[MatchResult, ...]] = {}

        for field in schema.fields:
            field_results = self._match_field(field)
            results[field.full_path] = tuple(field_results)

        return results

    def match_schema_session(
        self,
        schema_source: str | Path | dict[str, Any],
        schema_format: str | None = None,
    ) -> MatchingSession:
        """
        Match schema and return full session with metadata.

        Args:
            schema_source: Schema file path or content
            schema_format: Force specific parser

        Returns:
            Complete MatchingSession with all results and metrics
        """
        start_time = time.time()

        schema = self._parse_schema(schema_source, schema_format)
        results = self.match_schema(schema_source, schema_format)

        duration_ms = (time.time() - start_time) * 1000

        return MatchingSession(
            session_id=EntityId(),
            schema=schema,
            results=results,
            total_duration_ms=duration_ms,
        )

    def _parse_schema(
        self,
        source: str | Path | dict[str, Any],
        format_name: str | None = None,
    ) -> Schema:
        """Parse schema from source."""
        # If dict, try to detect parser
        if isinstance(source, dict):
            for parser in self._schema_parsers.values():
                if parser.can_parse(source):
                    result = parser.parse(source)
                    if result.is_success:
                        return result.unwrap()
            raise ValueError("Could not parse schema dict")

        # If path
        path = Path(source)

        # Auto-detect parser
        if format_name is None:
            for name, parser in self._schema_parsers.items():
                if path.suffix.lower() in parser.file_extensions:
                    format_name = name
                    break

        if format_name is None:
            raise ValueError(f"No parser found for extension {path.suffix}")

        parser = self._schema_parsers.get(format_name)
        if parser is None:
            raise ValueError(f"Unknown schema format: {format_name}")

        result = parser.parse_file(path)
        if result.is_failure:
            raise ValueError(result.error)

        return result.unwrap()

    def _match_field(self, field: SchemaField) -> list[MatchResult]:
        """Match a single field against the dictionary."""
        start_time = time.time()

        # Generate enriched query text with hierarchical context (GAP-006)
        # Research: Context injection is non-negotiable for nested schemas
        # "For user.addresses[].street_name, include 'user entity, addresses array, street_name field'"
        enriched_query = self._context_enricher.enrich(field)
        
        # Apply abbreviation expansion
        expanded = self._abbreviation_expander.expand(enriched_query)
        query_text = expanded.expanded

        # Generate embedding
        embed_result = self._embedding_provider.embed_single(query_text)
        if embed_result.is_failure:
            return []

        query_embedding = embed_result.unwrap()

        # Dense retrieval
        dense_results = self._vector_store.search(
            query_embedding,
            top_k=self._config.dense_top_k,
        )
        if dense_results.is_failure:
            return []

        dense_candidates = dense_results.unwrap()

        # Sparse retrieval (if available)
        sparse_candidates: dict[str, float] = {}
        if self._sparse_retriever:
            sparse_result = self._sparse_retriever.search(
                query_text,
                top_k=self._config.sparse_top_k,
            )
            if sparse_result.is_success:
                for sr in sparse_result.unwrap():
                    sparse_candidates[sr.id] = sr.score

        # Fuse results
        fused = self._fuse_results(dense_candidates, sparse_candidates)

        # Rerank if available
        if self._reranker and fused:
            candidates = [
                RerankCandidate(
                    id=doc_id,
                    text=self._dictionary_entries[doc_id].to_searchable_text(),
                    initial_score=score,
                )
                for doc_id, score in fused[:self._config.colbert_top_k]
            ]

            rerank_result = self._reranker.rerank(
                query_text,
                candidates,
                top_k=self._config.cross_encoder_top_k,
            )

            if rerank_result.is_success:
                reranked = rerank_result.unwrap()
                fused = [(r.id, r.score) for r in reranked]

        # Score and create results
        latency_ms = (time.time() - start_time) * 1000
        results: list[MatchResult] = []

        for rank, (doc_id, retrieval_score) in enumerate(fused[:self._config.results_per_field], 1):
            entry = self._dictionary_entries.get(doc_id)
            if entry is None:
                continue

            # Calculate detailed scores
            score_breakdown = self._calculate_scores(
                field, entry, query_embedding, retrieval_score
            )

            # Calculate final confidence
            final_confidence = self._calculate_final_confidence(score_breakdown)

            # Determine decision
            decision = self._determine_decision(final_confidence, rank, results)

            results.append(MatchResult(
                schema_field=field,
                dictionary_entry=entry,
                rank=rank,
                final_confidence=final_confidence,
                score_breakdown=score_breakdown,
                decision=decision,
                performance=PerformanceMetrics(
                    latency_ms=latency_ms,
                    cache_hit=False,
                    retrieval_stage="reranked" if self._reranker else "fused",
                    candidates_evaluated=len(fused),
                    reranking_applied=self._reranker is not None,
                ),
            ))

        return results

    def _fuse_results(
        self,
        dense: list[SearchResult],
        sparse: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Fuse dense and sparse retrieval results."""
        # Convex combination fusion
        alpha = self._config.fusion_alpha

        # Normalize scores
        max_dense = max((r.score for r in dense), default=1.0)
        max_sparse = max(sparse.values(), default=1.0) if sparse else 1.0

        fused_scores: dict[str, float] = {}

        # Add dense scores
        for r in dense:
            normalized = r.score / max_dense if max_dense > 0 else 0
            fused_scores[r.id] = alpha * normalized

        # Add sparse scores
        for doc_id, score in sparse.items():
            normalized = score / max_sparse if max_sparse > 0 else 0
            if doc_id in fused_scores:
                fused_scores[doc_id] += (1 - alpha) * normalized
            else:
                fused_scores[doc_id] = (1 - alpha) * normalized

        # Sort by fused score
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results

    def _calculate_scores(
        self,
        field: SchemaField,
        entry: DictionaryEntry,
        query_embedding: np.ndarray,
        retrieval_score: float,
    ) -> ScoreBreakdown:
        """Calculate detailed score breakdown."""
        # Semantic score from retrieval
        semantic_score = min(retrieval_score, 1.0)

        # Lexical score (simple token overlap)
        field_tokens = set(field.name.lower().replace("_", " ").split())
        entry_tokens = set(entry.logical_name.lower().replace("_", " ").split())
        entry_tokens.update(entry.business_name.lower().split())

        if field_tokens and entry_tokens:
            intersection = field_tokens & entry_tokens
            lexical_score = len(intersection) / max(len(field_tokens), 1)
        else:
            lexical_score = 0.0

        # Edit distance score
        edit_distance_score = self._edit_distance_score(
            field.name.lower(),
            entry.logical_name.lower(),
        )

        # Type compatibility
        type_score = entry.matches_type(field.data_type)

        # Domain score using domain matcher
        domain_score = self._calculate_domain_score(field, entry)

        return ScoreBreakdown(
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            edit_distance_score=edit_distance_score,
            type_compatibility_score=type_score,
            domain_score=domain_score,
        )

    def _calculate_domain_score(
        self,
        field: SchemaField,
        entry: DictionaryEntry,
    ) -> float:
        """
        Calculate domain compatibility score.

        Uses domain hierarchy matching when domain info is available,
        falls back to neutral score otherwise.
        """
        # Get entry domain (from dictionary)
        entry_domain = entry.domain

        # Try to infer field domain from:
        # 1. source_metadata if available
        # 2. parent_path inference
        # 3. field name patterns
        field_domain = field.source_metadata.get("domain")

        if not field_domain:
            # Try to infer from parent path
            field_domain = self._infer_domain_from_path(field.parent_path)

        if not field_domain:
            # Try to infer from field name
            field_domain = self._infer_domain_from_name(field.name)

        # If we have both domains, compute score
        if field_domain and entry_domain:
            return self._domain_matcher.score(field_domain, entry_domain)

        # If only entry has domain, give partial credit
        if entry_domain:
            return 0.5

        # No domain info available - neutral score
        return 0.5

    def _infer_domain_from_path(self, path: str) -> str | None:
        """Infer domain from field path."""
        if not path:
            return None

        # Common path patterns that indicate domain
        path_lower = path.lower()

        domain_patterns = {
            "account": "ACCOUNTS",
            "transaction": "TRANSACTIONS",
            "txn": "TRANSACTIONS",
            "balance": "BALANCES",
            "customer": "CUSTOMER",
            "cust": "CUSTOMER",
            "payment": "PAYMENTS",
            "card": "CARDS",
            "loan": "LOANS",
            "mortgage": "MORTGAGES",
            "address": "ADDRESS",
            "contact": "CONTACT",
            "phone": "PHONE",
            "email": "EMAIL",
            "product": "PRODUCT",
            "compliance": "COMPLIANCE",
            "kyc": "KYC",
            "aml": "AML",
        }

        for pattern, domain in domain_patterns.items():
            if pattern in path_lower:
                return domain

        return None

    def _infer_domain_from_name(self, name: str) -> str | None:
        """Infer domain from field name."""
        if not name:
            return None

        name_lower = name.lower()

        # Strong indicators in field names
        name_patterns = {
            "acct": "ACCOUNTS",
            "account": "ACCOUNTS",
            "txn": "TRANSACTIONS",
            "transaction": "TRANSACTIONS",
            "bal": "BALANCES",
            "balance": "BALANCES",
            "cust": "CUSTOMER",
            "customer": "CUSTOMER",
            "addr": "ADDRESS",
            "address": "ADDRESS",
            "payment": "PAYMENTS",
            "pmt": "PAYMENTS",
        }

        for pattern, domain in name_patterns.items():
            if pattern in name_lower:
                return domain

        return None

    def _edit_distance_score(self, s1: str, s2: str) -> float:
        """Calculate normalized edit distance score."""
        if not s1 or not s2:
            return 0.0

        # Levenshtein distance
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        distance = dp[m][n]
        max_len = max(m, n)

        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    def _calculate_final_confidence(self, scores: ScoreBreakdown) -> float:
        """Calculate weighted final confidence score."""
        config = self._config

        final = (
            config.semantic_weight * scores.semantic_score
            + config.lexical_weight * scores.lexical_score
            + config.edit_distance_weight * scores.edit_distance_score
            + config.type_weight * scores.type_compatibility_score
            + config.domain_weight * scores.domain_score
        )

        return min(max(final, 0.0), 1.0)

    def _determine_decision(
        self,
        confidence: float,
        rank: int,
        existing_results: list[MatchResult],
    ) -> MatchDecision:
        """Determine match decision based on confidence and gap."""
        config = self._config

        # Auto-approve if high confidence and significant gap
        if confidence >= config.auto_approve_threshold:
            if rank == 1:
                return MatchDecision.AUTO_APPROVE
            # Check gap from top result
            if existing_results:
                gap = existing_results[0].final_confidence - confidence
                if gap >= config.min_confidence_gap:
                    return MatchDecision.REJECT

        # Review if medium confidence
        if confidence >= config.review_threshold:
            return MatchDecision.REVIEW

        return MatchDecision.REJECT

    @property
    def dictionary_size(self) -> int:
        """Get number of loaded dictionary entries."""
        return len(self._dictionary_entries)

    @property
    def is_ready(self) -> bool:
        """Check if matcher is ready for queries."""
        return self._is_initialized
