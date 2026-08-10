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

import dataclasses
import json
import math
import operator
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nexus_matcher.core.fusion import fuse_linear_ids
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
from nexus_matcher.domain.services.alias_generation import expand_dictionary
from nexus_matcher.domain.services.context_enricher import ContextEnricher
from nexus_matcher.domain.services.domain_hierarchy import DomainMatcher
from nexus_matcher.shared.types.base import (
    EntityId,
    MatchDecision,
    PerformanceMetrics,
    ScoreBreakdown,
)

# Optional acceleration for the per-candidate string similarity on the hot path.
try:  # pragma: no cover - trivial import guard
    from rapidfuzz.distance import Levenshtein as _LEVENSHTEIN
except ImportError:  # pragma: no cover
    _LEVENSHTEIN = None  # type: ignore[assignment]

# C-level sort keys. These run once per candidate per field, where a lambda's frame setup
# is measurable against the handful of float operations it guards.
_second = operator.itemgetter(1)
_first = operator.itemgetter(0)


# =============================================================================
# MATCHING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for the matching process."""

    # Retrieval
    dense_top_k: int = 100
    sparse_top_k: int = 100
    # Weight for dense scores in linear fusion. 0.9 is the measured optimum for P@1 on
    # the combined BIRD+OMOP benchmark (see _fuse_results and benchmarks/exp_fusion.py):
    # dense=0.5 -> 0.657, 0.7 -> 0.682, 0.8 -> 0.686, 0.9 -> 0.702.
    # The lexical arm still earns its 10%: sparse-only reaches 0.542 P@1 on its own and
    # rescues exact-token matches that the embedding model misses.
    fusion_alpha: float = 0.90

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
    #
    # auto_approve_threshold is CALIBRATED, not arbitrary. Measured on the combined
    # BIRD+OMOP benchmark (688 labelled fields, benchmarks/exp_calibration.py):
    #
    #   threshold   coverage   auto-approve precision
    #   0.85          20.8%       0.916
    #   0.87          12.4%       0.953      <- default: ~95% precision target
    #
    # These are for the shipped configuration (dictionary_alias_count = 0). Turning
    # aliasing on also caps achievable auto-approve precision at ~91%, because max-pool
    # reports an entry's BEST fabricated spelling and a wrong entry can therefore look
    # confident -- a second, independent reason to leave it off.
    #
    # Auto-approving a wrong mapping is far more expensive than sending a field to
    # review, so the default targets ~95% auto-approve precision and accepts low
    # coverage; everything below the bar goes to a human rather than being guessed.
    # If your cost balance differs, 0.85 buys +8 points of coverage for -4 of precision.
    #
    # IMPORTANT: these numbers move with the retriever AND with the benchmark. They were
    # re-derived after a leakage fix in the OMOP split (its business name used to be
    # derived from the field name, making the task string-identity and inflating every
    # number downstream). Improving retrieval also SHIFTS THE SCORE DISTRIBUTION UPWARD,
    # pushing more candidates over a fixed bar and LOWERING precision -- that happened
    # twice during tuning. Re-run exp_calibration.py after any change to the model,
    # fusion, query representation, or benchmark.
    auto_approve_threshold: float = 0.87
    review_threshold: float = 0.50
    min_confidence_gap: float = 0.10

    # Results
    results_per_field: int = 5

    # Whether to run abbreviation expansion over the QUERY text.
    # Off by default: it measurably LOSES accuracy on the combined benchmark
    # (P@1 0.691 with context enrichment alone vs 0.671 with expansion added,
    # benchmarks/exp_query_repr.py). Once the parent-table context is present the
    # embedding model can already resolve most abbreviations from context, while the
    # expander's fixed dictionary fires on ambiguous short tokens and injects wrong
    # words -- "st" -> "state" inside a street field, and so on. A wrong expansion is
    # worse than no expansion. The expander is still used to enrich the DICTIONARY side
    # where entries are longer and more predictable.
    expand_query_abbreviations: bool = False

    # Index-time enrichment: how many fabricated technical spellings to index per
    # dictionary entry, max-pooled at query time. See domain/services/alias_generation.
    #
    # This is the inverse of query expansion, and unlike query expansion it works: a
    # wrong alias loses the max-pool instead of corrupting the single query vector.
    # OFF BY DEFAULT. It helps on a small dictionary and is CATASTROPHIC on a large one.
    #
    # Measured end-to-end, aliases=0 vs aliases=6 (benchmarks/results/exp_alias_scale.json):
    #     entries    off      on       delta
    #        688   0.5814  0.6003     +1.9
    #     10,000   0.5044  0.3677    -13.7
    #     30,000   0.4666  0.2791    -18.8
    #
    # The gain inverts between 688 and 10k entries and then keeps falling. The mechanism
    # is inherent to max-pooling, not a tuning problem: every DISTRACTOR also receives
    # `dictionary_alias_count` extra chances to beat the gold entry, so alias noise grows
    # with corpus size while the useful signal does not. A 6x larger index is a 6x larger
    # opportunity for a wrong entry to land inside the ~0.002 similarity margin that
    # separates right from wrong on this task.
    #
    # Enable ONLY if your dictionary is genuinely small (order 1000 entries) and you have
    # re-measured on your own data. On the 361-entry BIRD split it is worth +3.9 points.
    # Do not enable it on an enterprise glossary.
    dictionary_alias_count: int = 0

    @property
    def minimum_achievable_confidence(self) -> float:
        """
        The structural floor: the lowest `final_confidence` a rank-1 match can carry.

        For the shipped configuration this is **0.63**, and knowing it is not optional
        trivia. `MatchingSession.get_low_confidence_fields()` shipped with a default
        threshold of 0.6 -- below this floor -- so it returned an empty list on every
        schema ever matched, and told a governance lead there was nothing to review on
        a schema where nothing was trustworthy (DX-001, tests/museum/NM-0027). The floor
        was folklore; nothing computed it, so nothing could contradict the number.

        Why it exists. `fused_retrieval_score` is min-max normalised over the candidates
        retrieved for one field, so the best candidate's dense score normalises to
        exactly 1.0, and `fuse_linear_ids` renormalises the two arm weights to sum to 1 --
        so that candidate's fused score is at least `fusion_alpha`. It carries
        `semantic_weight` of the final confidence, and the other four signals cannot be
        negative. Hence:

            floor = semantic_weight * fusion_alpha = 0.70 * 0.90 = 0.63

        Move `fusion_alpha` and the floor moves with it, which is the proof that this
        score is rank-relative rather than a similarity: at fusion_alpha=0.5 the rank-1
        fused score drops to ~0.45 for the SAME embeddings.

        Two preconditions, stated because a bound that quietly does not hold is worse
        than no bound:

        1. **No reranker.** A reranker REPLACES the fused score with a squashed reranker
           score, which has no floor. `NexusMatcher.minimum_achievable_confidence`
           returns None when one is wired, rather than reporting a number that is wrong.
        2. **At least two distinct dense scores.** Min-max maps a constant map to all
           zeros, so a one-entry dictionary (or a perfect tie across every candidate)
           produces a fused score of 0.0 and there is no floor at all.

        Clamped to [0, 1] because `_weighted_confidence` clamps too: with a weight set
        that oversums, confidences pile up at 1.0 and so does the floor.
        """
        return min(max(self.semantic_weight * self.fusion_alpha, 0.0), 1.0)


def _load_matching_config(source: MatchingConfig | str | Path | None) -> MatchingConfig:
    """
    Coerce a config argument into a MatchingConfig.

    Accepts an instance (returned as-is), a path to a JSON or TOML file, or None for the
    calibrated defaults.

    An UNKNOWN KEY IS AN ERROR rather than a warning. Every field here is a tuned number
    whose default was measured; a typo like `auto_approve_treshold` would otherwise be
    dropped in silence and leave the user believing they had raised the bar while the
    matcher went on auto-approving at 0.87. A loud failure at startup is far cheaper than
    a mis-governed field discovered in an audit.
    """
    if source is None:
        return MatchingConfig()
    if isinstance(source, MatchingConfig):
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Matching config not found: {path}")

    if path.suffix.lower() == ".toml":
        # tomllib is stdlib from 3.11 only, and this package supports 3.10.
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - version-dependent
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"Reading {path.name} needs a TOML parser on Python "
                    f"{sys.version_info.major}.{sys.version_info.minor}. Either install "
                    f"one (pip install tomli) or use a JSON config file, which needs "
                    f"nothing extra."
                ) from None
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    # Tolerate a wrapping table so one project file can hold several sections.
    if "matching" in data and isinstance(data["matching"], dict):
        data = data["matching"]

    known = {f.name for f in dataclasses.fields(MatchingConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown matching config option(s) in {path}: {', '.join(unknown)}. "
            f"Valid options: {', '.join(sorted(known))}"
        )
    return MatchingConfig(**data)


# =============================================================================
# TOKENIZATION
# =============================================================================

# Splits identifiers on case transitions, digit boundaries and any punctuation, so
# "NumTstTakr", "num_tst_takr" and "enroll12" all yield comparable token sets.
_IDENT_SPLIT = re.compile(
    r"""
    (?<=[a-z0-9])(?=[A-Z])      # camelCase   -> camel | Case
  | (?<=[A-Z])(?=[A-Z][a-z])    # HTTPResponse -> HTTP | Response
  | (?<=[A-Za-z])(?=[0-9])      # enroll12    -> enroll | 12
  | (?<=[0-9])(?=[A-Za-z])      # 12grade     -> 12 | grade
    """,
    re.VERBOSE,
)


def _tokenize_identifier(name: str) -> set[str]:
    """Split a schema identifier into a lowercase token set."""
    if not name:
        return set()
    spaced = _IDENT_SPLIT.sub(" ", name)
    return {t for t in re.split(r"[^0-9A-Za-z]+", spaced.lower()) if t}


# =============================================================================
# SCORE NORMALIZATION
# =============================================================================


# The five scored signals, in weight order: fused retrieval, lexical, edit distance,
# type, domain. This is NOT ScoreBreakdown's field order -- that dataclass declares two
# reranker fields between type and domain -- so widening one into the other goes through
# `_breakdown`, never through positional expansion.
#
# Signal 0 is weighted by `MatchingConfig.semantic_weight`, whose name is a survivor of
# the same confusion as the old `ScoreBreakdown.semantic_score`: it weights the fused
# RETRIEVAL score. The config field keeps its name because it is a calibrated knob that
# users set in JSON/TOML files, and renaming it would break those files silently.
#
# Carried as a plain tuple on the hot path so a candidate that never reaches the result
# list does not cost a dataclass; see _match_field.
Signals = tuple[float, float, float, float, float]


def _signal_weights(config: MatchingConfig) -> Signals:
    """The configured weight of each signal, in signal order."""
    return (
        config.semantic_weight,
        config.lexical_weight,
        config.edit_distance_weight,
        config.type_weight,
        config.domain_weight,
    )


def _breakdown(signals: Signals, absolute_cosine: float | None = None) -> ScoreBreakdown:
    """
    Widen the five signals into the public ScoreBreakdown.

    Keyword arguments are not stylistic here: ScoreBreakdown declares `colbert_score` and
    `cross_encoder_score` between the type and domain fields, so positional expansion
    would quietly land the domain score in `colbert_score` and report a reranker result
    that never ran.

    `absolute_cosine` is carried ALONGSIDE the five signals and never inside them: it is
    reported to the caller and is deliberately not an input to the weighted confidence,
    so adding it cannot move a ranking. See `_match_field` for where it comes from.
    """
    sem, lex, edit, type_, domain = signals
    return ScoreBreakdown(
        fused_retrieval_score=sem,
        lexical_score=lex,
        edit_distance_score=edit,
        type_compatibility_score=type_,
        domain_score=domain,
        absolute_cosine=absolute_cosine,
    )


def _weighted_confidence(signals: Signals, weights: Signals) -> float:
    """
    The final confidence: a weighted sum of the five signals, clamped to [0, 1].

    One definition, two callers -- the per-candidate ranking loop, which holds signals as
    a tuple, and `_calculate_final_confidence`, which holds them as a ScoreBreakdown.
    """
    sem, lex, edit, type_, domain = signals
    w_sem, w_lex, w_edit, w_type, w_domain = weights
    total = w_sem * sem + w_lex * lex + w_edit * edit + w_type * type_ + w_domain * domain
    return min(max(total, 0.0), 1.0)


def _squash_score(score: float) -> float:
    """
    Map an arbitrary reranker score into [0, 1].

    Scores already in [0, 1] (ColBERT cosine MaxSim, normalized rerankers) pass through
    unchanged so that well-behaved rerankers keep their calibration. Anything outside
    that range is assumed to be a raw logit and goes through a logistic sigmoid, which
    is monotonic -- so it never changes the candidate ORDER, only the magnitudes that
    feed the weighted confidence and the auto-approve threshold.

    A per-query min-max normalization was deliberately rejected here: it would force
    the top candidate to exactly 1.0 for every field, making confidence meaningless
    across fields and defeating the whole point of a calibrated threshold.
    """
    if 0.0 <= score <= 1.0:
        return float(score)
    # Guard against overflow warnings on extreme logits.
    if score < -30.0:
        return 0.0
    if score > 30.0:
        return 1.0
    return float(1.0 / (1.0 + math.exp(-score)))


# =============================================================================
# RESULT IDENTITY
# =============================================================================


def field_result_key(field: SchemaField) -> str:
    """
    The name a caller uses to look this field up in a match result.

    Results come back as a dict, so every field needs a handle -- and the handle has to
    be the string the CALLER supplied, not one we derived. The flattened parser
    deliberately rewrites `cust_addr__city` into the dotted path `cust.addr.city`,
    because recovering the parent path is worth +19.3 P@1; keyed by that path, a caller
    asking for their own column name got a KeyError from a result set that did contain
    their field, and a caller iterating the keys saw names their schema never used.

    `flattened_name` is that original string, and both flattened entry points set it.
    Every other parser -- raw Avro, JSON Schema, SQL DDL -- has no such name and keeps
    its dotted `full_path`, which is what those callers have always addressed.
    """
    flattened = field.source_metadata.get("flattened_name")
    if isinstance(flattened, str) and flattened:
        return flattened
    # __post_init__ guarantees full_path falls back to the bare field name.
    return field.full_path


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
        self._indexed_ids: set[str] = set()
        # Maps a synthetic alias document id back to the entry that owns it.
        self._alias_owner: dict[str, str] = {}
        self._is_initialized = False

    @classmethod
    def from_config(cls, config: MatchingConfig | str | Path | None = None) -> NexusMatcher:
        """
        Build a fully wired matcher, with no components to assemble by hand.

        Args:
            config: A `MatchingConfig`, or a path to a JSON/TOML file holding its fields,
                or None for the calibrated defaults.

        Returns:
            Configured NexusMatcher instance.

        Example:
            matcher = NexusMatcher.from_config()
            matcher = NexusMatcher.from_config(MatchingConfig(auto_approve_threshold=0.85))
            matcher = NexusMatcher.from_config("matching.json")

        Note:
            This parameter used to be named `config_path` and was accepted and then never
            read, so passing a tuned config file silently produced default thresholds --
            a failure that is invisible precisely because the matcher still works.
        """
        matching_config = _load_matching_config(config)
        # Import infrastructure components
        from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import (
            CsvDictionaryLoader,
            ExcelDictionaryLoader,
        )
        from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
            default_embedding_provider,
        )
        from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
        from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
            FlattenedAvroParser,
        )
        from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
        from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore

        # Resolve the best provider AVAILABLE rather than hardcoding one.
        #
        # This used to construct SentenceTransformersProvider() directly, which needs
        # torch. The result was that every documented entry point -- the README
        # quickstart, `nexus-matcher match`, `nexus-matcher sync` -- raised
        # ImportError on a plain `pip install nexus-matcher`, while the 33.8 MB encoder
        # bundled into the wheel specifically to make that work was never reached.
        # default_embedding_provider() prefers the bundled offline encoder and falls
        # back to sentence-transformers when the extra is installed.
        embedding_provider = default_embedding_provider()
        vector_store = InMemoryVectorStore(
            VectorStoreConfig(
                collection_name="dictionary",
                dimension=embedding_provider.dimension,
            )
        )
        sparse_retriever = BM25Retriever()

        # Register parsers and loaders
        schema_parsers = {
            "avro": AvroSchemaParser(),
            # The production input shape: a flattened schema, not a nested .avsc.
            "flattened_avro": FlattenedAvroParser(),
        }
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
            config=matching_config,
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

        # Index-time enrichment: extra rows holding fabricated technical spellings of
        # each entry's business name. They are indexed as ordinary vectors but carry a
        # synthetic id, and `_alias_owner` maps them back so retrieval can max-pool per
        # entry. See domain/services/alias_generation for why this is done on the
        # dictionary side rather than by expanding the query.
        self._alias_owner = {}
        alias_rows: list[tuple[str, str]] = []
        if self._config.dictionary_alias_count > 0:
            alias_rows = expand_dictionary(
                [(e.id, e.business_name) for e in entries],
                max_aliases=self._config.dictionary_alias_count,
            )

        # Generate embeddings for the primary texts and the aliases in ONE batch.
        texts = [e.to_searchable_text() for e in entries]
        texts.extend(alias for _, alias in alias_rows)

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

        offset = len(entries)
        for j, (owner_id, alias) in enumerate(alias_rows):
            alias_id = f"{owner_id}\x00alias{j}"
            self._alias_owner[alias_id] = owner_id
            owner = self._dictionary_entries[owner_id]
            docs.append(
                VectorDocument(
                    id=alias_id,
                    embedding=embeddings.embeddings[offset + j],
                    payload={
                        "business_name": owner.business_name,
                        "logical_name": owner.logical_name,
                        "data_type": owner.data_type.value,
                        "domain": owner.domain,
                        "alias_of": owner_id,
                        "alias_text": alias,
                    },
                )
            )

        # Drop any previously indexed vectors first. `_dictionary_entries` is REPLACED
        # above, but the vector store was only ever upserted into, so loading a second
        # dictionary left the first one's vectors searchable while their entries were no
        # longer resolvable -- producing silent misses and stale matches.
        clear_result = self._vector_store.delete(list(self._indexed_ids))
        if clear_result.is_failure:
            raise RuntimeError(f"Failed to clear previous index: {clear_result.error}")

        upsert_result = self._vector_store.upsert(docs)
        if upsert_result.is_failure:
            raise RuntimeError(f"Vector indexing failed: {upsert_result.error}")

        self._indexed_ids = {entry.id for entry in entries} | set(self._alias_owner)

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
            # index() REPLACES the sparse index, so no explicit clear is needed here.
            # The Result was previously discarded: a failed sparse build left the
            # matcher running dense-only with no indication anything had gone wrong.
            sparse_result = self._sparse_retriever.index(sparse_docs)
            if sparse_result.is_failure:
                raise RuntimeError(f"Sparse indexing failed: {sparse_result.error}")

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
            One entry per parsed field, in schema order, keyed by the name the caller
            used for it -- the original flattened column name for a flattened schema,
            the dotted `full_path` for every other format. See `field_result_key`.
        """
        if not self._is_initialized:
            raise RuntimeError("Dictionary not loaded. Call load_dictionary() first.")

        # Parse schema
        schema = self._parse_schema(schema_source, schema_format)

        return self._match_fields(schema.fields)

    def _match_fields(
        self,
        fields: Sequence[SchemaField],
    ) -> dict[str, tuple[MatchResult, ...]]:
        """
        Match a batch of fields, encoding all queries in a single call.

        Transformer encoders are throughput-bound by batch size on CPU: encoding one
        text at a time reaches ~128 texts/sec where a batch of 128 reaches ~1690.
        Building every query string first and embedding them together is what turns a
        per-field loop into a batched pipeline.

        Returns exactly one entry per field, in input order. That count is the contract:
        see `_unique_result_key` for what used to happen when it did not hold.
        """
        if not fields:
            return {}

        query_texts = [self._build_query_text(f) for f in fields]

        embeddings: list[np.ndarray] | None = None
        embed_result = self._embedding_provider.embed(query_texts)
        if embed_result.is_success:
            batch = embed_result.unwrap()
            embeddings = [batch.embeddings[i] for i in range(len(query_texts))]

        dense_per_field = self._search_dense_batch(embeddings)

        results: dict[str, tuple[MatchResult, ...]] = {}
        for i, field in enumerate(fields):
            field_results = self._match_field(
                field,
                query_text=query_texts[i],
                query_embedding=embeddings[i] if embeddings is not None else None,
                dense_candidates=dense_per_field[i] if dense_per_field is not None else None,
            )
            results[self._unique_result_key(field, results)] = tuple(field_results)

        return results

    def _search_dense_batch(
        self, embeddings: list[np.ndarray] | None
    ) -> list[list[SearchResult]] | None:
        """
        Retrieve dense candidates for every field in ONE call, when the store supports it.

        The encoder was already batched; dense retrieval was not. Scoring one query is a
        matrix-VECTOR product that streams the whole corpus matrix out of RAM, so a
        688-field schema read a 153 MB matrix 688 times. Batching turns those into one
        matrix-MATRIX product per chunk, which reads the corpus once and reuses each
        cache line across every query in the block: measured 2.98x on the real FHIR
        dictionary (363 us -> 122 us per query).

        Returns None when batching is unavailable or fails, and the caller falls back to
        per-field search. That fallback is not decoration -- only InMemoryVectorStore
        implements search_batch today; Qdrant and HNSW do not, and a store supplied by a
        caller certainly need not. Degrading quietly here is right because the per-field
        path returns identical results, just slower.
        """
        if embeddings is None:
            return None
        search_batch = getattr(self._vector_store, "search_batch", None)
        if search_batch is None:
            return None

        batched = search_batch(embeddings, top_k=self._config.dense_top_k)
        if batched.is_failure:
            return None
        rows = batched.unwrap()
        # One row per query is the contract; anything else means the store disagrees with
        # us about it, and silently zipping mismatched lists would hand a field another
        # field's candidates -- the same class of defect as the result-key collision.
        if len(rows) != len(embeddings):
            return None
        return rows

    @staticmethod
    def _unique_result_key(field: SchemaField, taken: Mapping[str, Any]) -> str:
        """
        A key for `field` that cannot displace one already in `taken`.

        Results were keyed by `full_path`, which is NOT unique. The flattened parser maps
        both `contact__email` (an array of contacts) and `contact_email` (a scalar
        column) -- two legal, distinct fields of a single Avro record -- onto
        `contact.email`, so the second overwrote the first. `match_schema` then returned
        fewer entries than it was given: no exception, no warning, just a column absent
        from the results, inheriting no protection level, in a library whose entire job
        is to make a field inherit one. The only visible symptom was a count nobody had
        reason to check.

        Keying by the caller's own name removes the common case. A genuine duplicate --
        the same column listed twice in an export -- still has to go somewhere, so it
        takes a `#2`, `#3`, ... suffix. `#` occurs in neither an Avro name nor a dotted
        path, so a suffixed key reads as synthetic rather than as a real column, and each
        MatchResult still carries the SchemaField it belongs to.
        """
        key = field_result_key(field)
        if key not in taken:
            return key

        n = 2
        while f"{key}#{n}" in taken:
            n += 1
        return f"{key}#{n}"

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
        if not self._is_initialized:
            raise RuntimeError("Dictionary not loaded. Call load_dictionary() first.")

        start_time = time.time()

        # Parse once and match the parsed fields directly. Calling match_schema() here
        # re-parsed the same source a second time, doubling parse cost and risking a
        # mismatch between the returned schema and the results computed from it.
        schema = self._parse_schema(schema_source, schema_format)
        results = self._match_fields(schema.fields)

        duration_ms = (time.time() - start_time) * 1000

        return MatchingSession(
            session_id=EntityId(),
            schema=schema,
            results=results,
            total_duration_ms=duration_ms,
            # Carried into the session so `get_low_confidence_fields()` can REFUSE a
            # threshold no match could ever fall below, instead of returning [] and
            # reading as "nothing to review". The session is a domain object and must
            # not import MatchingConfig, so it gets the derived number, not the config.
            minimum_achievable_confidence=self._session_confidence_floor(results),
        )

    def _session_confidence_floor(
        self, results: Mapping[str, Sequence[MatchResult]]
    ) -> float | None:
        """
        The floor for THIS session, or None when the derivation's preconditions did not
        hold for it.

        `MatchingConfig.minimum_achievable_confidence` is a bound, and a bound has
        preconditions: no reranker, and at least two distinct dense scores so that min-max
        normalisation maps the top candidate to 1.0 rather than to 0.0. The second one is
        easy to violate by ordinary means -- a one-entry dictionary, `dense_top_k=1`, or a
        perfect tie across every candidate -- and when it is violated the real confidences
        sit around 0.13 while the config still reports 0.63.

        Handing that number to the session was worse than not having it. Every threshold in
        the range where the fields actually were got REFUSED, with a message asserting no
        match could fall below it, on precisely the sessions where they all did. That is
        NM-0027's own failure -- an API telling a reviewer there is nothing to see -- coming
        back as an exception instead of an empty list.

        So the bound is checked against the data it claims to bound. If any top match sits
        below it, the precondition demonstrably did not hold and the session reports no
        floor. A self-verifying claim cannot be wrong about its own session.
        """
        floor = self.minimum_achievable_confidence
        if floor is None:
            return None
        tops = [matches[0].final_confidence for matches in results.values() if matches]
        if not tops:
            # Nothing matched, so there is nothing to bound and nothing to refuse against.
            return None
        return floor if min(tops) >= floor else None

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

    def _build_query_text(self, field: SchemaField) -> str:
        """
        Build the retrieval query text for a field.

        Hierarchical context is injected first (GAP-006): a bare field name like
        `sname` carries almost no signal, while `satscores sname` is unambiguous. On the
        combined BIRD+OMOP benchmark this parent-path context is worth +20 points of
        P@1 -- by far the largest single accuracy factor in the pipeline.
        """
        enriched_query = self._context_enricher.enrich(field)

        if not self._config.expand_query_abbreviations:
            return enriched_query

        return self._abbreviation_expander.expand(enriched_query).expanded

    def _match_field(
        self,
        field: SchemaField,
        query_text: str | None = None,
        query_embedding: np.ndarray | None = None,
        dense_candidates: list[SearchResult] | None = None,
    ) -> list[MatchResult]:
        """
        Match a single field against the dictionary.

        `query_text` and `query_embedding` may be supplied by the caller so that a whole
        schema's embeddings can be produced in one batched encoder call; encoding fields
        one at a time costs roughly 13x throughput on CPU.
        """
        start_time = time.time()

        if query_text is None:
            query_text = self._build_query_text(field)

        if query_embedding is None:
            embed_result = self._embedding_provider.embed_single(query_text)
            if embed_result.is_failure:
                return []
            query_embedding = embed_result.unwrap()

        # Dense retrieval. `dense_candidates` arrives pre-computed when a whole schema was
        # retrieved in one batched call; searching per field is the fallback, and the
        # single-field public path.
        if dense_candidates is None:
            dense_results = self._vector_store.search(
                query_embedding,
                top_k=self._config.dense_top_k,
            )
            if dense_results.is_failure:
                return []
            dense_candidates = dense_results.unwrap()

        # Collapse alias hits onto the entry that owns them, keeping the BEST score per
        # entry (max-pool). Without this an entry could occupy several of the top-k slots
        # with different spellings of itself, crowding out real alternatives.
        # Primary-document score per entry, tracked separately from the pooled score
        # so the stricter confidence policy documented below remains available.
        primary_scores: dict[str, float] = {}
        if self._alias_owner:
            best: dict[str, SearchResult] = {}
            for r in dense_candidates:
                owner = self._alias_owner.get(r.id, r.id)
                if r.id == owner:
                    # This hit is the entry's own text, not a fabricated alias.
                    primary_scores[owner] = r.score
                current = best.get(owner)
                if current is None or r.score > current.score:
                    best[owner] = SearchResult(
                        id=owner, score=r.score, payload=r.payload, embedding=r.embedding
                    )
            dense_candidates = sorted(best.values(), key=lambda r: r.score, reverse=True)

        # The RAW dense score per entry, captured before fusion normalises it away.
        #
        # Under the shipped wiring this is the cosine similarity between the query vector
        # and the entry vector, and it is the only number in the breakdown an auditor can
        # compare ACROSS fields -- `fused_retrieval_score` is rescaled per field, so its
        # 0.9 says "won this field's shortlist", not "is 90% similar". It was computed on
        # every match and then discarded, so "how similar were they really?" had no
        # answer anywhere in the API. Reported only; never an input to the confidence.
        absolute_cosines = {r.id: r.score for r in dense_candidates}

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
                for doc_id, score in fused[: self._config.colbert_top_k]
            ]

            rerank_result = self._reranker.rerank(
                query_text,
                candidates,
                top_k=self._config.cross_encoder_top_k,
            )

            if rerank_result.is_success:
                reranked = rerank_result.unwrap()
                # Reranker scores are UNBOUNDED (cross-encoder logits can be negative or
                # far above 1; ColBERT MaxSim is a sum over query tokens). They feed
                # `fused_retrieval_score`, which carries 70% of the confidence weight, so
                # they must be squashed into [0, 1] first. Without this, min(score, 1.0)
                # saturated every plausible candidate to the same value and turned
                # `final_confidence` into a sign test on the raw logit.
                fused = [(r.id, _squash_score(r.score)) for r in reranked]

        # Score every surviving candidate, then rank by the multi-signal confidence.
        # Previously the loop scored only the first `results_per_field` entries and
        # emitted them in retrieval order, so the lexical/edit/type/domain signals
        # (30% of the weight) could never change which entry came back first.
        candidate_pool = fused[
            : max(self._config.cross_encoder_top_k, self._config.results_per_field)
        ]

        weights = _signal_weights(self._config)

        scored: list[tuple[float, Signals, DictionaryEntry, float | None]] = []
        for doc_id, retrieval_score in candidate_pool:
            entry = self._dictionary_entries.get(doc_id)
            if entry is None:
                continue

            # NOTE on aliasing and confidence. Ranking uses the max-pooled score, which
            # reports an entry's best-matching fabricated alias. Confidence uses it too,
            # and that is a deliberate compromise rather than an oversight:
            #
            #   pooled score -> confidence : ~91% max auto-approve precision, 15% coverage
            #   primary score -> confidence: ~95% precision but only 2.9% coverage
            #
            # Routing confidence through the primary text alone is more principled -- an
            # entry that only won via a fabricated spelling genuinely deserves less
            # confidence -- but it suppresses precisely the fields aliasing rescued, and
            # coverage collapses. `primary_scores` is computed and kept available for
            # callers that want the stricter behaviour.
            #
            # If a >=95% auto-approve guarantee matters more than ranking quality, set
            # dictionary_alias_count = 0: that restores 0.953 precision at 12.4% coverage
            # at the cost of ~2 points of P@1 (~4 on abbreviation-heavy schemas).
            signals = self._score_signals(field, entry, retrieval_score)
            scored.append(
                (
                    _weighted_confidence(signals, weights),
                    signals,
                    entry,
                    absolute_cosines.get(doc_id),
                )
            )

        # Stable sort by confidence descending; ties keep upstream retrieval order, which
        # `fuse_linear_ids` defines as dense rank rather than leaving it to string hashing.
        scored.sort(key=_first, reverse=True)

        latency_ms = (time.time() - start_time) * 1000
        results: list[MatchResult] = []

        for rank, (final_confidence, signals, entry, absolute_cosine) in enumerate(
            scored[: self._config.results_per_field], 1
        ):
            runner_up = scored[rank][0] if rank < len(scored) else None
            decision = self._determine_decision(final_confidence, rank, runner_up)

            results.append(
                MatchResult(
                    schema_field=field,
                    dictionary_entry=entry,
                    rank=rank,
                    final_confidence=final_confidence,
                    # Widened here, for the results that are actually returned.
                    score_breakdown=_breakdown(signals, absolute_cosine),
                    decision=decision,
                    performance=PerformanceMetrics(
                        latency_ms=latency_ms,
                        cache_hit=False,
                        retrieval_stage="reranked" if self._reranker else "fused",
                        candidates_evaluated=len(fused),
                        reranking_applied=self._reranker is not None,
                    ),
                )
            )

        return results

    def _fuse_results(
        self,
        dense: list[SearchResult],
        sparse: dict[str, float],
    ) -> list[tuple[str, float]]:
        """
        Fuse dense and sparse retrieval results.

        Uses weighted linear combination over MIN-MAX normalized scores, delegating to
        core.fusion so the arithmetic lives in one place. `fuse_linear_ids` is the same
        fusion as `HybridFuser`/`fuse_linear` with the per-item `ScoredItem` provenance
        left unbuilt -- this call site only ever kept the id and the score.

        Two deliberate choices, both measured on the combined BIRD+OMOP benchmark
        (benchmarks/exp_fusion.py, 793 labelled queries):

        1. NOT Reciprocal Rank Fusion, despite RRF being the conventional default. RRF
           keeps only rank and throws away score magnitude, so a confidently-correct
           dense hit gets averaged against a confidently-wrong lexical hit. Measured
           P@1: rrf 0.610, dense-alone 0.691, linear 0.702. RRF was the WORST method
           tried and was worse than using no fusion at all.

        2. Min-max rather than max-only normalization. Dividing by the max alone leaves
           a floor that varies per query, so scores are not comparable across the two
           arms; BM25 scores in particular never approach zero for a matched query.

        The 0.9 dense weight is the sweep optimum for P@1. If you tune for recall
        instead (e.g. to feed a reranker), balanced combsum scored higher R@10 (0.923
        vs 0.911) -- re-run the sweep rather than assuming.
        """
        dense_results = [(r.id, r.score) for r in dense]
        sparse_results = sorted(sparse.items(), key=_second, reverse=True)

        alpha = self._config.fusion_alpha
        return fuse_linear_ids(
            dense_results,
            sparse_results,
            semantic_weight=alpha,
            lexical_weight=1.0 - alpha,
            normalize_scores=True,
        )

    def _score_signals(
        self,
        field: SchemaField,
        entry: DictionaryEntry,
        retrieval_score: float,
    ) -> Signals:
        """
        The five raw signals for one (field, entry) pair, in `Signals` order.

        Returns a tuple rather than a ScoreBreakdown because this runs for every
        candidate in the pool while only `results_per_field` of them are ever returned to
        a caller -- three quarters of the dataclasses built here used to be discarded
        unread a few lines later. `_breakdown` widens the survivors.

        The arithmetic is deliberately the same work, in the same order, as before that
        change: a scoring change that silently moves rankings does not belong in a
        refactor that was only meant to stop building objects nobody reads.
        """
        # The fused RETRIEVAL score, not a semantic similarity. It is min-max normalised
        # per field, so it says how this entry placed among this field's candidates and
        # says almost nothing about how similar the two texts are. Named for what it is:
        # calling it `semantic_score` is what let "0.9" be read as 90% similarity for two
        # releases (DX-003). The absolute cosine is reported separately.
        fused_retrieval_score = min(retrieval_score, 1.0)

        # Lexical score (token overlap).
        # Identifiers must be split on case and digit boundaries, not just underscores:
        # `.replace("_", " ")` leaves "NumTstTakr" and "enroll12" as single tokens, so
        # every camelCase schema (Avro, JSON Schema, most SQL Server DDL) scored 0
        # lexical overlap no matter how well it actually matched.
        field_tokens = _tokenize_identifier(field.name)
        entry_tokens = _tokenize_identifier(entry.logical_name)
        entry_tokens |= _tokenize_identifier(entry.business_name)

        if field_tokens and entry_tokens:
            intersection = field_tokens & entry_tokens
            lexical_score = len(intersection) / max(len(field_tokens), 1)
        else:
            lexical_score = 0.0

        # Edit distance score. Compare on the normalized token strings so that
        # "NumTstTakr" and "num_tst_takr" are not penalized for punctuation alone.
        edit_distance_score = self._edit_distance_score(
            " ".join(sorted(field_tokens)),
            " ".join(sorted(_tokenize_identifier(entry.logical_name))),
        )

        # Type compatibility
        type_score = entry.matches_type(field.data_type)

        # Domain score using domain matcher
        domain_score = self._calculate_domain_score(field, entry)

        return (
            fused_retrieval_score,
            lexical_score,
            edit_distance_score,
            type_score,
            domain_score,
        )

    def _calculate_scores(
        self,
        field: SchemaField,
        entry: DictionaryEntry,
        query_embedding: np.ndarray,
        retrieval_score: float,
    ) -> ScoreBreakdown:
        """Calculate detailed score breakdown."""
        return _breakdown(self._score_signals(field, entry, retrieval_score))

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

        # No usable pair -- neutral score.
        #
        # This used to be two branches, "only the entry has a domain" and "neither side
        # has one", both returning 0.5. They are written as one return because the split
        # implied a partial-credit case that never existed, not because merging them is
        # faster: this is NOT a performance change. Both forms are one predicted branch
        # per candidate, and the whole collapse can save at most 0.16 ms of a 7.04 s
        # match. Do not cite it as an optimization.
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
        """
        Calculate normalized edit distance score.

        Uses rapidfuzz's C++ Levenshtein when available (~145x faster than the pure
        Python DP below, bit-identical results), falling back to the DP so the package
        still works without the optional dependency. This is on the hot path: it runs
        once per candidate per field.
        """
        if not s1 or not s2:
            return 0.0

        if _LEVENSHTEIN is not None:
            return float(_LEVENSHTEIN.normalized_similarity(s1, s2))

        return self._edit_distance_score_fallback(s1, s2)

    @staticmethod
    def _edit_distance_score_fallback(s1: str, s2: str) -> float:
        """Pure-Python Levenshtein, used only when rapidfuzz is unavailable."""
        m, n = len(s1), len(s2)

        # Single-row DP: O(min(m, n)) memory instead of a full (m+1)x(n+1) matrix.
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m

        previous = list(range(n + 1))
        for i in range(1, m + 1):
            current = [i] + [0] * n
            c1 = s1[i - 1]
            for j in range(1, n + 1):
                if c1 == s2[j - 1]:
                    current[j] = previous[j - 1]
                else:
                    current[j] = 1 + min(previous[j], current[j - 1], previous[j - 1])
            previous = current

        max_len = max(m, n)
        return 1.0 - (previous[n] / max_len) if max_len > 0 else 1.0

    def _calculate_final_confidence(self, scores: ScoreBreakdown) -> float:
        """Calculate weighted final confidence score."""
        return _weighted_confidence(
            (
                scores.fused_retrieval_score,
                scores.lexical_score,
                scores.edit_distance_score,
                scores.type_compatibility_score,
                scores.domain_score,
            ),
            _signal_weights(self._config),
        )

    def _determine_decision(
        self,
        confidence: float,
        rank: int,
        runner_up_confidence: float | None,
    ) -> MatchDecision:
        """
        Determine match decision from confidence and the margin over the next candidate.

        Only the top-ranked match can be auto-approved, and only when it is BOTH
        confident enough and clearly ahead of the runner-up. `min_confidence_gap`
        exists to catch the ambiguous case where two dictionary entries score almost
        identically -- exactly the situation a human should adjudicate. Previously the
        gap was compared against already-emitted lower-ranked results, so it could
        never prevent an ambiguous rank-1 auto-approval; a near-tie was silently
        auto-approved.
        """
        config = self._config

        if rank == 1 and confidence >= config.auto_approve_threshold:
            margin = (
                confidence - runner_up_confidence
                if runner_up_confidence is not None
                else float("inf")
            )
            if margin >= config.min_confidence_gap:
                return MatchDecision.AUTO_APPROVE
            # Confident but ambiguous: send it to a human rather than guessing.
            return MatchDecision.REVIEW

        if confidence >= config.review_threshold:
            return MatchDecision.REVIEW

        return MatchDecision.REJECT

    @property
    def minimum_achievable_confidence(self) -> float | None:
        """
        The lowest `final_confidence` a rank-1 match can carry, for THIS matcher.

        Ask this before choosing any confidence threshold. A threshold at or below it
        selects nothing, however bad the matches are -- which is how
        `get_low_confidence_fields()` came to answer "nothing to review" on a schema
        where nothing was trustworthy (DX-001).

        Returns None when a reranker is wired, because the reranker replaces the fused
        retrieval score with its own squashed output and the floor no longer holds. None
        means "unknown for this configuration", not "zero"; see
        `MatchingConfig.minimum_achievable_confidence` for the derivation and the second
        precondition.
        """
        if self._reranker is not None:
            return None
        return self._config.minimum_achievable_confidence

    @property
    def dictionary_size(self) -> int:
        """Get number of loaded dictionary entries."""
        return len(self._dictionary_entries)

    @property
    def is_ready(self) -> bool:
        """Check if matcher is ready for queries."""
        return self._is_initialized
