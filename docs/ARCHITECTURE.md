# NexusMatcher Architecture

Structural reference for the layering, ports, adapters and wiring.

> **Scope note.** This document describes structure, not measured performance. Latency
> figures embedded in the diagrams below are illustrative shapes of the pipeline, not
> results — several of them were carried over from claims that have since been
> retracted. For numbers, use [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md); for the
> interfaces that actually exist, use [API_REFERENCE.md](API_REFERENCE.md).
>
> Two specific corrections to keep in mind while reading:
>
> - **The matching endpoints are `/api/v1/match`, `/api/v1/match/batch` and
>   `/api/v1/feedback`**, not the `POST /match` and `POST /batch` earlier revisions of
>   this document described and then denied. §4.1 is the shipped flow. The `/dictionary`
>   CRUD routes still do not exist.
> - **The multi-layer cache is not wired into the matching pipeline.** L1/L2/L3 cache
>   adapters exist and are unit-tested, but `NexusMatcher` does not consult them. §6.1
>   is a design sketch.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Layer Breakdown](#2-layer-breakdown)
3. [Component Reference](#3-component-reference)
4. [Data Flow](#4-data-flow)
5. [Dependency Injection](#5-dependency-injection)
6. [Performance Architecture](#6-performance-architecture)
7. [Extension Points](#7-extension-points)
8. [Security Considerations](#8-security-considerations)
9. [Deployment Patterns](#9-deployment-patterns)

---

## 1. Architectural Overview

### 1.1 Design Philosophy

NexusMatcher follows **Clean Architecture** principles with **Hexagonal (Ports & Adapters)** implementation:

```
                              ┌─────────────────────────────────┐
                              │         EXTERNAL WORLD          │
                              │  (APIs, CLI, Files, Databases)  │
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ADAPTERS (Infrastructure)                       │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │   FastAPI   │  │   Click     │  │  Qdrant     │  │   Redis     │      │
│   │   Adapter   │  │   Adapter   │  │  Adapter    │  │   Adapter   │      │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│          │                │                │                │              │
└──────────┼────────────────┼────────────────┼────────────────┼──────────────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   PORTS                                      │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│   │  API Port   │  │  CLI Port   │  │VectorStore  │  │ Cache Port  │      │
│   │             │  │             │  │   Port      │  │             │      │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│          │                │                │                │              │
└──────────┼────────────────┼────────────────┼────────────────┼──────────────┘
           │                │                │                │
           └────────────────┴────────┬───────┴────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │                      DOMAIN CORE                       │
         │                                                        │
         │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
         │   │  Entities   │  │  Services   │  │   Rules     │  │
         │   │             │  │             │  │             │  │
         │   │ • Schema    │  │ • Matching  │  │ • Scoring   │  │
         │   │ • Field     │  │ • Context   │  │ • Threshold │  │
         │   │ • Match     │  │ • Type      │  │ • Validation│  │
         │   │ • Dict      │  │ • Hierarchy │  │             │  │
         │   └─────────────┘  └─────────────┘  └─────────────┘  │
         │                                                        │
         └───────────────────────────────────────────────────────┘
```

### 1.2 Key Principles

| Principle | Implementation |
|-----------|----------------|
| **Dependency Rule** | Dependencies point inward; domain has no external dependencies |
| **Separation of Concerns** | Each layer has distinct responsibilities |
| **Interface Segregation** | Ports are small, focused interfaces |
| **Dependency Inversion** | Domain defines ports; infrastructure implements them |
| **Single Responsibility** | Each class has one reason to change |

### 1.3 Layer Responsibilities

| Layer | Responsibility | Dependencies |
|-------|----------------|--------------|
| **Presentation** | External interface adaptation | Application |
| **Application** | Use case orchestration | Domain |
| **Domain** | Business logic and rules | None |
| **Infrastructure** | External system integration | Domain (ports) |

---

## 2. Layer Breakdown

### 2.1 Presentation Layer

**Location**: `src/nexus_matcher/presentation/`

Handles all external interfaces to the system.

#### 2.1.1 REST API (`presentation/api/`)

```python
# app.py - FastAPI application factory
from fastapi import FastAPI
from nexus_matcher.shared.container import Container

def create_app() -> FastAPI:
    """Create FastAPI application with dependency injection."""
    app = FastAPI(
        title="NexusMatcher API",
        version="1.0.0",
    )
    
    # Initialize container
    container = Container()
    app.state.container = container
    
    # Register routes
    app.include_router(health_router)
    app.include_router(match_router)
    app.include_router(dictionary_router)
    
    return app
```

> The factory sketch above is **not** the shipped `create_app()`. The real one registers a
> health router, a matching router and a feedback router, adds request-ID middleware, CORS
> and exception handlers, and does not use the DI container.

**Endpoints that exist** (enumerated from a live `create_app()`):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/match` | POST | Match schema fields; each candidate carries the protection class it would confer |
| `/api/v1/match/batch` | POST | The same service and the same projection, at a higher field cap |
| `/api/v1/feedback` | POST | Append a reviewer's verdict to an audit log |
| `/` | GET | Service identity |
| `/health` | GET | Health check |
| `/health/live` | GET | Kubernetes liveness probe |
| `/health/ready` | GET | Readiness probe (503 if not ready) |
| `/health/startup` | GET | Startup probe (503 while starting) |
| `/docs`, `/redoc`, `/openapi.json` | GET | Generated OpenAPI docs |

The matching routes live in `presentation/api/matching.py` and share one `MatchService`, so
the two cannot drift apart; admission control and the server-side deadline
(`presentation/api/limits.py`) sit between the handler and the matcher's thread pool.

The `/dictionary` CRUD routes appearing in earlier revisions of this table still do not
exist. `POST /match` and `POST /batch` do — under `/api/v1`.

#### 2.1.2 CLI (`presentation/cli/`)

> The shipped CLI uses **Typer**, not Click, and lives in `presentation/cli/main.py`.
> Its commands are `match`, `sync`, `api` and `info`. The Click sketch below is
> illustrative of the shape only; see [API_REFERENCE.md](API_REFERENCE.md#cli) for the
> real flags.

```python
# ILLUSTRATIVE ONLY - the real CLI is Typer-based
import click
from nexus_matcher.shared.container import Container

@click.group()
@click.pass_context
def cli(ctx):
    """NexusMatcher CLI."""
    ctx.ensure_object(dict)
    ctx.obj["container"] = Container()

@cli.command()
@click.argument("schema_path")
@click.option("--dictionary", "-d", required=True)
@click.option("--output", "-o", default=None)
@click.pass_context
def match(ctx, schema_path, dictionary, output):
    """Match a schema file to dictionary entries."""
    container = ctx.obj["container"]
    use_case = container.match_schema_use_case()
    
    results = use_case.execute(schema_path, dictionary)
    
    if output:
        save_results(results, output)
    else:
        print_results(results)
```

#### 2.1.3 Plugin System (`presentation/plugins/`)

Entry point-based extensibility:

```toml
# pyproject.toml
[project.entry-points."nexus_matcher.parsers"]
custom = "my_package.parser:CustomParser"

[project.entry-points."nexus_matcher.scorers"]
custom = "my_package.scorer:CustomScorer"
```

```python
# Plugin loader
from importlib.metadata import entry_points

def load_parsers() -> dict[str, type]:
    """Load all registered parser plugins."""
    eps = entry_points(group="nexus_matcher.parsers")
    return {ep.name: ep.load() for ep in eps}
```

### 2.2 Application Layer

**Location**: `src/nexus_matcher/application/`

Orchestrates use cases without business logic.

#### 2.2.1 Use Cases (`application/use_cases/`)

```python
# match_schema.py - Main matching use case
from dataclasses import dataclass
from nexus_matcher.domain.ports import (
    SchemaParser,
    EmbeddingProvider,
    VectorStore,
    Reranker,
)
from nexus_matcher.domain.services import (
    ContextEnricher,
    ConfidenceScorer,
)

@dataclass
class MatchSchemaUseCase:
    """Orchestrates schema matching workflow."""
    
    parser_registry: dict[str, SchemaParser]
    embedder: EmbeddingProvider
    vector_store: VectorStore
    reranker: Reranker
    context_enricher: ContextEnricher
    scorer: ConfidenceScorer
    cache: CacheProvider | None = None
    
    def execute(
        self,
        schema_path: str,
        options: MatchOptions | None = None,
    ) -> dict[str, list[Match]]:
        """Execute schema matching workflow."""
        options = options or MatchOptions()
        
        # 1. Parse schema
        schema = self._parse_schema(schema_path)
        
        # 2. Match each field
        results = {}
        for field in schema.fields:
            matches = self._match_field(field, options)
            results[field.path] = matches
        
        return results
    
    def _match_field(
        self,
        field: Field,
        options: MatchOptions,
    ) -> list[Match]:
        """Match single field through pipeline."""
        # Stage 1: Candidate retrieval
        query = self.context_enricher.enrich(field)
        
        # Check cache
        if self.cache:
            cached = self.cache.get(self._cache_key(query))
            if cached:
                return cached
        
        # Generate embedding
        embedding = self.embedder.encode(query)
        
        # Retrieve candidates
        candidates = self.vector_store.search(
            embedding,
            top_k=options.retrieval_top_k,
        )
        
        # Stage 2: Reranking
        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=options.rerank_top_k,
        )
        
        # Stage 3: Scoring
        matches = []
        for rank, (entry, score) in enumerate(reranked, 1):
            confidence = self.scorer.compute(field, entry, score)
            match = Match(
                dictionary_entry=entry,
                final_confidence=confidence.score,
                decision=confidence.decision,
                scores=confidence.components,
                rank=rank,
            )
            matches.append(match)
        
        # Cache results
        if self.cache:
            self.cache.set(self._cache_key(query), matches)
        
        return matches
```

#### 2.2.2 DTOs (`application/dto/`)

Data transfer objects for API boundaries:

```python
# request.py
from pydantic import BaseModel, Field

class MatchRequest(BaseModel):
    """Schema matching request."""
    schema: dict = Field(..., description="Schema to match")
    options: MatchOptions | None = Field(default=None)

class MatchOptions(BaseModel):
    """Matching configuration options."""
    top_k: int = Field(default=5, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_scores: bool = Field(default=False)

# response.py
class MatchResponse(BaseModel):
    """Schema matching response."""
    matches: dict[str, list[MatchResult]]
    metadata: MatchMetadata

class MatchResult(BaseModel):
    """Single match result."""
    dictionary_entry: DictionaryEntryDTO
    confidence: float
    decision: str
    scores: dict[str, float] | None = None
    rank: int
```

### 2.3 Domain Layer

**Location**: `src/nexus_matcher/domain/`

Contains business entities, rules, and service interfaces.

#### 2.3.1 Entities (`domain/models/`)

```python
# schema.py - Core domain entities
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class DataType(Enum):
    """Canonical data types."""
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    BINARY = "binary"
    ARRAY = "array"
    MAP = "map"
    RECORD = "record"

@dataclass(frozen=True)
class Field:
    """Schema field entity."""
    path: str
    name: str
    data_type: str
    description: str = ""
    nullable: bool = True
    default: Any = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def depth(self) -> int:
        """Nesting depth of field."""
        return self.path.count(".")
    
    @property
    def parent_path(self) -> str | None:
        """Path to parent field."""
        if "." not in self.path:
            return None
        return ".".join(self.path.split(".")[:-1])

@dataclass(frozen=True)
class Schema:
    """Parsed schema entity."""
    name: str
    namespace: str
    fields: tuple[Field, ...]
    format: str  # avro, json_schema, sql_ddl, csv
    raw: dict = field(default_factory=dict)
    
    def get_field(self, path: str) -> Field | None:
        """Get field by path."""
        for f in self.fields:
            if f.path == path:
                return f
        return None

@dataclass(frozen=True)
class DictionaryEntry:
    """Data dictionary entry entity."""
    id: str
    business_name: str
    technical_name: str
    data_type: str
    description: str
    domain: str = ""
    owner: str = ""
    sensitivity: str = "internal"
    metadata: dict = field(default_factory=dict)
    
    def to_searchable_text(self) -> str:
        """Generate text for embedding."""
        parts = [self.business_name, self.technical_name]
        if self.description:
            parts.append(self.description)
        return " ".join(parts)

@dataclass
class Match:
    """Matching result entity."""
    dictionary_entry: DictionaryEntry
    final_confidence: float
    decision: str  # AUTO_APPROVE, REVIEW, REJECT
    scores: dict[str, float]
    rank: int
```

#### 2.3.2 Ports (`domain/ports/`)

Abstract interfaces for external dependencies:

```python
# embedding.py - Embedding provider port
from abc import ABC, abstractmethod
import numpy as np

class EmbeddingProvider(ABC):
    """Abstract embedding provider."""
    
    @abstractmethod
    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass

# vector_store.py - Vector store port
class VectorStore(ABC):
    """Abstract vector store."""
    
    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        payloads: list[dict],
    ) -> None:
        """Add vectors to store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors."""
        pass

# cache.py - Cache provider port
class CacheProvider(ABC):
    """Abstract cache provider."""
    
    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set value in cache."""
        pass

# parser.py - Schema parser port
class SchemaParser(ABC):
    """Abstract schema parser."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Parser format name."""
        pass
    
    @property
    @abstractmethod
    def file_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        pass
    
    @abstractmethod
    def parse(self, source: str | Path | dict) -> Schema:
        """Parse schema from source."""
        pass
```

#### 2.3.3 Services (`domain/services/`)

Domain services encapsulating business logic:

```python
# context_enricher.py
@dataclass
class ContextEnricher:
    """Enriches field context for better matching."""
    
    config: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    
    def enrich(self, field: Field) -> str:
        """Generate enriched query text."""
        parts = []
        
        # Hierarchy context
        if self.config.include_hierarchy:
            hierarchy = self._extract_hierarchy(field.path)
            if hierarchy:
                parts.append(", ".join(hierarchy))
        
        # Field name (humanized)
        parts.append(self._humanize(field.name))
        
        # Type context
        if self.config.include_type:
            type_text = self._type_to_text(field.data_type)
            if type_text:
                parts.append(type_text)
        
        # Description
        if field.description and self.config.include_description:
            parts.append(field.description)
        
        return " ".join(parts)

# confidence_scorer.py
@dataclass
class ConfidenceScorer:
    """Computes final confidence scores."""
    
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: ScoringThresholds = field(default_factory=ScoringThresholds)
    
    def compute(
        self,
        field: Field,
        entry: DictionaryEntry,
        retrieval_score: float,
    ) -> Confidence:
        """Compute confidence with component breakdown."""
        components = {
            "semantic": retrieval_score,
            "lexical": self._lexical_overlap(field, entry),
            "type": self._type_compatibility(field, entry),
            "pattern": self._pattern_match(field, entry),
        }
        
        # Weighted combination
        final = sum(
            components[k] * getattr(self.weights, k)
            for k in components
        )
        
        # Decision
        if final >= self.thresholds.auto_approve:
            decision = "AUTO_APPROVE"
        elif final >= self.thresholds.review:
            decision = "REVIEW"
        else:
            decision = "REJECT"
        
        return Confidence(
            score=final,
            decision=decision,
            components=components,
        )
```

### 2.4 Infrastructure Layer

**Location**: `src/nexus_matcher/infrastructure/`

Implements ports with concrete external systems.

#### 2.4.1 Embeddings (`infrastructure/adapters/embedding_providers/`)

```python
# sentence_transformers.py
class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Sentence-Transformers embedding provider."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",   # actual shipped default
        device: str = "cpu",
    ):
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension

# quantized.py
class QuantizedEmbeddingProvider(EmbeddingProvider):
    """ONNX Runtime INT8 quantized embedding provider."""
    
    def __init__(
        self,
        model_name: str,
        use_int8: bool = True,
    ):
        # Export and quantize model
        self._session = self._create_session(model_name, use_int8)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            
            outputs = self._session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )
            
            # Mean pooling
            embeddings = self._mean_pool(outputs[0], inputs["attention_mask"])
            results.append(embeddings)
        
        return np.vstack(results)
```

#### 2.4.2 Caches (`infrastructure/adapters/caches/`)

```python
# memory.py
class L1LRUCache(CacheProvider):
    """Thread-safe in-memory LRU cache."""
    
    def __init__(
        self,
        max_size: int = 5000,
        default_ttl: int = 3600,
    ):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = RLock()
        self._stats = CacheStats()
    
    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                return None
            
            # Move to end (LRU update)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        with self._lock:
            # Update existing
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                # Evict if at capacity
                while len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1
            
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + (ttl or self._default_ttl),
            )

# content.py
class SemanticContentCache(CacheProvider):
    """Content-addressed cache using BLAKE3 hashing."""
    
    def __init__(
        self,
        max_size: int = 10000,
        normalize: bool = True,
    ):
        self._hasher = ContentHasher(normalize=normalize)
        self._backing_cache = L1LRUCache(max_size=max_size)
    
    def get_or_compute(
        self,
        content: str,
        compute_fn: Callable[[], T],
    ) -> T:
        """Get cached value or compute and cache."""
        key = self._hasher.hash(content)
        
        cached = self._backing_cache.get(key)
        if cached is not None:
            return cached
        
        result = compute_fn()
        self._backing_cache.set(key, result)
        
        return result
```

#### 2.4.3 Rerankers (`infrastructure/adapters/rerankers/`)

```python
# colbert.py
class ColBERTMaxSimReranker:
    """ColBERT-style MaxSim reranking with pre-computed embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._model = SentenceTransformer(model_name)
        self._doc_embeddings: dict[str, np.ndarray] = {}
    
    def precompute_embeddings(
        self,
        entries: list[DictionaryEntry],
    ) -> None:
        """Pre-compute and cache document token embeddings."""
        for entry in entries:
            text = entry.to_searchable_text()
            
            # Get token-level embeddings
            tokens = self._model.encode(
                text,
                output_value="token_embeddings",
                convert_to_numpy=True,
            )
            
            self._doc_embeddings[entry.id] = tokens
    
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using MaxSim scoring."""
        # Get query token embeddings
        query_tokens = self._model.encode(
            query,
            output_value="token_embeddings",
            convert_to_numpy=True,
        )
        
        # Score each candidate
        scores = []
        for doc_id, _ in candidates:
            if doc_id not in self._doc_embeddings:
                continue
            
            doc_tokens = self._doc_embeddings[doc_id]
            maxsim = self._compute_maxsim(query_tokens, doc_tokens)
            scores.append((doc_id, maxsim))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def _compute_maxsim(
        self,
        query_tokens: np.ndarray,
        doc_tokens: np.ndarray,
    ) -> float:
        """Compute MaxSim score."""
        # Similarity matrix: (Q, D)
        sim_matrix = query_tokens @ doc_tokens.T
        
        # Max over document for each query token
        max_sims = sim_matrix.max(axis=1)
        
        return float(max_sims.mean())
```

---

## 3. Component Reference

### 3.1 Complete Component List

| Component | Layer | Type | Purpose |
|-----------|-------|------|---------|
| `FastAPI App` | Presentation | Adapter | REST API server |
| `CLI Commands` | Presentation | Adapter | Command-line interface |
| `Plugin System` | Presentation | Adapter | Entry point extension |
| `MatchSchemaUseCase` | Application | Use Case | Schema matching orchestration |
| `BatchMatchUseCase` | Application | Use Case | Parallel batch processing |
| `SyncDictionaryUseCase` | Application | Use Case | Dictionary indexing |
| `Schema` | Domain | Entity | Parsed schema |
| `Field` | Domain | Entity | Schema field |
| `DictionaryEntry` | Domain | Entity | Dictionary entry |
| `Match` | Domain | Entity | Match result |
| `EmbeddingProvider` | Domain | Port | Embedding generation |
| `VectorStore` | Domain | Port | Vector storage/search |
| `CacheProvider` | Domain | Port | Caching |
| `SchemaParser` | Domain | Port | Schema parsing |
| `ContextEnricher` | Domain | Service | Query enrichment |
| `ConfidenceScorer` | Domain | Service | Score computation |
| `TypeCompatibility` | Domain | Service | Type checking |
| `SentenceTransformerProvider` | Infrastructure | Adapter | ST embeddings |
| `QuantizedEmbeddingProvider` | Infrastructure | Adapter | ONNX INT8 embeddings |
| `QdrantVectorStore` | Infrastructure | Adapter | Qdrant integration |
| `InMemoryVectorStore` | Infrastructure | Adapter | In-memory vectors |
| `L1LRUCache` | Infrastructure | Adapter | Memory cache |
| `RedisCache` | Infrastructure | Adapter | Distributed cache |
| `SemanticContentCache` | Infrastructure | Adapter | Content-addressed cache |
| `ColBERTMaxSimReranker` | Infrastructure | Adapter | MaxSim reranking |
| `AvroSchemaParser` | Infrastructure | Adapter | Avro parsing |
| `JsonSchemaParser` | Infrastructure | Adapter | JSON Schema parsing |
| `SqlDdlParser` | Infrastructure | Adapter | SQL DDL parsing |

### 3.2 Component Dependencies

```
                    ┌─────────────────────────────────────┐
                    │           Container                  │
                    │                                     │
                    │  Registers all components           │
                    │  Manages lifecycle                  │
                    │  Provides dependency injection      │
                    └───────────────┬─────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
   ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
   │  Use Cases    │       │   Services    │       │   Adapters    │
   │               │       │               │       │               │
   │ MatchSchema   │──────▶│ ContextEnrich │       │ Embeddings    │
   │ BatchMatch    │       │ ConfidenceScr │◀──────│ VectorStores  │
   │ SyncDict      │       │ TypeCompat    │       │ Caches        │
   └───────────────┘       └───────────────┘       │ Parsers       │
                                                   │ Rerankers     │
                                                   └───────────────┘
```

---

## 4. Data Flow

### 4.1 Match Request Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST FLOW                                    │
│                                                                              │
│   Client                API Layer              Application Layer             │
│     │                      │                         │                       │
│     │  POST /api/v1/match  │                         │                       │
│     │─────────────────────▶│                         │                       │
│     │                      │                         │                       │
│     │                      │  Validate Request       │                       │
│     │                      │──────────────────────┐  │                       │
│     │                      │◀─────────────────────┘  │                       │
│     │                      │                         │                       │
│     │                      │  MatchService.match     │                       │
│     │                      │────────────────────────▶│                       │
│     │                      │                         │                       │
│     │                      │                         │  Match every field    │
│     │                      │                         │───────────────┐       │
│     │                      │                         │◀──────────────┘       │
│     │                      │                         │                       │
│     │                      │                         │  For each field:      │
│     │                      │                         │  ┌─────────────────┐  │
│     │                      │                         │  │ 1. Enrich query │  │
│     │                      │                         │  │ 2. Embed query  │  │
│     │                      │                         │  │ 3. Retrieve     │  │
│     │                      │                         │  │ 4. Fuse + score │  │
│     │                      │                         │  │ 5. Rerank (opt) │  │
│     │                      │                         │  │ 6. Governance   │  │
│     │                      │                         │  │ 7. Decide       │  │
│     │                      │                         │  └─────────────────┘  │
│     │                      │                         │                       │
│     │                      │  MatchResult tuples     │                       │
│     │                      │◀────────────────────────│                       │
│     │                      │                         │                       │
│     │  JSON Response       │                         │                       │
│     │◀─────────────────────│                         │                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

Three things the boxes flatten, all of them load-bearing:

- **Nothing is parsed here.** The caller sends fields, not a schema file; `FieldSpec` is
  the wire type and `_to_schema_field` builds the domain object. Schema parsing is the
  CLI's and the library's path, not the endpoint's.
- **The matcher does not run on the event loop.** `run_bounded` admits the call into a
  bounded thread pool or sheds it with a **503**, and answers **504** if the server-side
  deadline fires first. Matching is CPU-bound, so without this a single large request
  would block `/health/live` in the same process.
- **The response is checked before it is sent.** `_project_results` asserts one result key
  per input field, each computed for the field it is keyed by, and refuses the response
  rather than returning it short — a field missing from the map would inherit no
  governance and nothing would say so.

No cache is consulted anywhere in this path; see the note at the top of this document.

### 4.2 Indexing Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING FLOW                                    │
│                                                                              │
│   Dictionary                 Application               Infrastructure        │
│   Source                     Layer                     Layer                  │
│     │                         │                         │                    │
│     │  Load Dictionary        │                         │                    │
│     │────────────────────────▶│                         │                    │
│     │                         │                         │                    │
│     │                         │  SyncDictionaryUseCase  │                    │
│     │                         │────────────────────────▶│                    │
│     │                         │                         │                    │
│     │                         │                         │  Parse entries     │
│     │                         │                         │──────────┐         │
│     │                         │                         │◀─────────┘         │
│     │                         │                         │                    │
│     │                         │                         │  Detect changes    │
│     │                         │                         │  (BLAKE3)          │
│     │                         │                         │──────────┐         │
│     │                         │                         │◀─────────┘         │
│     │                         │                         │                    │
│     │                         │                         │  Embed changed     │
│     │                         │                         │  entries only      │
│     │                         │                         │──────────┐         │
│     │                         │                         │◀─────────┘         │
│     │                         │                         │                    │
│     │                         │                         │  Index to          │
│     │                         │                         │  VectorStore       │
│     │                         │                         │──────────┐         │
│     │                         │                         │◀─────────┘         │
│     │                         │                         │                    │
│     │                         │                         │  Pre-compute       │
│     │                         │                         │  MaxSim embeddings │
│     │                         │                         │──────────┐         │
│     │                         │                         │◀─────────┘         │
│     │                         │                         │                    │
│     │  Sync Complete          │                         │                    │
│     │◀────────────────────────│                         │                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Dependency Injection

### 5.1 Container Implementation

```python
# shared/container.py
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """Dependency injection container."""
    
    # Configuration
    config = providers.Configuration()
    
    # Infrastructure - Embeddings
    embedding_provider = providers.Singleton(
        SentenceTransformersProvider,
        model_name=config.embedding.model_name,
        device=config.embedding.device,
    )
    
    quantized_embedding_provider = providers.Singleton(
        QuantizedEmbeddingProvider,
        model_name=config.embedding.model_name,
        use_int8=config.embedding.use_int8,
    )
    
    # Infrastructure - Vector Store
    vector_store = providers.Singleton(
        QdrantVectorStore,
        host=config.vector_store.host,
        port=config.vector_store.port,
        collection_name=config.vector_store.collection_name,
    )
    
    # Infrastructure - Caches
    l1_cache = providers.Singleton(
        L1LRUCache,
        max_size=config.cache.l1_max_size,
        default_ttl=config.cache.l1_ttl,
    )
    
    semantic_cache = providers.Singleton(
        SemanticContentCache,
        max_size=config.cache.l3_max_size,
    )
    
    # Infrastructure - Rerankers
    maxsim_reranker = providers.Singleton(
        ColBERTMaxSimReranker,
        model_name=config.embedding.model_name,
    )
    
    # Domain Services
    context_enricher = providers.Factory(
        ContextEnricher,
        config=providers.Factory(EnrichmentConfig),
    )
    
    confidence_scorer = providers.Factory(
        ConfidenceScorer,
        weights=providers.Factory(ScoringWeights),
        thresholds=providers.Factory(ScoringThresholds),
    )
    
    # Application - Use Cases
    match_schema_use_case = providers.Factory(
        MatchSchemaUseCase,
        parser_registry=providers.Dict({
            "avro": providers.Factory(AvroSchemaParser),
            "json_schema": providers.Factory(JsonSchemaParser),
            "sql_ddl": providers.Factory(SqlDdlParser),
        }),
        embedder=embedding_provider,
        vector_store=vector_store,
        reranker=maxsim_reranker,
        context_enricher=context_enricher,
        scorer=confidence_scorer,
        cache=l1_cache,
    )
```

### 5.2 Usage

```python
# In API handlers
from nexus_matcher.shared.container import Container

container = Container()
container.config.from_yaml("config.yaml")

use_case = container.match_schema_use_case()
results = use_case.execute("schema.avsc")
```

---

## 6. Performance Architecture

### 6.1 Caching Strategy — DESIGN ONLY, NOT WIRED

The three cache adapters exist and are unit-tested, but `NexusMatcher` never consults
them: there is no cache lookup anywhere in the matching path, and `PerformanceMetrics.cache_hit`
is hardcoded to `False`. The tiering below is the intended design, not current behaviour.
No latency figures are given because none have been measured through this path.

```
                    Request
                       │
                       ▼
                ┌─────────────┐
                │  L1 Cache   │ ◀─── In-process LRU
                │   (LRU)     │
                └──────┬──────┘
                       │ Miss
                       ▼
                ┌─────────────┐
                │  L2 Cache   │ ◀─── Redis, shared across processes
                │   (Redis)   │
                └──────┬──────┘
                       │ Miss
                       ▼
                ┌─────────────┐
                │  L3 Cache   │ ◀─── Content-addressed (semantic)
                │  (Content)  │
                └──────┬──────┘
                       │ Miss
                       ▼
                ┌─────────────┐
                │   Compute   │ ◀─── Full pipeline
                │             │
                └─────────────┘
```

### 6.2 Pre-computation Strategy

```
                 ┌─────────────────────────────────────────────────┐
                 │                INDEXING TIME                     │
                 │                                                  │
                 │   Dictionary Entries                             │
                 │         │                                        │
                 │         ▼                                        │
                 │   ┌─────────────┐                               │
                 │   │   Embed     │ ◀─── O(n) embedding calls     │
                 │   │   Entries   │      (expensive but offline)   │
                 │   └──────┬──────┘                               │
                 │          │                                       │
                 │          ▼                                       │
                 │   ┌─────────────┐                               │
                 │   │  Compute    │ ◀─── O(n) token embeddings    │
                 │   │  MaxSim     │      (expensive but offline)   │
                 │   │  Tokens     │                                │
                 │   └──────┬──────┘                               │
                 │          │                                       │
                 │          ▼                                       │
                 │   Store in Vector DB + Token Cache               │
                 │                                                  │
                 └─────────────────────────────────────────────────┘
                 
                 ┌─────────────────────────────────────────────────┐
                 │                 QUERY TIME                       │
                 │                                                  │
                 │   Query                                          │
                 │     │                                            │
                 │     ▼                                            │
                 │   ┌─────────────┐                               │
                 │   │   Embed     │ ◀─── O(1) embedding call      │
                 │   │   Query     │      (batched across fields)   │
                 │   └──────┬──────┘                               │
                 │          │                                       │
                 │          ▼                                       │
                 │   ┌─────────────┐                               │
                 │   │   Vector    │ ◀─── O(log n) HNSW search     │
                 │   │   Search    │      (~5ms)                    │
                 │   └──────┬──────┘                               │
                 │          │                                       │
                 │          ▼                                       │
                 │   ┌─────────────┐                               │
                 │   │   MaxSim    │ ◀─── O(k) lookups             │
                 │   │   Rerank    │      (optional, off by default)│
                 │   └──────┬──────┘                               │
                 │          │                                       │
                 │          ▼                                       │
                 │   Results                                        │
                 │                                                  │
                 └─────────────────────────────────────────────────┘
```

---

## 7. Extension Points

### 7.1 Custom Schema Parser

```python
from nexus_matcher.domain.ports import SchemaParser, Schema

class CustomFormatParser(SchemaParser):
    """Parser for custom schema format."""
    
    @property
    def format_name(self) -> str:
        return "custom_format"
    
    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".custom", ".cust"})
    
    def parse(self, source: str | Path | dict) -> Schema:
        # Load content
        if isinstance(source, Path):
            content = source.read_text()
        elif isinstance(source, str):
            content = Path(source).read_text()
        else:
            content = source
        
        # Parse into Schema
        fields = self._parse_fields(content)
        return Schema(
            name="custom_schema",
            namespace="",
            fields=tuple(fields),
            format=self.format_name,
        )

# Register via entry point
# pyproject.toml:
# [project.entry-points."nexus_matcher.parsers"]
# custom_format = "my_package:CustomFormatParser"
```

### 7.2 Custom Scoring Function

```python
from nexus_matcher.domain.services import ConfidenceScorer

class CustomScorer(ConfidenceScorer):
    """Custom scoring with domain-specific rules."""
    
    def compute(
        self,
        field: Field,
        entry: DictionaryEntry,
        retrieval_score: float,
    ) -> Confidence:
        # Get base scores
        base = super().compute(field, entry, retrieval_score)
        
        # Add custom rules
        if self._is_pii_field(field) and self._is_pii_entry(entry):
            # Boost PII matches
            base.score = min(1.0, base.score * 1.1)
        
        return base
```

### 7.3 Custom Cache Backend

```python
from nexus_matcher.domain.ports import CacheProvider

class CustomCache(CacheProvider):
    """Custom cache implementation (e.g., Memcached)."""
    
    def __init__(self, servers: list[str]):
        import pylibmc
        self._client = pylibmc.Client(servers)
    
    def get(self, key: str) -> Any | None:
        try:
            return self._client.get(key)
        except pylibmc.NotFound:
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        self._client.set(key, value, time=ttl or 0)
```

---

## 8. Security Considerations

### 8.1 Input Validation

```python
# All inputs validated at API boundary
class MatchRequest(BaseModel):
    schema: dict = Field(
        ...,
        max_length=1_000_000,  # 1MB max
    )
    
    @validator("schema")
    def validate_schema(cls, v):
        # Check structure
        if "fields" not in v and "properties" not in v:
            raise ValueError("Invalid schema structure")
        return v
```

### 8.2 Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/match")
@limiter.limit("100/minute")
async def match(request: MatchRequest):
    ...
```

### 8.3 Data Sensitivity

```python
# Dictionary entries can have sensitivity levels
class DictionaryEntry:
    sensitivity: str = "internal"  # public, internal, confidential, restricted
    
    def to_searchable_text(self) -> str:
        # Exclude sensitive metadata from embeddings
        parts = [self.business_name]
        if self.sensitivity in ("public", "internal"):
            parts.append(self.description)
        return " ".join(parts)
```

---

## 9. Deployment Patterns

### 9.1 Single Instance

```yaml
# docker-compose.yml
version: '3.8'
services:
  nexus-matcher:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEXUS_ENV=production
      - NEXUS_VECTOR_BACKEND=memory
    volumes:
      - ./data:/app/data
```

### 9.2 Production Cluster

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  nexus-matcher:
    build: .
    deploy:
      replicas: 3
    ports:
      - "8000:8000"
    environment:
      - NEXUS_ENV=production
      - NEXUS_VECTOR_BACKEND=qdrant
      - NEXUS_VECTOR_QDRANT_HOST=qdrant
      - NEXUS_CACHE_L2_ENABLED=true
      - NEXUS_CACHE_L2_REDIS_URL=redis://redis:6379/0
    depends_on:
      - qdrant
      - redis
  
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  qdrant_data:
  redis_data:
```

### 9.3 Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-matcher
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nexus-matcher
  template:
    metadata:
      labels:
        app: nexus-matcher
    spec:
      containers:
      - name: nexus-matcher
        image: nexus-matcher:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: NEXUS_ENV
          value: "production"
        - name: NEXUS_VECTOR_QDRANT_HOST
          value: "qdrant-service"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

*Architecture Document Version 1.0.0*
*Last Updated: December 2025*
