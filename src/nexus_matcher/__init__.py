"""
NexusMatcher - Enterprise Semantic Schema Matching System
=========================================================

NexusMatcher is a production-grade semantic schema matching system that
automatically maps schema fields to entries in a data dictionary using
neural embeddings, ColBERT reranking, and multi-signal scoring.

Key Features
------------
- **100% Precision@1** on benchmark datasets
- **86x faster** reranking with pre-computed ColBERT MaxSim
- **INT8 quantization** for 1.68x embedding speedup
- **Multi-layer caching** with 56.99% hit rate
- **Hexagonal architecture** for extensibility

Installation
------------
```bash
# Minimal
pip install nexus-matcher

# Full (all features)
pip install nexus-matcher[full]

# Specific features
pip install nexus-matcher[embeddings,api,cache]
```

Deployment Modes
----------------
NexusMatcher can be used in three modes:

1. **Library Mode**: Import and use directly in Python code
2. **Backend Mode**: Run as a REST API server
3. **Plugin Mode**: Extend with custom parsers, loaders, and providers

Quick Start (Library Mode)
--------------------------
```python
from nexus_matcher import NexusMatcher

# Initialize
matcher = NexusMatcher()

# Load dictionary
matcher.load_dictionary("data/dictionary.xlsx")

# Match a schema
results = matcher.match_schema("schemas/customer.avsc")

for field_path, matches in results.items():
    top_match = matches[0]
    print(f"{field_path} -> {top_match.dictionary_entry.business_name}")
    print(f"  Confidence: {top_match.final_confidence:.2%}")
    print(f"  Decision: {top_match.decision}")
```

Backend Mode
------------
```bash
# Start REST API server
nexus-matcher api --host 0.0.0.0 --port 8000
```

Links
-----
- Documentation: https://nexus-matcher.readthedocs.io
- Repository: https://github.com/pierce-lonergan/nexus_matcher
- PyPI: https://pypi.org/project/nexus-matcher/
"""

from __future__ import annotations

# This is the single source of truth for the package version:
# [tool.hatch.version] in pyproject.toml reads it from here. It said 1.0.0 while
# CHANGELOG's latest release, the built wheel, the API and the log envelope all
# said 2.0.0, so `python -m build` would have produced a 1.0.0 artifact and the
# publish workflow would have tried to release it over the existing 2.0.0.
__version__ = "2.0.0"
__author__ = "Pierce Lonergan"
__email__ = "lonerganpierce@gmail.com"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2025 Pierce Lonergan"

# Package metadata for programmatic access
__pkg_info__ = {
    "name": "nexus-matcher",
    "version": __version__,
    "author": __author__,
    "license": __license__,
}

# =============================================================================
# PUBLIC API
# =============================================================================

# Core types (always available)
# Domain models (always available)
from nexus_matcher.domain.models import (
    DictionaryEntry,
    MatchResult,
    Schema,
    SchemaField,
)

# Dependency injection
from nexus_matcher.shared import (
    Container,
    ContainerBuilder,
    Lifecycle,
)
from nexus_matcher.shared.types import (
    DataType,
    MatchDecision,
    ProtectionLevel,
    Score,
)

__all__ = [
    # --- embedding providers ---------------------------------------------
    "BundledOnnxProvider",
    "Cache",
    "Config",
    # DI Container
    "Container",
    "ContainerBuilder",
    # Types
    "DataType",
    "DictionaryEntry",
    "DictionaryLoader",
    "EmbeddingProvider",
    # --- schema parsing --------------------------------------------------
    "FlattenedAvroParser",
    "GlossaryIndex",
    "HnswVectorStore",
    # --- vector stores ---------------------------------------------------
    "InMemoryVectorStore",
    "Lifecycle",
    "MatchDecision",
    "MatchResult",
    # Lazy imports (documented for IDE completion)
    "NexusMatcher",
    "ProtectionLevel",
    "Reranker",
    "Schema",
    # Models
    "SchemaField",
    "SchemaParser",
    "Score",
    "SparseRetriever",
    "SyncReport",
    "VectorStore",
    "__author__",
    "__copyright__",
    "__email__",
    "__license__",
    "__pkg_info__",
    # Version info
    "__version__",
    "build_index",
    "create_app",
    "default_embedding_provider",
    "flatten_avro_schema",
    # --- ingestion -------------------------------------------------------
    "ingest",
    "load_entries",
    "sync",
]


# =============================================================================
# LAZY IMPORTS FOR OPTIONAL FEATURES
# =============================================================================


def __getattr__(name: str):
    """Lazy import for optional components."""

    # Config (requires pydantic-settings)
    if name == "Config":
        from nexus_matcher.infrastructure.config.settings import Config

        return Config

    # Main matcher class (requires full installation)
    if name == "NexusMatcher":
        from nexus_matcher.application.use_cases.match_schema import NexusMatcher

        return NexusMatcher

    # API app factory (requires fastapi)
    if name == "create_app":
        from nexus_matcher.presentation.api.app import create_app

        return create_app

    # Port interfaces
    if name in (
        "SchemaParser",
        "DictionaryLoader",
        "EmbeddingProvider",
        "VectorStore",
        "SparseRetriever",
        "Reranker",
        "Cache",
    ):
        from nexus_matcher.domain import ports

        return getattr(ports, name)

    # Ingestion: read a glossary from anywhere, and re-embed only what changed.
    if name in ("ingest", "build_index", "sync", "load_entries", "GlossaryIndex", "SyncReport"):
        from nexus_matcher.application import ingest as _ingest

        return _ingest if name == "ingest" else getattr(_ingest, name)

    # Flattened Avro: the production input shape.
    if name in ("FlattenedAvroParser", "flatten_avro_schema"):
        from nexus_matcher.infrastructure.adapters.schema_parsers import flattened_avro

        return getattr(flattened_avro, name)

    # Embedding providers. Loaded lazily so that importing nexus_matcher does NOT pull in
    # onnxruntime: import cost is ~80ms today and a user who only wants the domain models
    # should not pay for an inference runtime.
    if name in ("BundledOnnxProvider", "default_embedding_provider"):
        from nexus_matcher.infrastructure.adapters.embedding_providers import bundled_onnx

        return getattr(bundled_onnx, name)

    if name == "HnswVectorStore":
        from nexus_matcher.infrastructure.adapters.vector_stores.hnsw import HnswVectorStore

        return HnswVectorStore

    if name == "InMemoryVectorStore":
        from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore

        return InMemoryVectorStore

    raise AttributeError(f"module 'nexus_matcher' has no attribute '{name}'")
