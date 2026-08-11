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

# Initialize -- from_config() wires every component and needs no arguments.
# NexusMatcher(...) itself takes an embedding provider and a vector store, for
# callers who want to supply their own.
matcher = NexusMatcher.from_config()

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

Or build the ASGI app yourself, to mount it or to run it under your own server:

```python
from nexus_matcher import create_app  # needs pip install nexus-matcher[api]

app = create_app()
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
__version__ = "2.1.0"
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
# Governance (always available -- pure domain, no optional dependency). Exported eagerly
# because a caller has to build a GovernanceVocabulary BEFORE they can load a glossary
# that carries protection codes; behind a lazy import it would be the one name in the
# quickstart that does not autocomplete.
from nexus_matcher.domain.governance import (
    GovernanceVocabulary,
    ProtectionClass,
)

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
    # --- governance ------------------------------------------------------
    "GovernanceVocabulary",
    "HnswVectorStore",
    # --- vector stores ---------------------------------------------------
    "InMemoryVectorStore",
    "Lifecycle",
    "MatchDecision",
    "MatchResult",
    "MatchingConfig",
    # Lazy imports (documented for IDE completion)
    "NexusMatcher",
    "ProtectionClass",
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
    "default_embedding_provider",
    "flatten_avro_schema",
    # --- ingestion -------------------------------------------------------
    "ingest",
    "load_entries",
    "sync",
]

# Reachable through __getattr__, deliberately NOT in __all__: their dependency ships only
# in an extra, and __all__ is exactly the list `from nexus_matcher import *` walks. With
# create_app listed, that star-import died on a default install with
# `ModuleNotFoundError: No module named 'fastapi'` -- from a line the user never wrote,
# naming a package they never asked for, so it read as "nexus_matcher is broken" rather
# than "you skipped an extra". __all__ is a promise that a name imports; only names that
# hold on a bare `pip install nexus-matcher` belong in it. These stay listed in __dir__
# below, so tab-completion and dir() still surface them.
_OPTIONAL_EXPORTS = ("create_app",)

# Top-level module -> the extra that ships it. A missing optional dependency surfaces as
# a ModuleNotFoundError raised deep inside an adapter, naming a third-party package with
# no obvious link to this project; this turns it into one naming the extra to install.
# Kept for every extra, not just today's optional exports, so a lazy branch that grows a
# new dependency gets the good message for free.
#
# This covered 7 of the 15 runtime extras until 2026-08-09. The eight unmapped ones were
# the ones whose packages a user is LEAST likely to recognise -- a missing model2vec,
# ragatouille, cpuinfo or opentelemetry named nothing, suggested nothing, and left the
# reader to work out which of fifteen extras carries it.
#
# Both directions are now gated by tests/packaging/test_extras_graph.py: every optional
# distribution an extra installs has an entry here, and every entry points at an extra that
# genuinely installs it. Neither this dict nor pyproject.toml can drift alone.
#
# Two rules govern what does NOT belong here:
#
# 1. Core dependencies are excluded, even when an extra also lists them. typer, rich,
#    click, rapidfuzz and openpyxl all ship on a bare install; answering their absence with
#    "install nexus-matcher[cli]" would send someone with a broken core install chasing an
#    extra that was never the problem.
# 2. A module shipped by more than one extra maps to the one that is ABOUT it. torch and
#    transformers are in `embeddings` and `quantization`; they map to `embeddings`, whose
#    install fixes either caller. The mapping only has to name a command that works.
_EXTRA_FOR_MODULE = {
    "fastapi": "api",
    "uvicorn": "api",
    "sentence_transformers": "embeddings",
    "torch": "embeddings",  # also in `quantization`; see rule 2 above
    "transformers": "embeddings",  # also in `quantization`
    "qdrant_client": "vector-stores",
    "usearch": "vector-stores",
    "redis": "cache",
    "pyarrow": "loaders",
    "sqlalchemy": "loaders",
    "rank_bm25": "sparse",
    "blake3": "accel",
    "cpuinfo": "accel",  # the py-cpuinfo distribution imports as `cpuinfo`
    "networkx": "graph",
    "prometheus_client": "observability",
    "opentelemetry": "observability",  # both opentelemetry-api and -sdk import as this
    "onnx": "quantization",
    "model2vec": "static-embeddings",
    "ragatouille": "colbert",
}


# =============================================================================
# LAZY IMPORTS FOR OPTIONAL FEATURES
# =============================================================================


def _lazy_import(name: str):
    """Resolve one lazily-exported name. Wrapped by __getattr__, never called directly."""

    # Config (requires pydantic-settings)
    if name == "Config":
        from nexus_matcher.infrastructure.config.settings import Config

        return Config

    # Main matcher class (requires full installation)
    if name == "NexusMatcher":
        from nexus_matcher.application.use_cases.match_schema import NexusMatcher

        return NexusMatcher

    # The matcher's configuration object. Exported alongside NexusMatcher because it is
    # the only way to change a threshold or a fusion weight; leaving it out meant the
    # documented `NexusMatcher(config=...)` argument had no importable type.
    if name == "MatchingConfig":
        from nexus_matcher.application.use_cases.match_schema import MatchingConfig

        return MatchingConfig

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


def __getattr__(name: str):
    """Lazy import for optional components, with the missing extra named in the error."""
    try:
        return _lazy_import(name)
    except ModuleNotFoundError as exc:
        extra = _EXTRA_FOR_MODULE.get((exc.name or "").partition(".")[0])
        if extra is None:
            raise
        raise ModuleNotFoundError(
            f"nexus_matcher.{name} needs the optional '{extra}' extra, which is not "
            f"installed (no module named {exc.name!r}). Install it with: "
            f"pip install nexus-matcher[{extra}]",
            name=exc.name,
        ) from exc


# The module each optional export needs, so __dir__ can check availability without
# importing anything.
_OPTIONAL_EXPORT_REQUIRES = {"create_app": "fastapi"}


def __dir__() -> list[str]:
    """
    Advertise the optional exports -- but only the ones that will actually resolve.

    dir() is not merely a display list. inspect.getmembers(), help(), pydoc and
    rlcompleter tab-completion all walk it and getattr() every entry, so a name listed
    here that raises on access makes ALL of them blow up. Listing create_app
    unconditionally did exactly that on a bare install: `help(nexus_matcher)` died with
    ModuleNotFoundError: No module named 'fastapi'. Those four worked before the name was
    added, so advertising it cost more than it bought.

    find_spec only consults the import system's finders; it does not execute the module,
    so this stays cheap and free of side effects.
    """
    from importlib.util import find_spec

    available = []
    for name in _OPTIONAL_EXPORTS:
        required = _OPTIONAL_EXPORT_REQUIRES.get(name)
        try:
            if required is None or find_spec(required) is not None:
                available.append(name)
        except (ImportError, ValueError):  # a broken or namespace-shadowed install
            continue
    return sorted({*globals(), *__all__, *available})
