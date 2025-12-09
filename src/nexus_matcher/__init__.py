"""
NexusMatcher - Enterprise Semantic Schema Matching System
=========================================================

NexusMatcher is a production-grade semantic schema matching system that
automatically maps schema fields to entries in a data dictionary.

Deployment Modes
----------------
NexusMatcher can be used in three modes:

1. **Library Mode**: Import and use directly in Python code
2. **Backend Mode**: Run as a REST API server
3. **Plugin Mode**: Extend with custom parsers, loaders, and providers

Quick Start (Library Mode)
--------------------------
```python
from nexus_matcher import NexusMatcher, Config

# Initialize with defaults
matcher = NexusMatcher()

# Or with custom config
config = Config(
    embedding_model="BAAI/bge-base-en-v1.5",
    vector_store="qdrant",
    auto_approve_threshold=0.75,
)
matcher = NexusMatcher(config)

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

# Or programmatically
from nexus_matcher import create_app
app = create_app()
```

Plugin Mode
-----------
```python
from nexus_matcher.domain.ports import SchemaParser, BaseSchemaParser

class MyCustomParser(BaseSchemaParser):
    @property
    def format_name(self) -> str:
        return "my_format"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".myf"})

    def _parse_content(self, content: dict) -> Schema:
        # Custom parsing logic
        ...

# Register via entry points in pyproject.toml:
# [project.entry-points."nexus_matcher.schema_parsers"]
# my_format = "my_package.parser:MyCustomParser"
```

Version
-------
"""

__version__ = "2.0.0"
__author__ = "Pierce Lonergan"

# =============================================================================
# PUBLIC API
# =============================================================================

# Core types (always available)
from nexus_matcher.shared.types import (
    DataType,
    MatchDecision,
    ProtectionLevel,
    Score,
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

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Types
    "DataType",
    "MatchDecision",
    "ProtectionLevel",
    "Score",
    # Models
    "SchemaField",
    "DictionaryEntry",
    "MatchResult",
    "Schema",
    # DI Container
    "Container",
    "ContainerBuilder",
    "Lifecycle",
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
    if name in ("SchemaParser", "DictionaryLoader", "EmbeddingProvider",
                "VectorStore", "SparseRetriever", "Reranker", "Cache"):
        from nexus_matcher.domain import ports
        return getattr(ports, name)

    raise AttributeError(f"module 'nexus_matcher' has no attribute '{name}'")
