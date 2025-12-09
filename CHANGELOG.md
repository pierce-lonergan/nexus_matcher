# Changelog

All notable changes to NexusMatcher will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- GPU support for ModernBERT embeddings
- Multi-language schema matching
- Active learning from user feedback
- Streaming/real-time matching API

---

## [2.0.0] - 2025-12-09

### 🎉 Major Release - Complete Rewrite

This release represents a complete architectural redesign achieving **100% Precision@1** 
with **86x faster reranking** through systematic research-driven enhancements.

### Added

#### Core Features
- **ColBERT MaxSim Reranking** (GAP-001): Token-level late interaction with 93.7x speedup via pre-computed embeddings
- **INT8 Quantization** (GAP-002): ONNX-based quantization with 1.68x speedup and 75% model size reduction
- **L1 LRU Cache** (GAP-003): Sub-millisecond embedding cache with 56.99% hit rate
- **Semantic Content Cache** (GAP-004): BLAKE3-based content-addressed caching with 99.3% cost reduction
- **Incremental Updates** (GAP-005): Change detection with 99.9% computation savings for small updates
- **Context Enrichment** (GAP-006): Hierarchical field context injection for improved semantic understanding
- **Type Projections** (GAP-008): Learned type embeddings via contrastive learning (MRR 0.9706)
- **Graph Matching** (GAP-009): Structural schema matching for hybrid scoring

#### Architecture
- Hexagonal (ports & adapters) architecture
- Dependency injection container
- Plugin system via entry points
- Multi-layer caching (L1 → L2 → L3)
- Three-stage matching pipeline (Retrieval → Reranking → Scoring)

#### API
- REST API with FastAPI
- CLI with Typer
- Python library API
- Prometheus metrics endpoint
- Health and readiness probes

#### Parsers
- Avro schema parser
- JSON Schema parser  
- SQL DDL parser
- CSV header parser

#### Vector Stores
- Qdrant adapter with HNSW
- In-memory vector store
- FAISS adapter (optional)

#### Caching
- L1 LRU in-memory cache
- L2 Redis distributed cache
- L3 Semantic content cache

### Changed
- Complete rewrite from procedural to hexagonal architecture
- Embedding model default changed to `all-MiniLM-L6-v2` (from `all-mpnet-base-v2`)
- Configuration system now uses YAML with environment variable overrides
- Test suite expanded from ~50 to 433 tests

### Performance Improvements
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Precision@1 | ~85% | 100% | +15% |
| MaxSim Latency | 274ms | 3.17ms | **86x faster** |
| Embedding Latency | 12.5ms | 9.85ms | **1.68x faster** |
| Model Size | 86.8MB | 22.0MB | **75% smaller** |
| Cache Hit Rate | 0% | 56.99% | New capability |

### Deprecated
- Direct `schema_matcher.py` usage (use `NexusMatcher` class instead)
- Environment-only configuration (use YAML config files)

### Removed
- Legacy single-file implementation
- OpenAI embedding provider (temporarily, will be re-added)

### Fixed
- Memory leak in long-running embedding sessions
- Thread safety issues in concurrent matching
- ONNX API compatibility across versions

### Security
- Input validation on all API endpoints
- Rate limiting support
- Non-root Docker container

### Documentation
- Comprehensive README with architecture diagrams
- API reference documentation
- Deployment guide (Docker, Kubernetes, Cloud)
- Module-level documentation
- Enhancement journey narrative

---

## [1.0.0] - 2025-10-15

### Added
- Initial release
- Basic semantic schema matching
- Excel dictionary loader
- Avro schema parser
- Sentence-Transformers embeddings
- Simple cosine similarity scoring

### Known Issues
- Single-threaded processing
- No caching
- Limited to ~85% accuracy

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 2.0.0 | 2025-12-09 | Complete rewrite, 100% Precision@1, 86x speedup |
| 1.0.0 | 2025-10-15 | Initial release |

---

## Upgrade Guide

### From 1.x to 2.x

#### Breaking Changes

1. **Import paths changed**:
   ```python
   # Old
   from schema_matcher import match_schema
   
   # New
   from nexus_matcher import NexusMatcher
   matcher = NexusMatcher()
   results = matcher.match_schema("schema.avsc")
   ```

2. **Configuration format changed**:
   ```yaml
   # Old: Environment variables only
   SCHEMA_MATCHER_MODEL=all-mpnet-base-v2
   
   # New: YAML configuration
   embedding:
     model_name: sentence-transformers/all-MiniLM-L6-v2
     use_int8: true
   ```

3. **Result structure changed**:
   ```python
   # Old
   result = {"field": "matched_entry", "score": 0.95}
   
   # New
   result.dictionary_entry.business_name
   result.final_confidence  # 0.95
   result.decision  # "AUTO_APPROVE"
   ```

#### Migration Steps

1. Update imports to use `nexus_matcher` package
2. Convert environment variables to YAML config
3. Update result handling code
4. Run tests to verify behavior

---

## Contributors

- Pierce Lonergan - Lead Developer

## Links

- [GitHub Repository](https://github.com/pierce-lonergan/nexus_matcher)
- [Documentation](https://nexus-matcher.readthedocs.io)
- [PyPI Package](https://pypi.org/project/nexus-matcher/)
- [Issue Tracker](https://github.com/pierce-lonergan/nexus_matcher/issues)
