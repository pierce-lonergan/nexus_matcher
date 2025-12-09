"""
nexus_matcher.infrastructure.config.settings | Layer: INFRASTRUCTURE
Configuration system with environment profiles and validation.

## Relationships
# USED_BY    → application/* :: all use cases read config
# USED_BY    → infrastructure/adapters/* :: adapters use config
# USED_BY    → presentation/* :: API/CLI use config
# DEPENDS_ON → pydantic-settings :: environment loading

## Attributes
# Security: Secrets loaded from environment, never logged
# Performance: Config is immutable singleton
# Reliability: Validation on load prevents runtime errors
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# ENVIRONMENT ENUM
# =============================================================================


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def from_string(cls, value: str) -> Environment:
        """Parse environment string."""
        value_lower = value.lower()
        for env in cls:
            if env.value == value_lower:
                return env
        return cls.DEVELOPMENT


# =============================================================================
# COMPONENT CONFIGS
# =============================================================================


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding generation."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_EMBEDDING_")

    model_name: str = "BAAI/bge-base-en-v1.5"
    dimension: int = 768
    max_length: int = 512
    batch_size: int = 32
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    normalize: bool = True
    cache_dir: Path | None = None

    # Type augmentation
    type_embedding_dim: int = 32
    use_type_augmentation: bool = True

    # Abbreviation expansion
    expand_abbreviations: bool = True
    abbreviation_api_url: str | None = None


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_VECTOR_")

    store_type: Literal["qdrant", "faiss", "memory"] = "qdrant"
    collection_name: str = "dictionary_entries"
    dimension: int = 768

    # Qdrant specific
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_https: bool = False
    qdrant_path: Path | None = None  # For local mode

    # FAISS specific
    faiss_index_type: str = "IVFFlat"
    faiss_nlist: int = 100

    # Index parameters
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    use_scalar_quantization: bool = True


class SparseRetrieverConfig(BaseSettings):
    """Configuration for sparse retrieval."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_SPARSE_")

    retriever_type: Literal["bm25", "elasticsearch", "none"] = "bm25"
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 100

    # Elasticsearch specific
    es_host: str = "localhost"
    es_port: int = 9200
    es_index: str = "dictionary"


class RerankerConfig(BaseSettings):
    """Configuration for reranking."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_RERANKER_")

    # ColBERT
    use_colbert: bool = True
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_top_k: int = 50

    # Cross-encoder
    use_cross_encoder: bool = True
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    cross_encoder_top_k: int = 20

    # Batch sizes
    reranker_batch_size: int = 32


class CacheConfig(BaseSettings):
    """Configuration for caching."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_CACHE_")

    # L1 - In-memory
    l1_enabled: bool = True
    l1_max_size: int = 10000
    l1_ttl_seconds: int = 3600

    # L2 - Redis
    l2_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0
    l2_ttl_seconds: int = 3600

    # L3 - Semantic
    l3_enabled: bool = True
    l3_similarity_threshold: float = 0.92
    l3_max_size: int = 50000


class ScoringConfig(BaseSettings):
    """Configuration for confidence scoring."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_SCORING_")

    # Score weights (must sum to 1.0)
    semantic_weight: float = 0.70
    lexical_weight: float = 0.05
    edit_distance_weight: float = 0.05
    type_weight: float = 0.05
    domain_weight: float = 0.15

    # Thresholds
    auto_approve_threshold: float = 0.75
    review_threshold: float = 0.50
    reject_threshold: float = 0.30

    # Confidence gap
    min_confidence_gap: float = 0.10

    # Graph refinement
    use_graph_refinement: bool = True
    graph_propagation_factor: float = 0.1

    @model_validator(mode="after")
    def validate_weights(self) -> ScoringConfig:
        """Ensure weights sum to 1.0."""
        total = (
            self.semantic_weight
            + self.lexical_weight
            + self.edit_distance_weight
            + self.type_weight
            + self.domain_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Score weights must sum to 1.0, got {total}")
        return self


class FusionConfig(BaseSettings):
    """Configuration for dense/sparse fusion."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_FUSION_")

    fusion_method: Literal["convex", "rrf"] = "convex"
    dense_weight: float = 0.65  # alpha in convex combination
    rrf_k: int = 60  # k parameter for RRF


class APIConfig(BaseSettings):
    """Configuration for REST API."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_API_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: list[str] = ["*"]
    api_key: str | None = None
    rate_limit: int = 100  # requests per minute


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    file_path: Path | None = None
    redact_pii: bool = True


# =============================================================================
# MAIN CONFIG
# =============================================================================


class Config(BaseSettings):
    """
    Main configuration for NexusMatcher.

    Loads from environment variables with NEXUS_ prefix.
    Supports .env files and profile-based configuration.

    Example:
        # From environment
        config = Config()

        # With overrides
        config = Config(environment="production", embedding=EmbeddingConfig(device="cuda"))

        # From .env file
        config = Config(_env_file=".env.production")
    """

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Core settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")

    # Component configs
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    sparse_retriever: SparseRetrieverConfig = Field(default_factory=SparseRetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("environment", mode="before")
    @classmethod
    def parse_environment(cls, v: Any) -> Environment:
        """Parse environment from string."""
        if isinstance(v, str):
            return Environment.from_string(v)
        return v

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> Config:
        """Apply environment-specific defaults."""
        if self.environment == Environment.PRODUCTION:
            # Production: stricter settings
            self.debug = False
            self.logging.level = "INFO"
            self.api.reload = False
        elif self.environment == Environment.TESTING:
            # Testing: fast settings
            self.cache.l2_enabled = False  # No Redis in tests
            self.vector_store.store_type = "memory"
        elif self.environment == Environment.DEVELOPMENT:
            # Development: verbose settings
            self.debug = True
            self.logging.level = "DEBUG"
            self.api.reload = True

        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in tests."""
        return self.environment == Environment.TESTING


# =============================================================================
# CONFIG LOADING
# =============================================================================


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the global configuration singleton.

    Configuration is loaded once and cached.

    Returns:
        Config instance
    """
    return Config()


def load_config(
    env_file: str | Path | None = None,
    **overrides: Any,
) -> Config:
    """
    Load configuration with optional overrides.

    Args:
        env_file: Path to .env file
        **overrides: Config field overrides

    Returns:
        Config instance
    """
    kwargs: dict[str, Any] = {}

    if env_file:
        kwargs["_env_file"] = env_file

    kwargs.update(overrides)

    return Config(**kwargs)


def reset_config() -> None:
    """Reset the cached configuration."""
    get_config.cache_clear()


# =============================================================================
# PROFILE PRESETS
# =============================================================================


class ConfigProfiles:
    """Pre-built configuration profiles."""

    @staticmethod
    def development() -> Config:
        """Development profile with verbose logging and fast iteration."""
        return Config(
            environment=Environment.DEVELOPMENT,
            debug=True,
            vector_store=VectorStoreConfig(store_type="memory"),
            cache=CacheConfig(l2_enabled=False),
            logging=LoggingConfig(level="DEBUG", format="text"),
            api=APIConfig(reload=True, workers=1),
        )

    @staticmethod
    def testing() -> Config:
        """Testing profile with in-memory stores."""
        return Config(
            environment=Environment.TESTING,
            debug=True,
            vector_store=VectorStoreConfig(store_type="memory"),
            sparse_retriever=SparseRetrieverConfig(retriever_type="bm25"),
            cache=CacheConfig(l1_enabled=True, l2_enabled=False, l3_enabled=False),
            reranker=RerankerConfig(use_colbert=False, use_cross_encoder=False),
            logging=LoggingConfig(level="WARNING"),
        )

    @staticmethod
    def production() -> Config:
        """Production profile with full features."""
        return Config(
            environment=Environment.PRODUCTION,
            debug=False,
            vector_store=VectorStoreConfig(
                store_type="qdrant",
                use_scalar_quantization=True,
            ),
            cache=CacheConfig(
                l1_enabled=True,
                l2_enabled=True,
                l3_enabled=True,
            ),
            scoring=ScoringConfig(
                auto_approve_threshold=0.75,
            ),
            logging=LoggingConfig(level="INFO", format="json", redact_pii=True),
            api=APIConfig(reload=False, workers=4),
        )

    @staticmethod
    def high_accuracy() -> Config:
        """Profile optimized for accuracy over speed."""
        return Config(
            reranker=RerankerConfig(
                use_colbert=True,
                colbert_top_k=100,
                use_cross_encoder=True,
                cross_encoder_top_k=50,
            ),
            scoring=ScoringConfig(
                use_graph_refinement=True,
                auto_approve_threshold=0.80,
            ),
        )

    @staticmethod
    def low_latency() -> Config:
        """Profile optimized for speed over accuracy."""
        return Config(
            reranker=RerankerConfig(
                use_colbert=False,
                use_cross_encoder=True,
                cross_encoder_top_k=10,
            ),
            cache=CacheConfig(
                l1_enabled=True,
                l1_max_size=50000,
                l3_similarity_threshold=0.88,
            ),
            scoring=ScoringConfig(
                use_graph_refinement=False,
            ),
        )
