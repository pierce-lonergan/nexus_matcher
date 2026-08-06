"""
nexus_matcher.domain.ports.embedding_provider | Layer: DOMAIN
Port interface for embedding generation providers.

## Relationships
# USED_BY    → domain/services/embedding_service :: embedding generation
# USED_BY    → infrastructure/adapters/embedding_providers/* :: implementations
# DEPENDS_ON → shared/types/base :: EmbeddingVector type

## Attributes
# Security: API keys should be loaded from environment, never logged
# Performance: Should support batching for efficiency
# Reliability: Should handle model loading failures gracefully

## Extension Points
# Plugin: nexus_matcher.embedding_providers entry point
# Register: EmbeddingProviderRegistry.register()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from nexus_matcher.shared.types.base import EmbeddingVector, Result

# =============================================================================
# EMBEDDING CONFIG
# =============================================================================


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation."""

    normalize: bool = True
    batch_size: int = 32
    max_length: int = 512
    show_progress: bool = False
    device: str = "cpu"  # "cpu", "cuda", "mps"


# =============================================================================
# EMBEDDING RESULT
# =============================================================================


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of embedding generation."""

    embeddings: tuple[EmbeddingVector, ...]
    model_name: str
    dimension: int
    tokens_used: int = 0

    @property
    def count(self) -> int:
        """Number of embeddings."""
        return len(self.embeddings)

    def as_array(self) -> np.ndarray:
        """Get embeddings as 2D numpy array."""
        return np.vstack(self.embeddings)


# =============================================================================
# EMBEDDING PROVIDER PROTOCOL
# =============================================================================


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    Implementations generate vector embeddings from text using various
    models (sentence-transformers, OpenAI, etc.).

    Example usage:
        provider = SentenceTransformersProvider("BAAI/bge-base-en-v1.5")
        result = provider.embed(["customer email", "transaction amount"])
        if result.is_success:
            embeddings = result.unwrap()
            print(f"Generated {embeddings.count} embeddings of dim {embeddings.dimension}")
    """

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    @property
    def max_tokens(self) -> int:
        """Get maximum input tokens."""
        ...

    def embed(
        self,
        texts: Sequence[str],
        config: EmbeddingConfig | None = None,
    ) -> Result[EmbeddingResult]:
        """
        Generate embeddings for texts.

        Args:
            texts: Texts to embed
            config: Embedding configuration

        Returns:
            Result containing EmbeddingResult on success
        """
        ...

    def embed_single(
        self,
        text: str,
        config: EmbeddingConfig | None = None,
    ) -> Result[EmbeddingVector]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            config: Embedding configuration

        Returns:
            Result containing embedding vector on success
        """
        ...

    def similarity(
        self,
        embedding1: EmbeddingVector,
        embedding2: EmbeddingVector,
    ) -> float:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1)
        """
        ...


# =============================================================================
# BASE IMPLEMENTATION
# =============================================================================


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Provides:
    - Batch processing
    - Normalization
    - Similarity calculation
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    @property
    def max_tokens(self) -> int:
        """Maximum input tokens (default 512)."""
        return 512

    @abstractmethod
    def _encode_batch(
        self,
        texts: Sequence[str],
        config: EmbeddingConfig,
    ) -> np.ndarray:
        """
        Internal batch encoding.

        Args:
            texts: Texts to encode
            config: Configuration

        Returns:
            2D array of embeddings (n_texts, dimension)
        """
        ...

    def embed(
        self,
        texts: Sequence[str],
        config: EmbeddingConfig | None = None,
    ) -> Result[EmbeddingResult]:
        """Generate embeddings for texts."""
        if not texts:
            return Result.failure("No texts provided")

        cfg = config or EmbeddingConfig()

        try:
            # Process in batches
            all_embeddings: list[EmbeddingVector] = []

            for i in range(0, len(texts), cfg.batch_size):
                batch = texts[i : i + cfg.batch_size]
                batch_embeddings = self._encode_batch(batch, cfg)

                # Normalize if requested
                if cfg.normalize:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / (norms + 1e-9)

                all_embeddings.extend(batch_embeddings)

            result = EmbeddingResult(
                embeddings=tuple(all_embeddings),
                model_name=self.model_name,
                dimension=self.dimension,
            )

            return Result.success(result)

        except Exception as e:
            return Result.failure(f"Embedding failed: {e}")

    def embed_single(
        self,
        text: str,
        config: EmbeddingConfig | None = None,
    ) -> Result[EmbeddingVector]:
        """Generate embedding for single text."""
        result = self.embed([text], config)
        if result.is_failure:
            return Result.failure(result.error or "Unknown error")
        return Result.success(result.unwrap().embeddings[0])

    def similarity(
        self,
        embedding1: EmbeddingVector,
        embedding2: EmbeddingVector,
    ) -> float:
        """Calculate cosine similarity."""
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================


class EmbeddingProviderRegistry:
    """
    Registry for embedding provider implementations.

    Example:
        registry = EmbeddingProviderRegistry()
        registry.register("sentence_transformers", SentenceTransformersProvider)

        provider = registry.create("sentence_transformers", model="BAAI/bge-base-en-v1.5")
    """

    def __init__(self) -> None:
        self._factories: dict[str, type[EmbeddingProvider]] = {}

    def register(self, provider_type: str, factory: type[EmbeddingProvider]) -> None:
        """Register an embedding provider factory."""
        self._factories[provider_type] = factory

    def create(
        self,
        provider_type: str,
        **kwargs,
    ) -> EmbeddingProvider | None:
        """Create a provider instance."""
        factory = self._factories.get(provider_type)
        if factory:
            return factory(**kwargs)
        return None

    def list_providers(self) -> list[str]:
        """Get list of registered provider types."""
        return list(self._factories.keys())

    def load_entry_points(self) -> None:
        """Load providers from entry points."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group="nexus_matcher.embedding_providers")
            for ep in eps:
                try:
                    provider_class = ep.load()
                    self.register(ep.name, provider_class)
                except Exception as e:
                    import logging

                    logging.warning(f"Failed to load embedding provider {ep.name}: {e}")
        except ImportError:
            pass
