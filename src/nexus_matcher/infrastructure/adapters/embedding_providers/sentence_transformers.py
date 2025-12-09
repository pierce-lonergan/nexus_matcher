"""
nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers | Layer: INFRASTRUCTURE
SentenceTransformers embedding provider implementation.

## Relationships
# IMPLEMENTS → domain/ports/embedding_provider :: EmbeddingProvider protocol
# DEPENDS_ON → sentence-transformers :: embedding generation
# USED_BY    → application/use_cases/match_schema :: embedding generation

## Attributes
# Security: Models loaded from local cache or HuggingFace
# Performance: GPU acceleration when available, batched processing
# Reliability: Lazy model loading, graceful fallback to CPU
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nexus_matcher.domain.ports.embedding_provider import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
)


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Embedding provider using SentenceTransformers library.

    Supports all models from the sentence-transformers library and
    HuggingFace Hub. Recommended models:

    - BAAI/bge-base-en-v1.5 (768 dim) - Good balance of quality/speed
    - BAAI/bge-large-en-v1.5 (1024 dim) - Higher quality
    - all-MiniLM-L6-v2 (384 dim) - Fast, good for prototyping
    - all-mpnet-base-v2 (768 dim) - High quality general purpose

    Example:
        provider = SentenceTransformersProvider("BAAI/bge-base-en-v1.5")
        result = provider.embed(["customer email", "transaction amount"])
        if result.is_success:
            embeddings = result.unwrap()
            print(f"Generated {embeddings.count} embeddings")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize the embedding provider.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to use (cpu, cuda, mps, or None for auto)
            cache_dir: Directory for model cache
        """
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Get embedding dimension (lazy loaded)."""
        if self._dimension is None:
            self._load_model()
        return self._dimension or 768

    @property
    def max_tokens(self) -> int:
        """Get maximum input tokens."""
        return 512

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install nexus-matcher[embeddings]"
            )

        # Determine device
        device = self._device
        if device is None or device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Load model
        self._model = SentenceTransformer(
            self._model_name,
            device=device,
            cache_folder=self._cache_dir,
        )

        # Get dimension from model
        self._dimension = self._model.get_sentence_embedding_dimension()

    def _encode_batch(
        self,
        texts: Sequence[str],
        config: EmbeddingConfig,
    ) -> np.ndarray:
        """
        Encode a batch of texts.

        Args:
            texts: Texts to encode
            config: Encoding configuration

        Returns:
            2D array of embeddings
        """
        self._load_model()

        embeddings = self._model.encode(
            list(texts),
            batch_size=config.batch_size,
            show_progress_bar=config.show_progress,
            normalize_embeddings=False,  # We handle normalization in base class
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Mock embedding provider for testing.

    Generates deterministic random embeddings based on text hash.
    Useful for unit tests where actual embeddings aren't needed.
    """

    def __init__(self, dimension: int = 768) -> None:
        """
        Initialize mock provider.

        Args:
            dimension: Embedding dimension
        """
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return "mock"

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def _encode_batch(
        self,
        texts: Sequence[str],
        config: EmbeddingConfig,
    ) -> np.ndarray:
        """Generate mock embeddings."""
        embeddings = []

        for text in texts:
            # Use text hash as seed for reproducibility
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            embedding = rng.randn(self._dimension).astype(np.float32)
            embeddings.append(embedding)

        return np.vstack(embeddings)
