"""
nexus_matcher.infrastructure.adapters.embedding_providers.static_embedding | Layer: INFRASTRUCTURE
Static (model2vec / potion) embeddings -- the last-resort fallback, NOT a recommendation.

## Relationships
# IMPLEMENTS  → domain/ports/embedding_provider :: EmbeddingProvider protocol
# DEPENDS_ON  → model2vec :: numpy-only static embeddings
# USED_BY     → embedding_providers/bundled_onnx :: default_embedding_provider fallback

## Attributes
# Security: No network access once the model is local
# Performance: ~70,000 texts/sec on CPU -- roughly 60x the ONNX encoder
# Reliability: Pure numpy at inference; no onnxruntime, no torch

## Read this before choosing it

Static embeddings look irresistible on paper: ~30 MB, numpy-only, tens of thousands of
texts per second. They were measured and REJECTED as the default.

On a FHIR-derived corpus built to mirror the flattened-Avro governance use case:

    encoder                      P@1      parent-path gain
    bge-small int8 ONNX         0.4071        +19.0
    potion-retrieval-32M        0.3183        +16.3
    potion-base-8M              0.2981        +17.3

That is **8.9 points of P@1** for the speed. And on the general benchmark the gap was
6.5 points (0.5596 -> 0.4942).

A separate result -- that seven TRANSFORMER encoders spanning 22M to 335M parameters all
landed within 0.03 P@1 of each other -- does NOT license the conclusion that "model choice
does not matter". It does not transfer to static models, where the measured gap is about
three times what the published MTEB ratio would predict.

For a system whose output decides whether a field inherits a PII classification, nine
points of accuracy is not a reasonable purchase with latency. Use this only when
onnxruntime is genuinely unavailable, or as a fast first-stage retriever whose top-k is
rescored by a real encoder -- static top-25 followed by transformer rescoring measured
0.5581 against 0.5596 for transformer-only, i.e. no meaningful loss.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from nexus_matcher.shared.types.base import Result

DEFAULT_STATIC_MODEL = "minishlab/potion-base-8M"


class StaticEmbeddingProvider:
    """
    Embedding provider backed by a model2vec static model.

    Args:
        model_name: A model2vec model id or local path.
        query_instruction: Applied to queries. Empty by default -- static models are not
            instruction-tuned, so a prefix adds tokens without adding signal.

    Example:
        provider = StaticEmbeddingProvider()
        vectors = provider.embed_documents(["Customer Account Balance"])

    Note:
        Prefer `BundledOnnxProvider`. See the module docstring for the measured cost.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_STATIC_MODEL,
        query_instruction: str = "",
    ) -> None:
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "model2vec is required for StaticEmbeddingProvider. "
                "Install with: pip install model2vec"
            ) from exc
        self._model = StaticModel.from_pretrained(self._model_name)

    @property
    def model_name(self) -> str:
        return f"{self._model_name} (static)"

    @property
    def dimension(self) -> int:
        self._load()
        return int(self._model.dim)

    @property
    def max_tokens(self) -> int:
        # Static models bag tokens rather than attending over them, so there is no
        # architectural context limit; this is a practical cap on input size.
        return 512

    @property
    def is_offline(self) -> bool:
        """False: the model is fetched on first use unless already cached locally."""
        return False

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        return vectors / norms

    def embed_documents(self, texts: Sequence[str], batch_size: int = 1024) -> np.ndarray:
        return self._encode(list(texts))

    def embed_queries(self, texts: Sequence[str], batch_size: int = 1024) -> np.ndarray:
        return self._encode([self._query_instruction + t for t in texts])

    def embed(self, texts: Sequence[str], batch_size: int = 1024) -> Result:
        try:
            arr = self.embed_queries(list(texts))
        except Exception as exc:
            return Result.failure(f"Embedding failed: {type(exc).__name__}: {exc}")

        class _Batch:
            embeddings = arr
            count = len(arr)

        return Result.success(_Batch())

    def embed_single(self, text: str) -> Result:
        try:
            return Result.success(self.embed_queries([text])[0])
        except Exception as exc:
            return Result.failure(f"Embedding failed: {type(exc).__name__}: {exc}")
