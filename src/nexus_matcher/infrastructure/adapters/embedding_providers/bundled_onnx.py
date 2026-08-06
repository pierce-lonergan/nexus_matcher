"""
nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx | Layer: INFRASTRUCTURE
The zero-setup embedding provider: an int8 ONNX encoder shipped inside the wheel.

## Relationships
# IMPLEMENTS  → domain/ports/embedding_provider :: EmbeddingProvider protocol
# DEPENDS_ON  → onnxruntime, tokenizers :: inference only, NO torch
# USED_BY     → shared/container :: the default provider

## Attributes
# Security: Never touches the network. No download, no HuggingFace, no telemetry.
# Performance: ~1259 queries/sec on CPU, faster than the fp32 torch path it replaces
# Reliability: Works in an airgapped container with no configuration

## Why this is the default

The previous default (`SentenceTransformersProvider`) needs torch (~800 MB installed) and
downloads weights from HuggingFace on first use. That fails in an airgapped environment,
behind a proxy, in a locked-down CI image, or simply on a slow connection -- and it fails
at RUN time, not install time, which is the worst moment.

This provider carries its own weights. Measured against the torch fp32 path on the
labelled benchmark:

    build                 size            P@1      Recall@10   throughput   torch
    torch fp32       130 MB + ~800 MB   0.5596      0.8648       973 q/s     yes
    int8 ONNX (this)      33.8 MB       0.5378      0.8532      1238 q/s     no

int8 costs roughly 2 points of P@1 and buys a 27% throughput gain plus the removal of
torch.

"Roughly" is load-bearing. int8 inference is NOT batch-invariant: ONNX Runtime selects
different quantised GEMM kernels per input shape, so the SAME corpus scores
P@1 0.5276 at batch 8, 0.5378 at batch 64 and 0.5436 at batch 128 -- a 1.6-point spread
from batch size alone, wider than several effects that are worth acting on. Hold batch
size fixed when comparing anything, or you are measuring kernel selection rather than the
change you made. The fp32 torch path does not have this property. That is a good trade for eliminating setup, and the transformer path
stays available via `pip install nexus-matcher[embeddings]` when those points matter.

Both rows use the model's official CLS pooling. An earlier measurement of this same file
reported only -1.2 P@1, which was wrong: it compared torch-CLS against an ONNX export that
sentence-transformers had silently defaulted to MEAN pooling, because the exported
directory carries no 1_Pooling config. Measured directly, CLS 0.5378 vs mean 0.5407 is
within noise -- the pooling was never the story, the quantisation was. If you re-measure
this, pin the pooling on both sides or you will compare two variables at once.

## What is NOT used, and why

Static embeddings (model2vec / potion) are ~30 MB, need only numpy, and are ~15-70x
faster still. They were measured and rejected as a default: on a FHIR-derived corpus
built to mirror the flattened-Avro use case they cost **8.9 points of P@1** (0.407 ->
0.318). A separate finding that seven TRANSFORMER encoders from 22M to 335M parameters
all landed within 0.03 P@1 does NOT generalise to static models -- the gap is roughly
three times what their published MTEB ratio implies. For a tool whose output decides
whether a field inherits a PII classification, nine points is not purchasable with
latency. `StaticEmbeddingProvider` exists as a fallback for environments without
onnxruntime, not as a recommendation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nexus_matcher.shared.types.base import Result

# The bundled encoder. Query-side instruction is part of the model contract: BGE models
# are trained with it on the query side ONLY, and omitting it cost 5.3 points of P@1.
BUNDLED_MODEL_DIR = Path(__file__).resolve().parents[3] / "_models" / "bge-small-en-v1.5-onnx-int8"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384
MAX_TOKENS = 512


def bundled_model_available() -> bool:
    """True when the wheel actually carries the weights."""
    return (BUNDLED_MODEL_DIR / "model_quantized.onnx").is_file()


class BundledOnnxProvider:
    """
    Embedding provider backed by the int8 ONNX encoder shipped in this package.

    Requires `onnxruntime` and `tokenizers`, both pure-wheel installs. Does NOT require
    torch, transformers, or sentence-transformers, and never contacts the network.

    Args:
        model_dir: Override the model location. Defaults to the bundled copy.
        query_instruction: Prefix applied to QUERIES only. Pass "" to disable, but
            measure first -- dropping it cost 5.3 points of P@1 on our benchmark.
        num_threads: ONNX Runtime intra-op threads. None lets ORT decide, which is
            usually right; set it explicitly when running many workers per host to stop
            them oversubscribing the same cores.

    Example:
        provider = BundledOnnxProvider()
        vectors = provider.embed_documents(["Customer Account Balance"])
        query = provider.embed_single("cust_acct_bal")
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        query_instruction: str = QUERY_INSTRUCTION,
        num_threads: int | None = None,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else BUNDLED_MODEL_DIR
        self._query_instruction = query_instruction
        self._num_threads = num_threads
        self._session: Any = None
        self._tokenizer: Any = None

    # -- lifecycle --------------------------------------------------------

    def _load(self) -> None:
        if self._session is not None:
            return

        onnx_path = self._model_dir / "model_quantized.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"No bundled encoder at {onnx_path}. This usually means the package was "
                f"built without its model data. Reinstall nexus-matcher, or pass "
                f"model_dir=..., or use SentenceTransformersProvider with "
                f"pip install nexus-matcher[embeddings]."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "onnxruntime is required for the bundled encoder. "
                "Install with: pip install onnxruntime"
            ) from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "tokenizers is required for the bundled encoder. "
                "Install with: pip install tokenizers"
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._num_threads:
            options.intra_op_num_threads = self._num_threads
        # Deterministic single-provider execution; no GPU probing, no surprises.
        self._session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )

        tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer = tokenizer

    # -- properties -------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "bge-small-en-v1.5-onnx-int8 (bundled)"

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def max_tokens(self) -> int:
        return MAX_TOKENS

    @property
    def is_offline(self) -> bool:
        """True: this provider never reaches the network."""
        return True

    # -- encoding ---------------------------------------------------------

    def _encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        expected = {i.name for i in self._session.get_inputs()}
        out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            encoded = self._tokenizer.encode_batch(chunk)

            ids = np.asarray([e.ids for e in encoded], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in encoded], dtype=np.int64)

            feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in expected:
                feeds["token_type_ids"] = np.asarray([e.type_ids for e in encoded], dtype=np.int64)
            feeds = {k: v for k, v in feeds.items() if k in expected}

            hidden = self._session.run(None, feeds)[0]

            # BGE uses CLS pooling, not mean pooling. Using the wrong pooling silently
            # degrades every score rather than raising, so it is pinned here explicitly.
            vectors = hidden[:, 0, :].astype(np.float32)

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)
            out[start : start + len(chunk)] = vectors / norms

        return out

    def embed_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Encode DICTIONARY entries. No instruction prefix -- BGE is asymmetric."""
        return self._encode(list(texts), batch_size)

    def embed_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Encode QUERIES, applying the model's query instruction."""
        return self._encode([self._query_instruction + t for t in texts], batch_size)

    # -- EmbeddingProvider protocol ---------------------------------------

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> Result:
        """Batch-encode as queries. This is the interface NexusMatcher calls per schema."""
        try:
            arr = self.embed_queries(list(texts), batch_size)
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


def default_embedding_provider(prefer: str = "auto") -> Any:
    """
    Pick an embedding provider that will actually work here, in preference order.

    Order: bundled ONNX -> sentence-transformers -> static. The bundled encoder is first
    because it is the only option that cannot fail at run time for environmental reasons.

    Args:
        prefer: "auto" (default), "bundled", "transformer", or "static" to force one.

    Returns:
        A ready provider.

    Raises:
        RuntimeError: if nothing usable is installed, listing what to install.
    """
    tried: list[str] = []

    def try_bundled():
        if not bundled_model_available():
            tried.append("bundled ONNX: weights not present in this install")
            return None
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
        except ImportError as exc:
            tried.append(f"bundled ONNX: {exc}")
            return None
        return BundledOnnxProvider()

    def try_transformer():
        # Respect an explicit offline request rather than letting HF hang on a socket.
        if os.environ.get("HF_HUB_OFFLINE") == "1" and not os.environ.get("NEXUS_ST_MODEL_PATH"):
            tried.append("sentence-transformers: HF_HUB_OFFLINE=1 and no local model path")
            return None
        try:
            from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
                SentenceTransformersProvider,
            )

            return SentenceTransformersProvider(
                os.environ.get("NEXUS_ST_MODEL_PATH", "BAAI/bge-small-en-v1.5")
            )
        except Exception as exc:
            tried.append(f"sentence-transformers: {type(exc).__name__}: {exc}")
            return None

    def try_static():
        try:
            from model2vec import StaticModel  # noqa: F401
        except ImportError as exc:
            tried.append(f"static: {exc}")
            return None
        from nexus_matcher.infrastructure.adapters.embedding_providers.static_embedding import (
            StaticEmbeddingProvider,
        )

        return StaticEmbeddingProvider()

    order = {
        "auto": (try_bundled, try_transformer, try_static),
        "bundled": (try_bundled,),
        "transformer": (try_transformer,),
        "static": (try_static,),
    }.get(prefer)
    if order is None:
        raise ValueError(f"prefer must be auto/bundled/transformer/static, got {prefer!r}")

    for factory in order:
        provider = factory()
        if provider is not None:
            return provider

    raise RuntimeError(
        "No usable embedding provider. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\n\nFix with ONE of:\n"
        "  pip install nexus-matcher            # bundled encoder, no download (recommended)\n"
        "  pip install nexus-matcher[embeddings]  # torch + sentence-transformers\n"
        "  pip install model2vec                # static fallback, lower accuracy"
    )
