"""
nexus_matcher.infrastructure.adapters.embedding_providers.quantized | Layer: INFRASTRUCTURE
INT8 quantized embedding provider for CPU speedup.

## Relationships
# IMPLEMENTS → domain/ports/embedding_provider :: EmbeddingProvider protocol
# DEPENDS_ON → onnxruntime (+onnx) or torch :: quantized inference
# USED_BY    → application/use_cases/match_schema :: embedding generation

## Attributes
# Security: Models loaded from the local HuggingFace cache or an explicit path
# Performance: INT8 dynamic quantization, VNNI detection
# Reliability: Raises on any unavailable backend. NEVER synthesizes embeddings.

## Research Reference
# README_RESEARCH_2.md, Lines 9-18
# README_RESEARCH_3.md, Lines 9-11
# Target: 3-10x speedup, <2% accuracy loss, <=15ms batch-32 latency

## Correctness note (2026-08 hardening pass)
# `QuantizedEmbeddingProvider._encode_batch` previously returned
# `np.random.RandomState(hash(text)).randn(dim)` behind a `# TODO: implement`
# comment. It presented as an embedding provider and returned RANDOM NOISE, so
# every downstream similarity score was meaningless while looking entirely
# plausible -- normalized float32 vectors of the right shape, stable across
# calls for the same text. That has been removed. This module now either runs
# a real quantized model or raises.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nexus_matcher.domain.ports.embedding_provider import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CPU FEATURE DETECTION
# =============================================================================


def detect_cpu_features() -> dict[str, bool]:
    """
    Detect CPU features relevant for quantization.

    Returns:
        Dictionary of feature availability:
        - vnni: Vector Neural Network Instructions (required for best INT8 perf)
        - avx2: Advanced Vector Extensions 2
        - avx512: AVX-512 instruction set
        - amx: Advanced Matrix Extensions (Intel 4th gen+)
    """
    features = {
        "vnni": False,
        "avx2": False,
        "avx512": False,
        "amx": False,
    }

    system = platform.system().lower()

    try:
        if system == "linux":
            with Path("/proc/cpuinfo").open() as f:
                cpuinfo = f.read().lower()
                features["avx2"] = "avx2" in cpuinfo
                features["avx512"] = "avx512" in cpuinfo or "avx512f" in cpuinfo
                features["vnni"] = "avx512_vnni" in cpuinfo or "avx_vnni" in cpuinfo
                features["amx"] = "amx" in cpuinfo

        elif system == "darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-a"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                sysctl_output = result.stdout.lower()
                features["avx2"] = "avx2" in sysctl_output
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        elif system == "windows":
            try:
                import cpuinfo

                info = cpuinfo.get_cpu_info()
                flags = info.get("flags", [])
                features["avx2"] = "avx2" in flags
                features["avx512"] = any("avx512" in f for f in flags)
                features["vnni"] = any("vnni" in f for f in flags)
            except ImportError:
                pass

    except Exception:
        # Default to False for all features on error
        pass

    return features


def is_quantization_recommended() -> bool:
    """
    Check if INT8 quantization is recommended for this CPU.

    VNNI (Vector Neural Network Instructions) provides significant
    speedup for INT8 operations. Available on:
    - Intel Ice Lake (2019+)
    - Intel Tiger Lake, Alder Lake, etc.
    - AMD Zen 4 (2022+)

    Returns:
        True if VNNI available and INT8 quantization recommended
    """
    features = detect_cpu_features()

    if features.get("vnni"):
        return True

    if features.get("avx512"):
        return True

    return features.get("avx2", False)


def is_backend_available(backend: str) -> bool:
    """
    Check if a quantization backend is genuinely usable end to end.

    This checks for everything needed to BUILD and RUN a quantized model, not
    merely that a package name imports. In particular the "onnx" backend needs
    both `onnxruntime` (inference) and `onnx` (graph export + the
    `onnxruntime.quantization` toolchain). `onnxruntime` alone cannot quantize
    anything, and reporting it as available is what let the old code claim an
    ONNX path it could not actually take.

    Args:
        backend: Backend name ("onnx", "torch", "openvino", "auto")

    Returns:
        True if backend is available
    """
    if backend == "onnx":
        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
            import torch
            import transformers
            from onnxruntime.quantization import quantize_dynamic

            return True
        except ImportError:
            return False

    if backend == "torch":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from torch.ao.quantization import quantize_dynamic  # noqa: F401

            return True
        except ImportError:
            return False

    if backend == "openvino":
        # Deliberately always False: no OpenVINO inference path is implemented
        # in this module. See _load_openvino_model().
        return False

    if backend == "auto":
        return is_backend_available("onnx") or is_backend_available("torch")

    return False


# =============================================================================
# QUANTIZATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class QuantizationConfig:
    """
    Configuration for embedding quantization.

    Attributes:
        precision: "int8", "int4", "fp16" or "fp32". Only "int8" and "fp32" are
            implemented; "int4"/"fp16" raise at load time rather than silently
            falling back to a different precision.
        backend: "onnx", "torch", "openvino" or "auto".
        optimize_for_inference: Enable graph optimizations (ONNX backend).
        num_threads: Intra-op thread count, 0 = library default.
        enable_profiling: Reserved for backend profiling hooks.
        pooling: "auto", "cls" or "mean". How token vectors are reduced to one
            sentence vector. "auto" picks CLS for BGE/GTE-family models and mean
            pooling otherwise, matching how those models were trained.
    """

    precision: str = "int8"  # "int8", "int4", "fp16", "fp32"
    backend: str = "auto"  # "onnx", "torch", "openvino", "auto"
    optimize_for_inference: bool = True
    num_threads: int = 0  # 0 = auto
    enable_profiling: bool = False
    pooling: str = "auto"  # "auto", "cls", "mean"

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_precisions = ("int8", "int4", "fp16", "fp32")
        if self.precision not in valid_precisions:
            raise ValueError(f"precision must be one of {valid_precisions}")

        valid_backends = ("onnx", "torch", "openvino", "auto")
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        valid_pooling = ("auto", "cls", "mean")
        if self.pooling not in valid_pooling:
            raise ValueError(f"pooling must be one of {valid_pooling}")


# =============================================================================
# QUANTIZATION STATISTICS
# =============================================================================


@dataclass
class QuantizationStats:
    """Statistics for quantized inference."""

    total_inferences: int = 0
    total_texts_processed: int = 0
    total_latency_ms: float = 0.0
    baseline_latency_ms: float | None = None

    # Internal tracking
    _latencies: list[float] = field(default_factory=list)

    def record_inference(self, batch_size: int, latency_ms: float) -> None:
        """Record an inference operation."""
        self.total_inferences += 1
        self.total_texts_processed += batch_size
        self.total_latency_ms += latency_ms
        self._latencies.append(latency_ms)

    def set_baseline_latency_ms(self, latency_ms: float) -> None:
        """Set FP32 baseline latency for speedup calculation."""
        self.baseline_latency_ms = latency_ms

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if self.total_inferences == 0:
            return 0.0
        return self.total_latency_ms / self.total_inferences

    @property
    def throughput_texts_per_second(self) -> float:
        """Throughput in texts per second."""
        if self.total_latency_ms == 0:
            return 0.0
        return (self.total_texts_processed / self.total_latency_ms) * 1000

    @property
    def speedup_ratio(self) -> float:
        """Speedup ratio vs FP32 baseline."""
        if self.baseline_latency_ms is None or self.avg_latency_ms == 0:
            return 1.0
        return self.baseline_latency_ms / self.avg_latency_ms

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_inferences = 0
        self.total_texts_processed = 0
        self.total_latency_ms = 0.0
        self._latencies.clear()


# =============================================================================
# MOCK QUANTIZED PROVIDER (for testing)
# =============================================================================


class MockQuantizedProvider(BaseEmbeddingProvider):
    """
    Mock quantized embedding provider -- FOR TESTS AND LATENCY BENCHMARKS ONLY.

    !! THE VECTORS THIS RETURNS ARE RANDOM NOISE. !!

    Embeddings are drawn from `np.random.RandomState(hash(text))`. They are
    deterministic per text, correctly shaped, and completely meaningless: the
    cosine similarity between any two of them is noise around zero regardless of
    how related the texts are. Use this class ONLY to exercise batching, timing
    and plumbing (e.g. benchmarks/suite_002_quantization.py, which measures
    throughput and does not look at the vectors). Never use it to measure
    retrieval accuracy, and never wire it into an application path.

    For real quantized embeddings use QuantizedEmbeddingProvider.
    """

    def __init__(
        self,
        dimension: int = 768,
        simulated_speedup: float = 3.0,
        simulated_accuracy_loss: float = 0.015,
        simulated_latency_per_text_us: float = 100.0,
    ) -> None:
        """
        Initialize mock quantized provider.

        Args:
            dimension: Embedding dimension
            simulated_speedup: Simulated speedup vs FP32
            simulated_accuracy_loss: Simulated accuracy loss (0.0-1.0)
            simulated_latency_per_text_us: Simulated latency per text (microseconds)
        """
        self._dimension = dimension
        self._simulated_speedup = simulated_speedup
        self._simulated_accuracy_loss = simulated_accuracy_loss
        self._simulated_latency_per_text_us = simulated_latency_per_text_us
        self._last_inference_time_ms: float | None = None
        self._stats = QuantizationStats()

    @property
    def model_name(self) -> str:
        """Get model name."""
        return "mock-quantized-int8"

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def is_quantized(self) -> bool:
        """Check if model is quantized."""
        return True

    @property
    def quantization_precision(self) -> str:
        """Get quantization precision."""
        return "int8"

    @property
    def produces_real_embeddings(self) -> bool:
        """False -- this provider returns random noise. See class docstring."""
        return False

    @property
    def simulated_speedup(self) -> float:
        """Get simulated speedup ratio."""
        return self._simulated_speedup

    @property
    def simulated_accuracy_loss(self) -> float:
        """Get simulated accuracy loss."""
        return self._simulated_accuracy_loss

    @property
    def last_inference_time_ms(self) -> float | None:
        """Get last inference time in milliseconds."""
        return self._last_inference_time_ms

    @property
    def stats(self) -> QuantizationStats:
        """Get quantization statistics."""
        return self._stats

    def _encode_batch(
        self,
        texts: Sequence[str],
        config: EmbeddingConfig,
    ) -> np.ndarray:
        """Generate MEANINGLESS mock embeddings with simulated latency."""
        start_time = time.perf_counter()

        embeddings = []
        for text in texts:
            # Stable across processes, unlike the builtin hash() (PYTHONHASHSEED).
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
            seed = int.from_bytes(digest, "big")
            rng = np.random.RandomState(seed)
            embedding = rng.randn(self._dimension).astype(np.float32)
            embeddings.append(embedding)

        # Simulate latency
        simulated_latency_s = len(texts) * self._simulated_latency_per_text_us / 1_000_000
        elapsed = time.perf_counter() - start_time
        if elapsed < simulated_latency_s:
            time.sleep(simulated_latency_s - elapsed)

        # Record stats
        actual_latency_ms = (time.perf_counter() - start_time) * 1000
        self._last_inference_time_ms = actual_latency_ms
        self._stats.record_inference(len(texts), actual_latency_ms)

        return np.vstack(embeddings)


# =============================================================================
# QUANTIZED EMBEDDING PROVIDER
# =============================================================================


class QuantizedEmbeddingProvider(BaseEmbeddingProvider):
    """
    INT8 quantized embedding provider for CPU speedup.

    Runs a real transformer encoder with INT8 dynamically-quantized weights and
    pools the token vectors into one sentence vector.

    Backends:
        "onnx"     Export the HF model to ONNX, apply
                   `onnxruntime.quantization.quantize_dynamic` (INT8 weights,
                   dynamically quantized activations), then run it under
                   onnxruntime. Requires `onnx`, `onnxruntime`, `torch` and
                   `transformers`. Artifacts are cached on disk and reused.
        "torch"    Apply `torch.ao.quantization.quantize_dynamic` to the HF
                   model's Linear layers (INT8, per-tensor dynamic). Requires
                   only `torch` and `transformers`.
        "openvino" NOT IMPLEMENTED -- raises NotImplementedError at load.
        "auto"     Prefer "onnx", fall back to "torch", raise if neither is
                   installed.

    Failure policy:
        Every failure path raises. If the backend is unavailable, the model
        cannot be loaded, or the precision is unimplemented, this class raises
        rather than degrading to something that returns vectors anyway. Callers
        going through `embed()` receive `Result.failure` with the message; the
        exception is never swallowed into a plausible-looking array.

    Accuracy caveat -- INT8 output depends on batch composition:
        Dynamic quantization derives activation quantization scales from the
        range observed in each batch. The same text therefore embeds slightly
        differently depending on which other texts share its batch. Measured on
        all-MiniLM-L6-v2: embedding "id" alone vs. batched with a long sentence
        gives cosine ~0.88, where the fp32 model gives exactly 1.0. Against the
        FP32 reference, INT8 vectors land at cosine ~0.92-0.99
        (BAAI/bge-small-en-v1.5).

        Practical consequences:
        - INT8 embeddings are not bit-reproducible across differently-composed
          batches. Content-hash caches (see caches/content.py) will return a
          vector produced under a different batch shape; that is acceptable for
          retrieval but not for anything requiring exact reproducibility.
        - Embed the corpus and the queries with the same backend and precision.
          Mixing an FP32-indexed corpus with INT8 queries costs accuracy.
        - Use precision="fp32" when exact reproducibility matters.

    Example:
        config = QuantizationConfig(precision="int8", backend="torch")
        provider = QuantizedEmbeddingProvider(
            model_name="BAAI/bge-small-en-v1.5",
            quantization_config=config,
        )
        result = provider.embed(["customer email", "transaction amount"])
    """

    # Model families trained with CLS pooling rather than mean pooling.
    _CLS_POOLING_HINTS = ("bge", "gte", "e5-", "-e5")

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        quantization_config: QuantizationConfig | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize quantized embedding provider.

        Model loading is lazy: construction never touches the network or the
        filesystem, so an unusable model name surfaces on first `embed()` call
        rather than at wiring time.

        Args:
            model_name: HuggingFace model name or local path
            quantization_config: Quantization configuration
            cache_dir: Directory for model + ONNX artifact cache
        """
        self._model_name = model_name
        self._quantization_config = quantization_config or QuantizationConfig()
        self._cache_dir = cache_dir

        # Lazy-loaded state
        self._model = None  # torch backend: the quantized nn.Module
        self._tokenizer = None
        self._session = None  # onnx backend: ort.InferenceSession
        self._active_backend: str | None = None
        self._onnx_input_names: tuple[str, ...] = ()
        self._dimension: int | None = None
        self._dimension_is_authoritative = False

        # Statistics
        self._stats = QuantizationStats()
        self._last_inference_time_ms: float | None = None

        # CPU features
        self._cpu_features = detect_cpu_features()

    # -- introspection --------------------------------------------------

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """
        Embedding dimension.

        Before the model is loaded this is a NAME-BASED ESTIMATE (1024 for
        "large", 384 for "small"/"mini", else 768) so that wiring code can size
        buffers without forcing a model download. Once the model is loaded the
        value is replaced with the encoder's real hidden size, and
        `dimension_is_authoritative` flips to True. `_encode_batch` asserts the
        vectors it produces match, so a wrong estimate can never propagate.
        """
        if self._dimension is None:
            model_lower = self._model_name.lower()
            if "large" in model_lower:
                self._dimension = 1024
            elif "small" in model_lower or "mini" in model_lower:
                self._dimension = 384
            else:
                self._dimension = 768
        return self._dimension

    @property
    def dimension_is_authoritative(self) -> bool:
        """True once `dimension` reflects the loaded model rather than a guess."""
        return self._dimension_is_authoritative

    @property
    def is_quantized(self) -> bool:
        """Check if model is quantized."""
        return self._quantization_config.precision in ("int8", "int4")

    @property
    def quantization_precision(self) -> str:
        """Get quantization precision."""
        return self._quantization_config.precision

    @property
    def produces_real_embeddings(self) -> bool:
        """True -- this provider runs a real model or raises."""
        return True

    @property
    def active_backend(self) -> str | None:
        """Backend actually in use, or None before the model is loaded."""
        return self._active_backend

    @property
    def stats(self) -> QuantizationStats:
        """Get quantization statistics."""
        return self._stats

    @property
    def last_inference_time_ms(self) -> float | None:
        """Get last inference time in milliseconds."""
        return self._last_inference_time_ms

    @property
    def cpu_features(self) -> dict[str, bool]:
        """Get detected CPU features."""
        return self._cpu_features

    # -- loading --------------------------------------------------------

    def _resolve_backend(self) -> str:
        """Pick the backend to use, raising if none is usable."""
        backend = self._quantization_config.backend

        if backend != "auto":
            return backend

        if is_backend_available("onnx"):
            return "onnx"
        if is_backend_available("torch"):
            return "torch"

        raise ImportError(
            "No quantization backend is available. Install one of:\n"
            "  pip install onnx onnxruntime transformers torch   (ONNX INT8)\n"
            "  pip install torch transformers                    (torch INT8)\n"
            "This provider will not fall back to synthetic embeddings."
        )

    def _load_model(self) -> None:
        """Lazy load and quantize the model. Raises on any failure."""
        if self._session is not None or self._model is not None:
            return

        precision = self._quantization_config.precision
        if precision in ("int4", "fp16"):
            raise NotImplementedError(
                f"precision={precision!r} is accepted by QuantizationConfig but no "
                f"{precision} inference path is implemented in this module. Use "
                "'int8' (dynamic quantization) or 'fp32' (unquantized). Refusing "
                "to silently run a different precision than the one requested."
            )

        backend = self._resolve_backend()

        if backend == "onnx":
            self._load_onnx_model()
        elif backend == "torch":
            self._load_torch_model()
        elif backend == "openvino":
            self._load_openvino_model()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._active_backend = backend

    def _load_tokenizer(self):
        """Load the HF tokenizer, raising a clear error if unavailable."""
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"transformers is required for quantized inference: {e}. "
                "Install with: pip install transformers"
            ) from e

        return AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
        )

    def _artifact_dir(self) -> Path:
        """Directory for cached ONNX artifacts."""
        base = (
            Path(self._cache_dir) if self._cache_dir else Path.home() / ".cache" / "nexus_matcher"
        )
        safe = self._model_name.replace("/", "__").replace("\\", "__")
        target = base / "onnx" / safe
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _load_onnx_model(self) -> None:
        """
        Export the encoder to ONNX, quantize it to INT8, and open a session.

        The FP32 export and the INT8 graph are both cached on disk; subsequent
        constructions reuse them instead of re-exporting.
        """
        if not is_backend_available("onnx"):
            raise ImportError(
                "The ONNX INT8 backend needs onnx, onnxruntime, torch and "
                "transformers. `onnxruntime` on its own can run graphs but cannot "
                "export or quantize one, so it is not sufficient. Install with:\n"
                "  pip install onnx onnxruntime torch transformers\n"
                "Alternatively use QuantizationConfig(backend='torch'), which needs "
                "only torch + transformers."
            )

        import onnxruntime as ort
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from transformers import AutoModel

        self._tokenizer = self._load_tokenizer()

        artifacts = self._artifact_dir()
        fp32_path = artifacts / "model_fp32.onnx"
        int8_path = artifacts / "model_int8.onnx"
        want_int8 = self._quantization_config.precision == "int8"
        target_path = int8_path if want_int8 else fp32_path

        if not target_path.exists():
            model = AutoModel.from_pretrained(
                self._model_name,
                cache_dir=self._cache_dir,
            )
            model.eval()

            self._dimension = int(model.config.hidden_size)
            self._dimension_is_authoritative = True

            input_names = self._model_input_names(model)
            dummy = self._tokenizer(
                ["nexus matcher onnx export probe"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16,
            )
            args = tuple(dummy[name] for name in input_names)

            dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
            dynamic_axes["last_hidden_state"] = {0: "batch", 1: "sequence"}

            if not fp32_path.exists():
                logger.info("Exporting %s to ONNX at %s", self._model_name, fp32_path)
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        args,
                        str(fp32_path),
                        input_names=list(input_names),
                        output_names=["last_hidden_state"],
                        dynamic_axes=dynamic_axes,
                        opset_version=14,
                        do_constant_folding=True,
                        dynamo=False,
                    )

            if want_int8:
                logger.info("Quantizing %s to INT8 at %s", fp32_path, int8_path)
                quantize_dynamic(
                    model_input=str(fp32_path),
                    model_output=str(int8_path),
                    weight_type=QuantType.QInt8,
                )

        sess_options = ort.SessionOptions()
        if self._quantization_config.num_threads > 0:
            sess_options.intra_op_num_threads = self._quantization_config.num_threads
        if self._quantization_config.optimize_for_inference:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(target_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._onnx_input_names = tuple(i.name for i in self._session.get_inputs())

        if not self._dimension_is_authoritative:
            out_shape = self._session.get_outputs()[0].shape
            if isinstance(out_shape[-1], int):
                self._dimension = int(out_shape[-1])
                self._dimension_is_authoritative = True

    def _load_torch_model(self) -> None:
        """Load the encoder and apply torch INT8 dynamic quantization."""
        if not is_backend_available("torch"):
            raise ImportError(
                "The torch INT8 backend needs torch and transformers. "
                "Install with: pip install torch transformers"
            )

        import torch
        from torch import nn
        from torch.ao.quantization import quantize_dynamic as torch_quantize_dynamic
        from transformers import AutoModel

        if self._quantization_config.num_threads > 0:
            torch.set_num_threads(self._quantization_config.num_threads)

        self._tokenizer = self._load_tokenizer()

        model = AutoModel.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
        )
        model.eval()

        self._dimension = int(model.config.hidden_size)
        self._dimension_is_authoritative = True

        if self._quantization_config.precision == "int8":
            # Dynamic quantization: INT8 weights, activations quantized on the
            # fly per batch. This is the standard CPU recipe for transformer
            # encoders and needs no calibration data.
            model = torch_quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        self._model = model
        self._onnx_input_names = self._model_input_names(model)

    def _load_openvino_model(self) -> None:
        """
        Not implemented.

        The previous version of this method loaded a tokenizer, left a
        `# TODO: Implement actual OpenVINO model loading` comment, assigned
        `self._session = None`, and returned successfully -- so callers believed
        an OpenVINO model was loaded when nothing had been. Raising is the
        honest behaviour.
        """
        raise NotImplementedError(
            "The OpenVINO backend is not implemented in this module. No OpenVINO "
            "export, quantization or inference path exists here. Use "
            "QuantizationConfig(backend='onnx') or backend='torch'."
        )

    @staticmethod
    def _model_input_names(model: Any) -> tuple[str, ...]:
        """
        Determine which tokenizer outputs the encoder's forward() accepts.

        BERT-family models take token_type_ids; DistilBERT and friends do not,
        and passing an unexpected kwarg is a hard error.
        """
        candidates = ("input_ids", "attention_mask", "token_type_ids")
        try:
            params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return ("input_ids", "attention_mask")

        return tuple(name for name in candidates if name in params)

    def _pooling_mode(self) -> str:
        """Resolve the configured pooling mode for this model."""
        configured = self._quantization_config.pooling
        if configured != "auto":
            return configured

        lowered = self._model_name.lower()
        if any(hint in lowered for hint in self._CLS_POOLING_HINTS):
            return "cls"
        return "mean"

    # -- inference ------------------------------------------------------

    def _encode_batch(
        self,
        texts: Sequence[str],
        config: EmbeddingConfig,
    ) -> np.ndarray:
        """
        Encode a batch of texts using real quantized inference.

        Args:
            texts: Texts to encode
            config: Encoding configuration

        Returns:
            2D float32 array of shape (len(texts), dimension)

        Raises:
            ImportError / NotImplementedError / RuntimeError: If the backend is
                unavailable or inference fails. This method NEVER returns
                synthetic vectors -- a caller that gets an array back is
                guaranteed it came out of the model.
        """
        start_time = time.perf_counter()

        self._load_model()

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="np" if self._session is not None else "pt",
        )

        if self._session is not None:
            token_vectors, attention_mask = self._run_onnx(encoded)
        elif self._model is not None:
            token_vectors, attention_mask = self._run_torch(encoded)
        else:
            raise RuntimeError(
                "Quantized model failed to load and no inference session is "
                "available. Refusing to return synthetic embeddings."
            )

        embeddings = self._pool(token_vectors, attention_mask)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"Quantized inference returned shape {embeddings.shape} for "
                f"{len(texts)} texts; expected ({len(texts)}, {self.dimension})."
            )

        if self._dimension_is_authoritative and embeddings.shape[1] != self._dimension:
            raise RuntimeError(
                f"Quantized inference returned dimension {embeddings.shape[1]} but "
                f"the loaded model reports {self._dimension}."
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._last_inference_time_ms = elapsed_ms
        self._stats.record_inference(len(texts), elapsed_ms)

        return embeddings.astype(np.float32)

    def _run_onnx(self, encoded: Any) -> tuple[np.ndarray, np.ndarray]:
        """Run the INT8 ONNX session and return (token_vectors, attention_mask)."""
        feeds = {}
        for name in self._onnx_input_names:
            if name not in encoded:
                raise RuntimeError(
                    f"ONNX graph expects input {name!r} but the tokenizer did not "
                    f"produce it (got {sorted(encoded.keys())})."
                )
            feeds[name] = np.asarray(encoded[name], dtype=np.int64)

        outputs = self._session.run(None, feeds)
        token_vectors = np.asarray(outputs[0], dtype=np.float32)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.float32)
        return token_vectors, attention_mask

    def _run_torch(self, encoded: Any) -> tuple[np.ndarray, np.ndarray]:
        """Run the INT8 torch module and return (token_vectors, attention_mask)."""
        import torch

        inputs = {name: encoded[name] for name in self._onnx_input_names if name in encoded}
        if "input_ids" not in inputs:
            raise RuntimeError(f"Tokenizer produced no input_ids (got {sorted(encoded.keys())}).")

        with torch.no_grad():
            outputs = self._model(**inputs)

        token_vectors = outputs.last_hidden_state.numpy().astype(np.float32)
        attention_mask = encoded["attention_mask"].numpy().astype(np.float32)
        return token_vectors, attention_mask

    def _pool(self, token_vectors: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Reduce (batch, seq, hidden) token vectors to (batch, hidden).

        Mean pooling excludes padding positions via the attention mask; pooling
        over padding is a classic silent-accuracy bug because it still yields a
        well-shaped, plausible vector.
        """
        mode = self._pooling_mode()

        if mode == "cls":
            return token_vectors[:, 0, :]

        mask = attention_mask[..., None]
        summed = (token_vectors * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_quantized_provider(
    model_name: str = "BAAI/bge-base-en-v1.5",
    precision: str = "int8",
    backend: str = "auto",
) -> QuantizedEmbeddingProvider:
    """
    Create a quantized embedding provider with sensible defaults.

    Args:
        model_name: HuggingFace model name
        precision: Quantization precision ("int8", "int4", "fp16", "fp32")
        backend: Backend ("onnx", "torch", "openvino", "auto")

    Returns:
        Configured QuantizedEmbeddingProvider
    """
    config = QuantizationConfig(
        precision=precision,
        backend=backend,
        optimize_for_inference=True,
    )

    return QuantizedEmbeddingProvider(
        model_name=model_name,
        quantization_config=config,
    )


def get_quantization_info() -> dict:
    """
    Get information about quantization support on this system.

    Returns:
        Dictionary with CPU features, backend availability, and recommendations
    """
    cpu_features = detect_cpu_features()

    return {
        "cpu_features": cpu_features,
        "vnni_available": cpu_features.get("vnni", False),
        "avx512_available": cpu_features.get("avx512", False),
        "avx2_available": cpu_features.get("avx2", False),
        "onnx_available": is_backend_available("onnx"),
        "torch_available": is_backend_available("torch"),
        "openvino_available": is_backend_available("openvino"),
        "quantization_recommended": is_quantization_recommended(),
        "expected_speedup": "3-10x" if cpu_features.get("vnni") else "2-4x",
    }
