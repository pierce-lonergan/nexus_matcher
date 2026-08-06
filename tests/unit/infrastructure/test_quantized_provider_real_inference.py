"""
tests.unit.infrastructure.test_quantized_provider_real_inference | Layer: TEST
Regression guards: QuantizedEmbeddingProvider must never fabricate embeddings.

The provider used to implement `_encode_batch` as:

    for text in texts:
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype(np.float32)

behind a `# TODO: Implement actual quantized inference` comment. Every caller
received correctly-shaped, deterministic, normalized float32 vectors that were
RANDOM NOISE. Nothing failed. Nothing warned. Retrieval built on top of it
returned confident, meaningless rankings.

The guards below pin the two properties that defect violated:
  1. No model loaded => embed() FAILS. It does not return vectors.
  2. With a model loaded, the vectors carry real semantics.

Test (2) is the definitive one -- random embeddings cannot pass it -- but it
needs a real model, so it is marked slow and skips when the model is not in the
local HuggingFace cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.infrastructure.adapters.embedding_providers.quantized import (
    MockQuantizedProvider,
    QuantizationConfig,
    QuantizedEmbeddingProvider,
    get_quantization_info,
    is_backend_available,
)

MODEL = "BAAI/bge-small-en-v1.5"


def _model_is_cached(name: str = MODEL) -> bool:
    """True if the model can be loaded without network access."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return False
    try:
        AutoConfig.from_pretrained(name, local_files_only=True)
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _model_is_cached(),
    reason=f"{MODEL} is not in the local HuggingFace cache",
)


# =============================================================================
# THE CORE GUARD: NO MODEL => NO VECTORS
# =============================================================================


class TestNeverFabricatesEmbeddings:
    """A provider that cannot load a model must fail, not invent numbers."""

    def test_unloadable_model_fails_instead_of_returning_vectors(self, monkeypatch):
        """
        This is the guard for the original defect.

        Against the old implementation this test FAILS: embed() returned
        Result.success with two random 768-dim vectors for a model name that
        does not exist anywhere.
        """
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        provider = QuantizedEmbeddingProvider(
            model_name="nexus-matcher-nonexistent-model-please-fail",
            quantization_config=QuantizationConfig(backend="torch"),
        )

        result = provider.embed(["customer email", "transaction amount"])

        assert result.is_failure, (
            "embed() succeeded for a model that cannot be loaded -- "
            "the provider fabricated embeddings"
        )

    def test_encode_batch_raises_for_unloadable_model(self, monkeypatch):
        """The internal path raises rather than returning an array."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        from nexus_matcher.domain.ports.embedding_provider import EmbeddingConfig

        provider = QuantizedEmbeddingProvider(
            model_name="nexus-matcher-nonexistent-model-please-fail",
            quantization_config=QuantizationConfig(backend="torch"),
        )

        with pytest.raises(Exception) as exc_info:
            provider._encode_batch(["anything"], EmbeddingConfig())

        # Must not be an assertion-free "returned an array" path.
        assert exc_info.value is not None

    def test_provider_declares_it_produces_real_embeddings(self):
        """The real provider and the mock are distinguishable programmatically."""
        real = QuantizedEmbeddingProvider(model_name=MODEL)
        mock = MockQuantizedProvider(dimension=384)

        assert real.produces_real_embeddings is True
        assert mock.produces_real_embeddings is False


# =============================================================================
# BACKEND HONESTY
# =============================================================================


class TestBackendAvailabilityIsHonest:
    """is_backend_available must reflect what can actually run."""

    def test_onnx_backend_requires_the_full_toolchain(self):
        """
        `onnxruntime` alone cannot export or quantize a model.

        The old check returned True as soon as `import onnxruntime` worked,
        which is how the code claimed an ONNX path it could not take.
        """
        try:
            import onnxruntime  # noqa: F401

            has_runtime = True
        except ImportError:
            has_runtime = False

        try:
            import onnx  # noqa: F401
            from onnxruntime.quantization import quantize_dynamic  # noqa: F401

            has_toolchain = True
        except ImportError:
            has_toolchain = False

        if has_runtime and not has_toolchain:
            assert is_backend_available("onnx") is False, (
                "onnx backend reported available with only onnxruntime installed"
            )

    def test_openvino_backend_reports_unavailable(self):
        """No OpenVINO path is implemented, so it must never report available."""
        assert is_backend_available("openvino") is False

    def test_openvino_backend_raises_not_implemented(self):
        """
        Loading the OpenVINO backend must raise.

        The old `_load_openvino_model` set `self._session = None` and returned
        successfully, so callers believed a model was loaded.
        """
        provider = QuantizedEmbeddingProvider(
            model_name=MODEL,
            quantization_config=QuantizationConfig(backend="openvino"),
        )

        with pytest.raises(NotImplementedError, match="OpenVINO"):
            provider._load_model()

    def test_unimplemented_precision_raises(self):
        """int4/fp16 are accepted by config but have no implementation."""
        for precision in ("int4", "fp16"):
            provider = QuantizedEmbeddingProvider(
                model_name=MODEL,
                quantization_config=QuantizationConfig(precision=precision, backend="torch"),
            )
            with pytest.raises(NotImplementedError, match=precision):
                provider._load_model()

    def test_quantization_info_reports_torch_backend(self):
        info = get_quantization_info()
        assert "torch_available" in info
        assert isinstance(info["torch_available"], bool)


# =============================================================================
# REAL SEMANTICS (requires a cached model)
# =============================================================================


@requires_model
@pytest.mark.slow
class TestRealQuantizedEmbeddings:
    """With a model loaded, the vectors must carry real meaning."""

    @pytest.fixture(scope="class")
    def embeddings(self):
        provider = QuantizedEmbeddingProvider(
            model_name=MODEL,
            quantization_config=QuantizationConfig(precision="int8", backend="torch"),
        )
        texts = [
            "customer email address",
            "the e-mail of the client",
            "total transaction amount in dollars",
        ]
        result = provider.embed(texts)
        assert result.is_success, result.error
        return provider, result.unwrap().as_array()

    def test_related_texts_score_higher_than_unrelated(self, embeddings):
        """
        THE definitive guard against random embeddings.

        Random vectors give cosine ~0 for every pair, so a paraphrase scores no
        better than an unrelated field. Real embeddings must separate them.
        """
        _, e = embeddings
        related = float(e[0] @ e[1])  # email vs e-mail
        unrelated = float(e[0] @ e[2])  # email vs transaction amount

        assert related > unrelated, (
            f"paraphrase similarity {related:.4f} did not beat unrelated "
            f"{unrelated:.4f} -- embeddings carry no semantics"
        )
        assert related > 0.6, (
            f"paraphrase similarity {related:.4f} is near-orthogonal; "
            "these look like random vectors"
        )

    def test_dimension_becomes_authoritative_after_load(self, embeddings):
        provider, e = embeddings
        assert provider.dimension_is_authoritative is True
        assert provider.dimension == 384  # bge-small hidden size
        assert e.shape == (3, 384)
        assert provider.active_backend == "torch"

    def test_identical_text_gives_identical_vector(self, embeddings):
        provider, _ = embeddings
        a = provider.embed(["postal code"]).unwrap().as_array()
        b = provider.embed(["postal code"]).unwrap().as_array()
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    def test_matches_fp32_reference_within_quantization_error(self):
        """
        INT8 output must track the FP32 model.

        Real dynamic quantization costs a little accuracy (cosine ~0.92-0.99
        here). Random noise would score ~0.0 against the reference, so this
        bounds the answer from both sides.
        """
        pytest.importorskip("sentence_transformers")
        from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
            SentenceTransformersProvider,
        )

        texts = [
            "customer email address",
            "total transaction amount",
            "shipping postal code",
        ]

        quantized = QuantizedEmbeddingProvider(
            model_name=MODEL,
            quantization_config=QuantizationConfig(precision="int8", backend="torch"),
        )
        qe = quantized.embed(texts).unwrap().as_array()

        reference = SentenceTransformersProvider(MODEL, device="cpu")
        re_ = reference.embed(texts).unwrap().as_array()

        cosines = (qe * re_).sum(axis=1)
        assert cosines.min() > 0.85, f"INT8 embeddings diverge from FP32 reference: {cosines}"

    def test_mean_pooling_ignores_padding(self):
        """
        A short text must embed identically whether or not it shares a batch
        with a long one. Pooling over padding is a classic silent-accuracy bug:
        it still returns a perfectly plausible vector.

        Measured at fp32 so the assertion isolates POOLING. At int8 the same
        comparison lands around 0.877 -- see
        test_int8_is_batch_composition_sensitive, which pins that as
        quantization behaviour rather than a pooling defect.
        """
        model = "sentence-transformers/all-MiniLM-L6-v2"
        if not _model_is_cached(model):
            pytest.skip(f"{model} not cached")

        provider = QuantizedEmbeddingProvider(
            model_name=model,
            quantization_config=QuantizationConfig(precision="fp32", backend="torch"),
        )

        assert provider._pooling_mode() == "mean"

        alone = provider.embed(["id"]).unwrap().as_array()
        padded = (
            provider.embed(["id", "a considerably longer field description that forces padding"])
            .unwrap()
            .as_array()[0:1]
        )

        cosine = float((alone * padded).sum())
        assert cosine > 0.9999, (
            f"short text embedding changed when batched with a long one "
            f"(cosine {cosine:.6f}) -- padding is leaking into the pooled vector"
        )

    def test_int8_is_batch_composition_sensitive(self):
        """
        Documented caveat, pinned as a test.

        torch dynamic quantization computes activation quantization scales from
        the observed range of each batch. Batching a short text with a long one
        changes that range, so the short text's vector shifts (measured cosine
        ~0.88 on all-MiniLM-L6-v2 vs exactly 1.0 at fp32).

        Consequences for callers:
          - INT8 embeddings are NOT bit-reproducible across differently-composed
            batches, so caching one under a content hash and reusing it for a
            differently-batched call gives a slightly different vector.
          - Index and query embeddings should be produced with the same backend
            and comparable batch shapes.

        This test exists so the behaviour cannot silently change or be mistaken
        for the pooling bug above.
        """
        model = "sentence-transformers/all-MiniLM-L6-v2"
        if not _model_is_cached(model):
            pytest.skip(f"{model} not cached")

        pair = ["id", "a considerably longer field description that forces padding"]

        fp32 = QuantizedEmbeddingProvider(
            model_name=model,
            quantization_config=QuantizationConfig(precision="fp32", backend="torch"),
        )
        int8 = QuantizedEmbeddingProvider(
            model_name=model,
            quantization_config=QuantizationConfig(precision="int8", backend="torch"),
        )

        fp32_drift = float(
            (
                fp32.embed(["id"]).unwrap().as_array() * fp32.embed(pair).unwrap().as_array()[0:1]
            ).sum()
        )
        int8_drift = float(
            (
                int8.embed(["id"]).unwrap().as_array() * int8.embed(pair).unwrap().as_array()[0:1]
            ).sum()
        )

        # fp32 is exact; int8 drifts but stays clearly semantic (not noise).
        assert fp32_drift > 0.9999
        assert 0.7 < int8_drift < 0.9999


# =============================================================================
# POOLING SELECTION
# =============================================================================


class TestPoolingSelection:
    """Pooling must match how the model was trained."""

    def test_bge_family_uses_cls_pooling(self):
        provider = QuantizedEmbeddingProvider(model_name="BAAI/bge-base-en-v1.5")
        assert provider._pooling_mode() == "cls"

    def test_other_models_use_mean_pooling(self):
        provider = QuantizedEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert provider._pooling_mode() == "mean"

    def test_pooling_can_be_overridden(self):
        provider = QuantizedEmbeddingProvider(
            model_name="BAAI/bge-base-en-v1.5",
            quantization_config=QuantizationConfig(pooling="mean"),
        )
        assert provider._pooling_mode() == "mean"

    def test_invalid_pooling_rejected(self):
        with pytest.raises(ValueError, match="pooling"):
            QuantizationConfig(pooling="magic")
