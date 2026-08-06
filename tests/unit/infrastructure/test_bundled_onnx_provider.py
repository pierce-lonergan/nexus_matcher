"""
tests.unit.infrastructure.test_bundled_onnx_provider | Layer: TEST
Guards for the zero-setup encoder shipped inside the wheel.

Two of these tests exist because the corresponding mistakes were actually made:

  * POOLING. The exported ONNX directory carries no 1_Pooling config, so anything that
    loads it through sentence-transformers silently defaults to MEAN while the model was
    trained with CLS. That produced a benchmark comparison off by a full point and no
    error anywhere. `test_uses_cls_pooling` pins it.

  * TORCH. The entire value of this provider is that it does not need torch. Nothing in
    the import graph fails if torch sneaks back in -- the wheel just quietly regains an
    800 MB dependency. `test_does_not_import_torch` pins that too.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
    EMBEDDING_DIM,
    QUERY_INSTRUCTION,
    BundledOnnxProvider,
    bundled_model_available,
    default_embedding_provider,
)


def _runtime_available() -> bool:
    """Weights AND the runtime that loads them. Checking only the file made CI ERROR
    rather than skip when onnxruntime was missing, which hid the real problem (the
    runtime was never declared as a dependency) behind 20 identical tracebacks."""
    if not bundled_model_available():
        return False
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _runtime_available(),
    reason="bundled encoder unavailable (missing weights, onnxruntime or tokenizers)",
)


@pytest.fixture(scope="module")
def provider():
    return BundledOnnxProvider()


class TestBundledModelShipping:
    def test_weights_are_present(self):
        assert bundled_model_available() is True

    def test_notice_ships_with_the_weights(self):
        """MIT requires the licence notice travel with redistribution."""
        from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
            BUNDLED_MODEL_DIR,
        )

        notice = BUNDLED_MODEL_DIR / "NOTICE"
        assert notice.is_file(), "the bundled model has no NOTICE; shipping it violates MIT"
        text = notice.read_text(encoding="utf-8")
        assert "MIT" in text
        assert "bge-small-en-v1.5" in text

    def test_model_fits_under_the_pypi_file_limit(self):
        from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
            BUNDLED_MODEL_DIR,
        )

        size_mb = (BUNDLED_MODEL_DIR / "model_quantized.onnx").stat().st_size / 1e6
        assert size_mb < 100, f"model is {size_mb:.1f} MB; PyPI rejects files over 100 MB"


class TestEncoding:
    def test_dimension_and_shape(self, provider):
        vectors = provider.embed_documents(["customer account balance", "transaction date"])
        assert vectors.shape == (2, EMBEDDING_DIM)
        assert vectors.dtype == np.float32

    def test_vectors_are_l2_normalised(self, provider):
        vectors = provider.embed_documents(["a customer record", "an account balance"])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_input(self, provider):
        assert provider.embed_documents([]).shape == (0, EMBEDDING_DIM)

    def test_deterministic(self, provider):
        a = provider.embed_documents(["customer account balance"])
        b = provider.embed_documents(["customer account balance"])
        assert np.allclose(a, b)

    def test_semantically_close_beats_unrelated(self, provider):
        """A sanity floor: if this fails the encoder is not encoding anything useful."""
        vectors = provider.embed_documents(
            ["customer account balance", "account balance for a customer", "volcanic rock geology"]
        )
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        assert related > unrelated + 0.2, f"related={related:.3f} unrelated={unrelated:.3f}"

    def test_batching_is_close_but_not_bitwise_identical(self, provider):
        """
        int8 inference is NOT batch-invariant. ONNX Runtime picks different quantised GEMM
        kernels for different input shapes, so the same text encoded in a batch of 3 and a
        batch of 64 differs by ~0.005 cosine. This is not padding -- it reproduces with
        equal-length inputs.

        It matters more than it looks: measured end to end, P@1 on the labelled benchmark
        ranges 0.5276 (batch 8) to 0.5436 (batch 128), a 1.6-point swing from batch size
        alone. Any A/B comparison of encoders or configurations MUST hold batch size fixed
        or it is measuring kernel selection. Asserted as a similarity floor rather than
        equality so the test states the real contract.
        """
        texts = [f"customer field number {i}" for i in range(10)]
        small = provider.embed_documents(texts, batch_size=3)
        large = provider.embed_documents(texts, batch_size=64)
        cosines = [float(np.dot(a, b)) for a, b in zip(small, large, strict=True)]
        assert min(cosines) > 0.99, f"batch size changed embeddings too much: {min(cosines):.4f}"

    def test_long_text_is_truncated_not_crashed(self, provider):
        assert provider.embed_documents(["word " * 5000]).shape == (1, EMBEDDING_DIM)


class TestAsymmetricEncoding:
    """BGE applies its instruction to QUERIES only; symmetry here would cost accuracy."""

    def test_query_and_document_encoding_differ(self, provider):
        text = "customer account balance"
        assert not np.allclose(
            provider.embed_queries([text])[0], provider.embed_documents([text])[0], atol=1e-4
        )

    def test_query_encoding_applies_the_instruction(self, provider):
        text = "customer account balance"
        assert np.allclose(
            provider.embed_queries([text])[0],
            provider.embed_documents([QUERY_INSTRUCTION + text])[0],
            atol=1e-5,
        )

    def test_instruction_can_be_disabled(self):
        text = "customer account balance"
        plain = BundledOnnxProvider(query_instruction="")
        assert np.allclose(
            plain.embed_queries([text])[0], plain.embed_documents([text])[0], atol=1e-5
        )


class TestPooling:
    def test_uses_cls_pooling(self, provider):
        """
        BAAI/bge-small-en-v1.5 declares pooling_mode_cls_token=True. The exported ONNX
        directory has no pooling config, so any loader that infers one will pick MEAN and
        quietly change the embedding. Assert we take the CLS token.
        """
        provider._load()
        text = "customer account balance"
        encoded = provider._tokenizer.encode_batch([text])
        expected_inputs = {i.name for i in provider._session.get_inputs()}
        feeds = {
            "input_ids": np.array([encoded[0].ids], dtype=np.int64),
            "attention_mask": np.array([encoded[0].attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoded[0].type_ids], dtype=np.int64),
        }
        hidden = provider._session.run(
            None, {k: v for k, v in feeds.items() if k in expected_inputs}
        )[0]

        cls = hidden[0, 0, :]
        cls = cls / np.linalg.norm(cls)
        assert np.allclose(provider.embed_documents([text])[0], cls, atol=1e-5)


class TestProviderProtocol:
    def test_embed_returns_a_batch(self, provider):
        result = provider.embed(["a", "b"])
        assert result.is_success
        assert result.unwrap().embeddings.shape == (2, EMBEDDING_DIM)

    def test_embed_single(self, provider):
        result = provider.embed_single("customer id")
        assert result.is_success
        assert result.unwrap().shape == (EMBEDDING_DIM,)

    def test_reports_offline(self, provider):
        assert provider.is_offline is True

    def test_missing_model_gives_an_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"Reinstall nexus-matcher|model_dir"):
            BundledOnnxProvider(model_dir=tmp_path).embed_documents(["x"])


class TestProviderSelection:
    def test_auto_prefers_the_bundled_encoder(self):
        assert isinstance(default_embedding_provider(), BundledOnnxProvider)

    def test_explicit_bundled(self):
        assert isinstance(default_embedding_provider("bundled"), BundledOnnxProvider)

    def test_rejects_an_unknown_preference(self):
        with pytest.raises(ValueError, match="auto/bundled/transformer/static"):
            default_embedding_provider("magic")


class TestNoTorchDependency:
    def test_does_not_import_torch(self):
        """
        The point of this provider is a wheel that does not drag in torch. Run in a
        SUBPROCESS: this test session has torch loaded already, so an in-process check
        would pass regardless of what the provider does.
        """
        code = (
            "import sys;"
            "from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx "
            "import BundledOnnxProvider;"
            "p=BundledOnnxProvider();"
            "v=p.embed_documents(['customer account balance']);"
            "assert v.shape[1]==384;"
            "print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert "NO_TORCH" in out.stdout, "torch was imported by the bundled encoder path"
