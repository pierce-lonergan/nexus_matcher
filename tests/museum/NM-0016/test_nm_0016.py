"""
NM-0016 -- the encoder produced random numbers and every test still passed.

A provider returned `np.random.RandomState(hash(text)).randn(dim)`. The vectors had the
right shape, the right dtype and unit norm, so every assertion in the suite held. Matching
was pure noise.

The lesson generalises past this one provider: asserting a vector's SHAPE tests nothing
about whether it encodes meaning. These tests assert the two properties random numbers
cannot fake -- that the same text always encodes the same way, and that semantic distance
is ordered correctly.

`hash()` is salted per process, so the replayed provider is not even stable across runs;
the determinism check catches that, and the semantic checks catch the seeded case too.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
    BundledOnnxProvider,
    bundled_model_available,
)

pytestmark = pytest.mark.skipif(
    not bundled_model_available(), reason="bundled ONNX model not present"
)

NEAR = ("customer email address", "the email address of a customer")
FAR = ("customer email address", "quarterly seismic tolerance of the turbine housing")


@pytest.fixture(scope="module")
def provider():
    return BundledOnnxProvider()


def test_the_same_text_always_encodes_the_same_way(provider):
    """Random-per-call output fails here immediately."""
    a = provider.embed_documents(["customer email address"])[0]
    b = provider.embed_documents(["customer email address"])[0]
    assert np.allclose(a, b, atol=1e-6), "the encoder is not deterministic"


def test_semantically_close_text_scores_higher_than_unrelated_text(provider):
    """
    The property that survives a per-text seed. Seeded noise is deterministic but carries
    no meaning, so its similarity ordering is arbitrary.
    """
    anchor, near = provider.embed_documents(list(NEAR))
    _, far = provider.embed_documents(list(FAR))
    sim_near, sim_far = float(anchor @ near), float(anchor @ far)
    assert sim_near > sim_far + 0.15, (
        f"related text scored {sim_near:.3f} and unrelated text {sim_far:.3f} -- "
        f"the encoder is not encoding meaning"
    )


def test_paraphrases_are_not_merely_orthogonal(provider):
    """
    Random unit vectors in 384 dimensions are near-orthogonal: cosine clusters around 0.
    A real encoder puts a paraphrase far above that.
    """
    anchor, near = provider.embed_documents(list(NEAR))
    assert float(anchor @ near) > 0.5, "paraphrase similarity is at the noise floor"


def test_vectors_are_not_all_alike_either(provider):
    """Guards the opposite degenerate failure: a provider returning one constant vector."""
    vectors = provider.embed_documents(
        ["customer email address", "order total amount", "patient birth date"]
    )
    off_diagonal = (vectors @ vectors.T)[~np.eye(3, dtype=bool)]
    assert off_diagonal.max() < 0.99, "every text encodes to the same vector"
