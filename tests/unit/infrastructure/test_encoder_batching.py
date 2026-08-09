"""
tests.unit.infrastructure.test_encoder_batching | Layer: TEST
Length-sorted batching must not disturb which vector belongs to which text.

The encoder now sorts texts by length before batching, because the tokenizer pads each
batch to its own longest member and attention is quadratic in that padded length. On the
FHIR corpus the padded token count was 3.83x the real token count in natural order and
1.12x in length order -- 71% of the encoder's work was padding, and removing it measured
a 2.30x speedup.

The danger is not slowness, it is MISALIGNMENT: encode in sorted order and scatter the
results back wrongly, and every text silently gets some other text's vector. Nothing
raises; matching just quietly returns nonsense. Hence this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
    bundled_model_available,
)

pytestmark = pytest.mark.skipif(
    not bundled_model_available(), reason="bundled ONNX model not present"
)

# Deliberately wide length spread, so sorting genuinely reorders them.
TEXTS = [
    "customer email address",
    "the total monetary amount of an order including tax and shipping charges "
    "applied at checkout time by the billing subsystem",
    "id",
    "shipping street line one",
    "birth date",
    "a record describing the postal code portion of a delivery address " * 4,
    "status",
]


@pytest.fixture(scope="module")
def provider():
    from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
        BundledOnnxProvider,
    )

    p = BundledOnnxProvider()
    p.embed_documents(["warmup"])
    return p


class TestOrderPreservation:
    def test_shuffling_the_input_permutes_the_output_identically(self, provider):
        """
        The strongest alignment check available: if row i of the output is genuinely
        texts[i]'s vector, then encoding a permutation must give exactly the permuted
        output. Sensitive to any scatter-back error, and immune to int8 batch jitter
        only insofar as the batches happen to match -- so it is asserted exactly.
        """
        base = provider.embed_documents(TEXTS, batch_size=3)
        perm = np.array([4, 0, 6, 2, 5, 1, 3])
        shuffled = provider.embed_documents([TEXTS[i] for i in perm], batch_size=3)
        assert np.abs(shuffled - base[perm]).max() < 1e-6

    def test_each_row_matches_that_text_encoded_alone(self, provider):
        """
        Guards the failure that order-invariance alone would miss: a scatter that is
        self-consistent but assigns every text the wrong neighbour's vector.
        """
        batched = provider.embed_documents(TEXTS, batch_size=3)
        for i, text in enumerate(TEXTS):
            single = provider.embed_documents([text], batch_size=3)[0]
            # int8 inference is not batch-invariant, so require high cosine rather than
            # equality -- but a misalignment would score near zero, not near one.
            assert float(batched[i] @ single) > 0.95, f"row {i} is not {text[:30]!r}"

    def test_a_misaligned_row_would_actually_fail_that_check(self, provider):
        """Proves the previous test has teeth rather than passing vacuously."""
        batched = provider.embed_documents(TEXTS, batch_size=3)
        single_first = provider.embed_documents([TEXTS[0]], batch_size=3)[0]
        assert float(batched[2] @ single_first) < 0.95

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 64])
    def test_alignment_holds_at_every_batch_size(self, provider, batch_size):
        ref = provider.embed_documents(TEXTS, batch_size=1)
        got = provider.embed_documents(TEXTS, batch_size=batch_size)
        cos = (ref * got).sum(axis=1)
        assert cos.min() > 0.95

    def test_batch_boundary_exactly_divides_the_input(self, provider):
        """len(texts) % batch_size == 0 is where an off-by-one slice hides."""
        got = provider.embed_documents(TEXTS[:6], batch_size=3)
        assert got.shape == (6, provider.dimension)
        for i in range(6):
            single = provider.embed_documents([TEXTS[i]], batch_size=3)[0]
            assert float(got[i] @ single) > 0.95


class TestOutputShape:
    def test_empty_input(self, provider):
        assert provider.embed_documents([]).shape == (0, provider.dimension)

    def test_vectors_are_unit_normalised(self, provider):
        v = provider.embed_documents(TEXTS)
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    def test_duplicate_texts_get_near_identical_vectors(self, provider):
        """
        Duplicates land in the same place semantically, but NOT bit-identically: two
        copies of a text can fall in different batches, and int8 inference is not
        batch-invariant, so their vectors differ slightly (measured cosine 0.9944).
        That is a property of the quantised encoder, not of the length sort -- it was
        equally true before, whenever a duplicate straddled a batch boundary. Asserted
        as "clearly the same text" rather than "identical", which would be false.
        """
        v = provider.embed_documents(["same text", "other", "same text"], batch_size=2)
        assert float(v[0] @ v[2]) > 0.99
        assert float(v[0] @ v[1]) < float(v[0] @ v[2])

    def test_queries_are_prefixed_and_documents_are_not(self, provider):
        """
        BGE is asymmetric. If the sort refactor had dropped the instruction prefix the
        vectors would still look fine -- only accuracy would fall.
        """
        text = "customer email address"
        assert (
            float(provider.embed_queries([text])[0] @ provider.embed_documents([text])[0]) < 0.9999
        )

    def test_very_long_text_is_truncated_not_rejected(self, provider):
        v = provider.embed_documents(["word " * 5000])
        assert v.shape == (1, provider.dimension)
        assert np.isfinite(v).all()
