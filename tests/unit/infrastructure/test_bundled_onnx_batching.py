"""
tests.unit.infrastructure.test_bundled_onnx_batching | Layer: TEST
Guards for the encoder's batching, which reorders rows and pads them by hand.

`BundledOnnxProvider._encode` no longer walks the caller's list in order. It tokenises
everything up front, sorts by true token length, groups rows under a token budget, pads
each group in numpy, and scatters the results back. Every one of those steps is a place
where a vector can end up attached to the wrong text -- and nothing downstream would
raise if it did. A mis-scattered glossary still returns 384 floats per entry, still
normalises, still ranks: it just hands back the wrong mapping, and this tool's output
decides whether a column inherits a PII classification.

So these tests pin the invariants that no exception would ever announce:
  * a row's vector does not depend on which rows it was batched with (padding + mask)
  * the output row order is the CALLER's order, not the sorted one
  * the batch planner emits every index exactly once
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
    EMBEDDING_DIM,
    MAX_BATCH_TOKENS,
    MAX_DEFAULT_THREADS,
    BundledOnnxProvider,
    bundled_model_available,
)


def _runtime_available() -> bool:
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


def _mixed_length_texts() -> list[str]:
    """Texts whose lengths vary enough that sorting MUST reorder them.

    Deliberately interleaved short/long, so a sort is guaranteed to move rows and any
    failure to scatter back shows up. The long entries also force more than one batch
    under the token budget.
    """
    texts: list[str] = []
    for i in range(24):
        if i % 3 == 0:
            texts.append(f"code{i}")
        elif i % 3 == 1:
            texts.append(f"customer account balance field number {i}")
        else:
            texts.append(f"entry {i}: " + ("the total monetary amount owed on the account " * 12))
    return texts


class TestBatchPlanner:
    """`_plan_batches` decides which rows share a session.run. Getting it wrong loses or
    duplicates rows, which mis-associates vectors with entries downstream."""

    @staticmethod
    def _plan(lengths, max_rows=512):
        order = np.argsort(lengths, kind="stable")
        return BundledOnnxProvider._plan_batches(order, lengths, max_rows)

    def test_every_index_appears_exactly_once(self):
        """A dropped index leaves an uninitialised output row; a duplicated one silently
        overwrites a different entry's vector. Neither raises."""
        lengths = [5, 400, 12, 380, 7, 64, 300, 9, 128, 33]
        flat = [i for batch in self._plan(lengths) for i in batch]
        assert sorted(flat) == list(range(len(lengths)))

    def test_no_batch_exceeds_the_token_budget(self):
        """The budget is what keeps each call cache-resident and bounds peak memory."""
        lengths = [5, 400, 12, 380, 7, 64, 300, 9, 128, 33, 511, 511, 511]
        for batch in self._plan(lengths):
            width = max(lengths[i] for i in batch)
            assert width * len(batch) <= MAX_BATCH_TOKENS or len(batch) == 1

    def test_a_single_oversized_text_still_gets_its_own_batch(self):
        """One text longer than the whole budget must not be dropped for not fitting."""
        lengths = [MAX_BATCH_TOKENS * 2]
        assert self._plan(lengths) == [[0]]

    def test_max_rows_is_honoured(self):
        """`batch_size` remains a hard row cap, so a caller pinning it for comparability
        against an older run still gets the batch size they asked for."""
        lengths = [4] * 100
        for batch in self._plan(lengths, max_rows=8):
            assert len(batch) <= 8

    def test_short_texts_batch_far_wider_than_long_ones(self):
        """The whole point of a token budget: batch size adapts to length instead of
        being fixed, which is where the 2.2x came from."""
        short = self._plan([4] * 600)
        long = self._plan([500] * 600)
        assert max(len(b) for b in short) > 10 * max(len(b) for b in long)


class TestRowIndependence:
    def test_a_vector_does_not_depend_on_its_batchmates(self, provider):
        """
        Encoding rows together must match encoding them alone.

        This is the test that catches a bad attention mask or an off-by-one in the manual
        numpy padding: pad tokens that are not masked out leak into the CLS vector, and
        the damage lands only on the SHORT rows that share a batch with a long one --
        exactly the rows a uniform-length fixture would never expose. Tolerance is a
        cosine floor rather than equality because int8 inference is not batch-invariant.
        """
        texts = _mixed_length_texts()
        together = provider.embed_documents(texts)
        alone = np.vstack([provider.embed_documents([t]) for t in texts])
        cosines = np.einsum("ij,ij->i", together, alone)
        worst = int(np.argmin(cosines))
        assert cosines.min() > 0.98, (
            f"row {worst} ({texts[worst][:40]!r}) changed by {1 - cosines.min():.4f} "
            f"depending on its batchmates; suspect the attention mask or the padding"
        )

    def test_output_order_is_the_callers_order(self, provider):
        """
        Rows are encoded in length order, so the scatter back is load-bearing.

        Asserted semantically, not just by shape: a shuffled output has the right shape
        and the right norms, and only a content check notices. Each probe is matched
        against its own single-text encoding.
        """
        texts = _mixed_length_texts()
        batched = provider.embed_documents(texts)
        for i in (0, 1, 2, 11, 23):
            solo = provider.embed_documents([texts[i]])[0]
            best = int(np.argmax(batched @ solo))
            assert best == i, f"text {i} came back at row {best}: the scatter is wrong"

    def test_duplicate_texts_get_the_same_vector(self, provider):
        """Duplicates land in different batch positions after sorting; they must still
        agree, or an identical glossary row would rank differently from its twin."""
        texts = ["customer account balance"] * 3 + ["x " * 300] + ["customer account balance"]
        v = provider.embed_documents(texts)
        for i in (1, 2, 4):
            assert float(v[0] @ v[i]) > 0.999

    def test_batch_size_cap_does_not_change_the_answer(self, provider):
        """A caller pinning batch_size for a comparable A/B must get the same vectors."""
        texts = _mixed_length_texts()
        wide = provider.embed_documents(texts, batch_size=512)
        narrow = provider.embed_documents(texts, batch_size=2)
        assert np.einsum("ij,ij->i", wide, narrow).min() > 0.98


class TestTokenizerContract:
    def test_token_type_ids_are_all_zero(self, provider):
        """
        `_encode` feeds `np.zeros_like(ids)` as token_type_ids instead of reading them
        back from the tokenizer. That is correct only while every input is a SINGLE
        sequence. If a future tokenizer.json ever emitted a second segment, the shortcut
        would encode the wrong thing silently -- no shape error, just worse vectors.
        """
        provider._load()
        encoded = provider._tokenizer.encode_batch(
            ["customer account balance", "a much longer definition of the same field " * 8]
        )
        for e in encoded:
            assert set(e.type_ids) == {0}

    def test_tokenizer_padding_is_off(self, provider):
        """
        Padding must stay OFF. `_encode` tokenises the whole input in one call so it can
        sort by true token length; with tokenizer padding on, that single call pads every
        text to the longest in the entire input -- on the FHIR corpus that is 417 tokens
        for every 5-token entry, i.e. the 3.83x padding blowup this design removes.
        """
        provider._load()
        encoded = provider._tokenizer.encode_batch(["short", "a considerably longer text here"])
        assert len(encoded[0].ids) != len(encoded[1].ids)


class TestSessionDefaults:
    def test_intra_op_threads_are_capped(self, provider):
        """
        ORT's default is one thread per physical core, which measured 1.52x SLOWER than 8
        on a 32-thread box: bge-small's GEMMs at these batch sizes cannot keep that many
        threads busy, so ORT pays fork/join per op for cores with nothing to do.
        """
        assert BundledOnnxProvider._default_threads() <= MAX_DEFAULT_THREADS
        assert BundledOnnxProvider._default_threads() >= 1

    def test_explicit_thread_count_is_respected(self):
        """Hosts running many workers per box need to pin this, or the workers
        oversubscribe the same cores."""
        p = BundledOnnxProvider(num_threads=2)
        assert p.embed_documents(["customer account balance"]).shape == (1, EMBEDDING_DIM)


class TestDegenerateInputs:
    def test_empty(self, provider):
        assert provider.embed_documents([]).shape == (0, EMBEDDING_DIM)

    def test_single(self, provider):
        assert provider.embed_documents(["x"]).shape == (1, EMBEDDING_DIM)

    def test_empty_string_among_real_texts(self, provider):
        """An empty definition still tokenises to [CLS] [SEP], so it must not collapse
        the batch width to zero or produce a NaN after normalisation."""
        v = provider.embed_documents(["", "customer account balance", ""])
        assert v.shape == (3, EMBEDDING_DIM)
        assert np.isfinite(v).all()
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    def test_text_longer_than_max_tokens_is_truncated(self, provider):
        v = provider.embed_documents(["word " * 5000, "short"])
        assert v.shape == (2, EMBEDDING_DIM)
        assert np.isfinite(v).all()
