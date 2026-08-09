"""
tests.unit.infrastructure.test_bm25_index_build | Layer: TEST
The vectorised index build must produce exactly the postings the scalar build did.

Why this file exists
--------------------
test_bm25_vectorized.py pins the SCORES against rank_bm25, which is the contract that
matters to a user. This file pins the structures underneath it, because the index build
was rewritten from a Python loop over every (term, document) pair into a flatten /
compute / sort pipeline in numpy (~2x faster to build at 30k-100k entries), and that
rewrite has failure modes the score comparison cannot see:

  * A wrong stride between the offsets and the postings mixes one term's documents into
    another's. On a small corpus the scores can still look plausible.
  * `np.bincount(term_ids)` is used as the document frequency AND as the offset stride.
    If those two ever disagree, every term after the first bad one is misaligned.
  * The weights are accumulated with `scores[docs] += w`, which SILENTLY keeps only the
    last write instead of summing if a document ever appears twice in one term's
    postings. No exception, just a quietly wrong score.
  * The postings are grouped with a STABLE argsort. An unstable sort still scores
    correctly, so nothing else in the suite would notice it being changed.

Each test below names the specific thing that breaks, so a failure here points at a
cause rather than at "BM25 is wrong somewhere".
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from nexus_matcher.domain.ports.retrieval import SparseDocument
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever

VOCAB = [
    "customer",
    "order",
    "email",
    "address",
    "amount",
    "total",
    "status",
    "code",
    "date",
    "identifier",
    "the",
    "of",
    "a",
    "record",
    "payment",
    "invoice",
]


def _corpus(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(VOCAB) for _ in range(rng.randint(3, 14))) for _ in range(n)]


def _indexed(texts: list[str]) -> BM25Retriever:
    r = BM25Retriever()
    r.index([SparseDocument(id=f"d{i}", text=t) for i, t in enumerate(texts)])
    return r


def _postings(r: BM25Retriever, term: str) -> tuple[np.ndarray, np.ndarray]:
    tid = r._vocab[term]
    lo, hi = r._term_offsets[tid], r._term_offsets[tid + 1]
    return r._postings_docs[lo:hi], r._postings_weights[lo:hi]


class TestPostingsLayout:
    """The offsets, the document ids and the weights all have to agree with each other."""

    def test_each_terms_postings_are_exactly_the_documents_containing_it(self):
        """
        Pins offset/posting misalignment: a stride computed from the wrong array, or a
        vocabulary whose ids do not match the order the postings were grouped in, hands
        back another term's documents. Scores stay finite and plausible, so only a direct
        set comparison catches it.
        """
        texts = _corpus(300)
        r = _indexed(texts)
        tokenized = [r._tokenize(t) for t in texts]

        for term in r._vocab:
            expected = {i for i, toks in enumerate(tokenized) if term in toks}
            docs, _ = _postings(r, term)
            assert set(docs.tolist()) == expected, f"postings for {term!r} are not its documents"
            assert len(docs) == len(expected), f"duplicate postings for {term!r}"

    def test_offsets_span_the_whole_postings_array_without_gaps_or_overlap(self):
        """
        Pins the offsets being a true cumulative sum: they must start at 0, end at the
        posting count, and never step backwards. A gap silently drops a term's documents;
        an overlap double-counts them.
        """
        r = _indexed(_corpus(300))
        off = r._term_offsets
        assert off[0] == 0
        assert off[-1] == r._postings_docs.size == r._postings_weights.size
        assert len(off) == len(r._vocab) + 1
        assert np.all(np.diff(off) >= 0), "offsets must be non-decreasing"

    def test_document_frequency_matches_the_offset_stride(self):
        """
        Pins the dual use of np.bincount: it supplies BOTH the document frequency that
        feeds IDF and the per-term stride. If they ever diverge, the IDF is computed for
        one term while the postings are read for another.
        """
        texts = _corpus(400)
        r = _indexed(texts)
        tokenized = [r._tokenize(t) for t in texts]
        for term, tid in r._vocab.items():
            df = sum(1 for toks in tokenized if term in toks)
            assert r._term_offsets[tid + 1] - r._term_offsets[tid] == df

    def test_no_document_repeats_within_a_single_terms_postings(self):
        """
        Pins the precondition for `scores[docs] += weights`. Fancy-index accumulation
        applies only the LAST write when an index repeats rather than summing, so a
        duplicated posting understates a score with no error raised.
        """
        r = _indexed(["email email email address", *_corpus(200)])
        for term in r._vocab:
            docs, _ = _postings(r, term)
            assert len(docs) == len(set(docs.tolist())), f"{term!r} has a duplicate posting"

    def test_documents_stay_in_ascending_order_within_each_term(self):
        """
        Pins the STABLE argsort used to group the postings. An unstable sort scores
        identically, so no other test would fail -- but it scatters each term's document
        ids, turning the search-time gather from a sequential read into a random one.
        """
        r = _indexed(_corpus(400))
        for term in r._vocab:
            docs, _ = _postings(r, term)
            assert np.all(np.diff(docs) > 0), f"{term!r} postings are not ascending"


class TestWeights:
    """The precomputed weight is the entire BM25 formula; check it against the formula."""

    def test_weights_equal_the_bm25_term_weight_computed_independently(self):
        """
        Pins the in-place weight pipeline. The build computes `idf * f * (k1+1) / (f + norm)`
        as a chain of in-place numpy updates to avoid allocating a temporary per operator;
        an operand applied in the wrong order, or against the wrong document's length
        normalisation, still yields finite weights of roughly the right magnitude.
        """
        texts = _corpus(200)
        k1, b = 1.5, 0.75
        r = _indexed(texts)
        tokenized = [r._tokenize(t) for t in texts]
        lengths = np.array([len(t) for t in tokenized], dtype=np.float64)
        avgdl = lengths.mean()
        n = len(texts)

        # IDF straight from the definition, including rank_bm25's epsilon floor.
        raw = {}
        for term in r._vocab:
            df = sum(1 for toks in tokenized if term in toks)
            raw[term] = np.log(n - df + 0.5) - np.log(df + 0.5)
        floor = 0.25 * (sum(raw.values()) / len(raw))
        idf = {t: (v if v >= 0 else floor) for t, v in raw.items()}

        for term in r._vocab:
            docs, weights = _postings(r, term)
            for doc_id, w in zip(docs.tolist(), weights.tolist(), strict=True):
                f = tokenized[doc_id].count(term)
                norm = k1 * (1.0 - b + b * lengths[doc_id] / avgdl)
                expected = idf[term] * f * (k1 + 1.0) / (f + norm)
                assert w == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_weights_are_stored_as_float32_on_purpose(self):
        """
        Pins the storage width. float32 was verified sufficient by measurement -- against
        a full float64 build the worst relative score error is ~1e-7 and NO ranking moves,
        on the FHIR corpus (1556 queries) or a 100k synthetic one -- while halving the
        postings array and making the search-time gather 1.44x faster at 100k entries.
        Widening this back to float64 costs both for no accuracy gain.
        """
        r = _indexed(_corpus(50))
        assert r._postings_weights.dtype == np.float32
        assert r._postings_docs.dtype == np.int32
        assert r._term_offsets.dtype == np.int64


class TestTokenizer:
    """The tokenizer feeds both the index and every query; drift here moves both."""

    @pytest.mark.parametrize(
        "text",
        [
            "Customer Email Address",
            "  leading and trailing  ",
            "tabs\tand\nnewlines\r\nmixed",
            "punctuation!!! removed, entirely.",
            "hyphen-separated_and_underscored",
            "unicode\xa0nbsp thin　ideographic",
            "accented Ä é Ç and CJK 日本語",
            "digits 123 mixed4with5letters",
            "",
            "!!!",
            "\u200b\u200b",
        ],
    )
    def test_tokens_are_lowercase_alphanumeric_and_never_empty(self, text):
        """
        Pins the tokenizer contract after the per-token strip()/filter was removed as
        redundant (str.split() cannot yield an empty or whitespace-padded token). If the
        substitution pattern is ever changed so that it can, this catches it -- an empty
        token becomes a vocabulary entry matching nothing and shifts every later term id.
        """
        tokens = BM25Retriever()._tokenize(text)
        for t in tokens:
            assert t, "empty token"
            assert t == t.strip(), f"token {t!r} carries whitespace"
            assert t == t.lower(), f"token {t!r} was not lowercased"
            assert all(c.isalnum() and c.isascii() for c in t), f"token {t!r} has a stray char"

    def test_whitespace_variants_all_separate_tokens(self):
        """
        Pins that re's \\s and str.split() agree on what whitespace is. They are relied on
        to be interchangeable; if they ever diverge, two words fuse into one token that
        matches no query.
        """
        r = BM25Retriever()
        for sep in (" ", "\t", "\n", "\r", "\v", "\f", "\xa0", " ", "　", "\x1c"):
            assert r._tokenize(f"alpha{sep}beta") == ["alpha", "beta"], repr(sep)


class TestRebuildPaths:
    """add/remove/load all re-enter the same build; none may leave it inconsistent."""

    def test_add_remove_and_reindex_keep_the_postings_consistent(self):
        """
        Pins the incremental paths against the vectorised build. Each one rebuilds from
        _tokenized_corpus, so an off-by-one in the corpus bookkeeping shows up as postings
        that no longer match the documents -- the exact bug the remove() ordering comment
        in bm25.py describes.
        """
        texts = _corpus(120)
        r = _indexed(texts)

        r.add([SparseDocument(id="extra", text="unmistakable zebra quasar")])
        r.remove(["d3", "d7", "d11"])
        r.add([SparseDocument(id="d3", text="customer email restored")])

        corpus = {doc_id: r._tokenize(r._documents[doc_id].text) for doc_id in r._index_to_id}
        for term in r._vocab:
            docs, _ = _postings(r, term)
            expected = {r._id_to_index[doc_id] for doc_id, toks in corpus.items() if term in toks}
            assert set(docs.tolist()) == expected, f"{term!r} inconsistent after add/remove"

    def test_save_and_load_reproduce_the_index_bit_for_bit(self, tmp_path):
        """
        Pins that load() rebuilds an identical index rather than a merely similar one.
        load() re-runs the whole vectorised build from the pickled token lists, so any
        dependence on transient state during a build would show up here as drift.
        """
        r = _indexed(_corpus(200))
        path = str(tmp_path / "bm25.pkl")
        assert r.save(path).unwrap()

        loaded = BM25Retriever()
        assert loaded.load(path).unwrap()

        assert loaded._vocab == r._vocab
        assert np.array_equal(loaded._term_offsets, r._term_offsets)
        assert np.array_equal(loaded._postings_docs, r._postings_docs)
        assert np.array_equal(
            loaded._postings_weights.view(np.int32), r._postings_weights.view(np.int32)
        ), "weights differ bitwise after a save/load round trip"


class TestDegenerateCorpora:
    """The vectorised build has empty-array edge cases the scalar loop did not."""

    def test_corpus_of_only_empty_documents_does_not_raise(self):
        """
        Pins the zero-posting path. Every document tokenising to nothing makes avgdl 0,
        which leaves the length-normalisation array undefined; the scalar loop simply
        never executed its body, whereas the vectorised build would divide by zero or
        index an empty array if the guard were dropped.
        """
        r = BM25Retriever()
        assert r.index([SparseDocument(id=f"e{i}", text="   !!!  ") for i in range(5)]).is_success
        assert r._postings_docs.size == 0
        assert r.search("anything", top_k=5).unwrap() == []

    def test_single_document_corpus(self):
        r = _indexed(["customer email address"])
        assert r._term_offsets[-1] == r._postings_docs.size == 3
        assert r.search("customer", top_k=5).is_success

    def test_one_term_present_in_every_document_is_floored_not_negative(self):
        """
        Pins the vectorised epsilon floor. It is applied with np.copyto(where=idf < 0)
        rather than a Python loop over the negative terms; a mask built the wrong way
        round would floor exactly the terms that should not be floored.
        """
        r = _indexed([f"common token{i % 5}" for i in range(30)])
        _, weights = _postings(r, "common")
        assert np.all(weights > 0), "a term in every document must not score negatively"
