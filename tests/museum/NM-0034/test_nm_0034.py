"""
NM-0034 -- the answer depended on which row was listed first.

BM25 floors a negative IDF at `epsilon * average_idf`, the way rank_bm25 does. The average
was taken with `ndarray.sum()`, a pairwise reduction over an array laid out in first-seen
term order -- which is glossary ROW order. On the four-entry corpus below the raw IDFs
cancel exactly: five terms at -0.847298, four at +0.847298, exact total 0.0. A pairwise
reduction of those nine values returns +2.220446e-16 walking them in one order and
-2.220446e-16 walking them in another, so the FLOOR CHANGES SIGN when the rows are
re-ordered.

That sign is not a rounding difference by the time a caller sees it, because two things
downstream turn it into a decision:

  * `BM25Retriever.search` keeps a document only when its score is > 0. A negative floor
    makes every floored weight negative, so the sparse arm returns NOTHING; a positive one
    makes it return three documents scoring around 1.3e-17.
  * `fuse_linear_ids` min-max normalises each arm before weighting it, and min-max is
    scale-free. An arm made entirely of 1e-17 dust is therefore stretched across the whole
    [0, 1] range, and its best document is handed a lexical component of 1.0 rather than
    the 0.0 an absent arm gives.

At `fusion_alpha` = 0.90 that is 0.10 of fused retrieval score, and at `semantic_weight` =
0.70 it is 0.07 of published confidence: rank 1 for `customer.customer` came back at
0.88625 with the rows listed e0 e1 e2 e3 and at 0.95625 with the same rows listed
e2 e0 e1 e3. `auto_approve_threshold` is 0.87 and `review_threshold` is 0.50, so a 0.07
step is large enough to move a field across a governance boundary; here both values sit
above 0.87, and the honest statement is that the SCORE an auditor reads changed, not that
this particular field changed lane.

Same class as NM-0020, reached differently. There the order leaked out of a hash-ordered
container; here out of a float64 reduction. Both computed a property of the corpus as a
SET by reading the corpus as a SEQUENCE.

Both levels are asserted, as in NM-0020. The confidence is what a user meets. The BM25
output is where the order actually leaked, and pinning it there means a future regression
is reported against the stage that caused it instead of arriving as an unexplained
end-to-end difference.

SELF-CONTAINED ON PURPOSE. The encoder below is twenty lines rather than an import of
`tests.properties._support`, so that adding a word to another lane's fixture vocabulary
cannot quietly change the numbers this entry pins. Nothing here needs a real model: every
assertion is about how the pipeline handles vectors, not about which vectors it gets.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.retrieval import SparseDocument
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, Result

# The falsifying example hypothesis shrank the property down to, written out as literals.
# `.hypothesis/` is gitignored, so the database that found this does not travel and a fresh
# clone inherits no counterexample -- eight fresh runs of the property against the broken
# code went 0/8. The example is therefore the artifact, not the search that produced it.
#
# FOUR ROWS IS AS SMALL AS THIS GETS, and that was checked rather than assumed: a
# 400,000-trial search over randomly built 3- and 4-document corpora (vocabularies of 6 to
# 14 terms) turned up four-document sign flips readily and no three-document one. At three
# documents the raw IDFs can only take the values +0.5108, -0.5108 and -1.9459, so a total
# of exactly zero needs symmetric pairs -- and symmetric pairs cancel exactly whichever way
# a pairwise reduction blocks them.
ENTRIES: tuple[tuple[str, str, str, str], ...] = (
    ("e0", "Customer", "account", ""),
    ("e1", "Address", "balance", "email"),
    ("e2", "Customer Address Account", "email", ""),
    ("e3", "Transaction Identifier Opened", "account_balance", "customer email address amount"),
)
LISTED_ORDER: tuple[str, ...] = ("e0", "e1", "e2", "e3")
PERMUTED_ORDER: tuple[str, ...] = ("e2", "e0", "e1", "e3")

QUERY = "customer customer"

# What the pipeline returns once the average is summed exactly. 0.88625 is the LOWER of the
# two values the defect produced: the sparse arm is genuinely empty for this query, so rank
# 1's fused retrieval score is `fusion_alpha` and nothing else.
EXPECTED_TOP_ID = "e0"
EXPECTED_CONFIDENCE = 0.88625
EXPECTED_FUSED_RETRIEVAL = 0.90

# What the defect produced when the rows were permuted, recorded so the size of the
# movement is legible without re-deriving it.
DEFECTIVE_CONFIDENCE = 0.95625


# =============================================================================
# A DETERMINISTIC ENCODER
# =============================================================================

_TOKEN = re.compile(r"[0-9A-Za-z]+")

# One reserved dimension per word these fixtures can produce, so a vector is an exact
# bag of tokens and two texts sharing no word are exactly orthogonal. Never `hash()`:
# NM-0020 is the entry about ranking that moved with the hash seed, and a fixture that
# reintroduced that would make this one flaky for a reason unrelated to what it tests.
_VOCABULARY: tuple[str, ...] = (
    "customer",
    "address",
    "account",
    "balance",
    "email",
    "transaction",
    "identifier",
    "opened",
    "amount",
    "string",
    "text",
    "value",
    "characters",
)
_DIMENSION = len(_VOCABULARY) + 1  # +1 catch-all for anything the enricher appends
_INDEX = {word: position for position, word in enumerate(_VOCABULARY)}


def _encode(text: str) -> np.ndarray:
    vector = np.zeros(_DIMENSION, dtype=np.float32)
    for token in _TOKEN.findall(text.lower()):
        vector[_INDEX.get(token, _DIMENSION - 1)] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        vector[_DIMENSION - 1] = 1.0
        return vector
    return vector / norm


class _BagOfTokens:
    """An EmbeddingProvider-shaped encoder with no model behind it."""

    dimension = _DIMENSION
    model_name = "nm-0034-bag-of-tokens"

    @staticmethod
    def _stack(texts) -> np.ndarray:
        rows = [_encode(t) for t in texts]
        if not rows:
            return np.zeros((0, _DIMENSION), dtype=np.float32)
        return np.stack(rows)

    def embed(self, texts) -> Result:
        rows = self._stack(texts)

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text: str) -> Result:
        return Result.success(_encode(text))

    def embed_documents(self, texts) -> np.ndarray:
        return self._stack(texts)


# =============================================================================
# FIXTURES
# =============================================================================


def _entries(order: tuple[str, ...]) -> list[DictionaryEntry]:
    by_id = {row[0]: row for row in ENTRIES}
    return [
        DictionaryEntry(
            id=by_id[entry_id][0],
            business_name=by_id[entry_id][1],
            logical_name=by_id[entry_id][2],
            definition=by_id[entry_id][3],
            data_type=DataType.STRING,
            domain="CUSTOMER",
        )
        for entry_id in order
    ]


def _field() -> SchemaField:
    return SchemaField(
        name="customer",
        data_type=DataType.STRING,
        full_path="customer.customer",
        parent_path="customer",
        description="",
    )


def _matcher(order: tuple[str, ...]) -> NexusMatcher:
    matcher = NexusMatcher(
        embedding_provider=_BagOfTokens(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=_DIMENSION)
        ),
        sparse_retriever=BM25Retriever(),
        config=MatchingConfig(results_per_field=5),
    )
    matcher._index_dictionary(_entries(order))
    return matcher


def _retriever(order: tuple[str, ...]) -> BM25Retriever:
    retriever = BM25Retriever()
    result = retriever.index(
        [SparseDocument(id=e.id, text=e.to_searchable_text()) for e in _entries(order)]
    )
    assert result.is_success, "the fixture corpus failed to index"
    return retriever


def _top_matches(order: tuple[str, ...]):
    results = _matcher(order)._match_fields([_field()])
    matches = next(iter(results.values()), ())
    assert matches, "the matcher returned no candidates, so every assertion here is vacuous"
    return matches


# =============================================================================
# GUARDS AGAINST PASSING VACUOUSLY
# =============================================================================


def test_the_two_orders_really_are_the_same_rows():
    """
    If the two orders ever stopped being permutations of one another, every comparison
    below would be measuring a difference in CONTENT and would hold for reasons that have
    nothing to do with this defect.
    """
    assert sorted(LISTED_ORDER) == sorted(PERMUTED_ORDER)
    assert LISTED_ORDER != PERMUTED_ORDER
    assert sorted(LISTED_ORDER) == sorted(row[0] for row in ENTRIES)


def test_the_corpus_still_sits_on_the_knife_edge():
    """
    Guards the guards.

    The sign flip needed raw IDFs summing to EXACTLY zero, which is a property of these
    four definitions and nothing else. Edit any of them and the floor becomes an ordinary
    number, both orders agree for a reason unrelated to the fix, and this entry goes on
    printing PASS while covering nothing.
    """
    retriever = _retriever(LISTED_ORDER)
    tokenized = [retriever._tokenize(e.to_searchable_text()) for e in _entries(LISTED_ORDER)]
    n_docs = len(tokenized)
    raw = [
        math.log(n_docs - df + 0.5) - math.log(df + 0.5)
        for df in (sum(1 for tokens in tokenized if term in tokens) for term in retriever._vocab)
    ]

    assert any(value < 0 for value in raw), (
        "no term appears in more than half this corpus, so the epsilon floor is never "
        "applied and this fixture can no longer reproduce the defect"
    )
    assert math.fsum(raw) == 0.0, (
        f"the raw IDFs no longer cancel (exact total {math.fsum(raw)!r}), so the average "
        f"is no longer a quantity whose SIGN a reduction order can decide"
    )


def test_the_query_actually_reaches_the_floored_term():
    """
    `customer` is the term whose IDF is floored, and it is the only term in the query. If
    the query stopped hitting a floored term the sparse arm would be decided by ordinary
    positive weights and the sign of the floor would not matter.
    """
    retriever = _retriever(LISTED_ORDER)
    tokens = retriever._tokenize(QUERY)
    assert tokens, "the query tokenises to nothing"
    assert set(tokens) <= set(retriever._vocab), "the query term is not in the corpus vocabulary"
    n_docs = len(ENTRIES)
    for token in set(tokens):
        df = sum(
            1
            for e in _entries(LISTED_ORDER)
            if token in retriever._tokenize(e.to_searchable_text())
        )
        assert math.log(n_docs - df + 0.5) - math.log(df + 0.5) < 0, (
            f"query term {token!r} has a non-negative raw IDF, so it is never floored and "
            f"this entry no longer exercises the defect"
        )


# =============================================================================
# THE CAUSE
# =============================================================================


def test_bm25_returns_the_same_documents_whatever_order_they_were_indexed_in():
    """
    EXACT, ids and scores both.

    A BM25 score is a function of the corpus as a SET: document frequency, average document
    length and the epsilon floor are all statistics of that set, and none of them has any
    business knowing which row was listed first. This is the assertion that fails first
    when the average goes back to an order-dependent reduction, and it fails at full
    strength rather than as a rounding difference -- three documents against none.
    """
    runs = {}
    for label, order in (("listed", LISTED_ORDER), ("permuted", PERMUTED_ORDER)):
        found = _retriever(order).search(QUERY, top_k=100)
        assert found.is_success
        runs[label] = [(hit.id, hit.score) for hit in found.unwrap()]

    assert runs["listed"] == runs["permuted"], (
        f"BM25 returned {runs['listed']} for the glossary as listed and "
        f"{runs['permuted']} for a permutation of the same rows. A lexical score is "
        f"depending on where an entry sits in the corpus rather than on what it says -- "
        f"look at the epsilon IDF floor in _term_idf."
    )


def test_the_epsilon_floor_is_the_same_number_in_both_orders():
    """
    The defect, named at the exact quantity that carried it.

    Asserting the floor rather than only the scores means a regression is reported as
    "the average IDF moved" instead of as "some documents appeared", which is the
    difference between a diagnosis and a symptom.
    """
    floors = {}
    for label, order in (("listed", LISTED_ORDER), ("permuted", PERMUTED_ORDER)):
        retriever = _retriever(order)
        tokenized = retriever._tokenized_corpus
        df = np.array(
            [sum(1 for tokens in tokenized if term in tokens) for term in retriever._vocab],
            dtype=np.int64,
        )
        floors[label] = float(retriever._term_idf(df, len(tokenized)).min())

    assert floors["listed"] == floors["permuted"], (
        f"the floored IDF is {floors['listed']!r} for the glossary as listed and "
        f"{floors['permuted']!r} for a permutation of it. The average is being summed by "
        f"a reduction whose answer depends on the order of its input."
    )


# =============================================================================
# THE SYMPTOM
# =============================================================================


def test_the_published_confidence_does_not_move_when_the_rows_are_re_ordered():
    """
    The number a reviewer reads, and the one an auto-approve threshold is compared against.

    Asserted bit-exactly rather than within a float32 tolerance, which this fixture can
    afford and the hypothesis property cannot: four rows and a fourteen-dimensional encoder
    give a GEMM small enough that permuting the rows does not reassociate anything. If a
    future BLAS makes that untrue, the honest repair is a tolerance around 1e-7 -- the
    measured float32 band -- and NOT one that would also swallow the 7.0e-02 below.
    """
    listed = _top_matches(LISTED_ORDER)
    permuted = _top_matches(PERMUTED_ORDER)

    assert [m.dictionary_entry.id for m in listed] == [m.dictionary_entry.id for m in permuted], (
        "re-ordering the glossary rows changed the ranking"
    )
    for rank, (was, now) in enumerate(zip(listed, permuted, strict=True)):
        assert was.final_confidence == now.final_confidence, (
            f"rank {rank} confidence moved "
            f"{abs(was.final_confidence - now.final_confidence):.3e} on row order alone: "
            f"{was.final_confidence!r} -> {now.final_confidence!r}. This is NM-0034."
        )


def test_rank_one_is_the_entry_and_the_score_the_defect_moved():
    """
    Names the numbers, so the story above can be checked rather than trusted.

    `fused_retrieval_score` is the one worth pinning: it is exactly `fusion_alpha` here
    because the sparse arm is empty, and the defect showed as it becoming 1.00 -- the
    signature of a lexical arm that min-max manufactured out of 1e-17 dust.
    """
    for label, order in (("listed", LISTED_ORDER), ("permuted", PERMUTED_ORDER)):
        top = _top_matches(order)[0]
        assert top.dictionary_entry.id == EXPECTED_TOP_ID, (
            f"{label}: rank 1 is {top.dictionary_entry.id!r}, not {EXPECTED_TOP_ID!r}"
        )
        assert top.score_breakdown.fused_retrieval_score == pytest.approx(
            EXPECTED_FUSED_RETRIEVAL, abs=1e-9
        ), (
            f"{label}: fused retrieval score is "
            f"{top.score_breakdown.fused_retrieval_score!r}, not the {EXPECTED_FUSED_RETRIEVAL} "
            f"an empty sparse arm gives"
        )
        assert top.final_confidence == pytest.approx(EXPECTED_CONFIDENCE, abs=1e-9), (
            f"{label}: rank 1 confidence is {top.final_confidence!r}, not "
            f"{EXPECTED_CONFIDENCE}. The defect produced {DEFECTIVE_CONFIDENCE} here."
        )
