"""
tests.properties._support | Layer: TEST
The deterministic encoder and the strategies every property in this package draws from.

## Why a bag-of-tokens encoder and not the bundled ONNX model

Three things a property suite needs that a learned encoder cannot give:

1. **Determinism that does not depend on `PYTHONHASHSEED`.** Token -> dimension goes
   through blake2b, never `hash()`. H-005 is about ranking that moved with the hash seed;
   a fixture that reintroduced that dependency would make every property in this package
   flaky for a reason that has nothing to do with the code under test, and a flaky gate
   trains people to ignore red.

2. **A CONSTRUCTIBLE ZERO.** `test_metamorphic` needs a glossary entry that is provably
   irrelevant to a query -- zero token overlap *and* cosine below 0.2. Here that is not a
   hope: `BUSINESS_WORDS` and `DISJOINT_WORDS` own reserved, non-overlapping dimensions,
   so a text built only from one vocabulary is EXACTLY orthogonal to a text built only
   from the other. The tests assert the measured cosine rather than assuming it.

3. **Cost.** Hypothesis runs hundreds of examples per property and most of them build two
   matchers. A 33 MB ONNX load per matcher is not affordable, and the accuracy of the
   embeddings is irrelevant to every property here -- all of them are about how the
   pipeline handles vectors, not about which vectors it gets.

The encoder is otherwise a faithful stand-in: unit-norm float32 rows of the declared
dimension, so the store's normalization, the batched GEMM path and the min-max fusion all
run exactly as they do in production. That matters -- the float32 reassociation the GEMM
path introduces is precisely what forces one of the tolerances below.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import FlattenedAvroParser
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, Result

# =============================================================================
# HYPOTHESIS SETTINGS
# =============================================================================

# `deadline=None` and the `too_slow` suppression are NOT a way to let a slow property
# pass. They remove hypothesis's TIMING assertions, which this repo forbids outright:
# other agents run concurrently, and H-007 measured a 30.6% throughput band on identical
# code at 49.5% CPU busy against 0.9% idle. A per-example deadline under that load is a
# coin toss, and a gate that fails for reasons unrelated to correctness is the flaky gate
# this package exists to avoid. Everything asserted here is contention-immune.
PROPERTY_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# =============================================================================
# VOCABULARIES
# =============================================================================

# Ordinary glossary vocabulary. Schema fields and dictionary entries are both built from
# this, so a query has real overlap with real entries and ranking is not all ties.
BUSINESS_WORDS: tuple[str, ...] = (
    "customer",
    "email",
    "address",
    "account",
    "balance",
    "transaction",
    "amount",
    "identifier",
    "opened",
    "phone",
    "number",
    "status",
    "posting",
    "merchant",
    "settlement",
)

# Deliberately unpronounceable non-words. Nothing in BUSINESS_WORDS, in the type
# descriptions ContextEnricher appends, or in any domain name can collide with these, so
# "an entry with zero token overlap" is a fact about the fixture rather than a guess.
DISJOINT_WORDS: tuple[str, ...] = (
    "zqxvbn",
    "wprtly",
    "kjhgfd",
    "mnbvcx",
    "poiuyt",
    "vbnmqw",
    "lkjhgz",
)

# Parent paths, drawn from the business vocabulary so a query never contains a digit --
# a digit token would be shared by every field and would blunt the orthogonality above.
PARENT_PATHS: tuple[str, ...] = ("customer", "account", "merchant", "posting")

DOMAINS: tuple[str, ...] = ("CUSTOMER", "ACCOUNTS", "TRANSACTIONS", "")

DATA_TYPES: tuple[DataType, ...] = (
    DataType.STRING,
    DataType.LONG,
    DataType.DOUBLE,
    DataType.BOOLEAN,
    DataType.TIMESTAMP,
)

# =============================================================================
# THE ENCODER
# =============================================================================

DIMENSION = 256

# Reserved, injective dimensions for the two known vocabularies. This is what makes
# `cosine(disjoint text, business text) == 0.0` exact rather than merely likely: a
# hash-only mapping collides at roughly (tokens_a * tokens_b / DIMENSION) per pair, which
# over a few hundred hypothesis examples is a near-certainty, and a fixture that only
# usually holds its own precondition produces a gate that only usually means anything.
_RESERVED: dict[str, int] = {
    word: index for index, word in enumerate((*BUSINESS_WORDS, *DISJOINT_WORDS))
}
_TAIL_START = len(_RESERVED)
_TAIL_WIDTH = DIMENSION - _TAIL_START

_TOKEN = re.compile(r"[0-9A-Za-z]+")


def _dimension_for(token: str) -> int:
    """Reserved dimension for a known word, a blake2b-derived tail dimension otherwise."""
    reserved = _RESERVED.get(token)
    if reserved is not None:
        return reserved
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return _TAIL_START + int.from_bytes(digest, "big") % _TAIL_WIDTH


def encode(text: str) -> np.ndarray:
    """A unit-norm bag-of-tokens vector. Same text in, bit-identical vector out, always."""
    vector = np.zeros(DIMENSION, dtype=np.float32)
    for token in _TOKEN.findall(text.lower()):
        vector[_dimension_for(token)] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        # A text with no alphanumeric tokens at all -- "..." or "" -- still needs a unit
        # vector, because the store divides by the row norm. One fixed direction, so such
        # entries are identical to each other and orthogonal to everything else.
        vector[_TAIL_START] = 1.0
        return vector
    return vector / norm


class BagOfTokensProvider:
    """An EmbeddingProvider-shaped encoder with no model behind it."""

    dimension = DIMENSION
    model_name = "bag-of-tokens"

    def embed(self, texts: Any) -> Result:
        rows = _stack(texts)

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text: str) -> Result:
        return Result.success(encode(text))

    def embed_documents(self, texts: Any) -> np.ndarray:
        """`ingest._embed_documents` prefers this name, so the sync tests exercise it."""
        return _stack(texts)


def _stack(texts: Any) -> np.ndarray:
    rows = [encode(t) for t in texts]
    if not rows:
        return np.zeros((0, DIMENSION), dtype=np.float32)
    return np.stack(rows)


# =============================================================================
# MATCHER CONSTRUCTION
# =============================================================================


def build_matcher(
    entries: list[DictionaryEntry],
    *,
    sparse: bool = True,
    results_per_field: int = 5,
) -> NexusMatcher:
    """
    A matcher over `entries`, wired the way `from_config` wires the shipped one.

    `sparse` is a parameter rather than always-on because the two arms have genuinely
    different invariants under corpus change: a dense score is a property of one entry and
    one query, while a BM25 score is a property of the whole corpus (see
    `test_metamorphic.TestCorpusGrowth`). Conflating them is how a metamorphic test ends
    up asserting something that is true of one arm and false of the other.
    """
    matcher = NexusMatcher(
        embedding_provider=BagOfTokensProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=DIMENSION)
        ),
        sparse_retriever=BM25Retriever() if sparse else None,
        schema_parser_registry={"flattened_avro": FlattenedAvroParser()},
        config=MatchingConfig(results_per_field=results_per_field),
    )
    matcher._index_dictionary(entries)
    return matcher


# =============================================================================
# STRATEGIES
# =============================================================================


def _phrase(words: tuple[str, ...], min_size: int = 1, max_size: int = 3) -> st.SearchStrategy:
    return st.lists(st.sampled_from(words), min_size=min_size, max_size=max_size).map(" ".join)


def _identifier(words: tuple[str, ...], max_size: int = 2) -> st.SearchStrategy:
    return st.lists(st.sampled_from(words), min_size=1, max_size=max_size).map("_".join)


@st.composite
def glossary(
    draw: Any,
    words: tuple[str, ...] = BUSINESS_WORDS,
    id_prefix: str = "e",
    min_size: int = 2,
    max_size: int = 10,
) -> list[DictionaryEntry]:
    """
    A dictionary with ids assigned by position, so ids are unique but texts need not be.

    Colliding texts are left in on purpose. The 0.0024 measured margin of H-005 means
    near-ties are the normal regime here, and a strategy that quietly removed them would
    only ever exercise the easy half of the input space -- which is the half that was
    already working.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    entries = []
    for index in range(size):
        entries.append(
            DictionaryEntry(
                id=f"{id_prefix}{index}",
                business_name=draw(_phrase(words)).title(),
                logical_name=draw(_identifier(words)),
                definition=draw(_phrase(words, min_size=0, max_size=4)),
                data_type=draw(st.sampled_from(DATA_TYPES)),
                domain=draw(st.sampled_from(DOMAINS)),
            )
        )
    return entries


@st.composite
def schema_fields(draw: Any, min_size: int = 1, max_size: int = 6) -> list[SchemaField]:
    """Fields whose names, paths and descriptions come only from BUSINESS_WORDS."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    fields = []
    for _ in range(size):
        name = draw(_identifier(BUSINESS_WORDS))
        parent = draw(st.sampled_from(PARENT_PATHS))
        fields.append(
            SchemaField(
                name=name,
                data_type=draw(st.sampled_from(DATA_TYPES)),
                full_path=f"{parent}.{name}",
                parent_path=parent,
                description=draw(_phrase(BUSINESS_WORDS, min_size=0, max_size=3)),
            )
        )
    return fields


@st.composite
def disjoint_entries(
    draw: Any, id_prefix: str = "shard", min_size: int = 1, max_size: int = 8
) -> list[DictionaryEntry]:
    """
    Entries built only from DISJOINT_WORDS -- provably zero token overlap with any query.

    `data_type` is drawn from the same pool as the fields rather than pinned to something
    incompatible. Pinning it would hand every one of these entries a low type-compatibility
    score and let a property pass because of the FIXTURE rather than because of the code.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return [
        DictionaryEntry(
            id=f"{id_prefix}{index}",
            business_name=draw(_phrase(DISJOINT_WORDS)).title(),
            logical_name=draw(_identifier(DISJOINT_WORDS)),
            definition=draw(_phrase(DISJOINT_WORDS, min_size=0, max_size=3)),
            data_type=draw(st.sampled_from(DATA_TYPES)),
            domain=draw(st.sampled_from(DOMAINS)),
        )
        for index in range(size)
    ]


# =============================================================================
# READING RESULTS
# =============================================================================


def ranked_ids(matches: Any) -> tuple[str, ...]:
    return tuple(m.dictionary_entry.id for m in matches)


def confidences(matches: Any) -> tuple[float, ...]:
    return tuple(m.final_confidence for m in matches)


def strictly_separated_ranks(
    scores: tuple[float, ...] | list[float],
    *,
    tolerance: float = 0.0,
    truncated: bool = True,
) -> list[int]:
    """
    The ranks whose score is unique in the WHOLE scored list, not just the visible window.

    Ranks that are NOT separated are held by a tie, and a tie is settled by the tie-break
    rather than by the scores. That distinction is load-bearing rather than pedantic: the
    dense tie-break in this codebase is `numpy.argsort`/`argpartition`, neither of which is
    stable, so the order among equal cosines is arbitrary and CHANGES when the corpus
    changes size. Asserting on a tied rank therefore produces a test that reddens on
    correct code. See `test_metamorphic.TestDuplicateInsertion`, which documents the
    reproduction.

    A rank strictly below its predecessor and strictly above its successor is held by
    exactly one candidate: nothing earlier can share it (the list is sorted and the
    predecessor is strictly greater) and nothing later can either, including candidates a
    truncation hid, since all of those score at most the successor's score.

    `truncated` says whether the tail of the list was cut off. When it was, the last
    visible rank is excluded, because a tie straddling the cut is invisible from here.
    `tolerance` widens "strictly" to "by more than this", for lists whose scores carry
    float32 reassociation noise.
    """
    separated = []
    last = len(scores) - 1 if truncated else len(scores)
    for rank in range(last):
        above_ok = rank == 0 or scores[rank - 1] - scores[rank] > tolerance
        below_ok = rank == len(scores) - 1 or scores[rank] - scores[rank + 1] > tolerance
        if above_ok and below_ok:
            separated.append(rank)
    return separated
