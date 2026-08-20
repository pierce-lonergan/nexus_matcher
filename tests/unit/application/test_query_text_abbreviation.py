"""
tests.unit.application.test_query_text_abbreviation | Layer: TEST
Tests: NexusMatcher._build_query_text | Target: application/use_cases/match_schema.py

An enriched query is not one identifier. It is several -- one per parent-path level --
joined by `HIERARCHY_SEPARATOR`. Running abbreviation expansion over the whole string
therefore fed the expander a token with the separator stuck to it: it looked up
`"acct,"`, comma and all, missed, and passed it through raw, while the identical `acct`
expanded to `account` everywhere else in the same query. The level that failed to expand
was always a parent-path level -- the part of the query carrying the most signal.

Invisible on the committed benchmarks: `AbbreviationExpander.default()` holds a generic
list containing none of the tokens that sit before a separator in those corpora, so
whole-string and per-level expansion agree on 0 of 2244 queries there. With a governed
catalog -- the configuration that makes the flag worth enabling at all -- it changed 619
of 1556 FHIR queries, 96.3% of every query having any parent-path structure.

The flag itself stays OFF by default; `test_flag_is_off_by_default` pins that. These
tests are about what the flag does WHEN a caller turns it on.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.models.entities import SchemaField
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.domain.services.abbreviation import (
    AbbreviationDictionary,
    AbbreviationExpander,
)
from nexus_matcher.domain.services.context_enricher import HIERARCHY_SEPARATOR
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, Result

# A small GOVERNED catalog: generic transport/finance short forms, exact lookup only.
# Every key is a short form a caller's approved-abbreviation list would plausibly hold.
CATALOG = {
    "txn": "transaction",
    "amt": "amount",
    "acct": "account",
    "svc": "service",
    "psgr": "passenger",
    "dpt": "departure",
    "term": "terminal",
}


class _StubProvider:
    """No model, no network: these tests are about query TEXT, not retrieval."""

    dimension = 4
    model_name = "stub"

    def embed(self, texts):
        rows = np.zeros((len(list(texts)), 4), dtype=np.float32)

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        return Result.success(np.zeros(4, dtype=np.float32))


def _matcher(*, expand: bool, catalog=CATALOG) -> NexusMatcher:
    return NexusMatcher(
        embedding_provider=_StubProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=4)
        ),
        abbreviation_expander=AbbreviationExpander(AbbreviationDictionary.from_dict(catalog)),
        config=MatchingConfig(expand_query_abbreviations=expand),
    )


def _nested_field() -> SchemaField:
    """Two parent levels, so the enriched query carries a separator between them."""
    return SchemaField(
        name="txn_amt",
        data_type=DataType.DECIMAL,
        full_path="acct.svc.txn_amt",
        parent_path="acct.svc",
    )


class TestDefaultIsUnchanged:
    def test_flag_is_off_by_default(self):
        assert MatchingConfig().expand_query_abbreviations is False

    def test_with_the_flag_off_the_query_is_the_enriched_text_verbatim(self):
        matcher = _matcher(expand=False)
        field = _nested_field()
        assert matcher._build_query_text(field) == matcher._context_enricher.enrich(field)

    def test_with_the_flag_off_no_abbreviation_is_touched(self):
        assert "acct" in _matcher(expand=False)._build_query_text(_nested_field())


class TestPerLevelExpansion:
    def test_a_level_followed_by_a_separator_still_expands(self):
        """
        The regression. `acct` is the first parent level, so whole-string expansion saw
        `"acct,"` and left it alone -- in the highest-signal part of the query.
        """
        query = _matcher(expand=True)._build_query_text(_nested_field())
        assert "account" in query
        assert "acct" not in query

    def test_the_parent_path_structure_survives(self):
        matcher = _matcher(expand=True)
        field = _nested_field()
        query = matcher._build_query_text(field)
        # Two parent levels -> exactly one separator between them, and expansion must
        # neither consume it nor add one.
        enriched = matcher._context_enricher.enrich(field)
        assert enriched.count(HIERARCHY_SEPARATOR) == 1
        assert query.count(HIERARCHY_SEPARATOR) == 1
        assert query == "account, service transaction amount"

    def test_every_level_expands_not_just_the_last(self):
        field = SchemaField(
            name="dpt_term",
            data_type=DataType.STRING,
            full_path="psgr.svc.dpt_term",
            parent_path="psgr.svc",
        )
        query = _matcher(expand=True)._build_query_text(field)
        assert query == "passenger, service departure terminal"

    def test_a_flat_field_is_unaffected_by_the_split(self):
        """No parent path means no separator; per-level must degrade to whole-string."""
        field = SchemaField(name="txn_amt", data_type=DataType.STRING, full_path="txn_amt")
        assert _matcher(expand=True)._build_query_text(field) == "transaction amount"

    def test_tokens_absent_from_the_catalog_pass_through(self):
        """Exact lookup with passthrough -- an unknown short form is left alone."""
        field = SchemaField(
            name="zzz_amt",
            data_type=DataType.STRING,
            full_path="acct.zzz_amt",
            parent_path="acct",
        )
        assert _matcher(expand=True)._build_query_text(field) == "account zzz amount"


class TestResultIsLowercasedExplicitly:
    """
    The expander restores each replaced token's case style, so a descriptive phrase can
    put capitals back into the query. Today's BGE tokenizer is uncased and cannot see
    them; that is a property of the current encoder, not of this contract.
    """

    def test_case_from_a_description_does_not_leak_into_the_query(self):
        field = SchemaField(
            name="txn_amt",
            data_type=DataType.DECIMAL,
            full_path="acct.txn_amt",
            parent_path="acct",
            description="Total AMT settled per ACCT",
        )
        query = _matcher(expand=True)._build_query_text(field)
        assert query == query.lower()
        # and the uppercase short forms in the description did expand
        assert "amount" in query
        assert "amt" not in query


class TestAgreementWithWholeStringWhereItWasAlreadyRight:
    """
    Per-level expansion must only differ where the separator was in the way. On a query
    with no parent path the two are the same computation, and on the shipped generic
    dictionary they agree on both committed corpora.
    """

    @pytest.mark.parametrize("name", ["txn_amt", "svc_amt", "unknown_field", "id"])
    def test_flat_queries_match_whole_string_expansion(self, name):
        matcher = _matcher(expand=True)
        field = SchemaField(name=name, data_type=DataType.STRING, full_path=name)
        enriched = matcher._context_enricher.enrich(field)
        whole_string = matcher._abbreviation_expander.expand(enriched).expanded.lower()
        assert matcher._build_query_text(field) == whole_string
