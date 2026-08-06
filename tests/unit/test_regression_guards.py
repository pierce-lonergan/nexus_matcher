"""
tests.unit.test_regression_guards | Layer: TEST
Fast, model-free guards for defects that previously shipped silently.

Every test here corresponds to a bug that was live in the codebase and that the existing
suite could not detect. They are deliberately cheap (no model downloads, no network) so
they run on every commit, and each one fails loudly if the specific defect returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
    _squash_score,
)
from nexus_matcher.domain.models.entities import DictionaryEntry, SchemaField
from nexus_matcher.domain.ports.retrieval import SparseDocument
from nexus_matcher.domain.ports.vector_store import VectorDocument, VectorStoreConfig
from nexus_matcher.domain.services.abbreviation import AbbreviationExpander
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import DataType, MatchDecision

# =============================================================================
# ABBREVIATION EXPANSION
# =============================================================================


class TestAbbreviationSeparators:
    """
    The expander rebuilt text using the separator style of the INPUT. Natural-language
    input (which is exactly what ContextEnricher produces) has no underscores or
    hyphens, so it fell through to the camelCase branch and was concatenated into one
    out-of-vocabulary mega-token. That zeroed every BM25 score and halved dense recall.
    """

    def test_whitespace_text_stays_multi_word(self):
        expander = AbbreviationExpander.default()
        result = expander.expand("customer, account cust acct bal amt")

        assert " " in result.expanded, (
            f"multi-word text was collapsed into a single token: {result.expanded!r}"
        )
        # The classic failure signature: an interior capital with no separator.
        assert "AccountCustomer" not in result.expanded

    def test_whitespace_text_keeps_token_count(self):
        expander = AbbreviationExpander.default()
        text = "customer account balance amount field"
        expanded = expander.expand(text).expanded
        assert len(expanded.split()) >= len(text.split())

    @pytest.mark.parametrize(
        "text,separator",
        [
            ("cust_acct_bal", "_"),
            ("cust-acct-bal", "-"),
        ],
    )
    def test_identifier_separators_are_preserved(self, text, separator):
        expander = AbbreviationExpander.default()
        expanded = expander.expand(text).expanded
        assert separator in expanded
        assert " " not in expanded

    def test_camel_case_identifier_still_camel(self):
        expander = AbbreviationExpander.default()
        expanded = expander.expand("custAcctBal").expanded
        assert " " not in expanded
        assert "_" not in expanded


# =============================================================================
# SCORE NORMALIZATION
# =============================================================================


class TestSquashScore:
    """
    Reranker outputs are unbounded logits. They feed `semantic_score`, which carries 70%
    of final confidence, so leaving them unsquashed collapsed every plausible candidate
    to the same clamped value and made the auto-approve threshold meaningless.
    """

    def test_in_range_scores_pass_through_unchanged(self):
        for s in (0.0, 0.25, 0.5, 1.0):
            assert _squash_score(s) == pytest.approx(s)

    def test_out_of_range_scores_land_in_unit_interval(self):
        for s in (-50.0, -8.0, 1.5, 12.0, 500.0):
            assert 0.0 <= _squash_score(s) <= 1.0

    def test_squash_is_monotonic_so_ordering_is_preserved(self):
        raw = [-9.0, -3.0, -0.5, 1.4, 2.0, 7.5, 40.0]
        squashed = [_squash_score(s) for s in raw]
        assert squashed == sorted(squashed)

    def test_distinct_logits_stay_distinguishable(self):
        # The old min(score, 1.0) mapped every one of these to exactly 1.0.
        assert _squash_score(1.2) != _squash_score(6.0)


# =============================================================================
# DECISION POLICY
# =============================================================================


class TestDecisionPolicy:
    """`min_confidence_gap` must actually prevent ambiguous auto-approvals."""

    def _matcher(self):
        m = NexusMatcher.__new__(NexusMatcher)
        m._config = MatchingConfig()
        return m

    def test_confident_and_clear_is_auto_approved(self):
        m = self._matcher()
        assert m._determine_decision(0.92, 1, 0.40) == MatchDecision.AUTO_APPROVE

    def test_confident_but_ambiguous_goes_to_review(self):
        m = self._matcher()
        # 0.90 vs 0.88 is inside min_confidence_gap (0.10): a human should decide.
        assert m._determine_decision(0.90, 1, 0.88) == MatchDecision.REVIEW

    def test_only_rank_one_can_auto_approve(self):
        m = self._matcher()
        assert m._determine_decision(0.99, 2, 0.10) != MatchDecision.AUTO_APPROVE

    def test_sole_confident_candidate_is_auto_approved(self):
        m = self._matcher()
        assert m._determine_decision(0.95, 1, None) == MatchDecision.AUTO_APPROVE

    def test_low_confidence_is_rejected(self):
        m = self._matcher()
        assert m._determine_decision(0.10, 1, None) == MatchDecision.REJECT


# =============================================================================
# EDIT DISTANCE
# =============================================================================


class TestEditDistance:
    """The fast path must agree exactly with the reference implementation."""

    @pytest.mark.parametrize(
        "a,b",
        [
            ("cust_acct_bal", "customer_account_balance"),
            ("", "abc"),
            ("same", "same"),
            ("kitten", "sitting"),
            ("a", "bbbbbbbb"),
        ],
    )
    def test_fast_path_matches_reference(self, a, b):
        m = NexusMatcher.__new__(NexusMatcher)
        fast = m._edit_distance_score(a, b)
        ref = NexusMatcher._edit_distance_score_fallback(a, b) if a and b else 0.0
        assert fast == pytest.approx(ref, abs=1e-9)

    def test_score_is_bounded(self):
        m = NexusMatcher.__new__(NexusMatcher)
        assert m._edit_distance_score("kitten", "sitting") <= 1.0
        assert m._edit_distance_score("kitten", "sitting") >= 0.0


# =============================================================================
# VECTOR STORE
# =============================================================================


class TestInMemoryVectorStore:
    """Search must be exact, ordered, and stable across repeated calls."""

    def _store(self, n=200, d=32, seed=0):
        rng = np.random.default_rng(seed)
        store = InMemoryVectorStore(VectorStoreConfig(collection_name="c", dimension=d))
        vecs = rng.standard_normal((n, d)).astype(np.float32)
        store.upsert(
            [
                VectorDocument(id=str(i), embedding=vecs[i], payload={"g": "a" if i % 2 else "b"})
                for i in range(n)
            ]
        )
        return store, vecs

    def test_matches_brute_force_cosine(self):
        store, vecs = self._store()
        q = vecs[7] + 0.01

        got = [r.id for r in store.search(q, top_k=10).unwrap()]

        norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        sims = norm @ (q / np.linalg.norm(q))
        expected = [str(i) for i in np.argsort(-sims)[:10]]
        assert got == expected

    def test_results_are_sorted_descending(self):
        store, vecs = self._store()
        scores = [r.score for r in store.search(vecs[3], top_k=25).unwrap()]
        assert scores == sorted(scores, reverse=True)

    def test_repeated_search_is_stable(self):
        """Guards the old bug where the corpus was re-normalized in place each call."""
        store, vecs = self._store()
        first = [(r.id, r.score) for r in store.search(vecs[5], top_k=10).unwrap()]
        for _ in range(5):
            again = [(r.id, r.score) for r in store.search(vecs[5], top_k=10).unwrap()]
            assert again == first

    def test_filter_restricts_results(self):
        store, vecs = self._store()
        results = store.search(vecs[1], top_k=10, filter={"g": "a"}).unwrap()
        assert results
        assert all(int(r.id) % 2 == 1 for r in results)

    def test_top_k_larger_than_corpus(self):
        store, vecs = self._store(n=5, d=8)
        assert len(store.search(vecs[0], top_k=100).unwrap()) == 5


# =============================================================================
# BM25
# =============================================================================


class TestBM25Removal:
    """
    remove() reassigned _id_to_index before reading the old corpus position, so every
    surviving document silently inherited a different document's tokens.
    """

    def test_surviving_documents_keep_their_own_tokens(self):
        r = BM25Retriever()
        texts = {
            "a": "customer account balance",
            "b": "transaction posting date",
            "c": "product catalog code",
            "d": "customer email address",
            "e": "loan principal amount",
        }
        r.index([SparseDocument(id=k, text=v, metadata={}) for k, v in texts.items()])
        r.remove(["b", "c"])

        for doc_id in ("a", "d", "e"):
            idx = r._id_to_index[doc_id]
            assert r._tokenized_corpus[idx] == texts[doc_id].split(), (
                f"{doc_id} inherited the wrong token list"
            )

    def test_search_still_finds_the_right_document_after_removal(self):
        # A corpus of this size is needed for BM25 IDF to be non-degenerate: with only
        # two documents left, log((N-df+0.5)/(df+0.5)) collapses to 0 for every term and
        # the score filter discards everything regardless of correctness.
        r = BM25Retriever()
        docs = {
            "a": "customer account balance",
            "b": "transaction posting date",
            "c": "loan principal amount",
            "d": "product catalog code",
            "e": "shipping address city",
            "f": "employee hire date",
            "g": "invoice total value",
            "h": "vendor contact email",
        }
        r.index([SparseDocument(id=k, text=v, metadata={}) for k, v in docs.items()])
        r.remove(["b", "d"])

        hits = r.search("loan principal", top_k=3).unwrap()
        assert hits and hits[0].id == "c"

        hits = r.search("shipping city", top_k=3).unwrap()
        assert hits and hits[0].id == "e"

    def test_removed_documents_are_never_returned(self):
        r = BM25Retriever()
        docs = {
            "a": "customer account balance",
            "b": "customer account balance",
            "c": "loan principal amount",
            "d": "product catalog code",
            "e": "shipping address city",
            "f": "employee hire date",
        }
        r.index([SparseDocument(id=k, text=v, metadata={}) for k, v in docs.items()])
        r.remove(["a"])
        assert all(h.id != "a" for h in r.search("customer balance", top_k=6).unwrap())


# =============================================================================
# MULTI-SIGNAL RANKING
# =============================================================================


class TestMultiSignalRanking:
    """
    The multi-signal score was computed but never used to order results, so 30% of the
    scoring weight had no effect on which entry was returned first.
    """

    def test_results_are_ordered_by_final_confidence(self):
        class _Provider:
            dimension = 8
            model_name = "stub"

            def embed(self, texts):
                from nexus_matcher.shared.types.base import Result

                arr = np.tile(np.eye(1, 8, 0, dtype=np.float32), (len(list(texts)), 1))

                class _B:
                    embeddings = arr

                return Result.success(_B())

            def embed_single(self, text):
                from nexus_matcher.shared.types.base import Result

                return Result.success(np.eye(1, 8, 0, dtype=np.float32)[0])

        store = InMemoryVectorStore(VectorStoreConfig(collection_name="dictionary", dimension=8))
        matcher = NexusMatcher(
            embedding_provider=_Provider(),
            vector_store=store,
            config=MatchingConfig(results_per_field=5),
        )
        entries = [
            DictionaryEntry(
                id=f"e{i}",
                business_name=name,
                logical_name=name.lower().replace(" ", "_"),
                definition="",
                data_type=DataType.STRING,
                domain="d",
            )
            for i, name in enumerate(
                ["customer account balance", "totally unrelated", "customer account bal"]
            )
        ]
        matcher._index_dictionary(entries)

        field = SchemaField(
            name="customer_account_balance",
            data_type=DataType.STRING,
            full_path="t.customer_account_balance",
            parent_path="t",
        )
        results = matcher._match_field(field)

        assert results
        confidences = [r.final_confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))


# =============================================================================
# CONTEXT ENRICHMENT
# =============================================================================


class TestContextEnrichment:
    """
    Query-side context is the largest single accuracy factor in the pipeline
    (+20 points of P@1 on the combined benchmark), and scalar type words are a
    measured net negative. Both defaults are guarded here so neither silently flips.
    """

    def _field(self, name, parent="", path=None, **kw):
        from nexus_matcher.domain.models.entities import SchemaField

        return SchemaField(
            name=name,
            data_type=kw.pop("data_type", DataType.STRING),
            full_path=path if path is not None else (f"{parent}.{name}" if parent else name),
            parent_path=parent,
            **kw,
        )

    def test_parent_path_is_included(self):
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        result = ContextEnricher().enrich(self._field("sname", parent="satscores"))
        assert "satscores" in result
        assert "sname" in result

    def test_scalar_type_words_are_not_added_by_default(self):
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        result = ContextEnricher().enrich(self._field("email_address", parent="users"))
        assert "text" not in result
        assert "field" not in result

    def test_array_structure_is_still_signalled(self):
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        field = self._field("tags", parent="product", is_array=True)
        assert "array" in ContextEnricher().enrich(field)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("enroll12", ["enroll", "12"]),
            ("NumTstTakr", ["num", "tst", "takr"]),
            ("FRPM Count (K-12)", ["frpm", "count", "k", "12"]),
            ("HTTPResponseCode", ["http", "response", "code"]),
        ],
    )
    def test_identifiers_are_split_into_tokens(self, name, expected):
        """Unsplit identifiers become out-of-vocabulary blobs for BM25 and the tokenizer."""
        from nexus_matcher.domain.services.context_enricher import ContextEnricher

        tokens = ContextEnricher().enrich(self._field(name)).split()
        for want in expected:
            assert want in tokens, f"{name!r} -> {tokens} is missing {want!r}"


# =============================================================================
# DICTIONARY-SIDE ALIAS GENERATION
# =============================================================================


class TestAliasGeneration:
    """
    Index-time alias enrichment is the only technique that improved the corrected
    benchmark. Its value depends entirely on the SELECTIVITY gate: applied to every
    entry it cost 11.3 points on the OMOP split, because a business name shared by many
    entries produces short contentless vectors that win spurious max-pool matches.
    """

    def test_generates_plausible_spellings(self):
        from nexus_matcher.domain.services.alias_generation import generate_aliases

        aliases = generate_aliases("Number of Test Takers", 6)
        assert aliases
        # Stopwords dropped, so "of" must not survive as its own token.
        assert all("of" not in a.split() for a in aliases)
        # At least one contracted form.
        assert any("num" in a for a in aliases)

    def test_aliases_are_distinct(self):
        from nexus_matcher.domain.services.alias_generation import generate_aliases

        aliases = generate_aliases("Customer Account Balance", 8)
        assert len(aliases) == len(set(aliases))

    def test_respects_the_cap(self):
        from nexus_matcher.domain.services.alias_generation import generate_aliases

        assert len(generate_aliases("Free or Reduced Price Meal Count", 3)) <= 3

    def test_empty_name_yields_nothing(self):
        from nexus_matcher.domain.services.alias_generation import generate_aliases

        assert generate_aliases("", 6) == []
        assert generate_aliases("of the", 6) == []

    def test_deterministic(self):
        from nexus_matcher.domain.services.alias_generation import generate_aliases

        assert generate_aliases("Average Scholastic Reading Score", 6) == generate_aliases(
            "Average Scholastic Reading Score", 6
        )

    def test_shared_names_are_not_alias_worthy(self):
        """A name on 30 entries cannot identify one; aliasing it is pure noise."""
        from nexus_matcher.domain.services.alias_generation import is_alias_worthy

        assert is_alias_worthy("Number of Test Takers", share_count=1) is True
        assert is_alias_worthy("person", share_count=30) is False

    def test_single_word_names_are_not_alias_worthy(self):
        from nexus_matcher.domain.services.alias_generation import is_alias_worthy

        assert is_alias_worthy("person", share_count=1) is False

    def test_expand_dictionary_skips_shared_names(self):
        from nexus_matcher.domain.services.alias_generation import expand_dictionary

        entries = [("specific", "Number of Test Takers")] + [
            (f"shared{i}", "person visit") for i in range(10)
        ]
        rows = expand_dictionary(entries, max_aliases=6)
        assert rows
        assert {owner for owner, _ in rows} == {"specific"}

    def test_expand_dictionary_owner_ids_resolve(self):
        from nexus_matcher.domain.services.alias_generation import expand_dictionary

        entries = [("a", "Customer Account Balance"), ("b", "Transaction Posting Date")]
        rows = expand_dictionary(entries, max_aliases=4)
        assert {owner for owner, _ in rows} == {"a", "b"}
        assert all(text.strip() for _, text in rows)
