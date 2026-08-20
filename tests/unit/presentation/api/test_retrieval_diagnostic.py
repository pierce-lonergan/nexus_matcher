"""
tests.unit.presentation.api.test_retrieval_diagnostic | Layer: TEST
POST /api/v1/diag/retrieval -- "why did this field not match?"

## Relationships
# TESTS → presentation/api/introspect :: the retrieval trace, its channels and its ranks
# TESTS → presentation/api/matching :: that the trace drives the pipeline matching drives

## The oracle this file is built around

A diagnostic that reports a pipeline the matcher does not run is worse than no diagnostic:
it looks authoritative and sends an operator after the wrong stage. The route reproduces the
retrieval half of `NexusMatcher._match_field` by calling the same private methods, and
nothing about calling private methods guarantees they are still the ones matching calls.

So the load-bearing test here is not a shape assertion. It is arithmetic:
**matching can only score what retrieval returned**, so EVERY candidate a real
`POST /api/v1/match` returns for a field must appear in the fused list this route reports
for the same field. That holds under any change to the five-signal scoring pass, to the
weights, or to the decision layer -- and fails the moment this route stops driving the same
retrieval.

Every candidate, not rank 1: the rank-1 version of that oracle was written first and was too
weak to be worth having. On a five-entry fixture the dense arm's own top candidate is
usually the one matching chooses, so a mutation that threw the lexical arm away and truncated
fusion to a single candidate passed it. The set version fails that mutation.

## What is deliberately NOT asserted

That a candidate's position here equals its position in `/api/v1/match`. It does not, and
saying so is the point: scoring reorders the fused list. A test that pinned the two together
would be asserting a property the module docstring explicitly disclaims, and would go red on
the next legitimate scoring change.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.introspect import RetrievalDiagnosticView
from nexus_matcher.presentation.api.limits import MatchServiceLimits
from tests.unit.presentation.api._support import FakeMatcher, build_api_matcher

# A field whose glossary entry is known, so "where did the entry I expected land" has an
# answer worth asserting.
RESIDENT_FIELD = {
    "name": "resident_nm",
    "path": "account.resident_nm",
    "doc": "Name of the resident on the account",
    "type": "string",
}

CHANNELS = ("dense", "sparse", "fused")

DIAGNOSTIC_KEYS = (
    "field",
    "queryText",
    "encoderModel",
    "rerankerWired",
    "channels",
    "expected",
)


def client_for(matcher: object, **kwargs: object) -> TestClient:
    app = create_app(configure_logs=False, matcher=matcher, environ={}, **kwargs)
    return TestClient(app)


@pytest.fixture
def real_client():
    with client_for(build_api_matcher()) as client:
        yield client


def diagnose(client: TestClient, **body: object) -> dict:
    payload = {"field": RESIDENT_FIELD, **body}
    response = client.post("/api/v1/diag/retrieval", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# =============================================================================
# THE TRACE
# =============================================================================


class TestTheTrace:
    """What the query became, and what each channel returned."""

    def test_the_keys_are_the_contract_in_order(self, real_client):
        assert tuple(diagnose(real_client)) == DIAGNOSTIC_KEYS

    def test_real_bodies_validate_against_the_published_schema(self, real_client):
        RetrievalDiagnosticView.model_validate(diagnose(real_client))

    def test_the_query_text_is_the_enriched_query_not_the_field_name(self, real_client):
        """
        Parent-path context injected into the query is the largest single accuracy factor
        measured on this task, and it is invisible to a caller looking at their own field.
        Reporting the field name back would be a diagnostic that answers a question nobody
        asked.
        """
        body = diagnose(real_client)

        assert body["queryText"] != RESIDENT_FIELD["name"]
        assert "account" in body["queryText"]
        assert "resident" in body["queryText"]
        assert RESIDENT_FIELD["doc"].lower() in body["queryText"].lower()

    def test_the_field_is_echoed_so_the_artifact_is_self_describing(self, real_client):
        assert diagnose(real_client)["field"] == RESIDENT_FIELD

    def test_every_channel_reports_its_depth_and_its_candidates(self, real_client):
        channels = diagnose(real_client)["channels"]

        assert tuple(channels) == CHANNELS
        for name in CHANNELS:
            assert channels[name]["available"] is True, name
            assert channels[name]["detail"] is None, name
            assert channels[name]["candidates"], name

        assert channels["dense"]["requestedTopK"] == 100
        assert channels["sparse"]["requestedTopK"] == 100

    def test_a_candidate_carries_the_channels_own_raw_score(self, real_client):
        """
        Raw, and per channel: a dense cosine, a BM25 score and a normalised fused score are
        three different quantities, and the whole value of this surface is seeing them
        unmixed. The BM25 arm here scores well above 1.0, which a normalised number cannot.
        """
        channels = diagnose(real_client)["channels"]

        dense_top = channels["dense"]["candidates"][0]
        assert dense_top["governanceId"] == "LWP-0001"
        assert dense_top["businessName"] == "Resident Full Name"
        assert 0.0 < dense_top["score"] <= 1.0

        assert channels["sparse"]["candidates"][0]["score"] > 1.0
        assert channels["fused"]["candidates"][0]["score"] == 1.0

    def test_candidates_are_truncated_for_display_and_counted_in_full(self, real_client):
        body = diagnose(real_client, top_k=2)

        for name in CHANNELS:
            channel = body["channels"][name]
            assert len(channel["candidates"]) <= 2, name
            assert channel["returned"] >= len(channel["candidates"]), name

        assert body["channels"]["dense"]["returned"] == 5

    def test_two_identical_requests_produce_identical_bytes(self, real_client):
        first = real_client.post("/api/v1/diag/retrieval", json={"field": RESIDENT_FIELD})
        second = real_client.post("/api/v1/diag/retrieval", json={"field": RESIDENT_FIELD})

        assert first.content == second.content

    def test_alias_hits_are_reported_under_the_entry_that_owns_them(self):
        """
        With aliasing on, the store returns fabricated technical spellings under synthetic
        ids. Matching max-pools them onto the owning entry BEFORE fusing, so a trace that
        fused the raw list would report a different fused order than the one matching used,
        and would name ids that are not in the caller's glossary at all.

        Asserted on every channel, because the collapse happens in one arm and reaches the
        fused list through it -- which is exactly how an earlier version of this route
        collapsed the dense DISPLAY and still fused the raw candidates.
        """
        matcher = build_api_matcher(
            config=MatchingConfig(results_per_field=5, dictionary_alias_count=4)
        )
        real_ids = set(matcher._dictionary_entries)
        assert matcher._alias_owner, "aliasing produced no alias ids, so this proves nothing"
        assert not (set(matcher._alias_owner) & real_ids)

        with client_for(matcher) as client:
            body = diagnose(client, top_k=100)

        for name in CHANNELS:
            for candidate in body["channels"][name]["candidates"]:
                assert candidate["governanceId"] in real_ids, (name, candidate)
                assert candidate["businessName"] is not None, (name, candidate)

    def test_a_server_with_no_reranker_says_so(self, real_client):
        """
        `rerankerWired` is how a reader knows whether the fused list is the order matching
        used or merely the input to a stage this trace does not run.
        """
        assert diagnose(real_client)["rerankerWired"] is False


# =============================================================================
# THE EXPECTED ENTRY
# =============================================================================


class TestWhereTheExpectedEntryLanded:
    """The two diagnoses this route must never conflate."""

    def test_a_retrieved_entry_reports_its_rank_in_every_channel(self, real_client):
        body = diagnose(real_client, expected_governance_id="LWP-0001")

        assert body["expected"]["governanceId"] == "LWP-0001"
        assert body["expected"]["inDictionary"] is True
        assert body["expected"]["rankByChannel"] == {"dense": 1, "sparse": 1, "fused": 1}

    def test_an_entry_that_is_in_the_glossary_but_was_not_retrieved_reports_null(self):
        """
        `inDictionary` TRUE with a null rank in every channel is a RETRIEVAL problem, and it
        is the case that separates the two diagnoses. It needs a retriever that genuinely
        leaves an indexed entry out, so `dense_top_k=1` supplies one -- on a five-entry
        fixture every other setting returns the whole corpus and the distinction is
        untestable, which is how an `inDictionary` derived from the retrieval result rather
        than from the dictionary passed an earlier version of this file.
        """
        shallow = build_api_matcher(
            config=MatchingConfig(results_per_field=5, dense_top_k=1, sparse_top_k=1)
        )
        with client_for(shallow) as client:
            body = diagnose(client, expected_governance_id="LWP-0004")

        assert body["expected"]["inDictionary"] is True
        assert body["expected"]["rankByChannel"] == {
            "dense": None,
            "sparse": None,
            "fused": None,
        }

    def test_an_entry_that_is_not_in_the_dictionary_says_so(self, real_client):
        """
        FALSE IS THE ANSWER: a field cannot match an entry that was never indexed, and no
        amount of threshold tuning fixes it. An operator told "rank: null" alone would go
        tuning.
        """
        body = diagnose(real_client, expected_governance_id="LWP-9999")

        assert body["expected"]["inDictionary"] is False
        assert body["expected"]["rankByChannel"] == {
            "dense": None,
            "sparse": None,
            "fused": None,
        }

    def test_no_expected_id_means_no_block_at_all(self, real_client):
        assert diagnose(real_client)["expected"] is None

    def test_the_rank_is_computed_over_the_full_result_not_the_truncation(self, real_client):
        """
        An expected entry at rank 4 with `top_k=1` must still report 4. Reporting null
        because it fell outside the display would be a wrong answer produced by a display
        setting.
        """
        full = diagnose(real_client, expected_governance_id="LWP-0003", top_k=100)
        rank = full["expected"]["rankByChannel"]["dense"]
        assert rank is not None and rank > 1, "the fixture no longer exercises this"

        truncated = diagnose(real_client, expected_governance_id="LWP-0003", top_k=1)

        assert len(truncated["channels"]["dense"]["candidates"]) == 1
        assert truncated["expected"]["rankByChannel"]["dense"] == rank


# =============================================================================
# THE ORACLE
# =============================================================================


def test_the_trace_drives_the_same_retrieval_matching_drives(real_client):
    """
    THE ANTI-DRIFT ORACLE.

    Matching can only score what retrieval returned, so EVERY candidate `/api/v1/match`
    returns for a field must appear among the candidates the fused channel reports for the
    same field. This holds under any change to the scoring pass, the weights or the decision
    layer, and it fails the moment this route stops driving the same pipeline -- which is the
    only thing standing between a heavily coupled diagnostic and a confidently wrong one.

    Every candidate rather than just rank 1, because rank 1 alone is a weak oracle on a small
    fixture: with five entries the dense arm's own top candidate is usually the one matching
    chooses, so a trace that had thrown the sparse arm away and truncated fusion to one
    candidate passed the rank-1 version of this test. It does not pass this one.

    Run over EVERY field of the fixture, so a trace that happened to agree on the easy field
    cannot pass either.
    """
    fields = [
        RESIDENT_FIELD,
        {"name": "meter_key", "path": "meter.meter_key", "doc": "Technician access key"},
        {"name": "usage_litres", "path": "meter.usage_litres", "doc": "Water drawn this month"},
        {"name": "tariff_band", "path": "billing.tariff_band", "doc": "Price band"},
    ]
    matched = real_client.post("/api/v1/match", json={"fields": fields, "top_k": 5})
    assert matched.status_code == 200, matched.text
    results = matched.json()["results"]
    assert any(len(candidates) > 1 for candidates in results.values()), (
        "every field came back with at most one candidate, so this oracle is comparing "
        "almost nothing"
    )

    for spec in fields:
        scored = {candidate["governanceId"] for candidate in results[spec["path"]]}
        if not scored:
            continue

        traced = diagnose(real_client, field=spec, top_k=100)
        fused = {
            candidate["governanceId"] for candidate in traced["channels"]["fused"]["candidates"]
        }

        assert scored <= fused, (
            f"matching scored {sorted(scored - fused)} for {spec['path']!r}, which the "
            f"diagnostic's fused channel does not contain ({sorted(fused)}). The trace and "
            f"the matcher are no longer driving the same retrieval, so this diagnostic is "
            f"reporting a pipeline that is not the one being diagnosed."
        )


# =============================================================================
# REFUSALS AND DEGRADATION
# =============================================================================


class TestRefusals:
    """Everything this route will not answer, and what it says instead."""

    def test_no_dictionary_is_503(self):
        with client_for(None) as client:
            response = client.post("/api/v1/diag/retrieval", json={"field": RESIDENT_FIELD})

        assert response.status_code == 503, response.text

    def test_a_matcher_that_cannot_be_traced_is_a_named_500(self):
        """
        `FakeMatcher` has no query builder and no encoder. A diagnostic that answered with
        empty channels would report "retrieval returned nothing" for a matcher that was never
        asked -- the most misleading answer available.
        """
        with client_for(FakeMatcher()) as client:
            response = client.post("/api/v1/diag/retrieval", json={"field": RESIDENT_FIELD})

        assert response.status_code == 500, response.text
        assert response.json()["error"]["details"]["attribute"] == "_build_query_text"

    def test_an_unknown_request_key_is_refused(self, real_client):
        response = real_client.post(
            "/api/v1/diag/retrieval", json={"field": RESIDENT_FIELD, "gold_id": "LWP-0001"}
        )

        assert response.status_code == 422, response.text

    def test_a_missing_field_is_refused(self, real_client):
        response = real_client.post("/api/v1/diag/retrieval", json={"top_k": 5})

        assert response.status_code == 422, response.text

    def test_a_slow_trace_returns_504_rather_than_hanging(self):
        """
        Unlike lookup, this route encodes a query and scans the corpus -- the same CPU work
        matching does, and therefore the same deadline. A 0.05 s deadline over a 2 s trace
        cannot be resolved the other way by any amount of machine load, so no timing
        assertion is being smuggled in here.
        """
        matcher = build_api_matcher()
        slow = matcher._build_query_text

        def _crawl(field):
            time.sleep(2.0)
            return slow(field)

        matcher._build_query_text = _crawl
        limits = MatchServiceLimits(deadline_seconds=0.05, max_workers=2, max_queued=4)
        with client_for(matcher, limits=limits) as client:
            response = client.post("/api/v1/diag/retrieval", json={"field": RESIDENT_FIELD})

        assert response.status_code == 504, response.text
        assert response.json()["error"]["details"]["deadline_seconds"] == 0.05

    def test_a_dense_only_matcher_reports_the_sparse_channel_as_unwired(self):
        """
        Not every deployment wires a sparse retriever. "Unavailable, and here is why" is a
        different answer from "returned nothing", and an operator reading an empty lexical
        list would conclude their query has no matching tokens.
        """
        matcher = build_api_matcher()
        matcher._sparse_retriever = None

        with client_for(matcher) as client:
            body = diagnose(client)

        sparse = body["channels"]["sparse"]
        assert sparse["available"] is False
        assert "no sparse retriever" in sparse["detail"]
        assert sparse["candidates"] == []
        assert sparse["requestedTopK"] is None
        # The dense and fused channels still answer, because a dense-only deployment is a
        # supported one rather than a broken one.
        assert body["channels"]["dense"]["available"] is True
        assert body["channels"]["fused"]["available"] is True
