"""
tests.unit.presentation.api.test_lookup_endpoint | Layer: TEST
The wire contract of GET /api/v1/lookup/{id} and POST /api/v1/lookup.

## Relationships
# TESTS → presentation/api/lookup :: resolution, the found/not-found contract, admission
# TESTS → presentation/api/matching :: that the two enrichment surfaces stay the same one

Four properties are load-bearing here:

  COMPLETENESS  every requested id comes back exactly once, in the order sent, mapped to
                an entry or to an explicit null. A partial list is the conservation defect
                one surface over: the caller's key vanishes and nothing says so.
  EXACTNESS     a hit carries no rank, no confidence, no decision and no score. A lookup
                is exact by construction, and a number here would invite thresholding on
                something nobody measured.
  ALIGNMENT     the enrichment a looked-up entry carries is the enrichment a match
                candidate carries, member for member and in the same order, so a caller
                can feed either into one code path.
  ISOLATION     a lookup takes no permit from the bounded work pool. It does no CPU work,
                and letting the cheap route shed the expensive one inverts the purpose of
                admission control.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.limits import MatchServiceLimits, worst_case_body_bytes
from nexus_matcher.presentation.api.lookup import (
    MAX_DICTIONARY_ID_CHARS,
    LookupResponseView,
)
from nexus_matcher.shared.types.base import DataType, ProtectionLevel
from tests.unit.presentation.api._support import (
    GLOSSARY,
    TIER_OPEN,
    FakeMatcher,
    build_api_matcher,
    governance_vocabulary,
    request_fields,
)

# The exact keys of a looked-up entry, in the exact order the contract fixes them. A
# literal, not something read back off the response: an expectation derived from the code
# under test is an identity, and an identity holds just as well when both sides are wrong.
ENTRY_KEYS = (
    "governanceId",
    "businessName",
    "definition",
    "domain",
    "governance",
    # ENRICHMENT, not scoring, so it belongs on this plane: it describes the glossary row
    # and says nothing about a match. It is what lets this plane claim to return "the same
    # enrichment surface as a match candidate" -- before it, a caller resolving an id it
    # already held got this library's four members and none of its own columns. The round
    # trip and the truncation marker are pinned in `test_metadata_plane.py`.
    "sourceMetadata",
)

# The top level. `missing` sits between the answer and the block that interprets it.
RESPONSE_KEYS = ("results", "missing", "vocabulary")

# Candidate members that are claims about a MATCH rather than about the entry, and are
# therefore deliberately absent from a lookup. Listed by name so that a member added to a
# candidate by another lane is either enrichment (and must appear on a looked-up entry) or
# is named here as scoring -- and until somebody decides which,
# `test_the_entry_surface_is_the_candidate_surface_minus_the_match` says so out loud.
MATCH_ONLY_CANDIDATE_KEYS = frozenset(
    {"rank", "confidence", "decision", "absoluteScore", "explain"}
)


def client_for(matcher: object, **kwargs: object) -> TestClient:
    """A client over an app whose only configuration is the matcher under test."""
    app = create_app(configure_logs=False, matcher=matcher, environ={}, **kwargs)
    return TestClient(app)


@pytest.fixture
def real_client():
    """A real NexusMatcher over the fictional glossary."""
    with client_for(build_api_matcher()) as client:
        yield client


def lookup(client: TestClient, *ids: str) -> dict:
    """POST the batch route and return the parsed body, failing loudly on a non-200."""
    response = client.post("/api/v1/lookup", json={"ids": list(ids)})
    assert response.status_code == 200, response.text
    return response.json()


# =============================================================================
# COMPLETENESS
# =============================================================================


class TestEveryIdComesBack:
    """A partial list is refused as a design; both answers travel in the body."""

    def test_hits_and_misses_share_one_map_in_the_order_sent(self, real_client):
        body = lookup(real_client, "LWP-0003", "NOT-A-TERM", "LWP-0001")

        assert tuple(body) == RESPONSE_KEYS
        assert list(body["results"]) == ["LWP-0003", "NOT-A-TERM", "LWP-0001"]
        assert body["results"]["NOT-A-TERM"] is None
        assert body["results"]["LWP-0003"]["businessName"] == "Monthly Usage Litres"

    def test_missing_names_exactly_the_null_values(self, real_client):
        body = lookup(real_client, "LWP-0001", "GONE-1", "LWP-0002", "GONE-2")

        assert body["missing"] == ["GONE-1", "GONE-2"]
        assert [key for key, value in body["results"].items() if value is None] == body["missing"]

    def test_every_glossary_id_resolves(self, real_client):
        """
        The whole fixture in one request, so a resolver that happened to work for the ids
        an assertion above names cannot pass by coincidence.
        """
        ids = [entry.id for entry in GLOSSARY]
        body = lookup(real_client, *ids)

        assert body["missing"] == []
        assert [value["governanceId"] for value in body["results"].values()] == ids

    def test_a_miss_is_a_200_with_null_and_not_a_404(self, real_client):
        """
        404 on this service means the ROUTE does not exist, and it renders into the same
        error envelope every other failure uses. Spending it on "no such entry" would make
        "you called a path that does not exist" and "that term was retired" indistinguishable
        without reading prose.
        """
        response = real_client.get("/api/v1/lookup/NOT-A-TERM")

        assert response.status_code == 200, response.text
        assert response.json()["results"] == {"NOT-A-TERM": None}
        assert response.json()["missing"] == ["NOT-A-TERM"]

        unknown_route = real_client.get("/api/v1/lookups/NOT-A-TERM")
        assert unknown_route.status_code == 404
        assert "error" in unknown_route.json()

    def test_the_single_route_answers_the_same_body_as_a_one_id_batch(self, real_client):
        """One DTO for both routes, so a generated client has one model."""
        single = real_client.get("/api/v1/lookup/LWP-0002")
        batch = real_client.post("/api/v1/lookup", json={"ids": ["LWP-0002"]})

        assert single.content == batch.content

    def test_an_id_containing_a_slash_is_still_addressable(self):
        """
        Dictionary ids are opaque strings this library does not get to constrain. A route
        that could not express half of them would send exactly the callers this plane
        exists for back through matching.
        """
        slashed = DictionaryEntry(
            id="LWP/2004/A",
            business_name="Hydrant Flow Test",
            logical_name="hydrant_flow_test",
            definition="Flow measured at a hydrant during a scheduled test.",
            data_type=DataType.FLOAT,
            protection_level=ProtectionLevel.INTERNAL,
            governance_code="USAGEAGG",
            domain="OPERATIONS",
        )
        with client_for(build_api_matcher(entries=(*GLOSSARY, slashed))) as client:
            response = client.get("/api/v1/lookup/LWP/2004/A")

            assert response.status_code == 200, response.text
            assert response.json()["results"]["LWP/2004/A"]["businessName"] == "Hydrant Flow Test"


# =============================================================================
# EXACTNESS AND ALIGNMENT
# =============================================================================


class TestTheEnrichmentSurface:
    """What a hit carries, and what it deliberately does not."""

    def test_an_entry_carries_the_enrichment_keys_in_order_and_nothing_else(self, real_client):
        entry = lookup(real_client, "LWP-0001")["results"]["LWP-0001"]

        assert tuple(entry) == ENTRY_KEYS

    @pytest.mark.parametrize("absent", sorted(MATCH_ONLY_CANDIDATE_KEYS))
    def test_no_score_no_rank_and_no_decision(self, real_client, absent):
        """
        A hit is exact or it is absent. A confidence here would be either the constant 1.0
        -- inviting a threshold on a number nobody measured -- or a fiction.
        """
        entry = lookup(real_client, "LWP-0001")["results"]["LWP-0001"]

        assert absent not in entry

    def test_the_entry_surface_is_the_candidate_surface_minus_the_match(self, real_client):
        """
        THE COORDINATION GATE, and it is deliberately strict.

        "The same enrichment surface as a match candidate" is a promise that has to be
        checked against the candidate the service actually sends, not against a copy of the
        key list. A member added to a candidate is either a fact about the ENTRY -- in which
        case a looked-up entry must carry it too, or the same caller gets two different
        answers about one glossary row -- or it is a claim about the match, in which case it
        belongs in `MATCH_ONLY_CANDIDATE_KEYS`. Neither is a decision this test can make; it
        exists so that nobody makes it by accident.
        """
        matched = real_client.post("/api/v1/match", json={"fields": request_fields(), "top_k": 1})
        assert matched.status_code == 200, matched.text
        candidate = matched.json()["results"]["account.resident_nm"][0]

        undeclared = set(candidate) - set(ENTRY_KEYS) - MATCH_ONLY_CANDIDATE_KEYS
        assert not undeclared, (
            f"a match candidate now carries {sorted(undeclared)}, which the lookup plane "
            f"neither returns nor declares to be match-only. Decide which it is: add it to "
            f"`lookup.LookupEntryView` and `lookup._entry_payload` if it describes the "
            f"ENTRY, or to MATCH_ONLY_CANDIDATE_KEYS if it describes the MATCH."
        )

        # Same members AND the same relative order, so the two surfaces cannot drift into
        # spelling the same facts in a different sequence.
        assert [key for key in candidate if key in ENTRY_KEYS] == list(ENTRY_KEYS)

        looked_up = lookup(real_client, candidate["governanceId"])["results"][
            candidate["governanceId"]
        ]
        for key in ENTRY_KEYS:
            assert looked_up[key] == candidate[key], key

    def test_an_entry_with_no_code_is_an_explicit_null_the_vocabulary_explains(self, real_client):
        """
        "This entry has no class" and "this response forgot" must not look alike to a caller
        whose next step is applying a classification.
        """
        body = lookup(real_client, "LWP-0005")
        entry = body["results"]["LWP-0005"]

        assert "governance" in entry
        assert entry["governance"] is None
        assert body["vocabulary"]["openClassification"] == TIER_OPEN

    def test_a_coded_entry_carries_the_whole_class(self, real_client):
        entry = lookup(real_client, "LWP-0002")["results"]["LWP-0002"]

        assert entry["governance"] == {
            "code": "METERKEY",
            "name": "Meter Access Key",
            "classification": "LUMENPORT_SEALED",
            "personalInformation": False,
            "directIdentifier": False,
            "enhancement": "ROTATE_QUARTERLY",
        }

    def test_a_fabricated_alias_id_is_not_resolvable(self):
        """
        With aliasing on, the index also carries invented technical spellings of each entry
        under synthetic ids. Those exist to be RETRIEVED against, never to be named:
        resolving one would hand a caller an id that is not in their glossary and that
        changes meaning whenever the alias generator does.
        """
        matcher = build_api_matcher(
            config=MatchingConfig(results_per_field=5, dictionary_alias_count=4)
        )
        alias_ids = sorted(matcher._alias_owner)
        assert alias_ids, "aliasing produced no alias ids, so this test proves nothing"

        with client_for(matcher) as client:
            body = lookup(client, alias_ids[0], "LWP-0001")

        assert body["results"][alias_ids[0]] is None
        assert body["results"]["LWP-0001"] is not None


# =============================================================================
# DETERMINISM
# =============================================================================


class TestDeterminism:
    """This body is a governance artifact: it gets pasted into tickets and diffed."""

    def test_two_identical_requests_produce_identical_bytes(self, real_client):
        first = real_client.post("/api/v1/lookup", json={"ids": ["LWP-0005", "LWP-0001"]})
        second = real_client.post("/api/v1/lookup", json={"ids": ["LWP-0005", "LWP-0001"]})

        assert first.content == second.content

    def test_the_body_is_pure_ascii(self, real_client):
        """
        An accented business name travels as an escape, so the artifact is byte-stable
        through whichever console, log pipeline or code page it passes.
        """
        response = real_client.get("/api/v1/lookup/LWP-0005")

        response.content.decode("ascii")
        assert "\\u00c9" in response.text
        assert json.loads(response.text)["results"]["LWP-0005"]["businessName"] == (
            "Tariff Band (Étage)"
        )

    def test_real_bodies_validate_against_the_published_schema(self, real_client):
        """
        The models in `lookup.py` are documentation, attached through `responses=`. Nothing
        serialises through them, so without this the published schema would be free to drift
        away from what the service sends -- a worse lie than having no schema.
        """
        body = lookup(real_client, "LWP-0001", "LWP-0005", "NOT-A-TERM")

        LookupResponseView.model_validate(body)


# =============================================================================
# ADMISSION AND REFUSALS
# =============================================================================


class TestRefusals:
    """Everything this route will not answer, and what it says instead."""

    def test_duplicate_ids_are_refused_naming_them(self, real_client):
        response = real_client.post(
            "/api/v1/lookup", json={"ids": ["LWP-0001", "LWP-0002", "LWP-0001"]}
        )

        assert response.status_code == 422, response.text
        assert response.json()["error"]["details"]["duplicate_ids"] == ["LWP-0001"]

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_id_is_refused_rather_than_answered(self, real_client, blank):
        response = real_client.post("/api/v1/lookup", json={"ids": ["LWP-0001", blank]})

        assert response.status_code == 422, response.text
        assert response.json()["error"]["details"]["blank_positions"] == [1]

    def test_the_empty_single_route_path_is_refused(self, real_client):
        """`GET /api/v1/lookup/` matches the path converter with an empty id."""
        response = real_client.get("/api/v1/lookup/")

        assert response.status_code == 422, response.text

    def test_an_empty_id_list_is_refused_by_the_model(self, real_client):
        response = real_client.post("/api/v1/lookup", json={"ids": []})

        assert response.status_code == 422, response.text
        assert response.json()["error"]["details"]["violations"]

    def test_an_unknown_envelope_key_is_refused(self, real_client):
        """
        `extra="forbid"` on this envelope: it has one member and no optional knobs, so a key
        that is silently ignored has nothing in the response to show for it.
        """
        response = real_client.post(
            "/api/v1/lookup", json={"ids": ["LWP-0001"], "term_ids": ["LWP-0002"]}
        )

        assert response.status_code == 422, response.text

    def test_an_oversized_id_is_refused_rather_than_reported_as_a_miss(self, real_client):
        response = real_client.post(
            "/api/v1/lookup", json={"ids": ["x" * (MAX_DICTIONARY_ID_CHARS + 1)]}
        )

        assert response.status_code == 422, response.text
        assert response.json()["error"]["details"]["limit_chars"] == MAX_DICTIONARY_ID_CHARS

    def test_too_many_ids_is_413_naming_the_cap(self):
        limits = MatchServiceLimits(max_batch_fields=4)
        with client_for(build_api_matcher(), limits=limits) as client:
            response = client.post(
                "/api/v1/lookup", json={"ids": [f"LWP-{n:04d}" for n in range(5)]}
            )

        assert response.status_code == 413, response.text
        details = response.json()["error"]["details"]
        assert details == {"ids": 5, "limit": 4, "status_code": 413}

    def test_no_dictionary_is_503_naming_the_setting_to_change(self):
        with client_for(None) as client:
            response = client.post("/api/v1/lookup", json={"ids": ["LWP-0001"]})

        assert response.status_code == 503, response.text
        assert "NEXUS_API_DICTIONARY" in response.json()["error"]["message"]

    def test_a_matcher_with_no_entry_map_is_a_named_500(self):
        """
        `FakeMatcher` has no `_dictionary_entries`, which is exactly the drift this refuses:
        a lookup that answered "not found" for every id would report an empty glossary
        rather than a broken one.
        """
        with client_for(FakeMatcher()) as client:
            response = client.post("/api/v1/lookup", json={"ids": ["LWP-0001"]})

        assert response.status_code == 500, response.text
        assert response.json()["error"]["details"]["attribute"] == "_dictionary_entries"


class TestAdmissionControl:
    """The caps this plane shares with matching, and the one it deliberately does not."""

    def test_the_id_cap_is_the_batch_field_cap(self):
        """
        One knob for the operator. Pinned so that a lookup-specific cap cannot be introduced
        without this saying so, and so the relation the docstring claims is checked.
        """
        limits = MatchServiceLimits(max_batch_fields=7)
        with client_for(build_api_matcher(), limits=limits) as client:
            accepted = client.post("/api/v1/lookup", json={"ids": [f"MISS-{n}" for n in range(7)]})
            refused = client.post("/api/v1/lookup", json={"ids": [f"MISS-{n}" for n in range(8)]})

        assert accepted.status_code == 200, accepted.text
        assert refused.status_code == 413, refused.text

    def test_the_id_bound_admits_every_id_the_feedback_route_accepts(self):
        """
        A reviewer's verdict names a `chosenGovernanceId`, and that is the same identifier
        arriving from the other direction. A lookup that refused an id the feedback route
        recorded would be one service disagreeing with itself about how long its own ids may
        be, and the caller would have no way to tell which document to believe.
        """
        from nexus_matcher.presentation.api.schemas import _MAX_NAME

        assert MAX_DICTIONARY_ID_CHARS >= _MAX_NAME

    @pytest.mark.parametrize("max_batch_fields", [1, 7, 250, 1000])
    def test_the_declared_id_bounds_can_never_exceed_the_body_cap(self, max_batch_fields):
        """
        A byte cap below what this route's own bounds admit would refuse a body the schema
        says is legal -- the caller reads two documents from one service and they contradict
        each other. The two numbers are derived from different constants in different
        modules, so the relation is checked rather than assumed.
        """
        limits = MatchServiceLimits(max_batch_fields=max_batch_fields)
        # `{"ids":[` + n * (quote + 4 UTF-8 bytes per char + quote + comma) + `]}`.
        worst_case = 8 + max_batch_fields * (MAX_DICTIONARY_ID_CHARS * 4 + 3) + 2

        assert worst_case <= limits.body_byte_cap
        assert limits.body_byte_cap == worst_case_body_bytes(max_batch_fields)

    async def test_a_lookup_is_served_while_the_matching_pool_is_saturated(self):
        """
        THE ISOLATION PROPERTY. A lookup does a dict read per id; routing it through the
        bounded pool would let a burst of cheap lookups consume the permits that keep
        matching responsive -- the cheap route shedding the expensive one, which inverts
        what admission control is for.

        Driven with an in-process async client so both requests are tasks on one event loop:
        what is observed is the server's admission control, not the harness's threading.
        """

        class BlockingMatcher(FakeMatcher):
            """A matcher that hangs in `_match_fields` but can still resolve an id."""

            def __init__(self) -> None:
                super().__init__()
                self._dictionary_entries = {entry.id: entry for entry in GLOSSARY}
                self._governance = governance_vocabulary()

        matcher = BlockingMatcher()
        matcher.release.clear()
        limits = MatchServiceLimits(deadline_seconds=20.0, max_workers=1, max_queued=0)
        app = create_app(configure_logs=False, matcher=matcher, limits=limits, environ={})

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as ac:
            try:
                busy = asyncio.ensure_future(
                    ac.post("/api/v1/match", json={"fields": request_fields()})
                )
                assert await asyncio.to_thread(matcher.started.wait, 20.0)

                shed = await ac.post("/api/v1/match", json={"fields": request_fields()})
                assert shed.status_code == 503, "the pool was not actually saturated"

                served = await ac.post("/api/v1/lookup", json={"ids": ["LWP-0001"]})
                assert served.status_code == 200, served.text
                assert served.json()["results"]["LWP-0001"]["businessName"] == (
                    "Resident Full Name"
                )
            finally:
                matcher.release.set()

            assert (await busy).status_code == 200
