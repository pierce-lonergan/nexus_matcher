"""
tests.unit.presentation.api.test_match_endpoint | Layer: TEST
The wire contract of POST /api/v1/match and /api/v1/match/batch.

## Relationships
# TESTS → presentation/api/matching :: projection, conservation, determinism
# TESTS → presentation/api/schemas  :: request validation and the published response model

Three properties are load-bearing here, and each is the negation of a defect this
repository has already paid for:

  CONSERVATION  every input field comes back, under the caller's own path, even when
                nothing matched it. A field that vanishes from a result map inherits no
                governance and nothing raises -- NM-0005, whose only symptom was a count
                nobody had reason to check.
  DETERMINISM   two identical requests produce byte-identical bodies. This response is a
                governance artifact: it gets pasted into tickets and diffed, and a body
                whose key order wanders makes every diff unreadable.
  PASSTHROUGH   the governance id is always populated and the protection class is either
                a complete object or an explicit null -- never a missing key, because
                "this entry has no class" and "this response forgot" must not look alike.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.matching import _to_schema_field
from nexus_matcher.presentation.api.schemas import FieldSpec, MatchResponseView
from tests.unit.presentation.api._support import (
    FICTIONAL_VOCABULARY,
    GLOSSARY,
    STAND_IN_CONFIDENCE,
    FakeMatcher,
    build_api_matcher,
    request_fields,
)

# The exact keys of a candidate, in the exact order the contract fixes them. A literal,
# not something read back off the response: an expectation derived from the code under
# test is an identity, and an identity holds just as well when both sides are wrong.
CANDIDATE_KEYS = (
    "rank",
    "governanceId",
    "businessName",
    "definition",
    "domain",
    "governance",
    "confidence",
    "decision",
)

GOVERNANCE_KEYS = (
    "code",
    "name",
    "classification",
    "personalInformation",
    "directIdentifier",
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


@pytest.fixture
def fake_client():
    """
    A matcher whose scores this file chose, so `explain` can be checked against a
    hand-computed constant instead of against whatever the encoder produced.
    """
    with client_for(FakeMatcher()) as client:
        yield client


def post_match(client: TestClient, **body: object):
    payload = {"fields": request_fields()}
    payload.update(body)
    return client.post("/api/v1/match", json=payload)


# =============================================================================
# CONSERVATION
# =============================================================================


class TestConservation:
    """Every field in, exactly one entry out, addressable by the caller's own name."""

    def test_every_input_field_appears_in_the_output(self, real_client):
        """
        COUNT and ADDRESS together, through the real matcher.

        The expected keys are the paths this test SENT, not the keys the response happens
        to carry. Reading them back would make this assertion an identity that survives
        the exact defect it exists to catch.
        """
        sent = [field["path"] for field in request_fields()]
        response = post_match(real_client)

        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert list(results) == sent, (
            f"sent {len(sent)} fields and got {len(results)} back. A field missing from "
            f"the response inherits no governance and nothing said so."
        )

    def test_a_field_nothing_matched_still_comes_back_with_an_empty_list(self):
        """
        The dangerous half of conservation.

        Dropping a field because it has no candidates is the tempting bug: the response
        still parses, every key present is correct, and the caller simply never hears
        about the column. It is indistinguishable, to them, from a column they forgot to
        send -- so they never go looking for it, and it ships unclassified.
        """
        matcher = FakeMatcher(empty_paths=("misc.zzz_unmatchable",))
        with client_for(matcher) as client:
            results = post_match(client).json()["results"]

        assert "misc.zzz_unmatchable" in results, (
            "the field with no candidates was dropped from the response entirely"
        )
        assert results["misc.zzz_unmatchable"] == []

    def test_duplicate_paths_are_refused_rather_than_collapsed(self, real_client):
        """
        A map cannot hold two entries under one key, so this has to be a 4xx.

        Accepting it would silently drop one of the two columns -- NM-0005 reproduced
        through the front door. The message must name the offending path, because the
        caller's next move is to fix their own request.
        """
        duplicated = [
            {"name": "usage_litres", "path": "meter.usage_litres"},
            {"name": "usage_litres_v2", "path": "meter.usage_litres"},
        ]
        response = real_client.post("/api/v1/match", json={"fields": duplicated})

        assert response.status_code == 422, response.text
        body = response.json()
        assert "meter.usage_litres" in body["error"]["message"]
        assert body["error"]["details"]["duplicate_paths"] == ["meter.usage_litres"]

    def test_a_matcher_that_loses_a_field_produces_a_500_not_a_short_answer(self):
        """
        The invariant is checked, not assumed.

        If the layer below ever returns fewer results than fields, the honest answer is a
        refusal. Returning the short map would be a 200 that quietly under-classifies,
        which is the single outcome this endpoint must never produce.
        """
        with client_for(FakeMatcher(drop_last_field=True)) as client:
            response = post_match(client)

        assert response.status_code == 500, response.text
        details = response.json()["error"]["details"]
        assert details["fields_in"] == len(request_fields())
        assert details["results_out"] == len(request_fields()) - 1

    def test_a_result_the_caller_cannot_address_produces_a_500(self):
        """
        ADDRESS, checked separately from COUNT.

        The right NUMBER of results under keys the caller never sent is not conservation:
        they asked for `meter.usage_litres` and got a KeyError from a response that does
        contain their field. That really happened here, when the flattening rewrite began
        keying results by the reconstructed dotted path.
        """
        with client_for(FakeMatcher(mangle_keys=True)) as client:
            response = post_match(client)

        assert response.status_code == 500, response.text
        details = response.json()["error"]["details"]
        assert details["expected_path"] == "account.resident_nm"
        assert details["actual_key"] == "account.resident_nm.derived"

    def test_results_computed_for_another_field_produce_a_500(self):
        """
        IDENTITY, and the most expensive of the three failures.

        A count that holds while the mapping is wrong means the caller sees the right
        number of rows and one of them carries another column's matches -- so a column is
        not missing a classification, it has acquired somebody else's.

        This test does NOT distinguish `is` from `==`: the fixture's two fields differ in
        content, so both operators catch it. That gap is real and is recorded in
        `_project_results`; it is unreachable from the wire because duplicate paths are
        refused first.
        """
        with client_for(FakeMatcher(serve_wrong_field=True)) as client:
            response = post_match(client)

        assert response.status_code == 500, response.text
        details = response.json()["error"]["details"]
        assert details["path"] == "account.resident_nm"
        assert details["computed_for"] == "meter.meter_key"


# =============================================================================
# DETERMINISM
# =============================================================================


class TestDeterminism:
    """Byte-identical bodies for identical requests, in the caller's own order."""

    def test_two_identical_requests_produce_byte_identical_bodies(self, real_client):
        first = post_match(real_client)
        second = post_match(real_client)

        assert first.status_code == 200
        assert first.content == second.content, (
            "identical requests produced different bytes, so no two responses can be "
            "diffed against each other"
        )

    def test_key_order_follows_the_request_and_is_not_sorted(self, real_client):
        """
        Reversing the request must reverse the response.

        A sorted response would also be deterministic, and would also pass a
        byte-identity check -- so that check alone cannot tell "stable" from "in the
        caller's order". This one can: the fixture's paths are not in sorted order either
        way round, so both a sort and a fixed order fail it.
        """
        forward = [field["path"] for field in request_fields()]
        reverse = list(reversed(request_fields()))

        response = real_client.post("/api/v1/match", json={"fields": reverse})

        assert response.status_code == 200, response.text
        assert list(response.json()["results"]) == list(reversed(forward))

    def test_the_body_is_pure_ascii_even_for_an_accented_business_name(self, real_client):
        """
        A governance artifact travels through consoles, logs and ticket systems, and this
        repository has already been bitten by legacy code pages mangling one. Pure ASCII
        on the wire makes the body byte-stable everywhere; `json.loads` gives the original
        string back regardless.
        """
        response = post_match(real_client, top_k=5)

        assert response.status_code == 200, response.text
        response.content.decode("ascii")  # raises if any byte is not ASCII

        accented = [
            candidate
            for candidates in response.json()["results"].values()
            for candidate in candidates
            if candidate["governanceId"] == "LWP-0005"
        ]
        assert accented, "the entry with the accented business name never appeared"
        assert accented[0]["businessName"] == "Tariff Band (Étage)"
        assert b"\\u00c9" in response.content or b"\\u00e9" in response.content


# =============================================================================
# GOVERNANCE PASSTHROUGH
# =============================================================================


class TestGovernancePassthrough:
    """
    The two fields this whole feature exists to deliver, end to end.

    Driven through the REAL matcher and a REAL `GovernanceVocabulary` loaded from the
    caller-supplied JSON shape, so what is exercised is the whole chain -- vocabulary file
    to `DictionaryEntry.governance_code` to `MatchResult.governance` to the wire -- rather
    than this module's projection of a hand-built object.
    """

    def test_candidate_keys_are_exactly_the_contract_in_order(self, real_client):
        candidate = post_match(real_client).json()["results"]["account.resident_nm"][0]
        assert tuple(candidate) == CANDIDATE_KEYS

    def test_every_declared_code_passes_through_complete(self, real_client):
        """
        The whole vocabulary, pinned against its own declared values rather than against
        whatever the response returned. Every member matters on its own:
        `personalInformation` false where it should be true is a field that quietly stops
        being treated as personal data.

        Asserted over the RANK-1 candidates, one request field per glossary entry, so all
        four codes are genuinely exercised. Sampling whatever came back would drift into
        asserting mostly about nulls: below rank 1 this fixture is nearly all REJECT, and
        a REJECT carries no class by design.
        """
        top = {
            candidates[0]["governanceId"]: candidates[0]
            for candidates in post_match(real_client).json()["results"].values()
            if candidates
        }

        for code, expected in FICTIONAL_VOCABULARY.items():
            entry = next(e for e in GLOSSARY if e.governance_code == code)
            assert entry.id in top, f"{entry.id} never ranked first, so {code} proved nothing"
            assert tuple(top[entry.id]["governance"]) == GOVERNANCE_KEYS
            assert top[entry.id]["governance"] == {
                "code": expected.code,
                "name": expected.name,
                "classification": expected.classification,
                "personalInformation": expected.personal_information,
                "directIdentifier": expected.direct_identifier,
            }

    def test_an_entry_with_no_code_serialises_governance_as_an_explicit_null(self, real_client):
        """
        `null`, not a missing key. A client reading `body.governance` must be able to tell
        "no protection class" from "this response did not say", because only one of those
        is safe to act on.

        Read off a candidate that was NOT rejected. A rejected match also carries no class
        -- the domain clears it, so a novel field inherits nothing -- so asserting against
        one would prove the reject rule and say nothing about the uncoded-entry rule.
        """
        uncoded = next(e for e in GLOSSARY if e.governance_code is None)
        candidate = post_match(real_client).json()["results"]["billing.tariff_band"][0]

        assert candidate["governanceId"] == uncoded.id
        assert candidate["decision"] != "REJECT"
        assert "governance" in candidate, (
            "the key was omitted rather than nulled, so a client cannot tell 'no class' "
            "from 'not reported'"
        )
        assert candidate["governance"] is None

    def test_every_candidate_carries_a_populated_governance_id(self, real_client):
        """
        Not only rank 1. A reviewer choosing between rank 1 and rank 2 needs both ids, and
        a blank one on any rank is a candidate that inherits nothing while looking like
        it does.
        """
        results = post_match(real_client, top_k=5).json()["results"]
        ids = [c["governanceId"] for candidates in results.values() for c in candidates]

        assert ids, "no candidates at all, so nothing was proven about the id"
        assert all(ids), "a candidate came back with an empty governanceId"
        assert set(ids) <= {entry.id for entry in GLOSSARY}

    def test_governance_is_resolved_below_rank_one_too(self, real_client):
        """
        A response that resolved the class for rank 1 only would look complete and force a
        second lookup for exactly the comparison a reviewer is making -- "is rank 2 a
        direct identifier too?" -- which is usually the deciding fact.

        `misc.zzz_unmatchable` is the field that produces a second NON-rejected candidate
        on this fixture, which is what makes the assertion about rank 2 rather than about
        the reject rule.
        """
        candidates = post_match(real_client, top_k=5).json()["results"]["misc.zzz_unmatchable"]
        assert len(candidates) > 1, "only one candidate, so nothing was proven about rank 2"
        assert candidates[1]["governance"] is not None, (
            "rank 2 came back with no protection class while rank 1 had one"
        )
        assert candidates[1]["governance"]["code"] in FICTIONAL_VOCABULARY


# =============================================================================
# EXPLAIN
# =============================================================================


class TestExplain:
    """Present only when asked, and self-checking when present."""

    def test_explain_is_absent_unless_requested(self, fake_client):
        candidate = post_match(fake_client).json()["results"]["account.resident_nm"][0]
        assert "explain" not in candidate

    def test_explain_reproduces_the_confidence_it_ships_with(self, fake_client):
        """
        The auditor's arithmetic, done here on the EMITTED numbers.

        The expected confidence is hand-computed in `_support.STAND_IN_CONFIDENCE` from
        the five signals and the shipped weights, so this asserts an absolute value rather
        than recomputing the sum with the same table the endpoint used.
        """
        candidate = post_match(fake_client, explain=True).json()["results"]["account.resident_nm"][
            0
        ]

        assert candidate["confidence"] == pytest.approx(STAND_IN_CONFIDENCE, abs=1e-6)
        explain = candidate["explain"]
        assert tuple(explain) == ("scores", "weights", "absoluteCosine")
        assert tuple(explain["scores"]) == (
            "fusedRetrieval",
            "lexical",
            "editDistance",
            "type",
            "domain",
        )
        assert tuple(explain["weights"]) == tuple(explain["scores"])
        recomputed = sum(
            explain["scores"][key] * weight for key, weight in explain["weights"].items()
        )
        assert recomputed == pytest.approx(STAND_IN_CONFIDENCE, abs=1e-5)
        assert explain["absoluteCosine"] == pytest.approx(0.7657, abs=1e-6)

    def test_explain_carries_the_live_matchers_weights_not_the_shipped_defaults(self):
        """
        A deployment that tuned its weights must get a response that reproduces ITS
        numbers. Reading `MatchingConfig()` instead would emit five plausible constants
        that do not explain the confidence next to them -- self-consistently wrong, which
        is the worst state for an audit surface.
        """
        from nexus_matcher.application.use_cases.match_schema import MatchingConfig

        tuned = MatchingConfig(
            semantic_weight=0.60,
            lexical_weight=0.10,
            edit_distance_weight=0.10,
            type_weight=0.10,
            domain_weight=0.10,
        )
        with client_for(FakeMatcher(config=tuned)) as client:
            response = post_match(client, explain=True)

        # The stand-in confidence was computed with the SHIPPED weights, so a tuned
        # matcher makes the response fail its own reproducibility check -- which is the
        # refusal working, not a flaw in the fixture.
        assert response.status_code == 500, response.text
        assert "reproducible confidence" in response.json()["error"]["message"]

    def test_a_genuinely_tuned_matcher_still_answers_and_its_arithmetic_closes(self):
        """
        The other direction, and the reason the test above is not evidence that tuned
        deployments 500.

        A REAL matcher with non-default weights produces confidences that ARE its own
        weighted sum, so the response is served and reproduces itself. Without this, the
        refusal above would look like a rule against tuning rather than a check on
        consistency -- and somebody would eventually loosen it for the wrong reason.
        """
        from nexus_matcher.application.use_cases.match_schema import MatchingConfig

        tuned = MatchingConfig(
            semantic_weight=0.60,
            lexical_weight=0.10,
            edit_distance_weight=0.10,
            type_weight=0.10,
            domain_weight=0.10,
            results_per_field=3,
        )
        with client_for(build_api_matcher(config=tuned)) as client:
            response = post_match(client, top_k=3, explain=True)

        assert response.status_code == 200, response.text
        candidates = [
            candidate for values in response.json()["results"].values() for candidate in values
        ]
        assert candidates
        for candidate in candidates:
            explain = candidate["explain"]
            assert explain["weights"] == {
                "fusedRetrieval": 0.6,
                "lexical": 0.1,
                "editDistance": 0.1,
                "type": 0.1,
                "domain": 0.1,
            }
            recomputed = sum(
                explain["scores"][key] * weight for key, weight in explain["weights"].items()
            )
            assert min(max(recomputed, 0.0), 1.0) == pytest.approx(
                candidate["confidence"], abs=1e-5
            )


# =============================================================================
# REQUEST VALIDATION
# =============================================================================


class TestRequestValidation:
    """A 4xx that says what is wrong, in the same envelope as every other failure."""

    @pytest.mark.parametrize(
        ("body", "because"),
        [
            ({"fields": []}, "an empty request reads as 'nothing to classify'"),
            ({"fields": [{"path": "a.b"}]}, "name is required"),
            ({"fields": [{"name": "a", "documentation": "x"}]}, "unknown key"),
            ({"fields": [{"name": "a"}], "top_k": 0}, "top_k below 1"),
            ({"fields": [{"name": "a"}], "explain": "yes please"}, "explain is a bool"),
            ({"fields": "account.resident_nm"}, "fields is a list"),
        ],
    )
    def test_a_malformed_body_is_422_with_the_reason(self, real_client, body, because):
        response = real_client.post("/api/v1/match", json=body)

        assert response.status_code == 422, f"{because}: {response.text}"
        error = response.json()["error"]
        assert error["code"] == "NEXUS-8004"
        assert error["details"]["violations"], "a 422 that does not say what is wrong"

    def test_an_unknown_field_key_is_named_in_the_violation(self, real_client):
        """
        A silently ignored `documentation` would drop the column comment, and the comment
        is real retrieval signal -- the caller would get worse matches with no indication
        why. Same standard as a mistyped matching-config key.
        """
        response = real_client.post(
            "/api/v1/match", json={"fields": [{"name": "a", "documentation": "x"}]}
        )

        violations = response.json()["error"]["details"]["violations"]
        assert any("documentation" in v["location"] for v in violations), violations

    def test_top_k_above_the_servers_results_per_field_is_refused_with_the_cap(self):
        """
        Truncating silently would tell the caller there were only N candidates when the
        server simply never looked for more.
        """
        with client_for(build_api_matcher(results_per_field=3)) as client:
            response = post_match(client, top_k=5)

        assert response.status_code == 422, response.text
        assert response.json()["error"]["details"]["results_per_field"] == 3

    def test_top_k_truncates_to_the_number_asked_for(self, real_client):
        results = post_match(real_client, top_k=2).json()["results"]
        assert all(len(candidates) <= 2 for candidates in results.values())

    def test_path_defaults_to_name_when_omitted(self, real_client):
        response = real_client.post("/api/v1/match", json={"fields": [{"name": "usage"}]})
        assert response.status_code == 200, response.text
        assert list(response.json()["results"]) == ["usage"]


# =============================================================================
# FIELD CAPS
# =============================================================================


class TestFieldCaps:
    """/match and /match/batch differ only in how many fields they accept."""

    @staticmethod
    def _fields(count: int) -> list[dict[str, str]]:
        return [{"name": f"col_{i}", "path": f"t.col_{i}"} for i in range(count)]

    def test_the_batch_route_accepts_the_documented_chunk_size(self, real_client):
        response = real_client.post(
            "/api/v1/match/batch", json={"fields": self._fields(250), "top_k": 1}
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == 250

    def test_over_the_cap_is_413_naming_the_limit(self, real_client):
        response = real_client.post(
            "/api/v1/match/batch", json={"fields": self._fields(251), "top_k": 1}
        )
        assert response.status_code == 413, response.text
        assert response.json()["error"]["details"]["limit"] == 250

    def test_the_narrow_route_has_the_narrow_cap(self, real_client):
        response = real_client.post("/api/v1/match", json={"fields": self._fields(101), "top_k": 1})
        assert response.status_code == 413, response.text
        assert response.json()["error"]["details"]["limit"] == 100


# =============================================================================
# TRANSLATION AND PUBLISHED SCHEMA
# =============================================================================


def test_the_parent_path_is_derived_from_the_callers_dotted_path():
    """
    Hierarchical context is the largest single accuracy factor measured on this task
    (+20 points of P@1). Dropping the derivation here would cost the caller that with no
    error and no visible symptom other than worse matches, so it is pinned by value.
    """
    field = _to_schema_field(FieldSpec(name="email", path="customer.contact.email", doc="d"))

    assert field.full_path == "customer.contact.email"
    assert field.parent_path == "customer.contact"
    assert field.description == "d"
    assert field.source_metadata["flattened_name"] == "customer.contact.email"


def test_a_flat_path_yields_an_empty_parent_rather_than_a_fabricated_one():
    field = _to_schema_field(FieldSpec(name="email", path="email"))
    assert field.parent_path == ""


def test_the_published_openapi_model_describes_what_the_service_really_sends(real_client):
    """
    The response models in `schemas.py` are documentation, and documentation drifts.

    A Java client generates from `/openapi.json`, so a published schema that no longer
    matches the body is a lie with a build step behind it. Validating a REAL response
    against the published model is what stops the two diverging: this is the same class
    as `check_doc_numbers`, applied to a schema instead of a sentence.
    """
    body = post_match(real_client, explain=True).json()
    MatchResponseView.model_validate(body)

    schema = real_client.get("/openapi.json").json()
    match_200 = schema["paths"]["/api/v1/match"]["post"]["responses"]["200"]
    assert "MatchResponseView" in json.dumps(match_200), match_200
