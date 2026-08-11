"""
tests.unit.presentation.api.test_error_contract | Layer: TEST
The failure path is published, and the published version is what the service really sends.

## Relationships
# TESTS → presentation/api/schemas  :: ErrorResponse, ErrorDetail, ExplainView
# TESTS → presentation/api/matching :: the `responses=` table on both match routes

`schemas.py`'s own docstring says the response models are DOCUMENTATION, kept honest by
validating real bodies against them. Until now only the 200 was: `/openapi.json` described
zero error bodies, so a Java client generated a typed success DTO and a bare `Map` for
every way the request can fail -- which is the half a caller integrating against a new
endpoint spends most of their time in.

Two directions are asserted here, and both are needed. The spec must NAME the error body
(a generated client reads the spec, not this docstring), and real error bodies must
VALIDATE against it (a schema nothing checks is free to drift into a lie with a build step
behind it). All five statuses satisfy the model on day one, so these fire only on drift.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.limits import MatchServiceLimits
from nexus_matcher.presentation.api.schemas import ErrorResponse, ExplainView
from nexus_matcher.shared.types.base import MatchDecision
from tests.unit.presentation.api._support import FakeMatcher, request_fields

# Every failure `errors.py` documents by status. 500 is in this list and was missing from
# the published table, which is the status a client is least able to guess.
DOCUMENTED_FAILURES = ("413", "422", "500", "503", "504")

MATCH_ROUTES = ("/api/v1/match", "/api/v1/match/batch")

# Every route with a failure in the published spec. `/health` and `/health/live` are absent
# because they answer 200 unconditionally -- `/health` reports `status: "degraded"` in a
# 200 body rather than refusing, which is a deliberate choice `create_health_router`
# documents, not a missing entry.
ROUTES_THAT_PUBLISH_FAILURES = {
    "/api/v1/match",
    "/api/v1/match/batch",
    "/api/v1/feedback",
    "/health/ready",
    "/health/startup",
}


def client_for(matcher: object | None, limits: MatchServiceLimits | None = None) -> TestClient:
    return TestClient(create_app(configure_logs=False, matcher=matcher, limits=limits, environ={}))


def post_match(client: TestClient, **body: object):
    payload: dict[str, object] = {"fields": request_fields()}
    payload.update(body)
    return client.post("/api/v1/match", json=payload)


# =============================================================================
# THE PUBLISHED SPEC
# =============================================================================


@pytest.fixture
def spec() -> dict:
    with client_for(FakeMatcher()) as client:
        return client.get("/openapi.json").json()


def test_every_documented_failure_publishes_the_error_body(spec):
    """
    A description is not a schema.

    The table supplied prose only, so `/openapi.json` said a 413 could happen and nothing
    at all about what arrives in the body. A generated client cannot deserialise prose.
    """
    for route in MATCH_ROUTES:
        responses = spec["paths"][route]["post"]["responses"]
        for status in DOCUMENTED_FAILURES:
            assert status in responses, f"{route} publishes no {status} at all"
            body = responses[status].get("content", {})
            assert "ErrorResponse" in json.dumps(body), (
                f"{route} {status} is described but its body is not typed: {responses[status]!r}"
            )


def test_no_route_publishes_an_untyped_failure(spec):
    """
    The same standard as the test above, asked of the WHOLE spec instead of two routes.

    Attaching the model route by route is how `/api/v1/feedback` was left behind: both
    match routes shared one `_ERROR_RESPONSES` table and got it, feedback declared its own
    422 and 503 with descriptions only, and a client generated from the one spec came out
    with a DTO for every way a match can fail and a `Map` for every way a verdict can. The
    health 503 -- the answer `/health/ready` exists to give -- was published nowhere at all.

    So the assertion is over `spec["paths"]`, which a new route joins by existing. The
    expected set of failing routes is spelled out for the same reason: a route added with
    NO published failure would otherwise satisfy a walk over its zero failures.
    """
    untyped: list[str] = []
    publishing: set[str] = set()

    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            for status, response in operation.get("responses", {}).items():
                if status.startswith("2"):
                    continue
                publishing.add(path)
                if "ErrorResponse" not in json.dumps(response.get("content", {})):
                    untyped.append(f"{method.upper()} {path} -> {status}")

    assert not untyped, f"published failures with no body a client can generate: {untyped}"
    assert publishing == ROUTES_THAT_PUBLISH_FAILURES, (
        "a route's published failures changed. If this is a new route, add it here after "
        f"attaching ErrorResponse to its failures: {sorted(publishing)}"
    )


def test_the_published_422_is_this_services_envelope_and_not_fastapis(spec):
    """
    `app.py` installs `validation_exception_handler`, so this service sends
    `{"error": {...}}` and never FastAPI's `{"detail": [...]}`. Publishing
    `HTTPValidationError` while "fixing" the missing schema would trade one gap for an
    actively false schema, which is worse: the client would generate against a shape the
    server cannot produce.
    """
    for route in MATCH_ROUTES:
        published = json.dumps(spec["paths"][route]["post"]["responses"]["422"])
        assert "HTTPValidationError" not in published, published
        assert "ErrorResponse" in published


def test_the_error_body_is_typed_rather_than_an_open_map():
    """
    The load-bearing half of publishing it at all.

    `error: dict[str, Any]` renders as `{"type": "object", "additionalProperties": true}`,
    so attaching THAT to the routes would have moved the model into the spec and bought a
    generated client nothing -- it still gets a `Map` and still learns the three keys from
    a human. The sub-model is what makes the schema worth generating from.
    """
    schema = ErrorResponse.model_json_schema()
    detail = schema["$defs"]["ErrorDetail"]

    assert set(detail["properties"]) == {"code", "message", "details"}
    assert detail["properties"]["code"]["type"] == "string"
    assert detail["properties"]["message"]["type"] == "string"
    # `details` stays open on purpose: its keys vary by failure (`limit`, `violations`,
    # `deadline_seconds`, `duplicate_paths`, ...) and pinning them would either be a lie
    # or force every new failure mode to change the published schema.
    assert detail["properties"]["details"]["type"] == "object"


def test_decision_is_published_as_the_librarys_own_enum(spec):
    """
    A generated client should get an enum with three members, not `String`.

    Typed as `MatchDecision` rather than a hand-written `Literal`, so the published values
    and the values `_candidate_payload` emits are the same object. A second copy of the
    list is exactly the drift this repository keeps a `drift()` helper for.
    """
    published = spec["components"]["schemas"]["MatchDecision"]["enum"]
    assert set(published) == {member.value for member in MatchDecision}


def test_explain_is_typed_without_naming_the_five_score_components(spec):
    """
    `_verify_reproducible`'s docstring anticipates "a sixth weighted signal this file
    knows nothing about". Naming today's five components as fields would publish a schema
    that the sixth makes false, and a client generated from it would silently drop the new
    signal -- so the maps stay open and the KEYS are the contract, not the schema.
    """
    assert set(ExplainView.model_fields) == {"scores", "weights", "absoluteCosine"}

    properties = spec["components"]["schemas"]["ExplainView"]["properties"]
    assert set(properties) == {"scores", "weights", "absoluteCosine"}
    # An OPEN map of number, not an object with five fixed members: that is what lets a
    # sixth signal appear without the published schema becoming false.
    assert properties["scores"]["additionalProperties"] == {"type": "number"}
    assert properties["weights"]["additionalProperties"] == {"type": "number"}


# =============================================================================
# THE BODIES THE SERVICE REALLY SENDS
# =============================================================================


def _real_error_bodies() -> dict[int, dict]:
    """One genuine response per documented failure, produced through the real ASGI app."""
    bodies: dict[int, dict] = {}

    with client_for(FakeMatcher()) as client:
        too_many = [{"name": f"c_{i}", "path": f"t.c_{i}"} for i in range(101)]
        bodies[413] = client.post("/api/v1/match", json={"fields": too_many}).json()
        bodies[422] = client.post("/api/v1/match", json={"fields": []}).json()

    with client_for(FakeMatcher(raises=RuntimeError("onnxruntime session died"))) as client:
        bodies[500] = post_match(client).json()

    with client_for(None) as client:
        bodies[503] = post_match(client).json()

    slow = FakeMatcher(delay_seconds=1.0)
    limits = MatchServiceLimits(deadline_seconds=0.05, max_workers=2, max_queued=4)
    with client_for(slow, limits) as client:
        bodies[504] = post_match(client).json()

    return bodies


def test_real_error_bodies_validate_against_the_published_model():
    """
    The same standard the 200 is already held to, applied to the failure path.

    All five satisfy the model today, so this passes on day one and fires only when a
    handler starts sending something the spec does not describe -- which is the failure
    mode a published schema has, and the reason `schemas.py` calls its response models
    documentation rather than serialisers.
    """
    bodies = _real_error_bodies()
    assert set(bodies) == {413, 422, 500, 503, 504}

    for status, body in bodies.items():
        assert set(body) == {"error"}, f"{status} answered outside the one envelope: {body!r}"
        parsed = ErrorResponse.model_validate(body)
        assert parsed.error.code.startswith("NEXUS-"), (status, parsed.error.code)
        assert parsed.error.message
        assert parsed.error.details["status_code"] == status


def test_the_routes_that_just_gained_a_schema_really_send_it(tmp_path):
    """
    Publishing a schema and sending something else is worse than publishing nothing: the
    client generated against it compiles and then fails at runtime. Feedback and the
    readiness probe are asserted here because they are the routes whose failures were
    published for the first time, so nothing had ever compared the two.

    The 500 is produced the way `test_feedback_endpoint` produces it -- a parent path that
    is a FILE, so the append raises OSError -- rather than by patching the recorder, which
    would assert the handler against a failure only the test can cause.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file cannot also be a parent directory", encoding="utf-8")
    verdict = {
        "field": "delivery.grower_name",
        "doc": "Registered name of the grower",
        "chosenGovernanceId": "TCG-0001",
        "suggestedGovernanceId": "TCG-0004",
        "wasCorrect": False,
        "reviewer": "weighbridge.control",
        "ts": "2026-08-10T09:15:00Z",
    }

    bodies: dict[str, dict] = {}
    with client_for(FakeMatcher()) as client:
        # No feedback path configured, so recording is unavailable rather than broken.
        bodies["feedback 503"] = client.post("/api/v1/feedback", json=verdict).json()
        bodies["feedback 422"] = client.post("/api/v1/feedback", json={}).json()

    unwritable = create_app(
        configure_logs=False,
        matcher=FakeMatcher(),
        feedback_path=str(blocker / "feedback.jsonl"),
        environ={},
    )
    with TestClient(unwritable) as client:
        bodies["feedback 500"] = client.post("/api/v1/feedback", json=verdict).json()

    # No matcher and no opt-out, so the readiness gate is genuinely red.
    with client_for(None) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503, response.text
        bodies["health 503"] = response.json()

    for label, body in bodies.items():
        assert set(body) == {"error"}, f"{label} answered outside the one envelope: {body!r}"
        parsed = ErrorResponse.model_validate(body)
        assert parsed.error.code.startswith("NEXUS-"), (label, parsed.error.code)
        assert parsed.error.message
        assert parsed.error.details["status_code"] == int(label.split()[-1])
