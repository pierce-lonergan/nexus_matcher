"""
tests.unit.presentation.api.test_error_envelope | Layer: TEST
One service, one error shape -- including the failures the framework raises for us.

## Relationships
# TESTS → presentation/api/errors :: http_exception_handler
# TESTS → presentation/api/app :: the health probes and the handler registration

`errors.py`'s module docstring says raising `HTTPException` "would give the same service
two error shapes, which is the sort of thing a client library ends up handling with a
string test". That is what shipped: a census against a live app found 404, 405 and both
health 503s answering `{"detail": ...}` while every `/api/v1/*` failure answered
`{"error": {...}}`. The service never *chose* the second shape -- Starlette raises
`HTTPException` for an unknown path and a wrong verb whether this repository likes it or
not, and nothing was registered to render it.

## The header is half the contract

`PUT /api/v1/match` is a 405 carrying `Allow: POST`, which is how a client discovers the
verb without reading a document. A handler that builds a bare `JSONResponse` drops it --
trading an envelope defect for a protocol one -- so the `Allow` assertion below is not a
detail, it is the reason the handler forwards `exc.headers` at all.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from tests.unit.presentation.api._support import FakeMatcher, request_fields

BODY = {"fields": request_fields()}


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A started app with a matcher, so only the failures under test are failing."""
    app = create_app(configure_logs=False, matcher=FakeMatcher(), environ={})
    with TestClient(app) as started:
        yield started


def envelope(payload: object) -> dict[str, object]:
    """The `{"error": {...}}` body, asserting it is the ONLY top-level key."""
    assert isinstance(payload, dict), payload
    assert set(payload) == {"error"}, payload
    error = payload["error"]
    assert isinstance(error, dict), error
    assert set(error) == {"code", "message", "details"}, error
    assert isinstance(error["code"], str) and error["code"].startswith("NEXUS-"), error
    assert error["message"], "an error envelope with no message is a shape, not an answer"
    return error


# =============================================================================
# THE FAILURES THE FRAMEWORK RAISES
# =============================================================================


def test_an_unknown_path_answers_in_the_service_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/no-such-route")

    assert response.status_code == 404, response.text
    error = envelope(response.json())
    assert error["details"] == {"status_code": 404}


def test_a_wrong_verb_gains_the_envelope_and_keeps_its_allow_header(client: TestClient) -> None:
    """
    The regression that makes this handler worth writing carefully.

    `Allow` is how a client learns the verb. A bare `JSONResponse` -- which is what the
    422 handler builds and the obvious thing to copy -- drops it, so this asserts the
    header and the body together; either one alone passes on a broken implementation.
    """
    response = client.put("/api/v1/match", json=BODY)

    assert response.status_code == 405, response.text
    assert "POST" in response.headers.get("allow", ""), dict(response.headers)
    envelope(response.json())


def test_the_readiness_503_answers_in_the_envelope(client: TestClient) -> None:
    """`/health/ready` is raised through `HTTPException` too, and it 503s for real."""
    with TestClient(create_app(configure_logs=False, environ={})) as unready:
        response = unready.get("/health/ready")

    assert response.status_code == 503, response.text
    envelope(response.json())


def test_the_startup_probe_503_answers_in_the_envelope() -> None:
    """
    The second health 503, reached by never running the lifespan -- which is exactly the
    window the probe exists to describe.
    """
    response = TestClient(create_app(configure_logs=False, environ={})).get("/health/startup")

    assert response.status_code == 503, response.text
    error = envelope(response.json())
    assert "start" in str(error["message"]).lower(), error


# =============================================================================
# THE CENSUS
# =============================================================================


def test_no_failure_this_service_can_produce_answers_with_detail(client: TestClient) -> None:
    """
    The census, run as a gate.

    Each row is a failure a caller can provoke over HTTP. `detail` is FastAPI's own
    envelope; one of these answering with it is the defect, and the list is here so the
    next endpoint that raises `HTTPException` is covered without anybody remembering to
    add a test.
    """
    unready = TestClient(create_app(configure_logs=False, environ={}))
    responses = {
        "404": client.get("/no-such-route"),
        "405": client.put("/api/v1/match", json=BODY),
        "422": client.post("/api/v1/match", json={"fields": [{"name": 5}]}),
        "readiness 503": unready.get("/health/ready"),
        "startup 503": unready.get("/health/startup"),
    }

    for label, response in responses.items():
        assert response.status_code >= 400, f"{label} did not fail: {response.text}"
        payload = response.json()
        assert "detail" not in payload, f"{label} answered in FastAPI's envelope: {payload}"
        envelope(payload)


def test_body_validation_keeps_its_violations(client: TestClient) -> None:
    """
    The 422 is NOT collateral. It already had the envelope, and it carries the list that
    says what is wrong -- a unifying handler that swallowed that would be a downgrade.
    """
    response = client.post("/api/v1/match", json={"fields": [{"name": 5}]})

    assert response.status_code == 422, response.text
    error = envelope(response.json())
    details = error["details"]
    assert isinstance(details, dict)
    assert details["violations"], details


def test_the_documentation_surfaces_still_answer(client: TestClient) -> None:
    """
    A handler registered against `StarletteHTTPException` sits under `/docs` and
    `/openapi.json` too. Both are 200s, and a generated client starts at the second one.
    """
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


# =============================================================================
# THE ENVELOPE IS NOT AN ACCIDENT
# =============================================================================


def test_the_readiness_503_names_what_is_red(client: TestClient) -> None:
    """
    `details.components` is the diagnostic. Before it, an operator whose dictionary failed
    to load got the string "Service not ready" and no way to tell which component was.
    """
    with TestClient(create_app(configure_logs=False, environ={})) as unready:
        response = unready.get("/health/ready")

    details = envelope(response.json())["details"]
    assert isinstance(details, dict)
    assert details["components"] == {"api": True, "config": True, "matcher": False}
