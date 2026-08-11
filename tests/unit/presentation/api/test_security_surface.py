"""
tests.unit.presentation.api.test_security_surface | Layer: TEST
The service ships unauthenticated -- so it must not ADVERTISE a control it does not have.

## Relationships
# TESTS → presentation/api/app :: the published description and the CORS configuration

Two claims shipped in `/openapi.json`, which is the document a Java client is generated
from: "API key authentication can be enabled via the `NEXUS_API_KEY` environment
variable. Pass the key in the `X-API-Key` header", and a module header advertising rate
limiting. Neither is implemented -- `NEXUS_API_KEY` appeared exactly once in `src/`, in
that sentence, and `RateLimitError` is raised nowhere. With the variable set, an
unauthenticated `POST /api/v1/match` still answered 200 carrying a real protection class.

Under it sat `allow_origins=["*"]` with `allow_credentials=True`, so any page on any
origin could read the body of an endpoint that asks for no credentials -- and could
`POST /api/v1/feedback`, which fsyncs a reviewer verdict into the audit trail that
`feedback.py` calls "the property that makes it usable as evidence."

## Why the deletions are asserted, not just performed

Nothing in `tests/unit/presentation/` mentioned CORS, origins or API keys, so the wrong
configuration was invisible and the false sentence was free to write. A claim deleted
without a gate grows back; these tests are the gate. They assert the REPLACEMENT wording
too, because silence about authentication is what let the claim be invented in the first
place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from tests.unit.presentation.api._support import FakeMatcher, request_fields

REPO = Path(__file__).resolve().parents[4]
DEPLOYMENT = REPO / "docs" / "DEPLOYMENT.md"

BODY = {"fields": request_fields()}
ALLOWED = "https://ops.gravelbay.example"
STRANGER = "https://evil.example"


def app_with(environ: dict[str, str]):
    return create_app(configure_logs=False, matcher=FakeMatcher(), environ=environ)


def preflight(client: TestClient, origin: str, method: str = "POST"):
    return client.options(
        "/api/v1/match",
        headers={"Origin": origin, "Access-Control-Request-Method": method},
    )


# =============================================================================
# CORS IS CLOSED UNTIL AN OPERATOR OPENS IT
# =============================================================================


def test_no_cross_origin_caller_is_granted_anything_by_default() -> None:
    """
    The default is the one that ships, and it shipped reflecting every origin.

    Both halves are asserted because Starlette answers them separately: the preflight
    decides whether the browser sends the request at all, and the simple response decides
    whether the page may read the body.
    """
    with TestClient(app_with({})) as client:
        options = preflight(client, STRANGER)
        posted = client.post("/api/v1/match", json=BODY, headers={"Origin": STRANGER})

    assert "access-control-allow-origin" not in options.headers, dict(options.headers)
    assert "access-control-allow-origin" not in posted.headers, dict(posted.headers)
    assert posted.status_code == 200, "closing CORS must not break the server-to-server path"


def test_the_operator_names_the_origins_and_only_those_are_granted() -> None:
    with TestClient(app_with({"NEXUS_API_CORS_ORIGINS": ALLOWED})) as client:
        granted = preflight(client, ALLOWED)
        refused = preflight(client, STRANGER)

    assert granted.headers.get("access-control-allow-origin") == ALLOWED
    assert "access-control-allow-origin" not in refused.headers, dict(refused.headers)


def test_a_verb_this_service_does_not_serve_is_not_granted() -> None:
    """
    `allow_methods=["*"]` told a browser that DELETE was fine on every route. The service
    serves GET, POST and the preflight itself; nothing else should be advertised.

    Starlette answers a disallowed preflight with 400 and still stamps the origin header
    on it, so the assertion is on the verb list -- which is the field the browser reads
    to decide whether to send the request at all.
    """
    with TestClient(app_with({"NEXUS_API_CORS_ORIGINS": ALLOWED})) as client:
        granted = preflight(client, ALLOWED, "POST")
        refused = preflight(client, ALLOWED, "DELETE")

    assert granted.status_code == 200, granted.text
    assert granted.headers["access-control-allow-methods"] == "GET, OPTIONS, POST"
    assert refused.status_code == 400, refused.text


def test_credentials_with_a_wildcard_origin_is_refused_at_startup() -> None:
    """
    The combination that shipped. Starlette reflects the requesting origin rather than
    sending `*` when credentials are on, so "allow every origin" quietly becomes "allow
    THIS origin, with cookies" -- a config an operator cannot read off their own settings.

    Refused at construction, where every other impossible configuration in this service is
    refused, so it cannot be discovered from a browser console in production.
    """
    with pytest.raises(ValueError, match="NEXUS_API_CORS_ALLOW_CREDENTIALS"):
        app_with({"NEXUS_API_CORS_ORIGINS": "*", "NEXUS_API_CORS_ALLOW_CREDENTIALS": "true"})


def test_a_wildcard_without_credentials_is_still_allowed() -> None:
    """
    Deliberately public is a legitimate choice; the refusal above must not become a ban on
    `*`. What it bans is `*` plus credentials.
    """
    with TestClient(app_with({"NEXUS_API_CORS_ORIGINS": "*"})) as client:
        assert preflight(client, STRANGER).headers.get("access-control-allow-origin") == "*"


def test_an_unparseable_flag_is_refused_rather_than_read_as_off() -> None:
    """
    Same standard as `MatchServiceLimits.from_env`: a security setting that silently
    reverts to its default is a setting the operator believes they applied.
    """
    with pytest.raises(ValueError, match="NEXUS_API_CORS_ALLOW_CREDENTIALS"):
        app_with({"NEXUS_API_CORS_ORIGINS": ALLOWED, "NEXUS_API_CORS_ALLOW_CREDENTIALS": "maybe"})


# =============================================================================
# THE PUBLISHED CONTRACT
# =============================================================================


def published_description() -> str:
    with TestClient(app_with({})) as client:
        return client.get("/openapi.json").json()["info"]["description"]


def test_the_openapi_advertises_no_control_this_service_does_not_implement() -> None:
    """
    The exact strings that shipped, each of which a reader would act on.

    Retracting a claim by REPEATING it does not retract it -- the same rule
    `tests/packaging/test_documented_routes.py` had to learn three times about the
    matching endpoint. A developer skimming `/docs` for how to authenticate takes the same
    wrong answer from "we used to offer `X-API-Key`" as from "pass `X-API-Key`", so the
    replacement wording describes the retraction instead of quoting it.
    """
    description = published_description().lower()

    assert "x-api-key" not in description
    assert "nexus_api_key" not in description
    assert "rate limit" not in description and "rate-limit" not in description


def test_the_openapi_says_out_loud_that_it_ships_unauthenticated() -> None:
    """
    Deleting the false sentence is not enough. The claim was invented once against a
    silent document; a document that states the position is what stops it being invented
    again, which is the same reason `API_REFERENCE.md` keeps a "not implemented" table.
    """
    description = published_description().lower()

    assert "unauthenticated" in description
    assert "gateway" in description


def test_no_security_scheme_is_declared() -> None:
    """
    The half a generated client actually reads. The description could say anything; a
    Java client is built from `components.securitySchemes`, and an empty one is the honest
    statement that this service authenticates nobody.
    """
    with TestClient(app_with({})) as client:
        document = client.get("/openapi.json").json()

    assert "securitySchemes" not in document.get("components", {})
    assert "security" not in document


# =============================================================================
# THE DEPLOYMENT GUIDE
# =============================================================================

# Settings §9 told operators to set, in a fenced block they would paste into a ConfigMap.
# No code read any of them, so an operator hardening by the book got a server whose CORS
# was still open and whose request rate was still unlimited -- and believed otherwise,
# which is worse than knowing the control is absent.
_DEAD_SETTINGS = (
    "NEXUS_RATE_LIMIT_ENABLED",
    "NEXUS_RATE_LIMIT_REQUESTS",
    "NEXUS_RATE_LIMIT_WINDOW",
)
_ASSIGNMENT = re.compile(rf"^\s*({'|'.join(_DEAD_SETTINGS)})\s*=", re.MULTILINE)


def test_the_deployment_guide_does_not_hand_out_a_setting_nothing_reads() -> None:
    """
    The ASSIGNMENT form is what the gate looks for, not the name.

    §9 still names these three, in a table that says they were deleted -- naming a
    retracted setting is how an operator finds out their existing ConfigMap does nothing.
    What must never come back is a copy-pasteable `NAME=value` line, because that is the
    form somebody applies.
    """
    offenders = _ASSIGNMENT.findall(DEPLOYMENT.read_text(encoding="utf-8"))

    assert not offenders, "docs/DEPLOYMENT.md hands out settings no code reads: " + ", ".join(
        sorted(set(offenders))
    )


def test_the_dead_setting_scan_can_see_the_block_that_shipped() -> None:
    """The four lines as they stood, so `assert not offenders` is not green over a typo."""
    shipped = (
        'NEXUS_API_CORS_ORIGINS=["https://your-domain.com"]\n'
        "NEXUS_RATE_LIMIT_ENABLED=true\n"
        "NEXUS_RATE_LIMIT_REQUESTS=100\n"
        "NEXUS_RATE_LIMIT_WINDOW=60\n"
    )

    assert len(_ASSIGNMENT.findall(shipped)) == 3
    assert not _ASSIGNMENT.findall("| `NEXUS_RATE_LIMIT_ENABLED` | **deleted.** |")


def test_the_guide_documents_the_cors_syntax_the_code_parses() -> None:
    """
    Two spellings of one setting were documented and one was parsed. The application takes
    comma-separated; a JSON list resolves to a single origin literally spelled `["https://
    ...`, which matches nothing and fails silently.
    """
    text = DEPLOYMENT.read_text(encoding="utf-8")

    assert 'NEXUS_API_CORS_ORIGINS=["' not in text
    assert "NEXUS_API_CORS_ORIGINS=https://" in text


def test_setting_the_key_that_was_advertised_changes_nothing() -> None:
    """
    The measurement that made the claim a defect rather than a gap: with `NEXUS_API_KEY`
    set, an unauthenticated request still returned 200 and a real protection class. Pinned
    so nobody re-adds the sentence believing the variable does something.
    """
    with TestClient(app_with({"NEXUS_API_KEY": "s3cret"})) as client:
        response = client.post("/api/v1/match", json=BODY)

    assert response.status_code == 200, response.text
