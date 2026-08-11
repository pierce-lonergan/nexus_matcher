"""
tests.unit.presentation.api.test_readiness | Layer: TEST
Readiness answers the question a rollout gate asks: can this process do its job?

## Relationships
# TESTS → presentation/api/app :: AppState.check_ready and _bring_up_matcher
# TESTS → docker/Dockerfile, docker/docker-compose.yml, docs/DEPLOYMENT.md :: probe targets

Two defects, and the second one made the first one moot.

**The endpoint lied.** `matcher` was registered only when a dictionary loaded, and
`check_ready()` is `all()` over what is registered -- so with `NEXUS_API_DICTIONARY`
absent, empty, or misspelled, nothing was registered and `/health/ready` answered 200
while every `POST /api/v1/match` answered 503. A *broken* dictionary went red correctly; a
*missing* one did not, and missing is the state a rollout most often produces.

**Nothing read it.** Every shipped probe targeted `/health`, which answers 200 with
`status: "degraded"` even when the dictionary failed to load, so `curl -f` passes. The
Kubernetes manifests, the Dockerfile HEALTHCHECK and the compose healthcheck all did it,
which means any change to `/health/ready` was invisible to every deployment this
repository documents. The probe-target test at the bottom is therefore the load-bearing
one: without it the rest is a correction nobody would have received.

## The opt-out is inverted on purpose

`NEXUS_API_MATCHING_OPTIONAL`, not `NEXUS_API_REQUIRE_MATCHER`. A knob defaulting to the
unsafe value protects only the deployments whose operator remembered to set it, which are
not the misconfigured ones. So the safe behaviour is the default and the exemption is
deliberate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import AppState, create_app

REPO = Path(__file__).resolve().parents[4]


def started(environ: dict[str, str]) -> TestClient:
    return TestClient(create_app(configure_logs=False, environ=environ))


# =============================================================================
# A DEPLOYMENT THAT CANNOT MATCH IS NOT READY
# =============================================================================


@pytest.mark.parametrize(
    ("label", "environ"),
    [
        ("absent", {}),
        ("empty string", {"NEXUS_API_DICTIONARY": ""}),
        ("whitespace", {"NEXUS_API_DICTIONARY": "   "}),
        ("misspelled variable", {"NEXUS_API_DICTIONARYY": "glossary.csv"}),
    ],
    ids=["absent", "empty", "whitespace", "misspelled"],
)
def test_no_dictionary_configured_is_not_ready(label: str, environ: dict[str, str]) -> None:
    """
    All four are one state -- the process serves `POST /api/v1/match` and 503s it -- and
    all four are what a Helm template produces when a value does not resolve.
    """
    with started(environ) as client:
        ready = client.get("/health/ready")
        match = client.post("/api/v1/match", json={"fields": [{"name": "x"}]})

    assert match.status_code == 503, f"{label}: the endpoint must be the one telling the truth"
    assert ready.status_code == 503, f"{label}: {ready.text}"


def test_the_503_names_the_component_that_is_red() -> None:
    """
    An operator whose dictionary failed to load used to get the string "Service not ready"
    and nothing else. `details.components` is every component and its state, so the
    diagnosis does not require server access.
    """
    with started({}) as client:
        response = client.get("/health/ready")

    details = response.json()["error"]["details"]
    assert details["components"]["matcher"] is False, details
    assert "matcher" in response.json()["error"]["message"]


def test_health_reports_degraded_for_the_same_deployment() -> None:
    """
    `/health` is what the probes used to read. It must agree with `/health/ready` about
    whether this process can do its job, or pointing the probes at either one is a coin
    toss.
    """
    with started({}) as client:
        assert client.get("/health").json()["status"] == "degraded"


# =============================================================================
# THE OPT-OUT
# =============================================================================


def test_a_health_only_deployment_declares_itself_with_the_opt_out() -> None:
    """
    The supported way to keep the old answer. This service was health-and-introspection
    only before the matching endpoints existed, and that deployment stays supported -- but
    it now says so, rather than being indistinguishable from a broken rollout.
    """
    with started({"NEXUS_API_MATCHING_OPTIONAL": "true"}) as client:
        ready = client.get("/health/ready")

    assert ready.status_code == 200, ready.text
    assert "matcher" not in ready.json()["components"]


def test_the_opt_out_still_reports_a_dictionary_that_was_asked_for_and_broke() -> None:
    """
    The one case where a component is visible and does not gate.

    Opting out says "matching need not be ready", not "do not tell me about it". A
    configured dictionary that failed to load is a misconfiguration either way, and the
    only reason `gates_readiness=False` exists is so it can be reported without failing a
    gate the operator deliberately opened.
    """
    environ = {
        "NEXUS_API_DICTIONARY": "does-not-exist.xlsx",
        "NEXUS_API_MATCHING_OPTIONAL": "true",
    }
    with started(environ) as client:
        ready = client.get("/health/ready")
        health = client.get("/health")

    assert ready.status_code == 200, ready.text
    assert ready.json()["components"]["matcher"] is False
    assert health.json()["status"] == "degraded"


def test_an_unparseable_opt_out_is_refused_rather_than_read_as_off() -> None:
    """
    Reading `NEXUS_API_MATCHING_OPTIONAL=yes-please` as False gives the operator 503s they
    believe they turned off, and reading it as True opens the gate they believe they
    closed. Both are worse than refusing to start.
    """
    with pytest.raises(ValueError, match="NEXUS_API_MATCHING_OPTIONAL"):
        create_app(configure_logs=False, environ={"NEXUS_API_MATCHING_OPTIONAL": "yes-please"})


# =============================================================================
# WHAT `components` MAY CONTAIN
# =============================================================================


def test_only_components_that_were_really_checked_are_reported() -> None:
    """
    `vector_store` and `cache` were set True inside `try:` blocks whose bodies were a
    comment, so no failure could reach the `except:` and neither could ever be False. A
    component that reports True unconditionally is not a check; it is a claim, and
    `/health/ready` is read by rollout gates.
    """
    with started({"NEXUS_API_MATCHING_OPTIONAL": "true"}) as client:
        components = client.get("/health/ready").json()["components"]

    assert components == {"api": True, "config": True}


def test_a_non_gating_component_is_visible_without_deciding_readiness() -> None:
    """The unit behind the opt-out, so the aggregation rule is pinned on its own."""
    state = AppState()
    state.set_component_status("api", True)
    state.set_component_status("matcher", False, gates_readiness=False)

    assert state.check_ready() is True
    assert state.components["matcher"] is False

    state.set_component_status("matcher", False)
    assert state.check_ready() is False


def test_nothing_registered_is_not_ready() -> None:
    """Before the lifespan runs there is nothing to aggregate, and `all(())` is True."""
    assert AppState().check_ready() is False


# =============================================================================
# THE PROBES POINT AT AN ENDPOINT THAT CARRIES A VERDICT
# =============================================================================

# Each shipped artifact and the probe lines in it. `/health` answers 200 with
# `status: "degraded"` when the dictionary failed to load, so `curl -f` passes and the
# rollout proceeds -- which made every readiness fix above unobservable in production.
_PROBE_ARTIFACTS = ("docker/Dockerfile", "docker/docker-compose.yml", "docs/DEPLOYMENT.md")

# `/health` as a whole path: not `/health/ready`, not `/healthz` on another service.
_BARE_HEALTH = re.compile(r"localhost:8000/health(?![\w/])|path: /health(?![\w/])")


@pytest.mark.parametrize("artifact", _PROBE_ARTIFACTS)
def test_no_shipped_probe_targets_the_endpoint_that_cannot_fail(artifact: str) -> None:
    text = (REPO / artifact).read_text(encoding="utf-8")
    offenders = [line.strip() for line in text.splitlines() if _BARE_HEALTH.search(line)]

    assert not offenders, (
        f"{artifact} probes /health, which answers 200 while the dictionary is missing:\n  "
        + "\n  ".join(offenders)
        + "\nPoint liveness at /health/live and readiness at /health/ready."
    )


def test_the_probe_scan_can_see_a_probe_it_should_reject() -> None:
    """
    `assert not offenders` passes over a regex that matches nothing, which is the
    vacuous-green shape this repository keeps rediscovering. These are the four lines that
    shipped, plus the two spellings that must NOT trip it.
    """
    assert _BARE_HEALTH.search("    CMD curl -f http://localhost:8000/health || exit 1")
    assert _BARE_HEALTH.search('      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]')
    assert _BARE_HEALTH.search("            path: /health")
    assert not _BARE_HEALTH.search("            path: /health/ready")
    assert not _BARE_HEALTH.search('test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]')


@pytest.mark.parametrize("artifact", _PROBE_ARTIFACTS)
def test_every_probe_artifact_still_has_a_probe(artifact: str) -> None:
    """Deleting the healthcheck would satisfy the gate above and lose the signal entirely."""
    text = (REPO / artifact).read_text(encoding="utf-8")
    assert "/health/ready" in text or "/health/live" in text, artifact
