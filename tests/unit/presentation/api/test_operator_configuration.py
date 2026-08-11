"""
tests.unit.presentation.api.test_operator_configuration | Layer: TEST
The path an OPERATOR actually has: environment variables into a running server.

## Relationships
# TESTS → presentation/api/app :: _load_configured_matcher and create_app

Both shipped ways to start this service call `create_app()` with NO arguments --
`nexus-matcher api` runs `uvicorn ...app:create_app --factory`, and the documented
`uvicorn ...app:app` imports a module-level instance. So a keyword argument on
`create_app` is reachable by an embedder and by nobody else.

That is H-006's exact shape: a feature that is written, tested, reviewed and committed,
and simply never runs, because the half that would reach it belongs to another file. Its
most expensive instance in this repo shipped 2.5x faster and unreachable. The governance
vocabulary is the acute case here -- the domain layer resolves every
`MatchResult.governance` through it, and without `NEXUS_API_GOVERNANCE` the HTTP surface
would answer `"governance": null` for every field of every request, correctly and
uselessly.

This test therefore drives the WHOLE operator path: two files on disk, two environment
variables, one HTTP request, and a real protection class on the wire. It is the only test
in this package that loads the bundled encoder, which is the price of proving the wiring
rather than describing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app

# FICTIONAL EXAMPLE DATA. The Tallow Creek Grain Cooperative does not exist and neither
# does this taxonomy; both were invented for this file. A different fictional setting from
# the rest of the package on purpose -- if these tests ever start agreeing because they
# share a fixture rather than because the code works, the difference makes it visible.
VOCABULARY = {
    "notice": "FICTIONAL EXAMPLE DATA. Tallow Creek Grain Cooperative is invented.",
    "open_classification": "TALLOW_OPEN",
    "classes": [
        {
            "code": "GROWERID",
            "name": "Grower identity",
            "classification": "TALLOW_SEALED",
            "personal_information": True,
            "direct_identifier": True,
            "enhancement": "MASK_IN_LOGS",
        },
        {
            "code": "SILOWEIGHT",
            "name": "Silo intake weight",
            "classification": "TALLOW_OPEN",
            "personal_information": False,
            "direct_identifier": False,
        },
    ],
}

GLOSSARY_CSV = """id,business_name,logical_name,definition,data_type,domain,protection_class
TCG-0001,Grower Legal Name,grower_legal_nm,The registered name of the grower delivering grain.,string,Grower,GROWERID
TCG-0002,Silo Intake Weight,silo_intake_wt,Mass of grain recorded at the silo weighbridge on intake.,double,Silo,SILOWEIGHT
"""


@pytest.fixture
def configured(tmp_path: Path) -> dict[str, str]:
    """The two files and the two variables an operator would set."""
    vocabulary_path = tmp_path / "protection_classes.json"
    vocabulary_path.write_text(json.dumps(VOCABULARY), encoding="utf-8")
    glossary_path = tmp_path / "glossary.csv"
    glossary_path.write_text(GLOSSARY_CSV, encoding="utf-8")
    return {
        "NEXUS_API_DICTIONARY": str(glossary_path),
        "NEXUS_API_GOVERNANCE": str(vocabulary_path),
        "NEXUS_API_FEEDBACK_PATH": str(tmp_path / "feedback.jsonl"),
    }


def test_the_environment_alone_brings_up_a_server_that_classifies(configured):
    """
    No keyword arguments anywhere: exactly what `nexus-matcher api` can produce.

    The assertion that matters is the last one. A server whose vocabulary never arrived
    still returns 200, still returns the right entry, still returns a populated
    `governanceId` -- and returns `"governance": null` for every field, which is
    indistinguishable to the caller from a glossary that genuinely carries no classes.
    That is the failure this whole path exists to prevent, and it is invisible to every
    other test in this package because they inject a matcher directly.
    """
    with TestClient(create_app(configure_logs=False, environ=configured)) as client:
        assert client.get("/health/ready").json()["components"]["matcher"] is True

        response = client.post(
            "/api/v1/match",
            json={
                "fields": [
                    {
                        "name": "grower_name",
                        "path": "delivery.grower_name",
                        "doc": "Registered name of the grower delivering the load",
                        "type": "string",
                    }
                ],
                "top_k": 1,
            },
        )

    assert response.status_code == 200, response.text
    candidate = response.json()["results"]["delivery.grower_name"][0]
    assert candidate["governanceId"] == "TCG-0001"
    assert candidate["governance"] == {
        "code": "GROWERID",
        "name": "Grower identity",
        "classification": "TALLOW_SEALED",
        "personalInformation": True,
        "directIdentifier": True,
    }


def test_a_glossary_whose_codes_nobody_can_read_is_refused_at_startup(configured):
    """
    The near-miss, and the one no layer below can catch.

    The glossary is configured, the vocabulary is not. `load_entries` attaches a
    governance code only when a vocabulary is present, and the matcher only refuses codes
    it cannot resolve -- so with no vocabulary there are no codes, nothing to refuse, and
    a server that starts perfectly and answers `"governance": null` for a file whose
    header says `protection_class` in as many words.

    A 200 with a null class is the expensive answer here: the caller applies nothing, and
    believes the glossary told them nothing was needed. So it is a startup failure, an
    unhealthy `matcher` component, and a 503 that names the variable to set.
    """
    without_vocabulary = {k: v for k, v in configured.items() if k != "NEXUS_API_GOVERNANCE"}

    with TestClient(create_app(configure_logs=False, environ=without_vocabulary)) as client:
        assert client.get("/health/ready").status_code == 503
        response = client.post(
            "/api/v1/match", json={"fields": [{"name": "grower_name"}], "top_k": 1}
        )

    assert response.status_code == 503, response.text
    message = response.json()["error"]["message"]
    assert "NEXUS_API_GOVERNANCE" in message, message
    assert "protection_class" in message, message


def test_a_glossary_with_no_protection_column_needs_no_vocabulary(tmp_path):
    """
    The guard above must not become a demand that every deployment configure a
    vocabulary. A glossary that carries no protection-code column has nothing to
    interpret, so it starts and serves -- with a null class, honestly this time.
    """
    plain = tmp_path / "plain.csv"
    plain.write_text(
        "id,business_name,logical_name,definition,data_type,domain\n"
        "TCG-0002,Silo Intake Weight,silo_intake_wt,"
        "Mass of grain recorded at the silo weighbridge on intake.,double,Silo\n",
        encoding="utf-8",
    )

    with TestClient(
        create_app(configure_logs=False, environ={"NEXUS_API_DICTIONARY": str(plain)})
    ) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post(
            "/api/v1/match",
            json={"fields": [{"name": "intake_weight", "path": "load.intake_weight"}], "top_k": 1},
        )

    assert response.status_code == 200, response.text
    candidate = response.json()["results"]["load.intake_weight"][0]
    assert candidate["governanceId"] == "TCG-0002"
    assert candidate["governance"] is None


def test_the_feedback_path_from_the_environment_is_written_to(configured):
    """The third variable, proven by the file appearing rather than by the 201."""
    log_path = Path(configured["NEXUS_API_FEEDBACK_PATH"])

    with TestClient(create_app(configure_logs=False, environ=configured)) as client:
        response = client.post(
            "/api/v1/feedback",
            json={
                "field": "delivery.grower_name",
                "doc": "Registered name of the grower",
                "chosenGovernanceId": "TCG-0001",
                "suggestedGovernanceId": None,
                "wasCorrect": True,
                "reviewer": "silo-clerk-2",
                "ts": "2026-08-10T10:00:00Z",
            },
        )

    assert response.status_code == 201, response.text
    assert log_path.exists()
    assert json.loads(log_path.read_text(encoding="ascii"))["chosenGovernanceId"] == "TCG-0001"
