"""
tests.unit.presentation.api.test_feedback_vocabulary | Layer: TEST
Tests: FeedbackRequest.verdict, the stored record, the published schema
Target: presentation/api/feedback.py

WC-11: the verdict vocabulary a boolean cannot hold, on the wire and in the file.

WHAT IS LOAD-BEARING

  THE ADDITION IS ADDITIVE. Every body that was valid before this member existed is still
  valid, still 201, and still stores the same eight values -- now beside an explicit null
  rather than beside nothing. `TestTheOldShapeStillWorks` is the half that would go red if
  `verdict` had been made required, or if a default had been back-filled.

  THE RECORD IS EVIDENCE, SO IT IS NOT INFERRED. A `wasCorrect: true` body stores
  `verdict: null`, not `verdict: "APPROVED"`. The server knows what that boolean implies
  and writes down what it was SENT, because a trail that records inferences alongside
  observations is a trail whose citations cannot be checked.

  TWO MEMBERS, ONE DECISION, NO RECONCILIATION. A verdict that contradicts `wasCorrect` is
  refused rather than resolved. Answering 201 would put a self-contradicting line in a file
  kept specifically to be cited later.

  THE COMPONENT NAME AND THE OPEN-VOCABULARY RULE. The published schema is still called
  `FeedbackRequest` -- renaming it would change the generated type in every client -- and
  the three values render as an INLINE enum rather than as their own component, so no new
  closed type lands in a generated client.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.feedback import _RECORD_KEYS
from tests.unit.presentation.api._support import build_api_matcher

BASE = {
    "field": "account.resident_nm",
    "doc": "Name of the resident on the account",
    "chosenGovernanceId": "LWP-0001",
    "suggestedGovernanceId": "LWP-0004",
    "reviewer": "dispatcher.alethea",
    "ts": "2026-08-10T09:15:00Z",
}


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "feedback.jsonl"


@pytest.fixture
def client(log_path: Path):
    app = create_app(
        configure_logs=False,
        matcher=build_api_matcher(),
        feedback_path=str(log_path),
        environ={},
    )
    with TestClient(app) as test_client:
        yield test_client


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


# =============================================================================
# THE THIRD VERDICT
# =============================================================================


class TestTheVerdictABooleanCannotHold:
    def test_a_manual_override_is_accepted_and_stored_as_itself(self, client, log_path):
        """
        The record WC-11 is about: the reviewer chose a term the matcher never proposed.
        Stored as `MANUAL_OVERRIDE`, so a later reader can count RECALL failures separately
        from ranking ones -- which is the whole reason the member exists.
        """
        body = {**BASE, "wasCorrect": False, "verdict": "MANUAL_OVERRIDE"}
        response = client.post("/api/v1/feedback", json=body)

        assert response.status_code == 201, response.text
        assert response.json()["record"]["verdict"] == "MANUAL_OVERRIDE"
        assert _lines(log_path)[0]["verdict"] == "MANUAL_OVERRIDE"

    def test_a_rejection_is_distinguishable_from_an_override(self, client, log_path):
        """
        Both carry `wasCorrect: false`. Before this member they were the same record; the
        stored lines below differ in exactly one key, which is the loss closing.
        """
        client.post("/api/v1/feedback", json={**BASE, "wasCorrect": False, "verdict": "REJECTED"})
        client.post(
            "/api/v1/feedback", json={**BASE, "wasCorrect": False, "verdict": "MANUAL_OVERRIDE"}
        )

        first, second = _lines(log_path)
        differing = {k for k in first if first[k] != second[k]}
        assert differing - {"receivedAt"} == {"verdict"}, (
            f"the two records differ in {sorted(differing)}; without `verdict` they would "
            f"be identical apart from the server's own stamp"
        )

    def test_the_stored_key_order_is_pinned_and_verdict_is_last(self, client, log_path):
        client.post("/api/v1/feedback", json={**BASE, "wasCorrect": True, "verdict": "APPROVED"})

        record = _lines(log_path)[0]
        assert tuple(record) == _RECORD_KEYS
        assert _RECORD_KEYS[-1] == "verdict", (
            "appended so a line written by a previous build is still a prefix-compatible "
            "reading of this order, and a trail spanning an upgrade diffs cleanly"
        )


class TestTheOldShapeStillWorks:
    def test_a_body_with_no_verdict_is_still_accepted(self, client, log_path):
        response = client.post("/api/v1/feedback", json={**BASE, "wasCorrect": False})

        assert response.status_code == 201, response.text
        assert _lines(log_path)[0]["chosenGovernanceId"] == "LWP-0001"

    def test_an_absent_verdict_is_stored_as_null_and_never_inferred(self, client, log_path):
        """
        `wasCorrect: true` unambiguously IMPLIES `APPROVED`, and the server still does not
        write it. The trail records what it was sent; the inference belongs to whoever
        reads it, where it can be labelled as an inference.
        """
        client.post("/api/v1/feedback", json={**BASE, "wasCorrect": True})

        record = _lines(log_path)[0]
        assert record["verdict"] is None
        assert record["wasCorrect"] is True

    def test_a_misspelled_key_is_still_a_422(self, client):
        """
        The widening must not have relaxed `extra="forbid"`. A dropped key on this route is
        a lost reviewer decision, and a silently-ignored `verdcit` is exactly that.
        """
        response = client.post(
            "/api/v1/feedback", json={**BASE, "wasCorrect": True, "verdcit": "APPROVED"}
        )
        assert response.status_code == 422, response.text


# =============================================================================
# TWO MEMBERS, ONE DECISION
# =============================================================================


class TestTheTwoMembersMustAgree:
    @pytest.mark.parametrize(
        ("verdict", "was_correct"),
        [
            ("APPROVED", False),
            ("REJECTED", True),
            ("MANUAL_OVERRIDE", True),
        ],
    )
    def test_a_contradiction_is_refused_rather_than_reconciled(self, client, verdict, was_correct):
        response = client.post(
            "/api/v1/feedback", json={**BASE, "wasCorrect": was_correct, "verdict": verdict}
        )

        assert response.status_code == 422, response.text
        assert response.json()["error"]["details"]["violations"]

    @pytest.mark.parametrize(
        ("verdict", "was_correct"),
        [("APPROVED", True), ("REJECTED", False), ("MANUAL_OVERRIDE", False)],
    )
    def test_every_agreeing_combination_is_accepted(self, client, verdict, was_correct):
        response = client.post(
            "/api/v1/feedback", json={**BASE, "wasCorrect": was_correct, "verdict": verdict}
        )
        assert response.status_code == 201, response.text

    def test_a_contradicting_verdict_writes_nothing_at_all(self, client, log_path):
        """A refused record must not be half-recorded; a 422 and a line is the worst of both."""
        client.post(
            "/api/v1/feedback", json={**BASE, "wasCorrect": True, "verdict": "MANUAL_OVERRIDE"}
        )
        assert not log_path.exists()

    def test_a_verdict_outside_the_vocabulary_is_a_422(self, client):
        response = client.post(
            "/api/v1/feedback", json={**BASE, "wasCorrect": False, "verdict": "DEFERRED"}
        )
        assert response.status_code == 422, response.text


# =============================================================================
# THE PUBLISHED SHAPE
# =============================================================================


class TestThePublishedShape:
    @staticmethod
    def _schemas() -> dict:
        return create_app(configure_logs=False, environ={}).openapi()["components"]["schemas"]

    def test_the_component_is_still_called_feedbackrequest(self):
        """
        The schema NAME is part of the published contract: it is the generated type name in
        every client built from this spec, and the Java client's own gate keys on it.
        Widening a request must not rename it.
        """
        schemas = self._schemas()
        assert "FeedbackRequest" in schemas
        assert "verdict" in schemas["FeedbackRequest"]["properties"]

    def test_every_member_of_the_previous_shape_is_still_published(self):
        published = set(self._schemas()["FeedbackRequest"]["properties"])
        assert {
            "field",
            "doc",
            "chosenGovernanceId",
            "suggestedGovernanceId",
            "wasCorrect",
            "reviewer",
            "ts",
        } <= published

    def test_verdict_is_optional_in_the_published_schema(self):
        schema = self._schemas()["FeedbackRequest"]
        assert "verdict" not in schema.get("required", [])
        assert "wasCorrect" in schema["required"]

    def test_the_three_values_do_not_become_a_new_published_component(self):
        """
        An `enum.Enum` here would publish its own schema, become its own hand-written Java
        enum, and refuse a response the day a fourth value ships. A `Literal` renders inline
        on the property, so a client gets the values as documentation and not as a closed
        type it can choke on.
        """
        schemas = self._schemas()
        verdict = schemas["FeedbackRequest"]["properties"]["verdict"]
        options = [branch for branch in verdict["anyOf"] if branch.get("type") == "string"]

        assert options and options[0]["enum"] == ["APPROVED", "REJECTED", "MANUAL_OVERRIDE"]
        assert "$ref" not in json.dumps(verdict)
        assert not any("erdict" in name for name in schemas), (
            f"the verdict vocabulary became its own component: "
            f"{[n for n in schemas if 'erdict' in n]}"
        )


# =============================================================================
# THE TRAIL A CONSUMER ACTUALLY READS
# =============================================================================


class TestTheTrailIsReadableByTheConsumer:
    def test_verdicts_posted_here_drive_a_bypass_end_to_end(self, client, log_path):
        """
        The whole loop in one test: three reviewers post, the file is read by the shipped
        consumer, and the standing answers are what the reviewers decided. Without this the
        writer and the reader are two modules agreeing about a format only in prose.
        """
        from nexus_matcher.application.feedback_loop import ApprovedPairBypass
        from nexus_matcher.domain.models.entities import DictionaryEntry
        from nexus_matcher.domain.ports import MappingEntryLookup
        from nexus_matcher.shared.types.base import DataType, DocumentId

        posts = [
            {
                **BASE,
                "field": "a",
                "chosenGovernanceId": "T1",
                "wasCorrect": True,
                "verdict": "APPROVED",
            },
            {
                **BASE,
                "field": "b",
                "chosenGovernanceId": "T2",
                "wasCorrect": False,
                "verdict": "MANUAL_OVERRIDE",
            },
            {
                **BASE,
                "field": "c",
                "chosenGovernanceId": "T3",
                "wasCorrect": False,
                "verdict": "REJECTED",
            },
        ]
        for body in posts:
            assert client.post("/api/v1/feedback", json=body).status_code == 201

        entries = {
            term: DictionaryEntry(
                id=DocumentId(term),
                business_name=f"Term {term}",
                logical_name=term.lower(),
                definition=f"The governed element {term}.",
                data_type=DataType.STRING,
            )
            for term in ("T1", "T2", "T3")
        }
        bypass = ApprovedPairBypass.from_trail(log_path)
        bypass.bind(MappingEntryLookup(entries))

        def _field(key: str):
            from nexus_matcher.domain.models.entities import SchemaField

            return SchemaField(name=key, data_type=DataType.STRING, full_path=key)

        assert bypass.approved_pair(_field("a")).entry.id == "T1"
        assert bypass.approved_pair(_field("b")).entry.id == "T2"
        assert bypass.approved_pair(_field("c")) is None, (
            "a REJECTED verdict must never become a bypass -- it says nothing in the "
            "glossary governs this field"
        )
        assert bypass.bypass_report().standing == 2
