"""
tests.unit.presentation.api.test_feedback_endpoint | Layer: TEST
POST /api/v1/feedback records a reviewer's verdict, append-only, and changes nothing else.

## Relationships
# TESTS → presentation/api/feedback :: the recorder and the route

The scope is recording, and that is a MEASURED decision rather than an unfinished half.
Fine-tuning the encoder on exactly this signal was run on this project's benchmark and it
LOST accuracy -- P@1 0.5651 -> 0.5374, a delta of -0.0277, with the gold-vs-runner-up
margin falling too (`benchmarks/results/exp_finetune_transfer.json`). So the tests below
assert two different things with equal weight: that a verdict is durably written, and
that writing it does not move a single byte of a subsequent match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from tests.unit.presentation.api._support import build_api_matcher, request_fields

VERDICT = {
    "field": "account.resident_nm",
    "doc": "Name of the resident on the account",
    "chosenGovernanceId": "LWP-0001",
    "suggestedGovernanceId": "LWP-0004",
    "wasCorrect": False,
    "reviewer": "dispatcher.alethea",
    "ts": "2026-08-10T09:15:00Z",
}

# The key order of a stored record, pinned as a literal. A JSONL audit trail whose key
# order wanders cannot be diffed across hosts or across days.
RECORD_KEYS = (
    "ts",
    "receivedAt",
    "reviewer",
    "field",
    "doc",
    "chosenGovernanceId",
    "suggestedGovernanceId",
    "wasCorrect",
)


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


# =============================================================================
# RECORDING
# =============================================================================


class TestRecording:
    def test_a_verdict_is_written_and_echoed_back(self, client, log_path):
        response = client.post("/api/v1/feedback", json=VERDICT)

        assert response.status_code == 201, response.text
        record = response.json()["record"]
        assert tuple(record) == RECORD_KEYS
        assert record["field"] == VERDICT["field"]
        assert record["chosenGovernanceId"] == "LWP-0001"
        assert record["wasCorrect"] is False
        assert record["ts"] == VERDICT["ts"]
        assert record["receivedAt"] != VERDICT["ts"], (
            "the server stamped the client's clock instead of its own, so the trail can "
            "be reordered by a client with a wrong clock"
        )

        lines = log_path.read_text(encoding="ascii").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == record

    def test_writes_are_append_only(self, client, log_path):
        """
        Three verdicts, three lines, and the first two BYTES unchanged.

        Comparing parsed records would miss a rewrite that happened to produce equal
        objects; comparing the raw prefix is what makes "append-only" a property of the
        file rather than a claim about the code.
        """
        for index in range(3):
            body = dict(VERDICT, reviewer=f"reviewer.{index}")
            assert client.post("/api/v1/feedback", json=body).status_code == 201
            if index == 1:
                after_two = log_path.read_bytes()

        final = log_path.read_bytes()
        assert final.count(b"\n") == 3
        assert final.startswith(after_two), (
            "an earlier record was rewritten, so this file cannot be used as evidence"
        )
        reviewers = [json.loads(line)["reviewer"] for line in final.decode("ascii").splitlines()]
        assert reviewers == ["reviewer.0", "reviewer.1", "reviewer.2"]

    def test_a_non_ascii_reviewer_survives_as_an_escape(self, client, log_path):
        """
        The trail stays pure ASCII, so it reads identically on a legacy Windows code page
        and in a UTF-8 log pipeline. `json.loads` returns the original string either way.
        """
        client.post("/api/v1/feedback", json=dict(VERDICT, reviewer="Zoë Ngâteau"))

        raw = log_path.read_bytes()
        raw.decode("ascii")
        assert json.loads(raw.decode("ascii"))["reviewer"] == "Zoë Ngâteau"

    def test_the_environment_configures_the_path(self, tmp_path):
        target = tmp_path / "from-env.jsonl"
        app = create_app(
            configure_logs=False,
            matcher=build_api_matcher(),
            environ={"NEXUS_API_FEEDBACK_PATH": str(target)},
        )
        with TestClient(app) as client:
            assert client.post("/api/v1/feedback", json=VERDICT).status_code == 201
        assert target.exists()


# =============================================================================
# FAILURE MODES
# =============================================================================


class TestFailureModes:
    def test_an_unconfigured_server_answers_503_naming_the_setting(self):
        app = create_app(configure_logs=False, matcher=build_api_matcher(), environ={})
        with TestClient(app) as client:
            response = client.post("/api/v1/feedback", json=VERDICT)

        assert response.status_code == 503, response.text
        assert "NEXUS_API_FEEDBACK_PATH" in response.json()["error"]["message"]

    def test_a_write_that_fails_is_a_500_not_a_false_201(self, tmp_path):
        """
        The worst outcome available here is a 201 for a verdict that was never written:
        the reviewer believes their decision is on file and it is not, and nobody finds
        out until somebody goes looking for it as evidence.
        """
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("this is a file, so it cannot also be a parent directory")

        app = create_app(
            configure_logs=False,
            matcher=build_api_matcher(),
            feedback_path=str(blocker / "feedback.jsonl"),
            environ={},
        )
        with TestClient(app) as client:
            response = client.post("/api/v1/feedback", json=VERDICT)

        assert response.status_code == 500, response.text
        assert "Nothing was recorded" in response.json()["error"]["message"]

    @pytest.mark.parametrize(
        ("body", "because"),
        [
            ({k: v for k, v in VERDICT.items() if k != "wasCorrect"}, "verdict is required"),
            ({k: v for k, v in VERDICT.items() if k != "reviewer"}, "reviewer is required"),
            ({**VERDICT, "chosenGovernanceId": ""}, "an empty id records nothing useful"),
            ({**VERDICT, "reviewr": "typo"}, "unknown key"),
        ],
    )
    def test_a_malformed_verdict_is_422_with_the_reason(self, client, body, because):
        response = client.post("/api/v1/feedback", json=body)

        assert response.status_code == 422, f"{because}: {response.text}"
        assert response.json()["error"]["details"]["violations"], because


# =============================================================================
# NOT WIRED INTO RANKING
# =============================================================================


class TestFeedbackDoesNotAffectRanking:
    """
    Recording is the honest scope. Any claim that it improves matching would be false:
    fine-tuning on this signal was measured at -0.0277 P@1 on this project's benchmark.
    """

    def test_recording_a_contradicting_verdict_does_not_move_a_single_byte(self, client):
        """
        A behavioural oracle, not a grep.

        A reviewer says the top match was wrong and names a different governance id. If
        feedback leaked into ranking anywhere, the second response would differ. Byte
        comparison, because a ranking change small enough to move only a confidence in the
        sixth decimal is still a ranking change.
        """
        body = {"fields": request_fields(), "explain": True}
        before = client.post("/api/v1/match", json=body)
        assert before.status_code == 200, before.text

        for _ in range(5):
            assert client.post("/api/v1/feedback", json=VERDICT).status_code == 201

        after = client.post("/api/v1/match", json=body)
        assert after.content == before.content, (
            "matching output changed after feedback was recorded -- either feedback is "
            "wired into ranking, which the benchmark measured as a LOSS, or something "
            "else in this request path is not deterministic"
        )

    def test_the_matching_module_does_not_reference_the_feedback_store(self):
        """
        The structural half. The behavioural test above can only see leakage that changes
        THIS fixture's ranking; a dependency that fires on other data would slip past it.
        """
        from nexus_matcher.presentation.api import matching

        source = Path(matching.__file__).read_text(encoding="utf-8")
        assert "feedback" not in source.lower(), (
            "the matching path names the feedback store; recording must stay one-way"
        )
