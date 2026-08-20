"""
tests.unit.presentation.api.test_query_signal_channel | Layer: TEST
The wire half of the per-request query-side signal channel (AR-6).

## Relationships
# TESTS → presentation/api/schemas  :: FieldSpec.signals, MatchRequest.signals, the bounds
# TESTS → presentation/api/matching :: the hand-over into the application layer

## The one decision this file exists to pin

`FieldSpec` keeps `extra="forbid"`, AND a deployment can send query-side context this
library has never heard of. Those look like opposites and are not: a typo and an extension
are different events. A misspelled `doc` is silently dropped retrieval signal and stays a
422; an unrecognised SIGNAL is a deployment knowing something the library does not, and a
422 there is what made the channel unreachable in the first place.

So the extension point is DECLARED -- a named `signals` field whose value is open -- rather
than bought by relaxing the model. `test_the_typo_gate_survives_the_extension_point` is the
assertion that says the trade was not made.

## And the property that has to hold on every request

A caller who sends no signals must get exactly what they got before this channel existed,
down to the call the endpoint makes into the matcher. `TestAbsence` drives that against a
stub that records whether the extended keyword arrived at all -- a stub that merely
accepted the keyword could not tell "sent nothing" from "sent an empty map".
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.matching import _to_schema_field
from nexus_matcher.presentation.api.schemas import (
    MAX_REQUEST_SIGNAL_CHARS,
    FieldSpec,
    MatchRequest,
)
from tests.unit.presentation.api._support import FakeMatcher, build_api_matcher, request_fields

CATALOG = {"psgr": "passenger", "brth": "berth"}


def client_for(matcher: object, **kwargs: object) -> TestClient:
    app = create_app(configure_logs=False, matcher=matcher, environ={}, **kwargs)
    return TestClient(app)


@pytest.fixture
def real_client():
    with client_for(build_api_matcher()) as client:
        yield client


@pytest.fixture
def fake():
    matcher = FakeMatcher()
    with client_for(matcher) as client:
        yield client, matcher


def post(client: TestClient, **body: object):
    payload: dict[str, object] = {"fields": request_fields()}
    payload.update(body)
    return client.post("/api/v1/match", json=payload)


# =============================================================================
# THE CHANNEL IS OPEN
# =============================================================================


class TestTheChannelIsOpen:
    @pytest.mark.parametrize(
        ("signals", "because"),
        [
            ({"abbreviations": CATALOG}, "the headline signal"),
            ({"abbreviation_overlay": CATALOG}, "its v1 name"),
            ({"domain": "utilities"}, "the domain prior"),
            ({"namespace": "com.example.utilities"}, "its v1 name"),
            ({"entity": "Account"}, "the parent record"),
            ({"protection_hint": "restricted"}, "a signal this library has never heard of"),
            ({"nested": {"deployment": {"specific": ["values"]}}}, "an arbitrary structure"),
        ],
    )
    def test_a_request_level_signal_is_accepted(self, real_client, signals, because):
        response = post(real_client, signals=signals)
        assert response.status_code == 200, f"{because}: {response.text}"

    def test_a_field_level_signal_is_accepted(self, real_client):
        response = real_client.post(
            "/api/v1/match",
            json={
                "fields": [
                    {"name": "a", "path": "t.a", "signals": {"entity": "T", "unknown": [1, 2]}}
                ]
            },
        )
        assert response.status_code == 200, response.text

    def test_the_typo_gate_survives_the_extension_point(self, real_client):
        """
        Both halves of the trade in one test, because either alone reads as an accident.

        The SAME unknown key is a 422 as a sibling of `doc` and a 200 inside `signals`.
        The first is a dropped column comment; the second is a deployment's own context.
        """
        typo = {"fields": [{"name": "a", "path": "t.a", "documentation": "x"}]}
        extension = {"fields": [{"name": "a", "path": "t.a", "signals": {"documentation": "x"}}]}

        assert real_client.post("/api/v1/match", json=typo).status_code == 422
        assert real_client.post("/api/v1/match", json=extension).status_code == 200

    def test_an_unknown_signal_does_not_change_the_answer(self, real_client):
        """Ignored means ignored: byte-for-byte, not merely 'still a 200'."""
        without = post(real_client)
        with_noise = post(real_client, signals={"protection_hint": "x", "trace": {"id": 7}})
        assert without.content == with_noise.content

    def test_the_published_schema_declares_the_channel_on_both_models(self, real_client):
        schemas = real_client.get("/openapi.json").json()["components"]["schemas"]
        for model in ("FieldSpec", "MatchRequest"):
            assert "signals" in schemas[model]["properties"], model
            # The description is the only place a client learns which keys are read and
            # that the rest are carried. A channel nobody can discover is not declared.
            described = schemas[model]["properties"]["signals"]["description"]
            assert "abbreviations" in described
            assert "entity" in described
            assert "domain" in described
            assert "carried and ignored" in described

    def test_signals_is_not_required(self, real_client):
        schemas = real_client.get("/openapi.json").json()["components"]["schemas"]
        assert "signals" not in schemas["FieldSpec"].get("required", [])
        assert "signals" not in schemas["MatchRequest"].get("required", [])


# =============================================================================
# THE HAND-OVER
# =============================================================================


class TestHandOver:
    def test_request_signals_reach_the_matcher_verbatim(self, fake):
        client, matcher = fake
        sent = {"abbreviations": CATALOG, "domain": "utilities", "unknown": {"a": 1}}
        assert post(client, signals=sent).status_code == 200
        assert matcher.signals_seen == [sent]

    def test_field_signals_ride_on_the_field_not_the_request(self, fake):
        client, matcher = fake
        response = client.post(
            "/api/v1/match",
            json={"fields": [{"name": "a", "path": "t.a", "signals": {"entity": "T"}}]},
        )
        assert response.status_code == 200, response.text
        # Not promoted to the request: a per-field fact applied to every field would give
        # the other columns a parent record they do not have.
        assert matcher.signals_seen == [None]

    def test_a_field_spec_with_no_signals_produces_the_field_it_always_did(self):
        spec = FieldSpec(name="a", path="t.a", doc="", type="")
        assert _to_schema_field(spec).source_metadata == {"flattened_name": "t.a"}

    def test_a_field_spec_with_signals_carries_them_nested(self):
        spec = FieldSpec(name="a", path="t.a", signals={"entity": "T"})
        assert _to_schema_field(spec).source_metadata == {
            "flattened_name": "t.a",
            "query_signals": {"entity": "T"},
        }


# =============================================================================
# ABSENCE
# =============================================================================


class TestAbsence:
    @pytest.mark.parametrize("body", [{}, {"signals": {}}])
    def test_no_signals_means_the_unextended_call(self, fake, body):
        """
        The endpoint must not pass a new keyword to a duck-typed collaborator for a request
        that had nothing to say. `matcher` is reached through a private attribute and is
        explicitly not this layer's object; every implementation of today's signature has
        to keep working.
        """
        client, matcher = fake
        assert post(client, **body).status_code == 200
        assert matcher.signals_seen == [None]

    def test_the_response_is_unchanged_by_an_empty_signal_map(self, real_client):
        assert post(real_client).content == post(real_client, signals={}).content


# =============================================================================
# BOUNDS
# =============================================================================


class TestBounds:
    """
    A resource limit, not an opinion about content. Nothing here inspects a key or a value
    -- only how much of it there is -- which is the same standard `_MAX_DOC` applies to a
    column comment: refuse an out-of-memory, never refuse a meaning.
    """

    def test_a_realistic_catalog_is_inside_the_request_budget(self, real_client):
        overlay = {f"ab{i:05d}": "expanded business word" for i in range(8_000)}
        response = post(real_client, signals={"abbreviations": overlay})
        assert response.status_code == 200, response.text[:400]

    def test_an_over_budget_request_map_is_refused_with_the_bound(self, real_client):
        overlay = {f"ab{i:07d}": "expanded business word " * 4 for i in range(20_000)}
        response = post(real_client, signals={"abbreviations": overlay})
        assert response.status_code == 422, response.status_code
        assert str(MAX_REQUEST_SIGNAL_CHARS) in response.text

    def test_an_over_budget_field_map_is_refused(self, real_client):
        response = real_client.post(
            "/api/v1/match",
            json={"fields": [{"name": "a", "path": "t.a", "signals": {"k": "x" * 4096}}]},
        )
        assert response.status_code == 422, response.status_code

    def test_a_deeply_nested_signal_map_is_refused_rather_than_recursed(self):
        """
        The channel accepts arbitrary JSON, so a body that is small on the wire can still
        be a nesting bomb after parsing. Recursing until Python's own limit would turn a
        caller's mistake into a 500.
        """
        deep: object = "leaf"
        for _ in range(40):
            deep = {"n": deep}
        with pytest.raises(ValueError, match="nests deeper"):
            MatchRequest(fields=[{"name": "a"}], signals={"deep": deep})

    def test_a_signal_map_at_the_bound_is_accepted(self):
        # The refusal must be a bound and not a shape: one character under the cap passes.
        payload = {"k": "x" * (MAX_REQUEST_SIGNAL_CHARS - 1)}
        assert MatchRequest(fields=[{"name": "a"}], signals=payload).signals == payload


# =============================================================================
# DETERMINISM
# =============================================================================


def test_two_identical_signalled_requests_are_byte_identical(real_client):
    """
    The response is a governance artifact that gets diffed. A signal map is a dict, and a
    dict that reached the query text in a different order on the second call would produce
    a different body.
    """
    signals = {"abbreviations": CATALOG, "domain": "utilities", "entity": "Account"}
    first = post(real_client, signals=signals)
    second = post(real_client, signals=signals)
    assert first.content == second.content
    # ASCII-only, like every other body this service emits: a signal value is caller text
    # and must not be the thing that puts a raw non-ASCII byte on the wire.
    first.content.decode("ascii")
    assert json.loads(first.content)["results"]
