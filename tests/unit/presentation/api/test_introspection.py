"""
tests.unit.presentation.api.test_introspection | Layer: TEST
GET /api/v1/status -- what is loaded, and whether retrieval is degraded.

## Relationships
# TESTS → presentation/api/introspect :: the status body, the warnings, the encoder tier
# TESTS → presentation/api/app :: the dictionary provenance recorded at startup

The property this file exists for is the one a health probe cannot express. A process can
be live, ready, and answering 200 to every match out of an encoder nobody chose -- which
has already cost the adopting pipeline a six-hour bulk run. So the assertions below are
mostly about `degraded` and `warnings`: what turns them on, what must NOT turn them on, and
that the body still arrives when the thing it reports on is broken.

## Why the encoder truth table is unit-tested and not driven end to end

`fallbackInForce` is true when encoder selection FELL THROUGH -- a lower rung in force while
the ladder's first choice is unavailable. A healthy developer machine cannot produce that
state: the bundled weights are present, so selection never falls through. Faking it by
deleting weights from an installed package is not a test, it is vandalism with a fixture
attached. So the two facts that decide the flag are parameters of `_encoder_payload`, and
the whole table is driven directly; what goes through HTTP is the wiring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import find_spec

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.application.use_cases.match_schema import MatchingConfig
from nexus_matcher.presentation.api import app as app_module
from nexus_matcher.presentation.api.app import AppState, _bring_up_matcher, create_app
from nexus_matcher.presentation.api.introspect import (
    _ENCODER_TIER_BY_MODULE,
    IntrospectionService,
    ProvenanceRecorder,
    StatusResponseView,
    _encoder_payload,
    bundled_encoder_available,
    encoder_tier,
)
from nexus_matcher.presentation.api.limits import BoundedWorkPool, MatchServiceLimits
from nexus_matcher.presentation.api.matching import MatcherHandle
from tests.unit.presentation.api._support import FakeMatcher, build_api_matcher

STATUS_KEYS = (
    "ready",
    "degraded",
    "warnings",
    "dictionary",
    "encoder",
    "thresholds",
    "limits",
)


def client_for(matcher: object, **kwargs: object) -> TestClient:
    app = create_app(configure_logs=False, matcher=matcher, environ={}, **kwargs)
    return TestClient(app)


def status_of(client: TestClient) -> dict:
    response = client.get("/api/v1/status")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def real_client():
    with client_for(build_api_matcher()) as client:
        yield client


# =============================================================================
# THE BODY
# =============================================================================


class TestTheStatusBody:
    """The shape, and that it arrives at all."""

    def test_the_keys_are_the_contract_in_order(self, real_client):
        assert tuple(status_of(real_client)) == STATUS_KEYS

    def test_real_bodies_validate_against_the_published_schema(self, real_client):
        StatusResponseView.model_validate(status_of(real_client))

    def test_a_loaded_healthy_server_reports_neither_degraded_nor_warnings(self, real_client):
        body = status_of(real_client)

        assert body["ready"] is True
        assert body["degraded"] is False
        assert body["warnings"] == []
        assert body["dictionary"]["entryCount"] == 5

    def test_two_identical_requests_produce_identical_bytes(self, real_client):
        """
        Nothing here is read from a clock or from live load at request time, which is what
        lets an operator diff two hosts and see only the difference that matters.
        """
        assert real_client.get("/api/v1/status").content == (
            real_client.get("/api/v1/status").content
        )

    def test_an_unloaded_server_still_answers_200(self):
        """
        A pre-run degradation check that 503s when the thing it checks for is true would be
        unusable at exactly the moment it is needed. `/health/ready` is the route that says
        no by status code; this one says no in the body.
        """
        with client_for(None) as client:
            body = status_of(client)

            assert body["ready"] is False
            assert body["degraded"] is True
            assert [warning["code"] for warning in body["warnings"]] == ["NO_DICTIONARY"]
            assert "NEXUS_API_DICTIONARY" in body["warnings"][0]["message"]

    def test_an_unloaded_server_reports_no_thresholds_rather_than_the_shipped_ones(self):
        """
        Reporting defaults that are not in force would be a wrong answer in the surface an
        operator consults precisely to find out what IS in force.
        """
        with client_for(None) as client:
            body = status_of(client)

            assert body["encoder"] is None
            assert body["thresholds"] is None
            assert body["dictionary"] == {"entryCount": None, "source": None, "indexedAt": None}

    def test_an_empty_dictionary_is_a_warning_of_its_own(self):
        """
        A glossary that loaded and carries nothing matches nothing, inherits nothing, and
        looks perfectly healthy to every probe in this service.
        """
        with client_for(build_api_matcher(entries=())) as client:
            body = status_of(client)

            assert body["dictionary"]["entryCount"] == 0
            assert [warning["code"] for warning in body["warnings"]] == ["EMPTY_DICTIONARY"]
            assert body["degraded"] is True

    def test_the_limits_block_reports_the_caps_a_client_has_to_chunk_against(self):
        limits = MatchServiceLimits(max_fields=11, max_batch_fields=13, max_workers=2, max_queued=3)
        with client_for(build_api_matcher(), limits=limits) as client:
            reported = status_of(client)["limits"]

        assert reported["maxFields"] == 11
        assert reported["maxBatchFields"] == 13
        assert reported["deadlineSeconds"] == 25.0
        assert reported["capacity"] == 5
        assert reported["bodyByteCap"] == limits.body_byte_cap

    def test_live_load_is_deliberately_absent_from_the_limits_block(self):
        """
        `in_flight` would make two identical requests produce different bytes. Pinned so it
        cannot be added back as an obvious improvement.
        """
        with client_for(build_api_matcher()) as client:
            assert "inFlight" not in status_of(client)["limits"]


# =============================================================================
# THRESHOLDS
# =============================================================================


class TestThresholds:
    """The numbers in force, and the floor that bounds what they can mean."""

    def test_the_live_matchers_numbers_are_reported_not_the_shipped_defaults(self):
        tuned = MatchingConfig(
            auto_approve_threshold=0.91,
            review_threshold=0.71,
            min_confidence_gap=0.04,
            results_per_field=3,
            fusion_alpha=0.8,
        )
        with client_for(build_api_matcher(config=tuned)) as client:
            reported = status_of(client)["thresholds"]

        assert reported["autoApprove"] == 0.91
        assert reported["review"] == 0.71
        assert reported["minConfidenceGap"] == 0.04
        assert reported["resultsPerField"] == 3
        assert reported["fusionAlpha"] == 0.8

    def test_the_structural_floor_is_published_as_arithmetic(self):
        """
        `semantic_weight * fusion_alpha` -- 0.70 * 0.90 for the shipped wiring. Asserted
        against a hand-computed constant rather than against the property that produces it,
        so this states an absolute expectation instead of an identity.
        """
        with client_for(build_api_matcher()) as client:
            reported = status_of(client)["thresholds"]

        assert reported["minimumAchievableConfidence"] == 0.63

    def test_a_review_threshold_under_the_floor_is_reported_and_is_not_a_warning(self):
        """
        The shipped 0.50 sits below the shipped 0.63 floor, so a rank-1 candidate cannot
        fall below review on score alone -- DX-001's shape, and worth publishing.

        It is NOT a warning, deliberately. It is true of every default install, and a status
        surface that reports `degraded` on a stock deployment teaches operators to ignore the
        one field that must never be ignored.
        """
        with client_for(build_api_matcher()) as client:
            body = status_of(client)

        assert body["thresholds"]["reviewThresholdBelowFloor"] is True
        assert body["degraded"] is False
        assert body["warnings"] == []

    def test_a_setting_the_config_does_not_carry_is_null_and_never_zero(self):
        """
        `autoApprove: 0.0` would tell an operator that everything on this server
        auto-approves, and would be indistinguishable from a deployment that really
        configured 0.0. Null says "this matcher does not expose it", which is the truth.

        `reviewThresholdBelowFloor` goes null with it rather than to `false`, because false
        reads as "your review threshold is safely above the floor" and nothing here knows
        that.
        """

        class HalfConfig:
            review_threshold = 0.5
            fusion_alpha = 0.9

        class HalfMatcher:
            _config = HalfConfig()
            minimum_achievable_confidence = None

        handle = MatcherHandle()
        handle.bind(HalfMatcher())
        pool = BoundedWorkPool(max_workers=1, max_queued=0)
        try:
            reported = IntrospectionService(
                handle, MatchServiceLimits(), pool, ProvenanceRecorder()
            ).status()["thresholds"]
        finally:
            pool.shutdown()

        assert reported["autoApprove"] is None
        assert reported["minConfidenceGap"] is None
        assert reported["resultsPerField"] is None
        assert reported["review"] == 0.5
        assert reported["fusionAlpha"] == 0.9
        assert reported["minimumAchievableConfidence"] is None
        assert reported["reviewThresholdBelowFloor"] is None

    def test_a_review_threshold_above_the_floor_reports_false(self):
        """The other direction, so the flag is not a constant wearing a computation."""
        with client_for(build_api_matcher(config=MatchingConfig(review_threshold=0.7))) as client:
            body = status_of(client)

        assert body["thresholds"]["minimumAchievableConfidence"] == 0.63
        assert body["thresholds"]["reviewThresholdBelowFloor"] is False


# =============================================================================
# THE ENCODER, AND THE FLAG THIS SURFACE EXISTS FOR
# =============================================================================


class TestEncoderTier:
    """Which rung of the selection ladder is in force, and whether it was chosen."""

    @pytest.mark.parametrize(("module", "expected"), _ENCODER_TIER_BY_MODULE)
    def test_each_named_module_actually_exists(self, module, expected):
        """
        A rename would turn every tier into `custom` and every fallback into "somebody chose
        this", silently -- which is the one direction this table must not fail in.
        """
        assert find_spec(module) is not None, expected

    @pytest.mark.parametrize(("module", "expected"), _ENCODER_TIER_BY_MODULE)
    def test_a_provider_from_a_named_module_reports_its_tier(self, module, expected):
        class _Lookalike:
            pass

        _Lookalike.__module__ = module

        assert encoder_tier(_Lookalike()) == expected

    def test_a_provider_from_anywhere_else_is_custom(self):
        assert encoder_tier(object()) == "custom"

    @pytest.mark.parametrize(
        ("tier", "bundled_available", "expected"),
        [
            # The ladder's first choice, in force. Nothing fell through.
            ("bundled", True, False),
            # Fell through: a lower rung is running BECAUSE the first choice is unusable.
            # This is the silent degradation the whole surface exists for.
            ("static", False, True),
            ("transformer", False, True),
            # A lower rung running while the first choice IS available was chosen, not
            # fallen into. Reporting it as a fallback would train an operator to ignore the
            # flag.
            ("static", True, False),
            ("transformer", True, False),
            # A provider from outside this library was wired deliberately by whoever
            # constructed the matcher.
            ("custom", True, False),
            ("custom", False, False),
        ],
    )
    def test_the_fallback_truth_table(self, tier, bundled_available, expected):
        payload = _encoder_payload(
            build_api_matcher()._embedding_provider,
            tier=tier,
            bundled_available=bundled_available,
        )

        assert payload["fallbackInForce"] is expected

    def test_the_probe_answers_a_boolean_for_this_install(self):
        """
        Not "is it True here" -- that is a property of the machine. What is asserted is that
        asking is safe and answerable, because the status route calls it on every request.
        """
        assert isinstance(bundled_encoder_available(), bool)

    def test_the_encoder_block_names_the_model_actually_in_use(self, real_client):
        encoder = status_of(real_client)["encoder"]

        assert encoder["provider"] == "BagOfTokensProvider"
        assert encoder["modelName"] == "bag-of-tokens"
        assert encoder["tier"] == "custom"
        assert encoder["fallbackInForce"] is False


class TestFallbackWarningEndToEnd:
    """The warning, the `degraded` roll-up and the message, with the probe forced."""

    def _service(self, matcher: object, *, bundled: bool) -> IntrospectionService:
        handle = MatcherHandle()
        handle.bind(matcher)
        return IntrospectionService(
            handle,
            MatchServiceLimits(),
            BoundedWorkPool(max_workers=1, max_queued=0),
            ProvenanceRecorder(),
            bundled_probe=lambda: bundled,
        )

    def test_a_fallen_through_selection_is_reported_as_degraded(self, monkeypatch):
        matcher = build_api_matcher()
        # The one honest way to put a lower rung in force here: tell `encoder_tier` what it
        # would say about a provider that came from the static adapter module.
        monkeypatch.setattr(
            "nexus_matcher.presentation.api.introspect.encoder_tier", lambda _provider: "static"
        )
        service = self._service(matcher, bundled=False)

        body = service.status()

        assert body["encoder"]["fallbackInForce"] is True
        assert body["degraded"] is True
        assert [warning["code"] for warning in body["warnings"]] == ["FALLBACK_ENCODER"]
        assert "bulk run" in body["warnings"][0]["message"]

    def test_the_same_tier_with_the_first_choice_available_is_not_degraded(self, monkeypatch):
        monkeypatch.setattr(
            "nexus_matcher.presentation.api.introspect.encoder_tier", lambda _provider: "static"
        )
        service = self._service(build_api_matcher(), bundled=True)

        body = service.status()

        assert body["encoder"]["fallbackInForce"] is False
        assert body["degraded"] is False


# =============================================================================
# PROVENANCE
# =============================================================================


class TestProvenance:
    """Where the dictionary came from, and the one case where the answer is honestly null."""

    def test_an_injected_matcher_reports_neither_a_source_nor_an_indexed_at(self, real_client):
        """
        That matcher was indexed before it arrived. A stamp taken at bind time would record
        when the OBJECT was handed over and publish it as when the INDEX was built.
        """
        dictionary = status_of(real_client)["dictionary"]

        assert dictionary["source"] is None
        assert dictionary["indexedAt"] is None
        assert dictionary["entryCount"] == 5

    def test_a_server_loaded_dictionary_is_stamped_with_its_source_and_the_time(self, monkeypatch):
        """
        `_load_configured_matcher` is stubbed: what is under test is the RECORDING, and the
        real loader would spend a model load to prove something about a timestamp.
        """
        loaded = build_api_matcher()
        monkeypatch.setattr(app_module, "_load_configured_matcher", lambda _environ: loaded)

        handle = MatcherHandle()
        provenance = ProvenanceRecorder()
        before = datetime.now(timezone.utc)
        _bring_up_matcher(
            handle,
            AppState(),
            {"NEXUS_API_DICTIONARY": "  /srv/glossary.xlsx  "},
            app_module.get_logger("test"),
            optional=False,
            provenance=provenance,
        )

        recorded = provenance.current
        assert recorded.source == "/srv/glossary.xlsx"
        assert recorded.indexed_at is not None
        assert recorded.indexed_at >= before

    def test_a_failed_load_stamps_nothing(self, monkeypatch):
        """
        A load that failed indexed nothing. Naming a source for it would let the status body
        name a glossary this server is not answering out of.
        """

        def _explode(_environ):
            raise ValueError("no such file")

        monkeypatch.setattr(app_module, "_load_configured_matcher", _explode)

        provenance = ProvenanceRecorder()
        _bring_up_matcher(
            MatcherHandle(),
            AppState(),
            {"NEXUS_API_DICTIONARY": "/srv/missing.xlsx"},
            app_module.get_logger("test"),
            optional=False,
            provenance=provenance,
        )

        assert provenance.current.source is None
        assert provenance.current.indexed_at is None

    def test_the_recorded_instant_reaches_the_wire_as_utc_iso_8601(self):
        provenance = ProvenanceRecorder()
        provenance.record(
            source="/srv/glossary.csv",
            indexed_at=datetime(2026, 8, 19, 12, 30, 5, tzinfo=timezone.utc),
        )
        handle = MatcherHandle()
        handle.bind(build_api_matcher())
        pool = BoundedWorkPool(max_workers=1, max_queued=0)
        try:
            body = IntrospectionService(handle, MatchServiceLimits(), pool, provenance).status()
        finally:
            pool.shutdown()

        assert body["dictionary"]["source"] == "/srv/glossary.csv"
        assert body["dictionary"]["indexedAt"] == "2026-08-19T12:30:05+00:00"


# =============================================================================
# DEGRADED READS OFF A MATCHER THAT IS ONLY PARTLY THERE
# =============================================================================


def test_a_matcher_without_a_config_reports_no_thresholds_and_still_answers():
    """
    `FakeMatcher` carries a `_config` but no encoder and no entry map. The status route has
    to survive every one of those: a diagnostic that raises on a half-built matcher is a
    diagnostic that is unavailable exactly when somebody needs it.
    """
    with client_for(FakeMatcher()) as client:
        body = status_of(client)

    assert body["ready"] is True
    assert body["encoder"] is None
    assert body["dictionary"]["entryCount"] is None
    assert body["thresholds"]["autoApprove"] == 0.87
