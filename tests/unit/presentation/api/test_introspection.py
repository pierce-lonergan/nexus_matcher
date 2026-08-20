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
from tests.unit.presentation.api._support import (
    FakeMatcher,
    build_api_matcher,
    request_fields,
)

STATUS_KEYS = (
    "ready",
    "degraded",
    "warnings",
    "dictionary",
    "encoder",
    "thresholds",
    "limits",
    # APPENDED. `calibration` answers NM-V2-03 SC-7 -- which corpus the shipped defaults
    # were fitted on -- and NM-V2-01 AR-4's "expose the active profile"; a caller reading
    # keys positionally must find the seven above unmoved.
    "calibration",
)

# The `thresholds` block, in order. A literal rather than a read-back, so a member silently
# dropped from the payload fails here instead of being confirmed by itself.
THRESHOLD_KEYS = (
    "autoApprove",
    "review",
    "minConfidenceGap",
    "resultsPerField",
    "fusionAlpha",
    "minimumAchievableConfidence",
    "reviewThresholdBelowFloor",
    "absoluteScoreFloor",
    "absoluteScoreMetric",
)

# How many entries a dictionary needs before the shipped defaults stop being a statement
# about anything resembling it: ten times the 688-entry calibration corpus. Written out
# rather than imported, so a change to either number has to be made twice on purpose.
UNCALIBRATED_ENTRY_COUNT = 6881


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
# THE ABSOLUTE FLOOR -- the one threshold this surface used to send you elsewhere for
# =============================================================================


class TestTheAbsoluteScoreFloorIsVisible:
    """
    NM-V2-01 AR-4 / NM-V2-03 SC-6: an operator must be able to see which calibration is in
    force. `absolute_score_floor` is the threshold a deployment is most likely to have set,
    and the only one that produces a verdict -- NO_MATCH -- that nothing else on this body
    explains. It used to be readable only by sending a match and reading `scoring`.
    """

    def test_the_block_publishes_every_threshold_in_order(self, real_client):
        assert tuple(status_of(real_client)["thresholds"]) == THRESHOLD_KEYS

    def test_a_configured_floor_is_reported_with_the_metric_it_is_compared_against(self):
        tuned = MatchingConfig(absolute_score_floor=0.42)
        with client_for(build_api_matcher(config=tuned)) as client:
            reported = status_of(client)["thresholds"]

        assert reported["absoluteScoreFloor"] == 0.42
        # A floor is a number in whatever space the wired store measures. Publishing the
        # floor without the metric would let a caller compare it against a cosine on a
        # server whose store returns dot products.
        assert reported["absoluteScoreMetric"] == "cosine"

    def test_no_floor_configured_is_null_and_that_is_the_shipped_default(self, real_client):
        """
        Null here means NO FLOOR, not "unreadable" -- the opposite convention from the rest
        of this block, and the reason it is pinned: with no floor this deployment cannot
        emit NO_MATCH at all, which is a fact about its verdicts and not a missing field.
        """
        assert status_of(real_client)["thresholds"]["absoluteScoreFloor"] is None

    def test_the_status_floor_is_the_same_number_a_match_response_publishes(self):
        """
        Two surfaces, one server, one floor. A status body and a `scoring` block that
        disagreed would leave an operator unable to say which one governs the verdicts they
        are looking at.
        """
        tuned = MatchingConfig(absolute_score_floor=0.31)
        with client_for(build_api_matcher(config=tuned)) as client:
            from_status = status_of(client)["thresholds"]["absoluteScoreFloor"]
            matched = client.post("/api/v1/match", json={"fields": request_fields()[:1]})

        assert matched.status_code == 200, matched.text
        assert from_status == matched.json()["scoring"]["absoluteScoreFloor"] == 0.31


# =============================================================================
# CALIBRATION -- which profile is in force, and what the shipped one was fitted on
# =============================================================================


class TestTheCalibrationBlock:
    """
    NM-V2-03 SC-7. A threshold is a statement about a score distribution, so a number
    fitted on one corpus means something else on another. Until this block existed the
    corpus behind the shipped numbers reached no machine-readable interface at all -- it
    was a comment in `MatchingConfig`, which an HTTP consumer never sees.
    """

    def test_the_shipped_defaults_name_the_corpus_they_were_measured_on(self, real_client):
        corpus = status_of(real_client)["calibration"]["corpus"]

        # Size, so a caller can compare it against their own dictionary; the splits and
        # domains, so they can judge whether it is anything like their data; the artifact,
        # so they can go and read the measurement rather than trust this block.
        assert corpus["fields"] == 688
        assert corpus["dictionaryEntries"] == 688
        assert corpus["splits"] == {"bird": 361, "omop": 327}
        assert corpus["artifact"] == "benchmarks/results/exp_calibration_combined.json"
        assert corpus["autoApproveThreshold"] == 0.87
        assert corpus["autoApprovePrecision"] == 0.952941
        assert len(corpus["domains"]) == 2

    def test_the_published_threshold_is_the_one_the_corpus_block_was_measured_at(self, real_client):
        """
        The pairing is the whole point. A corpus block quoting a precision measured at a
        threshold this server does not run would be a fact about nothing.
        """
        body = status_of(real_client)

        assert (
            body["thresholds"]["autoApprove"]
            == body["calibration"]["corpus"]["autoApproveThreshold"]
        )
        assert body["calibration"]["defaultsInForce"] is True

    def test_a_stock_deployment_reports_no_overrides(self, real_client):
        assert status_of(real_client)["calibration"]["overrides"] == {}

    def test_every_changed_setting_is_named_with_its_live_value(self):
        """
        The answer to "which calibration is in force". Derived from the config dataclass's
        own fields, so a setting nobody thought to publish -- the way
        `absolute_score_floor` itself went unpublished -- still shows up the day a
        deployment changes it.
        """
        tuned = MatchingConfig(
            auto_approve_threshold=0.8,
            absolute_score_floor=0.25,
            dictionary_alias_count=6,
            expand_query_abbreviations=True,
            dense_top_k=50,
        )
        with client_for(build_api_matcher(config=tuned)) as client:
            response = client.get("/api/v1/status")
            calibration = response.json()["calibration"]

        # A populated block, against the PUBLISHED schema: a flag rendered as `1.0` or a
        # count widened to a float would validate against a loose model and mislead an
        # operator reading it back.
        StatusResponseView.model_validate(response.json())
        assert response.content.decode("ascii") == response.text
        assert calibration["overrides"] == {
            "dense_top_k": 50,
            "auto_approve_threshold": 0.8,
            "absolute_score_floor": 0.25,
            "expand_query_abbreviations": True,
            "dictionary_alias_count": 6,
        }
        assert calibration["overrides"]["expand_query_abbreviations"] is True
        # A deployment that moved a decision threshold has calibrated deliberately.
        assert calibration["defaultsInForce"] is False

    def test_the_override_keys_are_the_names_an_operator_actually_sets(self):
        """
        snake_case, not the camelCase of the rest of the wire, and deliberately: these keys
        are what goes into a `NEXUS_API_MATCHING_CONFIG` file, and a key an operator has to
        transliterate before using is a key they will get wrong.
        """
        with client_for(build_api_matcher(config=MatchingConfig(review_threshold=0.6))) as client:
            overrides = status_of(client)["calibration"]["overrides"]

        assert list(overrides) == ["review_threshold"]

    def test_a_setting_left_at_its_shipped_value_is_not_reported_as_an_override(self):
        """A profile that restates the defaults has changed nothing, and must not read as
        though it had."""
        restated = MatchingConfig(auto_approve_threshold=0.87, review_threshold=0.5)
        with client_for(build_api_matcher(config=restated)) as client:
            assert status_of(client)["calibration"]["overrides"] == {}

    def test_the_corpus_is_published_even_with_nothing_loaded(self):
        """
        It describes this BUILD, not this deployment -- so a consumer deciding whether to
        adopt the library at all can read what the shipped numbers were fitted on before
        pointing it at a dictionary. The three live members go null with `thresholds`.
        """
        with client_for(None) as client:
            calibration = status_of(client)["calibration"]

        assert calibration["corpus"]["fields"] == 688
        assert calibration["defaultsInForce"] is None
        assert calibration["overrides"] is None
        assert calibration["dictionarySizeRatio"] is None

    def test_the_size_ratio_is_this_dictionary_against_that_corpus(self, real_client):
        body = status_of(real_client)

        assert body["dictionary"]["entryCount"] == 5
        assert body["calibration"]["dictionarySizeRatio"] == round(5 / 688, 6)
        assert body["calibration"]["warnAboveSizeRatio"] == 10.0


class TestTheUncalibratedSizeWarning:
    """
    AR-4's fourth clause: warn loudly when a deployment runs on defaults against a
    dictionary that does not resemble the calibration corpus.

    Only SIZE is warned on, and only upward. Domain and naming style are described on
    `calibration.corpus` for a human to compare, because measuring them would need a
    similarity metric this library has never validated -- and a warning computed from an
    invented metric is wrong in a direction nobody can audit.
    """

    def _status_for(self, entry_count: int, config: MatchingConfig | None = None) -> dict:
        """
        A status body for a dictionary of a stated size, without indexing that many entries.

        `dictionary_size` is the public accessor the payload reads, so a stub that reports a
        number is the whole of what this rule consumes. Indexing 6,881 real entries to
        assert an arithmetic comparison would buy nothing and cost a minute per test.
        """

        class SizedMatcher:
            _config = config or MatchingConfig()
            dictionary_size = entry_count
            minimum_achievable_confidence = 0.63
            absolute_score_floor = None
            absolute_score_metric = "cosine"

        handle = MatcherHandle()
        handle.bind(SizedMatcher())
        pool = BoundedWorkPool(max_workers=1, max_queued=0)
        try:
            return IntrospectionService(
                handle, MatchServiceLimits(), pool, ProvenanceRecorder()
            ).status()
        finally:
            pool.shutdown()

    def test_shipped_defaults_against_an_order_of_magnitude_more_entries_is_a_warning(self):
        body = self._status_for(UNCALIBRATED_ENTRY_COUNT)

        assert body["degraded"] is True
        assert [warning["code"] for warning in body["warnings"]] == ["UNCALIBRATED_SIZE"]
        message = body["warnings"][0]["message"]
        # The message has to carry the arithmetic, not just the verdict: an operator who
        # cannot see both numbers cannot tell whether the rule applies to them.
        assert "688" in message and str(UNCALIBRATED_ENTRY_COUNT) in message
        assert "NEXUS_API_MATCHING_CONFIG" in message

    def test_exactly_the_ratio_is_not_yet_a_warning(self):
        """
        The rule is `>`, not `>=`. Pinned because an off-by-one here would fire on a
        dictionary at exactly the documented boundary, and the boundary is published on the
        wire as `warnAboveSizeRatio` -- a rule that disagreed with its own published number
        is worse than an unpublished one.
        """
        body = self._status_for(6880)

        assert body["calibration"]["dictionarySizeRatio"] == 10.0
        assert body["warnings"] == []

    def test_a_deployment_that_calibrated_its_own_thresholds_is_not_warned(self):
        """
        They have made the decision this warning exists to prompt. Telling them their
        numbers are uncalibrated because their dictionary is large would be false.
        """
        tuned = MatchingConfig(auto_approve_threshold=0.8)
        body = self._status_for(UNCALIBRATED_ENTRY_COUNT, config=tuned)

        assert body["calibration"]["defaultsInForce"] is False
        assert body["warnings"] == []
        assert body["degraded"] is False

    def test_a_dictionary_smaller_than_the_corpus_is_never_warned_about(self):
        """
        One direction only. Nothing here has measured the small direction, and a surface
        that reported `degraded` on every demo and every test fixture would teach operators
        to ignore the field this whole body exists for.
        """
        assert self._status_for(30)["warnings"] == []
        assert self._status_for(1)["warnings"] == []

    def test_a_setting_that_is_not_a_decision_threshold_does_not_suppress_the_warning(self):
        """
        Turning aliasing on is not calibrating a threshold. A deployment that changed a
        retrieval knob is still auto-approving at a number fitted on somebody else's corpus.
        """
        body = self._status_for(
            UNCALIBRATED_ENTRY_COUNT, config=MatchingConfig(dictionary_alias_count=6)
        )

        assert body["calibration"]["overrides"] == {"dictionary_alias_count": 6}
        assert [warning["code"] for warning in body["warnings"]] == ["UNCALIBRATED_SIZE"]


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
