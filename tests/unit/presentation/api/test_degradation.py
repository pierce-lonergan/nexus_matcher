"""
tests.unit.presentation.api.test_degradation | Layer: TEST
The endpoint fails deterministically: it sheds, it times out, it refuses -- it never hangs.

## Relationships
# TESTS → presentation/api/limits :: admission control and the server-side deadline
# TESTS → presentation/api/matching :: MatcherHandle and matcher-failure mapping

The adopter driving this feature runs 5 s connect / 30 s read timeouts and a fallback
path, and has already been burned by an endpoint that HUNG: the connect succeeded, the
read never returned, and the fallback never fired because nothing had failed. Every test
here is about the status code arriving at all.

## No timing assertions

Nothing below asserts "responded within N ms". Other agents run concurrently on this
machine and H-007 measured a 30.6% throughput band on identical code at 49.5% CPU busy,
so a latency assertion here would be a coin toss and a flaky gate teaches people to
ignore red. What IS asserted is correctness under a deadline that is orders of magnitude
smaller than the work it bounds -- a 0.05 s deadline over a 1 s sleep cannot be resolved
the other way by any amount of machine load.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.errors import DeadlineExceededError, OverloadedError
from nexus_matcher.presentation.api.limits import (
    BoundedWorkPool,
    MatchServiceLimits,
    run_bounded,
)
from tests.unit.presentation.api._support import FakeMatcher, request_fields

BODY = {"fields": request_fields()}


def app_for(matcher: object | None, limits: MatchServiceLimits | None = None):
    return create_app(configure_logs=False, matcher=matcher, limits=limits, environ={})


# =============================================================================
# NOT READY
# =============================================================================


class TestNoMatcher:
    """A server with no dictionary says so, on every request, with the fix in the body."""

    def test_matching_is_503_and_names_the_setting_to_change(self):
        with TestClient(app_for(None)) as client:
            response = client.post("/api/v1/match", json=BODY)

        assert response.status_code == 503, response.text
        message = response.json()["error"]["message"]
        assert "NEXUS_API_DICTIONARY" in message

    def test_a_broken_dictionary_leaves_readiness_red_rather_than_looking_healthy(self):
        """
        A configured-but-unloadable dictionary must not present as a healthy server.

        The alternative -- starting up green and failing one request at a time -- is the
        shape that gets a bad deploy through a rollout gate.
        """
        app = create_app(
            configure_logs=False,
            environ={"NEXUS_API_DICTIONARY": "does-not-exist.xlsx"},
        )
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 503
            assert client.get("/health").json()["status"] == "degraded"
            match = client.post("/api/v1/match", json=BODY)

        assert match.status_code == 503, match.text
        assert "does-not-exist.xlsx" in match.json()["error"]["message"]

    def test_health_and_readiness_are_untouched_when_no_matcher_is_configured(self):
        """
        The service was health-and-introspection only before this feature. Registering a
        `matcher` component unconditionally would turn every existing /health/ready into
        a 503 on upgrade -- a breaking change wearing a feature's clothes.
        """
        with TestClient(app_for(None)) as client:
            ready = client.get("/health/ready")

        assert ready.status_code == 200, ready.text
        assert "matcher" not in ready.json()["components"]


# =============================================================================
# MATCHER FAILURE
# =============================================================================


def test_a_matcher_that_raises_becomes_a_named_500_not_a_hang():
    """
    A model error has to reach the client as a status code, with the cause named, so the
    adopter's fallback fires and their operator can tell which 500 they got without
    server access.
    """
    matcher = FakeMatcher(raises=RuntimeError("onnxruntime session died"))
    with TestClient(app_for(matcher)) as client:
        response = client.post("/api/v1/match", json=BODY)

    assert response.status_code == 500, response.text
    error = response.json()["error"]
    assert "onnxruntime session died" in error["message"]
    assert error["details"]["cause"] == "RuntimeError"


# =============================================================================
# THE DEADLINE
# =============================================================================


def test_a_slow_match_returns_504_rather_than_hanging():
    """
    The one behaviour this endpoint exists to guarantee.

    A 1 s sleep under a 0.05 s deadline. If the deadline were removed the request would
    return 200 after a second and this assertion would fail -- which is what makes it a
    gate rather than a description.
    """
    matcher = FakeMatcher(delay_seconds=1.0)
    limits = MatchServiceLimits(deadline_seconds=0.05, max_workers=2, max_queued=4)

    with TestClient(app_for(matcher, limits)) as client:
        response = client.post("/api/v1/match", json=BODY)

    assert response.status_code == 504, response.text
    assert response.json()["error"]["details"]["deadline_seconds"] == 0.05


async def test_the_deadline_does_not_hand_the_permit_back_early():
    """
    The subtle half, and the reason shedding and the deadline are one design.

    A CPU-bound Python thread cannot be interrupted, so a 504 ends the RESPONSE and not
    the work. If the permit came back when the caller gave up, a client retrying on
    timeout would stack thread after thread behind requests nobody is waiting for -- the
    unbounded queue this module exists to prevent, reached by a different road. The
    permit must stay held until the worker really finishes.
    """
    pool = BoundedWorkPool(max_workers=1, max_queued=0)
    release = threading.Event()

    def slow() -> str:
        release.wait(timeout=30.0)
        return "done"

    try:
        with pytest.raises(DeadlineExceededError):
            await run_bounded(pool, slow, deadline_seconds=0.05)

        assert pool.in_flight == 1, (
            "the permit came back while the worker was still running, so a retrying "
            "client could stack unbounded work behind timed-out requests"
        )
        with pytest.raises(OverloadedError):
            await run_bounded(pool, lambda: "second", deadline_seconds=0.05)

        release.set()
        deadline = time.monotonic() + 30.0
        while pool.in_flight and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert pool.in_flight == 0, "the permit was never returned after the work finished"

        assert await run_bounded(pool, lambda: "third", deadline_seconds=5.0) == "third"
    finally:
        release.set()
        pool.shutdown()


async def test_work_cancelled_before_it_starts_returns_its_permit():
    """
    A request shed by the deadline while still QUEUED really is cancelled, and its permit
    comes back at once. Otherwise a burst of timeouts would permanently shrink capacity
    for the life of the process.
    """
    pool = BoundedWorkPool(max_workers=1, max_queued=4)
    release = threading.Event()

    def blocker() -> str:
        release.wait(timeout=30.0)
        return "blocker"

    try:
        occupying = asyncio.ensure_future(run_bounded(pool, blocker, deadline_seconds=30.0))
        await asyncio.sleep(0.05)

        with pytest.raises(DeadlineExceededError):
            await run_bounded(pool, lambda: "queued", deadline_seconds=0.05)

        deadline = time.monotonic() + 30.0
        while pool.in_flight > 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert pool.in_flight == 1, "a queued-and-cancelled unit of work kept its permit"

        release.set()
        assert await occupying == "blocker"
    finally:
        release.set()
        pool.shutdown()


# =============================================================================
# SHEDDING
# =============================================================================


class TestAdmission:
    """Over capacity the answer is 503 immediately, not a longer queue."""

    def test_capacity_is_workers_plus_queue_and_submit_refuses_past_it(self):
        pool = BoundedWorkPool(max_workers=2, max_queued=3)
        release = threading.Event()
        try:
            assert pool.capacity == 5
            futures = [pool.submit(lambda: release.wait(timeout=30.0)) for _ in range(5)]
            assert all(f is not None for f in futures)
            assert pool.in_flight == 5
            assert pool.submit(lambda: None) is None, "the sixth was admitted anyway"
        finally:
            release.set()
            pool.shutdown()

    def test_a_shut_down_pool_sheds_instead_of_raising(self):
        pool = BoundedWorkPool(max_workers=1, max_queued=0)
        pool.shutdown()
        assert pool.submit(lambda: None) is None

    async def test_the_endpoint_sheds_with_503_while_a_worker_is_busy(self):
        """
        End to end, through the real ASGI app, with two genuinely concurrent requests.

        Driven with an in-process async client rather than two threads: both requests run
        as tasks on one event loop, so what is being observed is the server's admission
        control and not the test harness's thread safety.
        """
        matcher = FakeMatcher()
        matcher.release.clear()
        limits = MatchServiceLimits(deadline_seconds=20.0, max_workers=1, max_queued=0)
        app = app_for(matcher, limits)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as ac:
            try:
                first = asyncio.ensure_future(ac.post("/api/v1/match", json=BODY))
                assert await asyncio.to_thread(matcher.started.wait, 20.0)

                shed = await ac.post("/api/v1/match", json=BODY)
                assert shed.status_code == 503, shed.text
                details = shed.json()["error"]["details"]
                assert details["capacity"] == 1
                assert details["in_flight"] == 1
            finally:
                matcher.release.set()

            accepted = await first
            assert accepted.status_code == 200, accepted.text


# =============================================================================
# LIMITS CONFIGURATION
# =============================================================================


class TestLimits:
    """The numbers an operator sets, and what happens when they set nonsense."""

    def test_defaults_leave_headroom_under_the_clients_read_timeout(self):
        """
        A server deadline above the client's read timeout is the hang, reintroduced by an
        off-by-one in seconds: the client always gives up first and never sees the 504.
        """
        assert MatchServiceLimits().deadline_seconds < 30.0
        assert MatchServiceLimits().max_batch_fields == 250

    def test_the_environment_is_read_for_every_limit(self):
        limits = MatchServiceLimits.from_env(
            {
                "NEXUS_API_DEADLINE_SECONDS": "2.5",
                "NEXUS_API_MAX_WORKERS": "7",
                "NEXUS_API_MAX_QUEUED": "0",
                "NEXUS_API_MAX_FIELDS": "11",
                "NEXUS_API_MAX_BATCH_FIELDS": "13",
            }
        )
        assert limits == MatchServiceLimits(2.5, 7, 0, 11, 13)

    def test_an_unset_variable_falls_back_to_the_default(self):
        assert MatchServiceLimits.from_env({}) == MatchServiceLimits()
        assert MatchServiceLimits.from_env({"NEXUS_API_MAX_WORKERS": "  "}) == (
            MatchServiceLimits()
        )

    @pytest.mark.parametrize(
        "environ",
        [
            {"NEXUS_API_DEADLINE_SECONDS": "25s"},
            {"NEXUS_API_MAX_WORKERS": "four"},
            {"NEXUS_API_MAX_BATCH_FIELDS": "2.5"},
        ],
    )
    def test_an_unparseable_value_is_refused_rather_than_silently_defaulted(self, environ):
        """
        `=.25s` silently reverting to 25.0 is a hundredfold difference between what the
        operator configured and what the server does. Same standard as a mistyped
        matching-config key: a quietly discarded tuned number is worse than a loud failure.
        """
        with pytest.raises(ValueError, match="NEXUS_API_"):
            MatchServiceLimits.from_env(environ)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"deadline_seconds": 0.0},
            {"deadline_seconds": -1.0},
            {"max_workers": 0},
            {"max_queued": -1},
            {"max_fields": 0},
            {"max_batch_fields": 0},
        ],
    )
    def test_a_configuration_that_could_not_work_fails_at_construction(self, kwargs):
        """
        A zero deadline times out every request and a zero worker count hangs every
        request. Both are silent until traffic arrives, which is the worst time to learn.
        """
        with pytest.raises(ValueError):
            MatchServiceLimits(**kwargs)
