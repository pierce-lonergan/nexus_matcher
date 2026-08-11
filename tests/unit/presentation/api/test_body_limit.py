"""
tests.unit.presentation.api.test_body_limit | Layer: TEST
An oversized body is refused on the wire, not after it has been read and parsed.

## Relationships
# TESTS → presentation/api/body_limit :: BodySizeLimitMiddleware
# TESTS → presentation/api/limits     :: the derived cap and the pair it is validated against

## Why these assertions and not `status_code == 413`

The endpoint ALREADY answered 413 to an oversized body, by parsing all of it first: a
304 MiB body cost +2179 MiB of RSS and 5.15 s, and eight concurrent 198 MiB bodies cost
+3588 MiB with zero 503s, because `BoundedWorkPool` never saw them. So a test asserting
only the status passes on the broken code, on the measured no-op FastAPI dependency, and
on the real fix alike -- it cannot tell them apart, which makes it worse than no test.

What is asserted instead is the mechanism: the handler is never entered, no chunk handed
over is still referenced afterwards, and the reads a refused request costs are bounded.
All three are false on every shape that does not intercept `receive`.

## And the second defect: the 413 that never arrived

Measured later against a real uvicorn on raw sockets, the refusal itself was being lost
about 1-6% of the time -- the server closed on unread bytes, which sends RST, which
discards the response the client had already buffered. The fix is a bounded drain, and
what these tests assert about it is the ORDER of ASGI events: whether the body had stopped
arriving when the refusal went out. That is precisely the condition deciding FIN vs RST,
and it needs no socket.

Which tests were watched red, and on what:

  * on the SHIPPED middleware, `..._is_not_sent_while_the_client_is_still_writing` and its
    streaming twin -- the two that pin the defect itself
  * on a PARTIAL drain (the obvious fix, measured worse than no drain at all),
    `test_a_partial_drain_is_never_what_happens`
  * on a drain with the per-receive deadline removed,
    `test_a_client_that_stops_sending_mid_body_is_still_answered`, which hangs

## No clock

`test_degradation.py` forbids latency assertions by name -- other agents run concurrently
on this machine and a flaky gate teaches people to ignore red -- so nothing here measures
time, memory or a rate. The end-to-end claim is statistical (8 losses in 400 trials), and
a gate that has to run 400 sockets to be 99.97% sure is exactly the flaky gate that rule
forbids; every assertion here is instead a deterministic property of one request.
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
from typing import Any

import pytest

from nexus_matcher.presentation.api import app as app_module
from nexus_matcher.presentation.api.app import create_app
from nexus_matcher.presentation.api.body_limit import BodySizeLimitMiddleware
from nexus_matcher.presentation.api.limits import MatchServiceLimits, worst_case_body_bytes
from nexus_matcher.presentation.api.matching import MatchService
from nexus_matcher.presentation.api.schemas import MAX_FIELD_SPEC_CHARS
from tests.unit.presentation.api._support import FakeMatcher

# Small enough that the tests are instant, and the arithmetic is the same at 9.9 MB.
CAP = 4096
CHUNK = 1024

# The shipped `_DRAIN_MAX_MULTIPLE` is 2, so a refused request may cost 2 x CAP in reads.
DRAIN_BUDGET = CAP * 2
# A body over the cap and INSIDE the drain budget: the mis-chunked batch, which is the
# case the 413 was being lost on. The real one is 10474502 bytes against a 9873024 cap.
IN_BUDGET = CAP * 2
# A body so far over the cap that no drain could finish: the attack, refused for free.
OVER_BUDGET = CAP * 16

# One character that costs four UTF-8 bytes, for the worst-case body below. `max_length`
# on `FieldSpec` counts CHARACTERS, so this is a legal 8192-character `doc` that puts
# 32 KiB on the wire -- the whole reason the derived cap carries a x4 factor.
FOUR_BYTE_CHAR = "\U0001d11e"


# =============================================================================
# DRIVING THE RAW ASGI APP
# =============================================================================


def _scope(content_length: int | None = None, request_id: bytes | None = None) -> dict[str, Any]:
    """A POST /api/v1/match scope, with or without a declared length."""
    headers = [(b"host", b"api.test"), (b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    if request_id is not None:
        headers.append((b"x-request-id", request_id))
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/match",
        "raw_path": b"/api/v1/match",
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 51234),
        "server": ("api.test", 80),
    }


class Driven:
    """What one raw ASGI request produced, including the order the ASGI events happened in.

    `events` is the load-bearing field. It records every `receive` and every `send` in the
    order they occurred, which is the only way to state the property this middleware was
    losing the 413 for: the refusal must go out AFTER the body has stopped arriving, not
    while the client is still writing.
    """

    def __init__(
        self,
        status: int,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
        pulled: int,
        events: list[tuple[str, Any]],
        chunks: list[bytes],
    ) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.pulled = pulled
        self.events = events
        self.handed_over = chunks

    def json(self) -> Any:
        return json.loads(self.body)

    @property
    def body_was_consumed_before_answering(self) -> bool:
        """
        Was the request body finished arriving at the moment the response started?

        True only if a `more_body=False` receive happened before the first send. That is
        exactly the socket-level condition that decides whether the server's close is a
        FIN or an RST, and therefore whether the client can still read the 413 it has
        already buffered.
        """
        for kind, detail in self.events:
            if kind == "send":
                return False
            if kind == "receive" and detail == "last":
                return True
        return False

    @property
    def receives_before_answering(self) -> int:
        count = 0
        for kind, _detail in self.events:
            if kind == "send":
                break
            if kind == "receive":
                count += 1
        return count

    def retained_chunks(self) -> int:
        """
        How many handed-over body chunks anything is STILL holding a reference to.

        This is what turns "draining must not reinstate the +3.5 GiB" into a deterministic
        assertion rather than an RSS measurement, which the no-latency-assertions rule at
        the top of this file rules out anyway. `self.handed_over` is the only list holding
        each chunk, so a chunk the code under test let go of is referenced by that list and
        by nothing else. Anything more means it did not let go, which is exactly what a
        drain that accumulates instead of discarding looks like, and what buffering the
        whole body looks like.

        The baseline is MEASURED against a control rather than typed as a number. The
        expected count is not the obvious 2: the list holds one, `getrefcount`'s argument
        holds one, and the comprehension's own loop variable holds a third. Hard-coding it
        made this read as "8 of 8 chunks retained" on a correct implementation -- a test
        that is red on the truth is worse than no test, in the other direction.

        Plain `bytes`, no subclass: `bytes` cannot be weakly referenced at all, and a
        subclass that could would no longer be what a real server hands over.
        """
        gc.collect()
        control = [b"\x00" * CHUNK]
        baseline = max(sys.getrefcount(chunk) for chunk in control)
        return sum(1 for chunk in self.handed_over if sys.getrefcount(chunk) > baseline)

    def header(self, name: str) -> str | None:
        """One response header, matched case-insensitively as HTTP defines it."""
        wanted = name.lower().encode("ascii")
        for key, value in self.headers:
            if key.lower() == wanted:
                return value.decode("latin-1")
        return None


async def drive(
    app: Any,
    body: bytes,
    *,
    declare_length: bool = False,
    request_id: bytes | None = None,
) -> Driven:
    """
    Send `body` through `app` as an ASGI request, counting what the app pulls.

    Driven at the ASGI level rather than through `TestClient` on purpose: the count of
    bytes actually pulled from `receive` is the assertion these tests exist to make, and
    no HTTP client exposes it. The body is handed over in fixed chunks, exactly as a
    server hands over socket reads.
    """
    chunks = [body[i : i + CHUNK] for i in range(0, len(body), CHUNK)] or [b""]
    state = {"index": 0, "pulled": 0}
    messages: list[dict[str, Any]] = []
    events: list[tuple[str, Any]] = []
    # Every chunk actually handed over, held here and nowhere else in this function, so
    # `Driven.retained_chunks` can ask whether the code under test is still holding one.
    handed_over: list[bytes] = []

    async def receive() -> dict[str, Any]:
        index = state["index"]
        if index >= len(chunks):
            events.append(("receive", "disconnect"))
            return {"type": "http.disconnect"}
        state["index"] = index + 1
        chunk = chunks[index]
        state["pulled"] += len(chunk)
        more = state["index"] < len(chunks)
        handed_over.append(chunk)
        events.append(("receive", "more" if more else "last"))
        return {"type": "http.request", "body": chunk, "more_body": more}

    async def send(message: dict[str, Any]) -> None:
        events.append(("send", message["type"]))
        messages.append(message)

    scope = _scope(len(body) if declare_length else None, request_id)
    await app(scope, receive, send)
    # `chunks` and `body` are the test's own second and third references to the same
    # objects; dropped so that a retained chunk can only be the code under test holding it.
    chunks.clear()
    del body

    starts = [m for m in messages if m["type"] == "http.response.start"]
    assert len(starts) == 1, f"the request was answered {len(starts)} times: {messages!r}"
    payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    headers = [(bytes(k), bytes(v)) for k, v in starts[0].get("headers", [])]
    return Driven(
        int(starts[0]["status"]), headers, payload, int(state["pulled"]), events, handed_over
    )


def guarded(max_bytes: int = CAP, **drain: Any) -> BodySizeLimitMiddleware:
    """The real app, wrapped exactly as `create_app` wraps it."""
    app = create_app(configure_logs=False, matcher=FakeMatcher(), environ={})
    return BodySizeLimitMiddleware(app, max_bytes=max_bytes, **drain)


def body_of_exactly(nbytes: int) -> bytes:
    """A well-formed match request padded to an exact byte length."""
    payload: dict[str, Any] = {
        "fields": [{"name": "usage_litres", "path": "meter.usage_litres", "doc": ""}],
        "top_k": 1,
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("ascii")
    payload["fields"][0]["doc"] = "x" * (nbytes - len(encoded))
    encoded = json.dumps(payload, separators=(",", ":")).encode("ascii")
    assert len(encoded) == nbytes, (len(encoded), nbytes)
    return encoded


@pytest.fixture
def handler_entries(monkeypatch) -> list[str]:
    """
    Records every entry into `MatchService.match`.

    The load-bearing fixture. `FakeMatcher.calls` would NOT do: the handler's own
    `max_fields` 413 is raised before the matcher is ever called, so a zero there is also
    what the fully-parsed-then-refused path produces. Entry into `MatchService.match` is
    the first thing that happens after the body has been read and validated, so a zero
    here means the body never got that far.
    """
    entries: list[str] = []
    original = MatchService.match

    async def spy(self: MatchService, request: Any, max_fields: int) -> Any:
        entries.append("entered")
        return await original(self, request, max_fields)

    monkeypatch.setattr(MatchService, "match", spy)
    return entries


# =============================================================================
# THE REFUSAL
# =============================================================================


async def test_an_oversized_body_is_refused_before_the_handler_is_entered(handler_entries):
    """
    The whole point. The body never becomes a `MatchRequest`, so it is never parsed.

    Sent WITHOUT a declared length, which is the case that carries the guarantee: httpx
    omits `Content-Length` for a generator body, and the measured chunked attack cost the
    same +600 MiB as the declared one.
    """
    oversized = body_of_exactly(CAP * 16)

    result = await drive(guarded(), oversized)

    assert result.status == 413, result.body
    assert handler_entries == [], "the oversized body was parsed and handed to the handler"


async def test_nothing_past_the_cap_is_ever_RETAINED(handler_entries):
    """
    The assertion that separates this fix from every shape that does not work.

    A dependency, a `max_length`, or the shipped `max_fields` check all answer 413 having
    first pulled and parsed every byte -- that is the +2179 MiB. What makes those shapes
    expensive is not that the bytes were READ, it is that they were KEPT: a drained chunk
    is counted and dropped, which is why bounding the drain on time and bandwidth is enough
    and no bound on memory is needed.

    So the claim is made against retention directly rather than against a proxy for it.
    Every chunk handed over is weakly referenced, and after the request not one of them may
    still be reachable. False on anything that buffers the body, false on a drain that
    accumulates what it drains, and it costs no clock and no RSS measurement.
    """
    result = await drive(guarded(), body_of_exactly(OVER_BUDGET))

    assert result.status == 413
    assert result.retained_chunks() == 0, (
        f"{result.retained_chunks()} of {len(result.handed_over)} body chunks are still "
        f"reachable after the request was refused; the body was retained, not discarded"
    )
    assert handler_entries == []


async def test_the_reads_a_refused_request_costs_are_bounded(handler_entries):
    """
    The other half. Retention is bounded by discarding; BANDWIDTH is bounded by counting.

    An unbounded drain is a slow-loris hole -- a client can make the server read forever by
    declaring a body it dribbles out -- so the total a refused request may pull is capped
    at `_DRAIN_MAX_MULTIPLE` x the byte cap. The allowance on top is exactly one chunk: the
    message that crosses a bound has already been handed over by the server when it is
    counted, and chunk size is the server's choice, not ours.
    """
    oversized = body_of_exactly(OVER_BUDGET)

    result = await drive(guarded(), oversized)

    assert result.status == 413
    assert result.pulled <= DRAIN_BUDGET + CHUNK, (
        f"pulled {result.pulled} of {len(oversized)} bytes for a {CAP}-byte cap; the drain "
        f"is unbounded and a client that keeps writing can make this server read forever"
    )
    assert handler_entries == []


async def test_a_declared_length_beyond_the_drain_budget_is_refused_without_reading_a_byte(
    handler_entries,
):
    """
    The cheap fast path, and the control that keeps the drain below from eating it.

    Content-Length is the one signal that says how big the body is BEFORE a byte is read,
    which is what lets this path decide up front whether a drain could finish. Beyond the
    budget it could not, and a drain that cannot finish is pure cost -- it does not save
    the 413 and it still spends the bandwidth -- so nothing is read at all. This is the
    row that measured +0 MiB against 8 x 304 MiB concurrent, and it must stay zero: an
    implementation that "fixed" the lost 413 by always draining would turn this into 2432
    MiB of reads and would go red here.
    """
    result = await drive(guarded(), body_of_exactly(OVER_BUDGET), declare_length=True)

    assert result.status == 413, result.body
    assert result.pulled == 0, f"{result.pulled} bytes were read despite a declared length"
    assert result.receives_before_answering == 0
    assert handler_entries == []


# =============================================================================
# THE REFUSAL HAS TO SURVIVE THE CLOSE
# =============================================================================
#
# Measured against a real uvicorn on raw sockets, one non-blocking loop writing the body
# and reading the response at once, 400 trials per arm:
#
#     as shipped, content-length     warm 392/400 readable, 8 LOST   fresh 392/400, 8 LOST
#     as shipped, chunked            warm 396/400,          4 LOST   fresh 396/400, 4 LOST
#     after this fix, all four arms  400/400,               0 LOST
#
# The server refuses and closes while the client is still writing. Closing a socket with
# unread bytes in its receive buffer sends RST instead of FIN, and an RST discards what the
# peer had already buffered -- so the client sees `fixed content-length: 403, bytes
# received: 0`: headers in, body gone.
#
# None of that is assertable without a socket. What IS assertable, and what is exactly the
# condition that decides FIN vs RST, is the ORDER: had the body stopped arriving at the
# moment the refusal went out? These tests assert that ordering, so they need no clock, no
# server, and no repetition -- and they are red on the shipped code, where the answer on
# the fast path is "no receives at all".


async def test_a_refusal_is_not_sent_while_the_client_is_still_writing(handler_entries):
    """
    THE regression. On the shipped code this is zero receives then a send, every time.

    A body over the cap but inside the drain budget is the mis-chunked batch that motivated
    this -- 10474502 bytes against a 9873024-byte cap, 1.06x. The whole of it is pulled and
    thrown away BEFORE the 413 is written, so the server's close finds nothing unread and
    the client can still read the response it has already buffered.
    """
    result = await drive(guarded(), body_of_exactly(IN_BUDGET), declare_length=True)

    assert result.status == 413, result.body
    assert result.body_was_consumed_before_answering, (
        f"the 413 was written while the body was still arriving, so the close will reset "
        f"the connection and discard it; ASGI events were {result.events!r}"
    )
    assert handler_entries == []


async def test_the_streaming_path_does_not_send_while_the_client_is_still_writing(
    handler_entries,
):
    """
    The same defect on the path with no Content-Length to consult, measured at 4/400 lost.

    It cannot look the size up, so it drains speculatively on what is left of the budget
    and gives up if that runs out. Here it does not run out, so the ordering is the same.
    """
    result = await drive(guarded(), body_of_exactly(IN_BUDGET))

    assert result.status == 413, result.body
    assert result.json()["error"]["details"]["source"] == "stream"
    assert result.body_was_consumed_before_answering, (
        f"the streaming refusal was written mid-body; ASGI events were {result.events!r}"
    )
    assert handler_entries == []


async def test_a_partial_drain_is_never_what_happens(handler_entries):
    """
    Draining PART of a body is measured worse than draining none, so it must not be a state
    this code can reach.

        drain 0 (as shipped)   warm 392/400   fresh 377/400
        drain 64 KiB           warm 387/400   fresh 382/400
        drain 256 KiB          warm 375/400   fresh 364/400   <- worse than doing nothing
        drain to the END       warm 400/400   fresh 400/400

    The client refills the receive buffer faster than a bounded read can empty it, so a
    partial drain still leaves unread bytes at close, still resets, and has spent the
    bandwidth for nothing. Every refusal must therefore be one of two things and never
    a third: drained to the end, or not started.
    """
    for size, declared in ((IN_BUDGET, True), (OVER_BUDGET, True)):
        result = await drive(guarded(), body_of_exactly(size), declare_length=declared)

        assert result.status == 413
        consumed = result.body_was_consumed_before_answering
        untouched = result.receives_before_answering == 0
        assert consumed or untouched, (
            f"a {size}-byte body was drained only partway before the 413 went out, which "
            f"spends the bandwidth AND loses the response; events {result.events!r}"
        )
    assert handler_entries == []


async def test_a_client_that_stops_sending_mid_body_is_still_answered():
    """
    The anti-hang property, and the reason the deadline is on EACH receive rather than
    between them.

    A drain that awaits a message the client will never send waits forever. That is the
    same shape as the client-side `expectContinue` fix that was reverted -- 654 s observed,
    and the 30 s request timeout never fired -- arriving from the server's end instead. A
    hang defeats every timeout and fallback an adopter has, which is worse than the flake
    being fixed here.

    No timing is ASSERTED. `receive` here never returns at all, so without the deadline
    this test does not fail slowly, it never terminates; the `asyncio.timeout` is a
    backstop so a regression fails the suite instead of hanging it. That makes the test
    robust to a loaded machine rather than sensitive to it: a slow box makes it take
    slightly longer, never makes it red.
    """
    sends: list[dict[str, Any]] = []
    stalled = asyncio.Event()  # deliberately never set

    async def receive() -> dict[str, Any]:
        await stalled.wait()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        sends.append(message)

    app = guarded(drain_seconds=0.05)
    async with asyncio.timeout(30):
        await app(_scope(content_length=IN_BUDGET), receive, send)

    assert [m["type"] for m in sends] == ["http.response.start", "http.response.body"]
    assert sends[0]["status"] == 413


async def test_the_drain_bounds_are_refused_at_construction_like_every_other_limit():
    """
    A drain budget below the cap could never finish -- a refused body is larger than the
    cap by definition -- so it would spend bandwidth on every refusal and still lose the
    413. Rejected where the zero-byte cap is, rather than discovered from a caller.
    """
    app = create_app(configure_logs=False, matcher=FakeMatcher(), environ={})
    with pytest.raises(ValueError, match="drain_multiple"):
        BodySizeLimitMiddleware(app, max_bytes=CAP, drain_multiple=0)
    with pytest.raises(ValueError, match="drain_seconds"):
        BodySizeLimitMiddleware(app, max_bytes=CAP, drain_seconds=0)


async def test_the_refusal_uses_the_services_one_error_envelope(handler_entries):
    """
    This middleware sits outside the exception handlers and renders its own body, so the
    envelope is the thing most likely to drift into a second shape. It names the cap,
    because a caller's next move is to chunk to fit it.
    """
    result = await drive(guarded(), body_of_exactly(CAP * 16))

    body = result.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "NEXUS-8004"
    assert body["error"]["details"]["limit_bytes"] == CAP
    assert body["error"]["details"]["status_code"] == 413
    assert str(CAP) in body["error"]["message"]


# =============================================================================
# CORRELATION
# =============================================================================


@pytest.fixture
def correlation_ids(monkeypatch) -> list[str]:
    """
    Every id `request_id_middleware` handed the logger for this request.

    Spying on the setter rather than reading log output: the join an operator performs is
    between the `X-Request-ID` on their 413 and the `correlation_id` on the server's line,
    and this is that value at the moment it is committed to, with no log format in the way.
    """
    seen: list[str] = []
    original = app_module.set_correlation_id

    def spy(value: str) -> Any:
        seen.append(value)
        return original(value)

    monkeypatch.setattr(app_module, "set_correlation_id", spy)
    return seen


async def test_the_refusal_carries_the_correlation_headers_every_other_status_carries():
    """
    Measured before the fix: 404, 422 and 503 carried `X-Request-ID` and
    `X-Response-Time-Ms`; the 413 carried neither. Structural rather than sloppy -- this
    middleware has to be OUTSIDE `request_id_middleware` to reach `receive` at all, so it
    is outside the layer that stamps them -- and the caller that loses them is the one
    sending oversized batches, i.e. the one with the most requests to tell apart.

    Both statuses are driven through the same registered stack, so this asserts the
    PARITY, not merely that two headers exist.
    """
    app = create_app(configure_logs=False, matcher=FakeMatcher(), environ={})
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=CAP)

    served = await drive(app, body_of_exactly(CAP))
    refused = await drive(app, body_of_exactly(CAP * 16))

    assert (served.status, refused.status) == (200, 413), refused.body
    for response in (served, refused):
        assert response.header("X-Request-ID"), response.headers
        assert float(response.header("X-Response-Time-Ms") or "nan") >= 0.0


async def test_a_clients_own_request_id_is_the_one_the_refusal_answers_under():
    """
    A caller who stamps its own id on a batch is the caller this header exists for: it is
    already in their log, and echoing it is what makes the 413 findable from their side
    rather than only from the server's.
    """
    result = await drive(guarded(), body_of_exactly(CAP * 16), request_id=b"batch-7719")

    assert result.status == 413
    assert result.header("X-Request-ID") == "batch-7719"


async def test_a_minted_id_is_also_the_id_the_layers_below_log_under(correlation_ids):
    """
    The half a stamped header alone does not buy.

    Driven with no declared length, so the app IS entered and then handed the disconnect.
    Measured on that path: the client got a 413 and the server logged one `http_request`
    line reading `status_code: 400` -- FastAPI's own response to the disconnect, suppressed
    on its way out -- under a correlation id nothing had ever sent to the client. Two
    records of one request agreeing on neither status nor id, which is not a join.

    So the minted id is put into the scope before the app is called, and this asserts the
    layer below adopted it rather than minting a second one.
    """
    result = await drive(guarded(), body_of_exactly(CAP * 16))

    assert result.status == 413
    assert correlation_ids == [result.header("X-Request-ID")], (
        f"the 413 says {result.header('X-Request-ID')!r} and the server logged this "
        f"request under {correlation_ids!r}"
    )


async def test_a_declared_oversize_never_reaches_the_layer_that_would_log_it(correlation_ids):
    """
    The control for the test above. On the content-length path the app is never called, so
    there is no second record to join to -- and the id on the 413 is the only one that ever
    existed, which is why it must still be there.
    """
    result = await drive(guarded(), body_of_exactly(CAP * 16), declare_length=True)

    assert result.status == 413
    assert correlation_ids == []
    assert result.header("X-Request-ID")


# =============================================================================
# THE BOUNDARY
# =============================================================================


async def test_a_body_at_the_cap_is_served_normally(handler_entries):
    """
    The half that makes the cap a limit rather than a wall. Exactly at the cap is INSIDE
    it -- an off-by-one here refuses a body the published bounds accept, which is the
    failure the derived cap exists to avoid.
    """
    result = await drive(guarded(), body_of_exactly(CAP))

    assert result.status == 200, result.body
    assert handler_entries == ["entered"]
    assert list(result.json()["results"]) == ["meter.usage_litres"]


async def test_one_byte_over_the_cap_is_refused(handler_entries):
    result = await drive(guarded(), body_of_exactly(CAP + 1))

    assert result.status == 413, result.body
    assert handler_entries == []


# =============================================================================
# IN THE REAL STACK
# =============================================================================


@pytest.mark.parametrize(
    ("size", "expected", "entered"),
    [(CAP, 200, ["entered"]), (CAP * 16, 413, [])],
)
async def test_it_behaves_the_same_registered_as_middleware(
    handler_entries, size, expected, entered
):
    """
    Registered the way it ships, not wrapped by the test.

    `add_middleware` puts it OUTSIDE `request_id_middleware`, which is a
    `BaseHTTPMiddleware` and therefore runs the rest of the app in its own task. The
    refusal has to survive that: the disconnect handed downstream surfaces as an exception
    somewhere inside, and this asserts the client still gets the 413 exactly once rather
    than a 500 from the unwinding, or two responses on one request.
    """
    app = create_app(configure_logs=False, matcher=FakeMatcher(), environ={})
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=CAP)

    result = await drive(app, body_of_exactly(size))

    assert result.status == expected, result.body
    assert handler_entries == entered


# =============================================================================
# THE DERIVED CAP
# =============================================================================


def test_the_derived_cap_admits_the_largest_body_the_declared_bounds_allow():
    """
    A cap below what `FieldSpec` accepts is a 413 the caller cannot reconcile with the
    schema they generated their client from.

    Built as a REAL worst-case body -- every string at its `max_length`, every character
    four UTF-8 bytes -- and measured, rather than asserting the formula against itself.
    Three fields rather than 250 because the property is per-field and this runs instantly;
    the framing allowance is what is actually under test.
    """
    fields = [
        {
            "name": FOUR_BYTE_CHAR * 512,
            "path": FOUR_BYTE_CHAR * 1024,
            "doc": FOUR_BYTE_CHAR * 8192,
            "type": FOUR_BYTE_CHAR * 128,
        }
        for _ in range(3)
    ]
    encoded = json.dumps({"fields": fields, "top_k": 5, "explain": True}, ensure_ascii=False)
    largest = len(encoded.encode("utf-8"))

    assert largest <= worst_case_body_bytes(3), (
        f"a body of {largest} bytes is inside every declared bound and outside the cap "
        f"of {worst_case_body_bytes(3)} derived for it"
    )
    # And the shipped default is the ~10 MB the derivation implies, not the ~2.4 MB a
    # character count alone would give: 250 x 9856 characters is 2.46 M characters and
    # 9.86 M bytes. An 8 MiB cap -- the obvious round number -- sits below it.
    assert MAX_FIELD_SPEC_CHARS == 9856
    assert MatchServiceLimits().body_byte_cap == worst_case_body_bytes(250)
    assert MatchServiceLimits().body_byte_cap > 8 * 1024 * 1024


def test_the_cap_follows_the_field_cap_rather_than_a_typed_number():
    """
    An operator who raises the field cap must not silently inherit a stale byte cap, and a
    413 naming the wrong limit is the symptom that would be debugged last.
    """
    assert MatchServiceLimits(max_batch_fields=1000).body_byte_cap > (
        MatchServiceLimits(max_batch_fields=250).body_byte_cap
    )


def test_a_byte_cap_below_the_field_cap_is_refused_at_construction():
    """
    The pair, validated where every other impossible configuration is.

    8 MiB is the number the eye reaches for, and it sits BELOW what 250 fields at every
    declared `max_length` can legitimately produce -- so it would refuse requests this
    same service's published schema accepts. Rejected at construction, because the
    alternative is discovering it from a caller's 413.
    """
    with pytest.raises(ValueError, match="MAX_BODY_BYTES"):
        MatchServiceLimits(max_body_bytes=8 * 1024 * 1024)

    # Raising the field cap without raising the byte cap is the same defect arriving by
    # the other door, and it is refused the same way.
    with pytest.raises(ValueError, match="MAX_BODY_BYTES"):
        MatchServiceLimits(max_batch_fields=1000, max_body_bytes=worst_case_body_bytes(250))


def test_the_environment_configures_the_byte_cap_like_every_other_limit():
    assert MatchServiceLimits.from_env({}).max_body_bytes is None
    assert MatchServiceLimits.from_env({}).body_byte_cap == worst_case_body_bytes(250)

    generous = worst_case_body_bytes(250) * 2
    configured = MatchServiceLimits.from_env({"NEXUS_API_MAX_BODY_BYTES": str(generous)})
    assert configured.body_byte_cap == generous

    with pytest.raises(ValueError, match="NEXUS_API_MAX_BODY_BYTES"):
        MatchServiceLimits.from_env({"NEXUS_API_MAX_BODY_BYTES": "10MB"})


def test_a_cap_that_could_not_admit_anything_is_refused_at_construction():
    """A zero-byte cap refuses every request, silently, until traffic arrives."""
    app = create_app(configure_logs=False, matcher=FakeMatcher(), environ={})
    with pytest.raises(ValueError, match="max_bytes"):
        BodySizeLimitMiddleware(app, max_bytes=0)
