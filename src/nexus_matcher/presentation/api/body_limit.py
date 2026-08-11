"""
nexus_matcher.presentation.api.body_limit | Layer: PRESENTATION
A byte cap on the request body, enforced before the body is buffered or parsed.

## Relationships
# DEPENDS_ON → presentation/api/errors :: RequestTooLargeError, the 413 envelope
# DEPENDS_ON → presentation/api/limits :: MatchServiceLimits.body_byte_cap supplies max_bytes
# USED_BY    → presentation/api/app :: added to the middleware stack by create_app

## The defect this exists to close

`MatchService.match` applies `max_fields` only AFTER FastAPI has read the whole body and
built every `FieldSpec`: `fastapi/routing.py` does `await request.body()` and then
`json.loads` on the event loop before any validation runs. So the 413 was real and the
memory was spent anyway. Measured against a real uvicorn, RSS on the child process:

    76 MiB body                    413 in 1.33 s     +546 MiB    /health/live 1262 ms
    198 MiB body                   413               +808 MiB    /health/live  275 ms
    198 MiB chunked, no length     413               +600 MiB    /health/live  269 ms
    304 MiB body, 2.0 s deadline   413 in 5.15 s    +2179 MiB    /health/live 4574 ms
    8 x 198 MiB concurrent         8 x 413          +3588 MiB    ZERO 503s

The last row is the one that matters. `BoundedWorkPool` wraps `_invoke_matcher` only, so
these requests were never admitted and never shed -- on a memory-capped container there is
no 413, there is an OOMKill, and nothing sheds first. That is the precise failure
`limits.py` opens by saying its design prevents. For scale: the largest LEGITIMATE batch
body, 250 fields at every declared `max_length`, is 2.36 MB.

## Three shapes that were tried and do not work

**`max_length` on `MatchRequest.field_specs`.** A wall-time REGRESSION on the attack path:
on the shipped pydantic 2.12.5 the identical 400k-field payload goes from 1.446 s to
2.147 s, because pydantic-core reports `list_too_long` AFTER validating every item. It
also does nothing about the `json.loads` half, which is the larger one, and it converts
the 413 into a 422 -- changing the status the adopter's chunking branch keys on.

**A FastAPI dependency.** Measured no-op: +605 MB against the shipped +611 MB, versus
+0 MB here. A dependency runs inside `solve_dependencies`, which is after the body is
buffered AND parsed. It refuses a request whose cost has already been paid.

**`app.middleware("http")`.** Wrong shape, not merely slower: that wraps the function in
`BaseHTTPMiddleware`, which hands the app a `Request` and never lets the middleware see
`receive`. Intercepting `receive` is the entire mechanism, so this has to be raw ASGI.

## Why it drains a BOUNDED body before answering, and refuses the rest outright

Answering and closing without reading anything loses the 413 outright, at a rate that only
shows up on a connection that already carried a request. Closing a socket that still has
unread bytes in its receive buffer sends RST rather than FIN, and an RST discards whatever
the peer had already buffered -- including the 413 itself. Measured against a real uvicorn,
raw sockets, one non-blocking loop writing the body and reading the response at once
(which is what a pooled `java.net.http.HttpClient` does):

    as shipped, content-length     warm 392/400 readable    8 LOST
    as shipped, content-length     fresh 392/400            8 LOST
    as shipped, chunked/streaming  warm 396/400             4 LOST
    as shipped, chunked/streaming  fresh 396/400            4 LOST

The loss surfaces client-side as `fixed content-length: 403, bytes received: 0`: headers
in, body gone. It is NOT specific to the fast path -- the streaming path loses it too --
and it is not specific to a pooled connection either; a fresh connection loses it just as
often, so the 0/12 that made this look solid was small-n, not a different mechanism.

The obvious fix, a bounded partial drain, is measured WORSE THAN NOTHING:

    drain 0 (as shipped)           warm 392/400    fresh 377/400
    drain 64 KiB                   warm 387/400    fresh 382/400
    drain 256 KiB                  warm 375/400    fresh 364/400
    drain to the END of the body   warm 400/400    fresh 400/400

A partial drain cannot help by construction. The client refills the receive buffer far
faster than a bounded read empties it, so there is still unread data at `close()` and it is
still an RST -- the drain only moved when. Only reaching the end of the body removes the
reset, because only then is there nothing left unread. **So a drain that cannot finish is
pure cost, and the rule this module follows is: never start one that cannot finish, and
abandon one that is not going to.**

That is what makes the Content-Length fast path the RIGHT place to drain rather than the
wrong one. It is the one path that knows the body's size before reading a byte, so it can
decide up front whether a drain would succeed. Inside the budget it drains and the 413 is
readable; outside it, nothing is read at all and the behaviour is exactly what shipped,
including the +0 MiB. Falling back to the counting path instead would be strictly worse:
it spends a whole cap of reads and STILL ends with unread bytes, which is precisely the
396/400 the streaming rows above measure.

Draining does not reinstate the +3.5 GiB, and the asymmetry is the reason: a drained chunk
is DISCARDED, so retained memory is O(chunk) no matter how long the drain runs. uvicorn
hands over at most `HIGH_WATER_LIMIT` (64 KiB) per `receive` and pauses reading past it.
The cost of draining is time and bandwidth, never memory, so both bounds are on those:

  * BYTES -- `_DRAIN_MAX_MULTIPLE` x `max_bytes` in total for a refused request. The cap
    already carries a x4 UTF-8 factor and generous framing, so a caller inside 2x it is a
    caller whose chunking is slightly off -- the case that motivated this is 10474502
    bytes against a 9873024-byte cap, 1.06x. Beyond 2x is not a mis-chunked batch, and it
    is refused for free exactly as before. Derived from the cap, so an operator who raises
    `NEXUS_API_MAX_BATCH_FIELDS` does not inherit a stale drain budget either.
  * TIME -- `_DRAIN_SECONDS`, enforced on EACH `receive` and not merely between them. An
    unbounded drain is a slow-loris hole, and a drain that waits forever on a client that
    stopped sending is the 654 s hang that `expectContinue` produced, reintroduced from
    the other side. 2.0 s is below uvicorn's own 5 s `timeout_keep_alive`, so a drain never
    holds a connection longer than an idle keep-alive already does -- this change does not
    widen the slow-loris window at all. Blowing either bound degrades to what shipped: the
    413 goes out immediately and may be lost. That is a worse outcome, not a new one.

`connection: close` stays on the refusal unconditionally. The measurement above says the
close is not the cause -- drain-to-end plus close is 400/400 -- the unread bytes are, and
keeping a connection alive on the strength of our own byte count is how a framing bug
becomes a request-smuggling bug. `observed_bytes` likewise still reports what made the
server refuse, not what the drain went on to count: this is a transport fix, and the 413's
body is a contract a generated client already parses.

Nothing past the cap is ever handed to the APP, and nothing is ever accumulated -- that,
not the status code, is the property worth testing: a test asserting only `413` passes on
the broken code and on the no-op dependency alike.

## Why the correlation headers are stamped HERE

Being outermost is what makes the mechanism work and it is also what put this response
outside `request_id_middleware`, which sets `X-Request-ID` and `X-Response-Time-Ms` on
everything it wraps. Measured on the shipped app: 404, 422 and 503 carry both headers and
the 413 carried neither -- lost by exactly the caller most likely to need it, the one
sending oversized batches and trying to find out which of them was refused.

The streaming path was worse than a missing header. Driving the real app with a chunked
39.5 MB body: the client got a **413** with no id, while the server logged one
`http_request` line reading **`status_code: 400`** under a correlation id the client never
saw -- FastAPI turning the disconnect into its own response, which `guarded_send` then
suppressed. Two records of one request, agreeing on neither the status nor the id.

So the id is minted here, with the same recipe `app.py` uses (an inbound `X-Request-ID`
wins, else eight hex characters of a uuid4), and when one had to be minted it is INJECTED
INTO THE SCOPE before the app is called. That is the part that makes the two records
joinable: `request_id_middleware` reads the header off the request, so it adopts the id
this layer already committed to instead of minting a second one.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from nexus_matcher.presentation.api.errors import RequestTooLargeError

# The ASGI protocol, spelled out rather than imported from `starlette.types`.
#
# Two reasons, and the second is the load-bearing one. This middleware depends on the
# PROTOCOL and on nothing Starlette provides -- that is what makes it able to intercept
# `receive`, which `BaseHTTPMiddleware` cannot. And `tests/packaging/test_extras_graph`
# gates every third-party module `src/` imports against a declared extra, so importing a
# package for four type aliases would put a name in that graph for no runtime dependency
# at all. Identical to Starlette's own definitions.
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# =============================================================================
# THE REFUSAL
# =============================================================================

_REQUEST_ID = b"x-request-id"

# How many bytes a REFUSED request may cost in reads, as a multiple of the cap, and how
# long those reads may take. Both are bounds on the drain and neither is a bound on
# memory: a drained chunk is discarded, so retained bytes are O(chunk) throughout. See the
# module docstring for the measurements behind each number -- in particular why a drain
# that cannot finish inside these is not worth starting at all.
_DRAIN_MAX_MULTIPLE = 2
_DRAIN_SECONDS = 2.0


def _refusal_body(max_bytes: int, observed: int, source: str) -> bytes:
    """
    The 413 body, in the same `{"error": {...}}` envelope as every other failure.

    Built from `RequestTooLargeError` rather than hand-written, so the code, the status
    and the envelope come from the one place `errors.py` defines them. This middleware
    sits outside the exception handlers -- it is raw ASGI, below nothing that would render
    an exception for it -- so it renders the error itself, and reusing the class is what
    stops that from becoming a second error shape.
    """
    error = RequestTooLargeError(
        # "without being parsed", not "without being read": an over-budget body really is
        # refused unread, but one inside the drain budget is now read and thrown away so
        # the connection closes cleanly enough for this very message to survive. Neither is
        # ever parsed, which is the claim the caller is acting on.
        message=(
            f"The request body is larger than this server's limit of {max_bytes} bytes, "
            f"and was refused without being parsed. The cap is derived from the field cap "
            f"on /api/v1/match/batch, so a chunked request inside that cap is inside this "
            f"one; send fewer fields per request."
        ),
        # `observed` is what the client DECLARED on the content-length path and what was
        # actually counted on the streaming one. `source` says which, because a caller
        # debugging a 413 against a proxy that rewrites content-length needs to know.
        details={"limit_bytes": max_bytes, "observed_bytes": observed, "source": source},
    )
    return json.dumps(
        error.to_dict(), ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")


async def _refuse(
    send: Send,
    max_bytes: int,
    observed: int,
    source: str,
    request_id: bytes,
    started: float,
) -> None:
    """Send the 413 as raw ASGI messages and ask the client not to reuse the socket."""
    body = _refusal_body(max_bytes, observed, source)
    elapsed_ms = (time.perf_counter() - started) * 1000
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                # Unconditional, even when the drain reached the end of the body and the
                # socket is clean. Measured, the close is not what loses the 413 -- unread
                # bytes at close time are, and drain-to-end plus close is 400/400 -- and
                # deciding to keep a connection alive on the strength of our own byte count
                # is how a framing bug turns into a request-smuggling bug.
                (b"connection", b"close"),
                # Spelled the same way `request_id_middleware` spells them -- Starlette
                # lowercases a header name on its way out, so these are the bytes the other
                # statuses already carry, and an operator's existing grep finds all of them.
                (_REQUEST_ID, request_id),
                (b"x-response-time-ms", f"{elapsed_ms:.2f}".encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _drain(receive: Receive, budget: int, seconds: float) -> None:
    """
    Pull and DISCARD the rest of a refused body, bounded on bytes and on wall time.

    Reaching the END of the body is the only outcome worth anything, because a partially
    drained request still has unread bytes on the socket when the server closes, so it
    still resets, so the 413 is still lost -- measured, a 256 KiB drain lost MORE 413s than
    draining nothing at all. Both callers already know whether a drain could finish before
    they start one, so there is nothing for this to report back and nothing either of them
    would do differently: over budget or out of time, the refusal is sent exactly as it was
    before this function existed.

    Each chunk is counted and dropped on the next iteration, never appended, so this holds
    O(chunk) no matter how big `budget` is. That is what makes draining affordable at all:
    the thing this middleware refuses to spend is memory, and a drain spends time.

    The deadline is applied to EACH `receive`, not just between them. A client that stops
    sending mid-body never produces another message, and awaiting one without a timeout is
    an unbounded hang -- the same defeat-every-fallback failure that ruled out answering
    `100 Continue` on the client side, arriving from the server's end instead.
    """
    drained = 0
    ends_at = time.monotonic() + seconds
    while drained < budget:
        remaining = ends_at - time.monotonic()
        if remaining <= 0:
            return
        try:
            message = await asyncio.wait_for(receive(), timeout=remaining)
        except TimeoutError:
            # Cancelling this `receive` is safe precisely here: the caller answers 413 and
            # asks for the socket to be closed, and never reads from `receive` again. Only
            # the timeout is caught -- a `CancelledError` from the server shutting down is
            # not this middleware's to swallow.
            return
        if message["type"] != "http.request":
            # `http.disconnect` -- the client is gone, so there is nothing left unread and
            # nothing left to lose.
            return
        drained += len(message.get("body", b""))
        if not message.get("more_body", False):
            return


def _correlation_id(scope: Scope) -> tuple[bytes, bool]:
    """
    The id this request will be answered and logged under, and whether it was minted here.

    Same rule as `app.request_id_middleware` -- the caller's own `X-Request-ID` wins, and
    otherwise eight hex characters of a uuid4 -- because an operator joining a client's
    413 to a server log is joining two records produced by these two layers. A different
    recipe here would produce two ids for one request that happen to look alike.
    """
    for name, value in scope.get("headers", ()):
        if name == _REQUEST_ID:
            return value, False
    return str(uuid.uuid4())[:8].encode("ascii"), True


def _declared_length(scope: Scope) -> int | None:
    """
    The client's own `Content-Length`, when it sent one and it parses.

    A cheap fast path and nothing more: httpx omits the header entirely for a generator
    body, and the measured chunked case cost the same +600 MiB, so the counting path below
    is the one that actually carries the guarantee. An unparseable value falls through to
    counting, which is the safe direction -- trusting a malformed header either way would
    let a lie decide.
    """
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


# =============================================================================
# THE MIDDLEWARE
# =============================================================================


class BodySizeLimitMiddleware:
    """
    Refuse an oversized request body while it is still on the wire.

    Register it so it wraps the router -- `app.add_middleware(BodySizeLimitMiddleware,
    max_bytes=limits.body_byte_cap)` -- and pass `MatchServiceLimits.body_byte_cap`, never
    a literal: the cap is derived from `max_batch_fields` and `FieldSpec`'s own
    `max_length`s, so an operator who raises the field cap does not silently inherit a
    stale byte cap and a 413 naming the wrong limit.

    The mechanism is one wrapped `receive`. Every `http.request` message is counted as it
    passes through, and the moment the running total crosses the cap the app is handed
    `http.disconnect` instead of the rest of the body. Nothing past the cap reaches the
    app and nothing is accumulated, so the memory is never allocated rather than being
    allocated and then regretted.

    Before the 413 goes out, a BOUNDED remainder of the refused body is pulled and thrown
    away, because a server that closes on unread bytes resets the connection and an RST
    destroys the 413 the client had already buffered. `drain_multiple` and `drain_seconds`
    bound that in bytes and in time; both default to the measured values in the module
    docstring and exist as arguments so a test can pin them without a clock.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        drain_multiple: int = _DRAIN_MAX_MULTIPLE,
        drain_seconds: float = _DRAIN_SECONDS,
    ) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes!r}")
        if drain_multiple < 1:
            raise ValueError(
                f"drain_multiple must be >= 1, got {drain_multiple!r}; a refused body is "
                f"larger than the cap by definition, so a multiple below 1 could never "
                f"finish a drain and would only spend bandwidth losing the 413 anyway"
            )
        if drain_seconds <= 0:
            raise ValueError(f"drain_seconds must be > 0, got {drain_seconds!r}")
        self.app = app
        self.max_bytes = max_bytes
        self.drain_seconds = drain_seconds
        # The TOTAL reads a refused request may cost, counted from the first byte rather
        # than from the refusal, so both paths are bounded by the same number even though
        # the streaming one has already spent a cap's worth by the time it refuses.
        self.drain_budget = max_bytes * drain_multiple

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        request_id, minted = _correlation_id(scope)
        if minted:
            # Handed DOWN, so the request-id middleware adopts this id rather than minting
            # a second one. Without it the refused request exists twice under two ids: the
            # 413 the client holds, and the `http_request` line the server wrote for the
            # 400 that FastAPI produced from the disconnect. A new list, not an append, so
            # nothing mutates a scope the server may still own.
            scope["headers"] = [*scope.get("headers", ()), (_REQUEST_ID, request_id)]

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            # The one path that knows the size before reading a byte, and therefore the one
            # path that can tell in advance whether a drain would finish. A drain that
            # cannot finish does not save the 413 and still spends the bandwidth, so an
            # over-budget body is refused having read NOTHING -- byte for byte the
            # behaviour that shipped, on exactly the requests the cap exists to refuse.
            if declared <= self.drain_budget:
                await _drain(receive, self.drain_budget, self.drain_seconds)
            await _refuse(send, self.max_bytes, declared, "content-length", request_id, started_at)
            return

        seen = 0
        refused = False
        answered = False

        async def limited_receive() -> Message:
            nonlocal seen, refused
            if refused:
                return {"type": "http.disconnect"}

            message = await receive()
            if message["type"] != "http.request":
                return message

            seen += len(message.get("body", b""))
            if seen <= self.max_bytes:
                return message

            refused = True
            # `answered` can only be True if the app responded before reading its body,
            # which no route here does. If it ever happens, the response already on the wire
            # wins -- two `http.response.start` messages is a protocol violation, and
            # breaking the protocol is worse than serving a 200 to an oversized body.
            if not answered:
                # No declared length here, so unlike the fast path there is no way to know
                # whether the rest fits the budget -- it is drained speculatively and
                # abandoned if it does not. `seen` is already spent, so the remaining
                # allowance is the difference, which keeps both paths under one number.
                # The RAW `receive`, never `limited_receive`: this IS `limited_receive`.
                await _drain(receive, self.drain_budget - seen, self.drain_seconds)
                await _refuse(send, self.max_bytes, seen, "stream", request_id, started_at)
            return {"type": "http.disconnect"}

        async def guarded_send(message: Message) -> None:
            nonlocal answered
            if refused:
                # We have already answered. The app is still unwinding -- FastAPI turns the
                # disconnect into its own error response -- and forwarding that would send
                # a second response on the same request.
                return
            if message["type"] == "http.response.start":
                answered = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # Only once the 413 is already sent. The disconnect makes the app raise
            # somewhere -- `ClientDisconnect` from Starlette, or whatever FastAPI wraps it
            # in -- and letting that escape would make the server log a failed request for
            # one this middleware answered correctly. Any exception before the refusal is
            # a real error and is re-raised untouched.
            if not refused:
                raise
