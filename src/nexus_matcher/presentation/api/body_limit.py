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

## Why it answers and closes rather than draining

Reading a refused 304 MiB body to EOF just to keep the connection reusable spends exactly
the bandwidth and time the cap exists to refuse. So the 413 goes out immediately with
`connection: close`, which tells a keep-alive client not to reuse the socket and gets the
status in front of it. Nothing beyond the cap is ever pulled from `receive` -- that, not
the status code, is the property worth testing: a test asserting only `413` passes on the
broken code and on the no-op dependency alike.

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
        message=(
            f"The request body is larger than this server's limit of {max_bytes} bytes, "
            f"and was refused without being read. The cap is derived from the field cap "
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
                # The request body was NOT read to EOF, so this connection has a half-sent
                # request on it. Saying so is what lets the client read the 413 instead of
                # discovering a reset on its next request.
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
    passes through, and the moment the running total crosses the cap the 413 goes out and
    the app is handed `http.disconnect` instead of the rest of the body. Nothing past the
    cap is pulled, so the memory is never allocated rather than being allocated and then
    regretted.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes!r}")
        self.app = app
        self.max_bytes = max_bytes

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
