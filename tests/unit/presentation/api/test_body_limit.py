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

What is asserted instead is the mechanism: the handler is never entered, and no more than
the cap plus the chunk that crossed it is ever pulled from `receive`. Both are false on
every shape that does not intercept `receive`.

## No clock

`test_degradation.py` forbids latency assertions by name -- other agents run concurrently
on this machine and a flaky gate teaches people to ignore red -- so nothing here measures
time or memory. Bytes pulled is the same claim, made deterministically.
"""

from __future__ import annotations

import json
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
    """What one raw ASGI request produced: status, headers, body, and the bytes pulled."""

    def __init__(
        self, status: int, headers: list[tuple[bytes, bytes]], body: bytes, pulled: int
    ) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.pulled = pulled

    def json(self) -> Any:
        return json.loads(self.body)

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

    async def receive() -> dict[str, Any]:
        index = state["index"]
        if index >= len(chunks):
            return {"type": "http.disconnect"}
        state["index"] = index + 1
        state["pulled"] += len(chunks[index])
        return {
            "type": "http.request",
            "body": chunks[index],
            "more_body": state["index"] < len(chunks),
        }

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = _scope(len(body) if declare_length else None, request_id)
    await app(scope, receive, send)

    starts = [m for m in messages if m["type"] == "http.response.start"]
    assert len(starts) == 1, f"the request was answered {len(starts)} times: {messages!r}"
    payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    headers = [(bytes(k), bytes(v)) for k, v in starts[0].get("headers", [])]
    return Driven(int(starts[0]["status"]), headers, payload, int(state["pulled"]))


def guarded(max_bytes: int = CAP) -> BodySizeLimitMiddleware:
    """The real app, wrapped exactly as `create_app` wraps it."""
    app = create_app(configure_logs=False, matcher=FakeMatcher(), environ={})
    return BodySizeLimitMiddleware(app, max_bytes=max_bytes)


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


async def test_nothing_past_the_cap_is_pulled_off_the_wire(handler_entries):
    """
    The assertion that separates this fix from every shape that does not work.

    A dependency, a `max_length`, or the shipped `max_fields` check all answer 413 having
    first pulled and parsed every byte -- that is the +2179 MiB. The allowance is exactly
    one chunk: the message that crosses the cap has already been handed over by the server
    when it is counted, and chunk size is the server's choice, not ours.
    """
    oversized = body_of_exactly(CAP * 16)

    result = await drive(guarded(), oversized)

    assert result.status == 413
    assert result.pulled <= CAP + CHUNK, (
        f"pulled {result.pulled} of {len(oversized)} bytes for a {CAP}-byte cap; the body "
        f"was read into memory before being refused"
    )
    assert handler_entries == []


async def test_a_declared_content_length_over_the_cap_is_refused_without_reading_a_byte(
    handler_entries,
):
    """
    The cheap fast path, and only that. It cannot be the mechanism -- a client that omits
    the header, or a proxy that rewrites it, would walk straight past -- but when the
    client does declare an oversized body there is no reason to read any of it.
    """
    result = await drive(guarded(), body_of_exactly(CAP * 16), declare_length=True)

    assert result.status == 413, result.body
    assert result.pulled == 0, f"{result.pulled} bytes were read despite a declared length"
    assert handler_entries == []


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
