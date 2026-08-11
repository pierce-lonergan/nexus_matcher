"""
nexus_matcher.presentation.api.errors | Layer: PRESENTATION
Every way this API is allowed to fail, each pinned to one status code.

## Relationships
# DEPENDS_ON → shared/exceptions :: NexusMatcherError envelope and ErrorCode
# USED_BY    → presentation/api/matching :: raised by the matching endpoints
# USED_BY    → presentation/api/feedback :: raised by the feedback endpoint
# USED_BY    → presentation/api/app :: registers the validation handler

## Why these exist rather than raising HTTPException

The adopter driving this feature runs a Java pipeline with a 5 s connect / 30 s read
timeout and a fallback path, and has already been burned once by an endpoint that HUNG.
A hang defeats every one of those controls: the connect succeeded, the read never
returned, and the fallback never fired because nothing failed. So the contract this
module encodes is *deterministic degradation* -- the server always answers, and the
status code alone is enough for a client to decide what to do next:

    413  the request is too big for this server; chunk it and retry
    422  the request is malformed, and the body says exactly what is wrong
    503  no dictionary is loaded, or the server is shedding load; retry later
    504  the server-side deadline fired; the work may still be running, retry later
    500  the matcher itself failed, or this layer detected its own invariant broken

Each subclasses `APIError`, so `nexus_exception_handler` in `app.py` already renders it
into the `{"error": {"code", "message", "details"}}` envelope the rest of the service
uses. Raising `fastapi.HTTPException` instead would produce `{"detail": ...}` and give
the same service two error shapes, which is the sort of thing a client library ends up
handling with a string test.

## And the failures nobody here chooses to raise

That paragraph was true and insufficient, because the second envelope shipped anyway.
Starlette raises `HTTPException` for an unknown path and for a wrong verb whether this
module approves or not, and a census against a live app found 404, 405 and both health
503s answering `{"detail": ...}` while every `/api/v1/*` failure answered `{"error":
{...}}`. So `http_exception_handler` below renders THOSE into the same envelope: the
choice of exception class is about which status and `code` a failure gets pinned to, not
about which shape reaches the client, because a client only ever gets one shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from nexus_matcher.shared.exceptions import APIError, ErrorCode

if TYPE_CHECKING:
    from fastapi import Request

# =============================================================================
# THE FAILURE MODES
# =============================================================================


@dataclass
class MatcherUnavailableError(APIError):
    """
    No matcher is loaded, so no field can inherit anything.

    503 and not 500: nothing is broken, the service is simply not carrying a dictionary
    yet. That is a retryable condition for the caller and a configuration problem for the
    operator, and conflating it with "the matcher blew up" would send the adopter's
    fallback down the wrong branch.
    """

    code: ErrorCode = ErrorCode.INITIALIZATION
    status_code: int = 503


@dataclass
class OverloadedError(APIError):
    """
    The bounded in-flight queue is full, so this request is SHED rather than queued.

    Queueing without limit is how a service turns a load spike into a timeout storm:
    every caller waits, every caller's read timeout expires, and the work is done anyway
    for responses nobody is still listening to. Refusing immediately is the honest
    answer, and it is the one the adopter's fallback path is built for.
    """

    code: ErrorCode = ErrorCode.API
    status_code: int = 503


@dataclass
class DeadlineExceededError(APIError):
    """
    The server-side deadline fired before matching finished.

    504 rather than 500 because the request was well-formed and the server was working;
    it simply ran out of the time the operator allowed. `details.deadline_seconds` says
    what that budget was, so an operator reading a client's logs can tell a too-tight
    server deadline from a genuinely slow match without server access.

    The worker thread is NOT killed -- CPU-bound Python cannot be interrupted -- so this
    is a promise about the RESPONSE, not about the work. The in-flight permit is held
    until the thread really finishes, which is what stops a stream of timed-out requests
    from piling threads up behind them.
    """

    code: ErrorCode = ErrorCode.MATCHING_TIMEOUT
    status_code: int = 504


@dataclass
class RequestTooLargeError(APIError):
    """Too many fields in one request. `details.limit` is the server's cap."""

    code: ErrorCode = ErrorCode.API_INVALID_REQUEST
    status_code: int = 413


@dataclass
class MalformedRequestError(APIError):
    """
    A request that parsed but cannot be answered, with the reason in `message`.

    422 rather than 400 so it sits alongside FastAPI's own body-validation failures: a
    client only has to learn one status for "your request is wrong", and the envelope is
    identical either way.
    """

    code: ErrorCode = ErrorCode.API_INVALID_REQUEST
    status_code: int = 422


@dataclass
class MatchFailedError(APIError):
    """
    The matcher raised. A clean 500 with the cause named, never a hang and never a 200.

    The alternative -- letting the exception escape to `generic_exception_handler` -- also
    produces a 500, but an anonymous one ("An unexpected error occurred"), and it produces
    it by a path that depends on middleware ordering. A caller integrating against this
    endpoint should not have to discover which 500 it got by reading server logs.
    """

    code: ErrorCode = ErrorCode.MATCHING
    status_code: int = 500


@dataclass
class ConservationViolationError(APIError):
    """
    A field went in and did not come back out under its own name.

    This is NM-0005 as an HTTP failure. The whole point of the endpoint is that a field
    inherits the governance of the entry it matched; a field missing from the response
    inherits nothing, and the only symptom is a map with fewer keys than the caller sent
    -- a count nobody has reason to check. A 500 that says so is strictly better than a
    200 that quietly drops a column, so the response is refused rather than trimmed.

    See `matching._project_results` for the three independent checks that raise this.
    """

    code: ErrorCode = ErrorCode.MATCHING
    status_code: int = 500


@dataclass
class ContractDriftError(APIError):
    """
    Another layer renamed or dropped something this layer reads.

    Same stance as the CLI's `_drift`: a governance document that omits the classification
    it exists to carry, or whose numbers do not reproduce its own confidence, is worse
    than no document -- it looks complete and gets used as evidence. So the response is
    refused and the message names both sides of the drift.
    """

    code: ErrorCode = ErrorCode.VALIDATION
    status_code: int = 500


def drift(owner: str, attribute: str, consequence: str) -> ContractDriftError:
    """Build the standard "this layer and `owner` have drifted" 500."""
    return ContractDriftError(
        message=(
            f"{owner} has no {attribute!r}, so {consequence} Refusing to answer with a "
            f"response that cannot be trusted. Report this: the HTTP matching surface and "
            f"{owner} have drifted."
        ),
        details={"owner": owner, "attribute": attribute},
    )


# =============================================================================
# THE FAILURES THE FRAMEWORK RAISES
# =============================================================================


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Render Starlette's own `HTTPException` into this service's one error envelope.

    404 and 405 are raised by the router, not by any code in this package, so they were
    the half of the two-envelope defect nobody could fix by choosing a better exception
    class. The health probes raise `HTTPException` deliberately -- see `app.py` -- and
    arrive here too.

    ## `headers` is forwarded, and that is not housekeeping

    A 405 on `PUT /api/v1/match` carries `Allow: POST`, which is how a client discovers
    the verb without reading a document. `JSONResponse(status_code=..., content=...)` --
    the obvious thing to copy from `validation_exception_handler` below -- drops it, so
    unifying the envelope that way would trade a client-library annoyance for a protocol
    defect. `getattr` rather than `exc.headers` because this is registered against a base
    class and a caller may raise a bare one.

    ## Why a `dict` detail is understood

    `HTTPException(detail=...)` takes any object. A string is the message; a mapping
    carrying `message` lets a raiser attach structured context -- which is how
    `/health/ready` puts its component map in the body instead of answering "Service not
    ready" and leaving the operator to guess which component is red.

    ## The codes

    Anything below 500 is the caller addressing this service wrongly, which is what
    `API_INVALID_REQUEST` already means for a malformed body. 5xx gets the generic API
    code and NOT `INITIALIZATION` (NEXUS-1002), whose docstring pins it to "no dictionary
    is loaded": a probe 503 means the process has not finished starting, and sending an
    operator down the dictionary branch for that is the mistake that class warns about.
    """
    status_code = int(getattr(exc, "status_code", 500))
    raw_detail: Any = getattr(exc, "detail", "")

    extra: dict[str, Any] = {}
    if isinstance(raw_detail, Mapping):
        message = str(raw_detail.get("message", ""))
        extra = {str(key): value for key, value in raw_detail.items() if key != "message"}
    else:
        message = str(raw_detail)

    code = ErrorCode.API_INVALID_REQUEST if status_code < 500 else ErrorCode.API
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code.value,
                "message": message,
                "details": {"status_code": status_code, **extra},
            }
        },
        headers=getattr(exc, "headers", None),
    )


# =============================================================================
# BODY VALIDATION
# =============================================================================


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Render FastAPI's own body-validation failure into this service's error envelope.

    Without this, a malformed body returns `{"detail": [...]}` while every other failure
    returns `{"error": {...}}`, so a client has to branch on the shape of the error before
    it can read the error. The pydantic violations are preserved verbatim under
    `details.violations` -- that list is the part that says WHAT is wrong, which is the
    whole reason this is a 422 and not a 500.

    `str()` on each violation's context, because pydantic puts the offending VALUE in
    `ctx`/`input` and those can be arbitrary objects that `json.dumps` refuses; a
    validation error that itself 500s on serialisation is the worst of both worlds.
    """
    violations: list[dict[str, Any]] = []
    if isinstance(exc, RequestValidationError):
        for raw in exc.errors():
            violations.append(
                {
                    "location": [str(part) for part in raw.get("loc", ())],
                    "message": str(raw.get("msg", "")),
                    "type": str(raw.get("type", "")),
                }
            )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": ErrorCode.API_INVALID_REQUEST.value,
                "message": (
                    "The request body is not valid. See details.violations for the exact "
                    "fields and why each was rejected."
                ),
                "details": {"status_code": 422, "violations": violations},
            }
        },
    )
