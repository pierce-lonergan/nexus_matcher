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
"""

from __future__ import annotations

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
