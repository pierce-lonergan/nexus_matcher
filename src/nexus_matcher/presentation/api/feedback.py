"""
nexus_matcher.presentation.api.feedback | Layer: PRESENTATION
POST /api/v1/feedback -- append-only recording of reviewer verdicts.

## Relationships
# DEPENDS_ON → presentation/api/errors :: MatcherUnavailableError, MatchFailedError
# DEPENDS_ON → presentation/api/schemas :: FeedbackRequest
# USED_BY    → presentation/api/app :: mounted by create_app

## Recording is the whole scope, and that is a MEASURED decision

This endpoint does not feed ranking, and nothing here is read back by the matcher. That
is not an unfinished half of the feature -- it is the honest scope.

Fine-tuning the encoder on this exact signal was measured on this repository's benchmark
and it LOST accuracy: P@1 fell by 0.0277 (0.5651 -> 0.5374) and the gold-vs-runner-up
margin fell with it, over 1628 training pairs and 15 minutes of training
(`benchmarks/results/exp_finetune_transfer.json`). Wiring reviewer feedback into ranking
and claiming it improves matching would therefore be a false claim, made about the one
number this library exists to protect.

What the record IS good for is the thing it is named after: an audit trail of who decided
what, when, and against which governance id. That is worth keeping on its own, and it is
also the raw material for measuring, later, whether any use of it helps -- which is a
different exercise from asserting that it does.

## Append-only, and what that costs

Every write is `open(..., "a")` plus one line plus flush plus fsync. Nothing in this module
opens the file for reading, truncating or seeking, so a recorded verdict cannot be edited
or lost by this code path -- which is the property that makes it usable as evidence.

fsync per request is genuinely slow, and that is accepted here and nowhere else: this is a
human-rate surface (a reviewer clicking a button), not the matching hot path. Without it,
a record is "written" into the OS page cache and a power loss silently drops the tail of
the audit trail -- and an audit trail that is silently incomplete is worse than none,
because it gets cited.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, status

from nexus_matcher.presentation.api.errors import MatcherUnavailableError, MatchFailedError
from nexus_matcher.presentation.api.schemas import FeedbackRequest, FeedbackResponseView

if TYPE_CHECKING:
    from nexus_matcher.presentation.api.matching import DeterministicJSONResponse as _Response

# The key order of a stored record. Fixed here so the file is diffable and two servers
# recording the same verdict produce the same line -- a JSONL audit trail whose key order
# wanders cannot be compared across hosts.
_RECORD_KEYS: tuple[str, ...] = (
    "ts",
    "receivedAt",
    "reviewer",
    "field",
    "doc",
    "chosenGovernanceId",
    "suggestedGovernanceId",
    "wasCorrect",
)


class FeedbackRecorder:
    """
    Appends one JSON object per line to a file, under a lock, and never reads it back.

    The lock is not belt-and-braces. Two uvicorn worker THREADS in one process appending
    concurrently can interleave partial writes, and a torn line in an audit trail is
    indistinguishable from a tampered one. Multiple uvicorn PROCESSES are a different
    problem this lock does not solve -- see the follow-up note in the module docstring of
    the tests; single-process is the documented deployment for this endpoint today.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Where records are appended. Reported in the response for the operator."""
        return self._path

    def record(self, record: dict[str, Any]) -> None:
        """
        Append exactly one line. Raises OSError if it cannot be written.

        `ensure_ascii=True` for the same reason as the HTTP response: the file stays pure
        ASCII, so a reviewer name with an accent survives a Windows console, a legacy code
        page, and every log pipeline in between as an escape rather than as mojibake.
        `json.loads` returns the original string either way.
        """
        line = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # "a" and nothing else. No "r+", no seek, no truncate: the append-only promise
            # is a property of this call, not of a convention somebody remembers to follow.
            with self._path.open("a", encoding="ascii", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def _stored_record(request: FeedbackRequest) -> dict[str, Any]:
    """
    Build the record in `_RECORD_KEYS` order.

    `ts` is the client's own timestamp, stored verbatim and NOT parsed: rejecting a
    reviewer's verdict because their clock format is unusual would lose the verdict, and
    the verdict is the thing worth keeping. `receivedAt` is the server's UTC stamp and is
    the field to order by -- a client clock is not evidence about when a review happened.
    """
    values: dict[str, Any] = {
        "ts": request.ts,
        "receivedAt": datetime.now(timezone.utc).isoformat(),
        "reviewer": request.reviewer,
        "field": request.field_path,
        "doc": request.doc,
        "chosenGovernanceId": request.chosenGovernanceId,
        "suggestedGovernanceId": request.suggestedGovernanceId,
        "wasCorrect": request.wasCorrect,
    }
    return {key: values[key] for key in _RECORD_KEYS}


def create_feedback_router(
    recorder: FeedbackRecorder | None,
    response_class: type[_Response],
) -> APIRouter:
    """
    Mount the feedback route.

    `recorder=None` means recording is not configured. The route still EXISTS and answers
    503 with the setting to change: a 404 would tell a client the endpoint does not exist
    in this build, which is a different and wrong diagnosis.
    """
    router = APIRouter(prefix="/api/v1", tags=["Feedback"])

    @router.post(
        "/feedback",
        status_code=status.HTTP_201_CREATED,
        response_class=response_class,
        response_model=None,
        responses={
            201: {"model": FeedbackResponseView},
            422: {"description": "Malformed record. The body names the offending field."},
            503: {"description": "Feedback recording is not configured on this server."},
        },
        summary="Record a reviewer's verdict on a match",
        description=(
            "Appended to an audit log. NOT used for ranking: fine-tuning on this signal "
            "was measured on this project's benchmark and lost 0.0277 P@1, so recording "
            "is the honest scope and no accuracy claim is made for it."
        ),
    )
    async def submit_feedback(request: FeedbackRequest) -> dict[str, Any]:
        if recorder is None:
            raise MatcherUnavailableError(
                message=(
                    "Feedback recording is not configured on this server. Set "
                    "NEXUS_API_FEEDBACK_PATH to a writable file, or pass "
                    "feedback_path=... to create_app()."
                ),
                details={"setting": "NEXUS_API_FEEDBACK_PATH"},
            )

        record = _stored_record(request)
        try:
            recorder.record(record)
        except OSError as exc:
            # A 201 for a verdict that was never written is the worst outcome available:
            # the reviewer believes their decision is on file and it is not.
            raise MatchFailedError(
                message=(
                    f"The feedback record could not be written to "
                    f"{recorder.path}: {type(exc).__name__}: {exc}. Nothing was recorded."
                ),
                details={"cause": type(exc).__name__},
                cause=exc,
            ) from exc

        # Echoed back so the reviewer sees exactly what was stored, `receivedAt` included.
        return {"recorded": True, "record": record}

    return router
