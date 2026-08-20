"""
nexus_matcher.presentation.api.feedback | Layer: PRESENTATION
POST /api/v1/feedback -- append-only recording of reviewer verdicts.

## Relationships
# DEPENDS_ON → presentation/api/errors :: MatcherUnavailableError, MatchFailedError
# DEPENDS_ON → presentation/api/schemas :: FeedbackRequest (the base shape), ErrorResponse
# USED_BY    → presentation/api/app :: mounted by create_app

## This endpoint still RECORDS ONLY, and that is still a MEASURED decision

Nothing here reads the trail back, and no route in this build ranks with it. That has not
changed and it is not an unfinished half of the feature.

Fine-tuning the encoder on this exact signal was measured on this repository's benchmark
and it LOST accuracy: P@1 fell by 0.0277 (0.5651 -> 0.5374) and the gold-vs-runner-up
margin fell with it, over 1628 training pairs and 15 minutes of training
(`benchmarks/results/exp_finetune_transfer.json`). Wiring reviewer feedback into ranking
and claiming it improves matching would therefore be a false claim, made about the one
number this library exists to protect.

What HAS changed is that the trail is now CONSUMABLE by something outside this module.
`domain/ports/review_feedback.py` declares the seam and `application/feedback_loop.py`
implements the obvious consumer -- an approved-pair bypass, where a pair a human has
already decided skips matching entirely on later runs. That is a different claim from the
one the measurement above refutes: it does not make the model better, it declines to ask
the model a question a human has answered. Its precision on a seen pair is 100% by
construction, which is a tautology and must never be reported as an accuracy result.

The shipped default still consumes nothing. `create_app()` builds no consumer and
`NexusMatcher` takes `feedback_consumer=None`, so on every server this package starts, the
trail is exactly what it has always been.

## What the record IS good for

An audit trail of who decided what, when, and against which governance id -- worth keeping
on its own, and the raw material for measuring, later, whether any use of it helps, which
is a different exercise from asserting that it does.

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
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, status
from pydantic import Field, model_validator

from nexus_matcher.presentation.api import schemas
from nexus_matcher.presentation.api.errors import MatcherUnavailableError, MatchFailedError

if TYPE_CHECKING:
    from nexus_matcher.presentation.api.matching import DeterministicJSONResponse as _Response

# The key order of a stored record. Fixed here so the file is diffable and two servers
# recording the same verdict produce the same line -- a JSONL audit trail whose key order
# wanders cannot be compared across hosts.
#
# `verdict` is APPENDED rather than placed beside `wasCorrect`, so every line a previous
# build wrote is still a prefix-compatible reading of this order and a trail spanning an
# upgrade diffs cleanly through the boundary.
_RECORD_KEYS: tuple[str, ...] = (
    "ts",
    "receivedAt",
    "reviewer",
    "field",
    "doc",
    "chosenGovernanceId",
    "suggestedGovernanceId",
    "wasCorrect",
    "verdict",
)


class FeedbackRequest(schemas.FeedbackRequest):
    """
    A reviewer's verdict on one match, widened so the most valuable one survives. WC-11.

    ## The verdict a boolean cannot hold

    `wasCorrect: bool` has exactly two states and the vocabulary has three. The one it
    cannot express is the reviewer who chose a term THE MATCHER NEVER PROPOSED -- not rank
    2, not rank 20: absent from the candidate list entirely.

    Collapsed into `false`, that record is byte-identical to "the top match was wrong and I
    took the third one". Those are opposite diagnoses. The second says the candidate list
    contained the answer and the ranking put it in the wrong place, which weights, fusion
    or a reranker can fix. The first says the answer was never retrieved at all, which no
    amount of re-ranking a list that never contained it will ever fix. A pipeline that
    stores both as `false` cannot count either, and therefore cannot tell which of the two
    problems it has.

    `verdict` is that third state, and it is OPTIONAL. Every request that was valid before
    this member existed is still valid and still records the same eight keys plus a null,
    so a client that has never heard of it keeps working unchanged.

    ## Why `wasCorrect` is still required, and still checked

    It is not deprecated and it is not derived. Two members that describe one decision can
    disagree, and this model REFUSES the disagreement rather than picking a winner: an
    `APPROVED` verdict beside `wasCorrect: false` is a client bug, and answering 201 to it
    would put a record in an audit trail that argues with itself. A trail is evidence, and
    evidence that contradicts itself is worse than a 422.

    ## Why this class is here and not in `schemas.py`

    Because the OpenAPI component name is part of the published contract. Renaming
    `FeedbackRequest` would change the generated type name in every client built against
    this service, for no benefit, so the widened model keeps the name and
    `schemas.FeedbackRequest` remains what it has always been: the shape without this
    member, and now also the base class. The two should be folded into one declaration in
    `schemas.py` -- that is a single-file edit and it is the right end state.
    """

    # Spelled as a `Literal` and not as an `enum.Enum`, which is a wire decision rather
    # than a style one: an Enum becomes its own component in the published schema and
    # therefore its own hand-written Java enum, a closed type that refuses a response the
    # day a fourth value is added. A Literal renders inline on the property, so a client
    # generated from this spec gets the three values as documentation without a type that
    # breaks on the fourth.
    #
    # The domain vocabulary this mirrors, `domain.ports.review_feedback.ReviewVerdict`,
    # carries a fourth value: `UNSPECIFIED`. It is deliberately NOT offered here. It is not
    # something a reviewer can decide -- it is what a record written before this member
    # existed reads as -- and offering it on the wire would let a client assert an absence
    # of information as though it were an observation.
    verdict: Literal["APPROVED", "REJECTED", "MANUAL_OVERRIDE"] | None = Field(
        default=None,
        description=(
            "What the reviewer did. Optional; null means the shape that predates this "
            "member, which can say a suggestion was wrong but not whether the reviewer's "
            "own choice had been proposed. `APPROVED` accepted the suggestion. `REJECTED` "
            "means nothing in the glossary governs this field. `MANUAL_OVERRIDE` means the "
            "reviewer chose a term that was NOT in the candidate list -- the highest-signal "
            "record there is, because it says retrieval missed rather than mis-ranked. Must "
            "agree with `wasCorrect`: APPROVED with `wasCorrect: false`, or either other "
            "value with `wasCorrect: true`, is refused rather than reconciled."
        ),
    )

    @model_validator(mode="after")
    def _verdict_agrees_with_the_boolean(self) -> FeedbackRequest:
        """Refuse two answers to one question. See the class docstring."""
        if self.verdict is None:
            return self
        approved = self.verdict == "APPROVED"
        if approved is not self.wasCorrect:
            raise ValueError(
                f"verdict {self.verdict!r} contradicts wasCorrect={self.wasCorrect!r}. "
                f"APPROVED means the matcher's suggestion was accepted and requires "
                f"wasCorrect=true; REJECTED and MANUAL_OVERRIDE both require false. "
                f"Recording a verdict that disagrees with itself would put a record in the "
                f"audit trail that cannot be cited."
            )
        return self


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

    `verdict` is stored EXACTLY AS SENT, including null. It is not back-filled from
    `wasCorrect`, and that is deliberate: a trail is evidence, and a value nobody sent is
    not evidence about what a reviewer did. A consumer reading an older line derives what
    it can and counts what it cannot -- see
    `application.feedback_loop.verdict_from_record`, where `wasCorrect: true` reads as
    APPROVED exactly and `wasCorrect: false` reads as UNSPECIFIED because the shape that
    wrote it could not say which of two opposite things happened. Writing `"REJECTED"` here
    on a hunch would erase the distinction between a fact and an inference in the one file
    kept precisely so facts can be cited.
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
        "verdict": request.verdict,
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
        # Every failure names its BODY, not only its meaning. A description is prose and a
        # generated client cannot deserialise prose: with the model missing here and
        # present on both match routes, one Java client got a typed DTO for every way a
        # match can fail and a bare `Map` for every way a verdict can -- from one service,
        # one spec, one build step. `ErrorResponse` is the same envelope this route really
        # sends, because `errors.py` renders all three of these.
        responses={
            201: {"model": schemas.FeedbackResponseView},
            422: {
                "model": schemas.ErrorResponse,
                "description": "Malformed record. The body names the offending field.",
            },
            # Reachable, and the one a caller is least able to guess: an OSError on the
            # append is answered 500 rather than 201, because a reviewer believing a
            # verdict is on file when it is not is the worst outcome this route has.
            500: {
                "model": schemas.ErrorResponse,
                "description": "The record could not be written. Nothing was recorded.",
            },
            503: {
                "model": schemas.ErrorResponse,
                "description": "Feedback recording is not configured on this server.",
            },
        },
        summary="Record a reviewer's verdict on a match",
        description=(
            "Appended to an audit log. NOT used for ranking by this server: fine-tuning on "
            "this signal was measured on this project's benchmark and lost 0.0277 P@1, so "
            "recording is the honest scope and no accuracy claim is made for it. Send "
            "`verdict` when you can -- `MANUAL_OVERRIDE` (the reviewer chose a term that "
            "was never proposed) is the one record `wasCorrect` cannot express, and it is "
            "the one that says retrieval missed rather than mis-ranked."
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
