"""
nexus_matcher.presentation.api.limits | Layer: PRESENTATION
Deterministic degradation: a bounded in-flight queue and a server-side deadline.

## Relationships
# DEPENDS_ON → presentation/api/errors :: OverloadedError, DeadlineExceededError
# USED_BY    → presentation/api/matching :: every match request goes through run_bounded
# USED_BY    → presentation/api/app :: constructed once per application

## The requirement, and why it is not just a timeout

Matching is CPU-bound: one call encodes a batch of queries, multiplies a corpus matrix,
and scores candidates. Running that on the event loop would block every other request in
the process, including `/health/live`, so the whole point of a liveness probe would be
lost exactly when it matters. It therefore runs in a thread pool.

That creates the failure this module exists to prevent. `ThreadPoolExecutor` has an
UNBOUNDED internal queue: submit faster than the workers drain and the queue grows
without limit, latency grows with it, and every caller eventually hits its own read
timeout while the server keeps doing work for responses nobody will read. Two controls,
and they are genuinely different:

  * **Admission** (`BoundedWorkPool`) caps how much work may be outstanding at all. Over
    the cap the request is SHED with a 503 immediately, rather than accepted and starved.
  * **Deadline** (`run_bounded`) caps how long the CALLER waits. It is not the same
    number and does not do the same job: admission protects the server, the deadline
    protects the client.

## What the deadline can and cannot promise

It promises a RESPONSE by the deadline. It cannot promise the work stops: a Python thread
running CPU-bound code cannot be interrupted, and pretending otherwise is how a "timeout"
turns into a slow leak. So the in-flight permit is released when the WORKER finishes, not
when the caller gives up. A client hammering a slow server therefore gets 504s and then
503s -- admission fills up with work that is still running -- instead of an ever-growing
pile of threads. That ordering is the point: shedding is what makes the deadline safe.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from nexus_matcher.presentation.api.errors import DeadlineExceededError, OverloadedError
from nexus_matcher.presentation.api.schemas import MAX_FIELD_SPEC_CHARS

T = TypeVar("T")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment names, so an operator running the documented
# `uvicorn nexus_matcher.presentation.api.app:app` (or `nexus-matcher api`, which starts
# the same factory with no arguments) can configure this without editing code. Every one
# of these is also a `create_app` keyword argument; the environment is the path that
# exists for the two entry points that cannot pass one.
_ENV_PREFIX = "NEXUS_API_"

# `FieldSpec`'s max_lengths count CHARACTERS; the wire carries UTF-8, where a character is
# up to four bytes. Deriving the byte cap without this factor gives ~2.4 MB at the shipped
# 250-field default and refuses bodies every declared bound accepts -- a 413 the caller
# cannot reconcile with the schema they generated their client from.
_UTF8_BYTES_PER_CHAR = 4

# JSON framing around one field: four key names, eight quotes, commas and braces. 47 bytes
# for `{"name":"","path":"","doc":"","type":""},` measured; 64 leaves room for escapes.
_FRAMING_BYTES_PER_FIELD = 64

# And around the whole body: `{"fields":[...],"top_k":5,"explain":true}`, plus slack for a
# client that pretty-prints. Generous on purpose -- this number is a FLOOR under a cap, so
# being over-generous costs a little memory headroom and being tight costs correctness.
_FRAMING_BYTES = 1024


def worst_case_body_bytes(max_batch_fields: int) -> int:
    """
    The largest body this service's own declared bounds admit, in bytes.

    Derived, never typed as a literal. This is a FLOOR under any byte cap rather than a
    guess at reasonable traffic: below it, a body that `MatchRequest` and `FieldSpec` both
    accept is refused before either of them sees it, and the caller is left holding a 413
    that contradicts the schema they generated their client from. ~9.4 MiB at the shipped
    250-field default; the largest body a realistic client sends is 2.36 MB.

    One thing it does NOT cover, said out loud: a client that escapes every character as
    `\\uXXXX` puts six bytes on the wire per character, so a body inside its declared
    bounds can still exceed this. `NEXUS_API_MAX_BODY_BYTES` is the escape hatch, and the
    413 names the cap it hit.
    """
    per_field = MAX_FIELD_SPEC_CHARS * _UTF8_BYTES_PER_CHAR + _FRAMING_BYTES_PER_FIELD
    return max_batch_fields * per_field + _FRAMING_BYTES


@dataclass(frozen=True)
class MatchServiceLimits:
    """
    The numbers that bound this service, and the reasoning behind each default.

    Attributes:
        deadline_seconds: How long a caller may wait before the server answers 504.
            25.0 is deliberately BELOW the adopter's 30 s read timeout. If the server
            deadline sat above the client's, the client would always time out first and
            never see the 504 -- which is precisely the hang this endpoint was built to
            avoid, reintroduced by an off-by-one in seconds. Leave headroom when tuning.
        max_workers: Threads actually running matches. Matching releases the GIL inside
            numpy and onnxruntime, so a small number is genuinely parallel; a large one
            mostly buys memory pressure, because each concurrent match holds its own
            candidate arrays.
        max_queued: How much work may WAIT for a worker before requests are shed. Zero is
            a legitimate setting and means "never queue"; the default allows a short
            burst to ride out without turning a spike into a 503 storm.
        max_fields: Cap for POST /api/v1/match.
        max_batch_fields: Cap for POST /api/v1/match/batch. 250 is the chunk size the
            chunked client sends, so the documented client works against the default and
            a client that ignores its own chunking gets a 413 that says the limit.
        max_body_bytes: Cap on the RAW REQUEST BODY, enforced by `BodySizeLimitMiddleware`
            before the body is buffered or parsed. None -- the default -- derives it from
            `max_batch_fields` and `FieldSpec`'s own max_lengths, so raising the field cap
            raises the byte cap with it. An explicit value is an escape hatch, not the
            source of truth: an operator who raises NEXUS_API_MAX_BATCH_FIELDS and leaves
            a stale byte cap behind would get a 413 naming the wrong limit, which is why
            `__post_init__` refuses the pair rather than the number.

            Note this object is SHARED by both match routes, so /api/v1/match inherits a
            byte cap derived from the batch route's 250 fields -- 2.5x looser than its own
            100-field limit needs. That is the right trade: the cap exists to stop an
            unparsed body from exhausting memory, and its own `max_fields` 413 still fires
            on anything that gets through.
    """

    deadline_seconds: float = 25.0
    max_workers: int = 4
    max_queued: int = 32
    max_fields: int = 100
    max_batch_fields: int = 250
    max_body_bytes: int | None = None

    def __post_init__(self) -> None:
        """
        Reject a configuration that could not work, at startup rather than under load.

        A non-positive deadline would make every request time out; a non-positive worker
        count would make every request hang forever waiting for a worker that does not
        exist. Both are silent until traffic arrives, which is the worst time to find out.

        `max_body_bytes` is checked against `max_batch_fields` rather than on its own,
        because a byte cap only means anything relative to the field cap it has to admit.
        """
        if self.deadline_seconds <= 0:
            raise ValueError(f"deadline_seconds must be > 0, got {self.deadline_seconds!r}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers!r}")
        if self.max_queued < 0:
            raise ValueError(f"max_queued must be >= 0, got {self.max_queued!r}")
        if self.max_fields < 1:
            raise ValueError(f"max_fields must be >= 1, got {self.max_fields!r}")
        if self.max_batch_fields < 1:
            raise ValueError(f"max_batch_fields must be >= 1, got {self.max_batch_fields!r}")
        if self.max_body_bytes is not None:
            floor = worst_case_body_bytes(self.max_batch_fields)
            if self.max_body_bytes < floor:
                raise ValueError(
                    f"{_ENV_PREFIX}MAX_BODY_BYTES={self.max_body_bytes!r} is below the "
                    f"{floor} bytes a max_batch_fields={self.max_batch_fields} request "
                    f"may legitimately carry, so this server would answer 413 to bodies "
                    f"every other limit it publishes accepts. Unset it to derive the cap, "
                    f"or lower {_ENV_PREFIX}MAX_BATCH_FIELDS to match."
                )

    @property
    def body_byte_cap(self) -> int:
        """
        The byte cap actually enforced: the operator's override, else the derived floor.

        One name for the effective number, so the middleware cannot be handed the raw
        `max_body_bytes` and silently enforce None.
        """
        if self.max_body_bytes is not None:
            return self.max_body_bytes
        return worst_case_body_bytes(self.max_batch_fields)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MatchServiceLimits:
        """
        Read the limits from the environment, falling back to the documented defaults.

        An unparseable value is an ERROR, not a fallback. `NEXUS_API_DEADLINE_SECONDS=25s`
        silently reverting to 25.0 happens to be harmless; `=0.25` typed as `.25s` and
        silently reverting to 25.0 is a hundredfold difference in what the operator
        believes they configured. Same standard as `_load_matching_config`: a tuned number
        that is quietly discarded is worse than a loud failure at startup.
        """
        env = os.environ if environ is None else environ

        def _read(suffix: str, cast: Callable[[str], float | int], fallback: float | int):
            raw = env.get(f"{_ENV_PREFIX}{suffix}")
            if raw is None or not raw.strip():
                return fallback
            try:
                return cast(raw.strip())
            except ValueError as exc:
                raise ValueError(
                    f"{_ENV_PREFIX}{suffix}={raw!r} is not a valid "
                    f"{cast.__name__}. Unset it to use the default {fallback!r}."
                ) from exc

        # Absent means DERIVE, which is not the same as any number an operator could type
        # -- so this one cannot go through `_read`'s fallback, which would have to pick a
        # literal and make the environment the source of truth for the cap.
        raw_body_bytes = env.get(f"{_ENV_PREFIX}MAX_BODY_BYTES")
        configured_body_bytes = (
            None
            if raw_body_bytes is None or not raw_body_bytes.strip()
            else int(_read("MAX_BODY_BYTES", int, 0))
        )

        return cls(
            deadline_seconds=float(_read("DEADLINE_SECONDS", float, cls.deadline_seconds)),
            max_workers=int(_read("MAX_WORKERS", int, cls.max_workers)),
            max_queued=int(_read("MAX_QUEUED", int, cls.max_queued)),
            max_fields=int(_read("MAX_FIELDS", int, cls.max_fields)),
            max_batch_fields=int(_read("MAX_BATCH_FIELDS", int, cls.max_batch_fields)),
            max_body_bytes=configured_body_bytes,
        )


# =============================================================================
# ADMISSION
# =============================================================================


class BoundedWorkPool:
    """
    A thread pool that refuses work instead of queueing it without limit.

    `capacity` counts everything ADMITTED and not yet finished -- running plus waiting.
    Counting only the waiting half would let `max_workers` extra items in on top of the
    queue bound, which is a bound that quietly means something other than what it says.
    """

    def __init__(self, max_workers: int, max_queued: int, name: str = "nexus-match") -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)
        self._capacity = max_workers + max_queued
        # A threading.Lock, not an asyncio primitive: `_release` runs on the WORKER
        # thread, from the future's done callback, and asyncio primitives are not
        # thread-safe. Getting this wrong would corrupt the counter under exactly the
        # concurrency the counter exists to survive.
        self._lock = threading.Lock()
        self._in_flight = 0
        self._closed = False

    @property
    def capacity(self) -> int:
        """The maximum number of admitted-and-unfinished units of work."""
        return self._capacity

    @property
    def in_flight(self) -> int:
        """How many are admitted and unfinished right now. Reported in the 503 body."""
        with self._lock:
            return self._in_flight

    def submit(self, work: Callable[[], T]) -> Future[T] | None:
        """
        Admit and schedule `work`, or return None when the pool is full or shut down.

        None rather than an exception because "full" is not an error here -- it is the
        designed response to load, and the caller turns it into a 503. The permit is taken
        BEFORE submitting and released from the future's done callback, so it covers the
        whole lifetime of the work including the part after a caller has stopped waiting.
        """
        with self._lock:
            if self._closed or self._in_flight >= self._capacity:
                return None
            self._in_flight += 1

        try:
            future = self._executor.submit(work)
        except RuntimeError:
            # The executor was shut down between the check above and here -- a real race
            # during application shutdown. Give the permit back and shed; a leaked permit
            # would permanently shrink capacity for the life of the process.
            self._release()
            return None

        future.add_done_callback(self._on_done)
        return future

    def _on_done(self, _future: Future[object]) -> None:
        self._release()

    def _release(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def shutdown(self) -> None:
        """
        Stop admitting, and let work already running finish.

        `cancel_futures=True` drops items that have not started, which is right: nobody is
        waiting for them any more. `wait=False` because shutdown runs from the lifespan
        handler on the event loop, and blocking there on a CPU-bound match would stall the
        shutdown of everything else in the process.
        """
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


# =============================================================================
# THE DEADLINE
# =============================================================================


async def run_bounded(pool: BoundedWorkPool, work: Callable[[], T], deadline_seconds: float) -> T:
    """
    Run `work` off the event loop, under admission control and a hard deadline.

    Raises OverloadedError (503) when the pool refuses, DeadlineExceededError (504) when
    the deadline fires, and whatever `work` raised otherwise.

    `asyncio.wait_for` cancels the wrapper on timeout. Cancelling a wrapped
    `concurrent.futures.Future` tries to cancel the underlying future: if the work has not
    started it really is cancelled and the permit comes back at once, and if it is already
    running the thread continues and the permit comes back when it finishes. Both are
    correct; neither leaks. The completed result is dropped -- `_copy_future_state` returns
    early on a cancelled destination -- so there is no stray "exception was never
    retrieved" from a match that finished just after its caller gave up.
    """
    future = pool.submit(work)
    if future is None:
        raise OverloadedError(
            message=(
                f"The matching service is at capacity ({pool.capacity} requests in flight "
                f"or queued) and shed this request rather than queueing it without limit. "
                f"Retry after a short backoff, or use your fallback path."
            ),
            details={"capacity": pool.capacity, "in_flight": pool.in_flight},
        )

    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=deadline_seconds)
    except asyncio.TimeoutError as exc:
        raise DeadlineExceededError(
            message=(
                f"Matching did not finish within this server's deadline of "
                f"{deadline_seconds}s, so the request was ended rather than left hanging. "
                f"Send fewer fields per request, raise "
                f"{_ENV_PREFIX}DEADLINE_SECONDS, or use your fallback path."
            ),
            details={"deadline_seconds": deadline_seconds},
        ) from exc
