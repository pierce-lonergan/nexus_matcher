"""
nexus_matcher.presentation.api.introspect | Layer: PRESENTATION
GET /api/v1/status and POST /api/v1/diag/retrieval -- what is loaded, and why it missed.

## Relationships
# DEPENDS_ON → presentation/api/matching :: the field translation and the response class
# DEPENDS_ON → presentation/api/errors :: MatcherUnavailableError, drift()
# DEPENDS_ON → presentation/api/limits :: the deadline and the bounded pool for the diagnostic
# DEPENDS_ON → presentation/api/lookup :: the one bound on a dictionary id
# USED_BY    → presentation/api/app :: mounted by create_app, which also records provenance

## Health answers a different question, and answering it is not enough

`/health/live`, `/health/ready` and `/health/startup` answer "is this process alive, and
has it finished starting" -- Kubernetes questions, correctly shaped for Kubernetes. An
operator about to start a bulk run over hundreds of subjects is asking something else:
**is retrieval currently degraded?** A process can be perfectly live, perfectly ready, and
answering every request out of an encoder that is not the one anybody intended.

That is not hypothetical. The adopting pipeline has already lost an entire bulk to a silent
encoder fallback -- matching ran, answered 200 on every field, and produced quietly worse
results for six hours before anyone noticed. Nothing in a liveness probe can catch that,
because nothing was down.

So `GET /api/v1/status` reports the state a bulk run depends on, and `degraded` is the one
field it exists for. The rest is the context that makes `degraded` actionable rather than
alarming.

### What `degraded` means, exactly

`degraded` is `warnings != []`, and a warning is emitted only for a condition under which a
bulk run would produce results the operator did not intend:

  NO_DICTIONARY      nothing is loaded; every match answers 503
  EMPTY_DICTIONARY   a dictionary loaded and carries no entries; every field matches nothing
  FALLBACK_ENCODER   retrieval is running on an encoder the selection ladder fell through to

One boolean and one list rather than a boolean per condition, so a caller has a single
thing to branch on and an operator has the sentence explaining it.

What is deliberately NOT a warning is the shipped threshold configuration. The structural
floor of a rank-1 confidence (0.63 with the shipped weights) sits ABOVE
`review_threshold = 0.50`, so a rank-1 candidate cannot fall below review on score alone --
a real and important property, and the direct descendant of DX-001, where a
`get_low_confidence_fields()` default of 0.6 sat below that floor and returned an empty list
on every call. But it is true of every default install, and a status surface that reports
`degraded` on a stock deployment teaches operators to ignore the field. It is published as
arithmetic instead: `thresholds.minimumAchievableConfidence` and the derived
`thresholds.reviewThresholdBelowFloor`, which is exactly `review < floor` and nothing more.

### This response is byte-stable

Nothing here is read from a clock or from live load at request time -- `indexedAt` is
stamped once at startup, and the pool's `capacity` is a constant while `in_flight` is
deliberately absent. Two GETs against one process therefore produce identical bytes, which
is what lets an operator diff two hosts and see only the difference that matters.

## The retrieval diagnostic

`POST /api/v1/diag/retrieval` answers "why did this field not match?" -- named in NM-V2-01
§AR-8 as the single highest-value diagnostic, and it is the one thing here that costs real
CPU, so it is the only route in this module that goes through the bounded work pool and the
server-side deadline.

It reports what the query text BECAME, what each retrieval channel returned with its RAW
scores, and -- when the caller names the entry they expected -- where that entry ranked in
each channel, or that it is not in the dictionary at all. Those last two are different
diagnoses and the answer says which: "retrieved at rank 34" is a scoring problem,
"not in the dictionary" is a glossary problem, and confusing them wastes a day.

### What it does NOT reproduce, said plainly

This route runs the RETRIEVAL half of `NexusMatcher._match_field` by calling the same
private methods in the same order. It stops before the five-signal scoring pass and before
the decision layer, so **a candidate's position here is not its final rank**: lexical, edit
distance, type compatibility and domain signals still reorder the fused list, and a
reranker replaces it outright. When one is wired, `rerankerWired` is true and the fused list
shown is the INPUT to reranking, not the order matching used.

That coupling is the heaviest in this package -- eight private names on the application
layer -- and it is held honest by arithmetic rather than by hope: `test_retrieval_diagnostic`
asserts that EVERY candidate a real `POST /api/v1/match` returns for a field appears in the
fused list this route reports for the same field. Matching can only score what retrieval
returned, so that invariant holds for any scoring change and fails the moment this route
stops driving the same pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field

from nexus_matcher.presentation.api.errors import MatcherUnavailableError, drift
from nexus_matcher.presentation.api.limits import run_bounded
from nexus_matcher.presentation.api.lookup import MAX_DICTIONARY_ID_CHARS
from nexus_matcher.presentation.api.matching import (
    _MATCHER_CONFIG_ATTR,
    DeterministicJSONResponse,
    _to_schema_field,
)
from nexus_matcher.presentation.api.schemas import ErrorResponse, FieldSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from nexus_matcher.domain.models.entities import SchemaField
    from nexus_matcher.presentation.api.limits import BoundedWorkPool, MatchServiceLimits
    from nexus_matcher.presentation.api.matching import MatcherHandle

# =============================================================================
# CONSTANTS
# =============================================================================

# Private application-layer names this module reads. Named constants rather than inline
# strings, for the reason `matching.py` gives for its three: the coupling has to be
# greppable from one place on the day another lane makes any of it public. A public
# retrieval-trace API on `NexusMatcher` would remove every one of them.
_EMBEDDING_PROVIDER_ATTR = "_embedding_provider"
_VECTOR_STORE_ATTR = "_vector_store"
_SPARSE_RETRIEVER_ATTR = "_sparse_retriever"
_RERANKER_ATTR = "_reranker"
_ALIAS_OWNER_ATTR = "_alias_owner"
_BUILD_QUERY_ATTR = "_build_query_text"
_FUSE_ATTR = "_fuse_results"
_DICTIONARY_ENTRIES_ATTR = "_dictionary_entries"

# Decimals kept for every emitted number, matching `matching._PRECISION` and the CLI's JSON
# writer. Raw retrieval scores are the thing this route exists to show, so they are rounded
# for byte-stability and not for presentation.
_PRECISION = 6

# Module path -> the rung of `default_embedding_provider`'s selection ladder it implements.
#
# Compared against `type(provider).__module__` rather than against a class, so this file
# does not import three adapter modules (and, through one of them, torch) merely to describe
# what is already loaded. A provider from anywhere else is `custom` -- somebody wired it
# deliberately, which is not a fallback and must not be reported as one.
#
# `test_introspection` asserts each module named here actually resolves, because a rename
# would otherwise turn every tier into `custom` silently, which is precisely the direction
# this table must not fail in.
_ENCODER_TIER_BY_MODULE: tuple[tuple[str, str], ...] = (
    ("nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx", "bundled"),
    (
        "nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers",
        "transformer",
    ),
    ("nexus_matcher.infrastructure.adapters.embedding_providers.static_embedding", "static"),
)

# The rungs BELOW the ladder's first choice. Reaching one of these while the first choice is
# unavailable is the silent fallback this whole surface exists to make loud.
_LADDER_FALLBACK_TIERS = frozenset({"transformer", "static"})

# Warning codes. Stable, machine-readable, and deliberately generic: they describe this
# library's own states and borrow no vocabulary from any deployment.
_NO_DICTIONARY = "NO_DICTIONARY"
_EMPTY_DICTIONARY = "EMPTY_DICTIONARY"
_FALLBACK_ENCODER = "FALLBACK_ENCODER"


# =============================================================================
# PROVENANCE -- recorded by app.py at startup, read here
# =============================================================================


@dataclass(frozen=True)
class DictionaryProvenance:
    """
    Where the loaded dictionary came from and when this process indexed it.

    Held here rather than on `MatcherHandle` because the handle lives in a module this lane
    does not own, and because provenance is a fact about the SERVER's startup rather than
    about the matcher object: an embedder who passes `matcher=` to `create_app` hands over
    an already-indexed matcher, and this server genuinely does not know where it came from
    or when it was built.

    Both members are therefore honestly nullable, and they are null together: `source` is
    the value of `NEXUS_API_DICTIONARY` this server loaded, and `indexed_at` is the instant
    it finished doing so. A caller-injected matcher reports null for both, which reads as
    "this server did not load it" rather than as a missing field.
    """

    source: str | None = None
    indexed_at: datetime | None = None


class ProvenanceRecorder:
    """
    A one-slot mutable holder, written once during startup and read on every status request.

    Mutable because the routes are registered at import time while the dictionary is loaded
    during the lifespan -- the same ordering problem `MatcherHandle` exists for, and solved
    the same way rather than by making the status route conditional on startup order.
    """

    def __init__(self) -> None:
        self._provenance = DictionaryProvenance()

    def record(self, *, source: str | None, indexed_at: datetime | None) -> None:
        """Record what this server loaded. Called once, from `app._bring_up_matcher`."""
        self._provenance = DictionaryProvenance(source=source, indexed_at=indexed_at)

    def stamp(self, source: str) -> None:
        """Record a dictionary this server has just finished indexing, stamped now."""
        self.record(source=source, indexed_at=datetime.now(timezone.utc))

    @property
    def current(self) -> DictionaryProvenance:
        """What is on record right now."""
        return self._provenance


# =============================================================================
# THE WIRE CONTRACT -- status
# =============================================================================


class StatusWarningView(BaseModel):
    """One reason not to start a bulk run yet."""

    code: str = Field(
        description=(
            "Stable machine-readable code: `NO_DICTIONARY`, `EMPTY_DICTIONARY` or "
            "`FALLBACK_ENCODER`. Branch on this, never on `message`."
        )
    )
    message: str = Field(
        description="What is wrong and what to change. Human-readable; not part of the contract."
    )


class DictionaryStatusView(BaseModel):
    """What dictionary this server is answering out of."""

    entryCount: int | None = Field(
        description="Indexed entries, or null when no dictionary is loaded."
    )
    source: str | None = Field(
        description=(
            "The `NEXUS_API_DICTIONARY` value this server loaded. Null when the matcher was "
            "supplied already-indexed by an embedder, in which case this server did not "
            "load it and cannot name a source."
        )
    )
    indexedAt: str | None = Field(
        description=(
            "UTC ISO-8601 instant this server finished indexing, or null when it did not do "
            "the indexing. This is when the INDEX was built, not when the file was written."
        )
    )


class EncoderStatusView(BaseModel):
    """
    Which encoder retrieval is actually running on, and whether that is the intended one.

    `fallbackInForce` is the field this whole surface exists for. It is true when the active
    provider sits on a rung BELOW the selection ladder's first choice AND the first choice is
    unavailable in this install -- that is, when the selection fell through rather than when
    somebody chose. A deliberately wired provider (`prefer="transformer"`, or a provider
    passed to `NexusMatcher` directly) reports false, because reporting a chosen encoder as a
    fallback would train an operator to ignore the one field that must never be ignored.
    """

    provider: str = Field(description="The provider class actually in use.")
    modelName: str | None = Field(description="The model it reports, or null if it reports none.")
    dimension: int | None = Field(description="Embedding dimension, or null if not reported.")
    tier: str = Field(
        description=(
            "Rung of the selection ladder: `bundled`, `transformer`, `static`, or `custom` "
            "for a provider from outside this library."
        )
    )
    bundledEncoderAvailable: bool = Field(
        description=(
            "Whether the bundled int8 ONNX encoder -- the ladder's first choice -- could be "
            "used in this install: weights present in the wheel AND `onnxruntime` and "
            "`tokenizers` importable."
        )
    )
    fallbackInForce: bool = Field(
        description=(
            "True when encoder selection FELL THROUGH to a lower rung because the first "
            "choice was unavailable here. This is the silent degradation that has cost an "
            "entire bulk run before; treat it as a reason to stop."
        )
    )


class ThresholdsView(BaseModel):
    """
    The numbers in force on the live matcher, and the floor that bounds them.

    Read off the running matcher's own config rather than off the shipped defaults, so a
    tuned deployment reports ITS numbers -- the same posture, and the same reason, as
    `matching._scoring_weights`.
    """

    # Every one is nullable, and null means ONE thing: this matcher does not expose that
    # setting, which is drift. It is never a defaulted number. `autoApprove: 0.0` on a
    # matcher whose config could not be read would tell an operator that everything
    # auto-approves, which is the most expensive wrong answer this block could give -- and
    # it would be indistinguishable from a deployment that had really configured 0.0.
    autoApprove: float | None
    review: float | None
    minConfidenceGap: float | None
    resultsPerField: int | None
    fusionAlpha: float | None
    minimumAchievableConfidence: float | None = Field(
        description=(
            "The lowest confidence a RANK-1 candidate can structurally carry: "
            "`semantic_weight * fusion_alpha` for the shipped wiring. Null when a reranker "
            "is wired, because a reranker replaces the fused score and the floor does not "
            "hold -- a bound that quietly does not hold is worse than no bound."
        )
    )
    reviewThresholdBelowFloor: bool | None = Field(
        description=(
            "Exactly `review < minimumAchievableConfidence`, and nothing more. When true, a "
            "rank-1 candidate cannot fall below the review threshold on score alone, so no "
            "field will ever be sent to review by score. Null when either side is null."
        )
    )


class ServiceLimitsView(BaseModel):
    """The caps a client has to chunk against, so an adapter can read them instead of guessing."""

    maxFields: int = Field(description="Field cap for POST /api/v1/match.")
    maxBatchFields: int = Field(
        description=(
            "Field cap for POST /api/v1/match/batch. Also the id cap for POST /api/v1/lookup."
        )
    )
    bodyByteCap: int = Field(description="Raw request body cap, enforced before parsing.")
    deadlineSeconds: float = Field(description="Server-side deadline before a 504.")
    capacity: int = Field(
        description=(
            "Requests that may be admitted-and-unfinished before the server sheds with 503. "
            "The live in-flight count is deliberately NOT reported: it would make two "
            "identical requests produce different bytes."
        )
    )


class StatusResponseView(BaseModel):
    """
    Everything an operator needs before starting a bulk run, in one byte-stable body.

    `ready` mirrors `/health/ready`'s verdict for the matcher; `degraded` is the question
    health probes cannot answer. A **not-ready** server still answers this route 200 --
    a diagnostic that fails when things are broken is a diagnostic nobody can use.
    """

    ready: bool = Field(description="Whether a dictionary is loaded and matching can be served.")
    degraded: bool = Field(
        description=(
            "`warnings != []`. True when a bulk run would produce results the operator did "
            "not intend. This is the field this surface exists for."
        )
    )
    warnings: list[StatusWarningView]
    dictionary: DictionaryStatusView
    encoder: EncoderStatusView | None = Field(
        description="Null when no matcher is loaded, because there is no encoder to describe."
    )
    thresholds: ThresholdsView | None = Field(
        description=(
            "Null when no matcher is loaded. Never the shipped defaults in that case: "
            "reporting thresholds that are not in force would be a wrong answer, not a "
            "missing one."
        )
    )
    limits: ServiceLimitsView


# =============================================================================
# THE WIRE CONTRACT -- the retrieval diagnostic
# =============================================================================


class RetrievalDiagnosticRequest(BaseModel):
    """
    One field to diagnose, and optionally the entry the caller expected it to find.

    A POST rather than the GET its v1 ancestor used: `doc` is a column comment up to 8 KiB
    and is real retrieval signal, so a diagnostic that could not carry it would diagnose a
    different query than the one that actually missed. A body is also what
    `BodySizeLimitMiddleware` protects; a query string is not.
    """

    model_config = ConfigDict(extra="forbid")

    # `field` shadows the `Field` imported into every module that touches this file, exactly
    # as `schemas.FeedbackRequest.field_path` does. Aliased for the same reason.
    field_spec: FieldSpec = Field(
        alias="field",
        description=(
            "The field to diagnose, in the same shape `/api/v1/match` takes -- so the query "
            "this route reports is the query that route would have built."
        ),
    )
    # Bounded by the same number the lookup plane bounds a dictionary id with, rather than
    # by a second literal. One service refusing an id here that it resolves two routes over
    # is a contradiction a caller cannot act on.
    expected_governance_id: str | None = Field(
        default=None,
        max_length=MAX_DICTIONARY_ID_CHARS,
        description=(
            "The dictionary id the caller believes this field should have matched. When "
            "given, the response says where that entry ranked in each channel, or that it "
            "is not in the dictionary at all -- two very different diagnoses."
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Candidates shown per channel. RANKS ARE COMPUTED OVER THE FULL CHANNEL RESULT, "
            "not over this truncation, so an expected entry at rank 34 is reported as 34 "
            "rather than as absent."
        ),
    )


class RetrievalCandidateView(BaseModel):
    """One candidate as a retrieval channel returned it, with the channel's own raw score."""

    rank: int = Field(description="1-based position within this channel, before any scoring.")
    governanceId: str
    businessName: str | None = Field(
        description="Null when the channel returned an id the dictionary does not carry."
    )
    score: float = Field(
        description=(
            "This channel's RAW score, rounded to six decimals: cosine similarity for "
            "`dense`, the retriever's own lexical score for `sparse`, and the min-max "
            "normalised weighted sum for `fused`. The three are NOT comparable with each "
            "other."
        )
    )


class RetrievalChannelView(BaseModel):
    """What one retrieval channel returned, and what it was asked for."""

    available: bool = Field(
        description="False when this channel is not wired, or could not run; see `detail`."
    )
    detail: str | None = Field(
        description="Why the channel is unavailable, or null when it ran normally."
    )
    requestedTopK: int | None = Field(description="The depth the channel was searched to.")
    returned: int | None = Field(
        description="How many candidates it returned, before the display truncation."
    )
    candidates: list[RetrievalCandidateView]


class ExpectedPlacementView(BaseModel):
    """Where the entry the caller named actually landed."""

    governanceId: str
    inDictionary: bool = Field(
        description=(
            "Whether this id exists in the loaded dictionary at all. FALSE IS THE ANSWER: a "
            "field cannot match an entry that was never indexed, and no amount of threshold "
            "tuning fixes it."
        )
    )
    rankByChannel: dict[str, int | None] = Field(
        description=(
            "1-based rank in each channel's FULL result, or null where that channel did not "
            "return it. Keys are the channel names."
        )
    )


class RetrievalDiagnosticView(BaseModel):
    """
    Why a field retrieved what it retrieved.

    Retrieval only. This is not the ranking `/api/v1/match` produces and must not be read as
    one -- see the module docstring for exactly which stages it stops before.
    """

    field: dict[str, str] = Field(
        description="The field as it was received, echoed so the artifact is self-describing."
    )
    queryText: str = Field(
        description=(
            "What the field BECAME before retrieval: parent-path context injected, "
            "abbreviations expanded if that is enabled. This string, not the field name, is "
            "what the encoder saw."
        )
    )
    encoderModel: str | None = Field(description="The model that encoded the query.")
    rerankerWired: bool = Field(
        description=(
            "When true, matching replaces the fused order with a reranker's, so the `fused` "
            "channel below is the INPUT to reranking rather than the order matching used."
        )
    )
    channels: dict[str, RetrievalChannelView] = Field(
        description="`dense`, `sparse` and `fused`, in that order."
    )
    expected: ExpectedPlacementView | None = Field(
        description="Null unless the request named an `expected_governance_id`."
    )


# =============================================================================
# ENCODER TIER
# =============================================================================


def encoder_tier(provider: object) -> str:
    """Which rung of the selection ladder this provider implements, or `custom`."""
    module = type(provider).__module__
    for name, tier in _ENCODER_TIER_BY_MODULE:
        if module == name:
            return tier
    return "custom"


def bundled_encoder_available() -> bool:
    """
    Whether the ladder's first choice could be used in this install.

    Weights present AND both runtime dependencies importable -- the same two conditions
    `default_embedding_provider`'s own `try_bundled` applies, which is what makes this a
    statement about the selection that happened rather than a separate guess at it.

    `find_spec` rather than `import`, so asking the question does not pay for onnxruntime's
    import in a process that is not using it. A broken or namespace-shadowed install raises
    from `find_spec`; that is answered False, because an install that cannot be inspected is
    not one the encoder can be counted on in.
    """
    from importlib.util import find_spec

    try:
        from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
            bundled_model_available,
        )

        if not bundled_model_available():
            return False
        return all(find_spec(module) is not None for module in ("onnxruntime", "tokenizers"))
    except (ImportError, ValueError, OSError):
        return False


def _encoder_payload(provider: object, *, tier: str, bundled_available: bool) -> dict[str, Any]:
    """
    The encoder block, given the two facts that decide `fallbackInForce`.

    Both facts are parameters rather than lookups so the truth table is testable without an
    install that can produce it: the interesting case -- a lower rung in force while the
    first choice is missing -- is exactly the one a healthy developer machine cannot create.
    """
    dimension = getattr(provider, "dimension", None)
    model_name = getattr(provider, "model_name", None)
    return {
        "provider": type(provider).__name__,
        "modelName": None if model_name is None else str(model_name),
        "dimension": int(dimension) if isinstance(dimension, int) else None,
        "tier": tier,
        "bundledEncoderAvailable": bundled_available,
        "fallbackInForce": tier in _LADDER_FALLBACK_TIERS and not bundled_available,
    }


# =============================================================================
# STATUS
# =============================================================================


def _config_of(matcher: object) -> object | None:
    """The live matcher's config, or None when it does not expose one."""
    return getattr(matcher, _MATCHER_CONFIG_ATTR, None)


def _number(source: object, attribute: str) -> float | None:
    """
    A numeric setting off the live config, or None when it is not there to read.

    NEVER a default. A missing `auto_approve_threshold` reported as `0.0` would tell an
    operator that everything on this server auto-approves, and would be indistinguishable
    from a deployment that had really configured 0.0 -- a wrong answer wearing the shape of
    a right one, in the surface consulted precisely to find out what is in force.

    `bool` is excluded because it is a subclass of `int` in Python, and a flag misread as a
    threshold of 1.0 is the same class of lie one type down.
    """
    value = getattr(source, attribute, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), _PRECISION)


def _thresholds_payload(matcher: object) -> dict[str, Any] | None:
    """
    The thresholds in force, or None when the matcher exposes no config at all.

    None rather than the shipped defaults, deliberately. A tuned deployment whose config
    could not be read would otherwise be told, in the surface it consults precisely to find
    out what is in force, the numbers that are not.
    """
    config = _config_of(matcher)
    if config is None:
        return None

    floor = _number(matcher, "minimum_achievable_confidence")
    review = _number(config, "review_threshold")
    results_per_field = _number(config, "results_per_field")
    return {
        "autoApprove": _number(config, "auto_approve_threshold"),
        "review": review,
        "minConfidenceGap": _number(config, "min_confidence_gap"),
        "resultsPerField": None if results_per_field is None else int(results_per_field),
        "fusionAlpha": _number(config, "fusion_alpha"),
        "minimumAchievableConfidence": floor,
        # Null when EITHER side is null: a comparison with a number nobody could read is not
        # a false answer, it is an unanswerable question, and `false` would read as "your
        # review threshold is safely above the floor".
        "reviewThresholdBelowFloor": None if floor is None or review is None else review < floor,
    }


def _entry_count(matcher: object) -> int | None:
    """
    How many entries are indexed.

    `dictionary_size` first because it is the public accessor; the entry map is the fallback
    so a matcher that has not grown the property still reports a number rather than a null
    an operator would read as "no dictionary".
    """
    size = getattr(matcher, "dictionary_size", None)
    if isinstance(size, int):
        return size
    entries = getattr(matcher, _DICTIONARY_ENTRIES_ATTR, None)
    return len(entries) if isinstance(entries, dict) else None


def _limits_payload(limits: MatchServiceLimits, pool: BoundedWorkPool) -> dict[str, Any]:
    """The caps, so a client can read them rather than hard-code them from documentation."""
    return {
        "maxFields": limits.max_fields,
        "maxBatchFields": limits.max_batch_fields,
        "bodyByteCap": limits.body_byte_cap,
        "deadlineSeconds": round(float(limits.deadline_seconds), _PRECISION),
        "capacity": pool.capacity,
    }


class IntrospectionService:
    """Assembles the status body, and runs the retrieval diagnostic under the deadline."""

    def __init__(
        self,
        handle: MatcherHandle,
        limits: MatchServiceLimits,
        pool: BoundedWorkPool,
        provenance: ProvenanceRecorder,
        *,
        bundled_probe: Callable[[], bool] = bundled_encoder_available,
    ) -> None:
        self._handle = handle
        self._limits = limits
        self._pool = pool
        self._provenance = provenance
        # Injectable so the fallback truth table can be driven end to end. The only honest
        # alternative is an install with the weights deleted, which is not a test.
        self._bundled_probe = bundled_probe

    def status(self) -> dict[str, Any]:
        """The whole status body, in fixed key order. Never raises for an unloaded matcher."""
        warnings: list[dict[str, str]] = []
        provenance = self._provenance.current

        try:
            matcher: object | None = self._handle.require()
        except MatcherUnavailableError as exc:
            matcher = None
            warnings.append(
                {
                    "code": _NO_DICTIONARY,
                    "message": (
                        f"No dictionary is loaded, so every match answers 503 and every "
                        f"lookup misses: {exc.details.get('reason', 'reason unavailable')}"
                    ),
                }
            )

        entry_count = None if matcher is None else _entry_count(matcher)
        if matcher is not None and entry_count == 0:
            warnings.append(
                {
                    "code": _EMPTY_DICTIONARY,
                    "message": (
                        "A dictionary is loaded and carries no entries, so every field will "
                        "match nothing and inherit nothing. Check the loader's row filter "
                        "and the column mapping before starting a bulk run."
                    ),
                }
            )

        encoder: dict[str, Any] | None = None
        if matcher is not None:
            provider = getattr(matcher, _EMBEDDING_PROVIDER_ATTR, None)
            if provider is not None:
                encoder = _encoder_payload(
                    provider,
                    tier=encoder_tier(provider),
                    bundled_available=self._bundled_probe(),
                )
                if encoder["fallbackInForce"]:
                    warnings.append(
                        {
                            "code": _FALLBACK_ENCODER,
                            "message": (
                                f"Retrieval is running on the {encoder['tier']} encoder "
                                f"({encoder['modelName']}) because the bundled encoder is "
                                f"not usable in this install. Accuracy is lower than every "
                                f"number this library publishes, and nothing else reports "
                                f"it. Do not start a bulk run until this is resolved."
                            ),
                        }
                    )

        return {
            "ready": self._handle.is_ready,
            "degraded": bool(warnings),
            "warnings": warnings,
            "dictionary": {
                "entryCount": entry_count,
                "source": provenance.source,
                "indexedAt": (
                    None if provenance.indexed_at is None else provenance.indexed_at.isoformat()
                ),
            },
            "encoder": encoder,
            "thresholds": None if matcher is None else _thresholds_payload(matcher),
            "limits": _limits_payload(self._limits, self._pool),
        }

    async def diagnose(self, request: RetrievalDiagnosticRequest) -> dict[str, Any]:
        """Run the retrieval half of matching for one field, off the event loop."""
        matcher = self._handle.require()
        field = _to_schema_field(request.field_spec)
        return await run_bounded(
            self._pool,
            lambda: _trace_retrieval(
                matcher,
                field,
                spec=request.field_spec,
                expected=request.expected_governance_id,
                top_k=request.top_k,
            ),
            self._limits.deadline_seconds,
        )


# =============================================================================
# THE RETRIEVAL TRACE
# =============================================================================


def _channel(
    *,
    available: bool,
    detail: str | None,
    requested_top_k: int | None,
    ranked: list[tuple[str, float]],
    entries: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """One channel's block, truncated for display but counted over the full result."""
    return {
        "available": available,
        "detail": detail,
        "requestedTopK": requested_top_k,
        "returned": None if not available else len(ranked),
        "candidates": [
            {
                "rank": position,
                "governanceId": doc_id,
                "businessName": (
                    getattr(entries.get(doc_id), "business_name", None)
                    if isinstance(entries, dict)
                    else None
                ),
                "score": round(float(score), _PRECISION),
            }
            for position, (doc_id, score) in enumerate(ranked[:top_k], 1)
        ],
    }


def _rank_of(expected: str, ranked: list[tuple[str, float]]) -> int | None:
    """1-based position of `expected` in a channel's FULL result, or None."""
    for position, (doc_id, _score) in enumerate(ranked, 1):
        if doc_id == expected:
            return position
    return None


def _require(matcher: object, attribute: str, consequence: str) -> Any:
    """Read a private application-layer name, or refuse the response that would lie."""
    value = getattr(matcher, attribute, None)
    if value is None:
        raise drift(type(matcher).__name__, attribute, consequence)
    return value


def _dense_retrieval(
    provider: object,
    store: object,
    query_text: str,
    *,
    top_k: int,
    alias_owner: dict[str, str],
) -> tuple[list[Any], str | None]:
    """
    The dense arm, run the way `_match_field` runs it: encode, search, collapse aliases.

    Returns the candidate list -- which is also FUSION'S INPUT, so it is the collapsed list
    and not the raw one whenever aliasing is on -- and the reason the channel is
    unavailable. Exactly one of the two is empty.

    The collapse is not cosmetic. With `dictionary_alias_count` above zero the store returns
    fabricated spellings under synthetic ids; matching max-pools them onto the owning entry
    before fusing, so a trace that fused the raw list would report a different fused order
    than the one matching used, and would show ids that are not in the caller's glossary.
    """
    embedded = provider.embed_single(query_text)
    if embedded.is_failure:
        return [], f"the query could not be encoded: {embedded.error}"

    found = store.search(embedded.unwrap(), top_k=top_k)
    if found.is_failure:
        return [], f"the vector store refused the search: {found.error}"

    results = list(found.unwrap())
    if not alias_owner:
        return results, None

    from nexus_matcher.domain.ports.vector_store import SearchResult

    best: dict[str, Any] = {}
    for result in results:
        owner = alias_owner.get(result.id, result.id)
        current = best.get(owner)
        if current is None or result.score > current.score:
            best[owner] = SearchResult(
                id=owner, score=result.score, payload=result.payload, embedding=result.embedding
            )
    return sorted(best.values(), key=lambda item: item.score, reverse=True), None


def _sparse_retrieval(
    retriever: object | None, query_text: str, *, top_k: int
) -> tuple[list[tuple[str, float]], str | None]:
    """
    The lexical arm, or the reason there is not one.

    "Unavailable, and here is why" is a different answer from "returned nothing", and an
    operator reading an empty lexical list would conclude their query shares no tokens with
    any entry rather than that no retriever was ever asked.
    """
    if retriever is None:
        return [], "no sparse retriever is wired, so retrieval is dense-only."

    searched = retriever.search(query_text, top_k=top_k)
    if searched.is_failure:
        return [], f"the sparse retriever refused the search: {searched.error}"
    return [(result.id, result.score) for result in searched.unwrap()], None


def _trace_retrieval(
    matcher: object,
    field: SchemaField,
    *,
    spec: FieldSpec,
    expected: str | None,
    top_k: int,
) -> dict[str, Any]:
    """
    Drive dense, sparse and fusion exactly as `_match_field` drives them, and report it.

    Runs on a worker thread: this encodes a query and scans the corpus, which is the same
    CPU work matching does and belongs under the same admission control.
    """
    build_query = _require(
        matcher,
        _BUILD_QUERY_ATTR,
        "the query text this field actually retrieved on cannot be reported, which is the "
        "first thing this diagnostic exists to say.",
    )
    provider = _require(
        matcher,
        _EMBEDDING_PROVIDER_ATTR,
        "the query cannot be encoded and the dense channel cannot be reported.",
    )
    store = _require(
        matcher,
        _VECTOR_STORE_ATTR,
        "the dense channel cannot be searched and this diagnostic would report a pipeline "
        "the matcher does not run.",
    )
    fuse = _require(
        matcher,
        _FUSE_ATTR,
        "the fused order cannot be reproduced, and a diagnostic that guessed at it would "
        "send an operator after the wrong stage.",
    )

    config = _config_of(matcher)
    entries = getattr(matcher, _DICTIONARY_ENTRIES_ATTR, {})
    alias_owner = getattr(matcher, _ALIAS_OWNER_ATTR, {}) or {}
    dense_top_k = int(getattr(config, "dense_top_k", 100))
    sparse_top_k = int(getattr(config, "sparse_top_k", 100))

    query_text = build_query(field)

    dense_results, dense_detail = _dense_retrieval(
        provider, store, query_text, top_k=dense_top_k, alias_owner=alias_owner
    )
    dense_ranked = [(result.id, result.score) for result in dense_results]

    retriever = getattr(matcher, _SPARSE_RETRIEVER_ATTR, None)
    sparse_ranked, sparse_detail = _sparse_retrieval(retriever, query_text, top_k=sparse_top_k)
    sparse_scores = dict(sparse_ranked)

    # -- fused --------------------------------------------------------------
    #
    # `_fuse_results` takes the dense SearchResults and the sparse score map, so the fused
    # order here is produced by the same call matching makes rather than by arithmetic this
    # file restates. When the dense arm failed there is nothing to fuse and saying so is the
    # honest answer.
    fused_ranked: list[tuple[str, float]] = []
    fused_detail: str | None = None
    if dense_detail is not None:
        fused_detail = "fusion was not run, because the dense channel did not return."
    else:
        fused_ranked = [(doc_id, score) for doc_id, score in fuse(dense_results, sparse_scores)]

    channels = {
        "dense": _channel(
            available=dense_detail is None,
            detail=dense_detail,
            requested_top_k=dense_top_k,
            ranked=dense_ranked,
            entries=entries,
            top_k=top_k,
        ),
        "sparse": _channel(
            available=sparse_detail is None,
            detail=sparse_detail,
            requested_top_k=None if retriever is None else sparse_top_k,
            ranked=sparse_ranked,
            entries=entries,
            top_k=top_k,
        ),
        "fused": _channel(
            available=fused_detail is None,
            detail=fused_detail,
            requested_top_k=None,
            ranked=fused_ranked,
            entries=entries,
            top_k=top_k,
        ),
    }

    placement: dict[str, Any] | None = None
    if expected is not None:
        placement = {
            "governanceId": expected,
            "inDictionary": expected in entries if isinstance(entries, dict) else False,
            "rankByChannel": {
                "dense": _rank_of(expected, dense_ranked),
                "sparse": _rank_of(expected, sparse_ranked),
                "fused": _rank_of(expected, fused_ranked),
            },
        }

    model_name = getattr(provider, "model_name", None)
    return {
        "field": {
            "name": spec.name,
            "path": spec.path,
            "doc": spec.doc,
            "type": spec.type,
        },
        "queryText": query_text,
        "encoderModel": None if model_name is None else str(model_name),
        "rerankerWired": getattr(matcher, _RERANKER_ATTR, None) is not None,
        "channels": channels,
        "expected": placement,
    }


# =============================================================================
# ROUTER
# =============================================================================

_DIAGNOSTIC_ERRORS: dict[int | str, dict[str, Any]] = {
    413: {
        "model": ErrorResponse,
        "description": "A body over this server's byte cap.",
    },
    422: {
        "model": ErrorResponse,
        "description": "Malformed request. `details.violations` names the offending field.",
    },
    500: {
        "model": ErrorResponse,
        "description": (
            "This layer refused a trace it could not trust -- the application layer no "
            "longer exposes a stage this diagnostic reads."
        ),
    },
    503: {
        "model": ErrorResponse,
        "description": "No dictionary is loaded, or the server shed this request.",
    },
    504: {
        "model": ErrorResponse,
        "description": "The server-side deadline fired before retrieval finished.",
    },
}


def create_introspection_router(service: IntrospectionService) -> APIRouter:
    """Mount the status surface and the retrieval diagnostic."""
    router = APIRouter(prefix="/api/v1", tags=["Introspection"])

    @router.get(
        "/status",
        status_code=status.HTTP_200_OK,
        response_class=DeterministicJSONResponse,
        response_model=None,
        # No failure responses, and that is the contract: this route answers 200 whether or
        # not a dictionary is loaded. A pre-run degradation check that 503s when the thing
        # it checks for is true would be unusable at exactly the moment it is needed.
        responses={200: {"model": StatusResponseView}},
        summary="What is loaded, and whether retrieval is degraded",
        description=(
            "Answers the question health probes cannot: is retrieval degraded right now. "
            "`degraded` and `warnings` are the answer; the dictionary, encoder, threshold "
            "and limit blocks are the context. Always 200, including when nothing is "
            "loaded. Byte-stable: nothing here is read from a clock at request time."
        ),
    )
    async def service_status() -> DeterministicJSONResponse:
        return DeterministicJSONResponse(service.status())

    @router.post(
        "/diag/retrieval",
        status_code=status.HTTP_200_OK,
        response_class=DeterministicJSONResponse,
        response_model=None,
        responses={200: {"model": RetrievalDiagnosticView}, **_DIAGNOSTIC_ERRORS},
        summary="Why did this field not match?",
        description=(
            "The retrieval half of matching for one field: what the query text became, what "
            "each channel returned with its raw scores, and -- when `expected_governance_id` "
            "is given -- where that entry ranked, or that it is not in the dictionary at "
            "all. RETRIEVAL ONLY: the five-signal scoring pass and the decision layer are "
            "not run, so a position here is not a final rank."
        ),
    )
    async def diagnose_retrieval(
        request: RetrievalDiagnosticRequest,
    ) -> DeterministicJSONResponse:
        return DeterministicJSONResponse(await service.diagnose(request))

    return router
