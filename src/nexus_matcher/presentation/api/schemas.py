"""
nexus_matcher.presentation.api.schemas | Layer: PRESENTATION
The wire contract: what a client may send, and what it is promised back.

## Relationships
# USED_BY → presentation/api/matching :: request parsing and the documented response shape
# USED_BY → presentation/api/feedback :: the feedback record a reviewer submits

## Two halves that are deliberately not symmetric

The REQUEST models are enforced. FastAPI parses the body through them, so a bad request
becomes a 422 naming the offending field before a single line of matching code runs.

The RESPONSE models are DOCUMENTATION. The handlers build plain dicts and render them
themselves, because "two identical requests produce byte-identical bodies" is a promise
about key ORDER and key PRESENCE, and serialising through a response model surrenders
both: `explain` has to be absent when it was not asked for (not present-and-null), while
`governance` has to be present-and-null when an entry carries no code, and no single
pydantic dump flag produces that pair. So the models are attached to the routes via
`responses=` for the OpenAPI schema a Java client generates from, and
`tests/unit/presentation/api/test_match_endpoint.py` validates real response bodies
against them -- otherwise the published schema would be free to drift away from what the
service actually sends, which is a worse lie than having no schema at all.

## Naming

Request fields are snake_case (`top_k`), response fields are camelCase (`governanceId`).
That is not an oversight, it is the agreed contract with the calling pipeline, and
"consistency" is not worth a silently renamed key on a surface another team has already
built against.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_matcher.shared.types.base import MatchDecision

# =============================================================================
# REQUEST
# =============================================================================

# Bounds on every string a client can send. These are not guesses about reasonable data;
# they are the difference between a 422 and an out-of-memory. A field `doc` is a column
# comment, so 8 KiB is already generous, and a caller who genuinely needs more is sending
# something this endpoint cannot use for retrieval anyway.
_MAX_NAME = 512
_MAX_PATH = 1024
_MAX_DOC = 8192
_MAX_TYPE = 128

# The largest one `FieldSpec` may be, in CHARACTERS, with every string at its bound.
#
# Exported because `limits.py` derives the raw-body byte cap from it rather than typing a
# number. The two have to move together: a byte cap BELOW what these bounds admit refuses
# a body `FieldSpec` itself accepts, so the caller reads two documents from this service
# and they contradict each other -- and a byte cap that stayed put while `_MAX_DOC` grew
# would do exactly that, silently, on the next edit to this block.
MAX_FIELD_SPEC_CHARS = _MAX_NAME + _MAX_PATH + _MAX_DOC + _MAX_TYPE


class FieldSpec(BaseModel):
    """
    One schema field a caller wants governance for.

    `extra="forbid"` on purpose. A misspelled `documentation` silently ignored would drop
    the column comment, and the column comment is real retrieval signal -- the caller
    would get measurably worse matches and no indication why. This is the same standard
    `_load_matching_config` applies to a mistyped `auto_approve_treshold`: a quietly
    discarded input is worse than a loud failure.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=_MAX_NAME, description="The column's own name.")
    path: str = Field(
        default="",
        max_length=_MAX_PATH,
        description=(
            "The caller's identifier for this field, and the key it comes back under. "
            "Dotted paths are strongly preferred: the segment before the last dot becomes "
            "the retrieval query's parent context, which is the single largest accuracy "
            "factor measured on this task. Defaults to `name`."
        ),
    )
    doc: str = Field(
        default="", max_length=_MAX_DOC, description="Column comment or description, if any."
    )
    type: str = Field(
        default="",
        max_length=_MAX_TYPE,
        description="Source type name, normalised server-side. Unknown types are accepted.",
    )

    @model_validator(mode="after")
    def _default_path_to_name(self) -> FieldSpec:
        """
        An omitted `path` falls back to `name`, so a caller with flat columns need not
        invent one. Done here rather than in the handler so the value the response is
        keyed by is fixed at parse time and there is exactly one place it comes from.
        """
        if not self.path:
            self.path = self.name
        return self


class MatchRequest(BaseModel):
    """
    A batch of fields to match. Identical for `/match` and `/match/batch`; only the
    server's field-count cap differs.
    """

    model_config = ConfigDict(extra="forbid")

    # The attribute is `field_specs` and the wire name is `fields`. `BaseModel.fields` is
    # a deprecated pydantic-v1 property, so a model attribute of that name is a trap
    # waiting for whoever next touches this file; the alias keeps the wire contract exact
    # without inheriting that.
    field_specs: list[FieldSpec] = Field(
        alias="fields",
        min_length=1,
        description="The fields to match. At least one -- see the note on empty requests.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "Candidates per field. Must not exceed the server's configured "
            "results_per_field; a larger value is refused with a 422 naming the cap, "
            "rather than silently returning fewer candidates than asked for."
        ),
    )
    explain: bool = Field(
        default=False,
        description=(
            "Include the score components and the weights that produced each confidence, "
            "so the number can be recomputed from the response itself."
        ),
    )


class FeedbackRequest(BaseModel):
    """
    A reviewer's verdict on one match.

    RECORDED ONLY. It is not read back into ranking, and this is a measured decision
    rather than a missing feature -- see `presentation/api/feedback.py`.
    """

    model_config = ConfigDict(extra="forbid")

    # `field` shadows nothing on BaseModel, but it does shadow the `Field` imported into
    # every module that touches this file. Aliased for the same reason as `fields` above.
    field_path: str = Field(alias="field", min_length=1, max_length=_MAX_PATH)
    doc: str = Field(default="", max_length=_MAX_DOC)
    chosenGovernanceId: str = Field(min_length=1, max_length=_MAX_NAME)
    suggestedGovernanceId: str | None = Field(default=None, max_length=_MAX_NAME)
    wasCorrect: bool
    reviewer: str = Field(min_length=1, max_length=_MAX_NAME)
    ts: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "The client's timestamp for this review. Stored verbatim and NOT trusted for "
            "ordering -- the server stamps its own `receivedAt` alongside it."
        ),
    )


# =============================================================================
# RESPONSE -- published schema, not the serialiser
# =============================================================================


class GovernanceView(BaseModel):
    """
    The protection class the matched entry carries, passed through from the caller's own
    controlled vocabulary.

    Null in exactly two cases, and this docstring is the only place a generated client can
    learn the second one:

      1. the matched entry carries no protection code at all -- the open tier;
      2. the candidate is `rank` 1 AND its `decision` is `REJECT`, meaning no entry in the
         glossary describes this field, so there is nothing for it to inherit.

    A REJECTED candidate below rank 1 DOES carry its class. Nothing inherits from a
    runner-up, and the class is what lets a reviewer see that rank 1 is a direct
    identifier and rank 2 is not.
    """

    code: str
    name: str
    classification: str
    personalInformation: bool
    directIdentifier: bool


class ExplainView(BaseModel):
    """
    Everything needed to recompute a candidate's confidence from the response alone.

    `scores` and `weights` are OPEN maps rather than five named fields, and that is the
    decision this model exists to record: `matching._verify_reproducible` is written to
    survive "a sixth weighted signal this file knows nothing about", so naming today's
    five components here would publish a schema that the sixth makes false. The two maps
    carry the same keys, and `sum(scores[k] * weights[k])`, clamped to [0, 1], is checked
    against the emitted `confidence` before the response is sent -- so a client can redo
    the arithmetic and a drifted server refuses rather than explains itself wrongly.
    """

    scores: dict[str, float] = Field(
        description=(
            "One entry per weighted signal, rounded to six decimals. "
            "`fusedRetrieval` is min-max normalised WITHIN this field's shortlist, so 0.9 "
            "means 'ranked first here', not '90% similar'."
        )
    )
    weights: dict[str, float] = Field(
        description=(
            "The weights of the LIVE matcher that produced these confidences, not the "
            "shipped defaults, so a tuned deployment gets numbers that reproduce ITS "
            "confidences."
        )
    )
    # `float | None`, not `float`: the endpoint emits null when the dense arm did not
    # return this candidate at all, and a narrower published type would be a schema the
    # service contradicts on its own fixture.
    absoluteCosine: float | None = Field(
        description=(
            "Dense cosine similarity -- the only number here comparable ACROSS fields. "
            "Null when the dense retriever did not return this candidate."
        )
    )


class MatchCandidateView(BaseModel):
    """One candidate for one field. `explain` is absent unless the request asked for it."""

    rank: int
    governanceId: str
    businessName: str
    definition: str
    domain: str
    governance: GovernanceView | None
    confidence: float
    # Typed as the library's own `MatchDecision` rather than a hand-written Literal, so
    # the published enum and the value `matching._candidate_payload` emits are the same
    # object by construction. A second copy of the value list is exactly the drift this
    # repository maintains a `drift()` helper for. Described because "REJECT" appeared
    # NOWHERE in the published schema, so a generated client could not learn that the
    # value interacts with `governance` at all.
    decision: MatchDecision = Field(
        description=(
            "AUTO_APPROVE, REVIEW or REJECT. Per CANDIDATE, not per field: every rank is "
            "compared against the server's review threshold, so runner-ups are routinely "
            "REJECT on a field whose top match is fine. Only a REJECT at rank 1 means "
            "'no entry describes this field', and only that combination nulls "
            "`governance`."
        )
    )
    explain: ExplainView | None = None


class MatchResponseView(BaseModel):
    """
    The whole response: one list per input field, keyed by the caller's own `path`.

    Every input path appears exactly once, in the order it was sent, whether or not
    anything matched it -- a field with no candidates gets an empty list, never a missing
    key. That is the conservation law this endpoint is built around (NM-0005).
    """

    results: dict[str, list[MatchCandidateView]]


class FeedbackResponseView(BaseModel):
    """The stored record, echoed back so the reviewer can see exactly what was written."""

    recorded: bool
    record: dict[str, Any]


# =============================================================================
# ERRORS -- the one envelope, published
# =============================================================================


class ErrorDetail(BaseModel):
    """
    One failure. The `error` member of every non-2xx body this service sends.

    Typed, and that is the whole point of it existing: the previous `error: dict[str,
    Any]` rendered as `{"type": "object", "additionalProperties": true}`, so a Java client
    generated from `/openapi.json` got a `Map` and had to learn the three keys from a
    human. Naming them is what turns the envelope into something a build step can use.
    """

    code: str = Field(
        description=(
            "Stable machine-readable code, e.g. `NEXUS-8004` for a request this server "
            "will not answer. Branch on this and the status code -- never on `message`."
        )
    )
    message: str = Field(
        description=(
            "What went wrong and what to do about it, addressed to whoever reads the log. "
            "Human-readable: the wording is not part of the contract."
        )
    )
    # Deliberately free-form. Its contents legitimately vary by failure -- `limit` and
    # `fields` on a 413, `violations` on a 422, `deadline_seconds` on a 504,
    # `duplicate_paths`, `results_per_field`, `capacity`, `reason`, `cause` -- so pinning
    # the keys would either be a lie or force every new failure mode to change the
    # published schema. `status_code` is the one key present on all of them.
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Failure-specific context, always including `status_code`. The other keys "
            "depend on the failure: `limit` on a 413, `violations` on a 422, "
            "`deadline_seconds` on a 504."
        ),
    )


class ErrorResponse(BaseModel):
    """
    The error envelope, so a generated client has a DTO for the failure path too.

    One shape for every failure is a property `errors.py` argues for in its own module
    docstring, and publishing it is what lets a client rely on it without a string test.
    """

    error: ErrorDetail
