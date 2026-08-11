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
    controlled vocabulary. Null on a candidate whose entry has no code at all.
    """

    code: str
    name: str
    classification: str
    personalInformation: bool
    directIdentifier: bool


class MatchCandidateView(BaseModel):
    """One candidate for one field. `explain` is absent unless the request asked for it."""

    rank: int
    governanceId: str
    businessName: str
    definition: str
    domain: str
    governance: GovernanceView | None
    confidence: float
    decision: str
    explain: dict[str, Any] | None = None


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
