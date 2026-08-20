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

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_matcher.domain.models.entities import FieldDecision
from nexus_matcher.domain.services.review_evidence import (
    Agreement,
    Separation,
    max_qualifier_segments,
)
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

# Bounds on the query-signal channel (see `FieldSpec.signals` and `MatchRequest.signals`).
#
# A resource limit, NOT an opinion about content. The channel exists so a deployment can
# send context this library has no vocabulary for, so nothing here inspects a key or a
# value -- only how much of it there is, and how deeply it nests. That is the same
# standard `_MAX_DOC` applies to a column comment: refuse an out-of-memory, never refuse
# a meaning.
_MAX_FIELD_SIGNAL_CHARS = 1024
_MAX_SIGNAL_DEPTH = 6

# The request-level budget is much larger because the abbreviation overlay lives here and
# a real approved-abbreviation catalog is thousands of rows. 512 KiB of characters holds
# roughly 11,000 `{short: long}` rows at realistic lengths, with room to spare over the
# largest catalogs this channel was designed for.
MAX_REQUEST_SIGNAL_CHARS = 524_288

# The largest one `FieldSpec` may be, in CHARACTERS, with every string at its bound.
#
# Exported because `limits.py` derives the raw-body byte cap from it rather than typing a
# number. The two have to move together: a byte cap BELOW what these bounds admit refuses
# a body `FieldSpec` itself accepts, so the caller reads two documents from this service
# and they contradict each other -- and a byte cap that stayed put while `_MAX_DOC` grew
# would do exactly that, silently, on the next edit to this block.
#
# `_MAX_FIELD_SIGNAL_CHARS` is in the sum for exactly that reason: `signals` is per-field
# payload, so it grows the per-field worst case and must grow the derived cap with it.
#
# WHAT THIS CONSTANT STILL DOES NOT COVER, said out loud. `MAX_REQUEST_SIGNAL_CHARS` is
# ENVELOPE-level, and `worst_case_body_bytes` has no envelope term to put it in -- its
# `_FRAMING_BYTES` covers `{"fields":[...],"top_k":5}` and nothing larger. It cannot be
# folded in here either: this number is multiplied by `max_batch_fields`, so an envelope
# allowance added to it would be counted up to 250 times and would raise the byte cap by
# hundreds of megabytes. The consequence is one narrow corner -- a maximal 250-field body
# in which every character is four UTF-8 bytes, carrying a maximal overlay, is inside
# every declared model bound and over the derived byte cap. It is the same shape as the
# `\\uXXXX` corner `worst_case_body_bytes` already documents, with the same escape hatch
# (`NEXUS_API_MAX_BODY_BYTES`), and the same fix: an envelope term in that function, whose
# module owns it. `MAX_REQUEST_SIGNAL_CHARS` is exported so that fix is a one-line change
# there rather than a second literal.
MAX_FIELD_SPEC_CHARS = _MAX_NAME + _MAX_PATH + _MAX_DOC + _MAX_TYPE + _MAX_FIELD_SIGNAL_CHARS

# The ceiling on `consistency_qualifier_segments`, DERIVED from the path bound rather than
# typed.
#
# It used to be the literal 8, which is a guess wearing a bound's clothes: it refused
# nothing this endpoint can distinguish and admitted five values it cannot. A concept key
# is built from the caller's own `path`, which is capped at `_MAX_PATH` characters, and
# `max_qualifier_segments` turns a character budget into the largest number of qualifier
# segments that can still change a key -- past that point the grouping slices the same
# segment list and produces byte-identical keys, so a larger value is inert rather than
# refused. Deriving it means a change to `_MAX_PATH` moves this with it instead of leaving
# a stale literal that says a value is out of range when the paths grew past it.
MAX_QUALIFIER_SEGMENTS = max_qualifier_segments(_MAX_PATH)


def _signal_map_chars(value: Any, depth: int = 0) -> int:
    """
    How many characters a signal map spends, and how deep it goes.

    Returns the character count, or -1 when the structure nests deeper than
    `_MAX_SIGNAL_DEPTH`. Depth is bounded rather than trusted because the channel accepts
    arbitrary JSON, and a body that is small on the wire can still be a nesting bomb after
    parsing -- and because the alternative, recursing until Python's own limit, turns a
    caller's mistake into a 500.
    """
    if depth > _MAX_SIGNAL_DEPTH:
        return -1
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            inner = _signal_map_chars(item, depth + 1)
            if inner < 0:
                return -1
            total += len(str(key)) + inner
        return total
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            inner = _signal_map_chars(item, depth + 1)
            if inner < 0:
                return -1
            total += inner
        return total
    return len(str(value))


def _check_signal_budget(signals: dict[str, Any], budget: int, where: str) -> None:
    """Refuse a signal map that is over budget or over-nested, naming which."""
    if not signals:
        return
    size = _signal_map_chars(signals)
    if size < 0:
        raise ValueError(
            f"`{where}` nests deeper than {_MAX_SIGNAL_DEPTH} levels. The query-signal "
            f"channel accepts any keys this server does not recognise, but it is bounded: "
            f"send a flatter structure."
        )
    if size > budget:
        raise ValueError(
            f"`{where}` spends {size} characters, over this server's budget of {budget}. "
            f"The query-signal channel is bounded so that an unrecognised key cannot "
            f"exhaust the server's memory; send a smaller map."
        )


_SIGNALS_DESCRIPTION = (
    "Query-side context this server does not derive: caller-supplied signals about the "
    "REQUEST rather than about the dictionary. Open by design -- keys this server does "
    "not recognise are carried and ignored, never refused, so a deployment can send "
    "signals this library has no opinion about. Recognised here: `abbreviations` (alias "
    "`abbreviation_overlay`), a `{short: long}` map merged into the query-side "
    "abbreviation expander FOR THIS REQUEST ONLY; `entity` (alias `parent_record`), the "
    "parent record a field came from, used as parent context when the path does not carry "
    "it; `domain` (aliases `domain_prior`, `namespace`), a domain hint that boosts "
    "dictionary terms declaring that domain. Every one is optional and omitting all of "
    "them is the shipped behaviour."
)


class FieldSpec(BaseModel):
    """
    One schema field a caller wants governance for.

    `extra="forbid"` on purpose, AND `signals` is an open map. Those two are not in
    tension -- they are the resolution of one, and it is worth stating which problem each
    solves.

    `forbid` exists because a misspelled `documentation` silently ignored would drop the
    column comment, and the column comment is real retrieval signal: the caller would get
    measurably worse matches with nothing in the response to show for it. That is the same
    standard `_load_matching_config` applies to a mistyped `auto_approve_treshold`.

    But `forbid` on this model also forecloses the extension the library most needs
    (AR-6): a deployment that knows something about the query -- its live abbreviation
    catalog, its parent record, its namespace -- had no way to say so, and got a 422 for
    trying. Relaxing to `extra="allow"` would buy that at the price of the typo gate,
    which is the wrong trade: a typo and an extension are different events and deserve
    different answers.

    So the extension point is DECLARED rather than implied. `signals` is a named field
    whose VALUE is open. A misspelled `doc` is still a 422; a signal this server has never
    heard of is carried and ignored; and the set of signals this server does interpret is
    published in the schema instead of living in the library's source.
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
    signals: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-FIELD query signals, overriding the request-level `signals` key by key. "
            "A flattened export can carry columns from several parent records, so `entity` "
            "cannot be request-scoped only. " + _SIGNALS_DESCRIPTION
        ),
    )

    @model_validator(mode="after")
    def _default_path_to_name(self) -> FieldSpec:
        """
        An omitted `path` falls back to `name`, so a caller with flat columns need not
        invent one. Done here rather than in the handler so the value the response is
        keyed by is fixed at parse time and there is exactly one place it comes from.

        The signal budget is checked in the same pass, so an over-large map is a 422
        naming the bound rather than a body the server buffers and then chokes on.
        """
        if not self.path:
            self.path = self.name
        _check_signal_budget(self.signals, _MAX_FIELD_SIGNAL_CHARS, "signals")
        return self


class MatchRequest(BaseModel):
    """
    A batch of fields to match. Identical for `/match` and `/match/batch`; only the
    server's field-count cap differs.

    `extra="ignore"` here, and `extra="forbid"` on `FieldSpec` one class up. The asymmetry
    is the point, and it is a judgement about which direction each envelope has to survive.

    THIS envelope has to be extensible, because it is the one a version skew lands on. Two
    services on one contract drift apart in both directions -- a caller sending a key a
    newer server understands gets a 422 from an older one, and the same caller keeps
    getting 422s from a newer server after the field is added but before their build is
    updated. Under `forbid` there is no way to add an optional request field to v1 at all
    without a coordinated deploy of somebody else's Java pipeline, so the first optional
    field this endpoint ever gains becomes a breaking change.

    What that costs is bounded, and worth stating rather than waving at. A misspelled
    `top_k` is now silently the default 5, and a misspelled `explain` silently false. Both
    are VISIBLE IN THE RESPONSE the caller already has -- five candidates when they asked
    for ten, no `explain` block when they asked for one -- and neither changes a
    classification. `fields` is required, so misspelling THAT is still a 422.

    `FieldSpec` keeps `forbid` for the opposite reason: a misspelled `doc` there is
    silently dropped retrieval signal, so the caller gets measurably worse matches with
    nothing in the response to show for it. That is the failure this endpoint exists to
    prevent, and it is exactly what an ignored key looks like.
    """

    model_config = ConfigDict(extra="ignore")

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
    signals: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Request-level query signals, applied to every field unless that field "
            "overrides the same key. The abbreviation overlay belongs here rather than on "
            "a field: it is a catalog, and it is scoped to this one request -- nothing "
            "about it survives into the next. " + _SIGNALS_DESCRIPTION
        ),
    )
    # The two evidence flags. Both default false and both are strictly ADDITIVE: with
    # neither set the response is byte-identical to the one this service sent before they
    # existed, which `test_review_evidence_wire.TestAdditive` asserts on the bytes rather
    # than on parsed JSON, because a re-ordering survives a parse and this body is diffed.
    contrast: bool = Field(
        default=False,
        description=(
            "Include a pairwise contrast between each field's rank 1 and rank 2: which "
            "signals separated them, by how much, and whether any single one of them "
            "accounts for the margin. Answers 'why not the other one', which the "
            "per-candidate `explain` block cannot -- it describes one candidate, and a "
            "reviewer's question is a subtraction. Independent of `explain`: the contrast "
            "carries the differences it needs, so a caller can have the comparison without "
            "the full weight breakdown on every candidate."
        ),
    )
    consistency: bool = Field(
        default=False,
        description=(
            "Include a cross-field consistency report: columns this request believes are "
            "the same business concept, and whether their rank-1 answers agree. REPORTING "
            "ONLY -- nothing in `results` or `fieldDecisions` changes, whatever it finds.\n\n"
            "UNPROVEN, AND OFF BY DEFAULT FOR THAT REASON. The grouping that decides which "
            "columns are 'the same concept' has been measured against schemas whose answers "
            "are known by construction, and on a repeated-leaf schema -- one leaf name "
            "governed separately in each of ~30 domains, the shape this feature was "
            "proposed for -- the loosest key scores pair-precision 0.0233 and emits four "
            "groups of which FOUR ARE COLLISIONS and none is a concept. Read "
            "`consistency_qualifier_segments` before turning this on."
        ),
    )
    consistency_qualifier_segments: int = Field(
        default=1,
        ge=0,
        le=MAX_QUALIFIER_SEGMENTS,
        description=(
            "How many of a column's nearest DECLARED path segments join its leaf in the "
            "concept key.\n\n"
            "1 -- the default -- means two columns are one concept only when they share a "
            "leaf AND the record they hang off. On every schema in this repository's "
            "generated corpus that setting produces NO GROUPS AT ALL, so it reports "
            "nothing and can therefore claim nothing false. It is the default because the "
            "alternative was measured and is worse: at 0 the key is the leaf alone, which "
            "scores pair-precision 0.86-1.00 on a parent-diverse mixture and 0.0233 on a "
            "repeated-leaf schema, where all four groups it emits merge distinct concepts. "
            "No setting of this parameter, or of any other, reaches precision above 0.024 "
            "while reporting anything at all on that shape -- established by searching 684 "
            "policies, not by observing one fixture.\n\n"
            "Segments are boundaries you declared (dots, or the `__` array boundary), never "
            "single underscores -- those are tokens inside a segment. A flattened name from "
            "a nested-but-array-free schema therefore has ONE segment, and every value of "
            "this parameter produces the same key for it. Values above the deepest path in "
            "your request are inert rather than refused; the ceiling is derived from the "
            "`path` length bound and is the largest value that can change any key this "
            "endpoint accepts."
        ),
    )

    @model_validator(mode="after")
    def _bound_request_signals(self) -> MatchRequest:
        """Refuse an over-large or over-nested request-level signal map at parse time."""
        _check_signal_budget(self.signals, MAX_REQUEST_SIGNAL_CHARS, "signals")
        return self


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

    THIS IS A CANDIDATE-LEVEL FACT, NOT AN INSTRUCTION TO INHERIT. It answers "what class
    does the matched entry carry", which stays true whatever the field's verdict is. A
    field whose `fieldDecisions` entry is `NO_MATCH` inherits NOTHING even though its
    rank-1 candidate here may show a populated class -- the candidates on a NO_MATCH field
    are evidence for a reviewer, not a classification. Read `fieldDecisions[path]` first;
    it is the field-level authority and this is not.

    `code`, `name` and `classification` are deliberately typed `str` and NOT as an enum or
    a `Literal`. They carry the caller's own controlled vocabulary; closing them would
    hard-code one organisation's taxonomy into the schema a Java client generates from, and
    this library ships no taxonomy at all.
    """

    code: str
    name: str
    classification: str
    personalInformation: bool
    directIdentifier: bool
    # Carried because the caller who reads this object is deciding HOW TO PROTECT a field,
    # and this is the only member that says what to do rather than what the field is. It
    # was resolved on every `MatchResult` and dropped at the wire, so the answer to "mask
    # it, tokenise it, or retain it seven years" lived only in a file the Java caller does
    # not have.
    #
    # `str | None`, and read with `getattr(..., None)` rather than through the `drift()`
    # path the five required members use: null is a documented value, not a defect. Five of
    # the nine classes in `examples/governance/protection_classes.json` declare it null, so
    # a required member here would refuse the repository's own example pack.
    enhancement: str | None = Field(
        description=(
            "The caller's own handling instruction for this class -- masking, "
            "tokenisation, a retention rule -- passed through untouched and never "
            "interpreted by this library. Null when the class declares none, which is not "
            "an error: it means the tier is the whole instruction. Free text in the "
            "caller's vocabulary, deliberately not a closed set."
        )
    )


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
            "THE SAME NUMBER as the candidate's `absoluteScore`, which is now present on "
            "every candidate without `explain`. Kept, and kept identical, for clients "
            "already reading it here; new clients should read `absoluteScore` and the "
            "`scoring` block that says what metric produced it. The only number in this "
            "object comparable ACROSS fields. Null when the dense retriever did not "
            "return this candidate."
        )
    )


class SourceMetadataView(BaseModel):
    """
    The dictionary entry's PASS-THROUGH PLANE: the deployment's own enrichment columns,
    carried the length of the pipeline and never interpreted.

    ## What it is for

    A deployment's glossary carries columns this library has no opinion about -- a
    steward, a review date, an upstream system's own identifier, a lifecycle token. The
    loader is told which columns to carry (`metadata_columns=`, or every unmapped column
    by default); they ride on the entry, through the index, and come back here. The
    library never has to know what any of them MEAN, and the deployment never has to join
    the response back against its own spreadsheet to find out.

    ## The three rules a consumer may rely on

    **NOTHING IN HERE IS READ.** No score, no ranking, no threshold and no governance
    decision depends on a key or a value in this map. Two responses that differ only in
    what these values say are identical everywhere else, byte for byte. That is the whole
    bargain: the plane is carried BECAUSE it is not interpreted, and a library that
    started branching on `values['...']` would be one specific enterprise's matcher
    wearing a generic name.

    **VALUES ARE THE CALLER'S OWN VOCABULARY, so they are open.** Deliberately not an
    enum, not a `Literal` and not typed narrower than the source: the keys come from the
    deployment's configuration and the values from its cells. Anything closed here would
    hard-code one organisation's spreadsheet into the schema a client generates from.

    **WHAT WENT IN COMES BACK.** No trimming, no case folding, no re-ordering, no
    re-encoding: keys keep the order the loader wrote them in, and a string is the same
    string. This body is ASCII-only (see `matching.DeterministicJSONResponse`), so a value
    carrying a non-breaking space, an em-dash or an accent travels as its `\\uXXXX`
    escape -- a JSON escape, not a substitution, so any conformant parser hands back the
    original characters. Leading and trailing whitespace is significant and is preserved.

    ## What is NOT in here

    The loader writes four keys of its own into the same map (`ingest.
    METADATA_RESERVED_KEYS`) and those are not the caller's enrichment, so they are not in
    `values`. `metadata_truncated` is surfaced properly as `droppedKeyCount` below. The
    other three are the loader's evidence about the SOURCE FILE -- the raw classification
    text, the raw protection-code token and the per-row governance problems -- and they
    stay on the library side of the boundary on purpose: the raw code token is the one
    place a token the caller's vocabulary REFUSED survives, and publishing it beside
    `governance.code` on a governance artifact is an invitation to apply a class nobody
    defined. A library caller reads them off `DictionaryEntry.source_metadata`; the load
    report counts them.
    """

    values: dict[str, Any] = Field(
        description=(
            "The deployment's own enrichment columns for the matched entry, in the order "
            "the loader carried them, passed through untouched and never interpreted by "
            "this library. Keys are the deployment's configured column names and values "
            "are its cells, so BOTH are open: this library ships no taxonomy and cannot "
            "enumerate either. Empty when the entry carries no pass-through columns -- "
            "an empty object, never a missing key. Values are whatever the source held "
            "(usually text; a JSON or Parquet glossary can hold numbers, booleans, nulls, "
            "lists or nested objects), preserved as-is; anything JSON cannot represent "
            "natively is rendered as text and named in `renderedKeys`."
        )
    )
    droppedKeyCount: int = Field(
        description=(
            "How many pass-through keys the LOADER dropped from this entry to fit its "
            "per-entry size cap (`metadata_max_bytes`, 1 KiB by default). 0 means this "
            "map is the whole plane the source row supplied; any other number means "
            "`values` is a BOUNDED SUBSEQUENCE of it and this response is not the place "
            "to read that row from. It is a count and not a list of names because the "
            "loader keeps the count -- the dropped names went with the dropped values. "
            "The cap exists so a careless column mapping cannot rake an entire "
            "spreadsheet row into the index; raise or lift it at load time if a "
            "deployment needs wider rows, having measured its own."
        )
    )
    renderedKeys: list[str] = Field(
        description=(
            "The keys of `values` whose value is the source value RENDERED AS TEXT rather "
            "than the source value itself, in the same order they appear in `values`. "
            "Empty for every source JSON can represent natively, which is every "
            "delimited-text glossary. It is non-empty when a spreadsheet or database "
            "column held something JSON has no form for -- a date or timestamp cell, a "
            "decimal, a binary blob, a non-finite number -- in which case that key's "
            "value is that object's text form and the exact original type is not "
            "recoverable from this response. Named rather than silently coerced, and "
            "rendered rather than refused, because one date column in a glossary must not "
            "turn every match response into a 500."
        )
    )


class MatchProvenanceView(str, Enum):
    """Where a candidate came from. The library's OWN vocabulary, so it closes.

    A caller-supplied vocabulary must never close -- a generated client that refuses a
    value a newer deployment sends is worse than one that carries it. This is the opposite
    case: these two values are decided here, and a third would be a change this library
    made deliberately.
    """

    RETRIEVAL = "RETRIEVAL"
    APPROVED_PAIR = "APPROVED_PAIR"


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
            "`governance`. For the ONE verdict per column, read `fieldDecisions[path]` -- "
            "this value cannot express 'nothing matched' (see `FieldDecision.NO_MATCH`)."
        )
    )
    # APPENDED to the candidate object, after `decision`, because `CANDIDATE_KEYS` and
    # their order are the contract a Java client generated against and appending is the
    # only additive edit to a key order. `explain` is still emitted last, when asked for,
    # so this key sits at a stable position whether or not `explain` was requested.
    #
    # `float | None`, not `float`: null when the dense retriever did not return this
    # candidate at all, so a narrower published type would be a schema the service
    # contradicts on its own fixture.
    absoluteScore: float | None = Field(
        description=(
            "The RAW dense-retrieval score for this candidate, present on EVERY "
            "candidate and not gated behind `explain`. The only number on a candidate "
            "comparable ACROSS fields: `confidence` is min-max normalised within one "
            "field's shortlist, so its floor is structural (see "
            "`scoring.confidenceFloor`) and a terrible rank-1 match still scores above "
            "it. This one has no floor. Identical to `explain.absoluteCosine`, which is "
            "kept for clients already reading it. Read `scoring.absoluteScoreMetric` "
            "before treating it as a cosine, and `scoring.absoluteScorePooledOverAliases` "
            "before treating it as a similarity to the entry's own text. Null when the "
            "dense retriever did not return this candidate -- it reached the shortlist "
            "through the lexical arm alone -- which is not the same as zero."
        )
    )
    # APPENDED after `absoluteScore` and BEFORE `explain`, which is the only position that
    # is additive twice over: the nine keys before it are the shape a Java client has
    # already generated against, and `explain` keeps its place as the last key of the
    # object whether or not it was asked for. Same rule, same reason, as `absoluteScore`
    # one member up and `enhancement` inside `governance`.
    #
    # Present on EVERY candidate, never null and never absent -- an entry with no
    # pass-through columns gets an empty `values`. "This entry carries no enrichment" and
    # "this response dropped it" must not look alike to a client whose pipeline is about
    # to write these columns into its own model.
    sourceMetadata: SourceMetadataView = Field(
        description=(
            "The deployment's own enrichment columns for the MATCHED ENTRY, carried "
            "through the pipeline and never interpreted -- see `SourceMetadataView`. This "
            "is a fact about the entry, not about the match: the identical object comes "
            "back from `GET /api/v1/lookup/{id}` for the same id, so a caller can feed a "
            "looked-up entry and a matched one into one code path. Present on every "
            "candidate, with an empty `values` when the entry carries none."
        )
    )
    # APPENDED after `sourceMetadata`, same rule as every member above it: `explain` stays
    # the last key of the object.
    #
    # This member exists because `confidence` alone CANNOT identify where an answer came
    # from. The bypass writes `confidence = 1.0`, and shipped source used to assert that
    # value was "outside the range the model can produce" -- it is not. The default weights
    # sum to exactly 1.0 and every signal caps at 1.0, so ordinary retrieval reaches 1.0
    # whenever all five are maximal; two independent constructions did it. A reviewer's
    # answer and a very good match were indistinguishable on `(confidence, decision)`.
    #
    # A value that says where the answer came from cannot collide with a score, which is
    # why this is a member rather than a cleverer sentinel. Closed set: these are the
    # library's own vocabulary, not the caller's.
    provenance: MatchProvenanceView = Field(
        description=(
            "Where this candidate came from. `RETRIEVAL` means the pipeline scored it; "
            "`APPROVED_PAIR` means a reviewer decided it and matching was skipped. Read "
            "this, NOT `confidence`, to tell the two apart: a retrieved candidate can "
            "legitimately reach `confidence` 1.0, so the number is not a discriminator. "
            "An `APPROVED_PAIR` candidate carries no `absoluteScore` and no `explain`, "
            "because nothing measured it -- absent rather than zero, since zero would be "
            "a measurement nobody took. Present on every candidate, never null."
        )
    )
    explain: ExplainView | None = None


class VocabularyView(BaseModel):
    """
    The two facts about the loaded vocabulary that a response cannot be READ without.

    Not a dump of the caller's catalog -- that is their file and they have it. This is the
    minimum needed to interpret the response it arrives with, which is why it rides on the
    response rather than sitting on an endpoint somebody has to know to call. The body is a
    governance artifact that gets pasted into a ticket and diffed; an artifact whose null
    means "ask a second system" is not one.

    Constant per deployment and roughly 120 bytes, so it is a rounding error against a
    response that carries candidates.
    """

    openClassification: str = Field(
        description=(
            "The tier a field with no protection code sits at, named by the caller's own "
            "vocabulary. This is what a `governance` of null MEANS on any candidate that "
            "is not a rejected rank 1 -- without it, null is a value a client cannot "
            "resolve without the vocabulary file. `UNCLASSIFIED` is this library's "
            "sentinel and indicates no vocabulary is configured; it is deliberately not a "
            "word a real taxonomy uses, so it cannot be mistaken for a real tier."
        )
    )
    tiersMostOpenFirst: list[str] = Field(
        description=(
            "The caller's declared tier ordering, most open first, from their own "
            "`tiers_most_open_first`. The ONLY thing that can rank two classifications "
            "against each other: this library ranks nothing and supplies no tiers. Empty "
            "when the vocabulary declares no ordering -- treat tiers as incomparable "
            "there, never as alphabetical, which sorts CONFIDENTIAL above PUBLIC."
        )
    )


class ScoringContractView(BaseModel):
    """
    What each number in this response MEANS, shipped in the response that carries them.

    The same instinct as `VocabularyView` one class up, and deliberately the same pattern
    rather than a second one: that block ships the tier ordering so a client need not
    hard-code somebody's taxonomy, and this block ships the scale contract so a client
    need not read this library's source to learn whether a number may be compared against
    a constant.

    ## The contradiction this exists to settle

    The library documents `confidence` as rank-relative and says do not threshold on it,
    and then ships `auto_approve_threshold = 0.87`, which is a threshold on it. Both
    cannot be right, and a consumer cannot tell which to believe. The resolution, stated
    here in machine-readable form: `confidence` is comparable WITHIN one field -- rank 1
    against rank 2 of the same field, and against a fixed cut point applied per field --
    and is NOT a cross-field quality score, because min-max normalisation puts every
    field's rank 1 at or above `confidenceFloor` whether the match is excellent or
    absurd. `absoluteScore` is the number that is comparable across fields.

    ## The scopes

    `WITHIN_FIELD` -- two values may be compared only when they come from the same field.
    `ACROSS_FIELDS` -- also comparable between two fields of the same response.
    `ACROSS_RUNS` -- also comparable between responses.

    Ordered narrowest first in `comparabilityScopesNarrowestFirst`, so a client can rank
    them without hard-coding the order, and a wider scope implies the narrower ones.

    **Nothing this library emits is `ACROSS_RUNS` today.** That is not an omission from
    this block, it is the honest state of the art here: none of these numbers is
    calibrated, so none behaves like a probability that a match is correct. `absoluteScore`
    comes closest and is stable between runs only while the embedding model, the
    dictionary and the vector store's metric are unchanged -- which is a precondition, not
    a property of the number.
    """

    confidenceFloor: float | None = Field(
        description=(
            "The lowest `confidence` a rank-1 candidate can carry under this server's "
            "configuration -- `semantic_weight * fusion_alpha`, which is 0.63 for the "
            "shipped defaults. NEVER SET A CONFIDENCE THRESHOLD AT OR BELOW IT: it "
            "selects nothing however bad the matches are, which is a filter that reports "
            "'nothing to review' on a schema where nothing is trustworthy. This library "
            "shipped exactly that defect once. Null means the floor does not hold or "
            "could not be verified for this response -- a reranker replaces the fused "
            "score with its own and the derivation lapses, and the value is also nulled "
            "rather than published if any rank-1 confidence in THIS response sits below "
            "it. A bound that is wrong about its own response is worse than no bound."
        )
    )
    absoluteScoreFloor: float | None = Field(
        description=(
            "The `absoluteScore` beneath which this server reports a field as NO_MATCH, "
            "or null when no floor is configured -- which is the default. This library "
            "ships no floor and will not invent one: a floor is a statement about a score "
            "distribution, and the distribution belongs to a dictionary and a set of "
            "field names this library has never seen. While it is null, `fieldDecisions` "
            "can only report NO_MATCH for a field that came back with no candidates at "
            "all."
        )
    )
    absoluteScoreMetric: str = Field(
        description=(
            "The distance metric the configured vector store declares, so a client never "
            "has to ASSUME `absoluteScore` is a cosine. `cosine` under the shipped "
            "wiring, in which case it is a genuine cosine similarity in [-1, 1]. A "
            "deployment supplying its own store may report `dot` or `euclidean`, in which "
            "case the number is monotone in similarity but is neither bounded nor a "
            "cosine, and a floor chosen for one metric is meaningless under another. "
            "`unknown` means the store declares nothing, which is NOT a synonym for "
            "cosine. Free text from the store, deliberately not a closed set."
        )
    )
    absoluteScorePooledOverAliases: bool = Field(
        description=(
            "True when this server indexes fabricated technical spellings of each "
            "dictionary entry (`dictionary_alias_count` above zero). `absoluteScore` is "
            "then the BEST score over an entry's spellings rather than the similarity to "
            "the entry's own text, so an entry can look confident on a spelling that was "
            "invented for it. False under the shipped configuration. Read it before "
            "comparing an absoluteScore against a floor measured on a deployment where it "
            "was the other value."
        )
    )
    thresholdableAcrossFields: list[str] = Field(
        description=(
            "The response numbers a client may legitimately compare against a CONSTANT "
            "across different fields. Everything absent from this list is comparable only "
            "within one field, or not at all. Derived from `comparability`, and stated "
            "separately because it is the one question a consumer actually has."
        )
    )
    comparabilityScopesNarrowestFirst: list[str] = Field(
        description=(
            "The scale vocabulary, narrowest first, so a client can rank two scopes "
            "without hard-coding the order. A wider scope implies every narrower one."
        )
    )
    comparability: dict[str, str | None] = Field(
        description=(
            "One entry per numeric field this response can carry, keyed by its path in "
            "the response body (`confidence`, `absoluteScore`, `explain.scores.lexical`, "
            "...), naming the WIDEST scope over which two of its values may be compared. "
            "Null means this server does not declare a scope for that number -- a signal "
            "added after this contract was written -- and an undeclared number must not "
            "be compared with anything."
        )
    )


class SignalDifferenceView(BaseModel):
    """
    One weighted signal's contribution to the margin between rank 1 and rank 2.

    `delta` is EXACTLY the subtraction a reader would do on the two candidates'
    `explain.scores` entries -- both operands are rounded to the published precision
    before they are subtracted, so a reviewer redoing it by hand gets this number and not
    one that disagrees in the last place.
    """

    signal: str = Field(
        description=(
            "The signal's name, the same key it carries in `explain.scores` and `explain.weights`."
        )
    )
    topScore: float
    runnerUpScore: float
    delta: float = Field(description="`topScore - runnerUpScore`. Negative where rank 2 won it.")
    weight: float = Field(
        description="The live matcher's weight for this signal, as `explain.weights` reports it."
    )
    weightedDelta: float = Field(
        description=(
            "`delta * weight`: this signal's share of the confidence gap. The shares sum "
            "to `confidenceGap`, and the service refuses the response rather than send one "
            "where they do not."
        )
    )
    separating: bool = Field(
        description=(
            "False when the two scores differ by no more than `contrast.resolution` -- the "
            "smallest difference the published numbers can express. A signal that is not "
            "separating is never named as a cause: a reason invisible in the artifact the "
            "reviewer is holding is an invented one."
        )
    )
    deciding: bool = Field(
        description=(
            "True when removing this signal's contribution would leave rank 2 level with "
            "or ahead of rank 1. Arithmetic, not judgement -- and it can be true of none "
            "of them, which means no single signal carried the margin. Always false on a "
            "`TIED` contrast, where nothing decided the order."
        )
    )


class ContrastView(BaseModel):
    """
    Rank 1 against rank 2 for one field: what separated them and what decided it.

    ## The question this answers, and the one `explain` answers

    `explain` reports why the winner scored what it did. A reviewer looking at a
    surprising match does not want a weight breakdown -- the weights are the same for
    every candidate and are already published. They want to know why not the other one,
    and that is a subtraction between two candidates rather than a description of one.

    ## What is deliberately NOT claimed

    A difference at or below `contrast.resolution` is not reported as separating and can
    never be named as a cause. When the whole margin is at or below it, `separation` is
    `TIED`, `largestDifference` is null and `decidingSignals` is empty: the order came
    from the matcher's own sort, and dressing a sort order up as a finding is how a
    review surface starts producing reasons that are not reasons. The per-signal
    differences are still reported on a tie, because two signals that disagree and cancel
    is precisely the case worth seeing.

    ## The two facts that are not about scoring at all

    `governanceDiffers` and `domainDiffers` come from the two dictionary ENTRIES rather
    than from any signal, and they are usually what settles a review: that rank 1 is a
    direct-identifier class and rank 2 is not is the deciding fact far more often than a
    fourth-decimal score difference is. They are read from the entries' own codes, so a
    rank-1 REJECT -- which carries no class by design -- does not read as "these two are
    classified differently".
    """

    topGovernanceId: str
    runnerUpGovernanceId: str
    topConfidence: float
    runnerUpConfidence: float
    confidenceGap: float = Field(
        description=(
            "`topConfidence - runnerUpConfidence`. Comparable WITHIN this field only -- "
            "see `contrast.comparability` -- because `confidence` is, and a difference is "
            "no more comparable than its operands."
        )
    )
    signalGap: float = Field(
        description=(
            "The same margin reached the other way: the sum of every `weightedDelta`. "
            "Published so the arithmetic can be checked from the response alone. The "
            "service verifies the two against each other and refuses to answer rather than "
            "send a contrast that does not close."
        )
    )
    # CLOSED, and a named component, for the same reason `MatchDecision` and
    # `FieldDecision` are: this is the LIBRARY'S OWN word for a state it computes, not a
    # caller's taxonomy, so the set of values is ours to freeze and a generated client is
    # entitled to a real enum for it. Nothing caller-supplied closes anywhere on this wire
    # -- governance codes, protection classes, tiers, domains and concepts all stay open
    # strings -- and this does not widen that rule, it applies it consistently.
    separation: Separation = Field(
        description=(
            "`SEPARATED` when the margin exceeds `contrast.resolution`; `TIED` when it does "
            "not, meaning the two candidates are level in every number this response "
            "publishes and the ordering between them came from the matcher's sort."
        )
    )
    largestDifference: str | None = Field(
        description=(
            "The separating signal with the largest weighted difference -- the headline "
            "answer to 'what separated these two'. Null on a `TIED` contrast, and null when "
            "no signal differs by more than the resolution."
        )
    )
    decidingSignals: list[str] = Field(
        description=(
            "Every signal whose removal would leave rank 2 level with or ahead of rank 1. "
            "EMPTY IS A REAL ANSWER and the common one on a wide margin: it means no single "
            "signal carried it. Always empty on a `TIED` contrast."
        )
    )
    governanceDiffers: bool = Field(
        description=(
            "Whether the two entries carry different protection codes. Read from the "
            "entries, not from the resolved `governance` on the candidates, so a rank-1 "
            "REJECT does not read as a difference that is not there."
        )
    )
    domainDiffers: bool = Field(description="Whether the two entries declare different domains.")
    signals: list[SignalDifferenceView] = Field(
        description=(
            "One entry per weighted signal, LARGEST WEIGHTED DIFFERENCE FIRST, with ties "
            "broken by the order the signals are declared in so two identical requests "
            "order this list identically."
        )
    )


class ContrastReportView(BaseModel):
    """
    The contrast block: one entry per input field, present only when `contrast` was asked
    for.

    `fields` carries EVERY input path in the order it was sent, exactly like `results` and
    `fieldDecisions`, with an explicit null for a field that has no runner-up to contrast.
    "This field had one candidate" and "this pass skipped it" must not look alike.
    """

    resolution: float = Field(
        description=(
            "The smallest difference the numbers in this response can express, which is "
            "the precision every published float is rounded to. Nothing below it is "
            "reported as separating and nothing below it is named as a cause."
        )
    )
    comparability: dict[str, Any] = Field(
        description=(
            "The scale contract for the contrast's own numbers, in the vocabulary "
            "`scoring.comparabilityScopesNarrowestFirst` publishes. `confidenceGap` names "
            "the scope of the gap; `signals` names the scope of each signal's `delta` and "
            "`weightedDelta`. Derived from `scoring.comparability` rather than restated, so "
            "a number whose scope changes cannot keep a stale entry here."
        )
    )
    fields: dict[str, ContrastView | None] = Field(
        description=(
            "One entry per input field, keyed and ordered exactly like `results`. Null "
            "where the field has fewer than two candidates."
        )
    )


class ConceptGroupView(BaseModel):
    """
    One group of columns this request believes are the same business concept, and the
    answers they got.

    `majorityGovernanceId` IS NOT AN INSTRUCTION. It is published so a reviewer can see
    where the weight of evidence sits; nothing in this library applies it, and
    `consistency.promotionApplied` is false for that reason.
    """

    concept: str = Field(
        description=(
            "The concept key, as a printable label: the qualifier segments, the leaf's "
            "normalised tokens, its class word and the data type, separated by `|`. Stable "
            "for a given request and grouping policy, so it can be quoted in a ticket, but "
            "it is a grouping artifact rather than a name anyone chose -- do not key a "
            "downstream system on it."
        )
    )
    fields: list[str] = Field(description="The group's members, in the order they were sent.")
    answers: dict[str, str | None] = Field(
        description=(
            "Each member's rank-1 governance id, or null where the field has no answer to "
            "give -- no candidates, or a `fieldDecisions` verdict of NO_MATCH, which "
            "inherits nothing. A null is SILENCE, not a dissenting answer: counting it as "
            "one would report a disagreement in a group where only one column was answered."
        )
    )
    distinctAnswers: int = Field(description="How many different non-null answers the group got.")
    # CLOSED, and a named component -- see the note on `ContrastView.separation`.
    agreement: Agreement = Field(
        description=(
            "`AGREE` when two or more members answered and all agree, `DISAGREE` when two "
            "or more answered and they do not, `UNDECIDED` when fewer than two members "
            "answered at all. `UNDECIDED` is deliberately not `AGREE`: one answer and five "
            "blanks is not five columns confirming each other.\n\n"
            "A `DISAGREE` is only evidence about the matcher if the GROUP is real. Read it "
            "beside `distinctAnswers`: a group whose `distinctAnswers` approaches its "
            "member count is a collision in the grouping, not a contradiction in the "
            "answers. Measured on a repeated-leaf schema at `qualifierSegments` 0, four of "
            "four `DISAGREE` findings were that."
        )
    )
    majorityGovernanceId: str | None = Field(
        description=(
            "The modal answer within the group, or null when no single answer holds a "
            "plurality. Evidence, never an instruction -- see this model's docstring."
        )
    )
    majorityCount: int = Field(
        description="How many members gave the majority answer; 0 when there is none."
    )


class ConsistencyReportView(BaseModel):
    """
    The consistency block, present only when `consistency` was asked for.

    ## Why this is deployable without labelled data

    Fields are matched one at a time and independently, which throws away a constraint
    that costs nothing to check: two columns that are the same concept should get the same
    answer. Nothing enforces that, and -- the part that matters -- nothing NOTICES when it
    fails. Detecting the disagreement does not require knowing which answer is right,
    which is what makes this shippable today.

    ## Why it reports and does not override

    Promoting a group's majority is a decision that can be wrong in a NEW way: it can move
    a correct answer to an incorrect one, which surfacing a disagreement cannot.
    `promotionApplied` is false and is published as a fact about this response rather than
    left implicit.

    ## Grouping is the whole difficulty, and IT IS NOT SOLVED

    Too loose and distinct concepts merge, which MANUFACTURES disagreement and fills a
    reviewer's queue with findings that were never real. Too tight and nothing groups.

    Measured against generated schemas whose answers are known by construction, pair-wise,
    over the columns the fixture has a single unambiguous answer for:

        profile             qualifierSegments   precision   recall   groups   collisions
        repeated-leaf                       0      0.0233   1.0000        4        4 of 4
        repeated-leaf                    >= 1         n/a   0.0000        0             -
        parent-diverse mix                  0      0.8571-      0.0647-     -    0-2 of 30+
                                                   1.0000       0.1371
        parent-diverse mix               >= 1         n/a   0.0000        0             -

    On the repeated-leaf shape -- one leaf name governed separately in each of ~30 domains,
    which is the construction this feature was proposed for -- the loose key merges 87
    columns spanning 29 distinct correct answers and reports them as contradicting each
    other. Every one of the four findings it produces is a false positive. Searching all
    684 policies in the published space finds none that reports anything on that shape at a
    precision above 0.024.

    SO THE DEFAULT REPORTS NOTHING RATHER THAN REPORTING WRONGLY. `consistency` is off, and
    when it is switched on `consistency_qualifier_segments` defaults to 1, which emits no
    group on any generated profile. The loose key is one integer away for a deployment that
    has measured its own schemas, and the numbers are on the parameter.

    HOW TO READ A FINDING YOU DID ASK FOR. `grouping` publishes the policy that produced
    these groups, because a finding cannot be judged without the rule that made it. Compare
    every group's `distinctAnswers` against its member count: when the two are close the
    group is a collision and the disagreement is the grouping's, not the matcher's.
    """

    grouping: dict[str, Any] = Field(
        description=(
            "The policy these groups were built under: `qualifierSegments`, "
            "`includeDataType`, `orderSensitive` and `minGroupSize`. Published because a "
            "finding cannot be judged without the rule that produced it."
        )
    )
    groupsFound: int
    fieldsGrouped: int = Field(
        description=(
            "How many of this request's fields fell into a group of two or more. A column "
            "that shares its concept with nothing else is not reported: it cannot disagree "
            "with anyone."
        )
    )
    groupsDisagreeing: int = Field(description="How many groups have `agreement` of DISAGREE.")
    promotionApplied: bool = Field(
        description=(
            "Always false. This block changed nothing in `results` or `fieldDecisions`, "
            "and states so machine-readably rather than leaving a consumer to infer it."
        )
    )
    groups: list[ConceptGroupView] = Field(
        description=(
            "The groups, ordered by where their first member appeared in the request, so "
            "two identical requests produce the same list."
        )
    )


class MatchResponseView(BaseModel):
    """
    The whole response: one list per input field, keyed by the caller's own `path`.

    Every input path appears exactly once, in the order it was sent, whether or not
    anything matched it -- a field with no candidates gets an empty list, never a missing
    key. That is the conservation law this endpoint is built around (NM-0005), and it
    holds for `fieldDecisions` too: the two maps carry the same keys in the same order.
    """

    results: dict[str, list[MatchCandidateView]]
    # Second, never first: `results` was the whole body and a Java client generated against
    # that shape must keep reading it at the same key. Appending is additive on the wire;
    # reordering is not. Everything below is appended for the same reason.
    vocabulary: VocabularyView
    fieldDecisions: dict[str, FieldDecision] = Field(
        description=(
            "ONE verdict per field, keyed and ordered exactly like `results`. This is the "
            "value a consumer writes into a per-column decision, and it is published "
            "rather than left to the client to derive, because a roll-up rule every "
            "client reconstructs for itself is a contract nobody wrote down.\n\n"
            "AUTO_APPROVE, REVIEW and REJECT are rank 1's own `decision`, passed through "
            "unchanged. NO_MATCH is the state the per-candidate vocabulary cannot "
            "express: rank-1 `confidence` has a structural floor (`scoring."
            "confidenceFloor`, 0.63 shipped) above the server's review threshold (0.50), "
            "so rank 1 can never be REJECT on score alone and every field would otherwise "
            "come back at least REVIEW however irrelevant its best candidate is.\n\n"
            "NO_MATCH means: this response carries nothing this field may inherit from. "
            "Either the field came back with no candidates, or `scoring."
            "absoluteScoreFloor` is configured and rank 1 does not clear it. The "
            "candidates are still returned either way -- they are evidence for a reviewer, "
            "not an inheritance -- so on NO_MATCH read THIS field, not `results[path][0]."
            "governance`."
        )
    )
    scoring: ScoringContractView
    # APPENDED, and OPTIONAL, for the two reasons every other addition to this model was:
    # the four keys above are the shape a Java client has already generated against, and
    # appending is the only additive edit to a key order. Absent -- not null -- unless the
    # request asked for the block, so a caller who did not ask gets the body they got
    # before these existed, byte for byte.
    contrast: ContrastReportView | None = None
    consistency: ConsistencyReportView | None = None


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
