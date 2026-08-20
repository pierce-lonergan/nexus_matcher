"""
nexus_matcher.presentation.api.matching | Layer: PRESENTATION
POST /api/v1/match and /api/v1/match/batch -- matching over HTTP.

## Relationships
# DEPENDS_ON → application/use_cases/match_schema :: the matcher this wraps
# DEPENDS_ON → application/ingest :: METADATA_RESERVED_KEYS -- which keys of the
#              pass-through plane the LOADER wrote, as against the deployment's own
# DEPENDS_ON → domain/models/entities :: SchemaField in, MatchResult out
# DEPENDS_ON → presentation/api/limits :: admission control and the deadline
# DEPENDS_ON → presentation/api/errors :: every failure mode
# DEPENDS_ON → presentation/api/schemas :: the wire contract
# USED_BY    → presentation/api/app :: mounted by create_app

## Why this exists

Matching was Python-and-CLI only. The adopter's pipeline is Java, so every documented way
to use this library was unreachable to them; the REST surface was health and
introspection. This is the one thing blocking them.

## The three properties this module is responsible for

**CONSERVATION (NM-0005).** Every input field appears in the output, under the caller's
own `path`. A dict silently absorbs a collision, and a field that vanishes from a result
map inherits no governance while nothing raises -- the only symptom is a count nobody has
reason to check. `_project_results` checks it three independent ways and refuses the
response rather than trimming it. It holds for `fieldDecisions` too: a column with
candidates and no verdict is the same failure wearing a different key.

**DETERMINISM.** Two identical requests produce byte-identical bodies: keys in input
order, a fixed key order inside every object, floats rounded to a fixed precision, and an
ASCII-only renderer. The last one matters because this is a governance artifact that gets
pasted into tickets and diffed.

**DETERMINISTIC DEGRADATION.** Overload sheds with 503, the deadline answers 504, a
matcher failure is a named 500. It never hangs. See `limits.py`.

## The pass-through plane, and the one rule that keeps it safe

A deployment's glossary carries columns this library has no opinion about. The loader
already carried them onto the entry and into the index; this module is where they finally
reach a response, as `sourceMetadata` on every candidate. Without that last hop the
capability was unreachable over HTTP -- a deployment could send its own glossary through
this service and get back none of its own columns -- so the plane ended at the index and
the feature was, from a client's side, absent.

THE RULE THAT MAKES IT SAFE IS THAT NOTHING READS IT. No score, no ranking, no threshold
and no verdict may depend on a key or a value in that map. A library that starts branching
on one is no longer a generic matcher; it is one enterprise's matcher with a generic name,
and the next enterprise's columns mean something else. This module reads the plane's key
NAMES once, to tell the loader's own four annotations from the deployment's columns using
the list the loader publishes for it, and never reads a value at all. `test_metadata_plane`
holds the line by matching one schema against two dictionaries that differ ONLY in these
values and diffing the two responses.

## Coupling that is deliberate, and how it is made safe

Three private names on the application layer are read here, because the application layer
exposes no public equivalent:

  * `NexusMatcher._match_fields` -- the only way to match a LIST OF FIELDS. The public
    `match_schema` takes a schema source and runs a parser; this endpoint's caller has
    already parsed their schema and is sending fields. `tests/properties/test_conservation`
    and three museum entries reach the same method for the same reason. Its optional
    `signals` keyword carries the query-signal channel (AR-6), and is passed ONLY when the
    caller sent signals -- see `_invoke_matcher` for why that conditional is deliberate.
  * `NexusMatcher._config` -- the weights `explain` reports, read off the LIVE matcher so
    a tuned deployment gets a response that reproduces ITS numbers, not the shipped ones.
    Same pattern, and same justification, as the CLI's `_MATCHER_CONFIG_ATTR`.
  * `NexusMatcher._governance` -- the vocabulary every `MatchResult.governance` was
    resolved through, for the response's `vocabulary` block. Read from the same object that
    produced the nulls it explains, so the two cannot disagree.

None is left to fail obscurely. A missing `_match_fields` raises `ContractDriftError`, a
500 that names both sides; the other two fall back rather than take matching down, because
a renamed attribute there costs one optional block and not the endpoint. And the weights
are not trusted on their name alone -- when `explain` is requested the emitted numbers must
reproduce the emitted confidence, or the response is refused. A governance document whose
arithmetic does not close is worse than no document, because it is the one that gets used
as evidence.

A public `match_fields` on `NexusMatcher`, and a public accessor for its vocabulary, would
remove all three couplings; that file belongs to another lane.

## The two evidence blocks, and why neither is on by default

`contrast` and `consistency` are opt-in and STRICTLY ADDITIVE. A request that asks for
neither gets the four keys this response has always carried, byte for byte -- asserted on
the bytes in `test_review_evidence_wire.TestAdditive`, not on parsed JSON, because a
re-ordering survives a parse and this body is diffed by hand.

`contrast` answers the question `explain` cannot. `explain` reports why the WINNER scored
what it did; a reviewer looking at a surprising match wants to know why not the other one,
which is a subtraction between two candidates rather than a description of one. It names
the signals that separated rank 1 from rank 2, by how much, and whether any single one of
them accounts for the margin -- and it refuses to name a cause below the resolution of the
numbers the response publishes, because a reason the reviewer cannot see in the artifact
they are holding is an invented one. Its arithmetic is checked before it is sent, exactly
as `explain`'s is.

`consistency` REPORTS AND DOES NOT OVERRIDE. Fields are matched one at a time and
independently, which throws away a constraint that costs nothing to check: two columns
that are the same concept should get the same answer. Nothing enforces it and nothing
notices when it fails. Detecting the disagreement needs no ground truth, which is what
makes it deployable; ACTING on it -- promoting a group's majority -- is a decision that
can be wrong in a new way, so nothing here does it, and `promotionApplied` says so on the
wire. Both passes live in `domain/services/review_evidence`, which is where the reasoning
belongs; this module only projects them.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

# The names the LOADER writes into the pass-through plane, imported rather than restated.
# That constant is published for exactly this reader: whatever emits the plane has to tell
# the loader's own annotations from the deployment's columns, and a second hand-maintained
# copy of the list in this layer is how the two drift. Import-time rather than deferred
# because `application.ingest` is stdlib plus domain -- it does not pull the matching
# stack, which is the reason `MatchingConfig` below is deferred.
from nexus_matcher.application.ingest import METADATA_RESERVED_KEYS
from nexus_matcher.domain.governance import OPEN_CLASSIFICATION
from nexus_matcher.domain.models.entities import (
    FieldDecision,
    SchemaField,
    derive_field_decision,
)

# The two evidence passes, imported from the DOMAIN rather than written here. Both are
# arithmetic over match results with no HTTP in them, so a copy in this layer would put
# domain reasoning behind a transport and make it reachable only over a socket. Imported
# by module path rather than through `domain.services.__init__` on purpose: that file is
# the package's export list and belongs to whoever curates it, and this module does not
# need it to be edited in order to work.
# The candidate provenance vocabulary and its single reader, from the port that defines
# the bypass. Import-time rather than deferred: `domain.ports.review_feedback` is domain
# plus stdlib and pulls no part of the matching stack, which is the same test
# `application.ingest` above passes and `MatchingConfig` below fails.
from nexus_matcher.domain.ports.review_feedback import MatchProvenance, provenance_of
from nexus_matcher.domain.services.review_evidence import (
    Agreement,
    GroupingPolicy,
    SignalSpec,
    assess_consistency,
    contrast_top_two,
    group_by_concept,
)
from nexus_matcher.presentation.api.errors import (
    ConservationViolationError,
    MalformedRequestError,
    MatcherUnavailableError,
    MatchFailedError,
    RequestTooLargeError,
    drift,
)
from nexus_matcher.presentation.api.limits import run_bounded
from nexus_matcher.presentation.api.schemas import (
    ErrorResponse,
    MatchRequest,
    MatchResponseView,
)
from nexus_matcher.shared.types.base import DataType

if TYPE_CHECKING:
    from nexus_matcher.domain.models.entities import MatchResult
    from nexus_matcher.presentation.api.limits import BoundedWorkPool, MatchServiceLimits
    from nexus_matcher.presentation.api.schemas import FieldSpec

# =============================================================================
# CONSTANTS
# =============================================================================

# Private application-layer names this module reads. Named constants rather than inline
# strings so the coupling is greppable from one place when the other lane makes them
# public -- H-006's shape is exactly a change whose two halves live in different lanes.
_MATCH_FIELDS_ATTR = "_match_fields"
_MATCHER_CONFIG_ATTR = "_config"
# The vocabulary the matcher resolved every `MatchResult.governance` through. Read for the
# response's `vocabulary` block, which is what makes a `governance` of null readable.
_MATCHER_GOVERNANCE_ATTR = "_governance"

# Decimals kept for every emitted number. Six, matching the CLI's JSON writer, and for the
# same reason: with `explain` the response has to be self-checking, and an auditor
# recomputing sum(scores * weights) at four decimals can disagree with the emitted
# confidence in the fourth place -- indistinguishable from the tool getting it wrong.
_PRECISION = 6

# How far the recomputed weighted sum may sit from the emitted confidence before the
# response is refused. Roughly two orders of magnitude of headroom over the worst-case
# rounding of eleven terms: loose enough never to reject honest rounding, tight enough
# that a real disagreement cannot pass.
_REPRODUCTION_TOLERANCE = 10 ** -(_PRECISION - 1)

# (response key, ScoreBreakdown attribute, MatchingConfig weight attribute).
#
# One table rather than five inline reads, so the SCORE-TO-WEIGHT PAIRING lives in a
# single visible place: pairing a component with the wrong weight produces a response that
# is self-consistently wrong, which is the worst failure an audit surface has. The first
# key is `fusedRetrieval`, not `semantic`: the number is the min-max-normalised fused
# retrieval score and is rank-relative, so calling it semantic claims 90% similarity and
# delivers "ranked first among this field's candidates".
_SCORE_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("fusedRetrieval", "fused_retrieval_score", "semantic_weight"),
    ("lexical", "lexical_score", "lexical_weight"),
    ("editDistance", "edit_distance_score", "edit_distance_weight"),
    ("type", "type_compatibility_score", "type_weight"),
    ("domain", "domain_score", "domain_weight"),
)

# The scale vocabulary, NARROWEST FIRST. Published in the response so a client can rank
# two scopes without hard-coding an order, exactly as `tiersMostOpenFirst` does for
# governance tiers. A wider scope implies every narrower one.
_COMPARABILITY_SCOPES: tuple[str, ...] = ("WITHIN_FIELD", "ACROSS_FIELDS", "ACROSS_RUNS")

_WITHIN_FIELD, _ACROSS_FIELDS = _COMPARABILITY_SCOPES[:2]

# The widest scope over which two values of each emitted number may be compared, keyed by
# the number's PATH IN THE RESPONSE BODY -- which is what a client actually holds.
#
# This is the answer to the question the library has so far given two contradictory
# answers to: `confidence` is documented as rank-relative with "do not threshold on it",
# and the server ships `auto_approve_threshold = 0.87`, a threshold on it. Both are true
# of different uses, and the resolution is the table below.
#
# `confidence` and `fusedRetrieval` are WITHIN_FIELD because the fused retrieval score is
# min-max normalised over the candidates retrieved FOR ONE FIELD: rank 1 lands at or above
# `fusion_alpha` whether the match is excellent or absurd, so 0.72 on one field and 0.72
# on another are not the same claim. The remaining four signals are computed per
# (field, entry) pair with no per-field rescaling, so they carry the same meaning
# everywhere in one response.
#
# NOTHING IS ACROSS_RUNS, and that is the honest state of this library rather than a gap
# in this table. None of these numbers is calibrated -- none behaves like P(correct) --
# and every one of them moves with the configuration, the embedding model or the
# dictionary. `absoluteScore` is the closest, and it is stable between runs only while all
# three are unchanged, which is a precondition rather than a property of the number.
_COMPARABILITY: dict[str, str] = {
    "confidence": _WITHIN_FIELD,
    "absoluteScore": _ACROSS_FIELDS,
    "explain.absoluteCosine": _ACROSS_FIELDS,
    "explain.scores.fusedRetrieval": _WITHIN_FIELD,
    "explain.scores.lexical": _ACROSS_FIELDS,
    "explain.scores.editDistance": _ACROSS_FIELDS,
    "explain.scores.type": _ACROSS_FIELDS,
    "explain.scores.domain": _ACROSS_FIELDS,
}

# Distinguishes "the attribute is absent" from "the attribute is None". The governance
# contract needs both: an absent `MatchResult.governance` means the field has not landed
# yet, while a present None means the entry genuinely has no code and the response must
# carry an explicit null.
_ABSENT = object()

# =============================================================================
# THE PASS-THROUGH METADATA PLANE
# =============================================================================

# The loader's own keys, as a set for the one membership test below.
_RESERVED_METADATA_KEYS = frozenset(METADATA_RESERVED_KEYS)

# The loader's marker for "this map is a bounded subsequence of the source row", carrying
# the number of keys it dropped to meet the per-entry cap. Named here because this layer
# has to LIFT it out of the caller's vocabulary and publish it as a declared member --
# until now it was reachable only by a library caller, so a consumer over HTTP could not
# tell a whole plane from a trimmed one, which is the half of AR-1's bound that a bound
# nobody can observe does not deliver. Pinned against `METADATA_RESERVED_KEYS` by
# `test_metadata_plane`, so a rename in the loader is a red test rather than a marker that
# quietly stops being surfaced and starts being emitted as a caller column.
_TRUNCATION_MARKER = "metadata_truncated"

# How deep a nested pass-through value is walked before it is rendered as text instead.
# A glossary read from JSON or Parquet can hold arrays and objects in one cell, and those
# are honest JSON that must survive; a structure deeper than this is either a whole
# document raked in by a careless mapping or a cycle, and neither should be able to put
# this renderer into unbounded recursion on the event loop. Six is well past any real
# enrichment column and far short of Python's recursion limit.
_MAX_METADATA_DEPTH = 6


def _renders_as_json(value: Any, depth: int = 0) -> bool:
    """
    Whether `json.dumps` can render this value AS ITSELF under this module's settings.

    The question is not "is it convenient", it is which of two failures the caller gets. A
    glossary column holding a date cell, a decimal or a blob is ordinary -- a spreadsheet
    or a database gives back real objects, not text -- and this renderer runs with no
    `default=` hook and `allow_nan=False`, so an unrenderable leaf is a `TypeError` inside
    the response and therefore a 500 on EVERY match against that dictionary. One date
    column would take the whole matching service down for a value nobody scores on.

    So an unrenderable value is rendered as text and NAMED in `renderedKeys`, which is the
    same posture the rest of this module takes towards a lossy conversion: never silent,
    always declared. Everything JSON can carry natively -- text, numbers, booleans, nulls,
    arrays and objects -- passes through untouched, which is every value a delimited-text
    glossary can produce and so is the case that AR-1's byte-for-byte round trip is about.

    Three exclusions are deliberate. A non-finite float is refused because `allow_nan=
    False` refuses it, and `NaN` is not JSON. A dict with a non-string key is refused
    because `json.dumps` would COERCE it, silently, and two keys can coerce to one. Depth
    beyond `_MAX_METADATA_DEPTH` is refused so a cycle cannot recurse forever.
    """
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if depth >= _MAX_METADATA_DEPTH:
        return False
    if isinstance(value, list | tuple):
        return all(_renders_as_json(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _renders_as_json(item, depth + 1)
            for key, item in value.items()
        )
    return False


def _source_metadata_payload(entry: object) -> dict[str, Any]:
    """
    The entry's pass-through plane, and what this response is not telling you about it.

    ONE reader for both planes: `matching._candidate_payload` and `lookup._entry_payload`
    both call this, so "a looked-up entry carries the same enrichment as a matched one" is
    a fact by construction rather than a convention two files have to remember. It is also
    why this lives here and not in `lookup.py` -- the candidate is the surface the lookup
    plane is defined against.

    THIS LAYER READS THE KEY NAMES AND NOTHING ELSE, and the distinction is AR-1's first
    rule. No score, no ranking, no threshold and no verdict in this response depends on
    anything in this map; what happens below is that the loader's own four annotations are
    told apart from the deployment's columns BY NAME, using the list the loader publishes
    for the purpose. That is emission, not interpretation: no branch here can change a
    number, and `test_metadata_plane` pins it by matching the same fields against two
    dictionaries that differ only in these values and diffing the responses.

    The three reserved keys that are not `metadata_truncated` are dropped rather than
    emitted. They are the loader's evidence about the source FILE -- the raw classification
    text, the raw protection-code token, the per-row governance problems -- and the second
    of those is the only place a token the caller's own vocabulary REFUSED survives.
    Publishing a refused code next to `governance.code`, on a body whose reader is deciding
    how to protect a column, is how a class nobody defined gets applied. A library caller
    still has all four on `DictionaryEntry.source_metadata`.

    A NEW dict is built rather than the entry's own handed out, so nothing downstream can
    reach the index's live map through a response. It is deliberately shallow, and worth
    saying rather than implying: a nested list or object from a JSON glossary is placed in
    it by reference. Nothing here mutates one, and the renderer only reads -- but a future
    reader of this function should not take "a copy is built" to mean more than it does.
    """
    plane = getattr(entry, "source_metadata", None)
    if not isinstance(plane, Mapping):
        # A caller's own entry-shaped object with no plane at all. An empty one is the
        # honest answer -- it carries no enrichment -- and is what an entry loaded from a
        # glossary with no spare columns reports too.
        plane = {}

    values: dict[str, Any] = {}
    rendered: list[str] = []
    for key, value in plane.items():
        if key in _RESERVED_METADATA_KEYS:
            continue
        # `str(key)` is defensive only: every loader path builds this map from a header
        # row, so the keys are already text. It is here because `json.dumps` would coerce
        # a non-string key silently and two of them can coerce to one, which is a key
        # quietly answering for another key -- the shape of defect this whole module is
        # written against.
        name = str(key)
        if _renders_as_json(value):
            values[name] = value
        else:
            values[name] = str(value)
            rendered.append(name)

    # The count the loader recorded when it trimmed this entry, read by the loader's own
    # marker name. `type(...) is int` rather than `isinstance` on purpose: `True` is an
    # `int` in Python, and a source column that happened to carry this reserved name with
    # a boolean in it would otherwise be published as "one key was dropped".
    marker = plane.get(_TRUNCATION_MARKER)
    dropped = marker if type(marker) is int and marker >= 0 else 0

    return {
        "values": values,
        "droppedKeyCount": dropped,
        "renderedKeys": rendered,
    }


# =============================================================================
# RESPONSE RENDERING
# =============================================================================


class DeterministicJSONResponse(JSONResponse):
    """
    Render exactly the dict the handler built: no re-ordering, no NaN, pure ASCII.

    Starlette's default renderer is already compact and preserves insertion order, so two
    of the three come free. The two changes are deliberate:

    `allow_nan=False`. Python's json module emits bare `NaN` and `Infinity`, which are not
    JSON. A Java client parsing that gets an exception in the middle of a governance
    payload, which is a far worse failure than the 500 this raises instead.

    `ensure_ascii=True`. The body becomes pure ASCII, so an accented business name travels
    as an escape rather than as bytes that some intermediary re-encodes. Same choice, and
    the same reason, as the CLI's JSON writer: `json.loads` returns the original string
    either way, and the artifact is then byte-stable no matter whose console or log
    pipeline it passes through.
    """

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("ascii")


# =============================================================================
# MATCHER HANDLE
# =============================================================================


class MatcherHandle:
    """
    Holds the matcher the endpoints use -- and, when there is none, WHY.

    Separated from the router because the matcher is built during application startup
    (loading a dictionary means loading an encoder) while the routes are registered at
    import time. Without the handle the routes would have to be conditional on startup
    order, and an endpoint that exists only sometimes is an endpoint whose 404 means two
    different things.
    """

    def __init__(self) -> None:
        self._matcher: object | None = None
        self._reason: str = (
            "no dictionary is configured. Set NEXUS_API_DICTIONARY to a dictionary file "
            "(.xlsx or .csv), or pass matcher=... to create_app()."
        )

    def bind(self, matcher: object) -> None:
        """Install the matcher these endpoints will use."""
        self._matcher = matcher

    def record_failure(self, reason: str) -> None:
        """Record why there is no matcher, so the 503 can say something actionable."""
        self._matcher = None
        self._reason = reason

    @property
    def is_ready(self) -> bool:
        """Whether matching can be served. Reported through /health/ready."""
        return self._matcher is not None

    def require(self) -> object:
        """The matcher, or a 503 that names the reason there is not one."""
        if self._matcher is None:
            raise MatcherUnavailableError(
                message=f"The matching service is not ready: {self._reason}",
                details={"reason": self._reason},
            )
        return self._matcher


# =============================================================================
# TRANSLATION -- request in
# =============================================================================


def _to_schema_field(spec: FieldSpec) -> SchemaField:
    """
    One wire field to one domain field.

    `parent_path` is not bookkeeping. Hierarchical context injected into the retrieval
    query is the largest single accuracy factor measured on this task (+20 points of P@1),
    so a caller who sends `customer.contact.email` gets a materially better answer than one
    who sends `email` -- and dropping the derivation here would silently cost them that.

    `flattened_name` is set to the caller's own path so the matcher's result keys ARE the
    caller's keys. `_project_results` checks that independently of the order-based mapping,
    which gives two oracles for the conservation law instead of one that agrees with itself.

    Per-FIELD query signals (AR-6) travel under `QUERY_SIGNALS_METADATA_KEY`, nested
    rather than spread across `source_metadata`'s top level. `source_metadata` is a shared
    bag that already holds `flattened_name` and that the domain scorer already reads a bare
    `domain` key out of, so spreading signals over it would silently reinterpret data an
    existing caller may already put there. The key is absent entirely when the caller sent
    no signals, so a field built from a signal-free spec is the object it always was.
    """
    # Deferred for the same reason `MatchingConfig` is deferred below: the module that
    # owns this name pulls numpy and the whole matching stack, and importing it at module
    # scope would put that on the OpenAPI generation path. Imported rather than restated
    # because a hand-copied key is how the two halves of one contract drift apart.
    from nexus_matcher.application.use_cases.match_schema import QUERY_SIGNALS_METADATA_KEY

    parent, _, _leaf = spec.path.rpartition(".")
    source_metadata: dict[str, Any] = {"flattened_name": spec.path}
    if spec.signals:
        source_metadata[QUERY_SIGNALS_METADATA_KEY] = spec.signals
    return SchemaField(
        name=spec.name,
        # from_string never raises: an unrecognised type normalises to UNKNOWN, which
        # scores neutrally rather than rejecting a request over a type name this library
        # has not seen. A caller's dialect is not their mistake.
        data_type=DataType.from_string(spec.type),
        full_path=spec.path,
        parent_path=parent,
        description=spec.doc,
        source_metadata=source_metadata,
    )


# =============================================================================
# TRANSLATION -- response out
# =============================================================================


def _governance_id(match: MatchResult) -> str:
    """
    The governance id this field inherits. ALWAYS populated -- that is the contract.

    Absent attribute means the domain lane's `MatchResult.governance_id` has not landed
    yet, and the documented promotion (`the dictionary entry id IS the governance id`)
    gives the same answer. Present-but-empty is different: it means the field exists and
    was not filled, so a caller would be told a field's governance is the empty string.
    That is the NM-0005 failure wearing a value instead of a missing key, and it is
    refused.
    """
    value = getattr(match, "governance_id", _ABSENT)
    if value is _ABSENT:
        return str(match.dictionary_entry.id)
    if not value:
        raise drift(
            "MatchResult",
            "governance_id",
            "a matched field would be told its governance id is empty, which is a field "
            "silently inheriting nothing.",
        )
    return str(value)


def _governance_payload(match: MatchResult) -> dict[str, Any] | None:
    """
    The protection class, or None when the entry carries no code.

    None is rendered as an explicit `null`, never omitted. "This entry has no protection
    class" and "this response forgot to tell you" must not look the same to a client whose
    next step is applying a classification.

    Read by the attribute names in the shared contract rather than by importing the
    dataclass, so this file works both before and after the domain lane lands
    `domain/governance.py` -- and so a caller's own ProtectionClass-shaped object works
    too. Every attribute is required: a partially-populated class is refused rather than
    defaulted, because a defaulted `personalInformation: false` is a wrong answer to the
    one question the caller asked.
    """
    protection_class = getattr(match, "governance", _ABSENT)
    if protection_class is _ABSENT or protection_class is None:
        return None

    payload: dict[str, Any] = {}
    for key, attribute, cast in (
        ("code", "code", str),
        ("name", "name", str),
        ("classification", "classification", str),
        ("personalInformation", "personal_information", bool),
        ("directIdentifier", "direct_identifier", bool),
    ):
        value = getattr(protection_class, attribute, _ABSENT)
        if value is _ABSENT:
            raise drift(
                type(protection_class).__name__,
                attribute,
                f"the {key!r} member of the governance payload cannot be emitted and the "
                f"caller would apply an incomplete classification.",
            )
        payload[key] = cast(value)

    # LAST, and by `getattr` rather than through the loop above.
    #
    # Last because the five keys before it are a shape a Java client has already generated
    # against, and appending is the only edit to a dict literal that is additive on the
    # wire.
    #
    # `getattr(..., None)` because null is a DECLARED value here, not a missing one: five
    # of the nine classes in `examples/governance/protection_classes.json` set
    # `"enhancement": null`, so putting it through the `drift()` path above would make the
    # repository's own example pack a 500. It is carried at all because the caller reading
    # this object is deciding how to protect a field, and this is the only member that says
    # what to DO -- it was resolved on every MatchResult and then dropped at the wire.
    enhancement = getattr(protection_class, "enhancement", None)
    payload["enhancement"] = None if enhancement is None else str(enhancement)
    return payload


def _vocabulary_payload(matcher: object) -> dict[str, Any]:
    """
    What the response's `governance` nulls MEAN, carried in the response that has them.

    `open_classification` was reachable nowhere over HTTP -- not on a route, not in
    `/openapi.json`. So a Java client receiving `"governance": null` could not tell which
    tier that field sits at without opening the vocabulary JSON, which is a file on the
    server that the caller's pipeline may never have seen. A governance artifact that
    requires a second source of truth to read is not one.

    Emitted on the RESPONSE rather than from a `/vocabulary` route on purpose. The body
    gets pasted into tickets and diffed; the interpretation has to travel with it, and a
    separate endpoint is a thing the reader of that ticket has to know to call.

    Read off the live matcher by the same private-attribute coupling as `_scoring_weights`,
    and with the same posture: falling back rather than refusing. The fallback is
    `OPEN_CLASSIFICATION`, the domain's sentinel, which exists precisely to say "nothing is
    configured" in a word no real taxonomy uses -- so an unreadable vocabulary reports
    itself as unconfigured instead of inventing a plausible tier.
    """
    vocabulary = getattr(matcher, _MATCHER_GOVERNANCE_ATTR, None)

    open_classification = OPEN_CLASSIFICATION
    resolve = getattr(vocabulary, "classification_for", None)
    if callable(resolve):
        # The documented accessor, given the value a field with no code carries. Asking the
        # vocabulary is what keeps this in step with the tier `MatchResult` resolves
        # through, instead of reading a private attribute that means the same thing today.
        open_classification = str(resolve(None))

    declared = getattr(vocabulary, "tiers_most_open_first", ())
    return {
        "openClassification": open_classification,
        "tiersMostOpenFirst": [str(tier) for tier in declared],
    }


def _scoring_weights(matcher: object) -> dict[str, float]:
    """
    The weights that actually produced this run's confidences, read off the live matcher.

    Falls back to the shipped defaults when the matcher does not expose its config, rather
    than refusing: weights that reproduce every emitted confidence ARE the weights that
    produced it, whichever object they came from, and `_verify_reproducible` is what
    decides. That keeps the guarantee resting on arithmetic anyone can check instead of on
    a private attribute name holding still.
    """
    config = getattr(matcher, _MATCHER_CONFIG_ATTR, None)
    if config is None:
        # Deferred: this pulls the whole matching stack, and importing it at module scope
        # would make `import nexus_matcher.presentation.api.matching` pay for it.
        from nexus_matcher.application.use_cases.match_schema import MatchingConfig

        config = MatchingConfig()

    weights: dict[str, float] = {}
    for key, _score_attr, weight_attr in _SCORE_COMPONENTS:
        value = getattr(config, weight_attr, None)
        if value is None:
            raise drift(
                type(config).__name__,
                weight_attr,
                f"the {key!r} weight cannot be emitted and the confidence cannot be "
                f"reproduced from the response.",
            )
        weights[key] = round(float(value), _PRECISION)
    return weights


def _results_per_field(matcher: object) -> int:
    """The server's cap on `top_k`, read off the live matcher's config."""
    config = getattr(matcher, _MATCHER_CONFIG_ATTR, None)
    value = getattr(config, "results_per_field", None)
    if isinstance(value, int) and value >= 1:
        return value

    from nexus_matcher.application.use_cases.match_schema import MatchingConfig

    return MatchingConfig().results_per_field


def _absolute_score(match: MatchResult) -> float | None:
    """
    The raw dense-retrieval score for one candidate, rounded for the wire.

    ONE reader, so the top-level `absoluteScore` and `explain.absoluteCosine` cannot come
    back as two different numbers for the same candidate -- a response that reported the
    same quantity twice with a disagreement in the sixth decimal would be a governance
    artifact arguing with itself.

    None stays None. It means the dense arm never returned this candidate, which is not
    zero: zero is a similarity the retriever measured, None is one it never took.
    """
    value = getattr(match.score_breakdown, "absolute_cosine", None)
    return None if value is None else round(float(value), _PRECISION)


def _explain_payload(
    match: MatchResult, weights: dict[str, float], absolute_score: float | None
) -> dict[str, Any]:
    """
    Everything needed to recompute the confidence from the response alone.

    `absoluteCosine` is the SAME number as the candidate's `absoluteScore` and is passed
    in rather than re-read, so the two cannot drift. It is kept here although it is now
    duplicated: a client already generated against `explain` reads it at this path, and
    removing a published key to tidy up a duplication is a breaking change bought with
    nothing.
    """
    breakdown = match.score_breakdown
    scores: dict[str, float] = {}
    for key, score_attr, _weight_attr in _SCORE_COMPONENTS:
        value = getattr(breakdown, score_attr, None)
        if value is None:
            raise drift(
                type(breakdown).__name__,
                score_attr,
                f"the {key!r} component cannot be emitted and the confidence cannot be "
                f"reproduced from the response.",
            )
        scores[key] = round(float(value), _PRECISION)

    return {
        "scores": scores,
        "weights": dict(weights),
        "absoluteCosine": absolute_score,
    }


def _verify_reproducible(confidence: float, explain: dict[str, Any]) -> None:
    """
    Do the auditor's arithmetic before handing them the answer.

    The whole promise of `explain` is that `sum(scores[k] * weights[k])`, clamped, comes
    back to `confidence`. Checked here on the EMITTED numbers, so a response that cannot
    keep it is never sent. It closes a class of drift no name matching can: a component
    paired with the wrong weight, a sixth weighted signal this file knows nothing about,
    or a matcher whose weights could not be read and whose confidences the shipped
    defaults do not explain. All three produce a response that is self-consistently wrong.

    Only requests that ASKED for `explain` are checked. Without it the response makes no
    arithmetic claim, so a scoring change in another lane degrades one optional field
    rather than taking matching down.

    Clamped exactly as `_weighted_confidence` clamps, so a weight set summing above 1.0 is
    not reported as an arithmetic failure that never happened.

    ONLY SCORED CANDIDATES REACH HERE, and the exclusion is made at the caller by
    PROVENANCE rather than by loosening anything in this function -- see
    `_candidate_payload`. A candidate a human decided never went through scoring, so it
    cannot satisfy an identity describing an arithmetic nobody performed, and refusing the
    whole response over it reported a scoring drift that had not happened while taking
    matching down for every other field in the batch. That is a narrowing of WHAT is
    checked, not of HOW: everything the scorer produced is still checked exactly as
    before, which is what `TestTheGuardKeepsItsTeeth` holds this to.
    """
    scores = explain["scores"]
    weights = explain["weights"]
    total = sum(scores[key] * weight for key, weight in weights.items())
    recomputed = round(min(max(total, 0.0), 1.0), _PRECISION)
    if abs(recomputed - confidence) > _REPRODUCTION_TOLERANCE:
        raise drift(
            "the matcher's scoring",
            "a reproducible confidence",
            f"the emitted components and weights give {recomputed!r} while the emitted "
            f"confidence is {confidence!r}.",
        )


def _candidate_payload(
    match: MatchResult,
    weights: dict[str, float] | None,
) -> dict[str, Any]:
    """
    One candidate, with its keys in the contract's order.

    The dict literal IS the wire order -- `DeterministicJSONResponse` does not sort -- so
    the order below is load-bearing and must match `MatchCandidateView`.

    `absoluteScore` is APPENDED to the literal, after `decision`. It is the raw dense
    score, and promoting it out of the optional `explain` block is what gives a client one
    number it may compare against a constant across fields: `confidence` cannot serve,
    because it is min-max normalised per field and its rank-1 value has a structural floor
    (0.63 shipped) that sits above the review threshold, so a rank 1 that matches nothing
    still scores well above it. `explain` is still added afterwards, so this key holds a
    stable position whether or not `explain` was requested.

    `sourceMetadata` is appended after it, in the same way and for the same reason, and it
    is the last key of the enrichment surface rather than the last key of the object:
    `explain` still goes last. It carries the entry's pass-through plane -- the
    deployment's own enrichment columns, which had reached the index and stopped there, so
    a deployment could send its glossary through this service and get back none of what its
    own pipeline needed. Unconditional: it does not wait for `explain`, because a column a
    deployment declared is not diagnostic output.

    ## `provenance`, and why it is a member rather than a number

    Appended after `sourceMetadata` and still before `explain`, by the same rule again.
    Unconditional and never null: a member that appears only when something unusual
    happened is a member a client learns about in production.

    It exists because the alternative was tried and was WRONG. A bypassed candidate carried
    `confidence` 1.0 and `decision` AUTO_APPROVE, and the library asserted that pair was
    outside the scorer's range. It is not -- the five default weights sum to exactly 1.0,
    so ordinary retrieval reaches 1.0 whenever all five signals are maximal, and a client
    reading those two members then cannot tell a human's answer from a very good match. A
    VALUE cannot collide with a score, and this repository's posture is that a number's
    meaning must never have to be inferred. See `MatchProvenance`.

    ## `explain` IS NOT ATTACHED TO A CANDIDATE THAT WAS NEVER SCORED

    The block promises `sum(scores * weights) == confidence`, and a candidate a human
    decided has honest 0.0 components against an honest 1.0 confidence. Three options, and
    only one is honest: emit the block and let it contradict itself; emit components of 1.0
    so the arithmetic closes, publishing five measurements nobody took into fields the
    scoring contract declares comparable ACROSS fields; or leave it out. It is left out --
    `ExplainView` is already `| None` and the key is already conditional on the request, so
    absence is a shape every generated client can read.

    THE PREVIOUS BEHAVIOUR WAS A 500 FOR THE WHOLE REQUEST. `_verify_reproducible` refused,
    correctly in principle, and took every other field in the batch down with it while
    telling the operator the library had drifted. The check is unchanged for everything the
    scorer produced; what changed is that a candidate which did not come from scoring is no
    longer offered to it as evidence of a scoring drift.
    """
    entry = match.dictionary_entry
    confidence = round(float(match.final_confidence), _PRECISION)
    decision = match.decision
    absolute_score = _absolute_score(match)
    provenance = provenance_of(match)
    payload: dict[str, Any] = {
        "rank": int(match.rank),
        "governanceId": _governance_id(match),
        "businessName": entry.business_name,
        "definition": entry.definition,
        "domain": entry.domain,
        "governance": _governance_payload(match),
        "confidence": confidence,
        "decision": getattr(decision, "value", str(decision)),
        "absoluteScore": absolute_score,
        "sourceMetadata": _source_metadata_payload(entry),
        "provenance": provenance.value,
    }
    if weights is not None and provenance is MatchProvenance.RETRIEVAL:
        explain = _explain_payload(match, weights, absolute_score)
        _verify_reproducible(confidence, explain)
        payload["explain"] = explain
    return payload


def _absolute_score_floor(matcher: object) -> float | None:
    """
    The absolute-score floor this server applies, or None when none is configured.

    None is the shipped default and it is not a stub: a floor is a statement about a score
    distribution, and the distribution belongs to a dictionary this library has never
    seen. Read through the PUBLIC property rather than the private config, unlike the
    weights and the vocabulary above, because that property was added in the same change
    as this reader and there is no older matcher to be compatible with. A matcher that
    does not have it -- a caller's own object, a test double -- reports no floor, which is
    the same answer as "not configured" and degrades to the documented default rather than
    taking matching down.
    """
    floor = getattr(matcher, "absolute_score_floor", None)
    return None if floor is None else float(floor)


def _scoring_payload(
    matcher: object,
    floor: float | None,
    projected: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    What every number in this response MEANS, travelling with the response that has them.

    Same argument as `_vocabulary_payload`: the body is a governance artifact that gets
    pasted into a ticket and diffed, so an artifact whose numbers can only be interpreted
    by reading this library's source is not one. It extends that block's pattern rather
    than inventing a second one.

    `confidenceFloor` IS VERIFIED AGAINST THE RESPONSE IT SHIPS WITH. The derivation
    `semantic_weight * fusion_alpha` is a bound with preconditions -- no reranker, and at
    least two distinct dense scores so min-max maps the top candidate to 1.0 rather than
    to 0.0. The second is easy to violate by ordinary means (a one-entry dictionary,
    `dense_top_k=1`, a perfect tie), and when it is violated the real confidences sit far
    below while the config still reports 0.63. Publishing that number would tell a client
    to set every threshold above a floor its own fields are underneath, which is NM-0027's
    failure re-shipped on the wire. So the bound is checked against the rank-1 confidences
    actually emitted, and reported as null if any of them is below it. A self-verifying
    claim cannot be wrong about its own response. `NexusMatcher._session_confidence_floor`
    does the same thing for a library caller, for the same reason.
    """
    declared = getattr(matcher, "minimum_achievable_confidence", None)
    confidence_floor: float | None = None
    if declared is not None:
        tops = [candidates[0]["confidence"] for candidates in projected.values() if candidates]
        if tops and min(tops) >= float(declared):
            confidence_floor = round(float(declared), _PRECISION)

    config = getattr(matcher, _MATCHER_CONFIG_ATTR, None)
    alias_count = getattr(config, "dictionary_alias_count", 0)

    return {
        "confidenceFloor": confidence_floor,
        "absoluteScoreFloor": None if floor is None else round(floor, _PRECISION),
        "absoluteScoreMetric": str(getattr(matcher, "absolute_score_metric", "unknown")),
        "absoluteScorePooledOverAliases": bool(alias_count),
        # Derived from the table rather than typed a second time: a number whose declared
        # scope changes must not keep a stale entry in a hand-written list saying a client
        # may still compare it against a constant.
        "thresholdableAcrossFields": [
            key
            for key, scope in _COMPARABILITY.items()
            if _COMPARABILITY_SCOPES.index(scope) >= _COMPARABILITY_SCOPES.index(_ACROSS_FIELDS)
        ],
        "comparabilityScopesNarrowestFirst": list(_COMPARABILITY_SCOPES),
        "comparability": dict(_COMPARABILITY),
    }


# =============================================================================
# CONTRAST -- WHY THE RUNNER-UP LOST
# =============================================================================


def _contrast_signals(weights: dict[str, float]) -> tuple[SignalSpec, ...]:
    """
    The signal table the domain pass works from, built from the ONE pairing table.

    `_SCORE_COMPONENTS` is the single visible place a component is paired with its
    weight, and the contrast has to use the same pairing or it will attribute a margin to
    the wrong signal -- a response that is self-consistently wrong, which is the worst
    failure an audit surface has.
    """
    return tuple(
        SignalSpec(name=key, score_attr=score_attr, weight=weights[key])
        for key, score_attr, _weight_attr in _SCORE_COMPONENTS
    )


def _contrast_comparability() -> dict[str, Any]:
    """
    The scale contract for the contrast's own numbers, DERIVED from `_COMPARABILITY`.

    A difference is exactly as comparable as its operands: the gap between two
    confidences carries `confidence`'s scope, and a signal's delta carries that signal's.
    Derived rather than typed a second time, so a number whose declared scope changes
    cannot leave a stale entry here saying a client may still compare it across fields.
    """
    return {
        "confidenceGap": _COMPARABILITY["confidence"],
        "signals": {
            key: _COMPARABILITY[f"explain.scores.{key}"] for key, _a, _w in _SCORE_COMPONENTS
        },
    }


def _verify_contrast_reproducible(contrast: Any) -> None:
    """
    Do the reviewer's subtraction before handing them the contrast.

    The promise is that the per-signal weighted differences sum to the gap between the two
    published confidences. Checked on the EMITTED numbers, exactly as `_verify_reproducible`
    checks a single candidate, and for the same reason: an explanation whose arithmetic
    does not close is worse than none, because it is the one that gets used as evidence.
    It closes the same class of drift -- a sixth weighted signal this file knows nothing
    about, or weights that do not explain these confidences -- on a request that asked for
    the contrast without asking for `explain`, where nothing else would check.

    THE CLAMP IS THE ONE CARVE-OUT. `_weighted_confidence` clamps to [0, 1], so a
    deployment whose weights sum above 1.0 can produce two candidates that both clamp to
    1.0: the gap is then legitimately 0 while the weighted differences are not, and
    refusing would turn a tuned-but-working configuration into a 500. A confidence sitting
    exactly on a bound is the only case where the two routes are allowed to disagree.
    """
    if contrast.top_confidence in (0.0, 1.0) or contrast.runner_up_confidence in (0.0, 1.0):
        return
    if abs(contrast.signal_gap - contrast.confidence_gap) > _REPRODUCTION_TOLERANCE:
        raise drift(
            "the matcher's scoring",
            "a reproducible contrast",
            f"the emitted per-signal differences sum to {contrast.signal_gap!r} while the "
            f"emitted confidences differ by {contrast.confidence_gap!r}.",
        )


def _contrast_payload(contrast: Any) -> dict[str, Any]:
    """One contrast, with its keys in the contract's order -- the dict literal IS the wire
    order, since `DeterministicJSONResponse` does not sort."""
    return {
        "topGovernanceId": contrast.top_governance_id,
        "runnerUpGovernanceId": contrast.runner_up_governance_id,
        "topConfidence": contrast.top_confidence,
        "runnerUpConfidence": contrast.runner_up_confidence,
        "confidenceGap": contrast.confidence_gap,
        "signalGap": contrast.signal_gap,
        "separation": contrast.separation.value,
        "largestDifference": contrast.largest_difference,
        "decidingSignals": list(contrast.deciding_signals),
        "governanceDiffers": contrast.governance_differs,
        "domainDiffers": contrast.domain_differs,
        "signals": [
            {
                "signal": difference.signal,
                "topScore": difference.top_score,
                "runnerUpScore": difference.runner_up_score,
                "delta": difference.delta,
                "weight": difference.weight,
                "weightedDelta": difference.weighted_delta,
                "separating": difference.separating,
                "deciding": difference.deciding,
            }
            for difference in contrast.differences
        ],
    }


def _contrast_block(
    specs: list[FieldSpec],
    matched: dict[str, tuple[MatchResult, ...]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """
    The contrast for every field, keyed like `results` and with an explicit null where
    there is no runner-up.

    EVERY INPUT PATH IS PRESENT. A field with one candidate has nothing it lost to, and
    that must not look like a field this pass skipped -- the same argument the response
    makes for `governance` being an explicit null and for a matchless field getting `[]`.

    Read from the FULL match list rather than from the `top_k` slice, which is the reading
    `fieldDecisions` already takes: the runner-up is a property of what the matcher found,
    not of how many candidates the caller asked to see. A caller who asks for one
    candidate and a contrast is told what the one they cannot see was.
    """
    signals = _contrast_signals(weights)
    contrasts: dict[str, Any] = {}
    for spec in specs:
        matches = matched.get(spec.path, ())
        try:
            contrast = contrast_top_two(matches, signals, _PRECISION)
        except ValueError as exc:
            raise drift(
                "ScoreBreakdown",
                "a readable component",
                f"a contrast could not be computed for {spec.path!r}: {exc}",
            ) from exc
        if contrast is None:
            contrasts[spec.path] = None
            continue
        _verify_contrast_reproducible(contrast)
        contrasts[spec.path] = _contrast_payload(contrast)

    return {
        "resolution": 10.0**-_PRECISION,
        "comparability": _contrast_comparability(),
        "fields": contrasts,
    }


# =============================================================================
# CONSISTENCY -- THE SAME CONCEPT, ANSWERED TWICE
# =============================================================================


def _consistency_answers(
    projected: dict[str, list[dict[str, Any]]],
    decisions: dict[str, str],
) -> dict[str, str | None]:
    """
    What each column ANSWERED, which is not the same as what it matched.

    A field whose verdict is NO_MATCH inherits nothing -- its candidates are evidence for
    a reviewer, not a classification -- so it has no answer to contribute. Feeding its
    rank-1 id in anyway would manufacture a disagreement between a column that answered
    and one that declined to, which is exactly the noise a report like this dies of.
    `fieldDecisions` is the field-level authority and this reads it rather than
    re-deriving a second opinion beside it.
    """
    answers: dict[str, str | None] = {}
    for path, candidates in projected.items():
        if not candidates or decisions.get(path) == FieldDecision.NO_MATCH.value:
            answers[path] = None
        else:
            answers[path] = str(candidates[0]["governanceId"])
    return answers


def _consistency_block(
    fields: list[SchemaField],
    projected: dict[str, list[dict[str, Any]]],
    decisions: dict[str, str],
    policy: GroupingPolicy,
) -> dict[str, Any]:
    """
    Which columns look like one concept, and whether they were given one answer.

    REPORTS, NEVER OVERRIDES. Nothing here touches `projected` or `decisions`; both are
    read. Promoting a group's majority is a decision that can be wrong in a new way --
    it can move a correct answer to an incorrect one -- while surfacing a disagreement
    cannot, and the measurement that would justify the former does not exist yet. What
    does exist is a measurement of the GROUPING, which is its prerequisite -- and it came
    back negative. See `tests/unit/domain/test_review_evidence_grouping.py`: on a
    repeated-leaf schema the loose key scores pair-precision 0.0233 and every group it
    emits is a collision, and no policy in the published space does better while reporting
    anything at all. The default `qualifier_segments` is 1 for that reason: on the
    generated corpus it emits nothing, which is the honest output for a grouping nobody has
    shown to work on that shape.

    The policy is published beside its findings because a finding cannot be judged without
    the rule that produced it: a group of six that disagree means one thing under a leaf-
    only key and another under a key that also matched their parent.
    """
    groups = group_by_concept(fields, policy)
    answers = _consistency_answers(projected, decisions)

    # Cheap insurance of the same shape as the conservation law: a group naming a column
    # this response does not carry would be a report about a field the caller cannot look
    # up. It cannot happen today -- both sides are built from the same field list -- and
    # that is exactly the kind of claim this repository has shipped as coverage twice, so
    # it is checked rather than asserted in a comment.
    for group in groups:
        for path in group.paths:
            if path not in projected:
                raise ConservationViolationError(
                    message=(
                        f"the consistency pass grouped {path!r}, which is not a field of "
                        f"this response, so the report would name a column the caller "
                        f"cannot look up."
                    ),
                    details={"path": path},
                )

    findings = assess_consistency(groups, answers)
    return {
        "grouping": {
            "qualifierSegments": policy.qualifier_segments,
            "includeDataType": policy.include_data_type,
            "orderSensitive": policy.order_sensitive,
            "minGroupSize": policy.min_group_size,
        },
        "groupsFound": len(findings),
        "fieldsGrouped": sum(len(finding.paths) for finding in findings),
        # Compared against the enum member, not against the string it renders as. The
        # spelling is now a CLOSED published component (`Agreement`), so a rename would be
        # a wire change caught by the schema gates -- but a literal here would keep
        # counting zero disagreements in silence while the wire said something else.
        "groupsDisagreeing": sum(
            1 for finding in findings if finding.agreement is Agreement.DISAGREE
        ),
        # A constant, and published rather than left implicit: a consumer reading this
        # block is entitled to a machine-readable statement that it changed nothing.
        "promotionApplied": False,
        "groups": [
            {
                "concept": finding.concept,
                "fields": list(finding.paths),
                "answers": dict(finding.answers),
                "distinctAnswers": finding.distinct_answers,
                "agreement": finding.agreement.value,
                "majorityGovernanceId": finding.majority_answer,
                "majorityCount": finding.majority_count,
            }
            for finding in findings
        ],
    }


def _project_results(
    specs: list[FieldSpec],
    fields: list[SchemaField],
    matched: dict[str, tuple[MatchResult, ...]],
    top_k: int,
    weights: dict[str, float] | None,
    floor: float | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """
    THE CONSERVATION LAW, checked three ways, before anything is sent.

    Each check is the negation of a way a column has actually been lost here:

      COUNT     as many result entries as fields, so nothing evaporated
      IDENTITY  the results at position i were computed for fields[i]
      ADDRESS   the matcher's key is the caller's own path, so nothing became unreachable

    IDENTITY compares by `is` rather than `==`, and the honest note about that is that
    NOTHING HERE PROVES THE DIFFERENCE MATTERS: mutating it to `==` leaves every test
    green. SchemaField is a frozen dataclass, so two distinct fields with identical
    contents compare equal -- but this endpoint refuses duplicate paths before it ever
    gets here, and two fields with different paths differ in `full_path`, so the case
    where the two operators disagree is unreachable from the wire. `is` stays because it
    is strictly stronger and costs nothing, not because a test defends it. Written down
    rather than left as a confident comment, because a claim no mutation can falsify is
    exactly the kind this repository has shipped as coverage twice.

    IDENTITY and ADDRESS are genuinely independent oracles rather than one restated: the
    first reads `MatchResult.schema_field`, the second reads the dict key the matcher
    chose. An error would have to be present in both to survive, which is the H-004 shape
    these are written to avoid.

    A field with no candidates gets `[]`. Dropping it would be the exact defect: the
    caller's map would be short one key and nothing would say so.

    THE FIELD DECISION IS BUILT IN THE SAME PASS and returned alongside, so the two maps
    carry the same keys in the same order by construction. The equality check afterwards
    is therefore not proof today -- it is an identity, and this file says elsewhere what
    an identity is worth. It stays as a guard against the edit that filters one map and
    not the other, which is exactly how a key goes missing from a response that still
    passes a count check: `fieldDecisions` short one key is a column with no verdict, and
    a client defaulting an absent verdict to "nothing matched" would silently unclassify
    it while a client defaulting the other way would silently classify it.
    """
    if len(matched) != len(fields):
        raise ConservationViolationError(
            message=(
                f"{len(fields)} fields were sent and {len(matched)} came back. A field "
                f"missing from the response inherits no governance and nothing would have "
                f"said so, so the response was refused instead of returned short."
            ),
            details={"fields_in": len(fields), "results_out": len(matched)},
        )

    projected: dict[str, list[dict[str, Any]]] = {}
    decisions: dict[str, str] = {}
    for (key, matches), spec, field in zip(matched.items(), specs, fields, strict=True):
        if key != spec.path:
            raise ConservationViolationError(
                message=(
                    f"result {key!r} is not addressable under the path the caller sent "
                    f"({spec.path!r}), so the caller could not look up their own field."
                ),
                details={"expected_path": spec.path, "actual_key": key},
            )
        for match in matches:
            if match.schema_field is not field:
                raise ConservationViolationError(
                    message=(
                        f"the results under {spec.path!r} were computed for a different "
                        f"field ({match.schema_field.full_path!r}), so this field would "
                        f"inherit another column's governance."
                    ),
                    details={"path": spec.path, "computed_for": match.schema_field.full_path},
                )
        projected[spec.path] = [_candidate_payload(m, weights) for m in matches[:top_k]]
        # Derived from the FULL match list rather than from `matches[:top_k]`, and the
        # honest note about that is that NOTHING HERE PROVES THE DIFFERENCE MATTERS:
        # `MatchRequest.top_k` is `ge=1`, `derive_field_decision` reads rank 1 only, so
        # the two expressions are equal for every request this endpoint accepts --
        # mutating it to the truncated list leaves every test green. It stays because the
        # verdict is a property of what the matcher FOUND rather than of how many
        # candidates the caller asked to see, and that reading does not depend on the
        # request contract keeping its lower bound. Written down rather than left as a
        # confident comment, because a claim no mutation can falsify is exactly the kind
        # this repository has shipped as coverage twice.
        decisions[spec.path] = derive_field_decision(matches, floor).value

    if list(decisions) != list(projected):
        raise ConservationViolationError(
            message=(
                "the field decisions and the results disagree about which fields this "
                "response covers, so at least one column would come back with candidates "
                "and no verdict, or with a verdict and no candidates."
            ),
            details={"results": len(projected), "field_decisions": len(decisions)},
        )

    return projected, decisions


# =============================================================================
# THE SERVICE
# =============================================================================


class MatchService:
    """Everything both match routes share; the routes differ only in their field cap."""

    def __init__(
        self,
        handle: MatcherHandle,
        limits: MatchServiceLimits,
        pool: BoundedWorkPool,
    ) -> None:
        self._handle = handle
        self._limits = limits
        self._pool = pool

    async def match(self, request: MatchRequest, max_fields: int) -> dict[str, Any]:
        """Validate, match under the deadline, and project -- in that order."""
        specs = request.field_specs

        if len(specs) > max_fields:
            raise RequestTooLargeError(
                message=(
                    f"{len(specs)} fields in one request exceeds this server's limit of "
                    f"{max_fields}. Send them in chunks of at most {max_fields}."
                ),
                details={"fields": len(specs), "limit": max_fields},
            )

        # The response is a map keyed by the caller's own path, and a map cannot hold two
        # entries for one key. Silently collapsing them is NM-0005 exactly -- one column
        # inherits nothing and the caller sees a shorter map. Refusing is the only answer
        # that keeps the promise, and it names the offending paths.
        repeated = sorted(p for p, n in Counter(s.path for s in specs).items() if n > 1)
        if repeated:
            raise MalformedRequestError(
                message=(
                    f"Every field needs a distinct `path`, because the response is keyed "
                    f"by it: {repeated!r} appear more than once. Two fields under one key "
                    f"would leave one of them with no governance and no error."
                ),
                details={"duplicate_paths": repeated},
            )

        matcher = self._handle.require()

        cap = _results_per_field(matcher)
        if request.top_k > cap:
            raise MalformedRequestError(
                message=(
                    f"top_k={request.top_k} exceeds this server's configured "
                    f"results_per_field={cap}, so it could only ever return {cap} "
                    f"candidates. Ask for at most {cap}, or raise results_per_field on "
                    f"the server."
                ),
                details={"top_k": request.top_k, "results_per_field": cap},
            )

        fields = [_to_schema_field(spec) for spec in specs]
        matched = await run_bounded(
            self._pool,
            lambda: _invoke_matcher(matcher, fields, request.signals),
            self._limits.deadline_seconds,
        )

        # Read ONCE when either surface needs them, and held in two separately-typed
        # names. The two uses are genuinely different -- `explain` publishes the weights
        # on every candidate, while the contrast only needs them to subtract with, so a
        # caller can have the comparison without the breakdown -- and separate names mean
        # each block's emission condition is "was this asked for" rather than "did the
        # weights happen to be readable". A block that silently vanished because a shared
        # variable was None is a request answered short with nothing saying so.
        explain_weights: dict[str, float] | None = None
        contrast_weights: dict[str, float] | None = None
        if request.explain or request.contrast:
            read = _scoring_weights(matcher)
            explain_weights = read if request.explain else None
            contrast_weights = read if request.contrast else None

        floor = _absolute_score_floor(matcher)
        projected, decisions = _project_results(
            specs, fields, matched, request.top_k, explain_weights, floor
        )
        # `results` first: it was the whole body, and the key order IS the wire contract.
        # `vocabulary` is what makes a `governance` of null readable without the caller
        # holding a copy of the server's vocabulary file. The two below are APPENDED for
        # the same reason: `fieldDecisions` is the one verdict per column a consumer
        # writes down, and `scoring` is what stops that verdict and the numbers beside it
        # from needing this library's source to interpret.
        body: dict[str, Any] = {
            "results": projected,
            "vocabulary": _vocabulary_payload(matcher),
            "fieldDecisions": decisions,
            "scoring": _scoring_payload(matcher, floor, projected),
        }
        # APPENDED LAST, and only when asked for. A request that sets neither flag gets
        # the four keys above and nothing else, byte for byte -- which is asserted on the
        # bytes rather than on parsed JSON, because a re-ordering survives a parse.
        if contrast_weights is not None:
            body["contrast"] = _contrast_block(specs, matched, contrast_weights)
        if request.consistency:
            body["consistency"] = _consistency_block(
                fields,
                projected,
                decisions,
                GroupingPolicy(qualifier_segments=request.consistency_qualifier_segments),
            )
        return body


def _invoke_matcher(
    matcher: object,
    fields: list[SchemaField],
    signals: dict[str, Any] | None = None,
) -> dict[str, tuple[MatchResult, ...]]:
    """
    Call the matcher on the worker thread, converting any failure into a named 5xx.

    Letting the raw exception escape also produces a 500, but an anonymous one, by a path
    that depends on middleware ordering. The adopter's fallback keys on the status code
    and their operator keys on the message; both deserve to be deterministic.

    THE CALL IS UNCHANGED WHEN NO SIGNALS ARE SENT, and that is deliberate rather than
    tidy. `matcher` is duck-typed -- this module reaches it through `_MATCH_FIELDS_ATTR`
    and raises `ContractDriftError` when the attribute is missing, precisely because it is
    not this layer's object. Passing a new keyword to it unconditionally would break every
    collaborator that implements today's signature, for requests that had nothing to say.
    So the extended call happens only when a caller has actually opted in, which is also
    the guarantee the channel is required to keep.
    """
    match_fields = getattr(matcher, _MATCH_FIELDS_ATTR, None)
    if match_fields is None:
        raise drift(
            type(matcher).__name__,
            _MATCH_FIELDS_ATTR,
            "there is no way to match a list of fields and this endpoint cannot serve "
            "anything at all.",
        )

    from nexus_matcher.shared.exceptions import NexusMatcherError

    try:
        if signals:
            return match_fields(fields, signals=signals)
        return match_fields(fields)
    except NexusMatcherError:
        # Already carries its own code and status; re-wrapping would hide a 503 behind a
        # 500 and send the caller's fallback down the wrong branch.
        raise
    except Exception as exc:
        raise MatchFailedError(
            message=(
                f"Matching failed: {type(exc).__name__}: {exc}. No field was classified; "
                f"treat this request as unanswered."
            ),
            details={"cause": type(exc).__name__},
            cause=exc,
        ) from exc


# =============================================================================
# ROUTER
# =============================================================================

# Every way these routes are allowed to fail, WITH the body each one sends.
#
# This table used to supply descriptions only, so `/openapi.json` described zero error
# bodies on either match route while `ErrorResponse` sat unused elsewhere in the package
# -- a generated Java client got a typed 200 and a `Map` for everything else.
#
# 500 is here and was not, although `errors.py` documents it as one of the five failure
# modes and three exception classes return it (MatchFailedError,
# ConservationViolationError, ContractDriftError). It is the status a client is least able
# to guess, so it is the one worst to omit.
#
# The 422 entry OVERRIDES FastAPI's own HTTPValidationError, deliberately: `app.py`
# installs `validation_exception_handler`, so this service never sends `{"detail": [...]}`
# and publishing that model would be an actively false schema. The defect was only ever
# that nothing replaced it.
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    413: {
        "model": ErrorResponse,
        "description": "Too many fields, or a body over this server's byte cap; chunk it.",
    },
    422: {
        "model": ErrorResponse,
        "description": "Malformed request. `details.violations` names the offending field.",
    },
    500: {
        "model": ErrorResponse,
        "description": (
            "The matcher failed, or this layer refused a response it could not trust. "
            "No field was classified; treat the request as unanswered."
        ),
    },
    503: {
        "model": ErrorResponse,
        "description": "No dictionary loaded, or the server shed this request.",
    },
    504: {
        "model": ErrorResponse,
        "description": "The server-side deadline fired before matching finished.",
    },
}


def create_matching_router(service: MatchService, limits: MatchServiceLimits) -> APIRouter:
    """Mount both match routes onto one service."""
    router = APIRouter(prefix="/api/v1", tags=["Matching"])

    @router.post(
        "/match",
        status_code=status.HTTP_200_OK,
        response_class=DeterministicJSONResponse,
        # The handler returns the RESPONSE rather than a dict, which short-circuits
        # FastAPI's `serialize_response` and the `jsonable_encoder` walk inside it.
        # Measured on a 250-field explain payload: 28.6 ms of encoding against 3.4 ms of
        # rendering, the single largest item by internal time at 58,843 calls per request
        # -- all of it on the event loop, so it was `/health/live` latency too. Bodies are
        # byte-identical, verified across 12 shapes: the payload is already pure
        # dict/list/str/int/float/bool/None, so the walk provably changed nothing.
        #
        # Two behaviour deltas come with it, both deliberate. `jsonable_encoder` was a
        # lenient net (datetime to ISO, Decimal to float, set to list -- all of which
        # `json.dumps` refuses), so a future non-primitive leaf is now a loud 500 instead
        # of a silent coercion, which is the choice this module makes everywhere else.
        # And returning a Response skips the merge of dependency-set response headers:
        # inert today (there is not one `Depends(` in src/, and X-Request-ID is set in
        # middleware) but a future dependency setting a header would lose it silently.
        #
        # response_model is deliberately absent: the handler renders the body itself so
        # that `explain` can be ABSENT rather than null while `governance` is null rather
        # than absent. The schema is published through `responses` instead, and
        # test_match_endpoint validates real bodies against it so the two cannot drift.
        response_model=None,
        responses={200: {"model": MatchResponseView}, **_ERROR_RESPONSES},
        summary="Match schema fields to dictionary entries",
        description=(
            "One entry per input field, keyed by the caller's own `path`, in the order "
            "sent. Every input field appears exactly once, with an empty list when "
            "nothing matched it. `fieldDecisions` carries the single verdict per column "
            "-- including NO_MATCH, which the per-candidate `decision` cannot express -- "
            "and `scoring` states, per number, whether it may be compared within a field, "
            "across fields, or not at all."
        ),
    )
    async def match(request: MatchRequest) -> DeterministicJSONResponse:
        return DeterministicJSONResponse(await service.match(request, limits.max_fields))

    @router.post(
        "/match/batch",
        status_code=status.HTTP_200_OK,
        response_class=DeterministicJSONResponse,
        response_model=None,
        responses={200: {"model": MatchResponseView}, **_ERROR_RESPONSES},
        summary="Match a chunk of schema fields",
        description=(
            "Identical to /match, with a higher field cap for chunked clients. The two "
            "share one implementation so their semantics cannot drift apart."
        ),
    )
    async def match_batch(request: MatchRequest) -> DeterministicJSONResponse:
        return DeterministicJSONResponse(await service.match(request, limits.max_batch_fields))

    return router
