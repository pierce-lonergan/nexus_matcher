"""
nexus_matcher.presentation.api.lookup | Layer: PRESENTATION
GET /api/v1/lookup/{id} and POST /api/v1/lookup -- resolving ids the caller already knows.

## Relationships
# DEPENDS_ON → domain/ports/entry_lookup :: EntryLookup, the port this plane is written to
# DEPENDS_ON → presentation/api/matching :: the candidate renderer and the response class,
#              reused rather than restated so the two enrichment surfaces cannot drift
# DEPENDS_ON → presentation/api/errors :: every failure mode
# DEPENDS_ON → presentation/api/limits :: the batch cap this shares with /match/batch
# DEPENDS_ON → domain/models/entities :: DictionaryEntry out
# USED_BY    → presentation/api/app :: mounted by create_app

## Why a lookup plane exists at all

This library had retrieval and no lookup, so a consumer holding a dictionary id it already
knows -- an operator-tagged column, a curated registry of audit columns, an id copied out
of a previous response -- had one way to turn it into an entry: send the field through
fuzzy matching and hope the right entry came back first.

That is wrong twice. It is expensive (an encoder call, a corpus scan and a five-signal
scoring pass to answer a question a dict already answers), and it is *less accurate than
doing nothing clever*: matching can rank the wrong entry first, and a caller who NAMED the
entry has no reason to accept that risk. A lookup hit is exact by construction. It is the
only surface in this service whose precision does not depend on a model.

So this is a separate plane, not a mode of matching:

  * no scoring, no ranking, no confidence, no decision -- a hit is exact or it is absent;
  * no encoder is touched and no vector is compared, so it holds none of the matching
    service's admission permits (see "Admission control" below);
  * it returns the same ENRICHMENT surface a match candidate carries, so a caller can feed
    a looked-up entry into the same code path that consumes a matched one.

## Found and not-found are both answers, and both are in the body

A partial list is refused as a design. Returning only the ids that resolved and leaving the
caller to diff that against what they asked for is the same defect as NM-0005 one surface
over: the caller's own key vanishes from the response, nothing says so, and the only
symptom is a count nobody has reason to check. So every requested id appears in `results`,
in the order it was sent, mapped either to an entry or to an explicit `null`.

`missing` is derived from that map in one pass -- never assembled separately -- so the two
cannot disagree. It exists because "did my registry resolve?" is the question an operator
actually asks, and answering it from the map alone means walking up to 250 entries in the
client.

## Not-found is 200, not 404

A `GET` for an id this dictionary does not carry answers **200** with `null` under that id,
not 404.

404 on this service already means one thing -- no such route -- and `errors.http_exception_handler`
renders it into the same `{"error": {...}}` envelope every other failure uses. Spending 404
on "no such entry" would make those two indistinguishable except by reading the prose in
`message`, and a client that mistakes "you called a path that does not exist" for "that
term was retired" reaches a wrong conclusion about the glossary rather than about its own
URL. Keeping the distinction in the BODY costs a client one field read and buys an
unambiguous status code, and it makes the single and the batch route the same DTO, so a
generated client has one model for both.

## Admission control

The byte cap (`BodySizeLimitMiddleware`) and the batch cap are both enforced, the second
against `MatchServiceLimits.max_batch_fields` -- the same number `/match/batch` uses, so an
operator tunes one knob rather than two. It is a generous cap here: the worst id this module
admits is 2,112 bytes on the wire against a field spec's 41,024, so any batch these bounds
accept is comfortably inside the derived body cap. `test_lookup_endpoint` pins that
relation rather than trusting it, because the two numbers are derived from different
constants and could drift apart on an edit to either.

What is deliberately NOT shared is the bounded work pool. `run_bounded` exists to shed CPU
work that would otherwise queue without limit; a lookup is a dict read per id and does no
CPU work worth shedding. Routing it through the pool would let a burst of cheap lookups
consume the permits that keep matching responsive -- the cheap route starving the expensive
one, which is the opposite of what admission control is for.

## Reading `matching.py`

`_governance_payload`, `_source_metadata_payload` and `_vocabulary_payload` are imported
from `matching` and called here rather than copied. That is what makes "the same
enrichment surface as a match candidate" a fact rather than a convention: a member added
to the governance payload for matched fields appears on looked-up entries in the same
edit, in the same key order, with the same present-but-null contract. A copy would have to
be remembered.

`_source_metadata_payload` is the one that makes the claim literally true rather than
nearly true. The deployment's own pass-through columns are the whole reason a lookup plane
was asked for -- a caller resolving an id it already holds wants ITS enrichment back, not
this library's four members -- and it takes a `DictionaryEntry`, which is exactly what
this module has, so no shim is needed on that side.

`_governance_payload` reads a `.governance` attribute, so it is handed `_ResolvedClass` --
the class this entry's code resolves to through the live vocabulary, resolved exactly as
`NexusMatcher._match_field` resolves it. That is a shim, and it is the honest price of
sharing the renderer with a module this lane does not own. `_vocabulary_payload` reads
`_governance` off a MATCHER, so it is handed `_VocabularySource`, which is the same shim
one level up and is built with the imported attribute-name constant rather than a literal,
so a rename in `matching.py` moves both ends together.

## The port this plane is written to

The routes below depend on `domain.ports.entry_lookup.EntryLookup` and on nothing else
that can resolve an id. That is the point of NM-V2-01 AR-5: lookup is architecturally
distinct from matching, so it gets a port in the domain rather than a private reach into
the application layer's entry map from an HTTP handler.

`LookupService` therefore never sees a matcher. It is driven end to end in
`test_lookup_endpoint` by `MappingEntryLookup` -- a domain object built from a plain dict,
with no `NexusMatcher` anywhere in the process -- which is the evidence that the
dependency is on the port and not on an object that happens to satisfy it.

`DictionaryLookup` below is the ADAPTER: the one place that knows a `NexusMatcher` is what
this server happens to be serving out of, and the one place left that reads a private
name. It prefers the port when the application layer offers it, so the day `NexusMatcher`
implements `EntryLookup` the private read stops executing without an edit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field

from nexus_matcher.domain.governance import GovernanceVocabulary
from nexus_matcher.domain.ports.entry_lookup import EntryLookup
from nexus_matcher.presentation.api.errors import (
    MalformedRequestError,
    RequestTooLargeError,
    drift,
)
from nexus_matcher.presentation.api.matching import (
    _MATCHER_GOVERNANCE_ATTR,
    DeterministicJSONResponse,
    _governance_payload,
    _source_metadata_payload,
    _vocabulary_payload,
)
from nexus_matcher.presentation.api.schemas import (
    ErrorResponse,
    GovernanceView,
    SourceMetadataView,
    VocabularyView,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nexus_matcher.domain.models.entities import DictionaryEntry
    from nexus_matcher.presentation.api.limits import MatchServiceLimits
    from nexus_matcher.presentation.api.matching import MatcherHandle

# =============================================================================
# CONSTANTS
# =============================================================================

# The private application-layer name the ADAPTER falls back to, as a named constant for
# the same reason `matching.py` names its three: the coupling is greppable from one place
# for whichever lane makes it public.
#
# It is now a fallback rather than the path. `DictionaryLookup.lookup_many` asks the
# matcher for `EntryLookup` first, so a `NexusMatcher` that implements the port is served
# through the port and this name is never read. Until it does, this is the seam, and it is
# confined to one class in one module -- the routes and `LookupService` below cannot see
# it at all.
_DICTIONARY_ENTRIES_ATTR = "_dictionary_entries"

# The longest dictionary id this service will accept, in characters. Public because it is
# the ONE bound on this identifier: `introspect.RetrievalDiagnosticRequest` reads it too,
# and a service that refused an id on one route while resolving it on another would be a
# contradiction a caller cannot act on. At least as generous as the bound
# `FeedbackRequest` puts on `chosenGovernanceId`, which is the same identifier arriving
# from the other direction.
MAX_DICTIONARY_ID_CHARS = 512

# The vocabulary reported when the server is answering out of an object that exposes none.
# Built once because it is immutable and carries no classes: `empty()` resolves every code
# to None and every classification to the open sentinel, which is the answer "nothing is
# configured here" rather than a guess at a tier.
_EMPTY_VOCABULARY = GovernanceVocabulary.empty()


# =============================================================================
# THE WIRE CONTRACT
# =============================================================================


class LookupRequest(BaseModel):
    """
    A batch of dictionary ids to resolve.

    `extra="forbid"`, which is the opposite of `MatchRequest` one module over, and the
    asymmetry is deliberate rather than an oversight.

    `MatchRequest` is `extra="ignore"` because it carries OPTIONAL knobs (`top_k`,
    `explain`) whose misspelling is visible in the response the caller already has -- five
    candidates when they asked for ten -- and because an envelope that must survive version
    skew in both directions cannot gain its first optional field under `forbid` without a
    coordinated deploy of somebody else's client.

    Neither half transfers here. This envelope has exactly one member and no optional knobs,
    so under `ignore` every extra key would be silently meaningless with nothing in the
    response to show for it; and the first optional field this route ever gains is a change
    to a route no client has generated against yet. `forbid` is also what NM-V2-01 §3 names
    as a property not to erode, and a new envelope is the cheapest place to keep it.
    """

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(
        min_length=1,
        description=(
            "The dictionary ids to resolve -- the same identifier a match candidate "
            "returns as `governanceId`. Each appears in `results` exactly once, in the "
            "order sent, whether or not it resolved. Duplicates are refused: the response "
            "is keyed by these strings and a map cannot hold two entries for one key."
        ),
    )


class LookupEntryView(BaseModel):
    """
    One dictionary entry, carrying exactly the enrichment surface a match candidate does.

    The members are `MatchCandidateView`'s minus the five that are claims about a MATCH
    rather than about the entry (`rank`, `confidence`, `decision`, `absoluteScore`,
    `explain`), in the same order, rendered by the same code. `sourceMetadata` is on this
    side of that line and not on the other: it describes the glossary row, so a caller who
    resolves an id and a caller who matches into it must get the same object back.
    There is no score here and there is not meant to be:
    a lookup hit is exact, so a confidence would be either the constant 1.0 -- a number that
    invites thresholding on something that was never measured -- or a fiction.

    `governance` follows the same rule as on a candidate: present and `null` when the entry
    carries no protection code, never absent. "This entry has no class" and "this response
    forgot to tell you" must not look alike to a caller whose next step is applying a
    classification. `vocabulary.openClassification` on the enclosing response is what makes
    that null readable.
    """

    governanceId: str = Field(
        description=(
            "The entry's own id -- the string that was looked up, echoed from the entry "
            "rather than from the request, so a dictionary whose id column disagrees with "
            "its key would be visible rather than hidden."
        )
    )
    businessName: str
    definition: str
    domain: str
    governance: GovernanceView | None
    # Last, and appended, exactly as it is on a candidate -- and it is WHY this plane can
    # claim to return "the same enrichment surface as a match candidate" at all. Before it
    # existed on either surface, the deployment's own columns reached the index and
    # stopped; a caller resolving an id it already knew got the four library-defined
    # members and none of its own. Rendered by `matching._source_metadata_payload`, the
    # same function the candidate uses, so the two objects are identical for one id rather
    # than merely similar.
    sourceMetadata: SourceMetadataView = Field(
        description=(
            "The deployment's own enrichment columns for this entry, carried through the "
            "pipeline and never interpreted -- see `SourceMetadataView`. BYTE-IDENTICAL "
            "to the `sourceMetadata` a match candidate for this same id carries: it is a "
            "fact about the entry, and neither plane is allowed to have its own opinion "
            "about one glossary row. Present on every hit, with an empty `values` when "
            "the entry carries none."
        )
    )


class LookupResponseView(BaseModel):
    """
    The whole response: one answer per requested id, plus what a `null` class means.

    Every requested id appears in `results` exactly once, in the order it was sent. An id
    the dictionary does not carry maps to `null`; it is never omitted, and `missing` names
    it a second time so the common question -- "did all of these resolve?" -- is one field
    read rather than a walk over the map.
    """

    results: dict[str, LookupEntryView | None] = Field(
        description=(
            "The caller's own ids, in the order sent, each mapped to its entry or to an "
            "explicit null. Never a partial list."
        )
    )
    missing: list[str] = Field(
        description=(
            "Exactly the ids whose `results` value is null, in the order sent. DERIVED "
            "from that map in the same pass, so the two cannot disagree; it is a "
            "convenience, not a second source of truth."
        )
    )
    # Third, and additions append after it. `results` first because it is the answer and
    # `missing` second because it is about `results`; `vocabulary` is constant per
    # deployment and is here for the same reason it rides on a match response -- a
    # governance artifact whose null means "ask a second system" is not one.
    vocabulary: VocabularyView


# =============================================================================
# THE PORT'S ADAPTER, AND THE TWO SHIMS THE SHARED RENDERERS NEED
# =============================================================================


@dataclass(frozen=True)
class _ResolvedClass:
    """
    A `.governance` attribute and nothing else, so `matching._governance_payload` can render
    an entry's class with the identical code path it renders a matched field's with.

    Not a `MatchResult` look-alike: it carries one member on purpose, so nothing here can
    grow into a second, quietly divergent, definition of what a candidate is.
    """

    governance: object | None


class _VocabularySource:
    """
    A carrier holding one attribute, so `matching._vocabulary_payload` can be called with a
    vocabulary this plane got from its PORT rather than from a matcher.

    `_vocabulary_payload` takes a matcher and reads `_governance` off it. That is right for
    the module that owns it and wrong for this one, which has no matcher after AR-5 -- so
    the vocabulary is wrapped in the shape that function expects instead of the payload
    being restated here. A copy of the renderer would be a second definition of the block
    that makes a `governance` of null readable, and the two would drift on the first edit.

    The attribute is set through the imported constant, never through a literal, so the
    rename that moves `_vocabulary_payload`'s reader moves this writer with it.
    """

    __slots__ = (_MATCHER_GOVERNANCE_ATTR,)

    def __init__(self, vocabulary: GovernanceVocabulary) -> None:
        setattr(self, _MATCHER_GOVERNANCE_ATTR, vocabulary)


class DictionaryLookup:
    """
    The `EntryLookup` adapter over whatever matcher this server was started with.

    This is the ONLY object in the lookup plane that knows a matcher exists. `LookupService`
    and both routes are written against `domain.ports.entry_lookup.EntryLookup`, so the
    application layer is reachable from here and from nowhere else on this plane.

    ## Two paths, and which one runs

    `lookup_many` asks the live matcher for the port first. A matcher that implements
    `EntryLookup` is served THROUGH the port, and the private name below is never read --
    which is what makes the fallback a fallback rather than the design. `NexusMatcher` does
    not implement it today, so the fallback is what runs today: it reads
    `NexusMatcher._dictionary_entries`, the same private-attribute coupling, with the same
    named constant and the same loud `drift()` on absence, that `matching.py` documents for
    `_match_fields`. When another lane makes that map public -- a `NexusMatcher.lookup(id)`,
    or the whole port -- nothing in this file changes and the private read stops executing.

    `isinstance` against a `runtime_checkable` Protocol checks that the members are PRESENT,
    not that they behave; that is exactly the strength wanted here. An object claiming the
    port keeps its promise or fails loudly in its own code, and this adapter is not the
    place to re-litigate whether somebody else's implementation is honest.

    ## Alias ids are not resolvable, and that is deliberate

    With `dictionary_alias_count > 0` the index also carries fabricated technical spellings
    of each entry under synthetic ids; those exist to be RETRIEVED against, not to be named.
    Resolving one would hand a caller an id that is not in their glossary and that changes
    meaning when the alias generator does. Neither path resolves them: the port's
    implementer answers for its own ids, and the fallback reads the ENTRY map, which the
    alias documents are deliberately not in.
    """

    def __init__(self, handle: MatcherHandle) -> None:
        self._handle = handle

    def matcher(self) -> object:
        """The live matcher, or the 503 that names why there is not one."""
        return self._handle.require()

    @property
    def vocabulary(self) -> GovernanceVocabulary:
        """
        The vocabulary the codes on these entries are spelled in.

        Read from the port when the matcher offers one, else off the matcher's own
        `_governance`. `empty()` when it exposes neither -- the same posture
        `matching._vocabulary_payload` takes, and for the same reason: a vocabulary that
        cannot be read must report itself as UNCONFIGURED rather than invent a tier, and
        `empty()` resolves every code to None and every classification to the open
        sentinel.
        """
        matcher = self.matcher()
        if isinstance(matcher, EntryLookup):
            return matcher.vocabulary
        vocabulary = getattr(matcher, _MATCHER_GOVERNANCE_ATTR, None)
        return vocabulary if isinstance(vocabulary, GovernanceVocabulary) else _EMPTY_VOCABULARY

    def lookup(self, entry_id: str) -> DictionaryEntry | None:
        """The entry with this id, or None. Absence is an answer, never an exception."""
        return self.lookup_many([entry_id])[0]

    def lookup_many(self, entry_ids: Sequence[str]) -> list[DictionaryEntry | None]:
        """
        One answer per id, positionally aligned to `entry_ids`.

        A list rather than a map, so the caller's map is built from the caller's own list
        and `zip(..., strict=True)` is a real oracle over the count instead of a check on a
        dict this method also built.
        """
        matcher = self.matcher()
        if isinstance(matcher, EntryLookup):
            return list(matcher.lookup_many(entry_ids))

        entries = getattr(matcher, _DICTIONARY_ENTRIES_ATTR, None)
        if entries is None:
            raise drift(
                type(matcher).__name__,
                _DICTIONARY_ENTRIES_ATTR,
                "there is no way to resolve an id and this endpoint cannot serve anything at all.",
            )
        return [entries.get(entry_id) for entry_id in entry_ids]


# =============================================================================
# RENDERING
# =============================================================================


def _entry_payload(entry: DictionaryEntry, vocabulary: object) -> dict[str, Any]:
    """
    One entry, with its keys in the enrichment order `MatchCandidateView` fixes.

    The dict literal IS the wire order -- `DeterministicJSONResponse` does not sort -- so
    the order below is load-bearing and must stay the candidate's order with the scoring
    members removed.

    The class is resolved through the LIVE vocabulary by `governance_code`, which is exactly
    what `NexusMatcher._match_field` does when it builds a candidate. Reading the entry's
    own `governance_code` a second time here rather than caching a resolution is what keeps
    the two answers to "which class does this entry confer?" the same answer.
    """
    resolve = getattr(vocabulary, "get", None)
    protection_class = resolve(entry.governance_code) if callable(resolve) else None
    return {
        "governanceId": str(entry.id),
        "businessName": entry.business_name,
        "definition": entry.definition,
        "domain": entry.domain,
        "governance": _governance_payload(_ResolvedClass(protection_class)),
        # The entry itself, not a shim: `_source_metadata_payload` reads
        # `.source_metadata`, which a `DictionaryEntry` has and a `MatchResult` does not --
        # the candidate renderer passes it the entry too. So this is the SAME function
        # over the SAME object on both planes, and the two cannot answer differently.
        "sourceMetadata": _source_metadata_payload(entry),
    }


# =============================================================================
# THE SERVICE
# =============================================================================


class LookupService:
    """
    Everything both lookup routes share; they differ only in how the ids arrive.

    Depends on the PORT, not on a matcher and not on `DictionaryLookup`. Anything that can
    resolve an id -- the adapter above, a `MappingEntryLookup` built from a dict, an
    application object that grows the port later -- drives these routes unchanged, which is
    the property AR-5 asks for and the property `test_lookup_endpoint` drives end to end
    with no `NexusMatcher` in the process at all.
    """

    def __init__(self, lookup: EntryLookup, limits: MatchServiceLimits) -> None:
        self._lookup = lookup
        self._limits = limits

    def resolve(self, ids: list[str]) -> dict[str, Any]:
        """Validate, resolve, and project -- in that order."""
        cap = self._limits.max_batch_fields
        if len(ids) > cap:
            raise RequestTooLargeError(
                message=(
                    f"{len(ids)} ids in one request exceeds this server's limit of {cap}. "
                    f"Send them in chunks of at most {cap}."
                ),
                details={"ids": len(ids), "limit": cap},
            )

        blank = [position for position, entry_id in enumerate(ids) if not entry_id.strip()]
        if blank:
            raise MalformedRequestError(
                message=(
                    f"An id must be a non-empty string, because it is the key this "
                    f"response is returned under: positions {blank!r} are empty or "
                    f"whitespace. An empty key would collide with the next empty one and "
                    f"silently answer one question twice."
                ),
                details={"blank_positions": blank},
            )

        oversized = sorted(
            {entry_id for entry_id in ids if len(entry_id) > MAX_DICTIONARY_ID_CHARS}
        )
        if oversized:
            raise MalformedRequestError(
                message=(
                    f"{len(oversized)} id(s) exceed this server's {MAX_DICTIONARY_ID_CHARS}-character "
                    f"limit for a dictionary id. No dictionary this library loads carries "
                    f"an id that long, so this is a malformed request rather than a miss."
                ),
                details={"limit_chars": MAX_DICTIONARY_ID_CHARS, "oversized": len(oversized)},
            )

        # The response is a map keyed by the caller's own ids, and a map cannot hold two
        # entries for one key. Collapsing them silently would return a map shorter than the
        # list that was sent, with nothing saying which answer was dropped -- the same
        # failure `/match` refuses duplicate paths for, arriving through a different door.
        seen: dict[str, int] = {}
        repeated: list[str] = []
        for entry_id in ids:
            seen[entry_id] = seen.get(entry_id, 0) + 1
            if seen[entry_id] == 2:
                repeated.append(entry_id)
        if repeated:
            raise MalformedRequestError(
                message=(
                    f"Every id must be distinct, because the response is keyed by it: "
                    f"{sorted(repeated)!r} appear more than once. De-duplicate before "
                    f"sending; the answer for a repeated id is the same answer."
                ),
                details={"duplicate_ids": sorted(repeated)},
            )

        # Resolution AFTER validation, in this order, because a request that is malformed
        # is malformed whether or not a dictionary is loaded: a caller who sent a duplicate
        # id must read the 422 that names it rather than a 503 that sends them to check
        # their server.
        vocabulary = self._lookup.vocabulary
        entries = self._lookup.lookup_many(ids)

        results: dict[str, dict[str, Any] | None] = {}
        # `strict=True` is the count oracle: a resolver that returned a different number of
        # answers than there were questions raises here rather than producing a map that is
        # quietly short one id.
        for entry_id, entry in zip(ids, entries, strict=True):
            results[entry_id] = None if entry is None else _entry_payload(entry, vocabulary)

        return {
            "results": results,
            # Derived from the map that was just built, in one pass over it, so "missing"
            # and "null" are two readings of one fact rather than two facts that can drift.
            "missing": [entry_id for entry_id, payload in results.items() if payload is None],
            "vocabulary": _vocabulary_payload(_VocabularySource(vocabulary)),
        }


# =============================================================================
# ROUTER
# =============================================================================

# Every way these routes are allowed to fail, WITH the body each one sends. Same table
# shape as `matching._ERROR_RESPONSES`, minus the two that cannot happen here: there is no
# deadline to exceed and no work pool to shed from, because a lookup does no CPU work.
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    413: {
        "model": ErrorResponse,
        "description": "Too many ids, or a body over this server's byte cap; chunk it.",
    },
    422: {
        "model": ErrorResponse,
        "description": (
            "Malformed request: a blank, oversized or repeated id, or a body that does not "
            "parse. `details` names the offending ids."
        ),
    },
    500: {
        "model": ErrorResponse,
        "description": (
            "This layer refused a response it could not trust -- the application layer no "
            "longer exposes the entry map. No id was resolved."
        ),
    },
    503: {
        "model": ErrorResponse,
        "description": "No dictionary is loaded, so no id can be resolved.",
    },
}

_NOT_FOUND_IS_A_BODY = (
    "An id the dictionary does not carry is **200** with `null` under that id and the id "
    "listed in `missing` -- not 404, which on this service means the route does not exist. "
    "Found and not-found are both answers and both are in the body."
)


def create_lookup_router(service: LookupService) -> APIRouter:
    """Mount both lookup routes onto one service."""
    router = APIRouter(prefix="/api/v1", tags=["Lookup"])

    @router.post(
        "/lookup",
        status_code=status.HTTP_200_OK,
        response_class=DeterministicJSONResponse,
        # Same posture as the match routes: the handler renders the body itself so that
        # `governance` is null rather than absent, and the schema is published through
        # `responses` instead of through `response_model`. `test_lookup_endpoint` validates
        # real bodies against `LookupResponseView` so the published schema and the service
        # cannot drift apart.
        response_model=None,
        responses={200: {"model": LookupResponseView}, **_ERROR_RESPONSES},
        summary="Resolve dictionary ids the caller already knows",
        description=(
            "Exact resolution by id: no scoring, no ranking, no confidence and no "
            "decision, because a hit is exact or it is absent. Every requested id comes "
            "back exactly once, in the order sent, carrying the same enrichment a match "
            "candidate carries. " + _NOT_FOUND_IS_A_BODY
        ),
    )
    async def lookup_batch(request: LookupRequest) -> DeterministicJSONResponse:
        return DeterministicJSONResponse(service.resolve(list(request.ids)))

    @router.get(
        # `:path` so an id containing a slash is addressable. Enterprise ids are opaque
        # strings this library does not get to constrain, and a route that could not
        # express half of them would send callers back to matching for exactly the ids
        # this plane exists to spare them. The cost is that `GET /api/v1/lookup/` matches
        # with an empty id, which is refused as malformed below rather than silently
        # answered.
        "/lookup/{governance_id:path}",
        status_code=status.HTTP_200_OK,
        response_class=DeterministicJSONResponse,
        response_model=None,
        responses={200: {"model": LookupResponseView}, **_ERROR_RESPONSES},
        summary="Resolve one dictionary id",
        description=(
            "The single-id form of POST /api/v1/lookup, answering the identical body with "
            "one key -- so a generated client has one model for both. " + _NOT_FOUND_IS_A_BODY
        ),
    )
    async def lookup_one(governance_id: str) -> DeterministicJSONResponse:
        return DeterministicJSONResponse(service.resolve([governance_id]))

    return router
