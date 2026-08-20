"""
nexus_matcher.domain.models.entities | Layer: DOMAIN
Core domain entities representing schemas and dictionary entries.

## Relationships
# DEPENDS_ON → shared/types/base :: base types and enums
# USED_BY    → domain/ports/* :: port interfaces use these models
# USED_BY    → application/use_cases/* :: use cases manipulate these
# USED_BY    → infrastructure/adapters/* :: adapters convert to/from these

## Attributes
# Security: Models contain no secrets, but DictionaryEntry may have PII classification
# Performance: Frozen dataclasses for immutability and hashability
# Reliability: Full validation in __post_init__
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from nexus_matcher.domain.governance import ProtectionClass
from nexus_matcher.shared.types.base import (
    DataType,
    DocumentId,
    EntityId,
    MatchDecision,
    Metadata,
    PerformanceMetrics,
    ProtectionLevel,
    Score,
    ScoreBreakdown,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# =============================================================================
# SCHEMA FIELD - Represents a field from any schema format
# =============================================================================


@dataclass(frozen=True, slots=True)
class SchemaField:
    """
    Domain model representing a field from a schema.

    This is the normalized representation of fields from any schema format
    (Avro, JSON Schema, SQL DDL, CSV headers, etc.).

    Attributes:
        name: Field name (e.g., "customer_email")
        data_type: Normalized data type
        full_path: Dot-separated path (e.g., "customer.contact.email")
        parent_path: Path to parent record (e.g., "customer.contact")
        description: Optional documentation/description
        is_nullable: Whether the field can be null
        is_array: Whether the field is an array type
        array_item_type: If array, the type of items
        default_value: Default value if specified
        constraints: Additional constraints (e.g., min/max, pattern)
        metadata: Additional metadata from source schema
    """

    name: str
    data_type: DataType
    full_path: str = ""
    parent_path: str = ""
    description: str = ""
    is_nullable: bool = True
    is_array: bool = False
    array_item_type: DataType | None = None
    default_value: Any = None
    constraints: frozenset[str] = field(default_factory=frozenset)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the schema field."""
        # Ensure full_path is set
        if not self.full_path:
            object.__setattr__(self, "full_path", self.name)

    @property
    def is_nested(self) -> bool:
        """Check if field is part of a nested structure."""
        return "." in self.full_path

    @property
    def depth(self) -> int:
        """Get nesting depth (0 for root fields)."""
        return self.full_path.count(".")

    @property
    def root_name(self) -> str:
        """Get the root-level name."""
        return self.full_path.split(".")[0]

    def with_path(self, parent: str) -> SchemaField:
        """Create new field with updated path."""
        new_path = f"{parent}.{self.name}" if parent else self.name
        new_parent = parent if parent else ""
        return SchemaField(
            name=self.name,
            data_type=self.data_type,
            full_path=new_path,
            parent_path=new_parent,
            description=self.description,
            is_nullable=self.is_nullable,
            is_array=self.is_array,
            array_item_type=self.array_item_type,
            default_value=self.default_value,
            constraints=self.constraints,
            source_metadata=self.source_metadata,
        )

    def to_searchable_text(self) -> str:
        """Generate text representation for embedding/search."""
        parts = [
            self.name.replace("_", " "),
            self.description,
        ]
        return " ".join(filter(None, parts))


# =============================================================================
# DICTIONARY ENTRY - Represents an entry from a data dictionary
# =============================================================================


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    """
    Domain model representing an entry in a data dictionary.

    Attributes:
        id: Unique identifier for this entry
        business_name: Human-readable business name
        logical_name: Technical/logical name (often abbreviated)
        definition: Full definition/description
        data_type: Expected data type
        protection_level: Data classification level
        governance_code: The controlled protection code this entry carries, in the
            spelling the caller's `GovernanceVocabulary` declares, or None when the entry
            has no code and therefore sits at the open tier. Only ever set from a code
            that vocabulary DEFINES -- an unrecognised token is rejected at load time and
            survives only as `source_metadata['governance_code_raw']`, because a stored
            code nobody defined reads as governance and is not.
        domain: Business domain (e.g., "Sales", "Finance")
        parent_table: Parent table/entity name
        sample_values: Example values
        synonyms: Alternative names/terms
        metadata: Additional metadata

    `id` doubles as the GOVERNANCE ID: it is the handle a caller quotes when they say
    "this field inherits that entry's class", and `MatchResult.governance_id` promotes it
    so nobody has to reach through the entry to find it.
    """

    id: DocumentId
    business_name: str
    logical_name: str
    definition: str
    data_type: DataType
    protection_level: ProtectionLevel = ProtectionLevel.INTERNAL
    # Deliberately NOT part of `to_searchable_text()`, and therefore not part of
    # `content_hash`. Governance is metadata about the term, not a description of it;
    # embedding it would change every vector the day a glossary is re-classified and turn
    # the next incremental sync into a full re-embed -- the exact cost `content_hash`
    # exists to avoid, and a documented property of this package (a governance-only edit
    # must encode zero texts).
    governance_code: str | None = None
    domain: str = ""
    parent_table: str = ""
    sample_values: tuple[str, ...] = field(default_factory=tuple)
    synonyms: frozenset[str] = field(default_factory=frozenset)
    is_enum: bool = False
    enum_values: tuple[str, ...] = field(default_factory=tuple)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the dictionary entry."""
        if not self.id:
            raise ValueError("DictionaryEntry.id cannot be empty")
        if not self.business_name:
            raise ValueError("DictionaryEntry.business_name cannot be empty")

    @property
    def content_hash(self) -> str:
        """Generate content hash for change detection."""
        content = (
            f"{self.business_name}|{self.logical_name}|{self.definition}|{self.data_type.value}"
        )
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

    def to_searchable_text(self) -> str:
        """Generate text representation for embedding/search."""
        parts = [
            self.business_name,
            self.logical_name.replace("_", " "),
            self.definition,
        ]
        if self.synonyms:
            parts.extend(self.synonyms)
        return " ".join(filter(None, parts))

    def matches_type(self, other_type: DataType, strict: bool = False) -> float:
        """
        Calculate type compatibility score.

        Args:
            other_type: Type to compare against
            strict: If True, require exact match

        Returns:
            Compatibility score (0.0 to 1.0)
        """
        if self.data_type == other_type:
            return 1.0

        if strict:
            return 0.0

        # Define compatible type groups
        numeric_types = {
            DataType.INTEGER,
            DataType.LONG,
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.DECIMAL,
        }
        string_types = {DataType.STRING, DataType.UUID, DataType.JSON}
        temporal_types = {DataType.DATE, DataType.TIMESTAMP}

        # Check group compatibility
        if self.data_type in numeric_types and other_type in numeric_types:
            return 0.8

        if self.data_type in string_types and other_type in string_types:
            return 0.9

        if self.data_type in temporal_types and other_type in temporal_types:
            return 0.9

        # String can represent most types
        if DataType.STRING in (self.data_type, other_type):
            return 0.5

        return 0.0


# =============================================================================
# MATCH RESULT - Represents a matching result
# =============================================================================


@dataclass(frozen=True, slots=True)
class MatchResult:
    """
    Domain model representing a match between a schema field and dictionary entry.

    A caller matches a field IN ORDER TO INHERIT the entry's governance, so the two
    things they came for -- which entry, and what class it carries -- are first-class
    fields here rather than something to fish out of `source_metadata`.

    Attributes:
        schema_field: The source schema field
        dictionary_entry: The matched dictionary entry
        rank: Rank among all matches for this field (1-based)
        final_confidence: Overall confidence score (0.0 to 1.0)
        score_breakdown: Detailed component scores
        decision: Classification decision
        performance: Performance metrics for this match
        governance: The protection class this match would confer, resolved through the
            caller's vocabulary. None when the entry carries no code (the open tier), and
            also on a RANK-1 REJECT, which confers nothing -- see `__post_init__`. A
            rejected runner-up keeps its class: no field inherits from rank 2.
        governance_id: The matched entry's id, which is the governance id. Always
            populated; derived from `dictionary_entry` when not supplied, and refused
            when supplied and different, because two answers to "which entry's class is
            this?" is worse than none.

    Populated on EVERY candidate, not only rank 1. A consumer deciding between rank 1 and
    rank 2 needs to see that one of them is a personal-identifier class and the other is
    not; that is usually the deciding fact, and having it on rank 1 alone makes the
    comparison impossible without a second lookup.
    """

    schema_field: SchemaField
    dictionary_entry: DictionaryEntry
    rank: int
    final_confidence: Score
    score_breakdown: ScoreBreakdown
    decision: MatchDecision
    performance: PerformanceMetrics
    # `ProtectionClass` is a plain frozen dataclass in the same layer, so this costs the
    # domain no new dependency. Typed as Any-free on purpose: the annotation is what
    # tells a reader that None is a real, meaningful state here.
    governance: ProtectionClass | None = None
    governance_id: str = ""

    def __post_init__(self) -> None:
        """Validate match result."""
        if not 0.0 <= self.final_confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.final_confidence}")
        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")

        if not self.governance_id:
            object.__setattr__(self, "governance_id", str(self.dictionary_entry.id))
        elif self.governance_id != str(self.dictionary_entry.id):
            raise ValueError(
                f"governance_id {self.governance_id!r} does not name the matched entry "
                f"{self.dictionary_entry.id!r}. The governance id IS the entry id; two "
                f"different answers to 'whose class is this?' is worse than none."
            )

        # A REJECTED RANK 1 INHERITS NOTHING, and that is enforced here rather than at the
        # call site so it cannot be forgotten by the next one.
        #
        # A rejected top match means "no entry in this glossary describes this field". A
        # novel field -- the case that most needs a human -- would otherwise arrive
        # carrying the class of the least-bad candidate, which is a confident-looking
        # classification derived from a match the matcher itself rejected. That is
        # NM-0005's shape inverted: instead of losing a class it should have had, the
        # field gains one it should not.
        #
        # `rank == 1` is a CORRECTION, not the original rule. REJECT is a per-CANDIDATE
        # verdict -- `_determine_decision` compares every rank against `review_threshold`
        # -- but the justification above is a per-FIELD one, and a field inherits from
        # rank 1 only. Unqualified, the strip also blanked the runner-ups: measured over
        # the 26-field Gravel Bay pack at top_k=5, 66 of 104 runner-up candidates came
        # back with no class although the indexed entry carried a real code, 16 of them
        # direct identifiers. `decision` could not disambiguate the two nulls, because
        # both read REJECT, so the wire said "no class" where it meant "withheld" -- and
        # it deleted exactly the rank-1-versus-rank-2 comparison this class's own
        # docstring says the field exists to provide.
        #
        # At the shipped thresholds the narrowed guard is LATENT: `review_threshold` is
        # 0.50 and a rank-1 confidence cannot fall below `semantic_weight * fusion_alpha`
        # = 0.63, so no top match can be rejected. It stays because that floor is a
        # consequence of two tunable numbers, not a law; it fires for a caller who raises
        # the threshold past it, which is what `TestRejectConfersNothing` constructs.
        if self.rank == 1 and self.decision == MatchDecision.REJECT and self.governance is not None:
            object.__setattr__(self, "governance", None)

    @property
    def is_auto_approved(self) -> bool:
        """Check if this match is auto-approved."""
        return self.decision == MatchDecision.AUTO_APPROVE

    @property
    def needs_review(self) -> bool:
        """Check if this match needs human review."""
        return self.decision == MatchDecision.REVIEW

    @property
    def is_rejected(self) -> bool:
        """Check if this match is rejected."""
        return self.decision == MatchDecision.REJECT


# =============================================================================
# FIELD-LEVEL VERDICT - one answer per column, including "nothing matched"
# =============================================================================


class FieldDecision(str, Enum):
    """
    ONE verdict per FIELD. `MatchDecision` is per CANDIDATE, and they are not the same
    question.

    A consumer of this library writes one decision per column -- into a metadata sheet, a
    model attribute, a review queue. `MatchDecision` cannot answer that on its own for two
    reasons:

    **It is a per-candidate verdict.** Every rank is compared against `review_threshold`,
    so runner-ups are routinely REJECT on a field whose top match is excellent. Rolling
    that up is a rule, and a rule nobody wrote down is a rule every client guesses
    differently.

    **It has no way to say "nothing matched".** Measured on the shipped configuration:
    `final_confidence` for rank 1 has a structural floor of `semantic_weight *
    fusion_alpha` = 0.70 * 0.90 = **0.63**, and `review_threshold` is **0.50**. 0.63 > 0.50,
    so a rank-1 candidate can never be REJECT on score alone and every field comes back at
    least REVIEW. `NO_MATCH` is the member that closes that hole.

    That is not a hypothetical. Measured 2026-08-19 on the shipped bundled encoder
    (`bge-small-en-v1.5-onnx-int8`) over the 30-entry fictional glossary in
    `examples/governance/`, all 26 fields of the pack's own schema:

        rank-1 confidence      0.8058 .. 0.8958      not one below review_threshold
        rank-1 absolute cosine 0.5830 .. 0.9688

    The two fields the pack declares `expected_id: null` -- the ones for which NO glossary
    term is a right answer -- score **0.5830** and **0.5943** absolute, and come back at
    confidence 0.8058 and 0.8792. Every field that does have a right answer scores
    **>= 0.7352** absolute. The confidence tells the two groups apart not at all; the
    absolute score separates them with a gap of 0.14 and nothing in it.

    ## Reading a floor off a fixture is how you get one that never fires

    An earlier version of this docstring illustrated the same point with the field
    `misc.zzz_unmatchable` in `tests/unit/presentation/api/_support.py` at an absolute
    similarity of **0.123**, and that number is real -- re-measured 2026-08-19 at 0.1231,
    confidence 0.7350. It is also a `BagOfTokensProvider` number: that fixture substitutes
    a deterministic stand-in for the encoder precisely so a unit test need not load a
    33 MB model, and a bag-of-tokens cosine between two texts sharing no tokens is near
    zero in a way a sentence encoder's never is.

    On the shipped encoder, total nonsense scores about **0.58**, not 0.12. A floor chosen
    from the fixture's spread -- 0.30, say -- is below every score any real field will ever
    produce, so it can never fire, and a caller who set it would believe they had
    configured a safety net they do not have. The useful range on this encoder and this
    pack is roughly 0.60 to 0.65; on another corpus it is another number, which is the
    whole reason this library ships none. Calibrate against your own glossary, on the
    encoder you will actually deploy, and read the spread before picking.

    ## What NO_MATCH claims, exactly

    *This response carries nothing this field may inherit from.* Two ways to earn it:

      1. the field came back with **no candidates at all**; or
      2. an **absolute floor is configured** (`MatchingConfig.absolute_score_floor`) and
         rank 1 does not clear it.

    Case 2 requires the caller to have chosen a floor. This library ships none and will
    not invent one -- a floor is a statement about a score distribution, and the
    distribution belongs to a corpus this library has never seen. With no floor
    configured, case 2 never fires and `NO_MATCH` can only come from case 1.

    Case 1 needs no calibration and is therefore not gated on the floor: the response
    already says the candidate list is empty, and this only names the claim the response
    was making anyway.

    ## What NO_MATCH does NOT distinguish

    An empty candidate list means "nothing came back", which is not quite the same as
    "the dictionary holds nothing relevant" -- a retrieval or encoding failure inside
    `_match_field` also produces an empty list. What it IS reliably distinguished from is
    **"this field was not processed"**: a field decision exists for every field that was,
    keyed the same way the results are, so a missing key means unprocessed and a present
    `NO_MATCH` means processed-and-empty. That distinction is the reason the recommended
    design returns candidates plus a verdict rather than an empty candidate list on its
    own -- an empty list alone cannot tell the two apart.

    ## Why this is a separate enum from `MatchDecision`

    Two reasons, and only the first is about meaning.

    A per-candidate REJECT ("this candidate is below the bar") and a per-field NO_MATCH
    ("no candidate is worth inheriting from") are different claims about the world, and a
    vocabulary that spells them the same way loses the difference.

    And on the wire, `decision` is an enum a Java client has already generated a closed
    Java `enum` from. Adding a value to it turns an ordinary 200 into a deserialisation
    failure on a client built against last week's schema. A NEW field carrying a WIDER
    vocabulary is additive; widening an existing one is not.

    The three shared members are the rank-1 candidate's own `MatchDecision`, passed
    through unchanged and spelled identically, which
    `tests/unit/domain/test_field_decision.py` pins so the two cannot drift apart.
    """

    AUTO_APPROVE = "AUTO_APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    NO_MATCH = "NO_MATCH"


def derive_field_decision(
    matches: Sequence[MatchResult],
    absolute_score_floor: float | None = None,
) -> FieldDecision:
    """
    The one verdict for one field, derived from rank 1 and an optional absolute floor.

    ONE implementation, so the HTTP surface, the CLI and a library caller cannot disagree
    about what a field's decision is. A roll-up rule that lives in three places is three
    rules.

    `absolute_score_floor` is compared against rank 1's `ScoreBreakdown.absolute_cosine` --
    the RAW dense score, which unlike `final_confidence` has no structural floor and is
    the only number in the breakdown comparable across fields. `None` means no floor is
    configured, which is the default and the only honest default for a library that has
    never seen the caller's corpus.

    A floor is configured and rank 1 has **no** absolute score -- the dense arm never
    returned it, so it reached the shortlist through the lexical arm alone -- also gives
    NO_MATCH. A candidate the dense retriever never proposed offers no evidence that it
    clears an absolute similarity floor, and clearing a floor on evidence that does not
    exist is the direction of this failure that costs a wrong classification rather than a
    human review.

    Reading `matches[0]` is deliberate and matches the domain's existing rule that a field
    inherits from rank 1 only (see `MatchResult.__post_init__`). The sequence is assumed
    to be in rank order, which is the order every producer in this package emits.
    """
    if not matches:
        return FieldDecision.NO_MATCH

    top = matches[0]
    if absolute_score_floor is not None:
        absolute = top.score_breakdown.absolute_cosine
        if absolute is None or absolute < absolute_score_floor:
            return FieldDecision.NO_MATCH

    # `MatchDecision` and `FieldDecision` share these three spellings by contract, so the
    # value carries across. Going through the VALUE rather than the name means a member
    # `FieldDecision` does not have raises here instead of being silently mapped to
    # something plausible.
    return FieldDecision(top.decision.value)


# =============================================================================
# SCHEMA - Container for a complete schema
# =============================================================================


@dataclass(frozen=True, slots=True)
class Schema:
    """
    Domain model representing a complete schema.

    Attributes:
        name: Schema name
        namespace: Schema namespace (e.g., "com.company.domain")
        fields: All fields in the schema (flattened)
        source_format: Original format (avro, json_schema, sql_ddl, etc.)
        source_metadata: Original schema metadata
    """

    name: str
    fields: tuple[SchemaField, ...]
    namespace: str = ""
    source_format: str = "unknown"
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def field_count(self) -> int:
        """Get total number of fields."""
        return len(self.fields)

    @property
    def field_paths(self) -> frozenset[str]:
        """Get all field paths."""
        return frozenset(f.full_path for f in self.fields)

    def get_field(self, path: str) -> SchemaField | None:
        """Get field by path."""
        for f in self.fields:
            if f.full_path == path:
                return f
        return None

    def get_fields_by_parent(self, parent_path: str) -> tuple[SchemaField, ...]:
        """Get all fields with given parent path."""
        return tuple(f for f in self.fields if f.parent_path == parent_path)


# =============================================================================
# MATCHING SESSION - Container for a complete matching operation
# =============================================================================


@dataclass(frozen=True, slots=True)
class MatchingSession:
    """
    Domain model representing a complete schema matching session.

    Attributes:
        session_id: Unique session identifier
        schema: The source schema being matched
        results: All match results grouped by field path
        started_at: Session start timestamp
        completed_at: Session completion timestamp
        metadata: Session metadata
        minimum_achievable_confidence: The structural floor of `final_confidence` for
            the configuration that produced this session, or None when it is unknown
            (a reranker was wired, or the session was not built by `NexusMatcher`).
            Supplied by `NexusMatcher.match_schema_session`; see
            `MatchingConfig.minimum_achievable_confidence` for the derivation.
        absolute_score_floor: The absolute-score floor beneath which a field is reported
            `NO_MATCH`, or None when no floor is configured -- which is the default, and
            the only default a library with no view of the caller's corpus may ship.
            Supplied by `NexusMatcher.match_schema_session` from
            `MatchingConfig.absolute_score_floor`.
    """

    session_id: EntityId
    schema: Schema
    results: dict[str, tuple[MatchResult, ...]]
    total_duration_ms: float
    metadata: Metadata = field(default_factory=Metadata)
    # A float rather than the MatchingConfig it comes from: this is a DOMAIN object and
    # must not import from the application layer. Defaults to None so a session built by
    # hand -- every existing caller and test -- keeps working, at the cost of losing the
    # threshold check below, which is the honest trade: the floor genuinely is unknown.
    minimum_achievable_confidence: float | None = None
    # Defaults to None -- no floor -- for the same reason `MatchingConfig` does: a floor
    # is a calibration decision about a score distribution, and the distribution belongs
    # to the caller's corpus.
    absolute_score_floor: float | None = None

    @property
    def field_count(self) -> int:
        """Get number of fields matched."""
        return len(self.results)

    def field_decisions(self) -> dict[str, FieldDecision]:
        """
        One verdict per field, in schema order -- the answer a consumer writes down.

        Every result key appears exactly once, including a field nothing matched, which
        gets `NO_MATCH` rather than being dropped. That is the conservation law applied to
        the verdict: a field missing from this map would inherit nothing while nothing
        said so.

        Derived through `derive_field_decision` using this session's own
        `absolute_score_floor`, so the numbers a library caller reads here and the ones an
        HTTP caller reads over the wire come out of the same rule.
        """
        return {
            path: derive_field_decision(matches, self.absolute_score_floor)
            for path, matches in self.results.items()
        }

    @property
    def total_matches(self) -> int:
        """Get total number of match results."""
        return sum(len(matches) for matches in self.results.values())

    @property
    def auto_approval_rate(self) -> float:
        """Calculate auto-approval rate."""
        if not self.results:
            return 0.0

        auto_approved = sum(
            1 for matches in self.results.values() if matches and matches[0].is_auto_approved
        )
        return auto_approved / len(self.results)

    def get_top_matches(self) -> dict[str, MatchResult]:
        """Get top match for each field."""
        return {path: matches[0] for path, matches in self.results.items() if matches}

    def get_low_confidence_fields(self, threshold: float | None = None) -> list[str]:
        """
        The fields a human still has to look at. Returns their result keys, in schema
        order.

        THE DEFAULT USED TO BE 0.6 AND SELECTED NOTHING, EVER (DX-001, museum NM-0027).
        `final_confidence` has a structural floor of about **0.63** in the shipped
        configuration -- `semantic_weight` 0.70 times `fusion_alpha` 0.90, because the
        fused retrieval score is min-max normalised per field and the rank-1 candidate
        therefore always lands at or above `fusion_alpha`. No top match could fall below
        0.6, so the one API whose name answers "which of these should I not trust?"
        answered "none of them" on every schema, including schemas where nothing was
        trustworthy. Measured on a 6-field schema: default 0 flagged, 0.87 -> 6 flagged,
        actual confidences 0.730-0.755. That is a silent governance failure, the same
        class as NM-0005. This session's exact floor is
        `minimum_achievable_confidence`, and it is None when a reranker was in play.

        `threshold=None` (the default) means **"was not auto-approved"**: the field is
        returned unless its top match carries `MatchDecision.AUTO_APPROVE`. That is the
        one definition of "low confidence" that cannot silently drift away from the
        configuration, because it reads the decision the matcher already made using the
        calibrated `auto_approve_threshold` AND `min_confidence_gap` -- so a confident
        but ambiguous near-tie, which the numeric comparison would clear, is correctly
        still flagged. A session has no access to the config, so no numeric default
        could track it.

        Pass a float to ask a different question ("show me everything under 0.8"). A
        threshold at or below `minimum_achievable_confidence` is REFUSED with a
        ValueError naming the floor, rather than returning [] -- an empty list reads as
        "nothing to review", which is exactly the lie this method used to tell.

        A field with NO matches at all is always included. Nothing matched it, which is
        the least trustworthy outcome there is; it used to be skipped, because
        `matches[0]` needs a match to exist and the guard dropped the field instead of
        flagging it.
        """
        floor = self.minimum_achievable_confidence
        if threshold is not None and floor is not None and threshold <= floor:
            raise ValueError(
                f"threshold={threshold!r} is at or below this configuration's structural "
                f"floor of {floor:.4f}, so no top match can fall below it and this call "
                f"could only ever return an empty list -- which reads as 'nothing to "
                f"review'. Use get_low_confidence_fields() with no argument to flag every "
                f"field that was not auto-approved, or pass a threshold above {floor:.4f}."
            )

        flagged: list[str] = []
        for path, matches in self.results.items():
            if not matches:
                flagged.append(path)
            elif threshold is None:
                if not matches[0].is_auto_approved:
                    flagged.append(path)
            elif matches[0].final_confidence < threshold:
                flagged.append(path)
        return flagged
