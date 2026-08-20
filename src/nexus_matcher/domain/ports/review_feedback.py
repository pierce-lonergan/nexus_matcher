"""
nexus_matcher.domain.ports.review_feedback | Layer: DOMAIN
Port interface for a consumer of reviewer verdicts that is allowed to influence matching.

## Relationships
# DEPENDS_ON → domain/models/entities :: SchemaField, DictionaryEntry, MatchResult
# DEPENDS_ON → domain/ports/entry_lookup :: EntryLookup, how a consumer re-reads the glossary
# USED_BY    → application/use_cases/match_schema :: the matcher consults it before retrieval
# USED_BY    → application/feedback_loop :: the reference consumer implements it

## Attributes
# Security: verdicts carry the caller's own field names and glossary ids; nothing here
#           interprets either
# Performance: `approved_pair` runs once per field on the match path and must be O(1)
# Reliability: "no opinion" is a RETURN VALUE, never an exception

## Why this port exists at all

A reviewer's verdict is the only input to this library that is *known* to be right, and
until now it went into an append-only file and stopped there. That file is a legitimate
default -- it is evidence, and evidence is worth keeping whether or not anything reads it
-- but a system whose most expensive input is human review and whose architecture cannot
express "and then the answer is used" has made the review permanently write-only.

This port is the seam that fixes the ARCHITECTURE without changing the DEFAULT. A
deployment attaches an implementation; nothing in the shipped wiring constructs one, and
`NullFeedbackConsumer` below is the reference implementation of consuming nothing.

## What an implementation is allowed to do, and what it must not

**It may answer "a human already decided this field."** That answer replaces retrieval for
that field. It is the only thing in this library entitled to do that, and it is entitled
because the answer did not come from a model.

**It must not rank, score or re-weight.** There is no hook here for "nudge the confidence
of a candidate a reviewer liked". That is a different feature with a different evidence
bar -- fine-tuning on exactly this signal was measured on this repository's own benchmark
and LOST 0.0277 P@1 (`benchmarks/results/exp_finetune_transfer.json`) -- and a port that
offered both would let a deployment believe it had turned on the measured-good one when it
had turned on the measured-bad one.

## The three decisions this port forces into the open

**WHAT IS THE KEY.** `approved_pair` is handed the whole `SchemaField`, not a name. The
same leaf name under a different parent is a DIFFERENT QUESTION -- `ADDRESS_LINE_1` under
a billing record and under a shipping record inherit different terms -- and a consumer
keyed on the leaf alone returns a confidently wrong answer with no symptom except a count
nobody checks. Passing the field makes the parent reachable; `application/feedback_loop`
documents which part of it the reference consumer uses and proves it against the
repeated-leaf fixture.

**WHAT INVALIDATES IT.** `bind` is called with the CURRENT glossary every time the
matcher's dictionary changes, and an implementation that ignores it is one that keeps
applying a verdict a reviewer gave about a term that has since been re-defined, re-classified
or deleted. `approval_binding` below is the fingerprint that makes those three detectable,
and it is deliberately NOT `DictionaryEntry.content_hash` alone -- see its docstring.

**HOW IT SHOWS ON THE WIRE.** The rendering is not this layer's decision, but the
VOCABULARY is, and it is here because it must be one vocabulary. `ApprovedPair` carries
the reviewer and the moment, so whatever surface renders it has the facts it needs to say
"a human decided this" rather than implying a retriever did -- and `MatchProvenance` below
is the value that says it, read through the single `provenance_of`. That member exists
because the alternative was tried and was wrong: a bypass identified by a magic confidence
of 1.0, asserted to be outside the scorer's range and demonstrably inside it.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from nexus_matcher.domain.models.entities import DictionaryEntry, MatchResult, SchemaField
from nexus_matcher.domain.ports.entry_lookup import EntryLookup

# =============================================================================
# WHERE AN ANSWER CAME FROM
# =============================================================================

# The value the matcher stamps on `PerformanceMetrics.retrieval_stage` for a candidate it
# did not retrieve. Named here rather than spelled inline at the two ends, because the
# matcher WRITES it and `provenance_of` READS it, and a string literal maintained in two
# layers is exactly the drift this repository keeps a `drift()` helper for.
APPROVED_PAIR_STAGE = "approved_pair"


class MatchProvenance(str, Enum):
    """
    Where a candidate's answer came from. A VALUE, deliberately, and not a number.

    ## Why this exists, stated plainly because it replaces a claim that was false

    A bypassed candidate used to be identifiable only by inference: `final_confidence` was
    set to 1.0 and shipped source asserted that value was "outside the range the model can
    produce". It is not. The five default scoring weights sum to exactly 1.0 and every
    signal is attainable at 1.0, so ordinary retrieval reaches 1.0 whenever all five are
    maximal -- and a client reading `(confidence, decision)` then cannot tell a human's
    answer from a very good match. The two tests that "proved" unreachability each matched
    one fixture and reported the maximum it happened to produce, which is an observation,
    not a property. See `tests/unit/application/test_approved_pair_bypass.py`.

    ## Why a member and not a cleverer sentinel

    A sentinel is a number carrying a meaning that is not its magnitude, so a reader has to
    know the convention to read the number at all, and any change to the scorer's range can
    collide with it silently. Making the sentinel genuinely unreachable -- refusing to emit
    1.0 from the scorer -- would also mean changing shipped scoring behaviour to buy a
    signalling channel, which is the wrong thing to spend a ranking on.

    A value cannot collide with a score. `RETRIEVAL` and `APPROVED_PAIR` say what happened;
    nothing has to be inferred from a conjunction, and nothing about the scorer's range can
    make them ambiguous.

    ## Closed, and that is deliberate

    This is the LIBRARY'S OWN vocabulary -- the same category as `MatchDecision` -- not the
    caller's. Nothing in a deployment's glossary, governance scheme or naming standard can
    add a value here, so publishing it closed costs a caller nothing and tells a generated
    client the whole set. A caller-supplied vocabulary would have to stay open; this one
    must not, or "some other provenance" becomes expressible and means nothing.
    """

    RETRIEVAL = "RETRIEVAL"
    APPROVED_PAIR = "APPROVED_PAIR"


def provenance_of(match: MatchResult) -> MatchProvenance:
    """
    Read a candidate's provenance off the result the matcher produced.

    ONE READER, so a library caller and the HTTP projection cannot come to different
    conclusions about the same candidate. `MatchResult` carries the fact already, in
    `performance.retrieval_stage`; this function is the single place that turns it into the
    published vocabulary, rather than each surface comparing its own copy of a string.

    `RETRIEVAL` is the default for every stage this function does not recognise, and that
    is the safe direction: a new retrieval stage added by the matcher reads as retrieval,
    which is what it is. The opposite default would let an unrecognised stage claim a human
    decided the field.

    Args:
        match: The candidate. Read defensively -- a caller's own `MatchResult`-shaped
            object may carry no `performance` at all, and this function must not be a way
            to take a response down.
    """
    stage = getattr(getattr(match, "performance", None), "retrieval_stage", None)
    if stage == APPROVED_PAIR_STAGE:
        return MatchProvenance.APPROVED_PAIR
    return MatchProvenance.RETRIEVAL


# =============================================================================
# THE VERDICT VOCABULARY
# =============================================================================


class ReviewVerdict(str, Enum):
    """
    What a reviewer actually did, in four values because three of them are answers and the
    fourth is the shape of a record that could not give one.

    `APPROVED` and `REJECTED` are the easy pair. The one that carries information nothing
    else can is `MANUAL_OVERRIDE`: the reviewer chose a term the matcher NEVER PROPOSED --
    not rank 2, not rank 20, absent from the candidate list entirely.

    THE DISTINCTION IS NOT COSMETIC AND IT IS NOT ABOUT THE BYPASS. "The top match was
    wrong and the reviewer took rank 3" and "the right answer was nowhere in the list" are
    the same record once they are collapsed into a boolean, and they imply opposite fixes:
    the first is a RANKING problem, addressable by weights, fusion or a reranker; the
    second is a RECALL problem, and no amount of re-ranking a list that never contained the
    answer will ever fix it. A pipeline that cannot count the second one cannot tell
    which of the two it has.

    `UNSPECIFIED` is not a verdict a reviewer gives. It is what a record recorded before
    this vocabulary existed reads as when its boolean says the suggestion was wrong: the
    reviewer chose *something*, and whether that something had been proposed is
    unrecoverable. It exists so that loss is COUNTABLE rather than silently absorbed into
    one of the other three.
    """

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    UNSPECIFIED = "UNSPECIFIED"

    @property
    def is_a_choice(self) -> bool:
        """
        Whether this verdict names a term the reviewer stood behind.

        `REJECTED` does not: it means "nothing in this glossary governs this field", which
        is an answer, and an answer that must never be turned into a bypass. Everything
        else does -- including `UNSPECIFIED`, where the reviewer's chosen id is on the
        record even though the provenance of that choice is not.
        """
        return self is not ReviewVerdict.REJECTED


# =============================================================================
# WHAT A CONSUMER IS FED
# =============================================================================


@dataclass(frozen=True, slots=True)
class ReviewedVerdict:
    """
    One reviewer verdict, normalised out of whatever trail it was read from.

    `field_key` is the field identity the verdict is about, spelled exactly as the match
    response keyed it -- see `application.use_cases.match_schema.field_result_key`. That
    string is the ONLY field identity an audit record carries, so it is also the only one a
    consumer can key on, and it is parent-qualified for every parser this library ships.

    `recorded_at` is the SERVER's stamp, not the client's. Ordering verdicts by a
    reviewer's own clock lets one workstation with a wrong time zone overwrite a later
    decision made by somebody else, and "whose verdict is in force" is exactly the question
    ordering answers.
    """

    field_key: str
    verdict: ReviewVerdict
    chosen_entry_id: str
    suggested_entry_id: str | None = None
    reviewer: str = ""
    recorded_at: str = ""

    def __post_init__(self) -> None:
        """Refuse a record that claims a choice and does not name one."""
        if not self.field_key:
            raise ValueError("ReviewedVerdict.field_key cannot be empty")
        if self.verdict.is_a_choice and not self.chosen_entry_id:
            raise ValueError(
                f"verdict {self.verdict.value} on {self.field_key!r} names no chosen entry. "
                f"A verdict that stands behind a term must say which one; REJECTED is the "
                f"value for 'nothing in this glossary governs this field'."
            )
        if self.verdict is ReviewVerdict.REJECTED and self.chosen_entry_id:
            raise ValueError(
                f"REJECTED verdict on {self.field_key!r} also names chosen entry "
                f"{self.chosen_entry_id!r}. Those are two different answers to one "
                f"question, and a consumer would have to guess which the reviewer meant."
            )


# =============================================================================
# WHAT A CONSUMER ANSWERS WITH
# =============================================================================


@dataclass(frozen=True, slots=True)
class ApprovedPair:
    """
    A human's standing answer for one field, resolved against the glossary in force.

    `entry` is the LIVE entry, not a copy taken when the verdict was given: a consumer
    that handed back a remembered entry would let a match inherit a definition and a
    protection class that no longer exist in the deployment's glossary, which is the exact
    failure `binding` is here to prevent.

    `binding` is `approval_binding(entry)` as it stood when this pair was last accepted.
    It is carried on the answer rather than kept private so that whatever renders the
    result can show WHICH generation of the term a human signed off on.
    """

    field_key: str
    entry: DictionaryEntry
    verdict: ReviewVerdict
    binding: str
    reviewer: str = ""
    decided_at: str = ""


def approval_binding(entry: DictionaryEntry) -> str:
    """
    The fingerprint an approval is bound to: what the term SAID and what it was WORTH.

    `DictionaryEntry.content_hash` is deliberately not enough, and this function exists to
    say so out loud. That hash covers `business_name | logical_name | definition |
    data_type` and DELIBERATELY EXCLUDES `governance_code`, because governance is metadata
    about a term rather than a description of it and folding it in would turn every
    re-classification into a full re-embed. That exclusion is right for its own purpose and
    exactly wrong for this one: a reviewer approving a field against a term is approving
    the CLASS the field will inherit, so a glossary that re-classifies the term from an
    open tier to a direct identifier has invalidated the approval without touching a single
    character the content hash covers.

    So the binding is both halves, joined and hashed:

        blake2b( content_hash | governance_code )

    A term that is re-defined moves the first half; a term that is re-classified moves the
    second; a term that is deleted resolves to nothing at all and never reaches this
    function. All three stop a stale approval, which is the whole requirement.

    Hashed rather than concatenated in the clear because `governance_code` is the caller's
    own vocabulary -- an arbitrary-length string that may contain any character, including
    the separator -- so a readable composite would be ambiguous exactly where two codes
    differ only by where the separator falls. A digest has no such seam.
    """
    material = f"{entry.content_hash}|{entry.governance_code or ''}"
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()


# =============================================================================
# THE PORT
# =============================================================================


@runtime_checkable
class FeedbackConsumer(Protocol):
    """
    Protocol for something that reads reviewer verdicts and may answer for a field.

    Example usage:
        consumer = ApprovedPairBypass.from_trail("feedback.jsonl")
        matcher = NexusMatcher(..., feedback_consumer=consumer)
        # matcher.load_dictionary(...) calls consumer.bind(...) for you.

    `runtime_checkable`, so a deployment holding an object that already answers these two
    questions can be recognised rather than wrapped. Both members are methods, so
    `issubclass` works here as well as `isinstance` -- unlike `EntryLookup`, which carries
    a property.
    """

    def bind(self, entries: EntryLookup) -> None:
        """
        Re-read every held verdict against the glossary now in force.

        Called by the matcher after EVERY successful index, including the first, and
        called before any field is matched. It is the invalidation contract: an
        implementation that does nothing here is promising that its answers do not depend
        on the glossary, and almost no useful implementation can promise that.

        Args:
            entries: Exact resolution over the dictionary that is now loaded. An id that
                resolves to None is a term this deployment no longer carries.
        """
        ...

    def approved_pair(self, field: SchemaField) -> ApprovedPair | None:
        """
        The standing human answer for this field, or None.

        None is by far the common case and must stay cheap: this runs once per field on
        the match path, so an implementation that talks to a network here has moved a
        per-field round trip into a batch that was built to avoid exactly that.

        Args:
            field: The field about to be matched, whole -- so an implementation can key on
                its parent as well as its name.

        Returns:
            The pair, or None for "no opinion; match it normally". Never raises to mean
            "no opinion": a consumer that cannot answer must not take matching down.
        """
        ...


class BaseFeedbackConsumer(ABC):
    """
    Base class for a consumer that has no glossary-dependent state.

    Supplies a `bind` that does nothing, which is the correct implementation for exactly
    one kind of consumer: one whose answers cannot go stale because it holds none. Anything
    that remembers an entry id MUST override it -- see `approval_binding` for what goes
    wrong when it does not.
    """

    def bind(self, entries: EntryLookup) -> None:
        """Re-read held verdicts against the current glossary. Nothing held, nothing to do."""

    @abstractmethod
    def approved_pair(self, field: SchemaField) -> ApprovedPair | None:
        """The standing human answer for this field, or None."""


class NullFeedbackConsumer(BaseFeedbackConsumer):
    """
    The consumer that consumes nothing. The shipped default's behaviour, made explicit.

    This exists so "an append-only audit trail is a legitimate default" is a POSITION the
    architecture states rather than a gap it happens to have. Attaching this to a matcher
    is indistinguishable from attaching nothing -- which is the property the test suite
    uses it to assert, and which is why it is in the port module beside the protocol rather
    than hidden in an application package a deployment would have to find.
    """

    __slots__ = ()

    def approved_pair(self, field: SchemaField) -> ApprovedPair | None:
        """Never has an opinion. Every field is matched by retrieval."""
        return None
