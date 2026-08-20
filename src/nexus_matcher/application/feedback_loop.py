"""
nexus_matcher.application.feedback_loop | Layer: APPLICATION
The reference consumer of the reviewer-verdict trail: an approved-pair bypass.

## Relationships
# DEPENDS_ON → domain/ports/review_feedback :: FeedbackConsumer, ReviewedVerdict, ApprovedPair
# DEPENDS_ON → domain/ports/entry_lookup :: EntryLookup, how the glossary is re-read
# DEPENDS_ON → application/use_cases/match_schema :: field_result_key, the field identity
# USED_BY    → external :: a deployment that opts in; nothing in this package constructs one

## Attributes
# Security: holds the caller's own field names and glossary ids; interprets neither
# Performance: `approved_pair` is one dict lookup; `bind` is one lookup per held verdict
# Reliability: a malformed record is refused at read time, never silently at match time

## OFF BY DEFAULT, and that is the whole design

Nothing in this library builds one of these. `create_app()` does not, `from_config()` does
not, and `NexusMatcher()` takes `feedback_consumer=None`. A deployment that wants the
bypass constructs it, hands it over, and owns the consequences -- and the consequences are
real, which is why they are enumerated in `ApprovedPairBypass` rather than implied.

## What it buys, stated so it cannot be overstated

Precision on a seen pair becomes 100% BY CONSTRUCTION, because the answer is a human's and
the matcher is not consulted. That is a tautology, not a measurement, and it must never be
reported as an accuracy improvement to retrieval: a run over a corpus whose verdicts you
have just replayed measures how many pairs you replayed, and nothing whatsoever about the
matcher. The honest quantity is COVERAGE -- what share of the fields in a run already have
a human answer -- and the honest claim is about re-deciding, not about deciding better.

`bypass_report()` returns the counts that support that claim and no others.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus_matcher.application.use_cases.match_schema import field_result_key
from nexus_matcher.domain.models.entities import SchemaField
from nexus_matcher.domain.ports.entry_lookup import EntryLookup
from nexus_matcher.domain.ports.review_feedback import (
    ApprovedPair,
    ReviewedVerdict,
    ReviewVerdict,
    approval_binding,
)

# The keys `presentation/api/feedback.py` writes. Named here rather than imported, and the
# reason is a layering one: the trail is a FILE FORMAT, and a file written by a v2.1 server
# is read by a v2.2 consumer whether or not the two agree about a Python module. Pinned
# against the recorder's own tuple by tests/unit/application/test_feedback_loop.py, so a
# rename over there is a red test here rather than a reader that silently stops finding a
# key it needs.
_TRAIL_FIELD = "field"
_TRAIL_CHOSEN = "chosenGovernanceId"
_TRAIL_SUGGESTED = "suggestedGovernanceId"
_TRAIL_WAS_CORRECT = "wasCorrect"
_TRAIL_VERDICT = "verdict"
_TRAIL_REVIEWER = "reviewer"
_TRAIL_RECEIVED_AT = "receivedAt"


# =============================================================================
# READING THE TRAIL
# =============================================================================


def verdict_from_record(record: Mapping[str, Any]) -> ReviewedVerdict:
    """
    One stored audit line, read into the domain vocabulary.

    ## The widening, and what reading a pre-widening line means

    A record carrying `verdict` says what the reviewer did. A record carrying only
    `wasCorrect` was written before the vocabulary could say it, and the two cases below
    are NOT symmetric:

      * `wasCorrect: true` -- unambiguous. The matcher's suggestion was accepted, so the
        verdict is APPROVED and the chosen id is the id that was accepted.
      * `wasCorrect: false` -- AMBIGUOUS, and permanently so. The reviewer chose the id on
        the record, and whether that id had been proposed is not recoverable from anything
        stored. It reads as `UNSPECIFIED`: still a choice, still usable as a bypass,
        useless for the question "is our problem ranking or recall?".

    That asymmetry is the loss WC-11 describes, in the one place it can be seen: half of
    the pre-widening records can be read exactly, and the half that carry the most
    information cannot.

    ## Why `receivedAt` and not `ts`

    `ts` is the reviewer's own clock, stored verbatim and never parsed, precisely so an
    unusual format cannot cost a verdict. That makes it unusable for ordering, and ordering
    is what decides whose verdict is in force when one field is reviewed twice.

    Raises:
        ValueError: naming the record and what is wrong with it. A trail line that cannot
            be read is refused at READ time rather than skipped, because a bypass silently
            built from 4,000 of 5,000 records is a bypass with 1,000 fields whose human
            answer is quietly not being applied.
    """
    field_key = record.get(_TRAIL_FIELD)
    if not isinstance(field_key, str) or not field_key:
        raise ValueError(f"feedback record has no usable {_TRAIL_FIELD!r}: {record!r}")

    chosen = record.get(_TRAIL_CHOSEN) or ""
    if not isinstance(chosen, str):
        raise ValueError(f"{_TRAIL_CHOSEN!r} on {field_key!r} is not a string: {chosen!r}")

    raw_verdict = record.get(_TRAIL_VERDICT)
    if raw_verdict is None:
        was_correct = record.get(_TRAIL_WAS_CORRECT)
        if not isinstance(was_correct, bool):
            raise ValueError(
                f"feedback record for {field_key!r} carries neither {_TRAIL_VERDICT!r} nor "
                f"a boolean {_TRAIL_WAS_CORRECT!r}, so nothing in it says what the "
                f"reviewer decided."
            )
        verdict = ReviewVerdict.APPROVED if was_correct else ReviewVerdict.UNSPECIFIED
    else:
        try:
            verdict = ReviewVerdict(raw_verdict)
        except ValueError as exc:
            known = ", ".join(v.value for v in ReviewVerdict)
            raise ValueError(
                f"unknown {_TRAIL_VERDICT!r} {raw_verdict!r} on {field_key!r}. This build "
                f"knows: {known}. A verdict nobody here can interpret must not be guessed "
                f"at -- it would be applied as whichever value happened to be closest."
            ) from exc

    suggested = record.get(_TRAIL_SUGGESTED)
    return ReviewedVerdict(
        field_key=field_key,
        verdict=verdict,
        chosen_entry_id="" if verdict is ReviewVerdict.REJECTED else chosen,
        suggested_entry_id=suggested if isinstance(suggested, str) else None,
        reviewer=str(record.get(_TRAIL_REVIEWER) or ""),
        recorded_at=str(record.get(_TRAIL_RECEIVED_AT) or ""),
    )


def read_feedback_trail(source: str | Path | Iterable[str]) -> tuple[ReviewedVerdict, ...]:
    """
    Every verdict in an append-only JSONL trail, in the order it was written.

    Order is preserved and not sorted. The file is append-only by construction, so its own
    order IS `receivedAt` order, and re-sorting would only introduce a second opinion about
    something the file already settles.

    Args:
        source: A path to the trail, or any iterable of lines (so a caller can feed a
            stream, a test fixture, or a file already open).

    Returns:
        The verdicts, oldest first.

    Raises:
        ValueError: naming the LINE NUMBER as well as the defect. A trail is evidence, and
            "something in this file is malformed" is not an actionable thing to tell an
            operator holding five thousand lines.
    """
    if isinstance(source, (str, Path)):
        lines: Iterable[str] = Path(source).read_text(encoding="utf-8").splitlines()
    else:
        lines = source

    out: list[ReviewedVerdict] = []
    for number, line in enumerate(lines, 1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"feedback trail line {number} is not JSON: {exc}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"feedback trail line {number} is not a JSON object")
        try:
            out.append(verdict_from_record(record))
        except ValueError as exc:
            raise ValueError(f"feedback trail line {number}: {exc}") from exc
    return tuple(out)


# =============================================================================
# THE REPORT
# =============================================================================


@dataclass(frozen=True, slots=True)
class BypassReport:
    """
    What this consumer is doing, in numbers an operator can act on.

    `standing` is the only one that describes the bypass in force. The other four describe
    verdicts that are NOT being applied, and each names a different thing to go and fix:

      * `revoked` -- a reviewer said nothing governs this field. Working as intended.
      * `unresolved` -- the chosen term has never been found in any glossary this consumer
        has been bound to. Almost always the wrong glossary is loaded, and the pair is
        still being retried on every bind.
      * `retired` -- the pair DID bind, and the term has since gone missing. Either it was
        deleted or the wrong glossary is now loaded, and those need different people.
      * `invalidated` -- the term is present but has been re-defined or re-classified since
        the approval was accepted. A human has to look again.

    `ambiguous` counts the standing pairs whose verdict is `UNSPECIFIED` -- pre-widening
    records that are being applied as bypasses but cannot answer whether retrieval's
    problem was ranking or recall. It is the size of WC-11's loss in this deployment.

    ## STATE AND EVENT ARE DIFFERENT KINDS OF NUMBER, AND MIXING THEM BREAKS THE IDENTITY

    `unresolved` is a STATE as of the last bind: those pairs are not applying right now,
    they are still held, and a later load can put them back. `retired` and `invalidated`
    are EVENT COUNTS since the consumer was attached, because both retire a pair FOR GOOD
    -- the pair is dropped from what this consumer holds, so a number recomputed from what
    it holds can no longer describe it, and would report zero for an approval that will
    never apply again.

    `retired` exists because that is exactly what went wrong. A deleted term used to be
    counted in `unresolved` alone: correct on the bind that dropped it, and gone from every
    later bind, because `unresolved` is recomputed from the pairs still held and the pair
    was no longer one of them. From the second bind onward the identity below was false and
    a verdict had vanished from the operator's only view of whether their review history
    still applies -- which is precisely the shortfall the identity is claimed to make
    visible. Deletion now gets the treatment `invalidated` always had.

    `unresolved` and `retired` are DISJOINT by construction: a pair that fails to resolve
    is one or the other according to whether it has ever bound, never both.

        standing + unresolved + retired + invalidated == verdicts

    at every moment, which is what makes a shortfall visible instead of hidden by a
    denominator that shrank to match. `revoked` is outside the identity on purpose: a
    REJECTED verdict removes the field from `verdicts` at construction, so it was never in
    the denominator to be accounted for.
    """

    verdicts: int
    standing: int
    revoked: int
    unresolved: int
    invalidated: int
    ambiguous: int
    # Appended rather than inserted beside `unresolved`, where it belongs by meaning: the
    # six above are positional on a frozen dataclass, and moving one would silently
    # re-point every positional construction. REQUIRED rather than defaulted to 0, because
    # a default here would let a future edit to `bypass_report` drop the count and report a
    # conservation identity that holds by omission -- which is the shape of the defect this
    # member was added to fix.
    retired: int


# =============================================================================
# THE CONSUMER
# =============================================================================


@dataclass(frozen=True, slots=True)
class _Standing:
    """One field's currently-binding verdict, before any glossary has been consulted."""

    verdict: ReviewedVerdict


class ApprovedPairBypass:
    """
    Once a reviewer approves (field -> entry), that pair skips matching on later runs.

    Implements `domain.ports.review_feedback.FeedbackConsumer`.

    ## THE KEY IS THE FIELD IDENTITY THE RESPONSE WAS KEYED BY, AND NOTHING SHORTER

    The key is `field_result_key(field)` -- the flattened column name a flattened schema
    supplied, or the dotted `full_path` for every other parser. Both carry the parent.

    A bypass keyed on the LEAF NAME would be a correctness bug, not a tuning matter. A
    warehouse schema repeats leaf names heavily; the same leaf under a different parent is
    a different question with a different governed term, and a leaf-keyed bypass answers
    all N occurrences with whichever one a reviewer happened to see first. The failure is
    not an exception and not a miss -- it is N-1 confidently wrong classifications with no
    symptom at all. `tests/unit/application/test_feedback_loop.py` proves the composed key
    against the repeated-leaf construction in `benchmarks/synthetic`.

    THIS KEY IS ALSO THE ONLY ONE AVAILABLE, and that is worth saying plainly rather than
    presenting the choice as freer than it is: a stored verdict carries exactly one field
    identity, the string the match response was keyed by. Two things follow, and a
    deployment must decide about both:

      * A field with NO parent -- a root-level column, a flattened name with no prefix --
        keys on its bare name, and a verdict given about `customer_id` in one schema will
        apply to `customer_id` in another. That is usually what a steward means and
        occasionally is not.
      * The `entity` query signal is NOT in the key, because the trail cannot carry it. Two
        requests that name different parent records for the same column get the same
        bypass. Widening the recorded shape to carry `entity` is the fix; until it does,
        this is the documented bound.

    ## WHAT INVALIDATES A PAIR

    `bind` is called with the live glossary on every index. The FIRST bind accepts each
    resolved pair and remembers `approval_binding(entry)` -- the generation of the term the
    deployment is accepting responsibility for. Every later bind re-resolves:

      * id never yet present  -> kept and retried, counted as `unresolved`
      * id present before, gone now -> dropped, counted as `retired`
      * binding moved         -> dropped, counted as `invalidated`
      * binding unchanged     -> still standing

    `approval_binding` covers the definition AND the protection code, so a re-classification
    invalidates as surely as a re-definition; `DictionaryEntry.content_hash` alone would not,
    because it deliberately excludes governance.

    THE BOUND ON THAT GUARANTEE, stated because it is the part that would otherwise be
    discovered: a verdict recorded before this consumer existed carries no fingerprint of
    its own, so the first bind can only bind it to the glossary in force at that moment.
    Invalidation is therefore exact from the first bind onward and cannot reach backwards
    over a change that happened before it. A deployment that needs that must record the
    fingerprint with the verdict.

    ## LAST VERDICT WINS

    A field reviewed twice takes the later record, ordered by the trail's own append order.
    A REJECTED verdict revokes an earlier approval rather than being ignored -- a reviewer
    who goes back and says "actually, nothing governs this" must be able to switch a bypass
    off, and a consumer that only ever accumulated approvals would make an approval
    permanent.
    """

    __slots__ = (
        "_ambiguous",
        "_bound",
        "_ever_bound",
        "_invalidated",
        "_retired",
        "_reviewed_fields",
        "_revoked",
        "_standing",
        "_unresolved",
    )

    def __init__(self, verdicts: Sequence[ReviewedVerdict] = ()) -> None:
        """
        Fold a trail into one standing verdict per field.

        Nothing is resolved here: this object is inert until `bind` is called with a
        glossary, and `approved_pair` answers None until it has been. That is deliberate --
        a consumer that answered from remembered ids before it had seen the loaded
        dictionary would be answering about a glossary it has never read.
        """
        standing: dict[str, _Standing] = {}
        revoked = 0
        for verdict in verdicts:
            if verdict.verdict is ReviewVerdict.REJECTED:
                if standing.pop(verdict.field_key, None) is not None:
                    revoked += 1
                continue
            standing[verdict.field_key] = _Standing(verdict)

        self._standing: dict[str, _Standing] = standing
        self._reviewed_fields = len(standing)
        self._revoked = revoked
        self._bound: dict[str, ApprovedPair] = {}
        # Keys this consumer has accepted at least once. The difference between "has never
        # bound" and "bound and then went missing" is what decides whether a pair is
        # retried or retired; see `bind`.
        self._ever_bound: set[str] = set()
        self._unresolved = 0
        self._retired = 0
        self._invalidated = 0
        self._ambiguous = 0

    @classmethod
    def from_trail(cls, source: str | Path | Iterable[str]) -> ApprovedPairBypass:
        """Build one from an append-only JSONL trail. See `read_feedback_trail`."""
        return cls(read_feedback_trail(source))

    # -- the port ------------------------------------------------------------

    def bind(self, entries: EntryLookup) -> None:
        """
        Re-resolve every standing verdict against the glossary now loaded.

        Idempotent for an unchanged glossary and destructive for a changed one, which is
        the point: a pair that cannot be re-confirmed stops applying rather than being kept
        on the chance that the change was harmless.

        THREE OUTCOMES, AND ONLY ONE OF THEM IS PERMANENT.

          * The id resolves and its binding is unchanged (or this is the first time it has
            ever resolved): the pair stands, bound to the entry as it is NOW.
          * The id does not resolve and this pair has NEVER bound: it is kept and retried,
            and counted in `unresolved` -- a STATE, re-evaluated on every bind. It has
            never been accepted against anything, so dropping it would mean that loading
            the wrong glossary once destroys a whole review history.
          * The id does not resolve and this pair HAS bound before: it is retired
            permanently and counted in `retired` -- an EVENT, for the same reason
            `invalidated` is one. What a reviewer approved was the term as it was, and a
            term that comes back may not be the term that left. It is an EVENT rather than
            a state because the pair is dropped here: a count recomputed from the pairs
            still held would report it once and then forget it, which is how the report's
            conservation identity came to be false from the second bind onward.
          * The id resolves and its binding has MOVED since this consumer accepted it: the
            pair is retired permanently and counted in `invalidated` -- an EVENT, and the
            one thing here that no later load undoes. A term that was re-defined or
            re-classified needs a human to look again, not a retry.
        """
        keys = sorted(self._standing)
        resolved = entries.lookup_many(
            [self._standing[key].verdict.chosen_entry_id for key in keys]
        )

        bound: dict[str, ApprovedPair] = {}
        # Keys to remove from `_standing` after the walk: retired by deletion or by a
        # moved binding. Both are permanent; neither is ever put back.
        dropped: list[str] = []
        ambiguous = 0
        unresolved = 0
        for key, entry in zip(keys, resolved, strict=True):
            verdict = self._standing[key].verdict
            if entry is None:
                # Disjoint on purpose: a pair is counted in exactly one of the two, so the
                # report's identity cannot double-count a single missing term.
                if key in self._ever_bound:
                    self._retired += 1
                    dropped.append(key)
                else:
                    unresolved += 1
                continue
            binding = approval_binding(entry)
            previous = self._bound.get(key)
            if previous is not None and previous.binding != binding:
                self._invalidated += 1
                dropped.append(key)
                continue
            if verdict.verdict is ReviewVerdict.UNSPECIFIED:
                ambiguous += 1
            self._ever_bound.add(key)
            bound[key] = ApprovedPair(
                field_key=key,
                entry=entry,
                verdict=verdict.verdict,
                binding=binding,
                reviewer=verdict.reviewer,
                decided_at=verdict.recorded_at,
            )

        self._bound = bound
        self._ambiguous = ambiguous
        self._unresolved = unresolved
        for key in dropped:
            del self._standing[key]

    def approved_pair(self, field: SchemaField) -> ApprovedPair | None:
        """The standing human answer for this field, or None. One dict lookup."""
        return self._bound.get(field_result_key(field))

    # -- reporting -----------------------------------------------------------

    def bypass_report(self) -> BypassReport:
        """
        The counts. See `BypassReport` for what each one means.

        `verdicts` counts FIELDS carrying a standing verdict when this consumer was built,
        not lines in the file: a field reviewed six times contributes one, and a field
        whose last verdict was REJECTED contributes none. It is fixed at construction, so
        `standing + unresolved + retired + invalidated` reconciles against it and a
        shortfall is visible rather than hidden by a denominator that shrank to match.
        """
        return BypassReport(
            verdicts=self._reviewed_fields,
            standing=len(self._bound),
            revoked=self._revoked,
            unresolved=self._unresolved,
            invalidated=self._invalidated,
            ambiguous=self._ambiguous,
            retired=self._retired,
        )
