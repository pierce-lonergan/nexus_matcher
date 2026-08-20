"""
nexus_matcher.domain.ports.entry_lookup | Layer: DOMAIN
Port interface for resolving a dictionary entry by an id the caller already holds.

## Relationships
# USED_BY    → presentation/api/lookup :: the HTTP lookup plane depends on THIS, not on a
#              matcher's private entry map
# DEPENDS_ON → domain/models/entities :: DictionaryEntry, the thing resolved
# DEPENDS_ON → domain/governance :: GovernanceVocabulary, without which a code is a token

## Attributes
# Security: entries carry the caller's own glossary text; nothing here interprets it
# Performance: `lookup` is expected to be O(1); `lookup_many` must not be worse than N
# Reliability: absence is a RETURN VALUE, never an exception

## Why lookup is a port and not a mode of matching

Retrieval answers "which entry is this field probably about?". Lookup answers "what is the
entry with this id?". They share a dictionary and nothing else:

  * retrieval is approximate, ranked, scored and configurable; lookup is exact or absent;
  * retrieval costs an encoder call and a corpus scan; lookup costs a dict read;
  * retrieval's answer changes when the model, the weights or the thresholds change;
    lookup's answer changes only when the glossary does.

Routing a known id through the matcher is therefore both expensive and LESS accurate than
doing nothing clever: a ranker can put the wrong entry first, and a caller who named the
entry has no reason to accept that risk. A separate port is what stops the two from being
implemented by one object that quietly applies retrieval's configuration to lookup's
question.

## Absence is a return value

`lookup` returns `None` for an id the dictionary does not carry, and `lookup_many` returns
a list with `None` in that position. It does not raise, and it does not omit.

An implementation that dropped misses would force every caller to diff what came back
against what they asked for, and the failure mode when they forget is silent: a shorter
result that still looks like an answer. Positional completeness makes the count itself an
oracle -- `len(out) == len(ids)` -- which is the cheapest possible check and the one a
caller gets for free.

Raising is reserved for "this lookup cannot be performed at all" -- no dictionary, an
index that has gone away -- which is a different question from "that id is not here", and
an implementation that conflated them would tell a caller their glossary had lost a term
when the truth is that nothing was loaded.

## Why the vocabulary is on this port

A resolved entry carries `governance_code`, and a code without the vocabulary that defines
it is an uninterpretable token: nothing in this package ships a taxonomy, so `PC-7` means
whatever the caller's own vocabulary says it means and nothing otherwise. A lookup plane
that returned the code but not the vocabulary would hand back an answer its caller has to
go somewhere else to read -- and "somewhere else" is a second system that can disagree.

`GovernanceVocabulary.empty()` is the honest answer for an implementation with no
vocabulary configured. It resolves every code to `None` and every classification to the
open sentinel, which reads as "nothing is classified here" rather than as a missing field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from nexus_matcher.domain.governance import GovernanceVocabulary
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.shared.types.base import DocumentId

# =============================================================================
# ENTRY LOOKUP PROTOCOL
# =============================================================================


@runtime_checkable
class EntryLookup(Protocol):
    """
    Protocol for exact resolution of a dictionary id to its entry.

    `runtime_checkable` so a caller holding an object that already answers these questions
    can be recognised rather than wrapped. It carries a non-method member (`vocabulary`),
    so `isinstance` works and `issubclass` does not -- the standard limitation of a data
    protocol, and `isinstance` is the only form anything here needs.

    Example usage:
        lookup: EntryLookup = MappingEntryLookup(entries_by_id, vocabulary)
        entry = lookup.lookup("GBF-0001")          # the entry, or None
        entries = lookup.lookup_many(["A", "B"])   # [entry_or_None, entry_or_None]
    """

    @property
    def vocabulary(self) -> GovernanceVocabulary:
        """
        The vocabulary the `governance_code` on these entries is spelled in.

        `GovernanceVocabulary.empty()` when none is configured -- never None, so a caller
        never has to branch on the vocabulary's existence before reading a class.
        """
        ...

    def lookup(self, entry_id: DocumentId) -> DictionaryEntry | None:
        """
        The entry with this id, or None when the dictionary does not carry it.

        Args:
            entry_id: The id to resolve, exactly as the dictionary spells it.

        Returns:
            The entry, or None. Absence is an answer, not an error.
        """
        ...

    def lookup_many(self, entry_ids: Sequence[DocumentId]) -> list[DictionaryEntry | None]:
        """
        One answer per id, positionally aligned to `entry_ids`.

        Args:
            entry_ids: The ids to resolve, in the caller's own order.

        Returns:
            A list the same length as `entry_ids`, holding the entry or None at each
            position. Never shorter, never reordered, never de-duplicated -- the caller's
            list is the key to the caller's answer.
        """
        ...


# =============================================================================
# BASE IMPLEMENTATION
# =============================================================================


class BaseEntryLookup(ABC):
    """
    Base class supplying the batch form in terms of the single one.

    Provided so an implementation cannot accidentally give the two forms different
    answers, which is the failure a batch API invites: a `lookup_many` that normalises,
    trims or de-duplicates while `lookup` does not means a caller's answer depends on
    which call they made. Override `lookup_many` only to make it FASTER, never to make it
    different, and keep the positional contract when you do.
    """

    @property
    @abstractmethod
    def vocabulary(self) -> GovernanceVocabulary:
        """The vocabulary these entries' codes are spelled in."""

    @abstractmethod
    def lookup(self, entry_id: DocumentId) -> DictionaryEntry | None:
        """The entry with this id, or None."""

    def lookup_many(self, entry_ids: Sequence[DocumentId]) -> list[DictionaryEntry | None]:
        """One answer per id, in the order asked, by calling `lookup` for each."""
        return [self.lookup(entry_id) for entry_id in entry_ids]


class MappingEntryLookup(BaseEntryLookup):
    """
    An entry lookup over a plain mapping of id -> entry.

    This exists so the port has an implementation that owes nothing to the application or
    infrastructure layers. A port with exactly one possible implementation is a private
    method with extra ceremony; this one can be constructed from a loader's output, from a
    test fixture, or from a caller's own dict, with no matcher anywhere.

    The mapping is COPIED at construction. A lookup surface whose answers change under a
    caller because somebody else mutated the dict they passed is not exact resolution, and
    the copy is what makes "a hit is exact by construction" true of this object rather
    than of the dict it was built from.
    """

    __slots__ = ("_entries", "_vocabulary")

    def __init__(
        self,
        entries: Mapping[DocumentId, DictionaryEntry],
        vocabulary: GovernanceVocabulary | None = None,
    ) -> None:
        self._entries: dict[DocumentId, DictionaryEntry] = dict(entries)
        # `empty()` rather than None: every caller reading a class off a looked-up entry
        # would otherwise have to check first, and the one that forgets gets an
        # AttributeError at the point of classification.
        self._vocabulary = GovernanceVocabulary.empty() if vocabulary is None else vocabulary

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[DictionaryEntry],
        vocabulary: GovernanceVocabulary | None = None,
    ) -> MappingEntryLookup:
        """
        Build one from entries, keyed by `DictionaryEntry.id`.

        Raises on a duplicate id rather than letting the last one win. Two entries sharing
        an id means one of them is unreachable through this port forever, and which one
        depends on iteration order -- a glossary defect that would present as a term that
        "does not exist" while sitting in the file.
        """
        by_id: dict[DocumentId, DictionaryEntry] = {}
        for entry in entries:
            if entry.id in by_id:
                raise ValueError(
                    f"duplicate dictionary id {entry.id!r}: two entries cannot share one "
                    f"id, or a lookup would resolve to whichever was indexed last."
                )
            by_id[entry.id] = entry
        return cls(by_id, vocabulary)

    @property
    def vocabulary(self) -> GovernanceVocabulary:
        """The vocabulary these entries' codes are spelled in."""
        return self._vocabulary

    @property
    def entry_count(self) -> int:
        """How many entries this lookup can resolve."""
        return len(self._entries)

    def lookup(self, entry_id: DocumentId) -> DictionaryEntry | None:
        """The entry with this id, or None."""
        return self._entries.get(entry_id)
