"""
nexus_matcher.domain.governance | Layer: DOMAIN
The CALLER-SUPPLIED controlled vocabulary of protection classes.

## Relationships
# USED_BY    → application/ingest :: validates and attaches a code to DictionaryEntry
# USED_BY    → application/use_cases/match_schema :: resolves MatchResult.governance
# USED_BY    → domain/models/entities :: MatchResult carries a ProtectionClass

## Attributes
# Security: Holds one organisation's classification policy. Never ships one.
# Performance: Lookup is a dict hit on a normalised key; column resolution is memoised
# Reliability: A row whose stated tier contradicts its code is REFUSED, not indexed

## Why this ships empty

This library assigns governance classifications to schema fields. The vocabulary of
classifications is not ours to invent: it belongs to the organisation whose glossary is
being matched, it differs between organisations, and in most of them it is confidential.
So the library ships **no taxonomy at all**. The caller hands it a JSON file, and every
code this module will accept comes from that file.

That is not a compromise for confidentiality's sake, it is the correct design twice over:
a library with one organisation's codes baked in is useless to every other organisation,
and it is exactly the shape that leaks a customer's internal policy into a public repo.

## The derivation invariant -- the whole point of the module

A protection code IMPLIES its classification tier. The tier is derived, never free text.

A glossary row that carries both a code and a tier, where the tier is not the one the
code implies, is a **data defect**, not a preference to be honoured. Indexing it would let
a field inherit a tier its own code disowns -- which is the class of bug this library
exists to prevent (NM-0005: a field silently losing the classification it should have
inherited). `problems_with()` reports the contradiction and `load_entries()` refuses the
load unless the caller explicitly asks for the softer behaviour.

The catalog also wins over the row for the personal-information and direct-identifier
flags, for the same reason and with the same report.

## Unknown codes are rejected, never stored

A code the vocabulary does not define is not a class. Storing it would leave a field
carrying a label nobody defined, which reads as governance and is not. Unknown tokens are
reported by `problems_with()` and never reach `DictionaryEntry.governance_code`.

Legacy spellings are handled by DECLARING them: an alias maps an old token onto a current
code, and an alias mapping to JSON `null` declares a token to be noise that must be
dropped. Both are explicit, so "we quietly dropped something" is not a thing that can
happen unnoticed.

## The JSON the caller supplies

```json
{
  "open_classification": "Open",
  "aliases": {"MTR#": "METERID", "n/a": null},
  "classes": [
    {
      "code": "METERID",
      "name": "Meter Serial Identifier",
      "classification": "Sealed",
      "personal_information": true,
      "direct_identifier": true,
      "enhancement": "tokenise",
      "aliases": ["LEGACY-METER"]
    }
  ]
}
```

A bare list of class objects is also accepted, as is a mapping of `code -> attributes`.
`open_classification` names the tier an uncoded field sits at; it defaults to the neutral
sentinel `OPEN_CLASSIFICATION` below, which is deliberately not a word any real taxonomy
uses, so an unconfigured vocabulary cannot be mistaken for a configured one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

# The tier an uncoded field sits at when the caller's file does not name one.
#
# Deliberately a sentinel rather than a plausible tier name. Any real-sounding default
# would be this library inventing one organisation's policy, and would read in an audit as
# a decision somebody made rather than as a gap in configuration.
OPEN_CLASSIFICATION = "UNCLASSIFIED"

# The two governance columns a glossary can carry, normalised the same way
# `ingest.map_columns` normalises everything else (lowercased, non-alphanumerics stripped).
#
# THESE TUPLES ARE THE SINGLE DEFINITION, and `ingest.COLUMN_ALIASES` imports them rather
# than restating them. Two independent notions of "which column means what" is how the
# loader path once ended up rejecting files the ingest path read without complaint.
#
# They live HERE rather than in the reader, for two reasons. What a protection code is, and
# what a classification tier is, are DOMAIN concepts; the reader only wires the columns in.
# And `problems_with()` takes a raw row, so it has to resolve these columns itself -- with
# the tuples in the application layer it could only do that by importing upward, which is
# an architecture violation the packaging gate correctly refuses (a deferred import is
# still an edge; it just hides from `import nexus_matcher.domain`).
#
# `CLASSIFICATION_COLUMN_ALIASES` is the FREE-TEXT tier column, historically
# `COLUMN_ALIASES["protection_level"]` and moved here unchanged.
CODE_COLUMN_ALIASES: tuple[str, ...] = (
    "governancecode",
    "protectionclass",
    "protectionclasscode",
    "protectioncode",
    "classificationcode",
    "dataclasscode",
    "governanceclass",
)
CLASSIFICATION_COLUMN_ALIASES: tuple[str, ...] = (
    "protectionlevel",
    "classification",
    "sensitivity",
    "governance",
    "governancestatus",
    "confidentiality",
    "pii",
    "securityclassification",
)

# The personal-information and direct-identifier columns are resolved HERE and are
# deliberately NOT added to `ingest.COLUMN_ALIASES`.
#
# Anything in that table is treated as a mapped column and is therefore EXCLUDED from
# `DictionaryEntry.source_metadata`. These two are advisory -- the catalog wins over the
# row for both -- so mapping them would silently delete the caller's own columns from the
# entry in exchange for values we then ignore. They are read for validation and left
# where they were.
_PI_COLUMN_ALIASES: tuple[str, ...] = (
    "personalinformation",
    "personaldata",
    "ispersonaldata",
    "ispersonalinformation",
    "containspersonaldata",
)
_DIRECT_ID_COLUMN_ALIASES: tuple[str, ...] = (
    "directidentifier",
    "isdirectidentifier",
    "directlyidentifying",
    "directidentifierflag",
)

_TRUE_WORDS = frozenset({"true", "yes", "y", "1", "t"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "f"})


def _norm_key(name: Any) -> str:
    """Normalise a COLUMN name. Mirrors `ingest._norm_key`; kept local so the domain can
    resolve its own columns without importing the application layer."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _norm_code(value: Any) -> str:
    """
    Normalise a protection CODE for lookup.

    Case, spacing and punctuation vary wildly between a catalog file and the glossary
    rows that reference it -- "METERID", "meter-id" and "Meter Id" are one code written
    three ways, and treating them as three codes would reject perfectly good rows. Note
    the consequence: a token that is nothing but punctuation normalises to the empty
    string and is therefore ABSENT, not unknown.
    """
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _norm_tier(value: Any) -> str:
    """Normalise a classification TIER for comparison only. The declared spelling is what
    gets reported; this exists so "Sealed " and "sealed" are not a contradiction."""
    return " ".join(str(value).split()).casefold()


def _as_bool(value: Any) -> bool | None:
    """
    Parse a flag out of glossary data, or None when the row makes no claim.

    None is not False. A blank cell means the row says nothing about whether the field is
    personal information, and reporting that as a disagreement with the catalog would
    bury the real contradictions under noise from every sparsely-filled glossary.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return None


# =============================================================================
# PROTECTION CLASS
# =============================================================================


@dataclass(frozen=True)
class ProtectionClass:
    """
    One entry of the caller's controlled vocabulary.

    Attributes:
        code: The controlled token a glossary row carries, e.g. "METERID".
        name: The human label for the class.
        classification: The tier this code DERIVES. Never read from a glossary row --
            see the module docstring on the derivation invariant.
        personal_information: Whether fields in this class hold personal information.
        direct_identifier: Whether fields in this class identify a person on their own.
        enhancement: Optional handling instruction the caller attaches to the class
            (masking, tokenisation, a retention rule). Passed through untouched; this
            library never interprets it.
    """

    code: str
    name: str
    classification: str
    personal_information: bool
    direct_identifier: bool
    enhancement: str | None = None


class _GovernanceColumns(NamedTuple):
    """Which of a row's columns hold the code, the stated tier, and the two flags."""

    code: str | None
    classification: str | None
    personal_information: str | None
    direct_identifier: str | None


@lru_cache(maxsize=64)
def _governance_columns(columns: tuple[str, ...]) -> _GovernanceColumns:
    """
    Resolve a row's column names once per HEADER, not once per row.

    `problems_with()` takes a row, so a naive implementation re-derives the mapping for
    every row of a 30k-entry glossary. Memoising on the header tuple keeps the published
    signature while paying for the resolution once; `ingest.load_entries` already hoists
    exactly this kind of per-mapping work out of its row loop for the same reason.

    Resolution is done HERE, over the tuples above, rather than by calling
    `ingest.map_columns`. Those are the same tuples the loader uses -- it imports them from
    this module -- so the two cannot disagree about which column is which, and the domain
    keeps no edge into the application layer.

    The one thing this does not reproduce is `map_columns`' first-come "taken" rule, where
    an earlier field can claim a column. It cannot bite: the four alias sets involved are
    pairwise disjoint, and no field resolved before them shares an alias with any of them.
    `test_the_two_resolvers_agree` in tests/unit/application/test_governance_ingest.py
    pins that, so a future alias that broke it fails a test instead of quietly making this
    module and the loader read the same file differently.
    """
    normalised = {_norm_key(c): c for c in columns}

    def _first(aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            found = normalised.get(alias)
            if found is not None:
                return found
        return None

    return _GovernanceColumns(
        code=_first(CODE_COLUMN_ALIASES),
        classification=_first(CLASSIFICATION_COLUMN_ALIASES),
        personal_information=_first(_PI_COLUMN_ALIASES),
        direct_identifier=_first(_DIRECT_ID_COLUMN_ALIASES),
    )


# =============================================================================
# GOVERNANCE VOCABULARY
# =============================================================================


class GovernanceVocabulary:
    """
    The set of protection classes a caller will accept, and the rules for reading them.

    Build one with `from_json()` from the caller's own file, or `empty()` when no
    vocabulary is configured. There is no third way to obtain codes: nothing in this
    package defines one.
    """

    __slots__ = ("_aliases", "_by_code", "_open")

    def __init__(
        self,
        classes: Iterable[ProtectionClass] = (),
        open_classification: str = OPEN_CLASSIFICATION,
        aliases: Mapping[str, str | None] | None = None,
    ) -> None:
        by_code: dict[str, ProtectionClass] = {}
        for protection_class in classes:
            key = _norm_code(protection_class.code)
            if not key:
                raise ValueError("a protection class has an empty code")
            if key in by_code:
                raise ValueError(
                    f"duplicate protection code {protection_class.code!r} -- two classes "
                    f"cannot share a code, or a row's tier would depend on which one won"
                )
            by_code[key] = protection_class

        resolved: dict[str, str | None] = {}
        for token, target in (aliases or {}).items():
            key = _norm_code(token)
            if not key:
                raise ValueError(f"alias {token!r} normalises to nothing")
            if key in by_code:
                raise ValueError(
                    f"alias {token!r} collides with the declared code "
                    f"{by_code[key].code!r}; an alias cannot shadow a real class"
                )
            if target is None:
                resolved[key] = None
                continue
            target_key = _norm_code(target)
            if target_key not in by_code:
                raise ValueError(
                    f"alias {token!r} points at {target!r}, which is not a declared code"
                )
            resolved[key] = target_key

        if not str(open_classification).strip():
            raise ValueError("open_classification cannot be blank")

        self._by_code = by_code
        self._aliases = resolved
        self._open = str(open_classification)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_json(
        cls, path_or_obj: str | Path | Mapping[str, Any] | Sequence[Any]
    ) -> GovernanceVocabulary:
        """
        Load a vocabulary from the caller's JSON file, or from an already-parsed object.

        Args:
            path_or_obj: A path to a JSON file, or a parsed list/mapping in any of the
                three accepted shapes (see the module docstring).

        Returns:
            A GovernanceVocabulary.

        Raises:
            FileNotFoundError: the path does not exist.
            ValueError: the document is malformed, a class is missing a required
                attribute, two classes share a code, or an alias points nowhere. A
                vocabulary that half-loads is worse than none: the codes it silently
                dropped become "unknown", and every row using them gets refused for a
                reason that is nowhere in the file the caller is reading.
        """
        if isinstance(path_or_obj, (str, Path)):
            path = Path(path_or_obj)
            if not path.is_file():
                raise FileNotFoundError(f"Governance vocabulary not found: {path}")
            try:
                document: Any = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        else:
            document = path_or_obj

        raw_classes, open_tier, raw_aliases = _split_document(document)

        classes: list[ProtectionClass] = []
        alias_table: dict[str, str | None] = {
            str(token): (None if target is None else str(target))
            for token, target in raw_aliases.items()
        }

        for index, raw in enumerate(raw_classes):
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"protection class #{index} is {type(raw).__name__}, not an object"
                )
            code = str(raw.get("code", "")).strip()
            if not code:
                raise ValueError(f"protection class #{index} has no 'code'")
            name = str(raw.get("name", "")).strip() or code
            classification = str(raw.get("classification", "")).strip()
            if not classification:
                raise ValueError(
                    f"protection class {code!r} has no 'classification'. The tier is what "
                    f"a matched field inherits; a class without one cannot classify."
                )
            enhancement = raw.get("enhancement")
            classes.append(
                ProtectionClass(
                    code=code,
                    name=name,
                    classification=classification,
                    personal_information=_required_bool(raw, "personal_information", code),
                    direct_identifier=_required_bool(raw, "direct_identifier", code),
                    enhancement=None if enhancement is None else str(enhancement),
                )
            )
            for token in raw.get("aliases", ()) or ():
                alias_table[str(token)] = code

        return cls(classes=classes, open_classification=open_tier, aliases=alias_table)

    @classmethod
    def empty(cls) -> GovernanceVocabulary:
        """
        The vocabulary of an installation that has configured none.

        Every code is unknown to it, which is the honest answer rather than a permissive
        one: with nothing configured, no code has been defined by anybody, and
        `problems_with()` says so.
        """
        return cls()

    # -- lookup ---------------------------------------------------------------

    @property
    def codes(self) -> frozenset[str]:
        """The declared codes, in their declared spelling. Aliases are not codes."""
        return frozenset(c.code for c in self._by_code.values())

    def get(self, code: str | None) -> ProtectionClass | None:
        """
        The class a code names, or None when it names nothing.

        None covers three distinct situations on purpose -- no code, a code declared as
        droppable noise, and a code nobody defined -- because all three mean "this field
        inherits no class". Use `problems_with()` to tell them apart; that is the
        validator, this is the accessor.
        """
        _, protection_class = self._lookup(code)
        return protection_class

    def classification_for(self, code: str | None) -> str:
        """
        The tier a code DERIVES, or the open tier when it derives none.

        This is the authority on a field's tier. A tier written next to a code in a
        glossary is not consulted here and never overrides this -- see the derivation
        invariant in the module docstring.

        NOT a validator: an unknown code returns the open tier, which is correct as an
        answer ("nothing classified this") and wrong as a safety check. Unknown codes are
        refused at load time by `problems_with()` precisely so they cannot reach here.
        """
        protection_class = self.get(code)
        return protection_class.classification if protection_class is not None else self._open

    def _lookup(self, code: str | None) -> tuple[str, ProtectionClass | None]:
        """(status, class) where status is absent | dropped | known | unknown."""
        key = _norm_code(code) if code is not None else ""
        if not key:
            return "absent", None
        found = self._by_code.get(key)
        if found is not None:
            return "known", found
        if key in self._aliases:
            target = self._aliases[key]
            if target is None:
                return "dropped", None
            return "known", self._by_code[target]
        return "unknown", None

    # -- validation -----------------------------------------------------------

    def problems_with(self, row: Mapping[str, Any]) -> list[str]:
        """
        Everything wrong with one glossary row's governance. `[]` means the row is valid.

        Three families, all of them the same underlying failure -- a row asserting
        something about governance that the vocabulary does not support:

        1. **An undefined code.** Rejected rather than stored, because a stored code
           nobody defined looks like governance and is not.
        2. **A contradicted tier.** The row states a classification that its own code
           does not derive. This is the derivation invariant and the reason this method
           exists; see the module docstring.
        3. **A contradicted flag.** The row states a personal-information or
           direct-identifier value the catalog disagrees with. The catalog wins; the
           disagreement is reported so somebody fixes the source.

        A blank cell is not a claim, so it is never a problem. A row with a tier but no
        code is not a problem either: plenty of glossaries classify in prose and have
        adopted no controlled vocabulary, and this method is not the place to insist
        they do.

        Args:
            row: A raw glossary row, keyed by its own column names.

        Returns:
            Human-readable problems, in a stable order. Each names the row's value AND
            the vocabulary's, because a message that reports only "invalid" sends the
            reader back to two files to work out which side is wrong.
        """
        columns = _governance_columns(tuple(str(k) for k in row))
        if columns.code is None:
            return []

        raw_code = row.get(columns.code)
        status, protection_class = self._lookup(raw_code)

        if status == "unknown":
            known = ", ".join(sorted(self.codes)) or "none -- no vocabulary is configured"
            return [
                f"protection code {str(raw_code).strip()!r} is not in the configured "
                f"vocabulary, so it is rejected rather than stored (declared codes: {known})"
            ]

        if protection_class is None:
            # absent (no code in this row) or dropped (a token the caller declared to be
            # noise). Neither is a defect, and neither leaves the row with a class.
            return []

        problems: list[str] = []

        if columns.classification is not None:
            stated = str(row.get(columns.classification) or "").strip()
            if stated and _norm_tier(stated) != _norm_tier(protection_class.classification):
                problems.append(
                    f"row states classification {stated!r} but code "
                    f"{protection_class.code!r} derives "
                    f"{protection_class.classification!r} -- the tier is derived from the "
                    f"code, so this row contradicts itself"
                )

        for column, catalog_value, label in (
            (
                columns.personal_information,
                protection_class.personal_information,
                "personal_information",
            ),
            (columns.direct_identifier, protection_class.direct_identifier, "direct_identifier"),
        ):
            if column is None:
                continue
            stated_flag = _as_bool(row.get(column))
            if stated_flag is not None and stated_flag != catalog_value:
                problems.append(
                    f"row states {label}={stated_flag} but code "
                    f"{protection_class.code!r} declares {catalog_value} -- the catalog "
                    f"wins and the row is wrong"
                )

        return problems


# =============================================================================
# DOCUMENT PARSING
# =============================================================================


def _split_document(
    document: Any,
) -> tuple[Sequence[Any], str, Mapping[str, Any]]:
    """Pull (classes, open tier, aliases) out of any of the three accepted shapes."""
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        return document, OPEN_CLASSIFICATION, {}

    if not isinstance(document, Mapping):
        raise ValueError(
            f"a governance vocabulary must be a JSON object or array, got {type(document).__name__}"
        )

    open_tier = str(document.get("open_classification") or OPEN_CLASSIFICATION)
    aliases = document.get("aliases") or {}
    if not isinstance(aliases, Mapping):
        raise ValueError("'aliases' must be an object mapping a legacy token to a code or null")

    for key in ("classes", "protection_classes"):
        candidate = document.get(key)
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return candidate, open_tier, aliases

    # A bare `{code: {...}}` mapping. Recognised only when EVERY value is an object, so a
    # document that meant to have a "classes" key and misspelled it fails loudly instead
    # of being read as a catalog of one class called "clases".
    reserved = {"open_classification", "aliases", "classes", "protection_classes"}
    body = {k: v for k, v in document.items() if k not in reserved}
    if body and all(isinstance(v, Mapping) for v in body.values()):
        return [{"code": code, **dict(attrs)} for code, attrs in body.items()], open_tier, aliases

    raise ValueError(
        "no protection classes found. Supply them as a 'classes' array, as a bare array, "
        "or as a mapping of code -> attributes."
    )


def _required_bool(raw: Mapping[str, Any], key: str, code: str) -> bool:
    """
    Read a catalog flag, refusing to guess.

    Tolerated in a glossary ROW (people type "Y"), refused in the CATALOG: the catalog is
    a file the caller authored to define policy, and a missing or unreadable flag there
    would silently become False -- i.e. "this class is not personal information", the
    permissive answer, asserted by nobody.
    """
    if key not in raw:
        raise ValueError(
            f"protection class {code!r} does not declare {key!r}. It must be true or "
            f"false; defaulting it would assert the permissive answer on the caller's "
            f"behalf."
        )
    parsed = _as_bool(raw[key])
    if parsed is None:
        raise ValueError(f"protection class {code!r} has an unreadable {key!r}: {raw[key]!r}")
    return parsed
