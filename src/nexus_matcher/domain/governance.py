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

One token means one thing. A token declared twice pointing at two different targets --
whether by two classes, by two spellings that normalise alike, or by a class overriding a
top-level `null` -- REFUSES the load. It used to resolve to whichever declaration came
last, which is a coin toss deciding whether a field inherits a restricted tier or an open
one. Restating the same mapping is not a conflict and stays legal.

## The JSON the caller supplies

```json
{
  "open_classification": "Open",
  "tiers_most_open_first": ["Open", "Guarded", "Sealed"],
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

## `tiers_most_open_first` -- the caller's own ordering, checked against itself

Tiers are strings to this library and it will never rank them. `tiers_most_open_first` is
OPTIONAL: a vocabulary that omits it works exactly as before, and nothing here invents an
order for one that does. What the key buys, when it is present, is that the ordering and
the classes have to AGREE -- a tier some class derives, or the declared open tier, that is
missing from the list refuses the load and the message names both sides.

Checking it is not this library forming an opinion about a taxonomy. The list is the
caller's, the tiers in it are the caller's, and the only thing compared is the caller's
file against itself. The alternative is what shipped: the key sat in the example pack and
was read by NO code, so an adopter copying that file inherited a no-op and a reasonable
belief that something enforced it.

Declaring a tier the classes do not use is fine and stays legal -- a policy ladder is
allowed to have rungs this particular vocabulary does not reach. The reverse is the defect.
"""

from __future__ import annotations

import json
import unicodedata
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
# `COLUMN_ALIASES["protection_level"]` and moved here almost unchanged -- see the note on
# "pii" below for the one member that had to leave.
CODE_COLUMN_ALIASES: tuple[str, ...] = (
    "governancecode",
    "protectionclass",
    "protectionclasscode",
    "protectioncode",
    "classificationcode",
    "dataclasscode",
    "governanceclass",
)
# "pii" is deliberately NOT in this tuple, and used to be. It came over with the rest of
# `COLUMN_ALIASES["protection_level"]`, where it was harmless: as a free-text tier alias, a
# column named "PII" was just another place somebody might write a tier. Under the
# derivation invariant the same membership asserts something else entirely -- that a row
# under a `PII` column claims its TIER is "Yes" -- so `term,definition,governance_code,PII`
# refused every coded row with a contradiction the row had never stated, and whether the
# load survived depended on whether some other tier column happened to be there to mask it.
#
# The silent half is why it had to move rather than just be dropped. When an earlier tier
# alias WAS present, "pii" lost the tier race and was then read by nothing, so a row
# claiming personal information against a class declaring `personal_information: false`
# reported nothing at all -- while `personal_data`, `is_personal_data` and
# `contains_personal_data` all reported it. It is a flag column; it now lives with the
# other flag columns below.
CLASSIFICATION_COLUMN_ALIASES: tuple[str, ...] = (
    "protectionlevel",
    "classification",
    "sensitivity",
    "governance",
    "governancestatus",
    "confidentiality",
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
    "pii",
    "ispii",
)
_DIRECT_ID_COLUMN_ALIASES: tuple[str, ...] = (
    "directidentifier",
    "isdirectidentifier",
    "directlyidentifying",
    "directidentifierflag",
)

# Every column name any of the four resolvers above can claim, behind the memoised
# predicate `_is_governance_column` below.
#
# `problems_with()` keys its column-resolution cache on the subset of a row's keys that
# passes that predicate, rather than on the whole row. CSV and Excel hand every row the same
# key tuple, so the whole-row key worked and hid the problem; JSON, JSONL and
# iterable-of-dicts sources hand each row back verbatim, so a sparse exporter produces one
# cache entry per COMBINATION OF PRESENT COLUMNS, and twelve optional columns is 4096 of
# them. The LRU is 64 wide, so one shape past that every lookup evicts the entry it is
# about to need and the hit rate goes to zero -- a cliff, not a slope.
#
# Measured over 30,000 rows, whole-row key -> filtered key (median of 5): 1 shape
# 34.2 -> 40.4 ms, 64 shapes 36.5 -> 44.0, 65 shapes 137.9 -> 44.5, 256 shapes
# 147.5 -> 45.6, 4096 shapes 170.3 -> 48.2.
#
# Filtering is NOT free, and saying so is the point: it costs ~6 ms per 30k rows on the
# uniform CSV shape, because it looks at every key of every row instead of hashing the
# tuple whole. That is the trade -- a flat ~18% on the path that was already fine, to
# delete a 3.5x cliff on the path that is not, inside an offline load that takes ~45 s.
_ALL_GOVERNANCE_ALIASES: frozenset[str] = frozenset(
    CODE_COLUMN_ALIASES
    + CLASSIFICATION_COLUMN_ALIASES
    + _PI_COLUMN_ALIASES
    + _DIRECT_ID_COLUMN_ALIASES
)

_TRUE_WORDS = frozenset({"true", "yes", "y", "1", "t"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "f"})


@lru_cache(maxsize=4096)
def _norm_key_text(name: str) -> str:
    """The memoised core of `_norm_key`, and of `_is_governance_column` below.

    Keyed on a `str` rather than on the caller's object, because `True` and `1` are equal
    and hash equal -- an object-keyed cache would answer `'1'` for a column literally named
    `True`."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _norm_key(name: Any) -> str:
    """
    Normalise a COLUMN name. Mirrors `ingest._norm_key`; kept local so the domain can
    resolve its own columns without importing the application layer.

    Memoised, because column names repeat massively -- a 30k-row glossary carries the same
    dozen of them 30,000 times.

    Deliberately NOT Unicode-normalised, unlike `_norm_code` below. This function has to
    agree character for character with `ingest._norm_key` or the validator and the loader
    read the same file differently, and `test_the_two_resolvers_agree` exists to catch the
    day one of them changes without the other.
    """
    return _norm_key_text(name if isinstance(name, str) else str(name))


@lru_cache(maxsize=4096)
def _is_governance_column(name: str) -> bool:
    """
    Whether a column is one of the four this module reads, as ONE memoised lookup.

    `problems_with()` runs this over every key of every row to build its cache key, so it is
    the hottest thing in the module -- 450,000 calls on a 30k-row glossary with fifteen
    columns. Folding the normalisation and the membership test into a single cached
    predicate rather than doing `_norm_key(k) in _ALL_GOVERNANCE_ALIASES` per key was
    measured at roughly half the cost of the two-step form.
    """
    return _norm_key_text(name) in _ALL_GOVERNANCE_ALIASES


def _norm_code(value: Any) -> str:
    """
    Normalise a protection CODE for lookup.

    Case, spacing and punctuation vary wildly between a catalog file and the glossary
    rows that reference it -- "METERID", "meter-id" and "Meter Id" are one code written
    three ways, and treating them as three codes would reject perfectly good rows. Note
    the consequence: a token that is nothing but punctuation normalises to the empty
    string and is therefore ABSENT, not unknown.

    NFC first, because a combining accent is not alphanumeric and this function strips it.
    An accented code therefore had two normalised forms -- the composed one an editor
    types and the decomposed one macOS's filesystem and several exporters emit -- which
    look identical everywhere a human reads them. Reproduced: a catalog declaring both
    loaded as TWO classes with two different tiers, and `codes` showed the same word twice.
    Which tier a row inherited then depended on which byte sequence its exporter chose.
    """
    text = value if isinstance(value, str) else str(value)
    return "".join(ch for ch in unicodedata.normalize("NFC", text).upper() if ch.isalnum())


def _norm_tier(value: Any) -> str:
    """
    Normalise a classification TIER for comparison only. The declared spelling is what
    gets reported; this exists so "Sealed " and "sealed" are not a contradiction.

    NFC for the same reason as `_norm_code`, and it bites harder here: `casefold()` does
    not normalise, so a decomposed accent in the row against a composed one in the catalog
    reads as a row contradicting itself, and at the default `governance_strict` that
    REFUSES the entire glossary over two words that render identically.
    """
    text = value if isinstance(value, str) else str(value)
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


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
    Resolve a row's GOVERNANCE column names once per shape, not once per row.

    `problems_with()` takes a row, so a naive implementation re-derives the mapping for
    every row of a 30k-entry glossary. Memoising keeps the published signature while paying
    for the resolution once; `ingest.load_entries` already hoists exactly this kind of
    per-mapping work out of its row loop for the same reason.

    The caller filters the row's keys through `_is_governance_column` before calling, so the
    key is the governance columns rather than the whole row -- see `_ALL_GOVERNANCE_ALIASES`
    for the numbers. A TUPLE and not a frozenset: two spellings can normalise to the same key
    and `normalised` below lets the last one win, so the order has to be the row's own.
    Measured across six processes under `PYTHONHASHSEED=random`, a frozenset key resolved
    the governance column three different ways, which would let the same file be refused on
    one run and accepted on the next.

    Do NOT hoist this to the header instead. `read_source` builds a header as the UNION of
    every row's keys, so on a merged export where some rows spell it "Governance Code" and
    others "Protection Class", a header-level resolver reads a column those rows do not
    have. Demonstrated on a two-row file with an undeclared code in the second spelling:
    refused today, and loaded SILENTLY under the hoist, carrying no code and not even a
    `governance_code_raw`. Per-row resolution is the only thing checking heterogeneous
    sources at all, and it is worth far more than the 0.4% of an index build it costs.

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

    __slots__ = ("_aliases", "_by_code", "_open", "_tiers")

    def __init__(
        self,
        classes: Iterable[ProtectionClass] = (),
        open_classification: str = OPEN_CLASSIFICATION,
        aliases: Mapping[str, str | None] | Iterable[tuple[str, str | None]] | None = None,
        tiers_most_open_first: Iterable[str] = (),
    ) -> None:
        by_code: dict[str, ProtectionClass] = {}
        for protection_class in classes:
            # The type check is here and not only in `from_json` because `from_json` is the
            # friendly message and this is the invariant nothing walks past. It accepted
            # `ProtectionClass(code=None)`: `_norm_code(None)` is 'NONE', so the empty-code
            # check below never fired, and a class asserting it had NO code became the class
            # every glossary cell spelling "none" inherited.
            for field, value in (
                ("code", protection_class.code),
                ("classification", protection_class.classification),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"a protection class has a {field} of {value!r}; it must be a "
                        f"non-blank string. Coercing it would invent a class nobody declared."
                    )
            key = _norm_code(protection_class.code)
            if not key:
                raise ValueError("a protection class has an empty code")
            if key in by_code:
                raise ValueError(
                    f"duplicate protection code {protection_class.code!r} -- two classes "
                    f"cannot share a code, or a row's tier would depend on which one won"
                )
            by_code[key] = protection_class

        # Aliases arrive as PAIRS, and every declaration in the file is one pair.
        #
        # `from_json` used to accumulate them into a dict keyed by the raw token, which
        # collapsed byte-identical duplicates before this loop could ever see them -- and
        # byte-identical is the likeliest duplicate there is, a maintainer copying a token
        # verbatim from one part of a file to another. A duplicate-check added here alone
        # was implemented and measured: it missed exactly those. Hence pairs.
        #
        # What the silence cost: two classes claiming one legacy spelling resolved
        # positionally to whichever was declared LAST, so a restricted class and an open
        # class both claiming "PAX-NM-OLD" loaded the passenger-name row as the open tier,
        # with no problems reported and a strict load not refused. That is NM-0005's harm
        # -- a field losing the classification it should have inherited -- reached through
        # the catalog instead of through the row. In the other direction, a `null` "drop
        # this token" declaration was silently promoted into a real class, contradicting
        # this module's own docstring promise that quiet dropping cannot happen unnoticed.
        declarations: Iterable[tuple[str, str | None]]
        if aliases is None:
            declarations = ()
        elif isinstance(aliases, Mapping):
            declarations = aliases.items()
        else:
            declarations = aliases

        resolved: dict[str, str | None] = {}
        as_written: dict[str, str] = {}
        for token, target in declarations:
            key = _norm_code(token)
            if not key:
                raise ValueError(f"alias {token!r} normalises to nothing")
            if key in by_code:
                raise ValueError(
                    f"alias {token!r} collides with the declared code "
                    f"{by_code[key].code!r}; an alias cannot shadow a real class"
                )
            target_key: str | None = None
            if target is not None:
                target_key = _norm_code(target)
                if target_key not in by_code:
                    raise ValueError(
                        f"alias {token!r} points at {target!r}, which is not a declared code"
                    )
            if key in resolved and resolved[key] != target_key:
                # Both spellings, as the caller wrote them: a normalised key is not
                # searchable in their own file. Restating the SAME mapping is allowed --
                # a rule that could not tell a restatement from a conflict would get worked
                # around by deleting the restatement that documented the intent. This also
                # handles null for free: None != 'CREWROSTER' conflicts, None == None does
                # not.
                raise ValueError(
                    f"alias {as_written[key]!r} is declared twice pointing at two "
                    f"different things -- {as_written[key]!r} -> "
                    f"{_alias_target_label(resolved[key], by_code)}, then {token!r} -> "
                    f"{_alias_target_label(target_key, by_code)}. One of them would "
                    f"silently win, so declare it once."
                )
            resolved[key] = target_key
            as_written.setdefault(key, token)

        if not str(open_classification).strip():
            raise ValueError("open_classification cannot be blank")

        self._by_code = by_code
        self._aliases = resolved
        self._open = str(open_classification)
        self._tiers = _checked_tier_order(tiers_most_open_first, by_code, self._open)

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

        raw_classes, open_tier, raw_aliases, tier_order = _split_document(document)

        classes: list[ProtectionClass] = []
        # A LIST of pairs, not a dict. See the alias loop in `__init__` for why the dict
        # was the bug rather than the storage.
        alias_pairs: list[tuple[str, str | None]] = []
        for token, target in raw_aliases.items():
            if not isinstance(token, str) or not token.strip():
                raise ValueError(
                    f"alias token {token!r} is not a non-blank string; the keys of "
                    f"'aliases' are the legacy tokens a glossary row can carry"
                )
            if target is not None and not isinstance(target, str):
                raise ValueError(
                    f"alias {token!r} points at {target!r}, which is "
                    f"{type(target).__name__}; it must be a declared code or null"
                )
            alias_pairs.append((token, target))

        for index, raw in enumerate(raw_classes):
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"protection class #{index} is {type(raw).__name__}, not an object"
                )
            code = _required_text(
                raw,
                "code",
                f"protection class #{index}",
                "The code is the token a glossary row carries; there is nothing to derive "
                "a tier from without one.",
            )
            subject = f"protection class {code!r}"
            classification = _required_text(
                raw,
                "classification",
                subject,
                "The tier is what a matched field inherits; a class without one cannot classify.",
            )
            classes.append(
                ProtectionClass(
                    code=code,
                    name=(_optional_text(raw, "name", subject) or "").strip() or code,
                    classification=classification,
                    personal_information=_required_bool(raw, "personal_information", code),
                    direct_identifier=_required_bool(raw, "direct_identifier", code),
                    enhancement=_optional_text(raw, "enhancement", subject),
                )
            )
            for token in _declared_list(raw, "aliases", subject, "alias"):
                alias_pairs.append((token, code))

        return cls(
            classes=classes,
            open_classification=open_tier,
            aliases=alias_pairs,
            tiers_most_open_first=tier_order,
        )

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

    @property
    def tiers_most_open_first(self) -> tuple[str, ...]:
        """
        The caller's declared tier ordering, most open first, or `()` when they declared
        none.

        A TUPLE and not a set, because the order is the entire content: it is the only
        thing in this module that can answer "is this field's tier more closed than that
        one's", and it answers it because the caller wrote it down, not because anything
        here ranked their taxonomy.

        Empty is a real answer and must stay distinguishable from a one-tier vocabulary.
        A consumer that cannot compare tiers should say so rather than fall back to
        alphabetical, which puts "CONFIDENTIAL" before "PUBLIC".
        """
        return self._tiers

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
        columns = _governance_columns(
            tuple(name for name in map(str, row) if _is_governance_column(name))
        )
        if columns.code is None:
            return []

        raw_code = row.get(columns.code)
        status, protection_class = self._lookup(raw_code)

        if status == "unknown":
            # The token and the SIZE of the vocabulary, never the vocabulary itself.
            #
            # `", ".join(sorted(self.codes))` used to sit here, in a branch that runs once
            # per defective ROW, building an identical list every time -- and `codes` is a
            # property that constructs a fresh frozenset per access. Measured over 30,000
            # unknown-code rows: 50 ms at 9 classes, 726 ms at 400, 804 ms at 800, against a
            # valid-code path that is flat at ~35 ms. Now flat at ~36 ms at every size.
            #
            # Caching the joined string does NOT fix it -- measured, still linear, because
            # interpolating a 4,000-character list into 30,000 f-strings copies it 30,000
            # times. Nor does caching touch the larger cost: under `governance_strict=False`,
            # the escape hatch this module's own refusal message recommends, every one of
            # those strings is RETAINED in `source_metadata`. Measured with tracemalloc at
            # 400 classes: 127.1 MB holding one distinct value, against 7.4 MB now.
            #
            # The declared list belongs in the load summary, which is built once. Keep the
            # exact phrase "no vocabulary is configured": two tests assert it by name,
            # because "configured nothing" and "configured permissively" are different
            # answers and the message is the only place a reader learns which one they have.
            declared = (
                f"{len(self._by_code)} code(s) are declared"
                if self._by_code
                else "no vocabulary is configured"
            )
            return [
                f"protection code {str(raw_code).strip()!r} is not in the configured "
                f"vocabulary, so it is rejected rather than stored ({declared})"
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
) -> tuple[Sequence[Any], str, Mapping[str, Any], list[str]]:
    """Pull (classes, open tier, aliases, tier order) out of any of the three shapes."""
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        return document, OPEN_CLASSIFICATION, {}, []

    if not isinstance(document, Mapping):
        raise ValueError(
            f"a governance vocabulary must be a JSON object or array, got {type(document).__name__}"
        )

    # `or OPEN_CLASSIFICATION` already handled a null correctly; only a dict or a list
    # slipped through, to be stringified into a Python repr and shipped as somebody's tier.
    # A blank string is still left to `__init__`, which refuses it by name rather than
    # quietly substituting the sentinel.
    declared_open = _optional_text(document, "open_classification", "the vocabulary document")
    open_tier = declared_open if declared_open else OPEN_CLASSIFICATION
    tier_order = _declared_list(
        document, "tiers_most_open_first", "the vocabulary document", "tier"
    )
    aliases = document.get("aliases") or {}
    if not isinstance(aliases, Mapping):
        raise ValueError("'aliases' must be an object mapping a legacy token to a code or null")

    for key in ("classes", "protection_classes"):
        candidate = document.get(key)
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return candidate, open_tier, aliases, tier_order

    # A bare `{code: {...}}` mapping. Recognised only when EVERY value is an object, so a
    # document that meant to have a "classes" key and misspelled it fails loudly instead
    # of being read as a catalog of one class called "clases".
    #
    # `tiers_most_open_first` is reserved here for the same reason the other three are, and
    # it was a real refusal before it was reserved: a code-keyed document that also declared
    # the ordering held one non-Mapping value, so `all(...)` failed and the whole file came
    # back "no protection classes found" -- a message pointing at the classes, which were
    # fine, rather than at the key it could not place.
    reserved = {
        "open_classification",
        "tiers_most_open_first",
        "aliases",
        "classes",
        "protection_classes",
    }
    body = {k: v for k, v in document.items() if k not in reserved}
    if body and all(isinstance(v, Mapping) for v in body.values()):
        return (
            [{"code": code, **dict(attrs)} for code, attrs in body.items()],
            open_tier,
            aliases,
            tier_order,
        )

    raise ValueError(
        "no protection classes found. Supply them as a 'classes' array, as a bare array, "
        "or as a mapping of code -> attributes."
    )


def _required_text(raw: Mapping[str, Any], key: str, subject: str, why: str) -> str:
    """
    Read a catalog string that CARRIES the classification, refusing to guess.

    `_required_bool` below has refused to guess since it was written. The two fields that
    actually decide what a field inherits had no such check, and the coercion ran before
    the emptiness test: `str(None)` is `'None'`, which is not empty, so neither guard fired.

    Both failures were silent and both were reproduced. A `"code": null` produced the code
    `'None'`, which then matched every glossary cell spelling "none" or "N/A-as-none" -- so
    a field asserting it has NO code inherited that class and `problems_with()` returned
    `[]`. A `"classification": null` derived the literal tier `'None'`, which is a `str` and
    therefore shipped over the wire as a tier nobody defined; worse, honest rows stating
    their real tier then contradicted it, and `governance_strict` refused the whole load
    blaming the glossary for a defect in the catalog.

    Applies to `code` and `classification` ONLY. Everything else optional goes through
    `_optional_text`, because null is a documented, load-bearing value in the rest of the
    file -- see the note there.
    """
    value = raw.get(key)
    if value is None:
        # Distinguished from absent on purpose. "You wrote null" points at the line; "you
        # have no 'code'" sends the reader looking for a key that is right in front of them.
        if key in raw:
            raise ValueError(f"{subject} has a {key!r} of null; it must be a string. {why}")
        raise ValueError(f"{subject} has no {key!r}. {why}")
    if not isinstance(value, str):
        raise ValueError(
            f"{subject} has a {key!r} of {value!r}, which is {type(value).__name__}; "
            f"it must be a string. {why}"
        )
    if not value.strip():
        raise ValueError(f"{subject} has no {key!r}. {why}")
    return value.strip()


def _optional_text(raw: Mapping[str, Any], key: str, subject: str) -> str | None:
    """
    Read an optional catalog string. Absent or null means absent; anything non-string is a
    defect rather than something to stringify.

    Null stays MEANINGFUL here, which is why this is a second helper and not a flag on the
    first. A single "reject null" rule applied to the field list this fix started from
    rejects this repo's own example pack: five of its nine classes declare
    `"enhancement": null`, and its alias map declares `"n/a": null` and `"tbc": null`. Both
    are documented ways for a caller to say "nothing here" out loud, which is the whole
    posture of this module.
    """
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{subject} has a {key!r} of {value!r}, which is {type(value).__name__}; "
            f"it must be a string or null"
        )
    return value


def _declared_list(raw: Mapping[str, Any], key: str, subject: str, item: str) -> list[str]:
    """
    A declared list of strings -- a class's `aliases`, or the document's
    `tiers_most_open_first` -- refusing the two shapes that silently half-work.

    The bare string is the one that mattered. `"aliases": "LEGACY-METER"` is iterable, so
    the loop walked it CHARACTER BY CHARACTER and declared L, E, G, A, C, Y and M as
    aliases of the class. Single-character aliases match nothing a glossary carries, so the
    legacy spelling the caller was trying to declare stays unknown and every row using it
    is refused -- for a reason that appears nowhere in their file. The tier ordering is the
    same shape and would fail the same way, one letter per rung.

    ONE function for both, rather than a second copy, because the two keys have exactly the
    same rules and a copy is how they stop having them.
    """
    declared = raw.get(key)
    if declared is None:
        return []
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ValueError(
            f"{key!r} on {subject} is {declared!r}, which is {type(declared).__name__}; "
            f"it must be a list of tokens. A bare string would be read one character at "
            f"a time."
        )
    tokens: list[str] = []
    for position, token in enumerate(declared):
        if not isinstance(token, str) or not token.strip():
            raise ValueError(
                f"{subject} declares {item} #{position} as {token!r}; every {item} must "
                f"be a non-blank string"
            )
        tokens.append(token)
    return tokens


def _checked_tier_order(
    declared: Iterable[str],
    by_code: Mapping[str, ProtectionClass],
    open_classification: str,
) -> tuple[str, ...]:
    """
    The caller's declared tier ordering, checked against the classes in the same file.

    This key shipped in the example pack and was read by NOTHING -- `grep` found exactly
    one occurrence in the repository, the declaration itself. An adopter copies that file
    to start their own vocabulary, so the no-op propagates along with a reasonable belief
    that the ordering means something. Enforcing it is the smaller change: the alternative
    is deleting a key that expresses something real about the caller's taxonomy.

    What is checked is the file against ITSELF. Every tier some class derives must appear,
    and so must the declared open tier, because those are the tiers that actually reach a
    consumer -- an ordering that cannot place the value on the wire cannot be used to
    compare it against anything. Nothing here decides what belongs in a taxonomy or what
    order it should be in.

    Two deliberate non-rules:

    * A declared tier NO class uses is legal. An organisation's ladder is allowed rungs
      this vocabulary does not reach, and refusing would make the key unusable for the
      thing it is for.
    * The sentinel open tier is exempt. `OPEN_CLASSIFICATION` is what an unset
      `open_classification` defaults to, so requiring it in the list would refuse a file
      over a value the caller never wrote -- this library's default failing this library's
      check.

    Comparison is by `_norm_tier`, the same normalisation a glossary row is compared
    through, so "Sealed " in the ladder and "sealed" on a class are one tier here exactly
    as they are one tier there. The DECLARED spellings are what get stored and reported.
    """
    tiers = tuple(declared)
    if not tiers:
        return ()

    seen: dict[str, str] = {}
    for tier in tiers:
        key = _norm_tier(tier)
        if key in seen:
            raise ValueError(
                f"'tiers_most_open_first' declares {seen[key]!r} and {tier!r}, which are "
                f"the same tier; an ordering cannot put one tier in two places"
            )
        seen[key] = tier

    for protection_class in by_code.values():
        if _norm_tier(protection_class.classification) not in seen:
            raise ValueError(
                f"protection class {protection_class.code!r} derives classification "
                f"{protection_class.classification!r}, which is not in the declared "
                f"'tiers_most_open_first' {list(tiers)!r}. An ordering that omits a tier "
                f"the vocabulary uses cannot place it, so add it to the list or remove the "
                f"key."
            )

    if open_classification != OPEN_CLASSIFICATION and _norm_tier(open_classification) not in seen:
        raise ValueError(
            f"open_classification {open_classification!r} is not in the declared "
            f"'tiers_most_open_first' {list(tiers)!r}. That is the tier an uncoded field "
            f"sits at -- the most common answer this vocabulary gives -- so an ordering "
            f"without it cannot place the majority of fields."
        )

    return tiers


def _alias_target_label(target_key: str | None, by_code: Mapping[str, ProtectionClass]) -> str:
    """How an alias target reads in a message: the code in its DECLARED spelling, or null."""
    if target_key is None:
        return "null (declaring the token droppable noise)"
    return repr(by_code[target_key].code)


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
