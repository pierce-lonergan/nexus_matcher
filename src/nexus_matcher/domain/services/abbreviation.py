"""
nexus_matcher.domain.services.abbreviation | Layer: DOMAIN
Abbreviation expansion service for improving semantic matching quality.

## Relationships
# DEPENDS_ON → None (pure domain service)
# USED_BY    → domain/models/entities :: SchemaField.to_searchable_text()
# USED_BY    → application/use_cases/match_schema :: field name expansion

## Invariants
# 1. Abbreviations are always lowercase (normalized)
# 2. Expansions preserve semantic meaning
# 3. Unknown abbreviations pass through unchanged

## Attributes
# Security: No external I/O, pure computation
# Performance: O(n) where n = number of tokens, dictionary lookup is O(1)
# Reliability: Never fails - unknown abbreviations pass through
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

# =============================================================================
# VALUE OBJECTS
# =============================================================================


@dataclass(frozen=True)
class AbbreviationMapping:
    """
    A single abbreviation → expansion mapping.

    Immutable value object representing one abbreviation and its full form.

    Example:
        mapping = AbbreviationMapping("acct", "account")
    """

    abbreviation: str
    expansion: str

    def __post_init__(self) -> None:
        """Validate and normalize the mapping."""
        # Normalize abbreviation to lowercase
        object.__setattr__(self, "abbreviation", self.abbreviation.lower().strip())

        # Validate
        if not self.abbreviation:
            raise ValueError("abbreviation must be non-empty")
        if not self.expansion or not self.expansion.strip():
            raise ValueError("expansion must be non-empty")
        if self.abbreviation == self.expansion.lower():
            raise ValueError("abbreviation cannot equal expansion")


@dataclass(frozen=True)
class ExpandedText:
    """
    Result of expanding abbreviations in text.

    Contains the original text, expanded text, and list of expansions applied.

    Example:
        result = ExpandedText(
            original="cust_acct",
            expanded="customer_account",
            expansions=[("cust", "customer"), ("acct", "account")]
        )
    """

    original: str
    expanded: str
    expansions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def was_expanded(self) -> bool:
        """Check if any expansions were applied."""
        return len(self.expansions) > 0

    @property
    def expansion_count(self) -> int:
        """Get number of expansions applied."""
        return len(self.expansions)


# =============================================================================
# ABBREVIATION DICTIONARY
# =============================================================================


class AbbreviationDictionary:
    """
    Collection of abbreviation mappings with efficient lookup.

    Supports:
    - Case-insensitive lookup
    - Loading from dict or file
    - Iteration over mappings

    Example:
        dictionary = AbbreviationDictionary.from_dict({
            "acct": "account",
            "cust": "customer",
        })
        expansion = dictionary.lookup("ACCT")  # Returns "account"
    """

    def __init__(self) -> None:
        """Initialize empty dictionary."""
        self._mappings: dict[str, str] = {}

    @property
    def size(self) -> int:
        """Get number of mappings."""
        return len(self._mappings)

    @property
    def is_empty(self) -> bool:
        """Check if dictionary is empty."""
        return len(self._mappings) == 0

    def add(self, mapping: AbbreviationMapping) -> None:
        """
        Add a mapping to the dictionary.

        Args:
            mapping: The mapping to add (overwrites if exists)
        """
        self._mappings[mapping.abbreviation] = mapping.expansion

    def lookup(self, abbreviation: str) -> str | None:
        """
        Look up expansion for an abbreviation.

        Args:
            abbreviation: The abbreviation to look up (case-insensitive)

        Returns:
            The expansion, or None if not found
        """
        return self._mappings.get(abbreviation.lower())

    def merged_with(self, overlay: Mapping[str, str]) -> AbbreviationDictionary:
        """
        A NEW dictionary holding this one's rows plus `overlay`'s, overlay winning.

        THIS OBJECT IS NOT MODIFIED, and that is the whole contract. The caller of this
        method is a per-request overlay merge (see `AbbreviationExpander.with_overlay`)
        running against a dictionary that is shared by every concurrent request, so a
        merge that mutated in place would let one request's approved-abbreviation list
        leak into the next one's query text -- a defect no accuracy measurement can
        attribute, because it moves a LATER request's answer.

        Overlay rows go through `AbbreviationMapping`'s validation, so an empty key, an
        empty expansion or a row whose short form equals its long form is SKIPPED rather
        than raised on -- the same admission rule as `from_dict`. A live reference-data
        feed is not a place to raise from: one malformed row must not cost the caller the
        other several thousand. A caller who wants to suppress a row this dictionary
        already holds supplies the correct long form for it; an identity row cannot do
        that job because it is not a valid mapping.

        Args:
            overlay: `{short -> long}` rows to layer on top. Keys are normalised (lowered
                and stripped) exactly as every other row's key is.

        Returns:
            A new AbbreviationDictionary. Never `self`.
        """
        merged = AbbreviationDictionary()
        # The base rows were already validated when they were added, so they are copied
        # rather than revalidated: re-running `AbbreviationMapping` over a 7,840-row
        # catalog on every request is real time spent proving something already proven.
        merged._mappings = dict(self._mappings)
        for abbrev, expansion in overlay.items():
            try:
                mapping = AbbreviationMapping(abbrev, expansion)
            except (ValueError, AttributeError):
                # AttributeError covers a non-string row from a JSON feed; both mean
                # "this row is not a mapping", and both are skipped.
                continue
            merged.add(mapping)
        return merged

    def __contains__(self, abbreviation: str) -> bool:
        """Check if abbreviation exists in dictionary."""
        return abbreviation.lower() in self._mappings

    def __iter__(self):
        """Iterate over (abbreviation, expansion) pairs."""
        return iter(self._mappings.items())

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> AbbreviationDictionary:
        """
        Create dictionary from a dict.

        Args:
            data: Dict mapping abbreviations to expansions

        Returns:
            Populated AbbreviationDictionary
        """
        dictionary = cls()
        for abbrev, expansion in data.items():
            try:
                mapping = AbbreviationMapping(abbrev, expansion)
                dictionary.add(mapping)
            except ValueError:
                # Skip invalid mappings
                pass
        return dictionary


# =============================================================================
# DEFAULT ABBREVIATIONS
# =============================================================================

# Common abbreviations in data/financial domain
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    # Account/Finance
    "acct": "account",
    "amt": "amount",
    "bal": "balance",
    "curr": "current",
    "pct": "percent",
    "txn": "transaction",
    "xfer": "transfer",
    # Customer/Business
    "cust": "customer",
    "clnt": "client",
    "org": "organization",
    "corp": "corporate",
    "bus": "business",
    "emp": "employee",
    # General Data
    "addr": "address",
    "cd": "code",
    "cnt": "count",
    "desc": "description",
    "dt": "date",
    "id": "identifier",
    "ind": "indicator",
    "msg": "message",
    "nm": "name",
    "no": "number",
    "num": "number",
    "qty": "quantity",
    "seq": "sequence",
    "stat": "status",
    "ts": "timestamp",
    "typ": "type",
    "val": "value",
    # Time
    "yr": "year",
    "mo": "month",
    "dy": "day",
    "hr": "hour",
    # NOTE: "min" is deliberately absent here - it is mapped to "minimum"
    # in the magnitude section below. Python keeps the last duplicate key, so
    # a "min" -> "minute" entry here never took effect.
    "sec": "second",
    # Status/State
    "actv": "active",
    "inactv": "inactive",
    "pend": "pending",
    "appr": "approved",
    "rej": "rejected",
    # Geography
    "cntry": "country",
    "st": "state",
    "cty": "city",
    "zip": "zipcode",
    "rgn": "region",
    # Technical
    "src": "source",
    "tgt": "target",
    "dest": "destination",
    "cfg": "configuration",
    "sys": "system",
    "env": "environment",
    "ref": "reference",
    "ver": "version",
    "fmt": "format",
    "len": "length",
    "sz": "size",
    "max": "maximum",
    "min": "minimum",
    "avg": "average",
    "tot": "total",
    "flg": "flag",
    "lvl": "level",
    "grp": "group",
    "cat": "category",
    # Names/People
    "fst": "first",
    "lst": "last",
    "mid": "middle",
    "pref": "prefix",
    "suf": "suffix",
    # Communication
    "tel": "telephone",
    "ph": "phone",
    "eml": "email",
    "fax": "facsimile",
}


# =============================================================================
# ABBREVIATION EXPANDER SERVICE
# =============================================================================


class AbbreviationExpander:
    """
    Domain service for expanding abbreviations in text.

    Handles multiple text formats:
    - underscore_separated
    - camelCase
    - PascalCase
    - SCREAMING_SNAKE_CASE

    Example:
        expander = AbbreviationExpander.default()
        result = expander.expand("cust_acct_bal")
        # result.expanded == "customer_account_balance"
    """

    # Singleton instance for default expander
    _default_instance: ClassVar[AbbreviationExpander | None] = None

    # Pattern for splitting tokens
    _SPLIT_PATTERN = re.compile(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

    def __init__(self, dictionary: AbbreviationDictionary) -> None:
        """
        Initialize expander with a dictionary.

        Args:
            dictionary: The abbreviation dictionary to use
        """
        self._dictionary = dictionary

    @classmethod
    def default(cls) -> AbbreviationExpander:
        """
        Get the default expander with built-in abbreviations.

        Returns:
            Singleton instance of default expander
        """
        if cls._default_instance is None:
            dictionary = AbbreviationDictionary.from_dict(DEFAULT_ABBREVIATIONS)
            cls._default_instance = cls(dictionary)
        return cls._default_instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset the default singleton (for testing)."""
        cls._default_instance = None

    def with_overlay(self, overlay: Mapping[str, str] | None) -> AbbreviationExpander:
        """
        A new expander carrying this one's catalog PLUS `overlay`, for one request.

        WHY THIS IS NOT A CONFIGURATION FILE. An approved-abbreviation catalog is live
        reference data: a term abbreviated one way this quarter may be abbreviated
        differently the next, and the authoritative copy lives in a service, not in the
        deployment's image. An expander fixed at start-up is therefore not a slower
        version of this -- it is a silently stale one, and staleness in this direction
        asserts the WRONG long form, which costs more than asserting none (a token absent
        from the catalog passes through untouched; a wrong row corrupts the one query
        vector the field gets).

        WHY IT RETURNS A NEW OBJECT. One `AbbreviationExpander` is shared by every
        concurrent request through the matcher that holds it. Merging in place would let
        request A's catalog decide request B's query text, non-deterministically, with no
        symptom in A's own response. So this never mutates `self`, and an empty overlay
        returns `self` unchanged rather than an equal copy -- the no-signal path pays
        nothing and cannot diverge from the shipped expander.

        Args:
            overlay: `{short -> long}` rows for this request, or None. Rows here outrank
                rows already in the catalog; invalid rows are skipped (see
                `AbbreviationDictionary.merged_with`).

        Returns:
            `self` when `overlay` is empty or None; otherwise a new expander.

        Example:
            per_request = matcher_expander.with_overlay({"psgr": "passenger"})
        """
        if not overlay:
            return self
        return AbbreviationExpander(self._dictionary.merged_with(overlay))

    def expand(self, text: str) -> ExpandedText:
        """
        Expand abbreviations in text.

        Handles underscore_separated, camelCase, and other formats.

        Args:
            text: The text to expand

        Returns:
            ExpandedText with original, expanded, and expansion list
        """
        if not text:
            return ExpandedText(original="", expanded="", expansions=[])

        # Detect separator used in original text.
        # Whitespace must be checked FIRST and takes precedence: the matching pipeline
        # feeds this method natural-language text from ContextEnricher (e.g.
        # "customer, account cust acct bal amt"). Such text has no underscores or
        # hyphens, so without this check it fell through to the camelCase branch and
        # was concatenated into a single out-of-vocabulary mega-token
        # ("customer,AccountCustomerAccountBalanceAmount"), which zeroed out every BM25
        # score and halved dense retrieval accuracy.
        has_whitespace = any(c.isspace() for c in text)
        has_underscores = "_" in text
        has_hyphens = "-" in text

        # Split into tokens
        tokens = self._tokenize(text)

        # Track expansions
        expansions: list[tuple[str, str]] = []
        expanded_tokens: list[str] = []

        for token in tokens:
            if not token:
                continue

            # Preserve case information
            is_upper = token.isupper()
            is_title = token.istitle()

            # Lookup expansion
            expansion = self._dictionary.lookup(token.lower())

            if expansion:
                expansions.append((token.lower(), expansion))
                # Apply original case style
                if is_upper:
                    expanded_tokens.append(expansion.upper())
                elif is_title:
                    expanded_tokens.append(expansion.title())
                else:
                    expanded_tokens.append(expansion)
            else:
                expanded_tokens.append(token)

        # Reconstruct with appropriate separator.
        # Whitespace wins over the identifier separators: multi-word text stays
        # multi-word, which is what both the tokenizer and the embedding model need.
        if has_whitespace:
            expanded = " ".join(expanded_tokens)
        elif has_underscores:
            expanded = "_".join(expanded_tokens)
        elif has_hyphens:
            expanded = "-".join(expanded_tokens)
        # camelCase - first token lowercase, rest title case
        elif len(expanded_tokens) > 1:
            expanded = expanded_tokens[0].lower() + "".join(t.title() for t in expanded_tokens[1:])
        else:
            expanded = expanded_tokens[0] if expanded_tokens else ""

        return ExpandedText(
            original=text,
            expanded=expanded,
            expansions=expansions,
        )

    def expand_tokens(self, tokens: list[str]) -> list[str]:
        """
        Expand a list of tokens.

        Args:
            tokens: List of tokens to expand

        Returns:
            List of expanded tokens
        """
        result = []
        for token in tokens:
            expansion = self._dictionary.lookup(token.lower())
            result.append(expansion if expansion else token)
        return result

    def get_candidates(self, abbreviation: str) -> list[str]:
        """
        Get expansion candidates for an abbreviation.

        Args:
            abbreviation: The abbreviation to look up

        Returns:
            List of possible expansions (usually 0 or 1)
        """
        expansion = self._dictionary.lookup(abbreviation)
        return [expansion] if expansion else []

    def _tokenize(self, text: str) -> list[str]:
        """
        Split text into tokens.

        Handles underscore, hyphen, space, and camelCase boundaries.
        """
        # Split on various boundaries
        tokens = self._SPLIT_PATTERN.split(text)
        # Filter empty tokens
        return [t for t in tokens if t]
