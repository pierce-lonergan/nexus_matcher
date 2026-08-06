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
