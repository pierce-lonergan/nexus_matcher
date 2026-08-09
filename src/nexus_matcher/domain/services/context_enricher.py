"""
nexus_matcher.domain.services.context_enricher | Layer: DOMAIN
Context enrichment for nested schema fields.

## Relationships
# DEPENDS_ON → domain/models/entities :: SchemaField
# USED_BY    → application/use_cases/match_schema :: query text generation
# USED_BY    → infrastructure/adapters :: embedding pipelines

## Attributes
# Security: No sensitive data handling
# Performance: String operations only, O(n) where n = hierarchy depth
# Reliability: Pure function, no side effects

## Research Reference
# README_RESEARCH_1.md, Lines 174-176
# "Context injection is non-negotiable for nested schemas... For user.addresses[].street_name,
# the query must include 'user entity, addresses array, street_name field' structure."
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nexus_matcher.domain.models.entities import SchemaField
from nexus_matcher.shared.types.base import DataType

# Type descriptions for embedding context
TYPE_DESCRIPTIONS: dict[DataType, str] = {
    DataType.STRING: "text",
    DataType.INTEGER: "integer number",
    DataType.LONG: "long integer",
    DataType.FLOAT: "decimal number",
    DataType.DOUBLE: "decimal number",
    DataType.DECIMAL: "decimal number",
    DataType.BOOLEAN: "boolean flag",
    DataType.DATE: "date",
    DataType.TIMESTAMP: "timestamp",
    DataType.BYTES: "binary data",
    DataType.UUID: "unique identifier",
    DataType.ARRAY: "array",
    DataType.RECORD: "record",
    DataType.ENUM: "enumeration",
    DataType.JSON: "json data",
    DataType.UNKNOWN: "data",
}


@dataclass
class EnrichmentConfig:
    """
    Configuration for context enrichment.

    `include_type` defaults to False. Appending a type descriptor such as "text field"
    measurably HURTS retrieval: on the combined BIRD+OMOP benchmark it cost 2.1 points
    of P@1 (0.493 -> 0.469). Nearly every dictionary entry is compatible with a generic
    type phrase, so the extra tokens add no discriminating signal while diluting the
    embedding of the part that does discriminate. Type compatibility is still used --
    but as a separate numeric scoring signal, which is where it belongs, not as text
    smuggled into the query.

    `include_array_indicator` is deliberately SEPARATE from `include_type` and stays on.
    Whether a field is a collection is structural information that changes what it can
    map to, unlike a scalar type word. The benchmark that showed type text to be harmful
    (BIRD + OMOP) is entirely flat relational schemas containing no array fields, so it
    provides no evidence either way about array markers -- and turning one off is not a
    reason to turn the other off.

    `include_hierarchy` is the single most valuable setting in this class: parent-path
    context is worth +20 points of P@1 on the same benchmark. Do not disable it without
    measuring.

    Level-wise hierarchy dedup -- dropping a parent level whose every token the field name
    already repeats, so `Account.subject` reads "account subject" rather than "account
    account subject" -- was tried and REMOVED. It genuinely improved ranking (FHIR P@1
    +0.0154, p=0.018), but it raised scores across the board and so pushed more candidates
    past the fixed auto-approve threshold: auto-approve precision fell 0.8049 -> 0.7391,
    auto-approvals went 41 -> 46, and of the 7 newly auto-approved 5 were WRONG, taking
    wrong-and-unreviewed approvals from 8 to 12. This service decides whether a field
    inherits a PII classification, so a wrong approval nobody reviews costs far more than
    a missed match that goes to a human. Any retrieval change here must be measured on
    auto-approve precision, not only on P@1/recall -- better retrieval lowering
    auto-approve precision at a fixed threshold has now happened three times.
    """

    include_type: bool = False
    include_hierarchy: bool = True
    include_description: bool = True
    include_array_indicator: bool = True
    max_depth: int = 5
    humanize_names: bool = True


class ContextEnricher:
    """
    Enriches SchemaField with hierarchical context for embedding.

    Research shows that nested schema fields lose 10-20% accuracy when
    hierarchical context is not included. This enricher builds rich
    contextual text that includes parent entities, array indicators,
    and type information.

    Example:
        enricher = ContextEnricher()

        # For user.addresses[].street_name:
        field = SchemaField(
            name="street_name",
            data_type=DataType.STRING,
            full_path="user.addresses.street_name",
            parent_path="user.addresses",
        )

        # Returns: "user, addresses street name"
        enriched = enricher.enrich(field)
    """

    def __init__(
        self,
        include_type: bool = False,
        include_hierarchy: bool = True,
        include_description: bool = True,
        include_array_indicator: bool = True,
        max_depth: int = 5,
        humanize_names: bool = True,
    ) -> None:
        """
        Initialize context enricher.

        Args:
            include_type: Include data type in context (default False; see
                EnrichmentConfig -- it measurably reduces P@1)
            include_hierarchy: Include parent hierarchy
            include_description: Include field description
            max_depth: Maximum hierarchy depth to include
            humanize_names: Convert snake_case/camelCase to human readable
        """
        self._config = EnrichmentConfig(
            include_type=include_type,
            include_hierarchy=include_hierarchy,
            include_description=include_description,
            include_array_indicator=include_array_indicator,
            max_depth=max_depth,
            humanize_names=humanize_names,
        )

    def enrich(self, field: SchemaField) -> str:
        """
        Enrich a schema field with hierarchical context.

        Args:
            field: SchemaField to enrich

        Returns:
            Enriched text suitable for embedding
        """
        parts: list[str] = []

        # Add hierarchy context (parent entities)
        if self._config.include_hierarchy and field.is_nested:
            hierarchy = self._build_hierarchy_context(field)
            if hierarchy:
                parts.append(hierarchy)

        # Add field name (humanized)
        field_name = self._humanize(field.name) if self._config.humanize_names else field.name
        parts.append(field_name)

        # Add type context
        if self._config.include_type:
            type_desc = self._get_type_description(field)
            if type_desc:
                parts.append(type_desc)
        elif self._config.include_array_indicator and field.is_array:
            # Keep the structural signal even when scalar type words are suppressed.
            parts.append("array")

        # Add description
        if self._config.include_description and field.description:
            parts.append(field.description.strip())

        # Join and normalize
        text = " ".join(filter(None, parts))
        return self._normalize_whitespace(text)

    def enrich_batch(self, fields: list[SchemaField]) -> list[str]:
        """
        Enrich multiple fields.

        Args:
            fields: List of SchemaFields

        Returns:
            List of enriched texts
        """
        return [self.enrich(field) for field in fields]

    def _build_hierarchy_context(self, field: SchemaField) -> str:
        """Build hierarchical context string from field path."""
        if not field.full_path or "." not in field.full_path:
            return ""

        # Split path and process each level
        path_parts = field.full_path.split(".")

        # Limit depth
        if len(path_parts) > self._config.max_depth + 1:
            # Keep first parts and last field
            path_parts = [*path_parts[: self._config.max_depth], path_parts[-1]]

        # Process parent parts (exclude the field name itself)
        parent_parts = path_parts[:-1]

        if not parent_parts:
            return ""

        # Build context for each parent level
        context_parts = []
        for part in parent_parts:
            # Skip namespace-like parts (com.company.etc)
            if self._is_namespace_part(part):
                continue

            humanized = self._humanize(part) if self._config.humanize_names else part
            context_parts.append(humanized)

        if not context_parts:
            return ""

        return ", ".join(context_parts)

    def _get_type_description(self, field: SchemaField) -> str:
        """Get human-readable type description."""
        base_type = TYPE_DESCRIPTIONS.get(field.data_type, "data")

        if field.is_array:
            item_type = TYPE_DESCRIPTIONS.get(field.array_item_type or DataType.UNKNOWN, "data")
            return f"array of {item_type}"

        return f"{base_type} field"

    def _humanize(self, name: str) -> str:
        """
        Convert identifier to human-readable text.

        Every boundary that is split here becomes a separate token for both BM25 and the
        embedding tokenizer, so missing one leaves an out-of-vocabulary blob. Digit
        boundaries and punctuation matter as much as case boundaries: "enroll12" and
        "FRPM Count (K-12)" have to become "enroll 12" and "frpm count k 12".
        """
        # Handle camelCase
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

        # Handle consecutive caps (e.g., "HTTPResponse" -> "HTTP Response")
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)

        # Split letter/digit boundaries in BOTH directions
        # ("enroll12" -> "enroll 12", "K12Grade" -> "K 12 Grade").
        name = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", name)
        name = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", name)

        # Any non-alphanumeric run is a separator: snake_case, kebab-case, dots,
        # parentheses and stray punctuation all collapse to whitespace.
        name = re.sub(r"[^0-9A-Za-z]+", " ", name)

        # Normalize whitespace and lowercase
        name = re.sub(r"\s+", " ", name).strip().lower()

        return name

    def _is_namespace_part(self, part: str) -> bool:
        """Check if path part looks like a namespace (e.g., 'com', 'org')."""
        namespace_patterns = {"com", "org", "net", "io", "ai", "gov", "edu"}
        return part.lower() in namespace_patterns or len(part) <= 2

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r"\s+", " ", text).strip()


# Convenience function for quick enrichment
def enrich_field(field: SchemaField) -> str:
    """
    Enrich a single field with default settings.

    Args:
        field: SchemaField to enrich

    Returns:
        Enriched text
    """
    enricher = ContextEnricher()
    return enricher.enrich(field)


def enrich_fields(fields: list[SchemaField]) -> list[str]:
    """
    Enrich multiple fields with default settings.

    Args:
        fields: List of SchemaFields

    Returns:
        List of enriched texts
    """
    enricher = ContextEnricher()
    return enricher.enrich_batch(fields)
