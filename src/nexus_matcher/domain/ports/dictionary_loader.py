"""
nexus_matcher.domain.ports.dictionary_loader | Layer: DOMAIN
Port interface for loading data dictionaries from various sources.

## Relationships
# USED_BY    → application/use_cases/sync_dictionary :: loading dictionaries
# USED_BY    → infrastructure/adapters/dictionary_loaders/* :: implementations
# DEPENDS_ON → domain/models/entities :: DictionaryEntry model

## Attributes
# Security: Loaders should sanitize input to prevent injection
# Performance: Should support streaming for large dictionaries
# Reliability: Should handle partial loads gracefully

## Extension Points
# Plugin: nexus_matcher.dictionary_loaders entry point
# Register: DictionaryLoaderRegistry.register()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.shared.types.base import DataType, ProtectionLevel, Result

# =============================================================================
# COLUMN MAPPING - Configuration for parsing dictionaries
# =============================================================================


@dataclass(frozen=True)
class ColumnMapping:
    """
    Configuration for mapping source columns to DictionaryEntry fields.

    Allows flexible parsing of dictionaries with varying column names.
    """

    id_column: str = "ID"
    business_name_column: str = "Business Name"
    logical_name_column: str = "Logical Name"
    definition_column: str = "Definition"
    data_type_column: str = "Data Type"
    protection_level_column: str = "Protection Level"
    domain_column: str = "Domain"
    parent_table_column: str = "Parent Table"
    sample_values_column: str | None = "Sample Values"
    synonyms_column: str | None = "Synonyms"

    @classmethod
    def default(cls) -> ColumnMapping:
        """Get default column mapping."""
        return cls()

    @classmethod
    def detect(cls, columns: Sequence[str]) -> ColumnMapping:
        """
        Infer a mapping from the columns a file actually has.

        Uses the same alias table as the ingest API, so "Term", "Business Term",
        "Business Definition", "Subject Area" and "Classification" are all understood.
        Fields with no match keep their literal default, which simply finds nothing --
        the same outcome as before, but only for the columns that are genuinely absent
        rather than for all of them.

        >>> ColumnMapping.detect(["Term", "Business Definition"]).business_name_column
        'Term'
        """
        return detect_column_mapping(columns)

    @classmethod
    def snake_case(cls) -> ColumnMapping:
        """Get snake_case column mapping."""
        return cls(
            id_column="id",
            business_name_column="business_name",
            logical_name_column="logical_name",
            definition_column="definition",
            data_type_column="data_type",
            protection_level_column="protection_level",
            domain_column="domain",
            parent_table_column="parent_table",
            sample_values_column="sample_values",
            synonyms_column="synonyms",
        )


def detect_column_mapping(columns: Sequence[str]) -> ColumnMapping:
    """
    Build a ColumnMapping from the columns a source actually has.

    The alias table lives in `application.ingest` and is shared deliberately: two
    independent notions of "which column is the business name" is how the loader path
    ended up rejecting files the ingest path read without complaint.

    Import is deferred to keep the domain layer free of an import-time dependency on the
    application layer.
    """
    from nexus_matcher.application.ingest import map_columns

    found = map_columns(list(columns))
    defaults = ColumnMapping()
    return ColumnMapping(
        id_column=found.get("id", defaults.id_column),
        business_name_column=found.get("business_name", defaults.business_name_column),
        logical_name_column=found.get("logical_name", defaults.logical_name_column),
        definition_column=found.get("definition", defaults.definition_column),
        data_type_column=found.get("data_type", defaults.data_type_column),
        protection_level_column=found.get("protection_level", defaults.protection_level_column),
        domain_column=found.get("domain", defaults.domain_column),
        parent_table_column=defaults.parent_table_column,
        sample_values_column=defaults.sample_values_column,
        synonyms_column=defaults.synonyms_column,
    )


# =============================================================================
# LOAD STATISTICS
# =============================================================================


@dataclass
class LoadStatistics:
    """Statistics from a dictionary load operation."""

    total_rows: int = 0
    valid_entries: int = 0
    skipped_rows: int = 0
    error_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_rows == 0:
            return 0.0
        return self.valid_entries / self.total_rows


# =============================================================================
# DICTIONARY LOADER PROTOCOL
# =============================================================================


@runtime_checkable
class DictionaryLoader(Protocol):
    """
    Protocol for dictionary loaders.

    Implementations load data dictionary entries from various sources
    (Excel, CSV, database, API, etc.) into DictionaryEntry domain models.

    Example usage:
        loader = ExcelDictionaryLoader()
        result = loader.load_file(Path("dictionary.xlsx"))
        if result.is_success:
            entries, stats = result.unwrap()
            print(f"Loaded {stats.valid_entries} entries")
    """

    @property
    def source_type(self) -> str:
        """
        Get the source type this loader handles.

        Returns:
            Source identifier (e.g., "excel", "csv", "database")
        """
        ...

    @property
    def supported_extensions(self) -> frozenset[str]:
        """
        Get supported file extensions (if file-based).

        Returns:
            Set of extensions (e.g., {".xlsx", ".xls"})
        """
        ...

    def load(
        self,
        source: str | Path,
        column_mapping: ColumnMapping | None = None,
        **options: Any,
    ) -> Result[tuple[list[DictionaryEntry], LoadStatistics]]:
        """
        Load dictionary entries from source.

        Args:
            source: Source path or connection string
            column_mapping: Column name mapping
            **options: Loader-specific options

        Returns:
            Result containing (entries, statistics) on success
        """
        ...

    def load_streaming(
        self,
        source: str | Path,
        column_mapping: ColumnMapping | None = None,
        batch_size: int = 100,
        **options: Any,
    ) -> Iterator[Result[list[DictionaryEntry]]]:
        """
        Load dictionary entries in streaming batches.

        Args:
            source: Source path or connection string
            column_mapping: Column name mapping
            batch_size: Number of entries per batch
            **options: Loader-specific options

        Yields:
            Result containing batch of entries
        """
        ...


# =============================================================================
# BASE IMPLEMENTATION
# =============================================================================


class BaseDictionaryLoader(ABC):
    """
    Abstract base class for dictionary loaders with common functionality.

    Provides:
    - Row to entry conversion
    - Validation
    - Statistics tracking
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source type identifier."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        ...

    @abstractmethod
    def _load_rows(
        self,
        source: str | Path,
        **options: Any,
    ) -> Iterator[dict[str, Any]]:
        """
        Load raw rows from source.

        Args:
            source: Data source
            **options: Loader-specific options

        Yields:
            Raw row dictionaries
        """
        ...

    def _convert_row(
        self,
        row: dict[str, Any],
        mapping: ColumnMapping,
        row_index: int,
    ) -> Result[DictionaryEntry]:
        """Convert a row dictionary to DictionaryEntry."""
        try:
            # Extract required fields
            entry_id = str(row.get(mapping.id_column, ""))
            if not entry_id:
                entry_id = f"auto_{row_index}"

            business_name = str(row.get(mapping.business_name_column, "")).strip()
            if not business_name:
                return Result.failure(f"Row {row_index}: Missing business name")

            logical_name = str(row.get(mapping.logical_name_column, "")).strip()
            definition = str(row.get(mapping.definition_column, "")).strip()
            data_type_str = str(row.get(mapping.data_type_column, "string")).strip()
            protection_str = str(row.get(mapping.protection_level_column, "")).strip()

            # Parse optional fields
            domain = str(row.get(mapping.domain_column, "")).strip()
            parent_table = str(row.get(mapping.parent_table_column, "")).strip()

            # Parse sample values
            sample_values: tuple[str, ...] = ()
            if mapping.sample_values_column:
                samples_raw = row.get(mapping.sample_values_column, "")
                if samples_raw:
                    samples_str = str(samples_raw)
                    sample_values = tuple(s.strip() for s in samples_str.split(",") if s.strip())

            # Parse synonyms
            synonyms: frozenset[str] = frozenset()
            if mapping.synonyms_column:
                synonyms_raw = row.get(mapping.synonyms_column, "")
                if synonyms_raw:
                    synonyms_str = str(synonyms_raw)
                    synonyms = frozenset(s.strip() for s in synonyms_str.split(",") if s.strip())

            entry = DictionaryEntry(
                id=entry_id,
                business_name=business_name,
                logical_name=logical_name,
                definition=definition,
                data_type=DataType.from_string(data_type_str),
                protection_level=ProtectionLevel.from_string(protection_str),
                domain=domain,
                parent_table=parent_table,
                sample_values=sample_values,
                synonyms=synonyms,
            )

            return Result.success(entry)

        except Exception as e:
            return Result.failure(f"Row {row_index}: {e}")

    def load(
        self,
        source: str | Path,
        column_mapping: ColumnMapping | None = None,
        **options: Any,
    ) -> Result[tuple[list[DictionaryEntry], LoadStatistics]]:
        """Load dictionary entries from source."""
        stats = LoadStatistics()
        entries: list[DictionaryEntry] = []
        mapping = column_mapping

        try:
            for row_index, row in enumerate(self._load_rows(source, **options), start=1):
                stats.total_rows += 1

                if mapping is None:
                    # Infer the mapping from the header actually present.
                    #
                    # ColumnMapping.default() names exact literal columns ("Business
                    # Name", "Definition"). Real glossaries say "Term", "Business Term",
                    # "Business Definition", "Classification" -- none of which matched, so
                    # every row failed on "Missing business name" and the load reported
                    # SUCCESS with zero entries. detect_column_mapping reuses the same
                    # alias table the ingest API already applies to these files.
                    mapping = detect_column_mapping(list(row))

                result = self._convert_row(row, mapping, row_index)

                if result.is_success:
                    entries.append(result.unwrap())
                    stats.valid_entries += 1
                else:
                    stats.error_rows += 1
                    stats.errors.append(result.error or "Unknown error")

            # Reading rows and producing nothing is a FAILURE, not a quiet success.
            # It means the columns were not understood; returning an empty dictionary
            # leaves the matcher with nothing to match against and no indication why.
            if stats.total_rows and not entries:
                sample = stats.errors[0] if stats.errors else "no rows converted"
                return Result.failure(
                    f"Read {stats.total_rows} row(s) from {source} but produced no "
                    f"entries -- the columns were not recognised (first error: {sample}). "
                    f"Pass an explicit column_mapping=ColumnMapping(...)."
                )

            return Result.success((entries, stats))

        except Exception as e:
            return Result.failure(f"Failed to load dictionary: {e}")

    def load_streaming(
        self,
        source: str | Path,
        column_mapping: ColumnMapping | None = None,
        batch_size: int = 100,
        **options: Any,
    ) -> Iterator[Result[list[DictionaryEntry]]]:
        """Load dictionary entries in streaming batches."""
        mapping = column_mapping or ColumnMapping.default()
        batch: list[DictionaryEntry] = []

        try:
            for row_index, row in enumerate(self._load_rows(source, **options), start=1):
                result = self._convert_row(row, mapping, row_index)

                if result.is_success:
                    batch.append(result.unwrap())

                    if len(batch) >= batch_size:
                        yield Result.success(batch)
                        batch = []

            # Yield remaining
            if batch:
                yield Result.success(batch)

        except Exception as e:
            yield Result.failure(f"Streaming error: {e}")


# =============================================================================
# LOADER REGISTRY
# =============================================================================


class DictionaryLoaderRegistry:
    """
    Registry for dictionary loader implementations.

    Example:
        registry = DictionaryLoaderRegistry()
        registry.register(ExcelDictionaryLoader())

        # By source type
        loader = registry.get_loader("excel")

        # By file extension
        loader = registry.get_loader_for_file(Path("dict.xlsx"))
    """

    def __init__(self) -> None:
        self._loaders: dict[str, DictionaryLoader] = {}
        self._extension_map: dict[str, str] = {}

    def register(self, loader: DictionaryLoader) -> None:
        """Register a dictionary loader."""
        self._loaders[loader.source_type] = loader
        for ext in loader.supported_extensions:
            self._extension_map[ext.lower()] = loader.source_type

    def get_loader(self, source_type: str) -> DictionaryLoader | None:
        """Get loader by source type."""
        return self._loaders.get(source_type)

    def get_loader_for_file(self, path: Path) -> DictionaryLoader | None:
        """Get loader by file extension."""
        source_type = self._extension_map.get(path.suffix.lower())
        if source_type:
            return self._loaders.get(source_type)
        return None

    def list_source_types(self) -> list[str]:
        """Get list of registered source types."""
        return list(self._loaders.keys())

    def load_entry_points(self) -> None:
        """Load loaders from entry points."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group="nexus_matcher.dictionary_loaders")
            for ep in eps:
                try:
                    loader_class = ep.load()
                    loader = loader_class()
                    self.register(loader)
                except Exception as e:
                    import logging

                    logging.warning(f"Failed to load dictionary loader {ep.name}: {e}")
        except ImportError:
            pass
