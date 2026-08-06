"""
nexus_matcher.infrastructure.adapters.dictionary_loaders.excel | Layer: INFRASTRUCTURE
Excel and CSV dictionary loaders.

## Relationships
# IMPLEMENTS → domain/ports/dictionary_loader :: DictionaryLoader protocol
# DEPENDS_ON → openpyxl :: .xlsx reading (no pandas; CSV needs nothing beyond stdlib)
# DELEGATES_TO → application/ingest :: _read_tabular, the shared hardened reader
# USED_BY    → application/use_cases/sync_dictionary :: dictionary loading

## Attributes
# Security: Validates file paths, sanitizes input
# Performance: Streams rows via openpyxl read-only mode; no DataFrame materialised
# Reliability: Handles malformed files gracefully
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus_matcher.domain.ports.dictionary_loader import (
    BaseDictionaryLoader,
)


class ExcelDictionaryLoader(BaseDictionaryLoader):
    """
    Loader for data dictionaries in Excel format (.xlsx, .xls).

    Supports:
    - Multiple sheets (first sheet by default)
    - Custom column mappings
    - Streaming for large files
    - Header row detection

    Example:
        loader = ExcelDictionaryLoader()
        result = loader.load(
            Path("dictionary.xlsx"),
            column_mapping=ColumnMapping(
                id_column="Field ID",
                business_name_column="Business Term",
            )
        )
        if result.is_success:
            entries, stats = result.unwrap()
            print(f"Loaded {stats.valid_entries} entries")
    """

    @property
    def source_type(self) -> str:
        """Source type identifier."""
        return "excel"

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        return frozenset({".xlsx", ".xls", ".xlsm"})

    def _load_rows(
        self,
        source: str | Path,
        **options: Any,
    ) -> Iterator[dict[str, Any]]:
        """
        Load rows from Excel file.

        Args:
            source: Path to Excel file
            **options:
                sheet_name: Sheet to read by NAME (default: the active sheet)
                header_row: 0-based row holding the headers (default: 0), for exports
                    that open with a title banner above the real header
                skip_rows: Data rows to skip after the header

        Yields:
            Row dictionaries
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        # openpyxl alone, without pandas.
        #
        # pandas was pulling ~50 MB of transitive dependency to read a spreadsheet that
        # openpyxl already reads, and the documented quickstart
        # (`matcher.load_dictionary("data/dictionary.xlsx")`) raised ImportError without
        # it. The shared reader also fills MERGED CELLS, which pandas leaves as NaN --
        # a classification merged down two rows silently downgraded the second one.
        from nexus_matcher.application.ingest import _read_tabular

        kwargs: dict[str, Any] = {"header_row": options.get("header_row", 0)}
        sheet_name = options.get("sheet_name")
        if isinstance(sheet_name, str):
            kwargs["sheet"] = sheet_name

        rows, header = _read_tabular(path, **kwargs)

        skip_rows = options.get("skip_rows")
        if skip_rows:
            rows = rows[skip_rows:]

        stripped = {col: col.strip() for col in header}
        for row in rows:
            yield {stripped.get(k, k): ("" if v is None else v) for k, v in row.items()}


class CsvDictionaryLoader(BaseDictionaryLoader):
    """
    Loader for data dictionaries in CSV format.

    Supports:
    - Various delimiters (comma, tab, pipe)
    - Custom encoding
    - Streaming for large files

    Example:
        loader = CsvDictionaryLoader()
        result = loader.load(Path("dictionary.csv"))
    """

    @property
    def source_type(self) -> str:
        """Source type identifier."""
        return "csv"

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        return frozenset({".csv", ".tsv", ".txt"})

    def _load_rows(
        self,
        source: str | Path,
        **options: Any,
    ) -> Iterator[dict[str, Any]]:
        """
        Load rows from CSV file.

        Args:
            source: Path to CSV file
            **options:
                delimiter: Field delimiter (default: auto-detect)
                encoding: File encoding (default: utf-8-sig, so an Excel BOM is stripped)
                header_row: 0-based row holding the headers (default: 0)
                skip_rows: Data rows to skip after the header

        Yields:
            Row dictionaries
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Read with the standard library, not pandas.
        #
        # CSV has never needed pandas, and requiring it broke the documented quickstart:
        # `matcher.load_dictionary("glossary.csv")` raised ImportError on a plain
        # `pip install nexus-matcher`, telling the user to install an extra in order to
        # read a comma-separated file. The shared reader in `application.ingest` is also
        # strictly more correct here -- it strips the BOM Excel writes, preserves newlines
        # inside quoted definitions, and keeps ragged rows off a None key.
        from nexus_matcher.application.ingest import _read_tabular

        kwargs: dict[str, Any] = {"header_row": options.get("header_row", 0)}
        if options.get("delimiter") is not None:
            kwargs["delimiter"] = options["delimiter"]
        if options.get("encoding") is not None:
            kwargs["encoding"] = options["encoding"]

        rows, header = _read_tabular(path, **kwargs)

        skip_rows = options.get("skip_rows")
        if skip_rows:
            rows = rows[skip_rows:]

        # Header cells often carry stray spaces from a spreadsheet export.
        stripped = {col: col.strip() for col in header}
        for row in rows:
            yield {stripped.get(k, k): ("" if v is None else v) for k, v in row.items()}
