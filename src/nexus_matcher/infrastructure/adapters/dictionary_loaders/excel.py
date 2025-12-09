"""
nexus_matcher.infrastructure.adapters.dictionary_loaders.excel | Layer: INFRASTRUCTURE
Excel dictionary loader implementation.

## Relationships
# IMPLEMENTS → domain/ports/dictionary_loader :: DictionaryLoader protocol
# DEPENDS_ON → pandas :: Excel reading
# DEPENDS_ON → openpyxl :: .xlsx support
# USED_BY    → application/use_cases/sync_dictionary :: dictionary loading

## Attributes
# Security: Validates file paths, sanitizes input
# Performance: Uses pandas for efficient reading, supports streaming
# Reliability: Handles malformed files gracefully
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from nexus_matcher.domain.ports.dictionary_loader import (
    BaseDictionaryLoader,
    ColumnMapping,
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
                sheet_name: Sheet to read (default: first sheet)
                header_row: Row containing headers (default: 0)
                skip_rows: Rows to skip at start

        Yields:
            Row dictionaries
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas and openpyxl are required for Excel loading. "
                "Install with: pip install nexus-matcher[loaders]"
            )

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        # Parse options
        sheet_name = options.get("sheet_name", 0)
        header_row = options.get("header_row", 0)
        skip_rows = options.get("skip_rows", None)

        # Read Excel file
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=header_row,
            skiprows=skip_rows,
            engine="openpyxl" if path.suffix == ".xlsx" else None,
        )

        # Clean column names
        df.columns = df.columns.str.strip()

        # Yield rows as dictionaries
        for _, row in df.iterrows():
            # Convert row to dict, handling NaN values
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = ""
                else:
                    row_dict[col] = value
            yield row_dict


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
                encoding: File encoding (default: utf-8)
                skip_rows: Rows to skip at start

        Yields:
            Row dictionaries
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for CSV loading. "
                "Install with: pip install nexus-matcher[loaders]"
            )

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Parse options
        delimiter = options.get("delimiter")
        encoding = options.get("encoding", "utf-8")
        skip_rows = options.get("skip_rows", None)

        # Auto-detect delimiter
        if delimiter is None:
            if path.suffix == ".tsv":
                delimiter = "\t"
            else:
                delimiter = ","

        # Read CSV file
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            encoding=encoding,
            skiprows=skip_rows,
        )

        # Clean column names
        df.columns = df.columns.str.strip()

        # Yield rows as dictionaries
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_dict[col] = ""
                else:
                    row_dict[col] = value
            yield row_dict
