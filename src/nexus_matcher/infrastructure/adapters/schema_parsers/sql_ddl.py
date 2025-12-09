"""
nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl | Layer: INFRASTRUCTURE
SQL DDL (CREATE TABLE) parser implementation.

## Relationships
# IMPLEMENTS → domain/ports/schema_parser :: SchemaParser protocol
# DEPENDS_ON → re :: Regular expression parsing (stdlib)
# USED_BY    → application/use_cases/match_schema :: schema parsing

## Attributes
# Security: Validates DDL structure to prevent malformed input
# Performance: O(n) parsing where n = number of columns
# Reliability: Handles common SQL dialects (PostgreSQL, MySQL, SQL Server)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nexus_matcher.domain.models.entities import Schema, SchemaField
from nexus_matcher.domain.ports.schema_parser import BaseSchemaParser
from nexus_matcher.shared.types.base import DataType, Result


class SqlDdlParser(BaseSchemaParser):
    """
    Parser for SQL DDL (CREATE TABLE) statements.

    Supports:
    - Common SQL types (VARCHAR, INT, TIMESTAMP, etc.)
    - Column constraints (NOT NULL, DEFAULT)
    - Multiple SQL dialects (standard, PostgreSQL, MySQL, SQL Server)
    - Inline comments (-- style)

    Example:
        parser = SqlDdlParser()
        result = parser.parse('''
            CREATE TABLE customers (
                id UUID PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                balance DECIMAL(10,2) DEFAULT 0.00
            );
        ''')
        if result.is_success:
            schema = result.unwrap()
            for field in schema.fields:
                print(f"{field.name}: {field.data_type}")
    """

    # SQL type to DataType mapping (case-insensitive patterns)
    TYPE_PATTERNS: list[tuple[str, DataType]] = [
        # String types
        (r"(?:VAR)?CHAR", DataType.STRING),
        (r"N?VARCHAR", DataType.STRING),
        (r"TEXT", DataType.STRING),
        (r"CLOB", DataType.STRING),

        # Integer types
        (r"BIGINT", DataType.LONG),  # Must be before INT
        (r"(?:SMALL|TINY)?INT(?:EGER)?", DataType.INTEGER),
        (r"SERIAL", DataType.INTEGER),
        (r"BIGSERIAL", DataType.LONG),

        # Floating point types
        (r"DOUBLE\s+PRECISION", DataType.DOUBLE),
        (r"DOUBLE", DataType.DOUBLE),
        (r"FLOAT", DataType.FLOAT),
        (r"REAL", DataType.FLOAT),

        # Decimal types
        (r"(?:DECIMAL|NUMERIC|NUMBER)", DataType.DECIMAL),

        # Boolean types
        (r"BOOL(?:EAN)?", DataType.BOOLEAN),
        (r"BIT", DataType.BOOLEAN),

        # Temporal types
        (r"TIMESTAMP(?:\s+WITH(?:OUT)?\s+TIME\s+ZONE)?", DataType.TIMESTAMP),
        (r"DATETIME2?", DataType.TIMESTAMP),
        (r"TIME", DataType.TIMESTAMP),
        (r"DATE", DataType.DATE),

        # Binary types
        (r"BYTEA", DataType.BYTES),
        (r"BLOB", DataType.BYTES),
        (r"(?:VAR)?BINARY", DataType.BYTES),
        (r"IMAGE", DataType.BYTES),

        # JSON types
        (r"JSONB?", DataType.JSON),

        # UUID types
        (r"UUID", DataType.UUID),
        (r"UNIQUEIDENTIFIER", DataType.UUID),

        # Array types
        (r"ARRAY", DataType.ARRAY),
    ]

    # Compile patterns for efficiency
    _compiled_patterns: list[tuple[re.Pattern, DataType]] = []

    def __init__(self):
        """Initialize parser with compiled regex patterns."""
        if not SqlDdlParser._compiled_patterns:
            SqlDdlParser._compiled_patterns = [
                (re.compile(rf"^{pattern}", re.IGNORECASE), dtype)
                for pattern, dtype in self.TYPE_PATTERNS
            ]

    @property
    def format_name(self) -> str:
        """Format identifier."""
        return "sql_ddl"

    @property
    def file_extensions(self) -> frozenset[str]:
        """Supported file extensions."""
        return frozenset({".sql", ".ddl"})

    def parse(self, content: str | dict[str, Any]) -> Result[Schema]:
        """
        Parse SQL DDL from string.

        Note: Does not support dict input (SQL is always text).

        Args:
            content: DDL string

        Returns:
            Result containing Schema on success
        """
        if isinstance(content, dict):
            return Result.failure("SQL DDL must be string, not dict", "INVALID_INPUT")

        try:
            schema = self._parse_ddl(content)
            return Result.success(schema)
        except ValueError as e:
            return Result.failure(str(e), "VALIDATION_ERROR")
        except Exception as e:
            return Result.failure(f"Unexpected error: {e}", "UNKNOWN_ERROR")

    def _parse_content(self, content: dict[str, Any]) -> Schema:
        """Not used for SQL DDL (string-only format)."""
        raise NotImplementedError("SQL DDL parser uses string input only")

    def _parse_ddl(self, ddl: str) -> Schema:
        """
        Parse DDL string into Schema.

        Args:
            ddl: CREATE TABLE statement

        Returns:
            Schema domain model

        Raises:
            ValueError: If DDL is invalid
        """
        # Remove inline comments
        ddl = self._remove_comments(ddl)

        # Extract table name and columns
        match = re.search(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)\s*\((.*)\)",
            ddl,
            re.IGNORECASE | re.DOTALL,
        )

        if not match:
            raise ValueError("Invalid DDL: No CREATE TABLE statement found")

        table_ref = match.group(1).strip()
        columns_str = match.group(2).strip()

        # Parse table name (may include schema)
        namespace = ""
        table_name = table_ref

        if "." in table_ref:
            parts = table_ref.split(".")
            namespace = parts[0].strip('"').strip("'").strip("`")
            table_name = parts[-1].strip('"').strip("'").strip("`")
        else:
            table_name = table_ref.strip('"').strip("'").strip("`")

        # Parse columns
        fields = self._parse_columns(columns_str)

        return Schema(
            name=table_name,
            fields=tuple(fields),
            namespace=namespace,
            source_format="sql_ddl",
            source_metadata={},
        )

    def _remove_comments(self, ddl: str) -> str:
        """Remove SQL comments from DDL."""
        # Remove -- style comments
        lines = []
        for line in ddl.split("\n"):
            # Find -- that's not inside a string
            comment_pos = self._find_comment_position(line)
            if comment_pos >= 0:
                line = line[:comment_pos]
            lines.append(line)
        return "\n".join(lines)

    def _find_comment_position(self, line: str) -> int:
        """Find position of -- comment, ignoring those in strings."""
        in_string = False
        string_char = None
        i = 0

        while i < len(line):
            char = line[i]

            if not in_string:
                if char in ("'", '"'):
                    in_string = True
                    string_char = char
                elif char == "-" and i + 1 < len(line) and line[i + 1] == "-":
                    return i
            else:
                if char == string_char:
                    in_string = False
                    string_char = None

            i += 1

        return -1

    def _parse_columns(self, columns_str: str) -> list[SchemaField]:
        """
        Parse column definitions from the columns portion of DDL.

        Args:
            columns_str: Content between parentheses in CREATE TABLE

        Returns:
            List of SchemaField instances
        """
        fields = []

        # Split by comma, but respect parentheses (for types like DECIMAL(10,2))
        column_defs = self._split_column_definitions(columns_str)

        for col_def in column_defs:
            col_def = col_def.strip()
            if not col_def:
                continue

            # Skip table-level constraints
            upper_def = col_def.upper()
            if upper_def.startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK", "CONSTRAINT", "INDEX")):
                continue

            field = self._parse_column(col_def)
            if field:
                fields.append(field)

        return fields

    def _split_column_definitions(self, columns_str: str) -> list[str]:
        """Split column definitions by comma, respecting parentheses."""
        result = []
        current = []
        paren_depth = 0

        for char in columns_str:
            if char == "(":
                paren_depth += 1
                current.append(char)
            elif char == ")":
                paren_depth -= 1
                current.append(char)
            elif char == "," and paren_depth == 0:
                result.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            result.append("".join(current))

        return result

    def _parse_column(self, col_def: str) -> SchemaField | None:
        """
        Parse a single column definition.

        Args:
            col_def: Single column definition string

        Returns:
            SchemaField or None if couldn't parse
        """
        # Normalize whitespace
        col_def = " ".join(col_def.split())

        # Extract column name (first word, handle quoted identifiers)
        match = re.match(r'(["`]?)(\w+)\1\s+(.+)', col_def, re.IGNORECASE)
        if not match:
            return None

        col_name = match.group(2)
        type_and_constraints = match.group(3)

        # Parse data type
        data_type = self._parse_type(type_and_constraints)

        # Parse constraints
        is_nullable = self._parse_nullable(type_and_constraints)
        default_value = self._parse_default(type_and_constraints)

        return SchemaField(
            name=col_name,
            data_type=data_type,
            full_path=col_name,
            parent_path="",
            description="",
            is_nullable=is_nullable,
            is_array=data_type == DataType.ARRAY,
            array_item_type=None,
            default_value=default_value,
            source_metadata={
                "sql_type": type_and_constraints.split()[0] if type_and_constraints else "",
            },
        )

    def _parse_type(self, type_str: str) -> DataType:
        """
        Parse SQL type string to DataType.

        Args:
            type_str: Type portion of column definition

        Returns:
            Matched DataType
        """
        # Normalize for matching
        type_upper = type_str.upper().strip()

        for pattern, dtype in self._compiled_patterns:
            if pattern.match(type_upper):
                return dtype

        return DataType.UNKNOWN

    def _parse_nullable(self, constraints: str) -> bool:
        """
        Parse nullable constraint.

        Args:
            constraints: Constraint portion of column definition

        Returns:
            True if nullable, False if NOT NULL
        """
        upper = constraints.upper()

        if "NOT NULL" in upper:
            return False

        # Default is nullable
        return True

    def _parse_default(self, constraints: str) -> str | None:
        """
        Parse DEFAULT value.

        Args:
            constraints: Constraint portion of column definition

        Returns:
            Default value string or None
        """
        # Match DEFAULT value (handles strings, numbers, keywords)
        match = re.search(
            r"DEFAULT\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))",
            constraints,
            re.IGNORECASE,
        )

        if match:
            # Return first non-None group (quoted string or unquoted value)
            return match.group(1) or match.group(2) or match.group(3)

        return None

    def can_parse(self, content: str | dict[str, Any]) -> bool:
        """
        Check if content is valid SQL DDL.

        Args:
            content: Content to check

        Returns:
            True if this parser can handle the content
        """
        if isinstance(content, dict):
            return False

        if not isinstance(content, str):
            return False

        # Check for CREATE TABLE pattern
        upper = content.upper()
        return "CREATE TABLE" in upper or "CREATE TABLE IF NOT EXISTS" in upper
