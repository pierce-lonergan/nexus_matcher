"""
nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl | Layer: INFRASTRUCTURE
SQL DDL (CREATE TABLE) parser implementation.

## Relationships
# IMPLEMENTS → domain/ports/schema_parser :: SchemaParser protocol
# DEPENDS_ON → re :: Regular expression parsing (stdlib)
# USED_BY    → application/use_cases/match_schema :: schema parsing

## Attributes
# Security: Validates DDL structure to prevent malformed input
# Performance: O(n) parsing where n = length of the DDL
# Reliability: Handles common SQL dialects (PostgreSQL, MySQL, SQL Server),
#              multiple statements per file, and all four identifier quoting
#              styles
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

from nexus_matcher.domain.models.entities import Schema, SchemaField
from nexus_matcher.domain.ports.schema_parser import BaseSchemaParser
from nexus_matcher.shared.types.base import DataType, Result

logger = logging.getLogger(__name__)


class SqlDdlParser(BaseSchemaParser):
    """
    Parser for SQL DDL (CREATE TABLE) statements.

    Supports:
    - Common SQL types (VARCHAR, INT, TIMESTAMP, etc.)
    - Column constraints (NOT NULL, DEFAULT)
    - Multiple SQL dialects (standard, PostgreSQL, MySQL, SQL Server)
    - Quoted identifiers in all four styles: "ansi", `mysql`, [sqlserver],
      including names containing spaces and dots
    - Multiple CREATE TABLE statements in one input (see parse_all)
    - Line (--) and block (/* */) comments

    Two defects this parser used to have:

    1. Statement boundaries. The table body was located with
       `CREATE\\s+TABLE\\s+([^\\s(]+)\\s*\\((.*)\\)` under re.DOTALL. `.*` is
       greedy, so on a file with several CREATE TABLE statements it captured
       from the first table's opening paren all the way to the LAST closing
       paren in the file. The result was one Schema named after the first table
       whose column list was a blend of every table in the file -- silently
       wrong rather than an error. Statement bodies are now found by scanning
       for the balanced closing parenthesis, skipping over quoted strings.

    2. SQL Server bracket identifiers. `[dbo].[Customers]` was kept verbatim as
       the table name (brackets and all), and a column written
       `[Customer Name] NVARCHAR(100)` failed the `(["`]?)(\\w+)\\1` name
       pattern entirely and was dropped from the output. Both now parse.

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

        # Several tables in one file:
        all_schemas = parser.parse_all(ddl_text).unwrap()
    """

    # SQL type to DataType mapping (case-insensitive patterns)
    TYPE_PATTERNS: ClassVar[list[tuple[str, DataType]]] = [
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
    _compiled_patterns: ClassVar[list[tuple[re.Pattern, DataType]]] = []

    # Locates the header of a CREATE TABLE statement up to its opening paren.
    _CREATE_TABLE_RE = re.compile(
        r"CREATE\s+(?:(?:GLOBAL|LOCAL)\s+)?(?:TEMP(?:ORARY)?\s+)?"
        r"(?:UNLOGGED\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?",
        re.IGNORECASE,
    )

    # Table-level constraint keywords that are not column definitions.
    _CONSTRAINT_PREFIXES = (
        "PRIMARY KEY",
        "FOREIGN KEY",
        "UNIQUE",
        "CHECK",
        "CONSTRAINT",
        "INDEX",
        "KEY ",
        "EXCLUDE",
        "PERIOD FOR",
    )

    # Matching close for each opening quote character.
    _QUOTE_PAIRS: ClassVar[dict[str, str]] = {'"': '"', "'": "'", "`": "`", "[": "]"}

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

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def parse(self, content: str | dict[str, Any]) -> Result[Schema]:
        """
        Parse SQL DDL from string, returning the FIRST table.

        Note: Does not support dict input (SQL is always text).

        When the input holds several CREATE TABLE statements, the first is
        returned and `source_metadata["additional_tables"]` names the rest, so a
        caller can tell that more was present. Use `parse_all()` to get every
        table.

        Args:
            content: DDL string

        Returns:
            Result containing Schema on success
        """
        if isinstance(content, dict):
            return Result.failure("SQL DDL must be string, not dict", "INVALID_INPUT")

        try:
            schemas = self._parse_ddl_all(content)
        except ValueError as e:
            return Result.failure(str(e), "VALIDATION_ERROR")
        except Exception as e:
            return Result.failure(f"Unexpected error: {e}", "UNKNOWN_ERROR")

        if not schemas:
            return Result.failure(
                "Invalid DDL: No CREATE TABLE statement found", "VALIDATION_ERROR"
            )

        first = schemas[0]
        if len(schemas) > 1:
            others = [s.name for s in schemas[1:]]
            logger.info(
                "DDL contained %d tables; parse() returns %r. Use parse_all() "
                "for all of them. Remaining: %s",
                len(schemas),
                first.name,
                others,
            )
            first = Schema(
                name=first.name,
                fields=first.fields,
                namespace=first.namespace,
                source_format=first.source_format,
                source_metadata={
                    **first.source_metadata,
                    "table_count": len(schemas),
                    "additional_tables": others,
                },
            )

        return Result.success(first)

    def parse_all(self, content: str | dict[str, Any]) -> Result[list[Schema]]:
        """
        Parse every CREATE TABLE statement in the input.

        Args:
            content: DDL string

        Returns:
            Result containing one Schema per table, in source order
        """
        if isinstance(content, dict):
            return Result.failure("SQL DDL must be string, not dict", "INVALID_INPUT")

        try:
            schemas = self._parse_ddl_all(content)
        except ValueError as e:
            return Result.failure(str(e), "VALIDATION_ERROR")
        except Exception as e:
            return Result.failure(f"Unexpected error: {e}", "UNKNOWN_ERROR")

        if not schemas:
            return Result.failure(
                "Invalid DDL: No CREATE TABLE statement found", "VALIDATION_ERROR"
            )

        return Result.success(schemas)

    def _parse_content(self, content: dict[str, Any]) -> Schema:
        """Not used for SQL DDL (string-only format)."""
        raise NotImplementedError("SQL DDL parser uses string input only")

    # =========================================================================
    # STATEMENT SCANNING
    # =========================================================================

    def _parse_ddl_all(self, ddl: str) -> list[Schema]:
        """Parse every CREATE TABLE statement into a Schema."""
        ddl = self._remove_comments(ddl)

        schemas: list[Schema] = []
        for table_ref, columns_str in self._iter_create_tables(ddl):
            namespace, table_name = self._split_qualified_name(table_ref)
            fields = self._parse_columns(columns_str)

            schemas.append(
                Schema(
                    name=table_name,
                    fields=tuple(fields),
                    namespace=namespace,
                    source_format="sql_ddl",
                    source_metadata={},
                )
            )

        return schemas

    def _iter_create_tables(self, ddl: str):
        """
        Yield (table_ref, columns_str) for each CREATE TABLE in the input.

        The column body is delimited by scanning for the parenthesis that
        balances the opening one, skipping anything inside quotes. That is what
        makes several statements per file work, and it also keeps a
        `DEFAULT '(pending)'` literal from ending the body early.
        """
        for header in self._CREATE_TABLE_RE.finditer(ddl):
            cursor = header.end()

            table_ref, cursor = self._read_identifier(ddl, cursor)
            if not table_ref:
                continue

            # Skip whitespace to the opening parenthesis.
            while cursor < len(ddl) and ddl[cursor].isspace():
                cursor += 1

            if cursor >= len(ddl) or ddl[cursor] != "(":
                # e.g. CREATE TABLE x AS SELECT ... -- no column list to read.
                continue

            body_end = self._find_matching_paren(ddl, cursor)
            if body_end < 0:
                raise ValueError(
                    f"Invalid DDL: unbalanced parentheses in CREATE TABLE {table_ref!r}"
                )

            yield table_ref, ddl[cursor + 1 : body_end]

    def _read_identifier(self, text: str, start: int) -> tuple[str, int]:
        """
        Read a possibly-qualified, possibly-quoted identifier.

        Handles `customers`, `public.customers`, `"My Schema"."My Table"`,
        `` `db`.`tbl` `` and `[dbo].[Customer Orders]`.

        Returns:
            (raw_identifier_text, index_after_it)
        """
        cursor = start
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        begin = cursor
        while cursor < len(text):
            char = text[cursor]

            if char in self._QUOTE_PAIRS:
                closer = self._QUOTE_PAIRS[char]
                cursor += 1
                while cursor < len(text) and text[cursor] != closer:
                    cursor += 1
                cursor += 1  # consume the closer
                continue

            if char.isspace() or char == "(":
                break

            cursor += 1

        return text[begin:cursor].strip(), cursor

    def _find_matching_paren(self, text: str, open_index: int) -> int:
        """
        Index of the ')' matching the '(' at open_index, or -1.

        Quoted regions are skipped so parentheses inside string literals and
        bracket identifiers do not affect the depth count.
        """
        depth = 0
        cursor = open_index

        while cursor < len(text):
            char = text[cursor]

            if char in ("'", '"', "`"):
                closer = char
                cursor += 1
                while cursor < len(text):
                    if text[cursor] == closer:
                        # Doubled quote is an escaped quote, not a terminator.
                        if cursor + 1 < len(text) and text[cursor + 1] == closer:
                            cursor += 2
                            continue
                        break
                    cursor += 1
                cursor += 1
                continue

            if char == "[":
                while cursor < len(text) and text[cursor] != "]":
                    cursor += 1
                cursor += 1
                continue

            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return cursor

            cursor += 1

        return -1

    def _remove_comments(self, ddl: str) -> str:
        """
        Remove -- line comments and /* */ block comments.

        Comment markers inside string literals and quoted identifiers are left
        alone; stripping those would corrupt DEFAULT values and column names.
        """
        out: list[str] = []
        cursor = 0
        length = len(ddl)

        while cursor < length:
            char = ddl[cursor]

            if char in ("'", '"', "`"):
                closer = char
                out.append(char)
                cursor += 1
                while cursor < length:
                    out.append(ddl[cursor])
                    if ddl[cursor] == closer:
                        cursor += 1
                        break
                    cursor += 1
                continue

            if char == "[":
                while cursor < length:
                    out.append(ddl[cursor])
                    if ddl[cursor] == "]":
                        cursor += 1
                        break
                    cursor += 1
                continue

            if char == "-" and cursor + 1 < length and ddl[cursor + 1] == "-":
                while cursor < length and ddl[cursor] != "\n":
                    cursor += 1
                continue

            if char == "/" and cursor + 1 < length and ddl[cursor + 1] == "*":
                cursor += 2
                while cursor + 1 < length and not (ddl[cursor] == "*" and ddl[cursor + 1] == "/"):
                    cursor += 1
                cursor += 2
                out.append(" ")
                continue

            out.append(char)
            cursor += 1

        return "".join(out)

    # =========================================================================
    # IDENTIFIERS
    # =========================================================================

    @classmethod
    def _unquote_identifier(cls, raw: str) -> str:
        """Strip one layer of ANSI, MySQL or SQL Server quoting."""
        value = raw.strip()
        if len(value) >= 2:
            first, last = value[0], value[-1]
            if first in cls._QUOTE_PAIRS and last == cls._QUOTE_PAIRS[first]:
                return value[1:-1]
        return value

    @classmethod
    def _split_identifier_parts(cls, raw: str) -> list[str]:
        """
        Split a qualified name on dots that sit OUTSIDE quotes.

        `[my.db].[dbo].[Customers]` must split into three parts, not five: the
        dot inside `[my.db]` belongs to the identifier.
        """
        parts: list[str] = []
        current: list[str] = []
        cursor = 0

        while cursor < len(raw):
            char = raw[cursor]

            if char in cls._QUOTE_PAIRS:
                closer = cls._QUOTE_PAIRS[char]
                current.append(char)
                cursor += 1
                while cursor < len(raw) and raw[cursor] != closer:
                    current.append(raw[cursor])
                    cursor += 1
                if cursor < len(raw):
                    current.append(closer)
                    cursor += 1
                continue

            if char == ".":
                parts.append("".join(current))
                current = []
                cursor += 1
                continue

            current.append(char)
            cursor += 1

        parts.append("".join(current))
        return [p for p in parts if p != ""]

    @classmethod
    def _split_qualified_name(cls, table_ref: str) -> tuple[str, str]:
        """
        Split a table reference into (namespace, table_name), unquoting both.

        For a three-part name (database.schema.table) the schema is used as the
        namespace, which is what `[db].[dbo].[Customers]` means in SQL Server.
        """
        parts = cls._split_identifier_parts(table_ref.strip())

        if not parts:
            return "", ""

        table_name = cls._unquote_identifier(parts[-1])
        namespace = cls._unquote_identifier(parts[-2]) if len(parts) >= 2 else ""

        return namespace, table_name

    # =========================================================================
    # COLUMNS
    # =========================================================================

    def _parse_columns(self, columns_str: str) -> list[SchemaField]:
        """
        Parse column definitions from the columns portion of DDL.

        Args:
            columns_str: Content between parentheses in CREATE TABLE

        Returns:
            List of SchemaField instances
        """
        fields = []

        column_defs = self._split_column_definitions(columns_str)

        for raw_col_def in column_defs:
            col_def = raw_col_def.strip()
            if not col_def:
                continue

            # Skip table-level constraints. A bracket/quote-opened definition is
            # always a column, so check the prefix only on bare identifiers --
            # otherwise a column legitimately named [Key] or "Check" is dropped.
            if col_def[0] not in self._QUOTE_PAIRS:
                upper_def = " ".join(col_def.upper().split())
                if upper_def.startswith(self._CONSTRAINT_PREFIXES):
                    continue

            field = self._parse_column(col_def)
            if field:
                fields.append(field)

        return fields

    def _split_column_definitions(self, columns_str: str) -> list[str]:
        """
        Split column definitions by comma, respecting parentheses and quotes.

        DECIMAL(10,2) must not split, and neither must DEFAULT 'a,b'.
        """
        result = []
        current: list[str] = []
        paren_depth = 0
        cursor = 0
        length = len(columns_str)

        while cursor < length:
            char = columns_str[cursor]

            if char in ("'", '"', "`"):
                closer = char
                current.append(char)
                cursor += 1
                while cursor < length:
                    current.append(columns_str[cursor])
                    if columns_str[cursor] == closer:
                        cursor += 1
                        break
                    cursor += 1
                continue

            if char == "[":
                while cursor < length:
                    current.append(columns_str[cursor])
                    if columns_str[cursor] == "]":
                        cursor += 1
                        break
                    cursor += 1
                continue

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

            cursor += 1

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
        col_def = " ".join(col_def.split())
        if not col_def:
            return None

        raw_name, cursor = self._read_identifier(col_def, 0)
        if not raw_name:
            return None

        col_name = self._unquote_identifier(raw_name)
        if not col_name:
            return None

        type_and_constraints = col_def[cursor:].strip()
        if not type_and_constraints:
            # A bare identifier with no type is not a column definition.
            return None

        data_type = self._parse_type(type_and_constraints)
        is_nullable = self._parse_nullable(type_and_constraints)
        default_value = self._parse_default(type_and_constraints)

        return SchemaField(
            name=col_name,
            data_type=data_type,
            full_path=col_name,
            parent_path="",
            description="",
            is_nullable=is_nullable,
            is_array=data_type == DataType.ARRAY or type_and_constraints.rstrip().endswith("[]"),
            array_item_type=None,
            default_value=default_value,
            source_metadata={
                "sql_type": type_and_constraints.split()[0] if type_and_constraints else "",
                "quoted_name": raw_name if raw_name != col_name else None,
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

        # A single-column PRIMARY KEY is implicitly NOT NULL.
        return "PRIMARY KEY" not in upper

    def _parse_default(self, constraints: str) -> str | None:
        """
        Parse DEFAULT value.

        Args:
            constraints: Constraint portion of column definition

        Returns:
            Default value string or None
        """
        match = re.search(
            r"DEFAULT\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))",
            constraints,
            re.IGNORECASE,
        )

        if match:
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

        return bool(self._CREATE_TABLE_RE.search(content))
