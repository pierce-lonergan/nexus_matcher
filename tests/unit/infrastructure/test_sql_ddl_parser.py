"""
tests.unit.infrastructure.test_sql_ddl_parser | Layer: TEST
Tests: SQL DDL parser | Target: src/infrastructure/adapters/schema_parsers/sql_ddl.py

TDD Phase: RED → Tests written before implementation
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from nexus_matcher.shared.types.base import DataType


class TestSqlDdlParserProperties:
    """Test basic parser properties."""

    def test_format_name(self):
        """Test format name is 'sql_ddl'."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        parser = SqlDdlParser()
        assert parser.format_name == "sql_ddl"

    def test_file_extensions(self):
        """Test supported file extensions."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        parser = SqlDdlParser()
        assert ".sql" in parser.file_extensions
        assert ".ddl" in parser.file_extensions


class TestSqlDdlParserStringTypes:
    """Test parsing string type columns."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_parse_varchar(self, parser):
        """Test parsing VARCHAR column."""
        ddl = "CREATE TABLE test (name VARCHAR(100));"

        result = parser.parse(ddl)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.name == "name"
        assert field.data_type == DataType.STRING

    def test_parse_char(self, parser):
        """Test parsing CHAR column."""
        ddl = "CREATE TABLE test (code CHAR(10));"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.STRING

    def test_parse_text(self, parser):
        """Test parsing TEXT column."""
        ddl = "CREATE TABLE test (description TEXT);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.STRING

    def test_parse_nvarchar(self, parser):
        """Test parsing NVARCHAR column (SQL Server)."""
        ddl = "CREATE TABLE test (name NVARCHAR(100));"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.STRING


class TestSqlDdlParserNumericTypes:
    """Test parsing numeric type columns."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_parse_integer(self, parser):
        """Test parsing INTEGER column."""
        ddl = "CREATE TABLE test (count INTEGER);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.INTEGER

    def test_parse_int(self, parser):
        """Test parsing INT column."""
        ddl = "CREATE TABLE test (count INT);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.INTEGER

    def test_parse_smallint(self, parser):
        """Test parsing SMALLINT column."""
        ddl = "CREATE TABLE test (count SMALLINT);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.INTEGER

    def test_parse_bigint(self, parser):
        """Test parsing BIGINT column."""
        ddl = "CREATE TABLE test (big_count BIGINT);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.LONG

    def test_parse_float(self, parser):
        """Test parsing FLOAT column."""
        ddl = "CREATE TABLE test (rate FLOAT);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.FLOAT

    def test_parse_double(self, parser):
        """Test parsing DOUBLE column."""
        ddl = "CREATE TABLE test (precise_rate DOUBLE);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.DOUBLE

    def test_parse_double_precision(self, parser):
        """Test parsing DOUBLE PRECISION column."""
        ddl = "CREATE TABLE test (rate DOUBLE PRECISION);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.DOUBLE

    def test_parse_decimal(self, parser):
        """Test parsing DECIMAL column."""
        ddl = "CREATE TABLE test (amount DECIMAL(10,2));"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.DECIMAL

    def test_parse_numeric(self, parser):
        """Test parsing NUMERIC column."""
        ddl = "CREATE TABLE test (amount NUMERIC(10,2));"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.DECIMAL


class TestSqlDdlParserTemporalTypes:
    """Test parsing temporal type columns."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_parse_date(self, parser):
        """Test parsing DATE column."""
        ddl = "CREATE TABLE test (birth_date DATE);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.DATE

    def test_parse_timestamp(self, parser):
        """Test parsing TIMESTAMP column."""
        ddl = "CREATE TABLE test (created_at TIMESTAMP);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.TIMESTAMP

    def test_parse_datetime(self, parser):
        """Test parsing DATETIME column."""
        ddl = "CREATE TABLE test (created_at DATETIME);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.TIMESTAMP


class TestSqlDdlParserOtherTypes:
    """Test parsing other type columns."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_parse_boolean(self, parser):
        """Test parsing BOOLEAN column."""
        ddl = "CREATE TABLE test (is_active BOOLEAN);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.BOOLEAN

    def test_parse_bool(self, parser):
        """Test parsing BOOL column."""
        ddl = "CREATE TABLE test (is_active BOOL);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.BOOLEAN

    def test_parse_uuid(self, parser):
        """Test parsing UUID column."""
        ddl = "CREATE TABLE test (id UUID);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.UUID

    def test_parse_bytea(self, parser):
        """Test parsing BYTEA column (PostgreSQL)."""
        ddl = "CREATE TABLE test (data BYTEA);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.BYTES

    def test_parse_blob(self, parser):
        """Test parsing BLOB column."""
        ddl = "CREATE TABLE test (data BLOB);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.BYTES

    def test_parse_json(self, parser):
        """Test parsing JSON column."""
        ddl = "CREATE TABLE test (metadata JSON);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.JSON

    def test_parse_jsonb(self, parser):
        """Test parsing JSONB column (PostgreSQL)."""
        ddl = "CREATE TABLE test (metadata JSONB);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].data_type == DataType.JSON


class TestSqlDdlParserConstraints:
    """Test parsing column constraints."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_not_null_constraint(self, parser):
        """Test NOT NULL makes field non-nullable."""
        ddl = "CREATE TABLE test (name VARCHAR(100) NOT NULL);"

        result = parser.parse(ddl)

        assert result.is_success
        assert not result.unwrap().fields[0].is_nullable

    def test_null_constraint(self, parser):
        """Test explicit NULL makes field nullable."""
        ddl = "CREATE TABLE test (name VARCHAR(100) NULL);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].is_nullable

    def test_default_nullable(self, parser):
        """Test columns are nullable by default."""
        ddl = "CREATE TABLE test (name VARCHAR(100));"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().fields[0].is_nullable

    def test_default_value_string(self, parser):
        """Test DEFAULT value extraction for strings."""
        ddl = "CREATE TABLE test (status VARCHAR(20) DEFAULT 'active');"

        result = parser.parse(ddl)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.default_value == "active"

    def test_default_value_number(self, parser):
        """Test DEFAULT value extraction for numbers."""
        ddl = "CREATE TABLE test (balance DECIMAL(10,2) DEFAULT 0.00);"

        result = parser.parse(ddl)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.default_value == "0.00"

    def test_default_value_boolean(self, parser):
        """Test DEFAULT value extraction for booleans."""
        ddl = "CREATE TABLE test (is_active BOOLEAN DEFAULT TRUE);"

        result = parser.parse(ddl)

        assert result.is_success
        field = result.unwrap().fields[0]
        assert field.default_value in ("TRUE", "true", True)


class TestSqlDdlParserTableExtraction:
    """Test table name extraction."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_extracts_table_name(self, parser):
        """Test table name becomes schema name."""
        ddl = "CREATE TABLE customers (id INT);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().name == "customers"

    def test_extracts_qualified_table_name(self, parser):
        """Test qualified table name (schema.table)."""
        ddl = "CREATE TABLE public.customers (id INT);"

        result = parser.parse(ddl)

        assert result.is_success
        schema = result.unwrap()
        assert schema.name == "customers"
        assert schema.namespace == "public"

    def test_case_insensitive_keywords(self, parser):
        """Test SQL keywords are case-insensitive."""
        ddl = "create table TEST (ID int);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().name == "TEST"


class TestSqlDdlParserMultipleColumns:
    """Test parsing multiple columns."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_parse_multiple_columns(self, parser):
        """Test parsing table with multiple columns."""
        ddl = """
        CREATE TABLE customers (
            id UUID,
            first_name VARCHAR(50) NOT NULL,
            last_name VARCHAR(50) NOT NULL,
            email VARCHAR(100),
            balance DECIMAL(10,2) DEFAULT 0.00
        );
        """

        result = parser.parse(ddl)

        assert result.is_success
        fields = result.unwrap().fields

        assert len(fields) == 5
        assert fields[0].name == "id"
        assert fields[1].name == "first_name"
        assert not fields[1].is_nullable
        assert fields[4].name == "balance"
        assert fields[4].default_value == "0.00"


class TestSqlDdlParserFileOperations:
    """Test file parsing operations."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_parse_file(self, parser):
        """Test parsing from file."""
        ddl = "CREATE TABLE test (name VARCHAR(100));"

        with NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
            f.write(ddl)
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result.is_success
            assert len(result.unwrap().fields) == 1

    def test_parse_file_not_found(self, parser):
        """Test error on file not found."""
        result = parser.parse_file(Path("/nonexistent/schema.sql"))

        assert result.is_failure

    def test_parse_file_wrong_extension(self, parser):
        """Test error on wrong extension."""
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("CREATE TABLE test (id INT);")
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result.is_failure


class TestSqlDdlParserCanParse:
    """Test can_parse detection."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_can_parse_create_table(self, parser):
        """Test can_parse returns True for CREATE TABLE."""
        ddl = "CREATE TABLE test (id INT);"

        assert parser.can_parse(ddl)

    def test_cannot_parse_json(self, parser):
        """Test can_parse returns False for JSON."""
        json_content = '{"type": "object", "properties": {}}'

        assert not parser.can_parse(json_content)

    def test_cannot_parse_avro(self, parser):
        """Test can_parse returns False for Avro dict."""
        avro = {"type": "record", "name": "Test", "fields": []}

        assert not parser.can_parse(avro)


class TestSqlDdlParserEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import (
            SqlDdlParser,
        )

        return SqlDdlParser()

    def test_empty_table(self, parser):
        """Test parsing empty table fails gracefully."""
        ddl = "CREATE TABLE empty ();"

        result = parser.parse(ddl)

        # Either succeeds with no fields or fails gracefully
        if result.is_success:
            assert len(result.unwrap().fields) == 0

    def test_invalid_ddl(self, parser):
        """Test invalid DDL fails gracefully."""
        ddl = "NOT VALID SQL"

        result = parser.parse(ddl)

        assert result.is_failure

    def test_source_format_set(self, parser):
        """Test source_format is set correctly."""
        ddl = "CREATE TABLE test (id INT);"

        result = parser.parse(ddl)

        assert result.is_success
        assert result.unwrap().source_format == "sql_ddl"

    def test_handles_inline_comments(self, parser):
        """Test handling inline comments."""
        ddl = """
        CREATE TABLE test (
            id INT, -- Primary key
            name VARCHAR(100) -- Customer name
        );
        """

        result = parser.parse(ddl)

        assert result.is_success
        assert len(result.unwrap().fields) == 2
