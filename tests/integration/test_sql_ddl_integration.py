"""
tests.integration.test_sql_ddl_integration | Layer: TEST
Integration tests for SQL DDL parser with NexusMatcher.
"""

import pytest
from pathlib import Path

from nexus_matcher.infrastructure.adapters.schema_parsers import (
    SqlDdlParser,
    AvroSchemaParser,
    JsonSchemaParser,
)
from nexus_matcher.domain.ports.schema_parser import SchemaParserRegistry
from nexus_matcher.shared.types.base import DataType


class TestSqlDdlParserIntegration:
    """Integration tests for SQL DDL parser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SqlDdlParser()

    def test_parse_realistic_customer_table(self, parser):
        """Test parsing a realistic customer table."""
        ddl = """
        CREATE TABLE customers (
            customer_id UUID PRIMARY KEY,
            first_name VARCHAR(50) NOT NULL,
            last_name VARCHAR(50) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            phone VARCHAR(20),
            date_of_birth DATE,
            account_balance DECIMAL(15,2) DEFAULT 0.00,
            credit_limit DECIMAL(15,2),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            metadata JSONB
        );
        """

        result = parser.parse(ddl)

        assert result.is_success
        schema = result.unwrap()

        # Check schema metadata
        assert schema.name == "customers"
        assert schema.source_format == "sql_ddl"

        # Check field count
        assert len(schema.fields) == 12

        # Verify specific field types
        fields_by_name = {f.name: f for f in schema.fields}

        assert fields_by_name["customer_id"].data_type == DataType.UUID
        assert fields_by_name["first_name"].data_type == DataType.STRING
        assert not fields_by_name["first_name"].is_nullable
        assert fields_by_name["date_of_birth"].data_type == DataType.DATE
        assert fields_by_name["account_balance"].data_type == DataType.DECIMAL
        assert fields_by_name["account_balance"].default_value == "0.00"
        assert fields_by_name["is_active"].data_type == DataType.BOOLEAN
        assert fields_by_name["created_at"].data_type == DataType.TIMESTAMP
        assert fields_by_name["metadata"].data_type == DataType.JSON

        # Check nullable fields
        assert fields_by_name["phone"].is_nullable
        assert not fields_by_name["email"].is_nullable

    def test_parse_banking_transaction_table(self, parser):
        """Test parsing a banking transaction table."""
        ddl = """
        CREATE TABLE IF NOT EXISTS banking.transactions (
            transaction_id BIGSERIAL PRIMARY KEY,
            account_number VARCHAR(20) NOT NULL,
            transaction_type VARCHAR(20) NOT NULL,
            amount DECIMAL(18,2) NOT NULL,
            currency CHAR(3) DEFAULT 'USD',
            transaction_date TIMESTAMP WITH TIME ZONE NOT NULL,
            description TEXT,
            reference_number VARCHAR(50),
            is_reversed BOOLEAN DEFAULT FALSE,
            reversal_date TIMESTAMP,
            created_by VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        result = parser.parse(ddl)

        assert result.is_success
        schema = result.unwrap()

        # Check qualified name parsing
        assert schema.name == "transactions"
        assert schema.namespace == "banking"

        fields_by_name = {f.name: f for f in schema.fields}

        # Check types
        assert fields_by_name["transaction_id"].data_type == DataType.LONG
        assert fields_by_name["amount"].data_type == DataType.DECIMAL
        assert fields_by_name["currency"].data_type == DataType.STRING
        assert fields_by_name["description"].data_type == DataType.STRING
        assert fields_by_name["is_reversed"].data_type == DataType.BOOLEAN
        assert fields_by_name["transaction_date"].data_type == DataType.TIMESTAMP


class TestSchemaParserRegistryWithSqlDdl:
    """Test schema parser registry with SQL DDL."""

    def test_registry_detects_sql_ddl(self):
        """Test registry can detect SQL DDL."""
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())
        registry.register(JsonSchemaParser())
        registry.register(SqlDdlParser())

        ddl = "CREATE TABLE test (id INT);"

        detected = registry.detect_parser(ddl)

        assert detected is not None
        assert detected.format_name == "sql_ddl"

    def test_registry_distinguishes_formats(self):
        """Test registry distinguishes all three formats."""
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())
        registry.register(JsonSchemaParser())
        registry.register(SqlDdlParser())

        sql_ddl = "CREATE TABLE test (id INT);"
        json_schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        avro_schema = {"type": "record", "name": "Test", "fields": [{"name": "id", "type": "int"}]}

        assert registry.detect_parser(sql_ddl).format_name == "sql_ddl"
        assert registry.detect_parser(json_schema).format_name == "json_schema"
        assert registry.detect_parser(avro_schema).format_name == "avro"

    def test_registry_get_parser_for_sql_file(self):
        """Test getting parser by .sql extension."""
        registry = SchemaParserRegistry()
        registry.register(SqlDdlParser())

        parser = registry.get_parser_for_file(Path("schema.sql"))

        assert parser is not None
        assert parser.format_name == "sql_ddl"

    def test_registry_lists_all_formats(self):
        """Test listing all registered formats."""
        registry = SchemaParserRegistry()
        registry.register(AvroSchemaParser())
        registry.register(JsonSchemaParser())
        registry.register(SqlDdlParser())

        formats = registry.list_formats()

        assert "avro" in formats
        assert "json_schema" in formats
        assert "sql_ddl" in formats


class TestSqlDdlWithComplexTypes:
    """Test SQL DDL parser with complex type scenarios."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SqlDdlParser()

    def test_postgresql_specific_types(self, parser):
        """Test PostgreSQL-specific types."""
        ddl = """
        CREATE TABLE pg_types (
            id SERIAL PRIMARY KEY,
            big_id BIGSERIAL,
            data BYTEA,
            metadata JSONB,
            tags TEXT[],
            ip_addr INET,
            mac_addr MACADDR
        );
        """

        result = parser.parse(ddl)
        assert result.is_success

        fields_by_name = {f.name: f for f in result.unwrap().fields}

        assert fields_by_name["id"].data_type == DataType.INTEGER
        assert fields_by_name["big_id"].data_type == DataType.LONG
        assert fields_by_name["data"].data_type == DataType.BYTES
        assert fields_by_name["metadata"].data_type == DataType.JSON

    def test_mysql_specific_types(self, parser):
        """Test MySQL-specific types."""
        ddl = """
        CREATE TABLE mysql_types (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tiny TINYINT,
            small SMALLINT,
            medium MEDIUMINT,
            big BIGINT,
            data BLOB,
            text_data LONGTEXT,
            ts DATETIME
        );
        """

        result = parser.parse(ddl)
        assert result.is_success

        fields_by_name = {f.name: f for f in result.unwrap().fields}

        assert fields_by_name["id"].data_type == DataType.INTEGER
        assert fields_by_name["tiny"].data_type == DataType.INTEGER
        assert fields_by_name["big"].data_type == DataType.LONG
        assert fields_by_name["data"].data_type == DataType.BYTES
        assert fields_by_name["ts"].data_type == DataType.TIMESTAMP

    def test_sql_server_specific_types(self, parser):
        """Test SQL Server-specific types."""
        ddl = """
        CREATE TABLE mssql_types (
            id UNIQUEIDENTIFIER PRIMARY KEY,
            name NVARCHAR(100),
            data VARBINARY(MAX),
            ts DATETIME2,
            active BIT
        );
        """

        result = parser.parse(ddl)
        assert result.is_success

        fields_by_name = {f.name: f for f in result.unwrap().fields}

        assert fields_by_name["id"].data_type == DataType.UUID
        assert fields_by_name["name"].data_type == DataType.STRING
        assert fields_by_name["data"].data_type == DataType.BYTES
        assert fields_by_name["ts"].data_type == DataType.TIMESTAMP
        assert fields_by_name["active"].data_type == DataType.BOOLEAN
