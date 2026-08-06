"""
tests.unit.infrastructure.test_sql_ddl_multi_table_and_quoting | Layer: TEST
Regression guards for SqlDdlParser statement boundaries and identifier quoting.

Defect 1 -- statement boundaries. The table body was located with

    re.search(r"CREATE\\s+TABLE\\s+(?:IF\\s+NOT\\s+EXISTS\\s+)?([^\\s(]+)\\s*\\((.*)\\)",
              ddl, re.IGNORECASE | re.DOTALL)

`.*` is greedy under DOTALL, so on a .sql file with several CREATE TABLE
statements it captured everything from the first table's opening paren to the
LAST closing paren in the file. The parser returned one Schema named after the
first table whose column list was a blend of every table -- confidently wrong,
never an error.

Defect 2 -- SQL Server bracket identifiers. `[dbo].[Customers]` was kept
verbatim as the table name, and a column written `[Customer Name] NVARCHAR(100)`
did not match the `(["`]?)(\\w+)\\1` name pattern at all, so it was silently
dropped from the field list.
"""

from __future__ import annotations

import pytest

from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import SqlDdlParser
from nexus_matcher.shared.types.base import DataType


@pytest.fixture
def parser() -> SqlDdlParser:
    return SqlDdlParser()


TWO_TABLES = """
CREATE TABLE customers (
    customer_id INT NOT NULL,
    email VARCHAR(255)
);

CREATE TABLE orders (
    order_id INT NOT NULL,
    order_total DECIMAL(12,2)
);
"""


# =============================================================================
# DEFECT 1: STATEMENT BOUNDARIES
# =============================================================================


class TestMultipleCreateTableStatements:
    def test_first_table_does_not_absorb_later_tables_columns(self, parser):
        """
        The headline symptom: `customers` came back carrying `orders`' columns.
        """
        schema = parser.parse(TWO_TABLES).unwrap()

        names = {f.name for f in schema.fields}

        assert schema.name == "customers"
        assert names == {"customer_id", "email"}, (
            f"first table absorbed columns from later statements: {sorted(names)}"
        )

    def test_parse_all_returns_every_table(self, parser):
        result = parser.parse_all(TWO_TABLES)

        assert result.is_success, result.error
        schemas = result.unwrap()

        assert [s.name for s in schemas] == ["customers", "orders"]
        assert {f.name for f in schemas[1].fields} == {"order_id", "order_total"}

    def test_parse_records_that_more_tables_were_present(self, parser):
        """A caller using parse() must be able to tell data was left behind."""
        schema = parser.parse(TWO_TABLES).unwrap()

        assert schema.source_metadata["table_count"] == 2
        assert schema.source_metadata["additional_tables"] == ["orders"]

    def test_single_table_has_no_extra_metadata(self, parser):
        schema = parser.parse("CREATE TABLE t (id INT);").unwrap()

        assert "additional_tables" not in schema.source_metadata

    def test_three_tables_all_parsed_independently(self, parser):
        ddl = """
        CREATE TABLE a (x INT);
        CREATE TABLE b (y VARCHAR(10), z DATE);
        CREATE TABLE c (w BOOLEAN);
        """

        schemas = parser.parse_all(ddl).unwrap()

        assert [s.name for s in schemas] == ["a", "b", "c"]
        assert [len(s.fields) for s in schemas] == [1, 2, 1]

    def test_paren_in_string_default_does_not_end_the_body(self, parser):
        ddl = """
        CREATE TABLE t (
            status VARCHAR(20) DEFAULT '(pending)',
            note VARCHAR(50)
        );
        """

        schema = parser.parse(ddl).unwrap()

        assert {f.name for f in schema.fields} == {"status", "note"}

    def test_comma_in_string_default_does_not_split_columns(self, parser):
        ddl = "CREATE TABLE t (a VARCHAR(20) DEFAULT 'x,y', b INT);"

        schema = parser.parse(ddl).unwrap()

        assert {f.name for f in schema.fields} == {"a", "b"}

    def test_block_comment_containing_create_table_is_ignored(self, parser):
        ddl = """
        /* CREATE TABLE decoy ( junk INT ) ; */
        CREATE TABLE real_table (id INT);
        """

        schemas = parser.parse_all(ddl).unwrap()

        assert [s.name for s in schemas] == ["real_table"]

    def test_line_comment_containing_create_table_is_ignored(self, parser):
        ddl = """
        -- CREATE TABLE decoy (junk INT);
        CREATE TABLE real_table (id INT);
        """

        schemas = parser.parse_all(ddl).unwrap()

        assert [s.name for s in schemas] == ["real_table"]

    def test_no_create_table_fails(self, parser):
        result = parser.parse("SELECT 1;")

        assert result.is_failure


# =============================================================================
# DEFECT 2: IDENTIFIER QUOTING
# =============================================================================


class TestSqlServerBracketIdentifiers:
    def test_bracket_quoted_table_name_is_unquoted(self, parser):
        schema = parser.parse("CREATE TABLE [dbo].[Customers] (id INT);").unwrap()

        assert schema.name == "Customers", f"got {schema.name!r}"
        assert schema.namespace == "dbo"

    def test_bracket_quoted_column_with_space_is_kept(self, parser):
        ddl = """
        CREATE TABLE [dbo].[Customer Orders] (
            [Order Id] INT NOT NULL,
            [Customer Name] NVARCHAR(200),
            [Order Total] DECIMAL(12,2)
        );
        """

        schema = parser.parse(ddl).unwrap()
        names = {f.name for f in schema.fields}

        assert names == {"Order Id", "Customer Name", "Order Total"}, (
            f"bracket-quoted columns were dropped; got {sorted(names)}"
        )

    def test_bracket_quoted_columns_get_correct_types(self, parser):
        ddl = """
        CREATE TABLE [t] (
            [Customer Name] NVARCHAR(200) NOT NULL,
            [Created On] DATETIME2,
            [Is Active] BIT
        );
        """

        fields = {f.name: f for f in parser.parse(ddl).unwrap().fields}

        assert fields["Customer Name"].data_type == DataType.STRING
        assert fields["Customer Name"].is_nullable is False
        assert fields["Created On"].data_type == DataType.TIMESTAMP
        assert fields["Is Active"].data_type == DataType.BOOLEAN

    def test_three_part_name_uses_schema_as_namespace(self, parser):
        schema = parser.parse("CREATE TABLE [analytics].[dbo].[Facts] (id INT);").unwrap()

        assert schema.namespace == "dbo"
        assert schema.name == "Facts"

    def test_dot_inside_bracket_identifier_is_not_a_separator(self, parser):
        schema = parser.parse("CREATE TABLE [my.schema].[my.table] (id INT);").unwrap()

        assert schema.namespace == "my.schema"
        assert schema.name == "my.table"

    def test_mysql_backtick_identifiers(self, parser):
        ddl = "CREATE TABLE `shop`.`order items` (`item id` BIGINT, `qty` INT);"

        schema = parser.parse(ddl).unwrap()

        assert schema.namespace == "shop"
        assert schema.name == "order items"
        assert {f.name for f in schema.fields} == {"item id", "qty"}

    def test_ansi_double_quoted_identifiers_with_spaces(self, parser):
        ddl = 'CREATE TABLE "public"."My Table" ("My Column" VARCHAR(10));'

        schema = parser.parse(ddl).unwrap()

        assert schema.namespace == "public"
        assert schema.name == "My Table"
        assert [f.name for f in schema.fields] == ["My Column"]

    def test_quoted_name_recorded_in_metadata(self, parser):
        schema = parser.parse("CREATE TABLE t ([Order Id] INT);").unwrap()

        field = schema.fields[0]
        assert field.source_metadata["quoted_name"] == "[Order Id]"

    def test_column_named_like_a_constraint_keyword_survives(self, parser):
        """
        Table-level constraints are skipped by prefix. A quoted column named
        [Key] or [Check] must not be caught by that filter.
        """
        ddl = "CREATE TABLE t ([Key] INT, [Check] VARCHAR(10), [Index] INT);"

        names = {f.name for f in parser.parse(ddl).unwrap().fields}

        assert names == {"Key", "Check", "Index"}

    def test_table_level_constraints_still_skipped(self, parser):
        ddl = """
        CREATE TABLE t (
            id INT NOT NULL,
            name VARCHAR(50),
            PRIMARY KEY (id),
            CONSTRAINT fk_x FOREIGN KEY (name) REFERENCES other(name),
            UNIQUE (name)
        );
        """

        names = {f.name for f in parser.parse(ddl).unwrap().fields}

        assert names == {"id", "name"}


# =============================================================================
# MIXED DIALECT SMOKE TEST
# =============================================================================


class TestMixedDialectFile:
    def test_realistic_multi_dialect_file(self, parser):
        ddl = """
        -- Warehouse definitions
        CREATE TABLE [dbo].[Dim Customer] (
            [Customer Key] INT NOT NULL PRIMARY KEY,
            [Full Name] NVARCHAR(200) NOT NULL,
            [Signup Date] DATE
        );

        /* fact table */
        CREATE TABLE IF NOT EXISTS analytics.fact_orders (
            order_id BIGSERIAL,
            customer_key INT NOT NULL,
            amount DECIMAL(18,4) DEFAULT 0.0000,
            payload JSONB
        );
        """

        schemas = parser.parse_all(ddl).unwrap()

        assert len(schemas) == 2

        dim, fact = schemas
        assert dim.namespace == "dbo"
        assert dim.name == "Dim Customer"
        assert {f.name for f in dim.fields} == {
            "Customer Key",
            "Full Name",
            "Signup Date",
        }

        assert fact.namespace == "analytics"
        assert fact.name == "fact_orders"
        fact_fields = {f.name: f for f in fact.fields}
        assert fact_fields["order_id"].data_type == DataType.LONG
        assert fact_fields["amount"].data_type == DataType.DECIMAL
        assert fact_fields["amount"].default_value == "0.0000"
        assert fact_fields["payload"].data_type == DataType.JSON
