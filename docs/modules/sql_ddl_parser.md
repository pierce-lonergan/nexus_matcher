# Module: SQL DDL Parser

## Purpose

Parse SQL Data Definition Language (DDL) statements (CREATE TABLE) into the unified Schema domain model. Supports common SQL dialects including standard SQL, PostgreSQL, MySQL, and SQL Server.

## Domain Model

### Entities

- **SqlDdlParser**: Parser implementation for SQL DDL format
  - Invariants: Input must contain valid CREATE TABLE statement
  - States: N/A (stateless service)
  - Events: N/A

### Value Objects

- **Schema**: Parsed schema with flattened fields
- **SchemaField**: Individual column with type, constraints, description

### Domain Services

- Uses `BaseSchemaParser` abstract base class from domain/ports

## SQL DDL Structure

### Supported Statements

```sql
CREATE TABLE table_name (
    column_name data_type [constraints],
    ...
);
```

### Type Mappings

| SQL Type | DataType |
|----------|----------|
| VARCHAR, CHAR, TEXT, NVARCHAR | STRING |
| INT, INTEGER, SMALLINT, TINYINT | INTEGER |
| BIGINT | LONG |
| FLOAT, REAL | FLOAT |
| DOUBLE, DOUBLE PRECISION | DOUBLE |
| DECIMAL, NUMERIC, NUMBER | DECIMAL |
| BOOLEAN, BOOL, BIT | BOOLEAN |
| DATE | DATE |
| TIMESTAMP, DATETIME, DATETIME2 | TIMESTAMP |
| TIME | TIMESTAMP |
| BLOB, BYTEA, VARBINARY, BINARY | BYTES |
| JSON, JSONB | JSON |
| UUID, UNIQUEIDENTIFIER | UUID |
| ARRAY | ARRAY |

### Constraint Recognition

| Constraint | Field Property |
|------------|----------------|
| NOT NULL | is_nullable = False |
| NULL | is_nullable = True |
| DEFAULT value | default_value = value |
| PRIMARY KEY | source_metadata["primary_key"] = True |
| UNIQUE | source_metadata["unique"] = True |
| REFERENCES | source_metadata["foreign_key"] = ref |

### Comment Extraction

```sql
-- Column comment inline
column_name VARCHAR(100), -- Description here

-- PostgreSQL COMMENT ON
COMMENT ON COLUMN table.column IS 'Description';
```

## Planned Implementation

- [x] Domain analysis complete
- [x] SqlDdlParser class
- [x] Type mapping (SQL → DataType)
- [x] Constraint parsing (NOT NULL, DEFAULT)
- [x] Comment removal
- [x] Multi-dialect support (PostgreSQL, MySQL, SQL Server)
- [x] Unit tests (45 tests)
- [x] Integration tests (9 tests)

## Configuration

```python
# Environment variables
NEXUS_SQL_DIALECT=postgresql  # Default dialect for ambiguous types
```

## Usage Example

```python
from nexus_matcher.infrastructure.adapters.schema_parsers.sql_ddl import SqlDdlParser

parser = SqlDdlParser()

# Parse from string
result = parser.parse('''
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    balance DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
''')

if result.is_success:
    schema = result.unwrap()
    for field in schema.fields:
        print(f"{field.name}: {field.data_type} nullable={field.is_nullable}")
```

## File Structure

```
src/nexus_matcher/infrastructure/adapters/schema_parsers/
├── __init__.py
├── avro.py           # Avro parser
├── json_schema.py    # JSON Schema parser
└── sql_ddl.py        # SQL DDL parser (new)
```
