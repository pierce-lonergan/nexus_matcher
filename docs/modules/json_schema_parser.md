# Module: JSON Schema Parser

## Purpose

Parse JSON Schema definitions (draft-04/06/07) into the unified Schema domain model. Supports standard JSON Schema features including object properties, arrays, references ($ref), and common validations.

## Domain Model

### Entities

- **JsonSchemaParser**: Parser implementation for JSON Schema format
  - Invariants: Schema must be valid JSON, must have properties or definitions
  - States: N/A (stateless service)
  - Events: N/A

### Value Objects

- **Schema**: Parsed schema with flattened fields
- **SchemaField**: Individual field with type, description, constraints

### Domain Services

- Uses `BaseSchemaParser` abstract base class from domain/ports

## JSON Schema Structure

### Supported Draft Versions

- Draft-04 (`http://json-schema.org/draft-04/schema#`)
- Draft-06 (`http://json-schema.org/draft-06/schema#`)
- Draft-07 (`http://json-schema.org/draft-07/schema#`)

### Type Mappings

| JSON Schema Type | DataType |
|------------------|----------|
| string | STRING |
| integer | INTEGER |
| number | DOUBLE |
| boolean | BOOLEAN |
| array | ARRAY |
| object | RECORD |
| null | UNKNOWN |

### Format Mappings

| Format | DataType |
|--------|----------|
| date | DATE |
| date-time | TIMESTAMP |
| time | TIMESTAMP |
| uuid | UUID |
| byte | BYTES |
| binary | BYTES |
| email | STRING |
| uri | STRING |

### Supported Features

- `properties` - Object properties
- `required` - Required field list
- `type` - Type specification (single or array)
- `format` - Type format hints
- `description` - Field documentation
- `default` - Default values
- `enum` - Enumeration values
- `items` - Array item type
- `$ref` - Local references (within same schema)
- `oneOf`, `anyOf`, `allOf` - Composition
- `additionalProperties` - Map-like objects

## Planned Implementation

- [x] Domain analysis complete
- [x] JsonSchemaParser class
- [x] Type mapping (JSON Schema → DataType)
- [x] Format mapping
- [x] Nested object flattening
- [x] Array item type extraction
- [x] Enum handling
- [x] Unit tests (31 tests)
- [x] Integration tests (8 tests)

## Configuration

```python
# Environment variables
NEXUS_JSON_SCHEMA_RESOLVE_REFS=true  # Resolve $ref references
NEXUS_JSON_SCHEMA_MAX_DEPTH=10       # Maximum nesting depth
```

## Usage Example

```python
from nexus_matcher.infrastructure.adapters.schema_parsers.json_schema import JsonSchemaParser

parser = JsonSchemaParser()

# Parse from string
result = parser.parse('''
{
  "type": "object",
  "properties": {
    "customer_id": {"type": "string", "format": "uuid"},
    "balance": {"type": "number"},
    "created_at": {"type": "string", "format": "date-time"}
  },
  "required": ["customer_id"]
}
''')

if result.is_success:
    schema = result.unwrap()
    for field in schema.fields:
        print(f"{field.full_path}: {field.data_type}")
        # customer_id: DataType.UUID
        # balance: DataType.DOUBLE
        # created_at: DataType.TIMESTAMP
```

## File Structure

```
src/nexus_matcher/infrastructure/adapters/schema_parsers/
├── __init__.py
├── avro.py           # Existing Avro parser
└── json_schema.py    # New JSON Schema parser
```
