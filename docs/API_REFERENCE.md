# NexusMatcher API Reference

> Complete API Documentation for REST, Python, and CLI Interfaces
> 
> **Version**: 1.0.0  
> **Base URL**: `http://localhost:8000` (default)

---

## Table of Contents

1. [REST API](#rest-api)
2. [Python API](#python-api)
3. [CLI Reference](#cli-reference)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)

---

## REST API

### Authentication

Authentication is optional by default. Enable via configuration:

```yaml
# config.yaml
api:
  auth:
    enabled: true
    type: api_key  # or jwt
```

When enabled, include header:
```
Authorization: Bearer <api-key>
```

---

### Health Endpoints

#### GET /health

Check system health status.

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "vector_store": "healthy",
    "cache_l1": "healthy",
    "cache_l2": "healthy",
    "embedding_model": "healthy"
  },
  "uptime_seconds": 3600
}
```

**Status Codes**
| Code | Description |
|------|-------------|
| 200 | All systems healthy |
| 503 | One or more components unhealthy |

---

#### GET /ready

Kubernetes readiness probe.

**Response**
```json
{
  "ready": true
}
```

---

### Matching Endpoints

#### POST /match

Match a single schema to dictionary entries.

**Request Body**
```json
{
  "schema": {
    "type": "record",
    "name": "Customer",
    "namespace": "com.example",
    "fields": [
      {
        "name": "id",
        "type": "long",
        "doc": "Customer identifier"
      },
      {
        "name": "email",
        "type": "string",
        "doc": "Customer email address"
      },
      {
        "name": "addresses",
        "type": {
          "type": "array",
          "items": {
            "type": "record",
            "name": "Address",
            "fields": [
              {"name": "city", "type": "string"},
              {"name": "zip", "type": "string"}
            ]
          }
        }
      }
    ]
  },
  "options": {
    "format": "avro",
    "top_k": 5,
    "min_confidence": 0.5,
    "include_scores": true,
    "include_alternatives": true
  }
}
```

**Request Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| schema | object | Yes | - | Schema to match (Avro, JSON Schema, etc.) |
| options.format | string | No | auto | Schema format: avro, json_schema, sql_ddl |
| options.top_k | integer | No | 5 | Max matches per field (1-100) |
| options.min_confidence | float | No | 0.0 | Minimum confidence threshold (0.0-1.0) |
| options.include_scores | boolean | No | false | Include component scores |
| options.include_alternatives | boolean | No | false | Include alternative matches |

**Response**
```json
{
  "matches": {
    "Customer.id": [
      {
        "dictionary_entry": {
          "id": "DE001",
          "business_name": "Customer Identifier",
          "technical_name": "cust_id",
          "data_type": "bigint",
          "description": "Unique customer identifier",
          "domain": "customer",
          "sensitivity": "internal"
        },
        "confidence": 0.94,
        "decision": "AUTO_APPROVE",
        "rank": 1,
        "scores": {
          "semantic": 0.96,
          "lexical": 0.88,
          "type": 1.0,
          "pattern": 0.85
        }
      },
      {
        "dictionary_entry": {
          "id": "DE015",
          "business_name": "Customer Key",
          "technical_name": "customer_key",
          "data_type": "integer",
          "description": "Alternative customer reference"
        },
        "confidence": 0.72,
        "decision": "REVIEW",
        "rank": 2
      }
    ],
    "Customer.email": [
      {
        "dictionary_entry": {
          "id": "DE002",
          "business_name": "Customer Email",
          "technical_name": "cust_email",
          "data_type": "varchar",
          "description": "Primary email address"
        },
        "confidence": 0.98,
        "decision": "AUTO_APPROVE",
        "rank": 1
      }
    ],
    "Customer.addresses.city": [
      {
        "dictionary_entry": {
          "id": "DE050",
          "business_name": "City Name",
          "technical_name": "city",
          "data_type": "varchar",
          "description": "City or municipality name"
        },
        "confidence": 0.89,
        "decision": "AUTO_APPROVE",
        "rank": 1
      }
    ]
  },
  "metadata": {
    "schema_name": "Customer",
    "fields_total": 4,
    "fields_matched": 4,
    "auto_approved": 3,
    "needs_review": 1,
    "rejected": 0,
    "processing_time_ms": 45,
    "cache_hits": 1,
    "model_version": "1.0.0"
  }
}
```

**Decision Values**

| Decision | Confidence Range | Description |
|----------|------------------|-------------|
| AUTO_APPROVE | ≥0.75 | High confidence, no review needed |
| REVIEW | 0.50-0.74 | Medium confidence, human review suggested |
| REJECT | <0.50 | Low confidence, likely no good match |

**Status Codes**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid schema format |
| 422 | Validation error |
| 500 | Internal server error |

---

#### POST /batch

Match multiple schemas in parallel.

**Request Body**
```json
{
  "schemas": [
    {
      "name": "customer",
      "schema": {...},
      "format": "avro"
    },
    {
      "name": "order",
      "schema": {...},
      "format": "json_schema"
    }
  ],
  "options": {
    "top_k": 3,
    "min_confidence": 0.5,
    "max_parallel": 4
  }
}
```

**Response**
```json
{
  "results": {
    "customer": {
      "matches": {...},
      "metadata": {...}
    },
    "order": {
      "matches": {...},
      "metadata": {...}
    }
  },
  "summary": {
    "schemas_processed": 2,
    "total_fields": 25,
    "auto_approved": 20,
    "needs_review": 4,
    "rejected": 1,
    "total_processing_time_ms": 120
  }
}
```

---

### Dictionary Endpoints

#### GET /dictionary

List dictionary entries with pagination and filtering.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number |
| per_page | integer | 50 | Items per page (max 100) |
| search | string | - | Search in name/description |
| domain | string | - | Filter by domain |
| data_type | string | - | Filter by data type |

**Request**
```
GET /dictionary?page=1&per_page=20&domain=customer
```

**Response**
```json
{
  "entries": [
    {
      "id": "DE001",
      "business_name": "Customer Identifier",
      "technical_name": "cust_id",
      "data_type": "bigint",
      "description": "Unique customer identifier",
      "domain": "customer",
      "owner": "data-team",
      "sensitivity": "internal",
      "created_at": "2025-01-01T00:00:00Z",
      "updated_at": "2025-12-01T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_items": 150,
    "total_pages": 8
  }
}
```

---

#### GET /dictionary/{id}

Get a single dictionary entry.

**Response**
```json
{
  "id": "DE001",
  "business_name": "Customer Identifier",
  "technical_name": "cust_id",
  "data_type": "bigint",
  "description": "Unique customer identifier",
  "domain": "customer",
  "owner": "data-team",
  "sensitivity": "internal",
  "aliases": ["customer_id", "cust_identifier"],
  "metadata": {
    "source_system": "CRM",
    "data_quality": "high"
  },
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-12-01T00:00:00Z"
}
```

---

#### POST /dictionary

Add a new dictionary entry.

**Request Body**
```json
{
  "business_name": "Customer Identifier",
  "technical_name": "cust_id",
  "data_type": "bigint",
  "description": "Unique customer identifier",
  "domain": "customer",
  "owner": "data-team",
  "sensitivity": "internal",
  "aliases": ["customer_id"]
}
```

**Response**
```json
{
  "id": "DE001",
  "business_name": "Customer Identifier",
  "status": "created",
  "indexed": true
}
```

---

#### PUT /dictionary/{id}

Update an existing dictionary entry.

**Request Body**
```json
{
  "description": "Updated description",
  "aliases": ["customer_id", "cust_identifier"]
}
```

**Response**
```json
{
  "id": "DE001",
  "status": "updated",
  "reindexed": true
}
```

---

#### DELETE /dictionary/{id}

Delete a dictionary entry.

**Response**
```json
{
  "id": "DE001",
  "status": "deleted"
}
```

---

### Cache Endpoints

#### POST /cache/clear

Clear all caches.

**Request Body**
```json
{
  "layers": ["l1", "l2", "l3"],  // Optional, defaults to all
  "pattern": "customer.*"         // Optional key pattern
}
```

**Response**
```json
{
  "cleared": {
    "l1": 1500,
    "l2": 3000,
    "l3": 500
  }
}
```

---

#### GET /cache/stats

Get cache statistics.

**Response**
```json
{
  "l1": {
    "size": 4500,
    "max_size": 5000,
    "hit_rate": 0.5699,
    "hits": 15000,
    "misses": 11305,
    "evictions": 2500
  },
  "l2": {
    "size": 12000,
    "hit_rate": 0.42,
    "memory_mb": 256
  },
  "l3": {
    "size": 8000,
    "max_size": 10000,
    "cost_reduction": 0.993
  }
}
```

---

### Metrics Endpoint

#### GET /metrics

Prometheus-compatible metrics.

**Response** (text/plain)
```
# HELP nexus_requests_total Total HTTP requests
# TYPE nexus_requests_total counter
nexus_requests_total{method="POST",endpoint="/match",status="200"} 15234

# HELP nexus_request_duration_seconds Request latency
# TYPE nexus_request_duration_seconds histogram
nexus_request_duration_seconds_bucket{le="0.01"} 5000
nexus_request_duration_seconds_bucket{le="0.05"} 12000
nexus_request_duration_seconds_bucket{le="0.1"} 14500
nexus_request_duration_seconds_bucket{le="0.5"} 15200
nexus_request_duration_seconds_bucket{le="+Inf"} 15234

# HELP nexus_cache_hit_rate Cache hit rate
# TYPE nexus_cache_hit_rate gauge
nexus_cache_hit_rate{layer="l1"} 0.5699
nexus_cache_hit_rate{layer="l2"} 0.42

# HELP nexus_embedding_latency_seconds Embedding generation latency
# TYPE nexus_embedding_latency_seconds histogram
nexus_embedding_latency_seconds_bucket{le="0.005"} 8000
nexus_embedding_latency_seconds_bucket{le="0.01"} 14000
nexus_embedding_latency_seconds_bucket{le="0.02"} 15000
```

---

## Python API

### NexusMatcher Class

Main entry point for schema matching.

```python
from nexus_matcher import NexusMatcher

# Initialize with defaults
matcher = NexusMatcher()

# Initialize with config file
matcher = NexusMatcher(config_path="config.yaml")

# Initialize with explicit configuration
matcher = NexusMatcher(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_backend="qdrant",
    qdrant_host="localhost",
    qdrant_port=6333,
    use_int8=True,
    enable_cache=True,
)
```

#### Methods

##### load_dictionary(path)

Load data dictionary from file.

```python
# From Excel
matcher.load_dictionary("data/dictionary.xlsx")

# From CSV
matcher.load_dictionary("data/dictionary.csv")

# From JSON
matcher.load_dictionary("data/dictionary.json")
```

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| path | str \| Path | Path to dictionary file |

**Raises**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format not supported

---

##### match_schema(path, **options)

Match all fields in a schema file.

```python
results = matcher.match_schema(
    "schemas/customer.avsc",
    top_k=5,
    min_confidence=0.5,
)

for field_path, matches in results.items():
    print(f"\n{field_path}:")
    for match in matches:
        print(f"  {match.dictionary_entry.business_name}")
        print(f"  Confidence: {match.final_confidence:.2%}")
        print(f"  Decision: {match.decision}")
```

**Parameters**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| path | str \| Path | - | Path to schema file |
| top_k | int | 5 | Max matches per field |
| min_confidence | float | 0.0 | Minimum confidence |
| include_scores | bool | False | Include component scores |

**Returns**
- `dict[str, list[Match]]`: Matches keyed by field path

---

##### match_field(field, **options)

Match a single field.

```python
from nexus_matcher.domain.models import Field

field = Field(
    path="customer.email",
    name="email",
    data_type="string",
    description="Customer email address",
)

matches = matcher.match_field(field, top_k=5)
```

**Parameters**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| field | Field | - | Field to match |
| top_k | int | 5 | Max matches |

**Returns**
- `list[Match]`: Sorted by confidence

---

##### parse_schema(path)

Parse a schema file into Schema object.

```python
schema = matcher.parse_schema("schemas/customer.avsc")

print(f"Name: {schema.name}")
print(f"Fields: {len(schema.fields)}")

for field in schema.fields:
    print(f"  {field.path}: {field.data_type}")
```

---

##### set_type_projection_manager(manager)

Enable type-aware matching.

```python
from nexus_matcher.core.type_projections import TypeProjectionManager

type_manager = TypeProjectionManager()
type_manager.load("models/type_projections.pt")

matcher.set_type_projection_manager(type_manager)
```

---

### Match Class

Represents a matching result.

```python
@dataclass
class Match:
    dictionary_entry: DictionaryEntry
    final_confidence: float      # 0.0 - 1.0
    decision: str                # AUTO_APPROVE, REVIEW, REJECT
    scores: dict[str, float]     # Component scores
    rank: int                    # Position in results
```

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| dictionary_entry | DictionaryEntry | Matched dictionary entry |
| final_confidence | float | Combined confidence score |
| decision | str | AUTO_APPROVE, REVIEW, or REJECT |
| scores | dict | Component scores (semantic, lexical, type, pattern) |
| rank | int | Position in results (1-based) |

---

### Field Class

Represents a schema field.

```python
@dataclass(frozen=True)
class Field:
    path: str           # Full path (e.g., "customer.addresses.city")
    name: str           # Field name (e.g., "city")
    data_type: str      # Data type (e.g., "string")
    description: str    # Optional description
    nullable: bool      # Whether field is nullable
    default: Any        # Default value
    metadata: dict      # Additional metadata
```

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| path | str | Full dot-separated path |
| name | str | Field name only |
| data_type | str | Data type string |
| description | str | Human-readable description |
| nullable | bool | True if field can be null |
| depth | int | Nesting level (computed) |
| parent_path | str \| None | Parent field path (computed) |

---

### DictionaryEntry Class

Represents a data dictionary entry.

```python
@dataclass(frozen=True)
class DictionaryEntry:
    id: str
    business_name: str
    technical_name: str
    data_type: str
    description: str
    domain: str
    owner: str
    sensitivity: str
    metadata: dict
```

---

## CLI Reference

### Global Options

```bash
nexus-matcher [OPTIONS] COMMAND [ARGS]

Options:
  --config PATH     Configuration file path
  --log-level TEXT  Logging level (DEBUG, INFO, WARNING, ERROR)
  --help            Show this message and exit
```

### Commands

#### match

Match a schema file to dictionary entries.

```bash
nexus-matcher match SCHEMA_PATH [OPTIONS]

Arguments:
  SCHEMA_PATH  Path to schema file

Options:
  -d, --dictionary PATH  Dictionary file path [required]
  -o, --output PATH      Output file path (default: stdout)
  -f, --format TEXT      Output format (json, csv, table) [default: table]
  -k, --top-k INTEGER    Max matches per field [default: 5]
  -c, --confidence FLOAT Minimum confidence [default: 0.0]
  --include-scores       Include component scores
```

**Examples**

```bash
# Basic usage
nexus-matcher match schema.avsc -d dictionary.xlsx

# JSON output to file
nexus-matcher match schema.avsc -d dictionary.xlsx -o results.json -f json

# Only high confidence
nexus-matcher match schema.avsc -d dictionary.xlsx -c 0.75

# With scores
nexus-matcher match schema.avsc -d dictionary.xlsx --include-scores
```

---

#### batch-match

Match multiple schemas in parallel.

```bash
nexus-matcher batch-match SCHEMA_DIR [OPTIONS]

Arguments:
  SCHEMA_DIR  Directory containing schema files

Options:
  -d, --dictionary PATH  Dictionary file path [required]
  -o, --output PATH      Output directory [default: ./results]
  -w, --workers INTEGER  Parallel workers [default: 4]
  -p, --pattern TEXT     File pattern [default: *.avsc]
```

**Examples**

```bash
# All Avro schemas
nexus-matcher batch-match schemas/ -d dictionary.xlsx

# JSON schemas with 8 workers
nexus-matcher batch-match schemas/ -d dictionary.xlsx -p "*.json" -w 8
```

---

#### sync

Sync dictionary to vector store.

```bash
nexus-matcher sync DICTIONARY_PATH [OPTIONS]

Arguments:
  DICTIONARY_PATH  Path to dictionary file

Options:
  --backend TEXT   Vector store backend (qdrant, memory) [default: qdrant]
  --host TEXT      Qdrant host [default: localhost]
  --port INTEGER   Qdrant port [default: 6333]
  --incremental    Only update changed entries
```

**Examples**

```bash
# Full sync
nexus-matcher sync dictionary.xlsx

# Incremental update
nexus-matcher sync dictionary.xlsx --incremental
```

---

#### api

Start the REST API server.

```bash
nexus-matcher api [OPTIONS]

Options:
  --host TEXT       Host to bind [default: 0.0.0.0]
  --port INTEGER    Port to bind [default: 8000]
  --workers INTEGER Number of workers [default: 4]
  --reload          Enable auto-reload (dev only)
```

**Examples**

```bash
# Production
nexus-matcher api --host 0.0.0.0 --port 8000 --workers 8

# Development
nexus-matcher api --reload
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid schema format",
    "details": {
      "field": "schema.fields[0].type",
      "reason": "Unknown type 'custom_type'"
    }
  },
  "request_id": "abc123"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid request format |
| SCHEMA_PARSE_ERROR | 400 | Failed to parse schema |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

---

## Rate Limiting

When enabled, rate limiting applies per-IP:

| Tier | Requests | Window |
|------|----------|--------|
| Default | 100 | 1 minute |
| Authenticated | 1000 | 1 minute |

**Rate Limit Headers**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

**Rate Limited Response**

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "retry_after": 30
  }
}
```

---

*API Reference Version 1.0.0*
*Last Updated: December 2025*
