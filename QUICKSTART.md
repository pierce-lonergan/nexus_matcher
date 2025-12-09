# NexusMatcher Quick Start Guide

> Get up and running with NexusMatcher in 5 minutes

---

## Installation

### Option 1: Full Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/nexus_matcher.git
cd nexus_matcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all features
pip install -e ".[full]"
```

### Option 2: Minimal Installation

```bash
pip install -e .
```

### Option 3: Docker

```bash
docker compose up -d
```

---

## 5-Minute Tutorial

### Step 1: Prepare Your Data Dictionary

Create a CSV file with your data dictionary entries:

```csv
id,business_name,technical_name,data_type,description,domain
DE001,Customer Identifier,cust_id,bigint,Unique customer identifier,customer
DE002,Customer Email,cust_email,varchar,Primary email address,customer
DE003,First Name,first_name,varchar,Customer first name,customer
DE004,Last Name,last_name,varchar,Customer last name,customer
DE005,Order Amount,order_amount,decimal,Total order value,order
```

Save as `data/dictionary.csv`.

### Step 2: Create a Schema to Match

Create an Avro schema file:

```json
{
  "type": "record",
  "name": "Customer",
  "namespace": "com.example",
  "fields": [
    {"name": "id", "type": "long", "doc": "Customer ID"},
    {"name": "email", "type": "string", "doc": "Email address"},
    {"name": "firstName", "type": "string"},
    {"name": "lastName", "type": "string"}
  ]
}
```

Save as `schemas/customer.avsc`.

### Step 3: Match Schema with Python

```python
from nexus_matcher import NexusMatcher

# Initialize
matcher = NexusMatcher()

# Load dictionary
matcher.load_dictionary("data/dictionary.csv")

# Match schema
results = matcher.match_schema("schemas/customer.avsc")

# Print results
for field_path, matches in results.items():
    top_match = matches[0]
    print(f"\n{field_path}")
    print(f"  → {top_match.dictionary_entry.business_name}")
    print(f"  Confidence: {top_match.final_confidence:.0%}")
    print(f"  Decision: {top_match.decision}")
```

**Expected Output:**
```
Customer.id
  → Customer Identifier
  Confidence: 94%
  Decision: AUTO_APPROVE

Customer.email
  → Customer Email
  Confidence: 98%
  Decision: AUTO_APPROVE

Customer.firstName
  → First Name
  Confidence: 92%
  Decision: AUTO_APPROVE

Customer.lastName
  → Last Name
  Confidence: 95%
  Decision: AUTO_APPROVE
```

### Step 4: Match with CLI

```bash
# Table output
nexus-matcher match schemas/customer.avsc -d data/dictionary.csv

# JSON output
nexus-matcher match schemas/customer.avsc -d data/dictionary.csv -f json -o results.json
```

### Step 5: Start API Server

```bash
# Start server
nexus-matcher api

# In another terminal, test it:
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "schema": {
      "type": "record",
      "name": "Customer",
      "fields": [
        {"name": "id", "type": "long"},
        {"name": "email", "type": "string"}
      ]
    }
  }'
```

---

## Common Use Cases

### Batch Processing

```python
from nexus_matcher.application.use_cases import BatchMatchUseCase

batch = BatchMatchUseCase(matcher, max_workers=4)

results = batch.execute([
    "schemas/customer.avsc",
    "schemas/order.avsc",
    "schemas/product.avsc",
])

for schema, schema_results in results.items():
    auto_approved = sum(
        1 for matches in schema_results.values()
        if matches[0].decision == "AUTO_APPROVE"
    )
    print(f"{schema}: {auto_approved}/{len(schema_results)} auto-approved")
```

### With Type-Aware Matching

```python
from nexus_matcher.core.type_projections import (
    TypeProjectionManager,
    TrainingDataGenerator,
)

# Generate and train type projections
generator = TrainingDataGenerator()
pairs = generator.generate_pairs(1000, 1000)

type_manager = TypeProjectionManager()
type_manager.train(pairs, matcher.embedder.encode)

# Use in matching
matcher.set_type_projection_manager(type_manager)
results = matcher.match_schema("schemas/customer.avsc")
```

### Export Results to Excel

```python
import pandas as pd

# Convert results to DataFrame
rows = []
for field_path, matches in results.items():
    top = matches[0]
    rows.append({
        "Source Field": field_path,
        "Matched Entry": top.dictionary_entry.business_name,
        "Technical Name": top.dictionary_entry.technical_name,
        "Confidence": f"{top.final_confidence:.0%}",
        "Decision": top.decision,
    })

df = pd.DataFrame(rows)
df.to_excel("matching_results.xlsx", index=False)
```

---

## Configuration Options

### Environment Variables

```bash
# Enable INT8 quantization (1.68x faster)
export NEXUS_EMBEDDING_USE_INT8=true

# Use in-memory vector store (no Qdrant needed)
export NEXUS_VECTOR_BACKEND=memory

# Adjust confidence thresholds
export NEXUS_SCORING_AUTO_APPROVE_THRESHOLD=0.80
export NEXUS_SCORING_REVIEW_THRESHOLD=0.60
```

### Configuration File

```yaml
# config.yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  use_int8: true

vector_store:
  backend: memory  # or qdrant

scoring:
  thresholds:
    auto_approve: 0.75
    review: 0.50
```

```python
matcher = NexusMatcher(config_path="config.yaml")
```

---

## Performance Tips

1. **Enable INT8 Quantization** - 1.68x faster embedding generation
   ```python
   matcher = NexusMatcher(use_int8=True)
   ```

2. **Use Caching** - 56.99% hit rate for repeated queries
   ```python
   matcher = NexusMatcher(enable_cache=True)
   ```

3. **Pre-compute MaxSim Embeddings** - 93.7x faster reranking
   ```python
   matcher.precompute_embeddings()
   ```

4. **Batch Processing** - Process multiple schemas in parallel
   ```python
   batch = BatchMatchUseCase(matcher, max_workers=8)
   ```

---

## Troubleshooting

### "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
# or
pip install -e ".[embeddings]"
```

### "Connection refused to Qdrant"

Use in-memory backend:
```bash
export NEXUS_VECTOR_BACKEND=memory
```

Or start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Low Confidence Scores

1. Check if dictionary has relevant entries
2. Verify field descriptions are meaningful
3. Try adding aliases to dictionary entries
4. Enable type-aware matching

---

## Next Steps

- 📖 [Full README](README.md) - Complete documentation
- 🏗️ [Architecture Guide](docs/ARCHITECTURE.md) - System design
- 🚀 [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- 📚 [API Reference](docs/API_REFERENCE.md) - All API endpoints
- 🔬 [Enhancement Journey](docs/ENHANCEMENT_JOURNEY.md) - How we built it

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pierce-lonergan/nexus_matcher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pierce-lonergan/nexus_matcher/discussions)
- **Documentation**: [docs/](docs/)

---

*Happy Matching! 🎯*
