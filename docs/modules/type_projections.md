# Type Projections Module (GAP-008)

> Learned Type Embeddings via Contrastive Learning
> 
> **Location**: `src/nexus_matcher/core/type_projections.py`
> **Status**: EXPERIMENTAL — not part of the benchmarked matching path
>
> The frequently quoted **MRR 0.9706** comes from `benchmarks/suite_008_combined.py`,
> which evaluates on **17 hand-written source fields against a 20-entry hand-written
> target set** and never calls `NexusMatcher`. It is not a system result. The two
> archived runs of that suite disagree with each other on the related figures
> (`test_accuracy` 0.89 vs 0.905), which is what you would expect from a 17-pair sample.
>
> Type projections are **not enabled in the default pipeline**. Type information is used
> as a numeric scoring signal (`type_weight = 0.05`) instead. Separately, injecting type
> information *as query text* was measured to cost 2.1 points of P@1.

---

## Overview

The Type Projections module implements learned type embeddings that encode data type semantics into the embedding space. This addresses the challenge of disambiguating fields with similar names but different data types.

### Problem Statement

Consider these field pairs:
- `customer_id` (string UUID) vs `customer_id` (integer autoincrement)
- `amount` (decimal currency) vs `amount` (string description)
- `date` (ISO date string) vs `date` (timestamp)

Pure semantic matching fails here because the names are identical. Type-aware embeddings solve this by incorporating type information into the similarity computation.

### Solution Approach

We use **contrastive learning** to train a projection network that:
1. Takes base text embeddings (384d from MiniLM)
2. Concatenates learned type embeddings (64d)
3. Projects to a unified space (384d)
4. Maximizes similarity for matching pairs, minimizes for non-matching

---

## Architecture

```
                           ┌──────────────────────────────────────────────┐
                           │              TYPE PROJECTION SYSTEM          │
                           │                                              │
Input Text ───────────────▶│  ┌─────────────────────────────────────────┐│
"customer email"           │  │         Base Embedding Model            ││
                           │  │         (MiniLM-L6-v2)                   ││
                           │  │                                          ││
                           │  │   "customer email" ──▶ [0.12, -0.34, ...]││
                           │  │                         (384 dimensions) ││
                           │  └────────────────┬────────────────────────┘│
                           │                   │                         │
                           │                   ▼                         │
Data Type ────────────────▶│  ┌─────────────────────────────────────────┐│
"varchar"                  │  │         Type Embedding Layer            ││
                           │  │                                          ││
                           │  │   TypeVocabulary: "varchar" ──▶ ID 5    ││
                           │  │   nn.Embedding(32, 64)                   ││
                           │  │                                          ││
                           │  │   ID 5 ──▶ [0.45, 0.12, ...]            ││
                           │  │             (64 dimensions)              ││
                           │  └────────────────┬────────────────────────┘│
                           │                   │                         │
                           │                   ▼                         │
                           │  ┌─────────────────────────────────────────┐│
                           │  │         Concatenation                   ││
                           │  │                                          ││
                           │  │   [base_384] + [type_64] = [448]        ││
                           │  └────────────────┬────────────────────────┘│
                           │                   │                         │
                           │                   ▼                         │
                           │  ┌─────────────────────────────────────────┐│
                           │  │         Projection Network              ││
                           │  │                                          ││
                           │  │   Linear(448 → 768)                     ││
                           │  │   ReLU                                   ││
                           │  │   Dropout(0.1)                          ││
                           │  │   Linear(768 → 384)                     ││
                           │  │   LayerNorm                              ││
                           │  └────────────────┬────────────────────────┘│
                           │                   │                         │
                           │                   ▼                         │
                           │         Type-Aware Embedding (384d)         │
                           └──────────────────────────────────────────────┘
```

---

## Components

### 1. TypeVocabulary

Maps data type names to integer IDs for embedding lookup.

```python
from nexus_matcher.core.type_projections import TypeVocabulary

vocab = TypeVocabulary()

# Built-in types
vocab.get_id("string")     # → 1
vocab.get_id("varchar")    # → 1 (normalized to "string")
vocab.get_id("int")        # → 2
vocab.get_id("integer")    # → 2 (normalized)
vocab.get_id("decimal")    # → 4
vocab.get_id("float")      # → 4 (normalized)

# Unknown types
vocab.get_id("custom_type")  # → 0 (UNK token)

# Properties
vocab.size  # → 32 (vocabulary size)
```

**Type Normalization**:

| Input Types | Normalized To | ID |
|-------------|---------------|-----|
| string, varchar, text, char | string | 1 |
| int, integer, bigint, smallint | integer | 2 |
| bool, boolean | boolean | 3 |
| decimal, numeric, float, double, real | decimal | 4 |
| date | date | 5 |
| timestamp, datetime | timestamp | 6 |
| binary, bytes, blob | binary | 7 |
| array, list | array | 8 |
| map, dict, object | map | 9 |
| record, struct | record | 10 |

### 2. TypeProjectionLayer

Neural network that combines base embeddings with type embeddings.

```python
from nexus_matcher.core.type_projections import TypeProjectionLayer
import torch

# Initialize layer
layer = TypeProjectionLayer(
    base_dim=384,      # MiniLM embedding dimension
    type_dim=64,       # Learned type embedding dimension
    output_dim=384,    # Output matches base for compatibility
    num_types=32,      # Vocabulary size
)

# Forward pass
base_embedding = torch.randn(1, 384)  # From MiniLM
type_id = torch.tensor([1])           # "string"

output = layer(base_embedding, type_id)  # Shape: (1, 384)
```

**Network Architecture**:

```
Input: base_embedding (B, 384) + type_id (B,)
    │
    ├── type_embedding = nn.Embedding(32, 64)[type_id]  → (B, 64)
    │
    └── combined = concat(base_embedding, type_embedding)  → (B, 448)
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  nn.Linear(448, 768)                    │
    │  nn.ReLU()                              │
    │  nn.Dropout(0.1)                        │
    │  nn.Linear(768, 384)                    │
    │  nn.LayerNorm(384)                      │
    └─────────────────────────────────────────┘
           │
           ▼
    Output: type_aware_embedding (B, 384)
```

### 3. ContrastiveTypeModel

Full model with InfoNCE-style contrastive loss for training.

```python
from nexus_matcher.core.type_projections import ContrastiveTypeModel

model = ContrastiveTypeModel(projection_layer)

# Training forward pass
source_emb = torch.randn(32, 384)
source_types = torch.randint(0, 10, (32,))
target_emb = torch.randn(32, 384)
target_types = torch.randint(0, 10, (32,))
labels = torch.randint(0, 2, (32,))  # 1 = positive pair, 0 = negative

loss, accuracy = model(
    source_emb, source_types,
    target_emb, target_types,
    labels
)
```

**Loss Function**:

```python
def contrastive_loss(source_proj, target_proj, labels):
    # Cosine similarity with temperature scaling
    source_norm = F.normalize(source_proj, dim=1)
    target_norm = F.normalize(target_proj, dim=1)
    similarity = (source_norm * target_norm).sum(dim=1) / temperature
    
    # Binary cross-entropy
    loss = F.binary_cross_entropy_with_logits(similarity, labels.float())
    return loss
```

### 4. TrainingDataGenerator

Generates synthetic training pairs for contrastive learning.

```python
from nexus_matcher.core.type_projections import TrainingDataGenerator

generator = TrainingDataGenerator()

# Generate training data
pairs = generator.generate_pairs(
    num_positive=1000,  # Matching pairs
    num_negative=1000,  # Non-matching pairs
)

# Each pair contains:
for pair in pairs[:3]:
    print(f"Source: {pair.source_text} ({pair.source_type})")
    print(f"Target: {pair.target_text} ({pair.target_type})")
    print(f"Label: {pair.label}")  # 1 = match, 0 = no match
    print()

# Example output:
# Source: customer_email (string)
# Target: cust_email_address (varchar)
# Label: 1
#
# Source: order_amount (decimal)
# Target: phone_number (string)
# Label: 0
```

**Built-in Field Templates**:

```python
FIELD_TEMPLATES = {
    "email": {
        "variants": ["email", "email_address", "cust_email", "customer_email", "e_mail"],
        "types": ["string", "varchar"],
    },
    "phone": {
        "variants": ["phone", "phone_number", "phone_num", "tel", "telephone", "mobile"],
        "types": ["string", "varchar"],
    },
    "amount": {
        "variants": ["amount", "total_amount", "order_amount", "payment_amount", "amt"],
        "types": ["decimal", "numeric", "float"],
    },
    "id": {
        "variants": ["id", "identifier", "key", "pk", "primary_key"],
        "types": ["integer", "bigint", "string"],
    },
    # ... 20+ more templates
}
```

### 5. TypeProjectionManager

High-level API for training and inference.

```python
from nexus_matcher.core.type_projections import TypeProjectionManager

# Initialize
manager = TypeProjectionManager(
    base_dim=384,
    type_dim=64,
    output_dim=384,
)

# Generate training data
generator = TrainingDataGenerator()
pairs = generator.generate_pairs(num_positive=1000, num_negative=1000)

# Train
manager.train(
    pairs=pairs,
    encode_fn=embedder.encode,  # Your embedding function
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-4,
)

# Training output:
# Epoch 1/5: Loss=0.5792, Accuracy=77.4%
# Epoch 2/5: Loss=0.2675, Accuracy=91.0%
# Epoch 3/5: Loss=0.1936, Accuracy=94.8%
# Epoch 4/5: Loss=0.1616, Accuracy=96.0%
# Epoch 5/5: Loss=0.1310, Accuracy=97.4%

# Inference
type_aware_embedding = manager.project(
    base_embedding=embedder.encode("customer email"),
    data_type="varchar",
)

# Save/Load
manager.save("models/type_projections.pt")
manager.load("models/type_projections.pt")
```

---

## Training Guide

### Step 1: Prepare Training Data

```python
from nexus_matcher.core.type_projections import (
    TypeProjectionManager,
    TrainingDataGenerator,
)
from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
    SentenceTransformersProvider,
)

# Initialize embedder
embedder = SentenceTransformersProvider(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate training pairs
generator = TrainingDataGenerator()
training_pairs = generator.generate_pairs(
    num_positive=2000,
    num_negative=2000,
)
```

### Step 2: Train the Model

```python
# Initialize manager
manager = TypeProjectionManager(
    base_dim=embedder.dimension,  # 384
    type_dim=64,
    output_dim=embedder.dimension,  # 384
)

# Train
history = manager.train(
    pairs=training_pairs,
    encode_fn=embedder.encode,
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    validation_split=0.1,
)

# Plot training curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig('training_curve.png')
```

### Step 3: Evaluate

```python
# Generate test pairs
test_pairs = generator.generate_pairs(
    num_positive=200,
    num_negative=200,
)

# Evaluate
metrics = manager.evaluate(
    pairs=test_pairs,
    encode_fn=embedder.encode,
)

print(f"Test Accuracy: {metrics['accuracy']:.2%}")
print(f"Positive Similarity: {metrics['avg_positive_sim']:.4f}")
print(f"Negative Similarity: {metrics['avg_negative_sim']:.4f}")
print(f"Separation: {metrics['separation']:.4f}")
```

### Step 4: Save and Deploy

```python
# Save model
manager.save("models/type_projections.pt")

# Load for production
production_manager = TypeProjectionManager()
production_manager.load("models/type_projections.pt")
```

---

## Integration with NexusMatcher

### Option 1: Direct Integration

```python
from nexus_matcher import NexusMatcher
from nexus_matcher.core.type_projections import TypeProjectionManager

# Initialize matcher
matcher = NexusMatcher.from_config()

# Load trained type projections
type_manager = TypeProjectionManager()
type_manager.load("models/type_projections.pt")

# Set type projection manager
matcher.set_type_projection_manager(type_manager)

# Match with type awareness
results = matcher.match_schema("schemas/customer.avsc")
```

### Option 2: Custom Pipeline

```python
def type_aware_match(field, dictionary_entries, embedder, type_manager):
    """Custom type-aware matching pipeline."""
    
    # Get base embedding
    query_text = f"{field.name} {field.description}"
    base_embedding = embedder.encode(query_text)
    
    # Project with type info
    query_projected = type_manager.project(
        base_embedding=base_embedding,
        data_type=field.data_type,
    )
    
    # Compare with dictionary entries
    scores = []
    for entry in dictionary_entries:
        entry_base = embedder.encode(entry.to_searchable_text())
        entry_projected = type_manager.project(
            base_embedding=entry_base,
            data_type=entry.data_type,
        )
        
        # Cosine similarity
        similarity = cosine_similarity(query_projected, entry_projected)
        scores.append((entry, similarity))
    
    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
```

---

## Benchmark Results

### Training Performance

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 0.5792 | 77.4% |
| 2 | 0.2675 | 91.0% |
| 3 | 0.1936 | 94.8% |
| 4 | 0.1616 | 96.0% |
| 5 | 0.1310 | **97.4%** |

### Evaluation Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Test Accuracy | 89.0% | - |
| Schema Matching MRR | 0.9706 on a 17-pair toy set | not a system result — see the header |
| Avg Positive Similarity | 0.6387 | - |
| Avg Negative Similarity | -0.0847 | - |
| Separation | 0.7233 | >0.5 ✓ |

### Latency Impact

| Operation | Latency |
|-----------|---------|
| Type ID lookup | 0.001ms |
| Projection forward pass | 0.5ms |
| Total overhead | ~0.5ms per embedding |

---

## Advanced Topics

### Custom Type Vocabulary

```python
from nexus_matcher.core.type_projections import TypeVocabulary

# Create custom vocabulary
custom_vocab = TypeVocabulary()

# Add domain-specific types
custom_vocab.add_type("ssn", normalize_to="string")
custom_vocab.add_type("ein", normalize_to="string")
custom_vocab.add_type("currency_code", normalize_to="string")
custom_vocab.add_type("country_code", normalize_to="string")

# Use with manager
manager = TypeProjectionManager(
    type_vocabulary=custom_vocab,
)
```

### Custom Training Data

```python
from nexus_matcher.core.type_projections import TrainingPair

# Create custom training pairs
custom_pairs = [
    TrainingPair(
        source_text="social_security_number",
        source_type="ssn",
        target_text="ssn",
        target_type="string",
        label=1,
    ),
    TrainingPair(
        source_text="tax_id",
        source_type="ein",
        target_text="employer_identification",
        target_type="string",
        label=1,
    ),
]

# Combine with generated pairs
all_pairs = generator.generate_pairs(1000, 1000) + custom_pairs
```

### Fine-tuning on Domain Data

```python
# Load pre-trained model
manager = TypeProjectionManager()
manager.load("models/type_projections.pt")

# Fine-tune on domain-specific data
domain_pairs = load_domain_training_data()

manager.train(
    pairs=domain_pairs,
    encode_fn=embedder.encode,
    num_epochs=3,
    learning_rate=1e-5,  # Lower LR for fine-tuning
)

# Save fine-tuned model
manager.save("models/type_projections_finetuned.pt")
```

---

## Troubleshooting

### Common Issues

**Issue**: Low training accuracy
```
Solution: Increase training data or epochs
- Try 5000+ positive/negative pairs
- Train for 10+ epochs
- Check for label noise in training data
```

**Issue**: High training accuracy but poor test performance
```
Solution: Model is overfitting
- Add more dropout (0.2-0.3)
- Reduce model capacity
- Use early stopping
- Add more diverse training data
```

**Issue**: Slow inference
```
Solution: Batch projections
- Project multiple embeddings at once
- Pre-compute dictionary projections at indexing time
```

---

## API Reference

### TypeVocabulary

```python
class TypeVocabulary:
    def __init__(self) -> None: ...
    def get_id(self, type_name: str) -> int: ...
    def add_type(self, type_name: str, normalize_to: str | None = None) -> int: ...
    @property
    def size(self) -> int: ...
```

### TypeProjectionLayer

```python
class TypeProjectionLayer(nn.Module):
    def __init__(
        self,
        base_dim: int = 384,
        type_dim: int = 64,
        output_dim: int = 384,
        num_types: int = 32,
    ) -> None: ...
    
    def forward(
        self,
        base_embedding: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor: ...
```

### TypeProjectionManager

```python
class TypeProjectionManager:
    def __init__(
        self,
        base_dim: int = 384,
        type_dim: int = 64,
        output_dim: int = 384,
        type_vocabulary: TypeVocabulary | None = None,
    ) -> None: ...
    
    def train(
        self,
        pairs: list[TrainingPair],
        encode_fn: Callable[[str], np.ndarray],
        num_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        validation_split: float = 0.1,
    ) -> dict[str, list[float]]: ...
    
    def project(
        self,
        base_embedding: np.ndarray,
        data_type: str,
    ) -> np.ndarray: ...
    
    def evaluate(
        self,
        pairs: list[TrainingPair],
        encode_fn: Callable[[str], np.ndarray],
    ) -> dict[str, float]: ...
    
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

---

*Module Documentation Version 1.0.0*
*Last Updated: December 2025*
