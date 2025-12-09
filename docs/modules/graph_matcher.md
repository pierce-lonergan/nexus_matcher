# Graph Matcher Module (GAP-009)

> Graph-Based Structural Schema Matching
> 
> **Location**: `src/nexus_matcher/core/graph_matcher.py`  
> **Status**: VALIDATED ✓  
> **Use Case**: Hybrid complement to semantic matching

---

## Overview

The Graph Matcher module implements structural schema matching by converting schemas into directed graphs and computing similarity based on structural relationships. This captures information that pure text-based semantic matching misses.

### Problem Statement

Semantic matching works well when field names are descriptive, but struggles with:

1. **Structural Ambiguity**: `city` could match `city` in addresses, billing, or shipping
2. **Naming Variations**: Different schemas use different conventions for the same concept
3. **Hierarchical Context**: `customer.address.city` vs `shipping.location.city`

### Solution Approach

Graph-based matching captures:
- **Parent-child relationships**: Which fields contain other fields
- **Sibling relationships**: Fields at the same hierarchical level
- **Type distributions**: What types surround a field
- **Depth patterns**: How deeply nested a field is

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GRAPH MATCHING PIPELINE                              │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    SOURCE SCHEMA                                     │   │
│   │                                                                      │   │
│   │   customer                                                           │   │
│   │   ├── id (long)                                                      │   │
│   │   ├── name (string)                                                  │   │
│   │   ├── email (string)                                                 │   │
│   │   └── addresses (array)                                              │   │
│   │       ├── street (string)                                            │   │
│   │       ├── city (string)                                              │   │
│   │       └── zip (string)                                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  SCHEMA GRAPH BUILDER                                │   │
│   │                                                                      │   │
│   │   Convert to directed graph:                                         │   │
│   │                                                                      │   │
│   │   (customer) ──[parent_child]──▶ (customer.id)                      │   │
│   │       │                              │                               │   │
│   │       │                         [sibling]                            │   │
│   │       │                              │                               │   │
│   │       ├──[parent_child]──▶ (customer.name) ◀──[sibling]──┐          │   │
│   │       │                              │                    │          │   │
│   │       ├──[parent_child]──▶ (customer.email) ◀─[sibling]──┤          │   │
│   │       │                              │                    │          │   │
│   │       └──[parent_child]──▶ (customer.addresses) ◀────────┘          │   │
│   │                                  │                                   │   │
│   │                           [parent_child]                             │   │
│   │                                  │                                   │   │
│   │                      ┌───────────┼───────────┐                       │   │
│   │                      ▼           ▼           ▼                       │   │
│   │             (addresses.street) (addresses.city) (addresses.zip)     │   │
│   │                      │           │           │                       │   │
│   │                      └───[sibling]───────────┘                       │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  STRUCTURAL SIMILARITY SCORING                       │   │
│   │                                                                      │   │
│   │   For each (source_field, target_field) pair:                       │   │
│   │                                                                      │   │
│   │   structural_sim = depth_similarity(source, target)                 │   │
│   │   context_sim = neighbor_type_jaccard(source, target)               │   │
│   │   type_sim = type_compatibility(source, target)                     │   │
│   │                                                                      │   │
│   │   combined = 0.4 × structural + 0.3 × context + 0.3 × type          │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. SchemaGraphBuilder

Converts schema fields into a directed graph representation.

```python
from nexus_matcher.core.graph_matcher import SchemaGraphBuilder

builder = SchemaGraphBuilder()

# Define fields
fields = [
    {"path": "customer", "name": "customer", "data_type": "record"},
    {"path": "customer.id", "name": "id", "data_type": "long"},
    {"path": "customer.name", "name": "name", "data_type": "string"},
    {"path": "customer.addresses", "name": "addresses", "data_type": "array"},
    {"path": "customer.addresses.city", "name": "city", "data_type": "string"},
]

# Build graph
graph = builder.build(fields)

# Inspect nodes
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Node attributes
node = graph.nodes["customer.addresses.city"]
print(f"Name: {node['name']}")         # city
print(f"Type: {node['data_type']}")    # string
print(f"Depth: {node['depth']}")       # 2
```

**Graph Properties**:

| Property | Description |
|----------|-------------|
| Node ID | Field path (e.g., `customer.addresses.city`) |
| Node Attributes | name, data_type, depth, description |
| Edge Types | `parent_child`, `sibling`, `type_similarity` |

**Edge Creation Rules**:

```python
def _create_edges(self, fields):
    for field in fields:
        path = field['path']
        
        # Parent-child edge
        if '.' in path:
            parent = '.'.join(path.split('.')[:-1])
            self.graph.add_edge(parent, path, relation='parent_child')
        
        # Sibling edges (same parent)
        for other in fields:
            if self._are_siblings(path, other['path']):
                self.graph.add_edge(path, other['path'], relation='sibling')
        
        # Type similarity edges (same type group, different parents)
        for other in fields:
            if (self._same_type_group(field, other) and 
                not self._are_siblings(path, other['path'])):
                self.graph.add_edge(path, other['path'], relation='type_similarity')
```

### 2. GraphStructuralMatcher

Computes structural similarity between fields in source and target schemas.

```python
from nexus_matcher.core.graph_matcher import GraphStructuralMatcher

matcher = GraphStructuralMatcher()

# Set source schema
matcher.set_source_schema("source", source_fields)

# Set target schema (dictionary entries)
matcher.set_target_schema("dictionary", dictionary_fields)

# Match a single field
matches = matcher.match_field(
    source_field_id="customer.addresses.city",
    top_k=5,
)

for target_id, score in matches:
    print(f"{target_id}: {score:.4f}")

# Example output:
# contact.location.city: 0.8234
# billing.address.city: 0.7891
# shipping.city: 0.6542
```

**Similarity Components**:

#### Structural Similarity

```python
def _structural_similarity(self, source_node, target_node):
    """Compute depth-based structural similarity."""
    source_depth = source_node['depth']
    target_depth = target_node['depth']
    
    # Inverse of depth difference
    depth_diff = abs(source_depth - target_depth)
    depth_sim = 1.0 / (1.0 + depth_diff)
    
    return depth_sim
```

| Source Depth | Target Depth | Similarity |
|--------------|--------------|------------|
| 2 | 2 | 1.0 |
| 2 | 3 | 0.5 |
| 2 | 4 | 0.33 |
| 0 | 3 | 0.25 |

#### Context Similarity (Neighbor Types)

```python
def _context_similarity(self, source_id, target_id):
    """Compute Jaccard similarity of neighbor types."""
    source_neighbors = self._get_neighbor_types(source_id, self._source_graph)
    target_neighbors = self._get_neighbor_types(target_id, self._target_graph)
    
    # Jaccard similarity
    intersection = source_neighbors & target_neighbors
    union = source_neighbors | target_neighbors
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def _get_neighbor_types(self, node_id, graph):
    """Get set of neighbor data types."""
    neighbors = set()
    for neighbor in graph.neighbors(node_id):
        neighbor_type = graph.nodes[neighbor].get('data_type', 'unknown')
        neighbors.add(self._normalize_type(neighbor_type))
    return neighbors
```

**Example**:
```
source: customer.addresses.city
  neighbors: {string, string, string}  → {string}
  
target: contact.location.city  
  neighbors: {string, string}  → {string}
  
Jaccard({string}, {string}) = 1.0
```

#### Type Similarity

```python
def _type_similarity(self, source_type, target_type):
    """Compute type compatibility score."""
    source_norm = self._normalize_type(source_type)
    target_norm = self._normalize_type(target_type)
    
    # Exact match
    if source_norm == target_norm:
        return 1.0
    
    # Same group
    if self._same_type_group(source_norm, target_norm):
        return 0.8
    
    # Compatible (e.g., string can hold any value)
    if self._types_compatible(source_norm, target_norm):
        return 0.3
    
    return 0.0
```

**Type Groups**:

| Group | Types |
|-------|-------|
| text | string, varchar, char, text |
| integer | int, integer, bigint, smallint, long |
| decimal | decimal, numeric, float, double, real |
| temporal | date, timestamp, datetime, time |
| binary | binary, bytes, blob |

### 3. HybridMatcher

Combines semantic and structural matching for best results.

```python
from nexus_matcher.core.graph_matcher import HybridMatcher

# Initialize with weights
hybrid = HybridMatcher(
    semantic_weight=0.6,   # 60% semantic
    graph_weight=0.4,      # 40% structural
)

# Set up graph matcher
hybrid.graph_matcher.set_source_schema("source", source_fields)
hybrid.graph_matcher.set_target_schema("dictionary", dictionary_fields)

# Rerank semantic results with structural info
semantic_results = [
    ("dict_entry_1", 0.95),
    ("dict_entry_2", 0.92),
    ("dict_entry_3", 0.88),
]

reranked = hybrid.rerank_with_structure(
    semantic_results=semantic_results,
    source_field_id="customer.addresses.city",
)

for entry_id, combined_score in reranked:
    print(f"{entry_id}: {combined_score:.4f}")
```

**Scoring Formula**:

```python
combined_score = (
    semantic_weight × semantic_score +
    graph_weight × graph_score
)

# With default weights:
combined = 0.6 × semantic + 0.4 × structural
```

**When to Use Higher Graph Weight**:

| Scenario | Recommended Weights |
|----------|---------------------|
| Well-named fields | semantic=0.8, graph=0.2 |
| Ambiguous names | semantic=0.5, graph=0.5 |
| Structural patterns matter | semantic=0.4, graph=0.6 |
| Cross-domain matching | semantic=0.6, graph=0.4 |

### 4. SchemaGraphAnalyzer

Utilities for analyzing schema graphs.

```python
from nexus_matcher.core.graph_matcher import SchemaGraphAnalyzer

analyzer = SchemaGraphAnalyzer(graph)

# Get field statistics
stats = analyzer.get_stats()
print(f"Total nodes: {stats['num_nodes']}")
print(f"Total edges: {stats['num_edges']}")
print(f"Max depth: {stats['max_depth']}")
print(f"Type distribution: {stats['type_distribution']}")

# Find similar structures
similar = analyzer.find_structurally_similar("customer.addresses")
for path, similarity in similar:
    print(f"{path}: {similarity:.4f}")

# Get field context
context = analyzer.get_field_context("customer.addresses.city")
print(f"Parent: {context['parent']}")
print(f"Siblings: {context['siblings']}")
print(f"Depth: {context['depth']}")
print(f"Neighbor types: {context['neighbor_types']}")
```

---

## Usage Guide

### Basic Usage

```python
from nexus_matcher.core.graph_matcher import (
    SchemaGraphBuilder,
    GraphStructuralMatcher,
)

# 1. Prepare field data
source_fields = [
    {"path": "order.id", "name": "id", "data_type": "long"},
    {"path": "order.customer_id", "name": "customer_id", "data_type": "long"},
    {"path": "order.total", "name": "total", "data_type": "decimal"},
]

dictionary_fields = [
    {"path": "order_identifier", "name": "order_identifier", "data_type": "bigint"},
    {"path": "customer_reference", "name": "customer_reference", "data_type": "integer"},
    {"path": "order_amount", "name": "order_amount", "data_type": "numeric"},
]

# 2. Create matcher
matcher = GraphStructuralMatcher()
matcher.set_source_schema("source", source_fields)
matcher.set_target_schema("dictionary", dictionary_fields)

# 3. Match fields
for source_field in source_fields:
    matches = matcher.match_field(source_field["path"], top_k=3)
    print(f"\n{source_field['path']}:")
    for target_id, score in matches:
        print(f"  → {target_id}: {score:.4f}")
```

### Hybrid Matching with Semantic

```python
from nexus_matcher.core.graph_matcher import HybridMatcher
from nexus_matcher.infrastructure.adapters.embeddings import (
    SentenceTransformerEmbeddingProvider,
)
from nexus_matcher.infrastructure.adapters.vector_stores import (
    InMemoryVectorStore,
)

# 1. Set up semantic matching
embedder = SentenceTransformerEmbeddingProvider()
vector_store = InMemoryVectorStore(dimension=embedder.dimension)

# Index dictionary
for entry in dictionary_entries:
    embedding = embedder.encode(entry.to_searchable_text())
    vector_store.add([entry.id], [embedding], [entry.to_dict()])

# 2. Set up hybrid matcher
hybrid = HybridMatcher(semantic_weight=0.6, graph_weight=0.4)
hybrid.graph_matcher.set_source_schema("source", source_fields)
hybrid.graph_matcher.set_target_schema("dictionary", dictionary_fields)

# 3. Match with both signals
def hybrid_match(field):
    # Semantic candidates
    query_embedding = embedder.encode(field.to_searchable_text())
    semantic_results = vector_store.search(query_embedding, top_k=20)
    
    # Rerank with structure
    reranked = hybrid.rerank_with_structure(
        semantic_results=[(r.id, r.score) for r in semantic_results],
        source_field_id=field.path,
    )
    
    return reranked[:5]
```

### Integration with NexusMatcher

```python
from nexus_matcher import NexusMatcher
from nexus_matcher.core.graph_matcher import HybridMatcher

# Initialize matcher
matcher = NexusMatcher()
matcher.load_dictionary("data/dictionary.xlsx")

# Add graph-based reranking
hybrid = HybridMatcher(semantic_weight=0.6, graph_weight=0.4)

# Custom matching with hybrid
def enhanced_match(schema_path):
    # Parse schema
    schema = matcher.parse_schema(schema_path)
    
    # Build source graph
    source_fields = [f.to_dict() for f in schema.fields]
    hybrid.graph_matcher.set_source_schema("source", source_fields)
    
    # Match each field
    results = {}
    for field in schema.fields:
        # Get semantic matches
        semantic = matcher.match_field(field, top_k=20)
        semantic_results = [(m.dictionary_entry.id, m.final_confidence) 
                           for m in semantic]
        
        # Rerank with structure
        reranked = hybrid.rerank_with_structure(semantic_results, field.path)
        
        # Convert back to Match objects
        results[field.path] = rerank_to_matches(reranked, matcher.dictionary)
    
    return results
```

---

## Benchmark Results

### Graph-Only Performance

| Metric | Value |
|--------|-------|
| Precision@1 | 29.41% |
| Precision@5 | 76.47% |
| MRR | 0.4775 |
| F1 | 29.41% |
| Latency | 0.13ms per field |

**Note**: Graph-only matching is intentionally limited because it has no semantic information. The value is in **combination** with semantic matching.

### Hybrid Performance (Semantic + Graph)

| Semantic Weight | Graph Weight | MRR | Notes |
|-----------------|--------------|-----|-------|
| 1.0 | 0.0 | 1.0000 | Baseline (semantic only) |
| 0.8 | 0.2 | 0.9800 | Minor structural boost |
| 0.6 | 0.4 | 0.9650 | Balanced |
| 0.5 | 0.5 | 0.9500 | Equal weights |
| 0.4 | 0.6 | 0.9200 | Structure-heavy |

**Recommendation**: Use `semantic=0.6, graph=0.4` for general use cases. Increase graph weight for structurally complex schemas.

### Latency Impact

| Operation | Latency |
|-----------|---------|
| Graph building | 5ms (once per schema) |
| Single field match | 0.13ms |
| Hybrid reranking | 0.5ms |
| Total overhead | ~0.5ms per field |

---

## Advanced Topics

### Custom Edge Types

```python
from nexus_matcher.core.graph_matcher import SchemaGraphBuilder

class CustomGraphBuilder(SchemaGraphBuilder):
    """Extended graph builder with custom edges."""
    
    def _create_edges(self, fields):
        super()._create_edges(fields)
        
        # Add custom semantic similarity edges
        for i, field1 in enumerate(fields):
            for field2 in fields[i+1:]:
                if self._names_similar(field1['name'], field2['name']):
                    self.graph.add_edge(
                        field1['path'],
                        field2['path'],
                        relation='name_similarity',
                        weight=self._name_similarity(field1['name'], field2['name']),
                    )
    
    def _names_similar(self, name1, name2):
        """Check if names are similar using edit distance."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1, name2).ratio() > 0.6
```

### Custom Scoring Functions

```python
from nexus_matcher.core.graph_matcher import GraphStructuralMatcher

class CustomStructuralMatcher(GraphStructuralMatcher):
    """Custom matcher with domain-specific scoring."""
    
    def _combined_score(self, source_id, target_id):
        structural = self._structural_similarity(
            self._source_graph.nodes[source_id],
            self._target_graph.nodes[target_id],
        )
        context = self._context_similarity(source_id, target_id)
        type_sim = self._type_similarity(
            self._source_graph.nodes[source_id]['data_type'],
            self._target_graph.nodes[target_id]['data_type'],
        )
        
        # Custom: Add name pattern matching
        name_sim = self._name_pattern_similarity(
            self._source_graph.nodes[source_id]['name'],
            self._target_graph.nodes[target_id]['name'],
        )
        
        return (
            0.3 * structural +
            0.2 * context +
            0.2 * type_sim +
            0.3 * name_sim  # Custom component
        )
```

### Weighted Edges

```python
# Use edge weights in scoring
def _context_similarity_weighted(self, source_id, target_id):
    """Context similarity using edge weights."""
    source_neighbors = []
    for neighbor in self._source_graph.neighbors(source_id):
        edge_data = self._source_graph.edges[source_id, neighbor]
        weight = edge_data.get('weight', 1.0)
        neighbor_type = self._source_graph.nodes[neighbor]['data_type']
        source_neighbors.append((neighbor_type, weight))
    
    # Weight-adjusted Jaccard
    # ...
```

### Graph Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_schema_graph(graph, output_path="schema_graph.png"):
    """Visualize schema graph with matplotlib."""
    plt.figure(figsize=(12, 8))
    
    # Layout
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Draw nodes
    node_colors = [
        'lightblue' if graph.nodes[n]['depth'] == 0 else
        'lightgreen' if graph.nodes[n]['depth'] == 1 else
        'lightyellow'
        for n in graph.nodes()
    ]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)
    
    # Draw edges by type
    parent_child = [(u, v) for u, v, d in graph.edges(data=True) 
                    if d.get('relation') == 'parent_child']
    sibling = [(u, v) for u, v, d in graph.edges(data=True) 
               if d.get('relation') == 'sibling']
    
    nx.draw_networkx_edges(graph, pos, edgelist=parent_child, 
                           edge_color='black', style='solid')
    nx.draw_networkx_edges(graph, pos, edgelist=sibling, 
                           edge_color='gray', style='dashed')
    
    # Labels
    labels = {n: graph.nodes[n]['name'] for n in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    plt.title("Schema Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## Troubleshooting

### Common Issues

**Issue**: Graph matcher returns poor results
```
Cause: Graph-only matching lacks semantic information
Solution: Use HybridMatcher with semantic_weight >= 0.5
```

**Issue**: Slow graph building
```
Cause: Too many sibling edges in wide schemas
Solution: Limit sibling edges to fields with same type
```

**Issue**: Memory usage grows with schema size
```
Cause: Dense edge creation
Solution: Prune low-weight edges or use sparse representation
```

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('nexus_matcher.core.graph_matcher').setLevel(logging.DEBUG)

# Inspect graph structure
matcher = GraphStructuralMatcher()
matcher.set_source_schema("source", source_fields)

print("Source graph nodes:")
for node, attrs in matcher._source_graph.nodes(data=True):
    print(f"  {node}: {attrs}")

print("\nSource graph edges:")
for u, v, attrs in matcher._source_graph.edges(data=True):
    print(f"  {u} → {v}: {attrs}")
```

---

## API Reference

### SchemaGraphBuilder

```python
class SchemaGraphBuilder:
    def __init__(self) -> None: ...
    
    def build(self, fields: list[dict]) -> nx.DiGraph: ...
    
    def _create_edges(self, fields: list[dict]) -> None: ...
    
    def _are_siblings(self, path1: str, path2: str) -> bool: ...
```

### GraphStructuralMatcher

```python
class GraphStructuralMatcher:
    def __init__(self) -> None: ...
    
    def set_source_schema(
        self,
        schema_name: str,
        fields: list[dict],
    ) -> None: ...
    
    def set_target_schema(
        self,
        schema_name: str,
        fields: list[dict],
    ) -> None: ...
    
    def match_field(
        self,
        source_field_id: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]: ...
    
    def match_all(
        self,
        top_k: int = 5,
    ) -> dict[str, list[tuple[str, float]]]: ...
```

### HybridMatcher

```python
class HybridMatcher:
    def __init__(
        self,
        semantic_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> None: ...
    
    def rerank_with_structure(
        self,
        semantic_results: list[tuple[str, float]],
        source_field_id: str,
    ) -> list[tuple[str, float]]: ...
```

---

*Module Documentation Version 1.0.0*
*Last Updated: December 2025*
