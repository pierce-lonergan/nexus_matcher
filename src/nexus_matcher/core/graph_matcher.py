"""
GAP-009: Graph-Based Structural Schema Matching
═══════════════════════════════════════════════════════════════════

Research Reference: README_RESEARCH_2.md, Lines 3-4
Target: F1 78-85%, 600x faster than traditional matchers

This module implements graph-based schema matching that captures
structural relationships between fields that pure text methods miss.

Key Features:
- Schema-to-graph conversion (fields as nodes, relationships as edges)
- Structural similarity scoring
- Graph-based candidate ranking
- Integration with existing semantic matching pipeline

Dependencies:
    pip install networkx
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

import numpy as np


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SchemaField:
    """Represents a field in a schema."""
    
    name: str
    data_type: str
    path: str  # Full path like "user.address.street"
    depth: int
    parent_path: Optional[str] = None
    description: str = ""
    nullable: bool = True
    
    @property
    def field_id(self) -> str:
        """Unique identifier for this field."""
        return self.path


@dataclass
class SchemaGraph:
    """Graph representation of a schema."""
    
    name: str
    graph: Any = field(default=None)  # nx.DiGraph
    fields: dict[str, SchemaField] = field(default_factory=dict)
    
    def __post_init__(self):
        if NETWORKX_AVAILABLE and self.graph is None:
            self.graph = nx.DiGraph()


@dataclass 
class GraphMatchResult:
    """Result of graph-based matching."""
    
    source_field: str
    target_field: str
    structural_score: float
    context_score: float
    type_score: float
    combined_score: float
    match_reasons: list[str] = field(default_factory=list)


# =============================================================================
# SCHEMA TO GRAPH CONVERTER
# =============================================================================

class SchemaGraphBuilder:
    """Converts schema definitions to graph representations."""
    
    # Type similarity groups
    TYPE_GROUPS = {
        "string": {"string", "text", "varchar", "char", "str"},
        "integer": {"integer", "int", "bigint", "smallint", "tinyint", "long"},
        "decimal": {"decimal", "float", "double", "real", "number", "numeric"},
        "boolean": {"boolean", "bool", "bit"},
        "date": {"date", "datetime", "timestamp", "time"},
        "binary": {"binary", "blob", "bytes", "bytea"},
        "array": {"array", "list", "set"},
        "object": {"object", "struct", "record", "map"},
    }
    
    def __init__(self):
        self._type_to_group = {}
        for group, types in self.TYPE_GROUPS.items():
            for t in types:
                self._type_to_group[t.lower()] = group
    
    def build_graph(self, schema_name: str, fields: list[dict]) -> SchemaGraph:
        """
        Build a graph from a list of field definitions.
        
        Args:
            schema_name: Name of the schema
            fields: List of field dicts with keys: name, type, path, description, etc.
            
        Returns:
            SchemaGraph with nodes (fields) and edges (relationships)
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for graph-based matching. Install with: pip install networkx")
        
        schema = SchemaGraph(name=schema_name)
        
        # Parse fields and add nodes
        for field_def in fields:
            field = self._parse_field(field_def)
            schema.fields[field.field_id] = field
            
            # Add node with attributes
            schema.graph.add_node(
                field.field_id,
                name=field.name,
                data_type=field.data_type,
                type_group=self._get_type_group(field.data_type),
                depth=field.depth,
                description=field.description,
            )
        
        # Add edges for structural relationships
        self._add_parent_child_edges(schema)
        self._add_sibling_edges(schema)
        self._add_type_similarity_edges(schema)
        
        return schema
    
    def _parse_field(self, field_def: dict) -> SchemaField:
        """Parse a field definition dictionary."""
        path = field_def.get("path", field_def.get("name", ""))
        parts = path.split(".")
        
        return SchemaField(
            name=field_def.get("name", parts[-1] if parts else ""),
            data_type=field_def.get("type", field_def.get("data_type", "string")),
            path=path,
            depth=len(parts) - 1,
            parent_path=".".join(parts[:-1]) if len(parts) > 1 else None,
            description=field_def.get("description", ""),
            nullable=field_def.get("nullable", True),
        )
    
    def _get_type_group(self, data_type: str) -> str:
        """Get the type group for a data type."""
        return self._type_to_group.get(data_type.lower(), "other")
    
    def _add_parent_child_edges(self, schema: SchemaGraph) -> None:
        """Add edges between parent and child fields."""
        for field_id, field in schema.fields.items():
            if field.parent_path and field.parent_path in schema.fields:
                # Parent -> Child edge
                schema.graph.add_edge(
                    field.parent_path,
                    field_id,
                    relation="parent_child",
                    weight=1.0,
                )
    
    def _add_sibling_edges(self, schema: SchemaGraph) -> None:
        """Add edges between sibling fields (same parent)."""
        # Group fields by parent
        siblings = defaultdict(list)
        for field_id, field in schema.fields.items():
            parent = field.parent_path or "__root__"
            siblings[parent].append(field_id)
        
        # Add sibling edges
        for parent, sibling_ids in siblings.items():
            for i, id1 in enumerate(sibling_ids):
                for id2 in sibling_ids[i+1:]:
                    schema.graph.add_edge(
                        id1, id2,
                        relation="sibling",
                        weight=0.5,
                    )
                    schema.graph.add_edge(
                        id2, id1,
                        relation="sibling",
                        weight=0.5,
                    )
    
    def _add_type_similarity_edges(self, schema: SchemaGraph) -> None:
        """Add edges between fields with same type group."""
        # Group by type
        by_type = defaultdict(list)
        for field_id, field in schema.fields.items():
            type_group = self._get_type_group(field.data_type)
            by_type[type_group].append(field_id)
        
        # Add type similarity edges (weaker than structural)
        for type_group, field_ids in by_type.items():
            if len(field_ids) > 1 and type_group != "other":
                for i, id1 in enumerate(field_ids):
                    for id2 in field_ids[i+1:]:
                        # Only add if not already connected
                        if not schema.graph.has_edge(id1, id2):
                            schema.graph.add_edge(
                                id1, id2,
                                relation="type_similar",
                                weight=0.3,
                            )


# =============================================================================
# GRAPH MATCHER
# =============================================================================

class GraphStructuralMatcher:
    """
    Matches schema fields using graph-based structural similarity.
    
    This complements semantic text matching by considering:
    - Structural position (depth, parent-child relationships)
    - Sibling context (fields at same level)
    - Type compatibility
    - Neighborhood similarity
    """
    
    def __init__(
        self,
        structural_weight: float = 0.4,
        context_weight: float = 0.3,
        type_weight: float = 0.3,
    ):
        """
        Initialize the graph matcher.
        
        Args:
            structural_weight: Weight for structural similarity (position, depth)
            context_weight: Weight for context similarity (neighbors)
            type_weight: Weight for type compatibility
        """
        self.structural_weight = structural_weight
        self.context_weight = context_weight
        self.type_weight = type_weight
        self.builder = SchemaGraphBuilder()
        
        self._source_graph: Optional[SchemaGraph] = None
        self._target_graph: Optional[SchemaGraph] = None
    
    def set_source_schema(self, name: str, fields: list[dict]) -> None:
        """Set the source schema for matching."""
        self._source_graph = self.builder.build_graph(name, fields)
    
    def set_target_schema(self, name: str, fields: list[dict]) -> None:
        """Set the target schema (data dictionary) for matching."""
        self._target_graph = self.builder.build_graph(name, fields)
    
    def match_field(
        self,
        source_field_id: str,
        candidate_ids: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[GraphMatchResult]:
        """
        Find best matches for a source field using graph similarity.
        
        Args:
            source_field_id: ID of the source field to match
            candidate_ids: Optional list of target field IDs to consider
            top_k: Number of top matches to return
            
        Returns:
            List of GraphMatchResult sorted by combined score
        """
        if not self._source_graph or not self._target_graph:
            raise ValueError("Both source and target schemas must be set")
        
        if source_field_id not in self._source_graph.fields:
            raise ValueError(f"Source field not found: {source_field_id}")
        
        source_field = self._source_graph.fields[source_field_id]
        
        # Get candidates
        if candidate_ids is None:
            candidate_ids = list(self._target_graph.fields.keys())
        
        results = []
        for target_id in candidate_ids:
            if target_id not in self._target_graph.fields:
                continue
            
            target_field = self._target_graph.fields[target_id]
            result = self._compute_match_score(source_field, target_field)
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]
    
    def match_all(
        self,
        top_k_per_field: int = 5,
    ) -> dict[str, list[GraphMatchResult]]:
        """
        Match all source fields to target fields.
        
        Returns:
            Dict mapping source field IDs to their top matches
        """
        if not self._source_graph or not self._target_graph:
            raise ValueError("Both source and target schemas must be set")
        
        all_matches = {}
        for source_id in self._source_graph.fields:
            matches = self.match_field(source_id, top_k=top_k_per_field)
            all_matches[source_id] = matches
        
        return all_matches
    
    def _compute_match_score(
        self,
        source: SchemaField,
        target: SchemaField,
    ) -> GraphMatchResult:
        """Compute comprehensive match score between two fields."""
        
        reasons = []
        
        # 1. Structural similarity (depth, position)
        structural_score = self._structural_similarity(source, target)
        if structural_score > 0.5:
            reasons.append(f"depth_match:{source.depth}→{target.depth}")
        
        # 2. Context similarity (neighbors)
        context_score = self._context_similarity(source, target)
        if context_score > 0.5:
            reasons.append("similar_context")
        
        # 3. Type compatibility
        type_score = self._type_similarity(source, target)
        if type_score > 0.5:
            reasons.append(f"type_match:{source.data_type}→{target.data_type}")
        
        # Combined score
        combined = (
            self.structural_weight * structural_score +
            self.context_weight * context_score +
            self.type_weight * type_score
        )
        
        return GraphMatchResult(
            source_field=source.path,
            target_field=target.path,
            structural_score=structural_score,
            context_score=context_score,
            type_score=type_score,
            combined_score=combined,
            match_reasons=reasons,
        )
    
    def _structural_similarity(
        self,
        source: SchemaField,
        target: SchemaField,
    ) -> float:
        """
        Compute structural similarity based on graph position.
        
        Considers:
        - Depth similarity
        - Path structure similarity
        """
        # Depth similarity (closer depths = higher score)
        depth_diff = abs(source.depth - target.depth)
        depth_sim = 1.0 / (1.0 + depth_diff)
        
        # Path component similarity
        source_parts = source.path.lower().split(".")
        target_parts = target.path.lower().split(".")
        
        # Compare path structure (not content, just shape)
        len_diff = abs(len(source_parts) - len(target_parts))
        len_sim = 1.0 / (1.0 + len_diff)
        
        return 0.6 * depth_sim + 0.4 * len_sim
    
    def _context_similarity(
        self,
        source: SchemaField,
        target: SchemaField,
    ) -> float:
        """
        Compute context similarity based on graph neighborhoods.
        
        Considers:
        - Sibling types
        - Parent field characteristics
        """
        if not NETWORKX_AVAILABLE:
            return 0.5  # Default if no graph
        
        # Get neighbor type distributions
        source_neighbors = self._get_neighbor_types(
            self._source_graph, source.path
        )
        target_neighbors = self._get_neighbor_types(
            self._target_graph, target.path
        )
        
        if not source_neighbors or not target_neighbors:
            return 0.5  # No context available
        
        # Jaccard similarity of neighbor types
        source_set = set(source_neighbors)
        target_set = set(target_neighbors)
        
        if not source_set and not target_set:
            return 0.5
        
        intersection = len(source_set & target_set)
        union = len(source_set | target_set)
        
        return intersection / union if union > 0 else 0.5
    
    def _get_neighbor_types(
        self,
        schema: SchemaGraph,
        field_id: str,
    ) -> list[str]:
        """Get the type groups of neighboring fields."""
        if field_id not in schema.graph:
            return []
        
        neighbor_types = []
        for neighbor in schema.graph.neighbors(field_id):
            node_data = schema.graph.nodes[neighbor]
            neighbor_types.append(node_data.get("type_group", "other"))
        
        return neighbor_types
    
    def _type_similarity(
        self,
        source: SchemaField,
        target: SchemaField,
    ) -> float:
        """
        Compute type compatibility score.
        
        Returns:
        - 1.0 for exact type match
        - 0.8 for same type group
        - 0.3 for compatible types
        - 0.0 for incompatible types
        """
        source_type = source.data_type.lower()
        target_type = target.data_type.lower()
        
        # Exact match
        if source_type == target_type:
            return 1.0
        
        # Same type group
        source_group = self.builder._get_type_group(source_type)
        target_group = self.builder._get_type_group(target_type)
        
        if source_group == target_group and source_group != "other":
            return 0.8
        
        # Compatible groups
        compatible_pairs = {
            ("integer", "decimal"),
            ("string", "date"),  # String can hold dates
        }
        
        if (source_group, target_group) in compatible_pairs or \
           (target_group, source_group) in compatible_pairs:
            return 0.3
        
        return 0.0
    
    def get_graph_stats(self) -> dict:
        """Get statistics about the loaded graphs."""
        stats = {}
        
        if self._source_graph:
            stats["source"] = {
                "name": self._source_graph.name,
                "nodes": self._source_graph.graph.number_of_nodes(),
                "edges": self._source_graph.graph.number_of_edges(),
                "max_depth": max(
                    (f.depth for f in self._source_graph.fields.values()),
                    default=0
                ),
            }
        
        if self._target_graph:
            stats["target"] = {
                "name": self._target_graph.name,
                "nodes": self._target_graph.graph.number_of_nodes(),
                "edges": self._target_graph.graph.number_of_edges(),
                "max_depth": max(
                    (f.depth for f in self._target_graph.fields.values()),
                    default=0
                ),
            }
        
        return stats


# =============================================================================
# HYBRID MATCHER (Combines Semantic + Graph)
# =============================================================================

class HybridMatcher:
    """
    Combines semantic text matching with graph structural matching.
    
    This is the recommended approach for production:
    - Semantic matching captures meaning similarity
    - Graph matching captures structural relationships
    - Combined scoring provides better precision
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        """
        Initialize hybrid matcher.
        
        Args:
            semantic_weight: Weight for semantic similarity scores
            graph_weight: Weight for graph structural scores
        """
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight
        self.graph_matcher = GraphStructuralMatcher()
    
    def rerank_with_structure(
        self,
        semantic_results: list[tuple[str, float]],  # (field_id, semantic_score)
        source_field_id: str,
    ) -> list[tuple[str, float, dict]]:
        """
        Re-rank semantic results using structural information.
        
        Args:
            semantic_results: List of (field_id, semantic_score) tuples
            source_field_id: The source field being matched
            
        Returns:
            List of (field_id, combined_score, details) tuples
        """
        candidate_ids = [r[0] for r in semantic_results]
        
        # Get graph scores
        graph_results = self.graph_matcher.match_field(
            source_field_id,
            candidate_ids=candidate_ids,
            top_k=len(candidate_ids),
        )
        
        # Build graph score lookup
        graph_scores = {
            r.target_field: r for r in graph_results
        }
        
        # Combine scores
        combined = []
        for field_id, semantic_score in semantic_results:
            graph_result = graph_scores.get(field_id)
            
            if graph_result:
                graph_score = graph_result.combined_score
                combined_score = (
                    self.semantic_weight * semantic_score +
                    self.graph_weight * graph_score
                )
                details = {
                    "semantic_score": semantic_score,
                    "graph_score": graph_score,
                    "structural": graph_result.structural_score,
                    "context": graph_result.context_score,
                    "type": graph_result.type_score,
                    "reasons": graph_result.match_reasons,
                }
            else:
                combined_score = semantic_score * self.semantic_weight
                details = {
                    "semantic_score": semantic_score,
                    "graph_score": 0.0,
                    "reasons": ["no_graph_data"],
                }
            
            combined.append((field_id, combined_score, details))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_networkx_available() -> bool:
    """Check if networkx is available."""
    return NETWORKX_AVAILABLE


def schema_to_fields(schema_dict: dict) -> list[dict]:
    """
    Convert a nested schema dictionary to a flat list of fields.
    
    Args:
        schema_dict: Nested schema like {"user": {"name": "string", "email": "string"}}
        
    Returns:
        List of field dicts with path, name, type
    """
    fields = []
    
    def _flatten(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, dict) and "type" not in value:
                    # Nested object
                    _flatten(value, new_path)
                elif isinstance(value, dict):
                    # Field with metadata
                    fields.append({
                        "path": new_path,
                        "name": key,
                        "type": value.get("type", "object"),
                        "description": value.get("description", ""),
                    })
                else:
                    # Simple type
                    fields.append({
                        "path": new_path,
                        "name": key,
                        "type": value if isinstance(value, str) else "object",
                    })
    
    _flatten(schema_dict)
    return fields


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test the graph matcher
    print("GAP-009: Graph-Based Structural Matcher")
    print("=" * 50)
    
    if not NETWORKX_AVAILABLE:
        print("ERROR: networkx not installed")
        print("Install with: pip install networkx")
        exit(1)
    
    # Example source schema
    source_fields = [
        {"path": "customer", "name": "customer", "type": "object"},
        {"path": "customer.id", "name": "id", "type": "integer"},
        {"path": "customer.name", "name": "name", "type": "string"},
        {"path": "customer.email", "name": "email", "type": "string"},
        {"path": "customer.address", "name": "address", "type": "object"},
        {"path": "customer.address.street", "name": "street", "type": "string"},
        {"path": "customer.address.city", "name": "city", "type": "string"},
        {"path": "order", "name": "order", "type": "object"},
        {"path": "order.id", "name": "id", "type": "integer"},
        {"path": "order.amount", "name": "amount", "type": "decimal"},
        {"path": "order.date", "name": "date", "type": "datetime"},
    ]
    
    # Example target schema (data dictionary)
    target_fields = [
        {"path": "cust_id", "name": "cust_id", "type": "int", "description": "Customer identifier"},
        {"path": "cust_name", "name": "cust_name", "type": "varchar", "description": "Customer full name"},
        {"path": "cust_email", "name": "cust_email", "type": "varchar", "description": "Customer email address"},
        {"path": "cust_addr_street", "name": "cust_addr_street", "type": "varchar", "description": "Street address"},
        {"path": "cust_addr_city", "name": "cust_addr_city", "type": "varchar", "description": "City"},
        {"path": "ord_id", "name": "ord_id", "type": "int", "description": "Order identifier"},
        {"path": "ord_total", "name": "ord_total", "type": "decimal", "description": "Order total amount"},
        {"path": "ord_date", "name": "ord_date", "type": "timestamp", "description": "Order date"},
        {"path": "prod_id", "name": "prod_id", "type": "int", "description": "Product identifier"},
        {"path": "prod_name", "name": "prod_name", "type": "varchar", "description": "Product name"},
    ]
    
    # Create matcher
    matcher = GraphStructuralMatcher()
    matcher.set_source_schema("source", source_fields)
    matcher.set_target_schema("data_dictionary", target_fields)
    
    # Print graph stats
    stats = matcher.get_graph_stats()
    print(f"\nSource graph: {stats['source']['nodes']} nodes, {stats['source']['edges']} edges")
    print(f"Target graph: {stats['target']['nodes']} nodes, {stats['target']['edges']} edges")
    
    # Match a field
    print("\nMatching 'customer.email' to target schema:")
    print("-" * 50)
    
    matches = matcher.match_field("customer.email", top_k=5)
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match.target_field}")
        print(f"   Combined: {match.combined_score:.4f}")
        print(f"   Structural: {match.structural_score:.4f}, Context: {match.context_score:.4f}, Type: {match.type_score:.4f}")
        if match.match_reasons:
            print(f"   Reasons: {', '.join(match.match_reasons)}")
    
    print("\n✓ Graph-based matching working!")
