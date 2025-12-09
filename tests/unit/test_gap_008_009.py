"""
Unit tests for GAP-008 (Type Projections) and GAP-009 (Graph Matching).
"""

import pytest
import numpy as np


# =============================================================================
# GAP-009: Graph Matcher Tests
# =============================================================================

class TestGraphMatcher:
    """Tests for GraphStructuralMatcher."""
    
    @pytest.fixture
    def matcher(self):
        """Create a graph matcher instance."""
        pytest.importorskip("networkx")
        from nexus_matcher.core.graph_matcher import GraphStructuralMatcher
        return GraphStructuralMatcher()
    
    @pytest.fixture
    def source_fields(self):
        """Sample source fields."""
        return [
            {"path": "customer.id", "name": "id", "type": "integer"},
            {"path": "customer.name", "name": "name", "type": "string"},
            {"path": "customer.email", "name": "email", "type": "string"},
            {"path": "customer.address.city", "name": "city", "type": "string"},
        ]
    
    @pytest.fixture
    def target_fields(self):
        """Sample target fields."""
        return [
            {"path": "cust_id", "name": "cust_id", "type": "int"},
            {"path": "cust_name", "name": "cust_name", "type": "varchar"},
            {"path": "cust_email", "name": "cust_email", "type": "varchar"},
            {"path": "cust_city", "name": "cust_city", "type": "varchar"},
            {"path": "emp_id", "name": "emp_id", "type": "int"},
        ]
    
    def test_build_source_graph(self, matcher, source_fields):
        """Test building source graph."""
        matcher.set_source_schema("source", source_fields)
        
        stats = matcher.get_graph_stats()
        assert "source" in stats
        assert stats["source"]["nodes"] == 4
        assert stats["source"]["edges"] > 0
    
    def test_build_target_graph(self, matcher, target_fields):
        """Test building target graph."""
        matcher.set_target_schema("target", target_fields)
        
        stats = matcher.get_graph_stats()
        assert "target" in stats
        assert stats["target"]["nodes"] == 5
    
    def test_match_field(self, matcher, source_fields, target_fields):
        """Test matching a single field."""
        matcher.set_source_schema("source", source_fields)
        matcher.set_target_schema("target", target_fields)
        
        matches = matcher.match_field("customer.email", top_k=3)
        
        assert len(matches) == 3
        assert matches[0].source_field == "customer.email"
        assert all(0 <= m.combined_score <= 1 for m in matches)
    
    def test_match_all(self, matcher, source_fields, target_fields):
        """Test matching all fields."""
        matcher.set_source_schema("source", source_fields)
        matcher.set_target_schema("target", target_fields)
        
        all_matches = matcher.match_all(top_k_per_field=3)
        
        assert len(all_matches) == 4
        assert "customer.id" in all_matches
        assert "customer.email" in all_matches
    
    def test_type_similarity(self, matcher, source_fields, target_fields):
        """Test that type similarity is computed correctly."""
        matcher.set_source_schema("source", source_fields)
        matcher.set_target_schema("target", target_fields)
        
        # Integer field should match better with int targets
        matches = matcher.match_field("customer.id", top_k=5)
        
        # cust_id and emp_id are both int types
        int_matches = [m for m in matches if "id" in m.target_field]
        assert len(int_matches) >= 1
        assert all(m.type_score > 0 for m in int_matches)


class TestSchemaGraphBuilder:
    """Tests for SchemaGraphBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Create a graph builder instance."""
        pytest.importorskip("networkx")
        from nexus_matcher.core.graph_matcher import SchemaGraphBuilder
        return SchemaGraphBuilder()
    
    def test_type_groups(self, builder):
        """Test type group mapping."""
        assert builder._get_type_group("string") == "string"
        assert builder._get_type_group("varchar") == "string"
        assert builder._get_type_group("integer") == "integer"
        assert builder._get_type_group("int") == "integer"
        assert builder._get_type_group("datetime") == "date"
        assert builder._get_type_group("unknown_xyz") == "other"
    
    def test_build_graph(self, builder):
        """Test building a complete graph."""
        fields = [
            {"path": "root", "name": "root", "type": "object"},
            {"path": "root.child1", "name": "child1", "type": "string"},
            {"path": "root.child2", "name": "child2", "type": "string"},
        ]
        
        schema = builder.build_graph("test", fields)
        
        assert schema.name == "test"
        assert len(schema.fields) == 3
        assert schema.graph.number_of_nodes() == 3
        # Should have parent-child and sibling edges
        assert schema.graph.number_of_edges() > 0


# =============================================================================
# GAP-008: Type Projections Tests
# =============================================================================

class TestTypeVocabulary:
    """Tests for TypeVocabulary."""
    
    @pytest.fixture
    def vocab(self):
        """Create a vocabulary instance."""
        torch = pytest.importorskip("torch")
        from nexus_matcher.core.type_projections import TypeVocabulary
        return TypeVocabulary()
    
    def test_standard_types(self, vocab):
        """Test that standard types are mapped correctly."""
        # String types should map to same ID
        string_id = vocab.get_type_id("string")
        varchar_id = vocab.get_type_id("varchar")
        assert string_id == varchar_id
        
        # Integer types should map to same ID
        int_id = vocab.get_type_id("integer")
        int_id2 = vocab.get_type_id("int")
        assert int_id == int_id2
    
    def test_unknown_type(self, vocab):
        """Test handling of unknown types."""
        id1 = vocab.get_type_id("custom_type_xyz")
        id2 = vocab.get_type_id("custom_type_xyz")
        
        # Same unknown type should get same ID
        assert id1 == id2
        
        # Different unknown type should get different ID
        id3 = vocab.get_type_id("another_custom_type")
        assert id3 != id1
    
    def test_partial_match(self, vocab):
        """Test partial type matching (e.g., varchar(255))."""
        # varchar(255) should match varchar
        varchar_id = vocab.get_type_id("varchar")
        varchar_255_id = vocab.get_type_id("varchar(255)")
        assert varchar_id == varchar_255_id


class TestTrainingDataGenerator:
    """Tests for TrainingDataGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        torch = pytest.importorskip("torch")
        from nexus_matcher.core.type_projections import TrainingDataGenerator
        return TrainingDataGenerator(seed=42)
    
    def test_generate_pairs(self, generator):
        """Test generating training pairs."""
        pairs = generator.generate_pairs(num_positive=50, num_negative=50)
        
        assert len(pairs) == 100
        
        positive = [p for p in pairs if p.is_match]
        negative = [p for p in pairs if not p.is_match]
        
        assert len(positive) == 50
        assert len(negative) == 50
    
    def test_pair_structure(self, generator):
        """Test that pairs have correct structure."""
        pairs = generator.generate_pairs(num_positive=10, num_negative=10)
        
        for pair in pairs:
            assert pair.source_text
            assert pair.source_type
            assert pair.target_text
            assert pair.target_type
            assert isinstance(pair.is_match, bool)


class TestTypeProjectionModel:
    """Tests for TypeProjectionLayer and ContrastiveTypeModel."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance."""
        torch = pytest.importorskip("torch")
        from nexus_matcher.core.type_projections import (
            ContrastiveTypeModel, TypeProjectionConfig
        )
        config = TypeProjectionConfig(
            base_embedding_dim=384,
            type_embedding_dim=64,
            projection_dim=384,
        )
        return ContrastiveTypeModel(config)
    
    def test_forward(self, model):
        """Test forward pass."""
        import torch
        
        batch_size = 4
        source_emb = torch.randn(batch_size, 384)
        source_types = torch.randint(0, 25, (batch_size,))
        target_emb = torch.randn(batch_size, 384)
        target_types = torch.randint(0, 25, (batch_size,))
        
        source_proj, target_proj = model(
            source_emb, source_types, target_emb, target_types
        )
        
        assert source_proj.shape == (batch_size, 384)
        assert target_proj.shape == (batch_size, 384)
    
    def test_similarity(self, model):
        """Test similarity computation."""
        import torch
        
        batch_size = 4
        source_proj = torch.randn(batch_size, 384)
        target_proj = torch.randn(batch_size, 384)
        
        sim = model.compute_similarity(source_proj, target_proj)
        
        assert sim.shape == (batch_size,)
    
    def test_contrastive_loss(self, model):
        """Test contrastive loss computation."""
        import torch
        
        batch_size = 4
        source_proj = torch.randn(batch_size, 384)
        target_proj = torch.randn(batch_size, 384)
        labels = torch.tensor([1., 0., 1., 0.])
        
        loss = model.contrastive_loss(source_proj, target_proj, labels)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0


class TestTypeProjectionManager:
    """Tests for TypeProjectionManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a manager instance."""
        torch = pytest.importorskip("torch")
        from nexus_matcher.core.type_projections import (
            TypeProjectionManager, TypeProjectionConfig
        )
        config = TypeProjectionConfig(
            base_embedding_dim=384,
            type_embedding_dim=64,
            projection_dim=384,
        )
        return TypeProjectionManager(config)
    
    def test_project_single(self, manager):
        """Test projecting a single embedding."""
        base_emb = np.random.randn(384).astype(np.float32)
        
        result = manager.project(base_emb, "string")
        
        assert result.shape == (384,)
    
    def test_project_batch(self, manager):
        """Test projecting a batch of embeddings."""
        base_embs = np.random.randn(8, 384).astype(np.float32)
        types = ["string", "integer", "decimal", "datetime",
                 "string", "boolean", "array", "object"]
        
        result = manager.project_batch(base_embs, types)
        
        assert result.shape == (8, 384)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHybridMatcher:
    """Tests for HybridMatcher (semantic + graph)."""
    
    @pytest.fixture
    def hybrid(self):
        """Create a hybrid matcher instance."""
        pytest.importorskip("networkx")
        from nexus_matcher.core.graph_matcher import HybridMatcher
        return HybridMatcher(semantic_weight=0.6, graph_weight=0.4)
    
    @pytest.fixture
    def setup_schemas(self, hybrid):
        """Set up source and target schemas."""
        source_fields = [
            {"path": "customer.email", "name": "email", "type": "string"},
        ]
        target_fields = [
            {"path": "cust_email", "name": "cust_email", "type": "varchar"},
            {"path": "emp_email", "name": "emp_email", "type": "varchar"},
        ]
        
        hybrid.graph_matcher.set_source_schema("source", source_fields)
        hybrid.graph_matcher.set_target_schema("target", target_fields)
        
        return hybrid
    
    def test_rerank_with_structure(self, setup_schemas):
        """Test re-ranking semantic results with structural info."""
        hybrid = setup_schemas
        
        # Simulated semantic results
        semantic_results = [
            ("cust_email", 0.9),
            ("emp_email", 0.85),
        ]
        
        reranked = hybrid.rerank_with_structure(
            semantic_results,
            "customer.email",
        )
        
        assert len(reranked) == 2
        # Should have field_id, combined_score, details
        assert all(len(r) == 3 for r in reranked)
        # Combined scores should be weighted
        assert all(r[1] > 0 for r in reranked)
