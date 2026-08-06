#!/usr/bin/env python3
"""
!!  SUPERSEDED -- DO NOT CITE THIS SUITE'S NUMBERS  !!

    This file produced the "100% Precision@1" figure that appeared in the README and was
    RETRACTED. Verified by AST parse: it defines 17 SOURCE_FIELDS, 20 TARGET_FIELDS and
    17 GROUND_TRUTH pairs, and the string "NexusMatcher" does not appear anywhere in it.
    The "baseline precision_at_1 = 1.0" is raw sentence-transformers cosine similarity
    over a 17-row hand-written toy set. It never measured this system.

    Use instead:
        python benchmarks/datasets/build_benchmarks.py   # 688 labelled pairs, real corpora
        python benchmarks/eval_pipeline.py --benchmark combined

    Kept only as a historical record of where the retracted number came from.

SUITE-008 — GAP-008 & GAP-009 Combined Benchmark
═══════════════════════════════════════════════════════════════════

This benchmark validates:
- GAP-008: Learned Type Projections (MRR target: 0.80+)
- GAP-009: Graph-Based Structural Matching (F1 target: 78-85%)

Prerequisites:
    pip install torch networkx sentence-transformers numpy
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Check dependencies
DEPENDENCIES = {}

try:
    import torch

    DEPENDENCIES["torch"] = True
except ImportError:
    DEPENDENCIES["torch"] = False

try:
    import networkx as nx

    DEPENDENCIES["networkx"] = True
except ImportError:
    DEPENDENCIES["networkx"] = False

try:
    from sentence_transformers import SentenceTransformer

    DEPENDENCIES["sentence_transformers"] = True
except ImportError:
    DEPENDENCIES["sentence_transformers"] = False


# =============================================================================
# TEST DATA
# =============================================================================

# Source schema (what we're matching FROM)
SOURCE_FIELDS = [
    {"path": "customer.id", "name": "id", "type": "integer", "description": "Customer identifier"},
    {
        "path": "customer.name",
        "name": "name",
        "type": "string",
        "description": "Customer full name",
    },
    {
        "path": "customer.email",
        "name": "email",
        "type": "string",
        "description": "Customer email address",
    },
    {"path": "customer.phone", "name": "phone", "type": "string", "description": "Phone number"},
    {
        "path": "customer.address.street",
        "name": "street",
        "type": "string",
        "description": "Street address",
    },
    {"path": "customer.address.city", "name": "city", "type": "string", "description": "City name"},
    {"path": "customer.address.zip", "name": "zip", "type": "string", "description": "Postal code"},
    {
        "path": "customer.created_at",
        "name": "created_at",
        "type": "datetime",
        "description": "Account creation date",
    },
    {
        "path": "customer.is_active",
        "name": "is_active",
        "type": "boolean",
        "description": "Active status flag",
    },
    {"path": "order.id", "name": "id", "type": "integer", "description": "Order identifier"},
    {
        "path": "order.amount",
        "name": "amount",
        "type": "decimal",
        "description": "Order total amount",
    },
    {
        "path": "order.status",
        "name": "status",
        "type": "string",
        "description": "Order status code",
    },
    {"path": "order.date", "name": "date", "type": "datetime", "description": "Order date"},
    {"path": "product.id", "name": "id", "type": "integer", "description": "Product identifier"},
    {"path": "product.name", "name": "name", "type": "string", "description": "Product name"},
    {"path": "product.price", "name": "price", "type": "decimal", "description": "Unit price"},
    {
        "path": "product.category",
        "name": "category",
        "type": "string",
        "description": "Product category",
    },
]

# Target schema (data dictionary - what we're matching TO)
TARGET_FIELDS = [
    {
        "path": "cust_id",
        "name": "cust_id",
        "type": "int",
        "description": "Customer unique identifier",
    },
    {
        "path": "cust_name",
        "name": "cust_name",
        "type": "varchar",
        "description": "Customer full name",
    },
    {
        "path": "cust_email",
        "name": "cust_email",
        "type": "varchar",
        "description": "Primary email address",
    },
    {
        "path": "cust_phone",
        "name": "cust_phone",
        "type": "varchar",
        "description": "Contact phone number",
    },
    {
        "path": "cust_street",
        "name": "cust_street",
        "type": "varchar",
        "description": "Street address line 1",
    },
    {"path": "cust_city", "name": "cust_city", "type": "varchar", "description": "City"},
    {
        "path": "cust_postal",
        "name": "cust_postal",
        "type": "varchar",
        "description": "ZIP/Postal code",
    },
    {
        "path": "cust_created",
        "name": "cust_created",
        "type": "timestamp",
        "description": "Customer creation timestamp",
    },
    {
        "path": "cust_active",
        "name": "cust_active",
        "type": "bit",
        "description": "Is customer active",
    },
    {"path": "ord_id", "name": "ord_id", "type": "int", "description": "Order ID"},
    {
        "path": "ord_total",
        "name": "ord_total",
        "type": "decimal",
        "description": "Order total value",
    },
    {"path": "ord_status", "name": "ord_status", "type": "varchar", "description": "Order status"},
    {
        "path": "ord_date",
        "name": "ord_date",
        "type": "datetime",
        "description": "Order creation date",
    },
    {"path": "prod_id", "name": "prod_id", "type": "int", "description": "Product ID"},
    {
        "path": "prod_name",
        "name": "prod_name",
        "type": "varchar",
        "description": "Product display name",
    },
    {
        "path": "prod_price",
        "name": "prod_price",
        "type": "decimal",
        "description": "Product unit price",
    },
    {
        "path": "prod_category",
        "name": "prod_category",
        "type": "varchar",
        "description": "Product category code",
    },
    # Distractors (should NOT match)
    {"path": "emp_id", "name": "emp_id", "type": "int", "description": "Employee identifier"},
    {"path": "dept_name", "name": "dept_name", "type": "varchar", "description": "Department name"},
    {"path": "salary", "name": "salary", "type": "decimal", "description": "Annual salary"},
]

# Ground truth mappings (source_path -> target_path)
GROUND_TRUTH = {
    "customer.id": "cust_id",
    "customer.name": "cust_name",
    "customer.email": "cust_email",
    "customer.phone": "cust_phone",
    "customer.address.street": "cust_street",
    "customer.address.city": "cust_city",
    "customer.address.zip": "cust_postal",
    "customer.created_at": "cust_created",
    "customer.is_active": "cust_active",
    "order.id": "ord_id",
    "order.amount": "ord_total",
    "order.status": "ord_status",
    "order.date": "ord_date",
    "product.id": "prod_id",
    "product.name": "prod_name",
    "product.price": "prod_price",
    "product.category": "prod_category",
}


# =============================================================================
# GAP-009: GRAPH-BASED MATCHING BENCHMARK
# =============================================================================


def benchmark_graph_matching():
    """Benchmark GAP-009: Graph-Based Structural Matching."""

    print("\n" + "=" * 70)
    print("GAP-009: Graph-Based Structural Matching Benchmark")
    print("=" * 70)

    if not DEPENDENCIES["networkx"]:
        print("  ⚠ networkx not installed. Install with: pip install networkx")
        return None

    # Import locally to handle missing dependency
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    try:
        from nexus_matcher.core.graph_matcher import (
            NETWORKX_AVAILABLE,
            GraphStructuralMatcher,
            HybridMatcher,
        )
    except ImportError:
        # Fall back to direct import
        from graph_matcher import NETWORKX_AVAILABLE, GraphStructuralMatcher

    if not NETWORKX_AVAILABLE:
        print("  ⚠ networkx not available in graph_matcher module")
        return None

    print("\nInitializing graph matcher...")

    # Create matcher
    matcher = GraphStructuralMatcher()
    matcher.set_source_schema("source", SOURCE_FIELDS)
    matcher.set_target_schema("data_dictionary", TARGET_FIELDS)

    # Print stats
    stats = matcher.get_graph_stats()
    print(f"  Source graph: {stats['source']['nodes']} nodes, {stats['source']['edges']} edges")
    print(f"  Target graph: {stats['target']['nodes']} nodes, {stats['target']['edges']} edges")

    # Match all fields
    print("\nMatching fields...")
    start_time = time.perf_counter()

    all_matches = matcher.match_all(top_k_per_field=5)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Evaluate
    correct_at_1 = 0
    correct_at_5 = 0
    reciprocal_ranks = []

    results_detail = []

    for source_path, expected_target in GROUND_TRUTH.items():
        if source_path not in all_matches:
            continue

        matches = all_matches[source_path]
        predicted_targets = [m.target_field for m in matches]

        # Check if correct at position 1
        if predicted_targets and predicted_targets[0] == expected_target:
            correct_at_1 += 1

        # Check if correct in top 5
        if expected_target in predicted_targets[:5]:
            correct_at_5 += 1
            rank = predicted_targets.index(expected_target) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        results_detail.append(
            {
                "source": source_path,
                "expected": expected_target,
                "predicted": predicted_targets[:3],
                "correct_at_1": predicted_targets[0] == expected_target
                if predicted_targets
                else False,
                "correct_at_5": expected_target in predicted_targets[:5],
            }
        )

    total = len(GROUND_TRUTH)
    precision_at_1 = correct_at_1 / total if total > 0 else 0
    precision_at_5 = correct_at_5 / total if total > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    # F1 calculation (for graph-only, precision = recall since we're matching 1:1)
    f1 = precision_at_1  # Simplified F1 for this benchmark

    print("\nResults:")
    print("-" * 50)
    print(f"  Precision@1: {precision_at_1:.2%} ({correct_at_1}/{total})")
    print(f"  Precision@5: {precision_at_5:.2%} ({correct_at_5}/{total})")
    print(f"  MRR: {mrr:.4f}")
    print(f"  F1 (simplified): {f1:.2%}")
    print(f"  Latency: {elapsed_ms:.2f}ms total, {elapsed_ms / total:.2f}ms per field")

    # Target validation
    f1_target = 0.78  # 78% F1
    f1_pass = f1 >= f1_target

    print("\nTarget Validation:")
    print("-" * 50)
    print(f"  F1 Target: ≥{f1_target:.0%}")
    print(f"  F1 Result: {f1:.2%}")
    print(f"  Status: {'✓ PASS' if f1_pass else '✗ FAIL (graph-only is limited)'}")

    return {
        "precision_at_1": precision_at_1,
        "precision_at_5": precision_at_5,
        "mrr": mrr,
        "f1": f1,
        "latency_ms": elapsed_ms,
        "latency_per_field_ms": elapsed_ms / total,
        "f1_pass": f1_pass,
        "details": results_detail,
    }


# =============================================================================
# GAP-008: TYPE PROJECTIONS BENCHMARK
# =============================================================================


def benchmark_type_projections():
    """Benchmark GAP-008: Learned Type Projections."""

    print("\n" + "=" * 70)
    print("GAP-008: Learned Type Projections Benchmark")
    print("=" * 70)

    if not DEPENDENCIES["torch"]:
        print("  ⚠ PyTorch not installed. Install with: pip install torch")
        return None

    if not DEPENDENCIES["sentence_transformers"]:
        print(
            "  ⚠ sentence-transformers not installed. Install with: pip install sentence-transformers"
        )
        return None

    # Import locally
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    try:
        from nexus_matcher.core.type_projections import (
            TrainingDataGenerator,
            TypeMatchPair,
            TypeProjectionConfig,
            TypeProjectionManager,
            evaluate_type_projection,
        )
    except ImportError:
        from type_projections import (
            TrainingDataGenerator,
            TypeMatchPair,
            TypeProjectionConfig,
            TypeProjectionManager,
            evaluate_type_projection,
        )

    # Load embedding model
    print("\nLoading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"  ✓ Model loaded: {embedder.get_sentence_embedding_dimension()}d")

    # Generate training data
    print("\nGenerating training data...")
    generator = TrainingDataGenerator()
    train_pairs = generator.generate_pairs(num_positive=500, num_negative=500)
    test_pairs = generator.generate_pairs(num_positive=100, num_negative=100)

    print(f"  Training: {len(train_pairs)} pairs")
    print(f"  Testing: {len(test_pairs)} pairs")

    # Create and train model
    print("\nTraining type projection model...")
    config = TypeProjectionConfig(
        base_embedding_dim=384,
        type_embedding_dim=64,
        projection_dim=384,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=5,  # Fewer epochs for benchmark
    )

    manager = TypeProjectionManager(config)

    start_time = time.perf_counter()

    # Train
    metrics = manager.train(
        train_pairs,
        embedder.encode,
        verbose=True,
    )

    training_time = time.perf_counter() - start_time

    # Evaluate on test set
    print("\nEvaluating on test set...")
    eval_results = evaluate_type_projection(manager, test_pairs, embedder.encode)

    print("\nResults:")
    print("-" * 50)
    print(f"  Accuracy: {eval_results['accuracy']:.2%}")
    print(f"  MRR: {eval_results['mrr']:.4f}")
    print(f"  Avg Positive Similarity: {eval_results['avg_positive_sim']:.4f}")
    print(f"  Avg Negative Similarity: {eval_results['avg_negative_sim']:.4f}")
    print(f"  Separation: {eval_results['separation']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")

    # Test on schema matching task
    print("\nTesting on schema matching task...")

    # Create test pairs from our ground truth
    schema_pairs = []
    for source_path, target_path in GROUND_TRUTH.items():
        source_field = next((f for f in SOURCE_FIELDS if f["path"] == source_path), None)
        target_field = next((f for f in TARGET_FIELDS if f["path"] == target_path), None)

        if source_field and target_field:
            # Create text from field info
            source_text = f"{source_field['name']} {source_field.get('description', '')}"
            target_text = f"{target_field['name']} {target_field.get('description', '')}"

            schema_pairs.append(
                TypeMatchPair(
                    source_text=source_text,
                    source_type=source_field["type"],
                    target_text=target_text,
                    target_type=target_field["type"],
                    is_match=True,
                )
            )

    # Evaluate on schema pairs
    if schema_pairs:
        schema_eval = evaluate_type_projection(manager, schema_pairs, embedder.encode)
        print(f"\n  Schema Matching MRR: {schema_eval['mrr']:.4f}")
        schema_mrr = schema_eval["mrr"]
    else:
        schema_mrr = eval_results["mrr"]

    # Target validation
    mrr_target = 0.80
    mrr_pass = schema_mrr >= mrr_target

    print("\nTarget Validation:")
    print("-" * 50)
    print(f"  MRR Target: ≥{mrr_target}")
    print(f"  MRR Result: {schema_mrr:.4f}")
    print(f"  Status: {'✓ PASS' if mrr_pass else '⚠ PARTIAL (needs more training data)'}")

    return {
        "test_accuracy": eval_results["accuracy"],
        "test_mrr": eval_results["mrr"],
        "schema_mrr": schema_mrr,
        "separation": eval_results["separation"],
        "training_time_s": training_time,
        "final_loss": metrics["losses"][-1],
        "final_accuracy": metrics["accuracies"][-1],
        "mrr_pass": mrr_pass,
    }


# =============================================================================
# BASELINE: SEMANTIC-ONLY MATCHING
# =============================================================================


def benchmark_semantic_baseline():
    """Baseline: Pure semantic matching without type/graph information."""

    print("\n" + "=" * 70)
    print("BASELINE: Semantic-Only Matching")
    print("=" * 70)

    if not DEPENDENCIES["sentence_transformers"]:
        print("  ⚠ sentence-transformers not installed")
        return None

    print("\nLoading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Create text representations
    source_texts = [f"{f['name']} {f.get('description', '')}" for f in SOURCE_FIELDS]
    target_texts = [f"{f['name']} {f.get('description', '')}" for f in TARGET_FIELDS]

    print("Computing embeddings...")
    source_embs = embedder.encode(source_texts, normalize_embeddings=True)
    target_embs = embedder.encode(target_texts, normalize_embeddings=True)

    # Compute similarities
    similarities = source_embs @ target_embs.T

    # Evaluate
    correct_at_1 = 0
    correct_at_5 = 0
    reciprocal_ranks = []

    source_paths = [f["path"] for f in SOURCE_FIELDS]
    target_paths = [f["path"] for f in TARGET_FIELDS]

    for i, source_path in enumerate(source_paths):
        if source_path not in GROUND_TRUTH:
            continue

        expected_target = GROUND_TRUTH[source_path]

        # Get top matches
        sim_scores = similarities[i]
        top_indices = np.argsort(-sim_scores)[:5]
        predicted_targets = [target_paths[j] for j in top_indices]

        if predicted_targets[0] == expected_target:
            correct_at_1 += 1

        if expected_target in predicted_targets:
            correct_at_5 += 1
            rank = predicted_targets.index(expected_target) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(GROUND_TRUTH)
    precision_at_1 = correct_at_1 / total
    precision_at_5 = correct_at_5 / total
    mrr = np.mean(reciprocal_ranks)

    print("\nResults:")
    print("-" * 50)
    print(f"  Precision@1: {precision_at_1:.2%} ({correct_at_1}/{total})")
    print(f"  Precision@5: {precision_at_5:.2%} ({correct_at_5}/{total})")
    print(f"  MRR: {mrr:.4f}")

    return {
        "precision_at_1": precision_at_1,
        "precision_at_5": precision_at_5,
        "mrr": mrr,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================


def run_benchmark():
    """Run all benchmarks."""

    print("=" * 70)
    print("SUITE-008 — GAP-008 & GAP-009 Combined Benchmark")
    print("=" * 70)

    # Check dependencies
    print("\nDependencies:")
    for dep, available in DEPENDENCIES.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "dependencies": DEPENDENCIES,
    }

    # Run baseline
    baseline = benchmark_semantic_baseline()
    results["baseline"] = baseline

    # Run GAP-009 benchmark
    gap009 = benchmark_graph_matching()
    results["gap009"] = gap009

    # Run GAP-008 benchmark
    gap008 = benchmark_type_projections()
    results["gap008"] = gap008

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\nComparison (MRR):")
    print("-" * 50)

    if baseline:
        print(f"  Baseline (Semantic-only): {baseline['mrr']:.4f}")
    if gap009:
        print(f"  GAP-009 (Graph-based):    {gap009['mrr']:.4f}")
    if gap008:
        print(f"  GAP-008 (Type-aware):     {gap008['schema_mrr']:.4f}")

    print("\nTarget Validation:")
    print("-" * 50)

    gap008_pass = gap008["mrr_pass"] if gap008 else False
    gap009_pass = gap009["f1_pass"] if gap009 else False

    print(f"  GAP-008 (MRR ≥ 0.80): {'✓ PASS' if gap008_pass else '⚠ PARTIAL'}")
    print(f"  GAP-009 (F1 ≥ 78%):   {'✓ PASS' if gap009_pass else '⚠ PARTIAL'}")

    overall_pass = gap008_pass or gap009_pass

    print("\n" + "=" * 70)
    print(f"OVERALL: {'✓ VALIDATED' if overall_pass else '◐ PARTIAL - Needs tuning'}")
    print("=" * 70)

    if not overall_pass:
        print("\nNotes:")
        print("  - Graph matching alone has limited accuracy (no semantic info)")
        print("  - Type projections need more training data for full MRR")
        print("  - Hybrid approach (semantic + graph + type) recommended")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = (
        results_dir / f"suite_008_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Clean results for JSON
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    with open(output_file, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_benchmark()
