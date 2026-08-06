#!/usr/bin/env python3
"""
!!  SUPERSEDED -- these numbers predate the current benchmark  !!

    This suite was written against a hand-constructed fixture, not the labelled BIRD+OMOP
    benchmark, and in several cases against a defective one. Its results are not
    comparable to anything in benchmarks/results/eval_pipeline_*.json.

    Use instead:
        python benchmarks/datasets/build_benchmarks.py
        python benchmarks/eval_pipeline.py --benchmark combined

    Kept as a historical record. Do not cite.

SUITE-007 — ModernBERT vs MiniLM Embedding Benchmark
═══════════════════════════════════════════════════════════════════

Research Reference: README_RESEARCH_2.md, Lines 9-12
Gap: GAP-007 — ModernBERT Integration

Target Metrics:
- 4x faster inference (claimed)
- Same or better accuracy
- Supports 8192 token context (vs 512 for MiniLM)

Models Compared:
1. sentence-transformers/all-MiniLM-L6-v2 (current baseline)
2. nomic-ai/modernbert-embed-base (ModernBERT-based)

Prerequisites:
    pip install sentence-transformers transformers>=4.48.0

Note: ModernBERT requires transformers >= 4.48.0
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Check dependencies
DEPENDENCIES_OK = True
MISSING = []

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    DEPENDENCIES_OK = False
    MISSING.append("sentence-transformers")

try:
    import transformers

    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    DEPENDENCIES_OK = False
    MISSING.append("transformers")
    TRANSFORMERS_VERSION = "0.0.0"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Models to compare
BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODERNBERT_MODEL = "nomic-ai/modernbert-embed-base"

# Benchmark configuration
BATCH_SIZES = [1, 8, 16, 32, 64]
WARMUP_ITERATIONS = 3
MEASUREMENT_ITERATIONS = 10

# Sample texts for benchmarking (schema field descriptions)
SAMPLE_TEXTS = [
    "customer email address primary contact",
    "transaction amount in USD currency",
    "account balance current available",
    "payment date timestamp",
    "user first name",
    "product category code",
    "order status description",
    "shipping address line 1",
    "credit card last four digits",
    "phone number primary",
    "date of birth",
    "annual income range",
    "employment status code",
    "loan amount requested",
    "interest rate percentage",
    "account open date",
    "last login timestamp",
    "security question answer",
    "preferred language code",
    "marketing consent flag",
    "risk score calculated",
    "fraud alert indicator",
    "kyc verification status",
    "document type identifier",
    "branch location code",
    "relationship manager id",
    "portfolio value total",
    "dividend payment amount",
    "tax identification number",
    "beneficiary name primary",
    "wire transfer reference",
    "check deposit amount",
] * 2  # 64 texts total

# ModernBERT requires prefixes
MODERNBERT_QUERY_PREFIX = "search_query: "
MODERNBERT_DOC_PREFIX = "search_document: "


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model_name: str
    batch_size: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_texts_per_sec: float
    embedding_dim: int
    latencies: list


@dataclass
class ModelComparison:
    """Comparison between baseline and ModernBERT."""

    batch_size: int
    baseline_latency_ms: float
    modernbert_latency_ms: float
    speedup_ratio: float
    meets_target: bool
    target_speedup: float


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================


def get_model_info(model: SentenceTransformer) -> dict:
    """Get model information."""
    return {
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
    }


def benchmark_model(
    model: SentenceTransformer,
    model_name: str,
    texts: list[str],
    batch_size: int,
    iterations: int,
    add_prefix: bool = False,
    prefix: str = "",
) -> BenchmarkResult:
    """Benchmark a sentence transformer model."""
    latencies = []

    # Prepare texts (add prefix for ModernBERT if needed)
    if add_prefix and prefix:
        batch_texts = [prefix + t for t in texts[:batch_size]]
    else:
        batch_texts = texts[:batch_size]

    for _ in range(iterations):
        start = time.perf_counter()
        embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    throughput = (batch_size / np.mean(latencies_arr)) * 1000

    return BenchmarkResult(
        model_name=model_name,
        batch_size=batch_size,
        avg_latency_ms=float(np.mean(latencies_arr)),
        p50_latency_ms=float(np.percentile(latencies_arr, 50)),
        p95_latency_ms=float(np.percentile(latencies_arr, 95)),
        p99_latency_ms=float(np.percentile(latencies_arr, 99)),
        throughput_texts_per_sec=throughput,
        embedding_dim=model.get_sentence_embedding_dimension(),
        latencies=[float(x) for x in latencies],
    )


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between embeddings."""
    # Normalize
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=-1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=-1, keepdims=True)

    # Cosine similarity
    if emb1_norm.ndim == 1:
        return float(np.dot(emb1_norm, emb2_norm))
    else:
        similarities = np.sum(emb1_norm * emb2_norm, axis=-1)
        return float(np.mean(similarities))


def test_semantic_quality(
    baseline_model: SentenceTransformer,
    modernbert_model: SentenceTransformer,
) -> dict:
    """Test semantic quality of embeddings."""

    # Test pairs (similar vs dissimilar)
    test_pairs = [
        # Similar pairs (should have high similarity)
        ("customer email address", "email contact information"),
        ("transaction amount", "payment value"),
        ("account balance", "available funds"),
        ("date of birth", "birth date"),
        ("phone number", "telephone contact"),
        # Dissimilar pairs (should have low similarity)
        ("customer email", "interest rate"),
        ("account balance", "shipping address"),
        ("date of birth", "product category"),
        ("phone number", "loan amount"),
        ("payment date", "risk score"),
    ]

    similar_baseline = []
    similar_modernbert = []
    dissimilar_baseline = []
    dissimilar_modernbert = []

    for i, (text1, text2) in enumerate(test_pairs):
        # Baseline embeddings
        emb1_base = baseline_model.encode(text1, convert_to_numpy=True)
        emb2_base = baseline_model.encode(text2, convert_to_numpy=True)
        sim_base = compute_similarity(emb1_base, emb2_base)

        # ModernBERT embeddings (with prefix)
        emb1_modern = modernbert_model.encode(MODERNBERT_DOC_PREFIX + text1, convert_to_numpy=True)
        emb2_modern = modernbert_model.encode(MODERNBERT_DOC_PREFIX + text2, convert_to_numpy=True)
        sim_modern = compute_similarity(emb1_modern, emb2_modern)

        if i < 5:  # First 5 are similar pairs
            similar_baseline.append(sim_base)
            similar_modernbert.append(sim_modern)
        else:  # Last 5 are dissimilar pairs
            dissimilar_baseline.append(sim_base)
            dissimilar_modernbert.append(sim_modern)

    return {
        "baseline": {
            "avg_similar": float(np.mean(similar_baseline)),
            "avg_dissimilar": float(np.mean(dissimilar_baseline)),
            "separation": float(np.mean(similar_baseline) - np.mean(dissimilar_baseline)),
        },
        "modernbert": {
            "avg_similar": float(np.mean(similar_modernbert)),
            "avg_dissimilar": float(np.mean(dissimilar_modernbert)),
            "separation": float(np.mean(similar_modernbert) - np.mean(dissimilar_modernbert)),
        },
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================


def run_benchmark():
    """Run the full ModernBERT vs MiniLM benchmark."""

    print("=" * 70)
    print("SUITE-007 — ModernBERT vs MiniLM Embedding Benchmark")
    print("=" * 70)
    print()

    # Check dependencies
    if not DEPENDENCIES_OK:
        print(f"ERROR: Missing dependencies: {', '.join(MISSING)}")
        print("Install with: pip install sentence-transformers transformers>=4.48.0")
        sys.exit(1)

    # Check transformers version
    print("Environment:")
    print(f"  Transformers version: {TRANSFORMERS_VERSION}")

    # Parse version
    try:
        major, minor, patch = TRANSFORMERS_VERSION.split(".")[:3]
        version_ok = int(major) > 4 or (int(major) == 4 and int(minor) >= 48)
    except:
        version_ok = False

    if not version_ok:
        print("  ⚠ WARNING: ModernBERT requires transformers >= 4.48.0")
        print(f"    Current version: {TRANSFORMERS_VERSION}")
        print("    Update with: pip install transformers>=4.48.0")
        print()
        print("Continuing with baseline model only...")
        modernbert_available = False
    else:
        print("  ✓ Transformers version OK")
        modernbert_available = True

    print()

    # Load baseline model
    print(f"Loading baseline model: {BASELINE_MODEL}")
    try:
        baseline_model = SentenceTransformer(BASELINE_MODEL)
        baseline_info = get_model_info(baseline_model)
        print(
            f"  ✓ Loaded: {baseline_info['embedding_dim']}d, max_seq={baseline_info['max_seq_length']}"
        )
    except Exception as e:
        print(f"  ✗ Failed to load baseline: {e}")
        sys.exit(1)
    print()

    # Load ModernBERT model
    modernbert_model = None
    modernbert_info = None

    if modernbert_available:
        print(f"Loading ModernBERT model: {MODERNBERT_MODEL}")
        try:
            modernbert_model = SentenceTransformer(MODERNBERT_MODEL, trust_remote_code=True)
            modernbert_info = get_model_info(modernbert_model)
            print(
                f"  ✓ Loaded: {modernbert_info['embedding_dim']}d, max_seq={modernbert_info['max_seq_length']}"
            )
        except Exception as e:
            print(f"  ⚠ Failed to load ModernBERT: {e}")
            print("  Continuing with baseline only...")
            modernbert_available = False
        print()

    # Semantic quality test
    if modernbert_model:
        print("Semantic Quality Test:")
        print("-" * 50)
        quality = test_semantic_quality(baseline_model, modernbert_model)

        print("  Baseline (MiniLM):")
        print(f"    Similar pairs avg:    {quality['baseline']['avg_similar']:.4f}")
        print(f"    Dissimilar pairs avg: {quality['baseline']['avg_dissimilar']:.4f}")
        print(f"    Separation:           {quality['baseline']['separation']:.4f}")
        print()
        print("  ModernBERT:")
        print(f"    Similar pairs avg:    {quality['modernbert']['avg_similar']:.4f}")
        print(f"    Dissimilar pairs avg: {quality['modernbert']['avg_dissimilar']:.4f}")
        print(f"    Separation:           {quality['modernbert']['separation']:.4f}")

        # Better separation = better quality
        quality_pass = (
            quality["modernbert"]["separation"] >= quality["baseline"]["separation"] * 0.9
        )
        print(f"  Quality (≥90% separation): {'✓ PASS' if quality_pass else '⚠ NEEDS REVIEW'}")
        print()
    else:
        quality = None
        quality_pass = None

    # Warmup
    print(f"Warmup ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        _ = baseline_model.encode(SAMPLE_TEXTS[:32], show_progress_bar=False)
        if modernbert_model:
            texts = [MODERNBERT_DOC_PREFIX + t for t in SAMPLE_TEXTS[:32]]
            _ = modernbert_model.encode(texts, show_progress_bar=False)
    print("  ✓ Warmup complete")
    print()

    # Run benchmarks
    print(f"Running benchmarks ({MEASUREMENT_ITERATIONS} iterations each)...")
    print()

    baseline_results = []
    modernbert_results = []
    comparisons = []

    target_speedup = 2.0  # Conservative target (research claims 4x)

    for batch_size in BATCH_SIZES:
        print(f"Batch size: {batch_size}")

        # Baseline benchmark
        baseline_result = benchmark_model(
            baseline_model,
            "MiniLM-L6",
            SAMPLE_TEXTS,
            batch_size,
            MEASUREMENT_ITERATIONS,
        )
        baseline_results.append(baseline_result)
        print(
            f"  MiniLM:     {baseline_result.avg_latency_ms:7.2f}ms (P95: {baseline_result.p95_latency_ms:.2f}ms)"
        )

        # ModernBERT benchmark
        if modernbert_model:
            modernbert_result = benchmark_model(
                modernbert_model,
                "ModernBERT",
                SAMPLE_TEXTS,
                batch_size,
                MEASUREMENT_ITERATIONS,
                add_prefix=True,
                prefix=MODERNBERT_DOC_PREFIX,
            )
            modernbert_results.append(modernbert_result)
            print(
                f"  ModernBERT: {modernbert_result.avg_latency_ms:7.2f}ms (P95: {modernbert_result.p95_latency_ms:.2f}ms)"
            )

            # Compute speedup
            speedup = baseline_result.avg_latency_ms / modernbert_result.avg_latency_ms
            meets_target = speedup >= target_speedup

            comparisons.append(
                ModelComparison(
                    batch_size=batch_size,
                    baseline_latency_ms=baseline_result.avg_latency_ms,
                    modernbert_latency_ms=modernbert_result.avg_latency_ms,
                    speedup_ratio=speedup,
                    meets_target=meets_target,
                    target_speedup=target_speedup,
                )
            )

            if speedup > 1:
                print(f"  Speedup:    {speedup:.2f}x {'✓' if meets_target else '⚠'}")
            else:
                print(f"  Speedup:    {speedup:.2f}x (ModernBERT slower)")

        print()

    # Summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print()

    print("Model Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'Dim':>6} {'Max Seq':>10} {'Batch-32 (ms)':>15}")
    print("-" * 60)

    base_32 = next((r for r in baseline_results if r.batch_size == 32), None)
    print(
        f"{'MiniLM-L6':<20} {baseline_info['embedding_dim']:>6} {baseline_info['max_seq_length']:>10} {base_32.avg_latency_ms if base_32 else 'N/A':>15.2f}"
    )

    if modernbert_model and modernbert_info:
        modern_32 = next((r for r in modernbert_results if r.batch_size == 32), None)
        print(
            f"{'ModernBERT':<20} {modernbert_info['embedding_dim']:>6} {modernbert_info['max_seq_length']:>10} {modern_32.avg_latency_ms if modern_32 else 'N/A':>15.2f}"
        )
    print()

    # Latency comparison
    if comparisons:
        print("Latency by Batch Size:")
        print("-" * 60)
        print(f"{'Batch':>8} {'MiniLM (ms)':>12} {'ModernBERT (ms)':>16} {'Speedup':>10}")
        print("-" * 60)
        for c in comparisons:
            print(
                f"{c.batch_size:>8} {c.baseline_latency_ms:>12.2f} {c.modernbert_latency_ms:>16.2f} {c.speedup_ratio:>9.2f}x"
            )
        print()

        avg_speedup = np.mean([c.speedup_ratio for c in comparisons])
        speedup_pass = avg_speedup >= target_speedup

        print("Target Validation (GAP-007):")
        print("-" * 50)
        print(f"  Average Speedup: {avg_speedup:.2f}x (target: {target_speedup}x)")
        print(f"  Speedup Target: {'✓ PASS' if speedup_pass else '✗ FAIL'}")

        if quality_pass is not None:
            print(f"  Quality Target: {'✓ PASS' if quality_pass else '✗ FAIL'}")

        # Context length advantage
        if modernbert_info:
            print(
                f"  Context Length: {modernbert_info['max_seq_length']} vs {baseline_info['max_seq_length']} ({modernbert_info['max_seq_length'] // baseline_info['max_seq_length']}x longer)"
            )
    else:
        speedup_pass = False
        avg_speedup = 0

    print()

    # Overall verdict
    if modernbert_model:
        all_pass = speedup_pass and (quality_pass or quality_pass is None)
    else:
        all_pass = False
        print("ModernBERT not available - update transformers to >= 4.48.0")

    print("=" * 70)
    print(f"GAP-007 VALIDATION: {'✓ VALIDATED' if all_pass else '◐ PARTIAL'}")
    print("=" * 70)

    if modernbert_model and not speedup_pass:
        print()
        print("Note: ModernBERT may be slower on CPU without Flash Attention.")
        print("      Full speed benefits require GPU with Flash Attention 2.")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "baseline": BASELINE_MODEL,
            "modernbert": MODERNBERT_MODEL if modernbert_model else None,
        },
        "transformers_version": TRANSFORMERS_VERSION,
        "config": {
            "batch_sizes": BATCH_SIZES,
            "warmup_iterations": WARMUP_ITERATIONS,
            "measurement_iterations": MEASUREMENT_ITERATIONS,
        },
        "model_info": {
            "baseline": baseline_info,
            "modernbert": modernbert_info,
        },
        "quality": quality,
        "baseline_results": [asdict(r) for r in baseline_results],
        "modernbert_results": [asdict(r) for r in modernbert_results] if modernbert_results else [],
        "comparisons": [asdict(c) for c in comparisons] if comparisons else [],
        "validation": {
            "speedup_pass": bool(speedup_pass),
            "quality_pass": bool(quality_pass) if quality_pass is not None else None,
            "avg_speedup": float(avg_speedup) if avg_speedup else None,
            "overall_pass": bool(all_pass),
        },
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = (
        results_dir / f"suite_007_modernbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_pass


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
