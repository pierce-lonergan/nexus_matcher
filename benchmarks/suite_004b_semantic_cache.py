"""
!!  SUPERSEDED -- these numbers predate the current benchmark  !!

    This suite was written against a hand-constructed fixture, not the labelled BIRD+OMOP
    benchmark, and in several cases against a defective one. Its results are not
    comparable to anything in benchmarks/results/eval_pipeline_*.json.

    Use instead:
        python benchmarks/datasets/build_benchmarks.py
        python benchmarks/eval_pipeline.py --benchmark combined

    Kept as a historical record. Do not cite.

benchmarks/suite_004b_semantic_cache.py | SUITE-004b: Semantic Content Cache Benchmark
Validates GAP-004 (Semantic Content Cache) implementation against research targets.

Research Targets (README_RESEARCH_3.md, Lines 21-37):
- Cache embeddings by content hash using BLAKE3
- 50-70% cost reduction for embedding computation
- 40-60% cache hit rate for typical workloads

Usage:
    python benchmarks/suite_004b_semantic_cache.py
"""

import gc
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class BenchmarkResult:
    """Benchmark result container."""

    suite: str
    run_id: str
    timestamp: str
    metrics: dict[str, float]
    passes: dict[str, bool]
    summary: str


def benchmark_cost_reduction() -> dict[str, float]:
    """
    Benchmark embedding computation cost reduction.

    Research target: 50-70% cost reduction via caching.
    """
    from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

    cache = SemanticContentCache(max_size=10000, normalize=True)

    # Simulate realistic field names with duplicates
    field_names = [
        "customer_email_address",
        "customer_phone_number",
        "account_balance",
        "transaction_date",
        "user_id",
        "customer_email_address",  # Duplicate
        "account_id",
        "customer_phone_number",  # Duplicate
        "billing_address",
        "shipping_address",
        "customer_email_address",  # Duplicate
        "order_total",
        "account_balance",  # Duplicate
    ] * 100  # 1300 total, with duplicates

    compute_count = [0]

    def simulate_embedding_compute(text: str) -> np.ndarray:
        compute_count[0] += 1
        # Simulate embedding computation time
        time.sleep(0.0001)  # 0.1ms per embedding
        return np.random.rand(768).astype(np.float32)

    # Run queries
    start = time.perf_counter()
    for field in field_names:
        _ = cache.get_or_compute(field, simulate_embedding_compute)
    elapsed = time.perf_counter() - start

    total_queries = len(field_names)
    computations = compute_count[0]
    cache_hits = total_queries - computations

    cost_reduction = (cache_hits / total_queries) * 100

    return {
        "total_queries": total_queries,
        "computations": computations,
        "cache_hits": cache_hits,
        "cost_reduction_pct": cost_reduction,
        "elapsed_sec": elapsed,
    }


def benchmark_hit_rate_patterns() -> dict[str, float]:
    """
    Benchmark cache hit rate for different workload patterns.
    """
    from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

    results = {}

    # Pattern 1: Low repetition (30%)
    cache = SemanticContentCache(max_size=10000)
    fields_low = [f"field_{i % 70}" for i in range(100)]
    for f in fields_low:
        cache.get_or_compute(f, lambda t: np.random.rand(768).astype(np.float32))
    stats = cache.get_stats()
    results["hit_rate_30pct_rep"] = stats.hit_rate

    # Pattern 2: Medium repetition (50%)
    cache = SemanticContentCache(max_size=10000)
    fields_med = [f"field_{i % 50}" for i in range(100)]
    for f in fields_med:
        cache.get_or_compute(f, lambda t: np.random.rand(768).astype(np.float32))
    stats = cache.get_stats()
    results["hit_rate_50pct_rep"] = stats.hit_rate

    # Pattern 3: High repetition (70%)
    cache = SemanticContentCache(max_size=10000)
    fields_high = [f"field_{i % 30}" for i in range(100)]
    for f in fields_high:
        cache.get_or_compute(f, lambda t: np.random.rand(768).astype(np.float32))
    stats = cache.get_stats()
    results["hit_rate_70pct_rep"] = stats.hit_rate

    return results


def benchmark_blake3_vs_sha256() -> dict[str, float]:
    """
    Benchmark BLAKE3 vs SHA-256 hashing performance.
    """
    import hashlib

    from nexus_matcher.infrastructure.adapters.caches.content import ContentHasher

    hasher = ContentHasher()
    content = "customer_email_address_field_with_some_description" * 10
    iterations = 10000

    # BLAKE3
    gc.collect()
    start = time.perf_counter()
    for _ in range(iterations):
        hasher.hash(content)
    blake3_time = time.perf_counter() - start

    # SHA-256
    gc.collect()
    start = time.perf_counter()
    for _ in range(iterations):
        hashlib.sha256(content.encode()).hexdigest()
    sha256_time = time.perf_counter() - start

    return {
        "blake3_time_ms": blake3_time * 1000,
        "sha256_time_ms": sha256_time * 1000,
        "speedup_ratio": sha256_time / blake3_time,
        "blake3_ops_per_sec": iterations / blake3_time,
    }


def benchmark_batch_efficiency() -> dict[str, float]:
    """
    Benchmark batch get_or_compute efficiency.
    """
    from nexus_matcher.infrastructure.adapters.caches.content import SemanticContentCache

    cache = SemanticContentCache(max_size=10000)

    # Pre-populate some entries
    for i in range(50):
        cache.set_by_content(f"field_{i}", np.random.rand(768).astype(np.float32))

    # Batch request with mix of cached and uncached
    batch = [f"field_{i}" for i in range(100)]
    compute_count = [0]

    def batch_compute(texts):
        compute_count[0] += len(texts)
        return [np.random.rand(768).astype(np.float32) for _ in texts]

    cache.batch_get_or_compute(batch, batch_compute)

    return {
        "batch_size": len(batch),
        "computations_needed": compute_count[0],
        "computations_saved": len(batch) - compute_count[0],
        "efficiency_pct": (1 - compute_count[0] / len(batch)) * 100,
    }


def run_suite_004b() -> BenchmarkResult:
    """Run complete SUITE-004b semantic content cache benchmark."""

    print("=" * 70)
    print("SUITE-004b: Semantic Content Cache Benchmark")
    print("=" * 70)
    print()

    all_metrics = {}
    passes = {}

    # Cost reduction benchmark
    print("Running cost reduction benchmark...")
    cost_metrics = benchmark_cost_reduction()
    all_metrics.update(cost_metrics)

    # Research target: 50-70% cost reduction
    passes["cost_reduction_50pct"] = cost_metrics["cost_reduction_pct"] >= 50
    print(f"  Total Queries: {cost_metrics['total_queries']}")
    print(f"  Computations: {cost_metrics['computations']}")
    print(f"  Cache Hits: {cost_metrics['cache_hits']}")
    print(f"  Cost Reduction: {cost_metrics['cost_reduction_pct']:.1f}%")
    print(f"  ✓ Cost reduction ≥50%: {'PASS' if passes['cost_reduction_50pct'] else 'FAIL'}")
    print()

    # Hit rate patterns benchmark
    print("Running hit rate patterns benchmark...")
    hit_metrics = benchmark_hit_rate_patterns()
    all_metrics.update(hit_metrics)

    # Research target: 40-60% hit rate
    passes["hit_rate_40pct"] = hit_metrics["hit_rate_50pct_rep"] >= 0.40
    print(f"  30% repetition: {hit_metrics['hit_rate_30pct_rep']:.1%}")
    print(f"  50% repetition: {hit_metrics['hit_rate_50pct_rep']:.1%}")
    print(f"  70% repetition: {hit_metrics['hit_rate_70pct_rep']:.1%}")
    print(f"  ✓ Hit rate ≥40%: {'PASS' if passes['hit_rate_40pct'] else 'FAIL'}")
    print()

    # Hashing performance
    print("Running hashing performance benchmark...")
    hash_metrics = benchmark_blake3_vs_sha256()
    all_metrics.update(hash_metrics)

    passes["hashing_functional"] = hash_metrics["blake3_ops_per_sec"] > 10000
    print(f"  BLAKE3 time: {hash_metrics['blake3_time_ms']:.2f}ms for 10K hashes")
    print(f"  SHA-256 time: {hash_metrics['sha256_time_ms']:.2f}ms for 10K hashes")
    print(f"  Speedup ratio: {hash_metrics['speedup_ratio']:.2f}x")
    print(f"  BLAKE3 throughput: {hash_metrics['blake3_ops_per_sec']:,.0f} ops/s")
    print(f"  ✓ Hashing functional: {'PASS' if passes['hashing_functional'] else 'FAIL'}")
    print()

    # Batch efficiency
    print("Running batch efficiency benchmark...")
    batch_metrics = benchmark_batch_efficiency()
    all_metrics.update(batch_metrics)

    passes["batch_efficiency"] = batch_metrics["efficiency_pct"] >= 40
    print(f"  Batch size: {batch_metrics['batch_size']}")
    print(f"  Computations needed: {batch_metrics['computations_needed']}")
    print(f"  Computations saved: {batch_metrics['computations_saved']}")
    print(f"  Efficiency: {batch_metrics['efficiency_pct']:.1f}%")
    print(f"  ✓ Batch efficiency ≥40%: {'PASS' if passes['batch_efficiency'] else 'FAIL'}")
    print()

    # Summary
    all_pass = all(passes.values())
    pass_count = sum(passes.values())
    total_count = len(passes)

    print("=" * 70)
    print(f"SUITE-004b RESULTS: {pass_count}/{total_count} targets met")
    print("=" * 70)

    for name, passed in passes.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    print()
    if all_pass:
        print(">>> GAP-004 VALIDATED: All semantic cache targets met <<<")
        summary = "VALIDATED"
    else:
        print(">>> GAP-004 NOT VALIDATED: Some targets not met <<<")
        summary = "NOT_VALIDATED"

    return BenchmarkResult(
        suite="SUITE-004b",
        run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        metrics=all_metrics,
        passes=passes,
        summary=summary,
    )


if __name__ == "__main__":
    result = run_suite_004b()
