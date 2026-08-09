"""
!!  SUPERSEDED -- these numbers predate the current benchmark  !!

    This suite was written against a hand-constructed fixture, not the labelled BIRD+OMOP
    benchmark, and in several cases against a defective one. Its results are not
    comparable to anything in benchmarks/results/eval_pipeline_*.json.

    Use instead:
        python benchmarks/datasets/build_benchmarks.py
        python benchmarks/eval_pipeline.py --benchmark combined

    Kept as a historical record. Do not cite.

benchmarks/suite_004_cache_performance.py | SUITE-004: Cache Performance Benchmark
Validates GAP-003 (L1 LRU Cache) implementation against research targets.

Research Targets (README_RESEARCH_3.md, Lines 21-37):
- L1 in-process LRU cache: 5K entries, sub-1ms access, 15-25% hit rate
- Combined L1+L2: 60-75% latency reduction
- Overall cache hit rate: ≥40%

Usage:
    python benchmarks/suite_004_cache_performance.py
"""

import gc
import statistics
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    suite: str
    run_id: str
    timestamp: str
    environment: dict
    metrics: dict[str, float]
    passes: dict[str, bool]
    summary: str


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    warmup_runs: int = 3
    measurement_runs: int = 5
    cache_size: int = 5000
    query_count: int = 1000
    embedding_dim: int = 768
    repetition_rate: float = 0.6  # 60% repeated queries


def get_environment_info() -> dict:
    """Capture environment configuration."""
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "timestamp": datetime.now().isoformat(),
    }


def generate_test_embeddings(
    count: int,
    dim: int = 768,
) -> list[tuple[str, np.ndarray]]:
    """Generate test embeddings with keys."""
    return [(f"query_hash_{i:06d}", np.random.rand(dim).astype(np.float32)) for i in range(count)]


def generate_query_sequence(
    embeddings: list[tuple[str, np.ndarray]],
    query_count: int,
    repetition_rate: float,
) -> list[str]:
    """
    Generate query sequence with specified repetition rate.

    Higher repetition simulates realistic cache-friendly workload.
    """
    unique_count = int(query_count * (1 - repetition_rate))
    repeated_count = query_count - unique_count

    # First queries are unique
    unique_keys = [emb[0] for emb in embeddings[:unique_count]]

    # Repeated queries sample from a smaller set
    repeat_pool = [emb[0] for emb in embeddings[: unique_count // 2]]
    repeated_keys = [repeat_pool[i % len(repeat_pool)] for i in range(repeated_count)]

    # Interleave
    sequence = []
    u_idx, r_idx = 0, 0
    for i in range(query_count):
        if i % 3 == 0 and u_idx < len(unique_keys):
            sequence.append(unique_keys[u_idx])
            u_idx += 1
        elif r_idx < len(repeated_keys):
            sequence.append(repeated_keys[r_idx])
            r_idx += 1
        elif u_idx < len(unique_keys):
            sequence.append(unique_keys[u_idx])
            u_idx += 1

    return sequence


def benchmark_l1_latency(
    config: BenchmarkConfig,
) -> dict[str, float]:
    """
    Benchmark L1 cache access latency.

    Research target: sub-1ms (<1ms) access time.
    """
    from nexus_matcher.domain.ports.cache import CacheConfig
    from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache

    cache_config = CacheConfig(max_size=config.cache_size)
    cache = L1LRUCache(cache_config)

    # Generate test data
    embeddings = generate_test_embeddings(config.cache_size, config.embedding_dim)

    # Populate cache
    for key, embedding in embeddings:
        cache.set(key, embedding)

    # Warmup
    for _ in range(config.warmup_runs):
        for key, _ in embeddings[:100]:
            _ = cache.get(key)

    # Measure GET latency
    get_latencies = []
    for _ in range(config.measurement_runs):
        gc.collect()
        for key, _ in embeddings[:1000]:
            start = time.perf_counter()
            _ = cache.get(key)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            get_latencies.append(elapsed)

    # Measure SET latency
    set_latencies = []
    new_embeddings = generate_test_embeddings(1000, config.embedding_dim)
    for _ in range(config.measurement_runs):
        gc.collect()
        # Clear and repopulate to measure SET
        temp_cache = L1LRUCache(cache_config)
        for key, embedding in new_embeddings:
            start = time.perf_counter()
            temp_cache.set(key, embedding)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            set_latencies.append(elapsed)

    return {
        "get_p50_ms": statistics.median(get_latencies),
        "get_p95_ms": sorted(get_latencies)[int(len(get_latencies) * 0.95)],
        "get_p99_ms": sorted(get_latencies)[int(len(get_latencies) * 0.99)],
        "get_avg_ms": statistics.mean(get_latencies),
        "get_max_ms": max(get_latencies),
        "set_p50_ms": statistics.median(set_latencies),
        "set_p95_ms": sorted(set_latencies)[int(len(set_latencies) * 0.95)],
        "set_avg_ms": statistics.mean(set_latencies),
    }


def benchmark_l1_hit_rate(
    config: BenchmarkConfig,
) -> dict[str, float]:
    """
    Benchmark L1 cache hit rate.

    Research target: 15-25% for L1 alone, 40%+ combined.
    """
    from nexus_matcher.domain.ports.cache import CacheConfig
    from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache

    cache_config = CacheConfig(max_size=config.cache_size)
    cache = L1LRUCache(cache_config)

    # Generate embeddings
    embeddings = generate_test_embeddings(
        int(config.cache_size * 1.5),  # More than cache size
        config.embedding_dim,
    )

    # Create query sequence with repetition
    query_sequence = generate_query_sequence(
        embeddings,
        config.query_count,
        config.repetition_rate,
    )

    # Run queries - first populate, then measure
    embedding_map = dict(embeddings)

    for _ in range(config.warmup_runs):
        cache.clear()
        for key in query_sequence[:100]:
            result = cache.get(key)
            if result is None and key in embedding_map:
                cache.set(key, embedding_map[key])

    # Measurement runs
    hit_rates = []
    for _ in range(config.measurement_runs):
        cache.clear()
        for key in query_sequence:
            result = cache.get(key)
            if result is None and key in embedding_map:
                cache.set(key, embedding_map[key])

        stats = cache.get_stats()
        hit_rates.append(stats.hit_rate)

    return {
        "hit_rate_avg": statistics.mean(hit_rates),
        "hit_rate_min": min(hit_rates),
        "hit_rate_max": max(hit_rates),
        "hits_total": stats.hits,
        "misses_total": stats.misses,
        "cache_size": stats.size,
        "evictions": stats.evictions,
    }


def benchmark_l1_throughput(
    config: BenchmarkConfig,
) -> dict[str, float]:
    """
    Benchmark L1 cache throughput (operations per second).
    """
    from nexus_matcher.domain.ports.cache import CacheConfig
    from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache

    cache_config = CacheConfig(max_size=config.cache_size)
    cache = L1LRUCache(cache_config)

    # Generate test data
    embeddings = generate_test_embeddings(config.cache_size, config.embedding_dim)
    for key, embedding in embeddings:
        cache.set(key, embedding)

    keys = [key for key, _ in embeddings]

    # Warmup
    for _ in range(config.warmup_runs):
        for key in keys[:100]:
            _ = cache.get(key)

    # Measure throughput
    throughputs = []
    for _ in range(config.measurement_runs):
        gc.collect()

        start = time.perf_counter()
        for key in keys:
            _ = cache.get(key)
        elapsed = time.perf_counter() - start

        ops_per_second = len(keys) / elapsed
        throughputs.append(ops_per_second)

    return {
        "throughput_avg_ops_s": statistics.mean(throughputs),
        "throughput_min_ops_s": min(throughputs),
        "throughput_max_ops_s": max(throughputs),
    }


def benchmark_l1_memory(
    config: BenchmarkConfig,
) -> dict[str, float]:
    """
    Benchmark L1 cache memory usage.

    Research target: ~50MB for L1 cache.
    """
    import tracemalloc

    from nexus_matcher.domain.ports.cache import CacheConfig
    from nexus_matcher.infrastructure.adapters.caches.memory import L1LRUCache

    gc.collect()
    tracemalloc.start()

    cache_config = CacheConfig(max_size=config.cache_size)
    cache = L1LRUCache(cache_config)

    # Generate and store embeddings
    embeddings = generate_test_embeddings(config.cache_size, config.embedding_dim)
    for key, embedding in embeddings:
        cache.set(key, embedding)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "memory_current_mb": current / 1024 / 1024,
        "memory_peak_mb": peak / 1024 / 1024,
        "entries": config.cache_size,
        "bytes_per_entry": current / config.cache_size,
    }


def run_suite_004(config: BenchmarkConfig | None = None) -> BenchmarkResult:
    """
    Run complete SUITE-004 cache performance benchmark.
    """
    if config is None:
        config = BenchmarkConfig()

    print("=" * 70)
    print("SUITE-004: Cache Performance Benchmark")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Cache Size: {config.cache_size}")
    print(f"  Query Count: {config.query_count}")
    print(f"  Embedding Dim: {config.embedding_dim}")
    print(f"  Repetition Rate: {config.repetition_rate:.0%}")
    print(f"  Warmup Runs: {config.warmup_runs}")
    print(f"  Measurement Runs: {config.measurement_runs}")
    print()

    environment = get_environment_info()
    all_metrics = {}
    passes = {}

    # Latency benchmark
    print("Running latency benchmark...")
    latency_metrics = benchmark_l1_latency(config)
    all_metrics.update(latency_metrics)

    # Research target: sub-1ms
    passes["latency_sub_1ms"] = latency_metrics["get_p95_ms"] < 1.0
    print(f"  GET P50: {latency_metrics['get_p50_ms']:.4f}ms")
    print(f"  GET P95: {latency_metrics['get_p95_ms']:.4f}ms")
    print(f"  GET P99: {latency_metrics['get_p99_ms']:.4f}ms")
    print(f"  SET P50: {latency_metrics['set_p50_ms']:.4f}ms")
    print(f"  ✓ Sub-1ms target: {'PASS' if passes['latency_sub_1ms'] else 'FAIL'}")
    print()

    # Hit rate benchmark
    print("Running hit rate benchmark...")
    hit_metrics = benchmark_l1_hit_rate(config)
    all_metrics.update(hit_metrics)

    # Research target: L1 15-25%, overall 40%+
    # With 60% repetition, L1 should hit 40%+
    passes["hit_rate_target"] = hit_metrics["hit_rate_avg"] >= 0.15
    print(f"  Hit Rate Avg: {hit_metrics['hit_rate_avg']:.2%}")
    print(
        f"  Hit Rate Range: {hit_metrics['hit_rate_min']:.2%} - {hit_metrics['hit_rate_max']:.2%}"
    )
    print(f"  Evictions: {hit_metrics['evictions']}")
    print(f"  ✓ Hit rate ≥15%: {'PASS' if passes['hit_rate_target'] else 'FAIL'}")
    print()

    # Throughput benchmark
    print("Running throughput benchmark...")
    throughput_metrics = benchmark_l1_throughput(config)
    all_metrics.update(throughput_metrics)

    # Target: >100K ops/sec for in-memory
    passes["throughput_target"] = throughput_metrics["throughput_avg_ops_s"] > 100000
    print(f"  Throughput Avg: {throughput_metrics['throughput_avg_ops_s']:,.0f} ops/s")
    print(
        f"  Throughput Range: {throughput_metrics['throughput_min_ops_s']:,.0f} - {throughput_metrics['throughput_max_ops_s']:,.0f}"
    )
    print(f"  ✓ Throughput >100K: {'PASS' if passes['throughput_target'] else 'FAIL'}")
    print()

    # Memory benchmark
    print("Running memory benchmark...")
    memory_metrics = benchmark_l1_memory(config)
    all_metrics.update(memory_metrics)

    # Target: <100MB for 5K embeddings
    passes["memory_target"] = memory_metrics["memory_peak_mb"] < 100
    print(f"  Memory Current: {memory_metrics['memory_current_mb']:.2f} MB")
    print(f"  Memory Peak: {memory_metrics['memory_peak_mb']:.2f} MB")
    print(f"  Bytes/Entry: {memory_metrics['bytes_per_entry']:.0f}")
    print(f"  ✓ Memory <100MB: {'PASS' if passes['memory_target'] else 'FAIL'}")
    print()

    # Summary
    all_pass = all(passes.values())
    pass_count = sum(passes.values())
    total_count = len(passes)

    print("=" * 70)
    print(f"SUITE-004 RESULTS: {pass_count}/{total_count} targets met")
    print("=" * 70)

    for name, passed in passes.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    print()
    if all_pass:
        print(">>> GAP-003 VALIDATED: All cache performance targets met <<<")
        summary = "VALIDATED"
    else:
        print(">>> GAP-003 NOT VALIDATED: Some targets not met <<<")
        summary = "NOT_VALIDATED"

    return BenchmarkResult(
        suite="SUITE-004",
        run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        environment=environment,
        metrics=all_metrics,
        passes=passes,
        summary=summary,
    )


def format_benchmark_markdown(result: BenchmarkResult) -> str:
    """Format benchmark result as markdown for BENCHMARK_REGISTRY.md."""
    lines = [
        f"### {result.run_id} — {result.timestamp[:10]}",
        "",
        "**Environment:**",
        f"- Python: {result.environment['python']}",
        f"- Platform: {result.environment['platform']}",
        "",
        "**Latency Results:**",
        "| Metric | Value |",
        "|--------|-------|",
        f"| GET P50 | {result.metrics['get_p50_ms']:.4f}ms |",
        f"| GET P95 | {result.metrics['get_p95_ms']:.4f}ms |",
        f"| GET P99 | {result.metrics['get_p99_ms']:.4f}ms |",
        f"| SET P50 | {result.metrics['set_p50_ms']:.4f}ms |",
        "",
        "**Hit Rate & Throughput:**",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Hit Rate | {result.metrics['hit_rate_avg']:.2%} |",
        f"| Throughput | {result.metrics['throughput_avg_ops_s']:,.0f} ops/s |",
        f"| Memory | {result.metrics['memory_peak_mb']:.2f} MB |",
        "",
        "**Validation:**",
        "| Target | Status |",
        "|--------|--------|",
    ]

    for name, passed in result.passes.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"| {name} | {status} |")

    lines.extend(
        [
            "",
            f"**Summary:** {result.summary}",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_suite_004()

    print("\n" + "=" * 70)
    print("MARKDOWN OUTPUT FOR BENCHMARK_REGISTRY.md:")
    print("=" * 70)
    print(format_benchmark_markdown(result))
