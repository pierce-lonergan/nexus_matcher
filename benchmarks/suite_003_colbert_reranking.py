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

SUITE-003: ColBERT MaxSim Reranking Benchmark | GAP-001 Validation

Research Reference: README_RESEARCH_3.md, Lines 5-8
Research Targets:
- 60ms for top-100 reranking
- +10-20% accuracy improvement over bi-encoder
- MS MARCO MRR improvement from 36.0% to 39.7%
- Near cross-encoder quality at 10x lower latency

Benchmark Scenarios:
1. Rerank 10 candidates (typical batch)
2. Rerank 50 candidates (moderate)
3. Rerank 100 candidates (target from research)
4. Rerank 200 candidates (stress test)
5. Batch reranking efficiency
"""

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nexus_matcher.domain.ports.retrieval import RerankCandidate
from nexus_matcher.infrastructure.adapters.rerankers.colbert import (
    ColBERTMaxSimReranker,
    MockColBERTReranker,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    scenario: str
    num_candidates: int
    num_iterations: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    throughput_candidates_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "num_candidates": self.num_candidates,
            "num_iterations": self.num_iterations,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "throughput_candidates_per_sec": self.throughput_candidates_per_sec,
        }


@dataclass
class SuiteResult:
    """Result of the benchmark suite."""

    suite_id: str = "SUITE-003"
    suite_name: str = "ColBERT MaxSim Reranking Performance"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: dict = field(default_factory=dict)
    reranker_type: str = "mock"
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "reranker_type": self.reranker_type,
            "results": [r.to_dict() for r in self.results],
        }


def generate_candidates(n: int) -> list[RerankCandidate]:
    """Generate synthetic candidates for reranking."""
    candidates = []
    for i in range(n):
        text = (
            f"Dictionary entry {i}: this field contains customer account "
            f"information including transaction details and balance data "
            f"for account type {i % 10}"
        )
        candidates.append(
            RerankCandidate(
                id=f"candidate_{i}",
                text=text,
                initial_score=1.0 - (i / n),  # Simulate initial ranking
                metadata={"source": f"batch_{i // 10}"},
            )
        )
    return candidates


def benchmark_scenario(
    reranker: MockColBERTReranker,
    num_candidates: int,
    num_iterations: int = 50,
    warmup: int = 5,
) -> BenchmarkResult:
    """Run benchmark for a specific scenario."""
    candidates = generate_candidates(num_candidates)
    query = "find customer email address field"

    # Warmup
    for _ in range(warmup):
        reranker.rerank(query, candidates)

    # Measurement
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = reranker.rerank(query, candidates)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.is_failure:
            raise RuntimeError(f"Reranking failed: {result.error}")

        latencies.append(elapsed_ms)

    # Calculate statistics
    latencies.sort()
    p50_idx = int(len(latencies) * 0.50)
    p95_idx = int(len(latencies) * 0.95)
    p99_idx = int(len(latencies) * 0.99)

    avg_latency = statistics.mean(latencies)
    total_candidates = num_candidates * num_iterations
    total_time_sec = sum(latencies) / 1000
    throughput = total_candidates / total_time_sec if total_time_sec > 0 else 0

    return BenchmarkResult(
        scenario=f"{num_candidates}_candidates",
        num_candidates=num_candidates,
        num_iterations=num_iterations,
        p50_latency_ms=latencies[p50_idx],
        p95_latency_ms=latencies[min(p95_idx, len(latencies) - 1)],
        p99_latency_ms=latencies[min(p99_idx, len(latencies) - 1)],
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        avg_latency_ms=avg_latency,
        throughput_candidates_per_sec=throughput,
    )


def get_environment_info() -> dict:
    """Collect environment information."""
    import platform

    # Check RAGatouille availability
    ragatouille_available = False
    try:
        from ragatouille import RAGPretrainedModel

        ragatouille_available = True
    except ImportError:
        pass

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ragatouille_available": ragatouille_available,
    }


def run_benchmark_suite(use_real_model: bool = False) -> SuiteResult:
    """Run the complete benchmark suite."""
    print(f"\n{'=' * 60}")
    print("SUITE-003: ColBERT MaxSim Reranking Benchmark")
    print(f"{'=' * 60}\n")

    suite_result = SuiteResult()
    suite_result.environment = get_environment_info()

    # Select reranker
    if use_real_model and suite_result.environment["ragatouille_available"]:
        print("Using REAL ColBERT model (RAGatouille)")
        reranker = ColBERTMaxSimReranker()
        suite_result.reranker_type = "colbert_maxsim"
    else:
        print("Using MOCK ColBERT reranker")
        # Use minimal simulated latency for benchmarking mock
        reranker = MockColBERTReranker(simulated_latency_ms=0.01)
        suite_result.reranker_type = "mock"

    print("Environment:")
    print(f"  RAGatouille Available: {suite_result.environment['ragatouille_available']}")
    print()

    # Scenarios
    scenarios = [10, 50, 100, 200]

    for num_candidates in scenarios:
        print(f"[Scenario] {num_candidates} candidates")
        result = benchmark_scenario(reranker, num_candidates, num_iterations=50)
        suite_result.results.append(result)

        print(f"  P50: {result.p50_latency_ms:.2f}ms")
        print(f"  P95: {result.p95_latency_ms:.2f}ms")
        print(f"  P99: {result.p99_latency_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_candidates_per_sec:,.0f} candidates/sec")
        print()

    # Summary
    print(f"{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Check research target: 60ms for 100 candidates
    result_100 = next((r for r in suite_result.results if r.num_candidates == 100), None)

    targets_met = 0
    targets_total = 2

    if result_100:
        if result_100.p95_latency_ms <= 60:
            print(f"✓ 100 candidates P95: {result_100.p95_latency_ms:.2f}ms (target: ≤60ms)")
            targets_met += 1
        else:
            print(f"○ 100 candidates P95: {result_100.p95_latency_ms:.2f}ms (target: ≤60ms)")

        if result_100.throughput_candidates_per_sec >= 1000:
            print(
                f"✓ Throughput: {result_100.throughput_candidates_per_sec:,.0f}/s (target: ≥1000)"
            )
            targets_met += 1
        else:
            print(
                f"○ Throughput: {result_100.throughput_candidates_per_sec:,.0f}/s (target: ≥1000)"
            )

    print(f"\nTargets Met: {targets_met}/{targets_total}")

    # Note about mock vs real
    if suite_result.reranker_type == "mock":
        print("\n⚠️  Note: Results from MockColBERTReranker.")
        print("   Full validation requires RAGatouille + answerai-colbert-small-v1 model.")
        print("   Install with: pip install ragatouille")

    return suite_result


def main():
    """Run benchmark and save results."""
    use_real = os.environ.get("USE_REAL_MODEL", "").lower() in ("1", "true", "yes")

    result = run_benchmark_suite(use_real_model=use_real)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"suite_003_run_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
