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

SUITE-005: Incremental Update Benchmark | GAP-005 Validation

Research Reference: README_RESEARCH_3.md, Lines 33-35
Research Targets:
- 90-99% computation savings for typical updates
- 100-1000x speedup for <1% corpus changes
- 10-50x speedup for 1-10% corpus changes
- BLAKE3 hashing (8-10x faster than SHA-256)

Benchmark Scenarios:
1. Initial load (all entries are new)
2. 0.1% change rate (100/100,000)
3. 1% change rate (1,000/100,000)
4. 5% change rate (5,000/100,000)
5. 10% change rate (10,000/100,000)
6. Hash computation throughput
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nexus_matcher.infrastructure.adapters.incremental import (
    IncrementalUpdateManager,
    compute_hash,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    scenario: str
    corpus_size: int
    change_count: int
    change_rate_pct: float
    detection_time_ms: float
    savings_pct: float
    throughput_entries_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "corpus_size": self.corpus_size,
            "change_count": self.change_count,
            "change_rate_pct": self.change_rate_pct,
            "detection_time_ms": self.detection_time_ms,
            "savings_pct": self.savings_pct,
            "throughput_entries_per_sec": self.throughput_entries_per_sec,
        }


@dataclass
class SuiteResult:
    """Result of the benchmark suite."""

    suite_id: str = "SUITE-005"
    suite_name: str = "Incremental Update Performance"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: dict = field(default_factory=dict)
    results: list[BenchmarkResult] = field(default_factory=list)
    hash_benchmark: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "results": [r.to_dict() for r in self.results],
            "hash_benchmark": self.hash_benchmark,
        }


def generate_dictionary_entries(n: int) -> dict[str, str]:
    """Generate synthetic dictionary entries."""
    entries = {}
    for i in range(n):
        # Simulate realistic field descriptions
        entry_id = f"DD_{i:06d}"
        content = (
            f"Field {i}: customer account information containing "
            f"transaction details for account type {i % 10} with "
            f"data classification level {i % 5}"
        )
        entries[entry_id] = content
    return entries


def modify_entries(entries: dict[str, str], count: int) -> dict[str, str]:
    """Modify a specified number of entries."""
    modified = dict(entries)
    keys = list(modified.keys())[:count]
    for key in keys:
        modified[key] = modified[key] + " [MODIFIED]"
    return modified


def benchmark_hash_throughput(iterations: int = 10000) -> dict:
    """Benchmark hash computation throughput."""
    test_content = "Sample content for hash computation benchmark " * 10  # ~500 chars

    # Warmup
    for _ in range(100):
        compute_hash(test_content)

    # Measurement
    start = time.perf_counter()
    for _ in range(iterations):
        compute_hash(test_content)
    elapsed = time.perf_counter() - start

    throughput = iterations / elapsed
    avg_time_us = (elapsed / iterations) * 1_000_000

    return {
        "iterations": iterations,
        "total_time_ms": elapsed * 1000,
        "avg_time_us": avg_time_us,
        "throughput_per_sec": throughput,
        "content_length": len(test_content),
    }


def benchmark_scenario(
    manager: IncrementalUpdateManager,
    entries: dict[str, str],
    scenario_name: str,
    change_count: int,
) -> BenchmarkResult:
    """Run a single benchmark scenario."""
    corpus_size = len(entries)

    # Time the change detection
    start = time.perf_counter()
    result = manager.compute_changes(entries)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Calculate throughput
    throughput = corpus_size / (elapsed_ms / 1000) if elapsed_ms > 0 else float("inf")

    return BenchmarkResult(
        scenario=scenario_name,
        corpus_size=corpus_size,
        change_count=change_count,
        change_rate_pct=(change_count / corpus_size) * 100 if corpus_size > 0 else 0,
        detection_time_ms=elapsed_ms,
        savings_pct=result.savings_pct,
        throughput_entries_per_sec=throughput,
    )


def get_environment_info() -> dict:
    """Collect environment information."""
    import platform

    # Check BLAKE3 availability
    blake3_available = False
    try:
        import blake3

        blake3_available = True
    except ImportError:
        pass

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "blake3_available": blake3_available,
        "hash_algorithm": "BLAKE3" if blake3_available else "SHA-256",
    }


def run_benchmark_suite(corpus_size: int = 100_000) -> SuiteResult:
    """Run the complete benchmark suite."""
    print(f"\n{'=' * 60}")
    print("SUITE-005: Incremental Update Benchmark")
    print(f"{'=' * 60}\n")

    suite_result = SuiteResult()
    suite_result.environment = get_environment_info()

    print("Environment:")
    print(f"  Hash Algorithm: {suite_result.environment['hash_algorithm']}")
    print(f"  Corpus Size: {corpus_size:,} entries")
    print()

    # Generate base corpus
    print("Generating corpus...")
    base_entries = generate_dictionary_entries(corpus_size)
    print(f"  Generated {len(base_entries):,} entries")

    # Initialize manager with baseline
    manager = IncrementalUpdateManager()

    # Scenario 1: Initial load (all new)
    print("\n[Scenario 1] Initial Load (100% new)")
    result = benchmark_scenario(manager, base_entries, "initial_load", corpus_size)
    suite_result.results.append(result)
    print(f"  Detection Time: {result.detection_time_ms:.2f}ms")
    print(f"  Throughput: {result.throughput_entries_per_sec:,.0f} entries/sec")

    # Apply initial load
    initial_result = manager.compute_changes(base_entries)
    manager.apply_changes(initial_result)

    # Scenario 2: 0.1% change rate
    change_count = max(1, corpus_size // 1000)
    modified = modify_entries(base_entries, change_count)
    print(f"\n[Scenario 2] 0.1% Change Rate ({change_count} entries)")
    result = benchmark_scenario(manager, modified, "0.1%_change", change_count)
    suite_result.results.append(result)
    print(f"  Detection Time: {result.detection_time_ms:.2f}ms")
    print(f"  Savings: {result.savings_pct:.1f}%")
    print(f"  Throughput: {result.throughput_entries_per_sec:,.0f} entries/sec")

    # Scenario 3: 1% change rate
    change_count = corpus_size // 100
    modified = modify_entries(base_entries, change_count)
    print(f"\n[Scenario 3] 1% Change Rate ({change_count} entries)")
    result = benchmark_scenario(manager, modified, "1%_change", change_count)
    suite_result.results.append(result)
    print(f"  Detection Time: {result.detection_time_ms:.2f}ms")
    print(f"  Savings: {result.savings_pct:.1f}%")
    print(f"  Throughput: {result.throughput_entries_per_sec:,.0f} entries/sec")

    # Scenario 4: 5% change rate
    change_count = corpus_size // 20
    modified = modify_entries(base_entries, change_count)
    print(f"\n[Scenario 4] 5% Change Rate ({change_count} entries)")
    result = benchmark_scenario(manager, modified, "5%_change", change_count)
    suite_result.results.append(result)
    print(f"  Detection Time: {result.detection_time_ms:.2f}ms")
    print(f"  Savings: {result.savings_pct:.1f}%")
    print(f"  Throughput: {result.throughput_entries_per_sec:,.0f} entries/sec")

    # Scenario 5: 10% change rate
    change_count = corpus_size // 10
    modified = modify_entries(base_entries, change_count)
    print(f"\n[Scenario 5] 10% Change Rate ({change_count} entries)")
    result = benchmark_scenario(manager, modified, "10%_change", change_count)
    suite_result.results.append(result)
    print(f"  Detection Time: {result.detection_time_ms:.2f}ms")
    print(f"  Savings: {result.savings_pct:.1f}%")
    print(f"  Throughput: {result.throughput_entries_per_sec:,.0f} entries/sec")

    # Hash throughput benchmark
    print("\n[Hash Throughput Benchmark]")
    hash_result = benchmark_hash_throughput(10000)
    suite_result.hash_benchmark = hash_result
    print(f"  Avg Hash Time: {hash_result['avg_time_us']:.2f}µs")
    print(f"  Throughput: {hash_result['throughput_per_sec']:,.0f} hashes/sec")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Check research targets
    targets_met = 0
    targets_total = 3

    # Target 1: 99% savings for 0.1% changes
    scenario_01 = next((r for r in suite_result.results if r.scenario == "0.1%_change"), None)
    if scenario_01 and scenario_01.savings_pct >= 99.0:
        print(f"✓ 0.1% change: {scenario_01.savings_pct:.1f}% savings (target: ≥99%)")
        targets_met += 1
    elif scenario_01:
        print(f"○ 0.1% change: {scenario_01.savings_pct:.1f}% savings (target: ≥99%)")

    # Target 2: 95% savings for 5% changes
    scenario_5 = next((r for r in suite_result.results if r.scenario == "5%_change"), None)
    if scenario_5 and scenario_5.savings_pct >= 95.0:
        print(f"✓ 5% change: {scenario_5.savings_pct:.1f}% savings (target: ≥95%)")
        targets_met += 1
    elif scenario_5:
        print(f"○ 5% change: {scenario_5.savings_pct:.1f}% savings (target: ≥95%)")

    # Target 3: 90% savings for 10% changes
    scenario_10 = next((r for r in suite_result.results if r.scenario == "10%_change"), None)
    if scenario_10 and scenario_10.savings_pct >= 90.0:
        print(f"✓ 10% change: {scenario_10.savings_pct:.1f}% savings (target: ≥90%)")
        targets_met += 1
    elif scenario_10:
        print(f"○ 10% change: {scenario_10.savings_pct:.1f}% savings (target: ≥90%)")

    print(f"\nTargets Met: {targets_met}/{targets_total}")
    print(f"Status: {'PASS' if targets_met == targets_total else 'PARTIAL'}")

    return suite_result


def main():
    """Run benchmark and save results."""
    # Run with smaller corpus for quick validation, larger for full benchmark
    corpus_size = int(os.environ.get("CORPUS_SIZE", "100000"))

    result = run_benchmark_suite(corpus_size)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"suite_005_run_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
