#!/usr/bin/env python3
"""
SUITE-004c: Context Enrichment Benchmark
Gap: GAP-006 - Enhanced Context Injection

Research Target:
- +10-20% accuracy improvement on nested schemas
- Context injection is "non-negotiable" for nested schemas (README_RESEARCH_1.md:174)
- Must include full hierarchy: "user entity, addresses array, street_name field"

Metrics Measured:
1. Enrichment throughput (fields/second)
2. Context depth coverage (verify 3+ levels captured)
3. Output quality (hierarchy tokens present)
4. Latency per field
"""

import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

from nexus_matcher.domain.services.context_enricher import ContextEnricher
from nexus_matcher.domain.models.entities import SchemaField
from nexus_matcher.shared.types.base import DataType


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    suite: str = "SUITE-004c"
    gap: str = "GAP-006"
    target: str = "Enhanced Context Injection"
    timestamp: str = ""
    
    # Throughput metrics
    fields_processed: int = 0
    total_time_ms: float = 0.0
    throughput_fields_per_sec: float = 0.0
    
    # Latency metrics
    latency_per_field_p50_us: float = 0.0
    latency_per_field_p95_us: float = 0.0
    latency_per_field_p99_us: float = 0.0
    
    # Quality metrics
    depth_1_coverage_pct: float = 0.0  # Root level
    depth_2_coverage_pct: float = 0.0  # 1 parent
    depth_3_coverage_pct: float = 0.0  # 2 parents (critical)
    depth_4_plus_coverage_pct: float = 0.0  # 3+ parents
    
    # Content validation
    avg_enriched_length: float = 0.0
    avg_hierarchy_tokens: float = 0.0
    humanization_rate_pct: float = 0.0
    
    # Targets
    targets_met: int = 0
    targets_total: int = 4


def generate_test_fields() -> list[SchemaField]:
    """Generate realistic nested schema fields for testing."""
    fields = []
    
    # Depth 1 - Root level fields
    for name in ["customer_id", "account_number", "created_at", "status", "email"]:
        fields.append(SchemaField(
            name=name,
            data_type=DataType.STRING,
            full_path=name,
        ))
    
    # Depth 2 - One parent
    for parent, children in [
        ("customer", ["first_name", "last_name", "phone", "email_address"]),
        ("order", ["order_id", "total_amount", "currency", "status"]),
        ("product", ["sku", "name", "description", "price"]),
    ]:
        for child in children:
            fields.append(SchemaField(
                name=child,
                data_type=DataType.STRING,
                full_path=f"{parent}.{child}",
                parent_path=parent,
            ))
    
    # Depth 3 - Two parents (critical for research compliance)
    nested_structures = [
        ("user", "addresses", ["street_name", "city", "postal_code", "country"]),
        ("order", "line_items", ["product_id", "quantity", "unit_price"]),
        ("customer", "preferences", ["email_opt_in", "sms_opt_in", "language"]),
        ("account", "settings", ["timezone", "currency", "notifications"]),
    ]
    for parent, child, grandchildren in nested_structures:
        for gc in grandchildren:
            fields.append(SchemaField(
                name=gc,
                data_type=DataType.STRING,
                full_path=f"{parent}.{child}.{gc}",
                parent_path=f"{parent}.{child}",
            ))
    
    # Depth 4+ - Three or more parents
    deep_fields = [
        ("company", "departments", "employees", "contact", "email"),
        ("organization", "teams", "members", "profile", "avatar_url"),
        ("system", "modules", "components", "config", "value"),
        ("order", "shipments", "packages", "items", "barcode"),
    ]
    for *parents, field_name in deep_fields:
        path = ".".join(parents + [field_name])
        parent_path = ".".join(parents)
        fields.append(SchemaField(
            name=field_name,
            data_type=DataType.STRING,
            full_path=path,
            parent_path=parent_path,
        ))
    
    # Array fields
    array_fields = [
        ("product", "tags", True, DataType.STRING),
        ("order", "items", True, DataType.RECORD),
        ("customer", "addresses", True, DataType.RECORD),
    ]
    for parent, name, is_array, item_type in array_fields:
        fields.append(SchemaField(
            name=name,
            data_type=DataType.ARRAY,
            full_path=f"{parent}.{name}",
            parent_path=parent,
            is_array=is_array,
            array_item_type=item_type,
        ))
    
    # Fields with descriptions
    described_fields = [
        ("transaction_id", "Unique identifier for the financial transaction"),
        ("balance_amount", "Current account balance in local currency"),
        ("expiration_date", "Date when the item or subscription expires"),
    ]
    for name, desc in described_fields:
        fields.append(SchemaField(
            name=name,
            data_type=DataType.STRING,
            full_path=name,
            description=desc,
        ))
    
    return fields


def count_hierarchy_tokens(text: str, field: SchemaField) -> int:
    """Count how many hierarchy elements are present in enriched text."""
    if not field.is_nested:
        return 0
    
    path_parts = field.full_path.split(".")
    parent_parts = path_parts[:-1]
    
    count = 0
    text_lower = text.lower()
    
    for part in parent_parts:
        # Check for the part or its humanized version
        humanized = part.replace("_", " ").lower()
        if part.lower() in text_lower or humanized in text_lower:
            count += 1
    
    return count


def check_humanization(text: str, field: SchemaField) -> bool:
    """Check if underscores were humanized."""
    if "_" not in field.name:
        return True
    
    # If the original name with underscores is NOT in the output, it was humanized
    return field.name not in text


def run_benchmark() -> BenchmarkResult:
    """Run the context enrichment benchmark."""
    result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
    )
    
    # Setup
    enricher = ContextEnricher()
    fields = generate_test_fields()
    result.fields_processed = len(fields)
    
    print(f"SUITE-004c: Context Enrichment Benchmark")
    print(f"=" * 60)
    print(f"Generated {len(fields)} test fields")
    
    # Warm-up
    print("\nWarm-up (3 runs)...")
    for _ in range(3):
        for field in fields:
            _ = enricher.enrich(field)
    
    # Measurement runs
    print("Measurement (5 runs)...")
    latencies = []
    all_enriched = []
    
    for run in range(5):
        run_latencies = []
        enriched_texts = []
        
        start_time = time.perf_counter()
        for field in fields:
            field_start = time.perf_counter()
            enriched = enricher.enrich(field)
            field_end = time.perf_counter()
            
            run_latencies.append((field_end - field_start) * 1_000_000)  # microseconds
            enriched_texts.append((field, enriched))
        end_time = time.perf_counter()
        
        latencies.extend(run_latencies)
        if run == 4:  # Last run - save for analysis
            all_enriched = enriched_texts
        
        run_time_ms = (end_time - start_time) * 1000
        print(f"  Run {run+1}: {run_time_ms:.2f}ms")
    
    # Calculate metrics
    result.total_time_ms = sum(latencies) / 1000  # Convert back to ms
    result.throughput_fields_per_sec = (len(fields) * 5) / (result.total_time_ms / 1000)
    
    sorted_latencies = sorted(latencies)
    result.latency_per_field_p50_us = sorted_latencies[int(len(sorted_latencies) * 0.50)]
    result.latency_per_field_p95_us = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    result.latency_per_field_p99_us = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    # Depth coverage analysis
    depth_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    depth_success = {1: 0, 2: 0, 3: 0, 4: 0}
    
    hierarchy_tokens = []
    enriched_lengths = []
    humanization_success = 0
    humanization_checked = 0
    
    for field, enriched in all_enriched:
        enriched_lengths.append(len(enriched))
        depth = field.depth + 1  # depth 0 = level 1
        depth_key = min(depth, 4)
        depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
        
        # For nested fields, check if hierarchy context is present
        if field.is_nested:
            tokens = count_hierarchy_tokens(enriched, field)
            hierarchy_tokens.append(tokens)
            expected_tokens = len(field.full_path.split(".")) - 1
            if tokens >= expected_tokens * 0.8:  # 80% of parents mentioned
                depth_success[depth_key] = depth_success.get(depth_key, 0) + 1
        else:
            depth_success[1] = depth_success.get(1, 0) + 1  # Root fields always pass
        
        # Check humanization
        if "_" in field.name:
            humanization_checked += 1
            if check_humanization(enriched, field):
                humanization_success += 1
    
    # Calculate coverage percentages
    result.depth_1_coverage_pct = (depth_success.get(1, 0) / max(depth_counts.get(1, 1), 1)) * 100
    result.depth_2_coverage_pct = (depth_success.get(2, 0) / max(depth_counts.get(2, 1), 1)) * 100
    result.depth_3_coverage_pct = (depth_success.get(3, 0) / max(depth_counts.get(3, 1), 1)) * 100
    result.depth_4_plus_coverage_pct = (depth_success.get(4, 0) / max(depth_counts.get(4, 1), 1)) * 100
    
    result.avg_enriched_length = statistics.mean(enriched_lengths)
    result.avg_hierarchy_tokens = statistics.mean(hierarchy_tokens) if hierarchy_tokens else 0
    result.humanization_rate_pct = (humanization_success / max(humanization_checked, 1)) * 100
    
    # Validate targets
    targets = [
        ("Depth 3+ Coverage ≥ 80%", result.depth_3_coverage_pct >= 80),
        ("Hierarchy Tokens ≥ 1.5 avg", result.avg_hierarchy_tokens >= 1.5),
        ("Humanization Rate ≥ 95%", result.humanization_rate_pct >= 95),
        ("Throughput ≥ 50K fields/s", result.throughput_fields_per_sec >= 50_000),
    ]
    
    result.targets_met = sum(1 for _, passed in targets if passed)
    result.targets_total = len(targets)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    print(f"\nThroughput:")
    print(f"  Fields processed: {result.fields_processed}")
    print(f"  Throughput: {result.throughput_fields_per_sec:,.0f} fields/s")
    
    print(f"\nLatency (per field):")
    print(f"  P50: {result.latency_per_field_p50_us:.2f} µs")
    print(f"  P95: {result.latency_per_field_p95_us:.2f} µs")
    print(f"  P99: {result.latency_per_field_p99_us:.2f} µs")
    
    print(f"\nDepth Coverage:")
    print(f"  Depth 1 (root):    {result.depth_1_coverage_pct:.1f}%")
    print(f"  Depth 2 (1 parent): {result.depth_2_coverage_pct:.1f}%")
    print(f"  Depth 3 (2 parents): {result.depth_3_coverage_pct:.1f}% [CRITICAL]")
    print(f"  Depth 4+ (3+ parents): {result.depth_4_plus_coverage_pct:.1f}%")
    
    print(f"\nQuality Metrics:")
    print(f"  Avg enriched length: {result.avg_enriched_length:.1f} chars")
    print(f"  Avg hierarchy tokens: {result.avg_hierarchy_tokens:.2f}")
    print(f"  Humanization rate: {result.humanization_rate_pct:.1f}%")
    
    print(f"\n{'='*60}")
    print("TARGET VALIDATION")
    print(f"{'='*60}")
    
    for target_name, passed in targets:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {target_name}")
    
    print(f"\nOverall: {result.targets_met}/{result.targets_total} targets met")
    
    if result.targets_met >= result.targets_total:
        print(f"\n{'='*60}")
        print("GAP-006 VALIDATED ✓")
        print(f"{'='*60}")
    
    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"suite_004c_context_enrichment_{run_id}.json"
    
    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return result


def show_enrichment_examples():
    """Show example enrichment outputs."""
    enricher = ContextEnricher()
    
    examples = [
        SchemaField(
            name="email_address",
            data_type=DataType.STRING,
            full_path="email_address",
        ),
        SchemaField(
            name="street_name",
            data_type=DataType.STRING,
            full_path="user.addresses.street_name",
            parent_path="user.addresses",
        ),
        SchemaField(
            name="amount",
            data_type=DataType.DECIMAL,
            full_path="order.line_items.price.amount",
            parent_path="order.line_items.price",
        ),
        SchemaField(
            name="tags",
            data_type=DataType.ARRAY,
            full_path="product.tags",
            parent_path="product",
            is_array=True,
            array_item_type=DataType.STRING,
        ),
    ]
    
    print("\nEnrichment Examples:")
    print("=" * 60)
    
    for field in examples:
        basic = field.to_searchable_text()
        enriched = enricher.enrich(field)
        
        print(f"\nField: {field.full_path}")
        print(f"  Basic:    '{basic}'")
        print(f"  Enriched: '{enriched}'")


if __name__ == "__main__":
    result = run_benchmark()
    show_enrichment_examples()
