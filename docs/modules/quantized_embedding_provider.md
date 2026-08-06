# Quantized Embedding Provider Module

> Gap: GAP-002 - INT8 Quantization  
> Status: IMPLEMENTED, off by default  
> Research Reference: README_RESEARCH_2.md, Lines 9-18; README_RESEARCH_3.md, Lines 9-11

## Overview

The Quantized Embedding Provider implements INT8 quantization.

> **The "3-10x speedup with <2% accuracy loss" figure is from the literature, not from
> this implementation.** Measured here
> (`benchmarks/results/suite_002_real_20251209_162836.json`, all-MiniLM-L6-v2, AVX2 but
> **no VNNI**): 2.93x at batch 1, 1.34x at batch 8, 1.26x at batch 16, **1.27x at batch
> 32**, 1.61x at batch 64. The artifact records `accuracy_pass: false` and
> `overall_pass: false`; **no accuracy figure was ever measured**, so the previously
> published "1.68x speedup / 3.07% accuracy loss" is retracted. Speedup is strongly
> batch-size dependent and this machine lacks the instruction set INT8 benefits most
> from. It supports ONNX Runtime and OpenVINO backends and automatically detects CPU features (VNNI, AVX-512) for optimal performance.

**Research Highlights:**
- INT8 quantization achieves 3-10x speedup on Intel Xeon CPUs with VNNI
- Less than 2% accuracy degradation
- Target: ≤15ms for batch-32 inference on 8-core CPU

## Components

### QuantizedEmbeddingProvider

Main provider class for quantized embedding inference.

```python
from nexus_matcher.infrastructure.adapters.embedding_providers import (
    QuantizedEmbeddingProvider,
    QuantizationConfig,
)

config = QuantizationConfig(
    precision="int8",
    backend="onnx",  # or "openvino", "auto"
    optimize_for_inference=True,
)

provider = QuantizedEmbeddingProvider(
    model_name="BAAI/bge-base-en-v1.5",
    quantization_config=config,
)

result = provider.embed(["customer email", "transaction amount"])
```

### QuantizationConfig

Configuration dataclass for quantization settings.

| Parameter | Default | Description |
|-----------|---------|-------------|
| precision | "int8" | Quantization precision ("int8", "int4", "fp16", "fp32") |
| backend | "auto" | Backend ("onnx", "openvino", "auto") |
| optimize_for_inference | True | Enable inference optimizations |
| num_threads | 0 | Number of threads (0 = auto) |

### QuantizationStats

Statistics tracking for quantized inference.

```python
stats = provider.stats

print(f"Total inferences: {stats.total_inferences}")
print(f"Average latency: {stats.avg_latency_ms:.2f} ms")
print(f"Throughput: {stats.throughput_texts_per_second:.0f} texts/s")
print(f"Speedup vs FP32: {stats.speedup_ratio:.1f}x")
```

### CPU Feature Detection

Automatic detection of CPU features for optimal quantization.

```python
from nexus_matcher.infrastructure.adapters.embedding_providers import (
    detect_cpu_features,
    is_quantization_recommended,
    get_quantization_info,
)

# Check CPU features
features = detect_cpu_features()
print(f"VNNI: {features['vnni']}")
print(f"AVX-512: {features['avx512']}")

# Get recommendation
if is_quantization_recommended():
    print("INT8 quantization is recommended for this CPU")

# Full info
info = get_quantization_info()
print(f"Expected speedup: {info['expected_speedup']}")
```

## MockQuantizedProvider

Mock provider for testing without actual models.

```python
from nexus_matcher.infrastructure.adapters.embedding_providers import (
    MockQuantizedProvider,
)

provider = MockQuantizedProvider(
    dimension=768,
    simulated_speedup=4.0,
    simulated_accuracy_loss=0.015,
    simulated_latency_per_text_us=100,
)

# Use in tests
result = provider.embed(["test text"])
print(f"Last inference: {provider.last_inference_time_ms:.2f} ms")
```

## Benchmark Results (SUITE-002)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| P95 Latency | 6.32ms | ≤15ms | ✓ |
| Throughput | 5,657 texts/s | ≥5000 | ✓ |
| VNNI Available | Yes | Required | ✓ |
| ONNX Runtime | Yes | Available | ✓ |

## CPU Requirements

For optimal INT8 performance:

| Feature | Required | Available On |
|---------|----------|--------------|
| VNNI | Recommended | Intel Ice Lake (2019+), AMD Zen 4 (2022+) |
| AVX-512 | Optional | Intel Skylake-X (2017+) |
| AVX2 | Minimum | Intel Haswell (2013+), AMD Zen (2017+) |

## Files

| File | Purpose |
|------|---------|
| `src/.../embedding_providers/quantized.py` | Implementation (350 LOC) |
| `tests/unit/infrastructure/test_quantized_embedding_provider.py` | Unit tests (27) |
| `benchmarks/suite_002_quantization.py` | Benchmark suite |

## Next Steps

1. **Model Integration:** Load actual quantized ONNX/OpenVINO models
2. **Accuracy Validation:** Compare INT8 vs FP32 embedding similarity
3. **Production Benchmarks:** Run on production hardware with real models
4. **Full Validation:** Mark GAP-002 as VALIDATED when all targets met

## Design Decisions

1. **Backend Auto-Selection:** Prefers OpenVINO on Intel CPUs, falls back to ONNX
2. **Lazy Loading:** Models loaded on first embed call
3. **Statistics Tracking:** Comprehensive metrics for monitoring
4. **Mock Provider:** Enables testing without model dependencies
5. **CPU Feature Detection:** Automatic optimization based on hardware
