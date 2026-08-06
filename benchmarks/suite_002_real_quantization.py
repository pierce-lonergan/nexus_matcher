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

SUITE-002 (Real) — INT8 Quantization Benchmark with Actual Models
═══════════════════════════════════════════════════════════════════

Research Reference: README_RESEARCH_2.md, Lines 9-18; README_RESEARCH_3.md, Lines 9-11
Gap: GAP-002 — INT8 Quantization

Target Metrics:
- 3-10x speedup (VNNI) or 2-4x (AVX2 only)
- <2% accuracy loss
- ≤15ms batch-32 latency

This benchmark compares:
1. FP32 baseline (sentence-transformers native)
2. ONNX FP32 (ONNX Runtime)
3. INT8 quantized (ONNX Runtime with dynamic quantization)

Prerequisites:
    pip install sentence-transformers onnxruntime onnx
"""

import json
import os
import platform
import sys
import tempfile
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
    import onnxruntime as ort
except ImportError:
    DEPENDENCIES_OK = False
    MISSING.append("onnxruntime")

try:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError:
    DEPENDENCIES_OK = False
    MISSING.append("onnx")

try:
    from transformers import AutoTokenizer
except ImportError:
    DEPENDENCIES_OK = False
    MISSING.append("transformers")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model to benchmark (small model for faster testing)
MODEL_NAME = os.environ.get("BENCHMARK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Test configuration
BATCH_SIZES = [1, 8, 16, 32, 64]
WARMUP_ITERATIONS = 3
MEASUREMENT_ITERATIONS = 10

# Sample texts for benchmarking
SAMPLE_TEXTS = [
    "customer_email_address",
    "transaction_amount_usd",
    "account_balance_current",
    "payment_date_timestamp",
    "user_first_name",
    "product_category_code",
    "order_status_description",
    "shipping_address_line_1",
    "credit_card_last_four",
    "phone_number_primary",
    "date_of_birth",
    "annual_income_range",
    "employment_status_code",
    "loan_amount_requested",
    "interest_rate_percentage",
    "account_open_date",
    "last_login_timestamp",
    "security_question_answer",
    "preferred_language_code",
    "marketing_consent_flag",
    "risk_score_calculated",
    "fraud_alert_indicator",
    "kyc_verification_status",
    "document_type_identifier",
    "branch_location_code",
    "relationship_manager_id",
    "portfolio_value_total",
    "dividend_payment_amount",
    "tax_identification_number",
    "beneficiary_name_primary",
    "wire_transfer_reference",
    "check_deposit_amount",
] * 2  # 64 texts total


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    backend: str
    precision: str
    batch_size: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_texts_per_sec: float
    latencies: list


@dataclass
class ComparisonResult:
    """Comparison between FP32 and INT8."""

    batch_size: int
    fp32_latency_ms: float
    int8_latency_ms: float
    speedup_ratio: float
    meets_target: bool
    target_speedup: float


# =============================================================================
# CPU FEATURE DETECTION
# =============================================================================


def detect_cpu_features() -> dict:
    """Detect CPU features for quantization."""
    features = {
        "vnni": False,
        "avx2": False,
        "avx512": False,
        "cpu_name": platform.processor() or "Unknown",
        "platform": platform.system(),
    }

    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read().lower()
                features["avx2"] = "avx2" in cpuinfo
                features["avx512"] = "avx512" in cpuinfo
                features["vnni"] = "avx512_vnni" in cpuinfo or "avx_vnni" in cpuinfo

        elif platform.system() == "Windows":
            # Try py-cpuinfo if available
            try:
                import cpuinfo

                info = cpuinfo.get_cpu_info()
                flags = [f.lower() for f in info.get("flags", [])]
                features["avx2"] = "avx2" in flags
                features["avx512"] = any("avx512" in f for f in flags)
                features["vnni"] = any("vnni" in f for f in flags)
                features["cpu_name"] = info.get("brand_raw", features["cpu_name"])
            except ImportError:
                # Assume modern CPU has AVX2
                features["avx2"] = True
    except Exception:
        pass

    return features


# =============================================================================
# MODEL EXPORT AND QUANTIZATION
# =============================================================================


def export_model_to_onnx(model: SentenceTransformer, output_path: Path) -> Path:
    """Export sentence-transformers model to ONNX format."""
    print(f"  Exporting model to ONNX: {output_path}")

    # Get the transformer model
    transformer = model[0].auto_model
    tokenizer = model.tokenizer

    # Sample input for tracing
    sample_text = "sample input for export"
    inputs = tokenizer(
        sample_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Export to ONNX
    import torch

    torch.onnx.export(
        transformer,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"  ✓ ONNX model exported: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def quantize_model_int8(input_path: Path, output_path: Path) -> Path:
    """Apply INT8 dynamic quantization to ONNX model."""
    print(f"  Quantizing model to INT8: {output_path}")

    try:
        # Try newer API first (onnxruntime >= 1.16)
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
    except TypeError:
        # Fall back to older API
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            optimize_model=True,
        )

    original_size = input_path.stat().st_size
    quantized_size = output_path.stat().st_size
    compression = (1 - quantized_size / original_size) * 100

    print(
        f"  ✓ INT8 model created: {quantized_size / 1024 / 1024:.1f} MB ({compression:.1f}% smaller)"
    )
    return output_path


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================


def benchmark_sentence_transformers(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark native sentence-transformers FP32."""
    latencies = []

    for _ in range(iterations):
        batch = texts[:batch_size]

        start = time.perf_counter()
        _ = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    throughput = (batch_size / np.mean(latencies_arr)) * 1000

    return BenchmarkResult(
        backend="sentence-transformers",
        precision="fp32",
        batch_size=batch_size,
        avg_latency_ms=float(np.mean(latencies_arr)),
        p50_latency_ms=float(np.percentile(latencies_arr, 50)),
        p95_latency_ms=float(np.percentile(latencies_arr, 95)),
        p99_latency_ms=float(np.percentile(latencies_arr, 99)),
        throughput_texts_per_sec=throughput,
        latencies=[float(x) for x in latencies],
    )


class ONNXEmbedder:
    """ONNX-based embedder for benchmarking."""

    def __init__(self, model_path: str, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Auto

        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        # Mean pooling over sequence dimension
        last_hidden_state = outputs[0]
        attention_mask = inputs["attention_mask"]

        # Expand attention mask for broadcasting
        mask_expanded = np.expand_dims(attention_mask, -1)

        # Sum and normalize
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1).clip(min=1e-9)

        return sum_embeddings / sum_mask


def benchmark_onnx(
    embedder: ONNXEmbedder,
    texts: list[str],
    batch_size: int,
    iterations: int,
    precision: str,
) -> BenchmarkResult:
    """Benchmark ONNX Runtime inference."""
    latencies = []

    for _ in range(iterations):
        batch = texts[:batch_size]

        start = time.perf_counter()
        _ = embedder.encode(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    throughput = (batch_size / np.mean(latencies_arr)) * 1000

    return BenchmarkResult(
        backend="onnxruntime",
        precision=precision,
        batch_size=batch_size,
        avg_latency_ms=float(np.mean(latencies_arr)),
        p50_latency_ms=float(np.percentile(latencies_arr, 50)),
        p95_latency_ms=float(np.percentile(latencies_arr, 95)),
        p99_latency_ms=float(np.percentile(latencies_arr, 99)),
        throughput_texts_per_sec=throughput,
        latencies=[float(x) for x in latencies],
    )


def compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between embeddings."""
    # Normalize
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=-1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=-1, keepdims=True)

    # Cosine similarity
    similarities = np.sum(emb1_norm * emb2_norm, axis=-1)
    return float(np.mean(similarities))


# =============================================================================
# MAIN BENCHMARK
# =============================================================================


def run_benchmark():
    """Run the full INT8 quantization benchmark."""

    print("=" * 70)
    print("SUITE-002 (Real) — INT8 Quantization Benchmark")
    print("=" * 70)
    print()

    # Check dependencies
    if not DEPENDENCIES_OK:
        print(f"ERROR: Missing dependencies: {', '.join(MISSING)}")
        print("Install with: pip install sentence-transformers onnxruntime onnx transformers")
        sys.exit(1)

    # Detect CPU features
    print("Environment:")
    cpu_features = detect_cpu_features()
    print(f"  CPU: {cpu_features['cpu_name']}")
    print(f"  Platform: {cpu_features['platform']}")
    print(f"  AVX2: {'✓' if cpu_features['avx2'] else '✗'}")
    print(f"  AVX-512: {'✓' if cpu_features['avx512'] else '✗'}")
    print(f"  VNNI: {'✓' if cpu_features['vnni'] else '✗'}")
    print(f"  ONNX Runtime: {ort.__version__}")
    print()

    # Determine expected speedup based on CPU features
    if cpu_features["vnni"]:
        expected_speedup = 3.0  # 3-10x with VNNI
        target_label = "3-10x (VNNI)"
    elif cpu_features["avx512"]:
        expected_speedup = 2.5  # Better than AVX2
        target_label = "2.5-5x (AVX-512)"
    else:
        expected_speedup = 1.5  # Modest gains with AVX2 only
        target_label = "1.5-3x (AVX2)"

    print(f"Target Speedup: {target_label}")
    print()

    # Create temp directory for models
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Load sentence-transformers model
        print(f"Loading model: {MODEL_NAME}")
        print("  (This may take a minute on first run...)")
        st_model = SentenceTransformer(MODEL_NAME)
        print(f"  ✓ Model loaded: {st_model.get_sentence_embedding_dimension()} dimensions")
        print()

        # Export to ONNX
        print("Exporting model to ONNX format...")
        onnx_fp32_path = tmpdir / "model_fp32.onnx"
        try:
            export_model_to_onnx(st_model, onnx_fp32_path)
        except Exception as e:
            print(f"  ⚠ ONNX export failed: {e}")
            print("  Continuing with sentence-transformers baseline only...")
            onnx_fp32_path = None
        print()

        # Quantize to INT8
        onnx_int8_path = None
        if onnx_fp32_path and onnx_fp32_path.exists():
            print("Quantizing model to INT8...")
            onnx_int8_path = tmpdir / "model_int8.onnx"
            try:
                quantize_model_int8(onnx_fp32_path, onnx_int8_path)
            except Exception as e:
                print(f"  ⚠ INT8 quantization failed: {e}")
                onnx_int8_path = None
            print()

        # Create ONNX embedders
        onnx_fp32_embedder = None
        onnx_int8_embedder = None

        if onnx_fp32_path and onnx_fp32_path.exists():
            try:
                onnx_fp32_embedder = ONNXEmbedder(str(onnx_fp32_path), MODEL_NAME)
                print("  ✓ ONNX FP32 embedder ready")
            except Exception as e:
                print(f"  ⚠ ONNX FP32 embedder failed: {e}")

        if onnx_int8_path and onnx_int8_path.exists():
            try:
                onnx_int8_embedder = ONNXEmbedder(str(onnx_int8_path), MODEL_NAME)
                print("  ✓ ONNX INT8 embedder ready")
            except Exception as e:
                print(f"  ⚠ ONNX INT8 embedder failed: {e}")
        print()

        # Accuracy check (INT8 vs FP32)
        print("Accuracy Check (INT8 vs FP32):")
        if onnx_fp32_embedder and onnx_int8_embedder:
            test_texts = SAMPLE_TEXTS[:16]
            fp32_emb = onnx_fp32_embedder.encode(test_texts)
            int8_emb = onnx_int8_embedder.encode(test_texts)
            similarity = compute_embedding_similarity(fp32_emb, int8_emb)
            accuracy_loss = (1 - similarity) * 100
            print(f"  Cosine similarity: {similarity:.4f}")
            print(f"  Accuracy loss: {accuracy_loss:.2f}%")
            accuracy_ok = accuracy_loss < 2.0
            print(f"  Target (<2% loss): {'✓ PASS' if accuracy_ok else '✗ FAIL'}")
        else:
            print("  Skipped (ONNX models not available)")
            accuracy_ok = None
        print()

        # Warmup
        print(f"Warmup ({WARMUP_ITERATIONS} iterations)...")
        for _ in range(WARMUP_ITERATIONS):
            _ = st_model.encode(SAMPLE_TEXTS[:32], show_progress_bar=False)
            if onnx_fp32_embedder:
                _ = onnx_fp32_embedder.encode(SAMPLE_TEXTS[:32])
            if onnx_int8_embedder:
                _ = onnx_int8_embedder.encode(SAMPLE_TEXTS[:32])
        print("  ✓ Warmup complete")
        print()

        # Run benchmarks
        print(f"Running benchmarks ({MEASUREMENT_ITERATIONS} iterations each)...")
        print()

        results = {
            "st_fp32": [],
            "onnx_fp32": [],
            "onnx_int8": [],
        }

        comparisons = []

        for batch_size in BATCH_SIZES:
            print(f"Batch size: {batch_size}")

            # Sentence-transformers FP32
            st_result = benchmark_sentence_transformers(
                st_model, SAMPLE_TEXTS, batch_size, MEASUREMENT_ITERATIONS
            )
            results["st_fp32"].append(st_result)
            print(
                f"  ST FP32:   {st_result.avg_latency_ms:7.2f}ms (P95: {st_result.p95_latency_ms:.2f}ms)"
            )

            # ONNX FP32
            if onnx_fp32_embedder:
                onnx_fp32_result = benchmark_onnx(
                    onnx_fp32_embedder, SAMPLE_TEXTS, batch_size, MEASUREMENT_ITERATIONS, "fp32"
                )
                results["onnx_fp32"].append(onnx_fp32_result)
                print(
                    f"  ONNX FP32: {onnx_fp32_result.avg_latency_ms:7.2f}ms (P95: {onnx_fp32_result.p95_latency_ms:.2f}ms)"
                )

            # ONNX INT8
            if onnx_int8_embedder:
                onnx_int8_result = benchmark_onnx(
                    onnx_int8_embedder, SAMPLE_TEXTS, batch_size, MEASUREMENT_ITERATIONS, "int8"
                )
                results["onnx_int8"].append(onnx_int8_result)
                print(
                    f"  ONNX INT8: {onnx_int8_result.avg_latency_ms:7.2f}ms (P95: {onnx_int8_result.p95_latency_ms:.2f}ms)"
                )

                # Compute speedup vs sentence-transformers FP32
                speedup = st_result.avg_latency_ms / onnx_int8_result.avg_latency_ms
                meets_target = speedup >= expected_speedup
                print(f"  Speedup:   {speedup:.2f}x {'✓' if meets_target else '⚠'}")

                comparisons.append(
                    ComparisonResult(
                        batch_size=batch_size,
                        fp32_latency_ms=st_result.avg_latency_ms,
                        int8_latency_ms=onnx_int8_result.avg_latency_ms,
                        speedup_ratio=speedup,
                        meets_target=meets_target,
                        target_speedup=expected_speedup,
                    )
                )

            print()

        # Summary
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print()

        print("Latency Comparison (batch=32):")
        print("-" * 50)

        # Find batch=32 results
        st_32 = next((r for r in results["st_fp32"] if r.batch_size == 32), None)
        onnx_fp32_32 = (
            next((r for r in results["onnx_fp32"] if r.batch_size == 32), None)
            if results["onnx_fp32"]
            else None
        )
        onnx_int8_32 = (
            next((r for r in results["onnx_int8"] if r.batch_size == 32), None)
            if results["onnx_int8"]
            else None
        )

        if st_32:
            print(f"  Sentence-Transformers FP32: {st_32.avg_latency_ms:7.2f}ms")
        if onnx_fp32_32:
            print(f"  ONNX Runtime FP32:          {onnx_fp32_32.avg_latency_ms:7.2f}ms")
        if onnx_int8_32:
            print(f"  ONNX Runtime INT8:          {onnx_int8_32.avg_latency_ms:7.2f}ms")

        print()

        # Target validation
        print("Target Validation:")
        print("-" * 50)

        # Speedup target
        if comparisons:
            avg_speedup = np.mean([c.speedup_ratio for c in comparisons])
            speedup_pass = avg_speedup >= expected_speedup
            print(f"  Average Speedup: {avg_speedup:.2f}x (target: {expected_speedup}x)")
            print(f"  Speedup Target:  {'✓ PASS' if speedup_pass else '✗ FAIL'}")
        else:
            speedup_pass = False
            print("  Speedup: Not measured (ONNX export failed)")

        # Accuracy target
        if accuracy_ok is not None:
            print(f"  Accuracy Target: {'✓ PASS' if accuracy_ok else '✗ FAIL'} (<2% loss)")

        # Latency target (≤15ms for batch-32)
        latency_target_ms = 15.0
        if onnx_int8_32:
            latency_pass = onnx_int8_32.avg_latency_ms <= latency_target_ms
            print(
                f"  Latency Target:  {'✓ PASS' if latency_pass else '✗ FAIL'} (≤{latency_target_ms}ms @ batch-32)"
            )
        else:
            latency_pass = False

        print()

        # Overall verdict
        all_pass = speedup_pass and (accuracy_ok or accuracy_ok is None)
        print("=" * 70)
        print(f"GAP-002 VALIDATION: {'✓ VALIDATED' if all_pass else '◐ PARTIAL / ✗ FAILED'}")
        print("=" * 70)

        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "environment": {
                "platform": cpu_features["platform"],
                "cpu": cpu_features["cpu_name"],
                "avx2": cpu_features["avx2"],
                "avx512": cpu_features["avx512"],
                "vnni": cpu_features["vnni"],
                "onnxruntime_version": ort.__version__,
            },
            "config": {
                "batch_sizes": BATCH_SIZES,
                "warmup_iterations": WARMUP_ITERATIONS,
                "measurement_iterations": MEASUREMENT_ITERATIONS,
            },
            "results": {
                "st_fp32": [asdict(r) for r in results["st_fp32"]],
                "onnx_fp32": [asdict(r) for r in results["onnx_fp32"]]
                if results["onnx_fp32"]
                else [],
                "onnx_int8": [asdict(r) for r in results["onnx_int8"]]
                if results["onnx_int8"]
                else [],
            },
            "comparisons": [asdict(c) for c in comparisons],
            "validation": {
                "speedup_pass": speedup_pass,
                "accuracy_pass": accuracy_ok,
                "latency_pass": latency_pass if onnx_int8_32 else False,
                "overall_pass": all_pass,
            },
        }

        # Save to file
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        output_file = (
            results_dir / f"suite_002_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            # Convert numpy bools to Python bools for JSON serialization
            def convert_bools(obj):
                if isinstance(obj, dict):
                    return {k: convert_bools(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_bools(v) for v in obj]
                elif isinstance(obj, (np.bool_, np.generic)):
                    return bool(obj)
                return obj

            json.dump(convert_bools(output), f, indent=2)

        print(f"\nResults saved to: {output_file}")

        return all_pass


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
