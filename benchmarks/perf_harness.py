"""
benchmarks.perf_harness | Layer: BENCHMARK
Throughput and latency of the matching hot path -- the SPEED baseline, not the accuracy one.

Everything else under benchmarks/ measures whether the answer is RIGHT. This measures how
fast it arrives, which is a separate question with separate failure modes: a change can
leave P@1 untouched and still triple the cost per field.

What it measures
----------------
  * index throughput   -- entries/sec for load + embed + vector index + BM25 index
  * match throughput    -- fields/sec end to end, the number a user actually feels
  * per-field latency   -- p50 / p95 / p99, because a mean hides a bimodal tail
  * peak memory         -- tracemalloc high-water mark, and the resident set if psutil is
                           available (tracemalloc misses numpy/onnxruntime buffers, which
                           on this workload are most of the footprint)
  * a cProfile breakdown of cumulative time, to point at what to fix next

Why the scales are what they are
--------------------------------
Corpus size is the dominant term in this system's cost AND in its accuracy (see the alias
scale inversion in MatchingConfig). A single-size benchmark would hide that, so the
default sweep spans 1k -> 30k entries, which brackets the enterprise glossaries this is
built for.

Determinism
-----------
Every glossary and schema is generated from a fixed seed, so two runs compare like with
like. The encoder is the bundled int8 ONNX model, which the README notes is NOT
batch-invariant -- so keep batch sizes fixed when comparing runs, and prefer comparing
timings rather than scores across code changes.

Usage
-----
    python benchmarks/perf_harness.py                      # default sweep
    python benchmarks/perf_harness.py --entries 10000 --fields 500
    python benchmarks/perf_harness.py --profile            # add a cProfile breakdown
    python benchmarks/perf_harness.py --baseline out.json  # write a high-score file
    python benchmarks/perf_harness.py --compare out.json   # diff against it
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import pstats
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "results"

# Vocabulary for synthetic glossaries. Deliberately drawn from the domain the library
# targets, so token statistics (length, overlap, abbreviation density) resemble the real
# thing rather than random strings, which would make BM25 look artificially cheap.
_SUBJECTS = (
    "customer",
    "account",
    "order",
    "invoice",
    "payment",
    "product",
    "shipment",
    "employee",
    "claim",
    "policy",
    "patient",
    "encounter",
    "transaction",
    "merchant",
    "subscription",
    "device",
    "session",
    "campaign",
    "vendor",
    "contract",
)
_ATTRS = (
    "identifier",
    "full name",
    "email address",
    "postal code",
    "street line",
    "created timestamp",
    "updated timestamp",
    "status code",
    "total amount",
    "currency code",
    "quantity",
    "description",
    "phone number",
    "birth date",
    "expiry date",
    "tax amount",
    "discount rate",
    "balance",
    "category",
    "region",
)
_QUALIFIERS = ("primary", "secondary", "billing", "shipping", "legal", "preferred", "")

_ABBREV = {
    "identifier": "id",
    "full name": "nm",
    "email address": "eml",
    "postal code": "zip",
    "street line": "str",
    "created timestamp": "crt_ts",
    "updated timestamp": "upd_ts",
    "status code": "stat_cd",
    "total amount": "tot_amt",
    "currency code": "ccy",
    "quantity": "qty",
    "description": "desc",
    "phone number": "phn",
    "birth date": "dob",
    "expiry date": "exp_dt",
    "tax amount": "tax_amt",
    "discount rate": "disc_rt",
    "balance": "bal",
    "category": "cat",
    "region": "rgn",
}


@dataclass
class Measurement:
    """One scale point."""

    entries: int
    fields: int
    index_seconds: float
    index_entries_per_sec: float
    match_seconds: float
    match_fields_per_sec: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_mean: float
    peak_tracemalloc_mb: float
    rss_delta_mb: float | None = None
    notes: list[str] = dc_field(default_factory=list)


def _make_glossary(n: int, seed: int = 20260806) -> list[Any]:
    """Build n distinct dictionary entries with realistic token statistics."""
    from nexus_matcher.domain.models.entities import DictionaryEntry
    from nexus_matcher.shared.types.base import DataType, ProtectionLevel

    types = list(DataType)
    levels = list(ProtectionLevel)
    entries: list[Any] = []
    seen: set[str] = set()
    i = 0
    while len(entries) < n:
        i += 1
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        attr = _ATTRS[(i // len(_SUBJECTS)) % len(_ATTRS)]
        qual = _QUALIFIERS[(i // (len(_SUBJECTS) * len(_ATTRS))) % len(_QUALIFIERS)]
        # A numeric suffix only once the combinatorial space is exhausted, so the first
        # few thousand entries stay natural-looking.
        base = f"{qual} {subject} {attr}".strip()
        name = base.title()
        if name in seen:
            name = f"{name} {len(entries)}"
        seen.add(name)

        entries.append(
            DictionaryEntry(
                id=f"dict-{len(entries):06d}",
                business_name=name,
                logical_name=f"{subject}_{_ABBREV.get(attr, attr).replace(' ', '_')}",
                definition=(
                    f"The {attr} associated with the {subject} record, "
                    f"maintained by the {subject} domain and used for reporting."
                ),
                data_type=types[i % len(types)],
                protection_level=levels[i % len(levels)],
                domain=subject,
            )
        )
    return entries


def _make_fields(n: int, seed: int = 20260806) -> list[Any]:
    """Build n schema fields shaped like flattened Avro leaves."""
    from nexus_matcher.domain.models.entities import SchemaField
    from nexus_matcher.shared.types.base import DataType

    types = list(DataType)
    fields: list[Any] = []
    for i in range(n):
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        attr = _ATTRS[(i // len(_SUBJECTS)) % len(_ATTRS)]
        abbrev = _ABBREV.get(attr, attr).replace(" ", "_")
        leaf = f"{abbrev}_{i}" if i >= len(_SUBJECTS) * len(_ATTRS) else abbrev
        fields.append(
            SchemaField(
                name=leaf,
                data_type=types[i % len(types)],
                full_path=f"{subject}.{leaf}",
                parent_path=subject,
                # Half the fields carry a doc, matching the real mix: flatteners often
                # drop `doc`, so a benchmark where every field has one is optimistic.
                description=(f"The {attr} of the {subject}." if i % 2 == 0 else ""),
            )
        )
    return fields


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1e6


def measure(entries_n: int, fields_n: int, warmup: bool = True) -> Measurement:
    """Run one scale point end to end."""
    from nexus_matcher.application.use_cases.match_schema import NexusMatcher

    entries = _make_glossary(entries_n)
    fields = _make_fields(fields_n)

    matcher = NexusMatcher.from_config()

    if warmup:
        # The first encode pays ONNX session setup and lazy imports. Charging that to the
        # first scale point would make the sweep look superlinear for the wrong reason.
        matcher._embedding_provider.embed_documents(["warmup text"])

    gc.collect()
    tracemalloc.start()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    matcher._index_dictionary(entries)
    index_seconds = time.perf_counter() - t0

    # Per-field latency: measured one field at a time so the distribution is real.
    # Throughput is measured separately on the BATCHED path, because that is what a user
    # invoking match_schema actually gets and it is several times faster.
    sample = fields[: min(len(fields), 200)]
    latencies: list[float] = []
    for f in sample:
        t = time.perf_counter()
        matcher._match_field(f)
        latencies.append((time.perf_counter() - t) * 1000.0)

    t0 = time.perf_counter()
    matcher._match_fields(fields)
    match_seconds = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_mb()

    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        k = min(len(latencies) - 1, round((p / 100.0) * (len(latencies) - 1)))
        return latencies[k]

    return Measurement(
        entries=entries_n,
        fields=fields_n,
        index_seconds=round(index_seconds, 4),
        index_entries_per_sec=round(entries_n / index_seconds, 1) if index_seconds else 0.0,
        match_seconds=round(match_seconds, 4),
        match_fields_per_sec=round(fields_n / match_seconds, 1) if match_seconds else 0.0,
        latency_ms_p50=round(pct(50), 3),
        latency_ms_p95=round(pct(95), 3),
        latency_ms_p99=round(pct(99), 3),
        latency_ms_mean=round(statistics.mean(latencies), 3) if latencies else 0.0,
        peak_tracemalloc_mb=round(peak / 1e6, 2),
        rss_delta_mb=(
            round(rss_after - rss_before, 1)
            if rss_before is not None and rss_after is not None
            else None
        ),
    )


def profile(entries_n: int, fields_n: int, top: int = 30) -> str:
    """cProfile the MATCH path only -- indexing is a one-off, matching is per request."""
    from nexus_matcher.application.use_cases.match_schema import NexusMatcher

    entries = _make_glossary(entries_n)
    fields = _make_fields(fields_n)
    matcher = NexusMatcher.from_config()
    matcher._embedding_provider.embed_documents(["warmup"])
    matcher._index_dictionary(entries)

    prof = cProfile.Profile()
    prof.enable()
    matcher._match_fields(fields)
    prof.disable()

    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def _fmt_table(rows: list[Measurement]) -> str:
    head = (
        f"{'entries':>8} {'fields':>7} {'index s':>8} {'ent/s':>8} "
        f"{'match s':>8} {'fld/s':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'peak MB':>8}"
    )
    lines = [head, "-" * len(head)]
    for m in rows:
        lines.append(
            f"{m.entries:>8} {m.fields:>7} {m.index_seconds:>8.2f} "
            f"{m.index_entries_per_sec:>8.0f} {m.match_seconds:>8.2f} "
            f"{m.match_fields_per_sec:>8.1f} {m.latency_ms_p50:>8.2f} "
            f"{m.latency_ms_p95:>8.2f} {m.latency_ms_p99:>8.2f} {m.peak_tracemalloc_mb:>8.1f}"
        )
    return "\n".join(lines)


def _diff(current: list[Measurement], baseline_path: Path) -> str:
    """Compare against a saved high score. Positive percentages are improvements."""
    prior = {
        (r["entries"], r["fields"]): r
        for r in json.loads(baseline_path.read_text(encoding="utf-8"))["measurements"]
    }
    head = (
        f"{'entries':>8} {'fields':>7} {'fld/s base':>11} {'fld/s now':>10} {'delta':>9} "
        f"{'p95 base':>9} {'p95 now':>8} {'delta':>9}"
    )
    lines = [head, "-" * len(head)]
    for m in current:
        b = prior.get((m.entries, m.fields))
        if not b:
            lines.append(f"{m.entries:>8} {m.fields:>7}   (no baseline)")
            continue
        thr = (
            (m.match_fields_per_sec - b["match_fields_per_sec"]) / b["match_fields_per_sec"] * 100
            if b["match_fields_per_sec"]
            else 0.0
        )
        # Latency: lower is better, so invert the sign to keep "positive = better".
        lat = (
            (b["latency_ms_p95"] - m.latency_ms_p95) / b["latency_ms_p95"] * 100
            if b["latency_ms_p95"]
            else 0.0
        )
        lines.append(
            f"{m.entries:>8} {m.fields:>7} {b['match_fields_per_sec']:>11.1f} "
            f"{m.match_fields_per_sec:>10.1f} {thr:>+8.1f}% "
            f"{b['latency_ms_p95']:>9.2f} {m.latency_ms_p95:>8.2f} {lat:>+8.1f}%"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entries", type=int, action="append", help="glossary size (repeatable)")
    ap.add_argument("--fields", type=int, default=500, help="schema fields to match")
    ap.add_argument("--profile", action="store_true", help="print a cProfile breakdown")
    ap.add_argument("--baseline", type=str, help="write results here as the new high score")
    ap.add_argument("--compare", type=str, help="diff against a saved high score")
    ap.add_argument("--label", type=str, default="", help="name this run in the output file")
    args = ap.parse_args()

    scales = args.entries or [1000, 5000, 10000, 30000]

    print(f"\nnexus-matcher performance baseline  (fields={args.fields})")
    print("=" * 92)
    rows: list[Measurement] = []
    for n in scales:
        print(f"  measuring {n} entries ...", flush=True)
        rows.append(measure(n, args.fields))

    print()
    print(_fmt_table(rows))

    if args.compare:
        path = Path(args.compare)
        if path.exists():
            print("\nDiff vs baseline (positive = faster):")
            print(_diff(rows, path))
        else:
            print(f"\n(no baseline at {path}, nothing to compare)")

    if args.baseline:
        path = Path(args.baseline)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"label": args.label, "measurements": [asdict(m) for m in rows]}, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote baseline -> {path}")

    if args.profile:
        print("\ncProfile (cumulative), 10k entries:\n")
        print(profile(10000, args.fields))


if __name__ == "__main__":
    main()
