"""
benchmarks.datasets.extract_instances | Layer: BENCHMARK
Extract real column VALUES from the BIRD SQLite databases.

Why this exists
---------------
The abbreviation ceiling (bird P@1 0.598 vs omop 0.831) comes from column names that carry
almost no signal: "sname", "NumTstTakr", "AvgScrRead". But those columns hold real data,
and the data is unambiguous -- "sname" contains "FAME Public Charter", "Envision Academy
for Arts & Technology". A human reading the values knows instantly what the column is.

This script profiles every column in the BIRD dev databases so that signal becomes
available to the matcher. It emits, per column:

    sample_values   : most frequent distinct non-null values (informative for categoricals)
    n_distinct      : cardinality
    null_rate       : fraction NULL
    sql_type        : declared SQLite type
    is_numeric      : whether the values parse as numbers
    numeric_stats   : min / max / mean when numeric
    pattern         : coarse character-class signature of the values

Output: data/benchmarks/bird/instances.jsonl, keyed by the same id used in
dictionary.jsonl / queries.jsonl so it joins cleanly.

Note on generality: in a real deployment the SOURCE schema has data while the business
glossary does not, so this is an ASYMMETRIC signal -- values enrich the QUERY side only.
That is exactly how the benchmark uses it.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_ROOT = REPO_ROOT / "data" / "raw" / "dev_20240627" / "dev_databases"
OUT = REPO_ROOT / "data" / "benchmarks" / "bird" / "instances.jsonl"

MAX_SAMPLE = 25
MAX_VALUE_CHARS = 60
SCAN_LIMIT = 5000  # rows scanned per column; enough to profile, fast on 1.4 GB


def char_pattern(value: str) -> str:
    """Coarse character-class signature: 'Xxx 999' style, collapsed."""
    s = re.sub(r"[A-Z]", "X", value)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"[0-9]", "9", s)
    s = re.sub(r"(.)\1+", r"\1+", s)
    return s[:24]


def profile_column(conn: sqlite3.Connection, table: str, column: str) -> dict | None:
    """Profile one column. Returns None if the column cannot be read."""
    q_table = f'"{table}"'
    q_col = f'"{column}"'
    try:
        rows = conn.execute(f"SELECT {q_col} FROM {q_table} LIMIT {SCAN_LIMIT}").fetchall()
    except sqlite3.Error:
        return None

    total = len(rows)
    if total == 0:
        return None

    values = [r[0] for r in rows]
    non_null = [v for v in values if v is not None and str(v).strip() != ""]
    null_rate = 1.0 - (len(non_null) / total)

    if not non_null:
        return {
            "sample_values": [],
            "n_distinct": 0,
            "null_rate": 1.0,
            "is_numeric": False,
            "numeric_stats": None,
            "pattern": "",
        }

    as_text = [str(v)[:MAX_VALUE_CHARS] for v in non_null]

    # Most frequent distinct values: for a categorical column these ARE the meaning.
    counts = Counter(as_text)
    sample = [v for v, _ in counts.most_common(MAX_SAMPLE)]

    numeric_vals: list[float] = []
    for v in non_null:
        try:
            numeric_vals.append(float(v))
        except (TypeError, ValueError):
            break
    is_numeric = len(numeric_vals) == len(non_null)

    numeric_stats = None
    if is_numeric and numeric_vals:
        numeric_stats = {
            "min": min(numeric_vals),
            "max": max(numeric_vals),
            "mean": sum(numeric_vals) / len(numeric_vals),
        }

    pattern = Counter(char_pattern(v) for v in as_text[:200]).most_common(1)[0][0]

    return {
        "sample_values": sample,
        "n_distinct": len(counts),
        "null_rate": round(null_rate, 4),
        "is_numeric": is_numeric,
        "numeric_stats": numeric_stats,
        "pattern": pattern,
    }


def main() -> None:
    if not DB_ROOT.exists():
        raise SystemExit(f"BIRD databases not found at {DB_ROOT}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with OUT.open("w", encoding="utf-8") as out:
        for db_dir in sorted(p for p in DB_ROOT.iterdir() if p.is_dir()):
            sqlite_files = list(db_dir.glob("*.sqlite"))
            if not sqlite_files:
                continue
            db_id = db_dir.name

            conn = sqlite3.connect(f"file:{sqlite_files[0]}?mode=ro", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            try:
                tables = [
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    )
                ]
                for table in tables:
                    try:
                        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
                    except sqlite3.Error:
                        continue
                    for col in cols:
                        name, sql_type = col[1], col[2]
                        prof = profile_column(conn, table, name)
                        if prof is None:
                            skipped += 1
                            continue
                        rec = {
                            # Matches the id scheme in build_benchmarks.py
                            "id": f"bird::{db_id}::{table}::{name}",
                            "db": db_id,
                            "table": table,
                            "column": name,
                            "sql_type": sql_type,
                            **prof,
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
            finally:
                conn.close()

    print(f"profiled {written} columns ({skipped} unreadable) -> {OUT}")


if __name__ == "__main__":
    main()
