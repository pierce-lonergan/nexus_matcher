"""
benchmarks.datasets.build_benchmarks | Layer: BENCHMARK
Convert raw downloaded corpora into a canonical schema-matching benchmark format.

Canonical format (one directory per benchmark under data/benchmarks/<name>/):

    dictionary.jsonl  - the "data dictionary" / business glossary we retrieve against
        {"id", "business_name", "logical_name", "description", "data_type", "domain"}

    queries.jsonl     - the source-schema fields we must match
        {"id", "field_name", "field_path", "data_type", "parent_path", "gold_id"}

Every query has exactly one gold dictionary entry, so P@1 / MRR / Recall@k are well defined.

Sources
-------
bird   : BIRD-SQL dev set (11 databases, 798 columns). Each database ships a
         `database_description/*.csv` giving original_column_name -> column_name +
         column_description. The technical name is the QUERY, the business name +
         description is the DICTIONARY ENTRY. Heavily abbreviated (sname -> "school name"),
         which is exactly the hard case for schema matching.

omop   : OHDSI OMOP Common Data Model v5.4 field-level specification (327 fields with
         a definition, across 39 tables). cdmFieldName is the QUERY; the DICTIONARY
         ENTRY is the TABLE name plus that field's `userGuidance` prose. The entry's
         business name is deliberately NOT derived from the field name -- see the
         leakage note on build_omop(), which explains why an earlier version of this
         builder made the task string-identity and inflated every downstream number.

Both dictionaries are pooled into a single `combined` benchmark (688 pairs) so that
every query competes against 687 distractors drawn from two unrelated domains.

Scale note: 688 entries is small for an enterprise glossary. Accuracy decays as the
corpus grows -- and the synthetic distractors used by benchmarks/exp_scale.py are
measurably LESS confusable than real entries (24.9% vs 42.3% outrank the gold), so the
decay curve that script reports is an optimistic bound, not a pessimistic one.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
OUT = REPO_ROOT / "data" / "benchmarks"


# =============================================================================
# CANONICAL RECORDS
# =============================================================================


@dataclass(frozen=True)
class DictEntry:
    id: str
    business_name: str
    logical_name: str
    description: str
    data_type: str
    domain: str


@dataclass(frozen=True)
class Query:
    id: str
    field_name: str
    field_path: str
    data_type: str
    parent_path: str
    gold_id: str


@dataclass
class Benchmark:
    name: str
    entries: list[DictEntry] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)

    def write(self, out_root: Path) -> None:
        d = out_root / self.name
        d.mkdir(parents=True, exist_ok=True)
        with (d / "dictionary.jsonl").open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        with (d / "queries.jsonl").open("w", encoding="utf-8") as f:
            for q in self.queries:
                f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


# =============================================================================
# HELPERS
# =============================================================================


def _read_text(path: Path) -> str:
    """Read a CSV that may be utf-8-sig, utf-8, or latin-1."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1", errors="replace")


def _clean(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _humanise(name: str) -> str:
    """person_id -> 'person id'; AvgScrRead -> 'Avg Scr Read'."""
    s = name.replace("_", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    return _clean(s)


def _normalise_type(raw: str) -> str:
    t = (raw or "").strip().lower()
    if not t:
        return "unknown"
    if any(k in t for k in ("int", "serial", "bigint", "smallint")):
        return "integer"
    if any(k in t for k in ("float", "double", "real", "numeric", "decimal", "number")):
        return "float"
    if any(k in t for k in ("bool", "bit")):
        return "boolean"
    if any(k in t for k in ("datetime", "timestamp")):
        return "datetime"
    if "date" in t:
        return "date"
    if "time" in t:
        return "time"
    if any(k in t for k in ("char", "text", "string", "varchar", "clob")):
        return "string"
    return "unknown"


# =============================================================================
# BIRD
# =============================================================================


def build_bird() -> Benchmark:
    """
    Build from BIRD dev `database_description/*.csv`.

    Only rows where the business name (or description) actually DIFFERS from the
    technical name are kept -- otherwise the task is a trivial string identity match
    and would inflate scores meaninglessly.
    """
    bm = Benchmark("bird")
    root = RAW / "dev_20240627" / "dev_databases"
    if not root.exists():
        print(f"  [bird] SKIP - {root} not found")
        return bm

    seen: set[str] = set()
    for csv_path in sorted(root.glob("*/database_description/*.csv")):
        db = csv_path.parents[1].name
        table = csv_path.stem
        try:
            reader = csv.DictReader(io.StringIO(_read_text(csv_path)))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [bird] unreadable {csv_path}: {exc}")
            continue

        for row in reader:
            tech = _clean(row.get("original_column_name"))
            if not tech:
                continue

            business = _clean(row.get("column_name"))
            desc = _clean(row.get("column_description"))
            dtype = _normalise_type(row.get("data_format", ""))

            # The business name is the retrieval target. Fall back to the
            # description when no separate name is given.
            label = business or desc
            if not label:
                continue

            # Reject trivial identity pairs: if the business label is just the
            # technical name (modulo case/underscores), there is nothing to match.
            if label.lower().replace(" ", "") == tech.lower().replace("_", "").replace(" ", ""):
                continue

            entry_id = f"bird::{db}::{table}::{tech}"
            if entry_id in seen:
                continue
            seen.add(entry_id)

            bm.entries.append(
                DictEntry(
                    id=entry_id,
                    business_name=label,
                    logical_name=tech,
                    description=desc if desc.lower() != label.lower() else "",
                    data_type=dtype,
                    domain=db,
                )
            )
            bm.queries.append(
                Query(
                    id=f"q::{entry_id}",
                    field_name=tech,
                    field_path=f"{table}.{tech}",
                    data_type=dtype,
                    parent_path=table,
                    gold_id=entry_id,
                )
            )

    print(f"  [bird] {len(bm.entries)} entries / {len(bm.queries)} queries")
    return bm


# =============================================================================
# OMOP
# =============================================================================


def build_omop() -> Benchmark:
    """
    Build from the OHDSI OMOP CDM v5.4 field-level spec.

    LEAKAGE NOTE -- read before changing this.

    OMOP publishes no human-authored *business name*; the only prose it gives is
    `userGuidance`. An earlier version of this builder synthesised a business name as
    humanise(table) + humanise(field), e.g. person_id -> "person person id". That made
    the benchmark DEGENERATE: the query representation is also humanise(table) +
    humanise(field), so query and gold label were token-identical (measured mean token
    overlap 1.000 against 0.240 for BIRD). Retrieval scored 0.796 P@1 for free and rose
    to 0.988 with alias generation -- a string-identity result masquerading as semantic
    matching, which inflated the combined headline and would have credited techniques
    for a gain they did not produce.

    So the business name here is the TABLE only, which is genuinely available in a
    glossary and is shared by every field in that table. All discriminating signal must
    come from `userGuidance`. The task becomes: given a technical field name, find the
    entry whose human definition describes it. That is real, and it is hard.

    Rows without guidance are dropped -- with no business name AND no description they
    would be unmatchable by construction.
    """
    bm = Benchmark("omop")
    path = RAW / "omop_cdm_field_level.csv"
    if not path.exists():
        print(f"  [omop] SKIP - {path} not found")
        return bm

    dropped = 0
    reader = csv.DictReader(io.StringIO(_read_text(path)))
    for row in reader:
        tech = _clean(row.get("cdmFieldName"))
        table = _clean(row.get("cdmTableName"))
        if not tech or not table:
            continue

        guidance = _clean(row.get("userGuidance"))
        if not guidance or guidance.upper() == "NA":
            dropped += 1
            continue

        dtype = _normalise_type(row.get("cdmDatatype", ""))

        entry_id = f"omop::{table}::{tech}"
        bm.entries.append(
            DictEntry(
                id=entry_id,
                # Table-level label only: NOT derived from the field name.
                business_name=_humanise(table),
                logical_name=tech,
                description=guidance,
                data_type=dtype,
                domain=table,
            )
        )
        bm.queries.append(
            Query(
                id=f"q::{entry_id}",
                field_name=tech,
                field_path=f"{table}.{tech}",
                data_type=dtype,
                parent_path=table,
                gold_id=entry_id,
            )
        )

    print(
        f"  [omop] {len(bm.entries)} entries / {len(bm.queries)} queries "
        f"({dropped} dropped for having no description)"
    )
    return bm


# =============================================================================
# COMBINED
# =============================================================================


def build_combined(parts: list[Benchmark]) -> Benchmark:
    """
    Pool every dictionary into one corpus.

    This is the headline benchmark: each query must find its gold entry among all
    ~1200 entries drawn from two unrelated domains (healthcare + assorted OLTP),
    which is a realistic enterprise-glossary retrieval setting with real distractors.
    """
    bm = Benchmark("combined")
    for p in parts:
        bm.entries.extend(p.entries)
        bm.queries.extend(p.queries)
    print(f"  [combined] {len(bm.entries)} entries / {len(bm.queries)} queries")
    return bm


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building benchmarks...")

    parts = [b for b in (build_bird(), build_omop()) if b.entries]
    if not parts:
        raise SystemExit("No source data found. Run benchmarks/datasets/download.py first.")

    for b in parts:
        b.write(OUT)

    combined = build_combined(parts)
    combined.write(OUT)

    print(f"\nWrote benchmarks to {OUT}")
    for b in [*parts, combined]:
        print(f"  {b.name:10s} dictionary={len(b.entries):5d}  queries={len(b.queries):5d}")


if __name__ == "__main__":
    main()
