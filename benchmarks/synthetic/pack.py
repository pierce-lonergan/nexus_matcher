"""
benchmarks.synthetic.pack | Layer: BENCHMARK
The five artifacts as one object, with scale and difficulty as dials.

A pack is a pure function of `(scale, difficulty, seed)`. Nothing here reads a clock, a
process id, an environment variable or a set's iteration order, so two runs of the same
spec produce byte-identical files -- which is the only reason a checksum in the manifest
means anything.

Why the pools grow with the scale
---------------------------------
A 100,000-row glossary drawn from 900 subject words is not a large glossary; it is a small
one repeated. The vocabulary therefore scales with the row count, which is also what keeps
the abbreviation catalog honest: the catalog covers every token the glossary can emit, so
at 100,000 rows it is a catalog of some thousands of entries rather than a thousand-entry
catalog stretched over a corpus it cannot describe.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .abbreviations import (
    AbbreviationCatalog,
    AbbreviationDelta,
    build_catalog,
    build_delta,
    catalog_as_json,
    delta_as_json,
)
from .feedback import FeedbackEvent, build_feedback, wire_loss, write_feedback_jsonl
from .glossary import (
    CSV_HEADER,
    Glossary,
    GlossaryProfile,
    build_glossary,
    glossary_rows_as_dicts,
)
from .pools import Pools, build_pools
from .schemas import SyntheticSchema, build_schemas
from .truth import TruthRow, class_counts, write_truth_csv

NOTICE = (
    "SYNTHETIC DATA. Every subject word, domain and abbreviation in this pack was "
    "manufactured by a seeded syllable grammar in benchmarks/synthetic/pools.py. Nothing "
    "here was sampled from any glossary, and this pack is not a template for one."
)

DEFAULT_SEED = 20260819


@dataclass(frozen=True)
class PackSpec:
    """The dials. Everything else is derived."""

    rows: int = 10_000
    difficulty: float = 1.0
    seed: int = DEFAULT_SEED
    schema_scale: float = 1.0
    feedback_events: int = 5_000
    paraphrase_strength: float = 0.6
    repeats_per_domain: int = 2

    @property
    def subjects(self) -> int:
        """Vocabulary size for this row count, so a large glossary is not a small one
        repeated. Floored so a 1,000-row pack still has room for its own clusters."""
        return max(400, min(8_000, self.rows // 14))


@dataclass
class SyntheticPack:
    spec: PackSpec
    pools: Pools
    glossary: Glossary
    catalog: AbbreviationCatalog
    delta: AbbreviationDelta
    schemas: tuple[SyntheticSchema, ...]
    feedback: tuple[FeedbackEvent, ...] = field(default_factory=tuple)

    @classmethod
    def generate(cls, spec: PackSpec | None = None) -> SyntheticPack:
        spec = spec or PackSpec()
        pools = build_pools(spec.seed, n_subjects=spec.subjects)
        profile = GlossaryProfile(rows=spec.rows).scaled(spec.difficulty)
        glossary = build_glossary(pools, profile, spec.seed)
        catalog = build_catalog(pools, spec.seed)
        delta = build_delta(catalog, spec.seed)
        schemas = build_schemas(
            pools,
            glossary,
            catalog,
            spec.seed,
            scale=spec.schema_scale,
            paraphrase_strength=spec.paraphrase_strength,
            repeats_per_domain=spec.repeats_per_domain,
        )
        pack = cls(
            spec=spec,
            pools=pools,
            glossary=glossary,
            catalog=catalog,
            delta=delta,
            schemas=schemas,
        )
        pack.feedback = build_feedback(
            glossary, pack.truth_rows(), spec.seed, count=spec.feedback_events
        )
        return pack

    # -- accessors -----------------------------------------------------------

    def truth_rows(self) -> tuple[TruthRow, ...]:
        return tuple(row for schema in self.schemas for row in schema.truth)

    def schema(self, name: str) -> SyntheticSchema:
        for s in self.schemas:
            if s.name == name:
                return s
        raise KeyError(f"no schema named {name!r}; have {[s.name for s in self.schemas]}")

    def glossary_dicts(self) -> list[dict[str, str]]:
        """The glossary in the shape `load_entries` reads, without touching disk."""
        return glossary_rows_as_dicts(self.glossary)

    # -- writing -------------------------------------------------------------

    def write(self, out_dir: Path) -> dict[str, object]:
        """Write every artifact and return the manifest (which is also written)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "schemas").mkdir(parents=True, exist_ok=True)

        self._write_glossary(out_dir / "glossary.csv")
        _write_json(out_dir / "abbreviations.json", catalog_as_json(self.catalog))
        _write_json(out_dir / "abbreviations-delta.json", delta_as_json(self.delta))
        for schema in self.schemas:
            _write_json(
                out_dir / "schemas" / f"{schema.name}.flattened.json",
                {"notice": NOTICE, "schema": schema.name, "fields": schema.flattened},
            )
            _write_json(out_dir / "schemas" / f"{schema.name}.avsc", schema.as_avro())
        write_truth_csv(out_dir / "truth.csv", self.truth_rows())
        write_feedback_jsonl(out_dir / "feedback.jsonl", self.feedback)

        manifest = self.manifest(out_dir)
        _write_json(out_dir / "manifest.json", manifest)
        return manifest

    def _write_glossary(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_HEADER), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.glossary_dicts())

    def manifest(self, out_dir: Path | None = None) -> dict[str, object]:
        """
        What was generated, in numbers, with a checksum per file when they were written.

        The distribution figures are MEASURED off the generated rows rather than copied
        from the profile that asked for them. A generator that reports its inputs as if
        they were its outputs cannot notice that it failed to hit them -- which is the
        same failure as a quality gate that reports the threshold instead of the reading.
        """
        rows = self.glossary.rows
        approved = sum(1 for r in rows if r.is_approved)
        clustered = sum(1 for r in rows if r.cluster_id)
        echoing = sum(1 for r in rows if r.definition_echoes_name)
        name_lengths = sorted(len(r.tokens) for r in rows)
        wide = self.glossary.wide_clusters()

        manifest: dict[str, object] = {
            "notice": NOTICE,
            "spec": asdict(self.spec),
            "glossary": {
                "rows": len(rows),
                "approved": approved,
                "non_approved_share": round(1 - approved / len(rows), 4) if rows else 0.0,
                "in_near_duplicate_cluster_share": round(clustered / len(rows), 4) if rows else 0.0,
                "definition_echoes_name_share": round(echoing / len(rows), 4) if rows else 0.0,
                "clusters": len(self.glossary.clusters),
                "widest_cluster": len(wide[0]) if wide else 0,
                "name_tokens_median": name_lengths[len(name_lengths) // 2] if name_lengths else 0,
                "name_tokens_max": name_lengths[-1] if name_lengths else 0,
                "distinct_domains": len({r.domain for r in rows}),
                "distinct_class_words": len({r.class_word for r in rows}),
            },
            "abbreviations": {
                "entries": len(self.catalog.expansions),
                "ambiguous_shorts": len(self.catalog.ambiguous),
                "multi_word_rules": len(self.catalog.multi_word),
                "stopword_collisions": len(self.catalog.stopword_collisions),
                "never_expand": len(self.catalog.never_expand),
                "not_contracted": len(self.catalog.identity),
                "delta_entries": len(self.delta.changed),
                "delta_version": self.delta.version,
            },
            "schemas": {
                s.name: {
                    "fields": len(s.flattened),
                    "with_doc": sum(1 for r in s.flattened if r.get("doc")),
                    "distinct_leaf_names": len({r["leafName"] for r in s.flattened}),
                    "max_leaf_repetition": _max_repetition(s),
                    "max_path_depth": max(
                        (r["flattenedName"].count("_") + 1 for r in s.flattened), default=0
                    ),
                }
                for s in self.schemas
            },
            "truth": class_counts(self.truth_rows()),
            "feedback": wire_loss(self.feedback),
        }
        if out_dir is not None:
            manifest["checksums"] = _checksums(out_dir)
        return manifest


def _max_repetition(schema: SyntheticSchema) -> int:
    counts: dict[str, int] = {}
    for row in schema.flattened:
        counts[row["leafName"]] = counts.get(row["leafName"], 0) + 1
    return max(counts.values(), default=0)


def _write_json(path: Path, payload: object) -> None:
    """Sorted keys, two-space indent, `\\n` endings, pure ASCII.

    ASCII because this repository has already shipped a surface that raised
    `'charmap' codec can't encode character` on a console using a legacy Windows code
    page, and a fixture nobody can print on their machine is a fixture nobody reads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    path.write_text(text + "\n", encoding="ascii", newline="\n")


def _checksums(out_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[path.relative_to(out_dir).as_posix()] = digest
    return out
