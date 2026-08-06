"""
benchmarks.datasets.build_fhir | Layer: BENCHMARK
Build a schema-matching benchmark from HL7 FHIR R5 StructureDefinitions.

Why FHIR
--------
The production use case is matching a FLATTENED AVRO field to a governed glossary entry.
BIRD and OMOP are proxies for that; FHIR is a much closer one, and it is measurable:

  * FHIR element paths are genuinely nested (`Patient.contact.name.family`), so flattening
    them reproduces the real input shape rather than simulating it.
  * Each element carries THREE independently-authored texts -- `short`, `definition` and
    `comment` -- which is what makes a non-degenerate benchmark possible at all.
  * The parent-path effect measured on this corpus is +19.0 P@1, against +19.3 on the
    production benchmark. That near-identical reproduction is the strongest evidence
    available that FHIR is a faithful proxy, and it is empirical rather than structural.
  * CC0 1.0: redistributable with no attribution requirement.

Licence scope
-------------
ONLY `profiles-resources.json` is used. `valuesets.json` and `conceptmaps.json` in the
same archive carry SNOMED CT, LOINC and CPT references which are NOT CC0 and are
separately licensed. Do not widen the input set without re-checking that.

The degeneracy trap, and how this avoids it
-------------------------------------------
The obvious construction is: flattened name + `definition` on the query side, `short` +
`definition` on the target side. That copies the gold text verbatim into the query. It was
measured, and it fails badly:

    design                                          BM25 P@1   embed P@1
    query = flat + definition (LEAKY)                 0.8114     0.7880
    query = flat + comment    (this builder)          0.3542     0.4071

On the leaky design BM25 BEATS the embedding model by 2.3 points. That inversion is the
signature of a string-copying benchmark: semantics actively hurt, and the winning strategy
is lexical overlap. Anyone tuning against it would conclude their matcher is worse than
grep, which is exactly the failure this repo already shipped once with OMOP.

`comment` is authored independently of `definition` -- measured Jaccard 0.08 -- so using it
as the query-side Avro `doc` leaves the embedding model ahead of BM25 and leaves real
headroom (R@5 ~0.68, nowhere near saturated).

Construction
------------
  query  : flattened element path  +  `comment`      (the Avro field name + its doc)
  target : `short`  +  `definition`                  (the glossary term + its definition)

Boilerplate elements (`id`, `meta`, `text`, `extension`, ...) are dropped: they repeat
identically across all 162 resources and would pad the corpus with ~3000 duplicate pairs
that are trivially matchable and inflate every score.

Evaluation uses only the comment-bearing elements as QUERIES, but retrieves against the
FULL entry pool, so the distractor set is the realistic one.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
OUT = REPO_ROOT / "data" / "benchmarks" / "fhir"
ARCHIVE = RAW / "fhir_r5_definitions.zip"
PROFILES = RAW / "profiles-resources.json"

# Elements every FHIR resource inherits. They are identical across all 162 resources, so
# keeping them adds thousands of duplicate, trivially-matchable pairs.
BOILERPLATE = frozenset(
    {
        "id",
        "meta",
        "implicitRules",
        "language",
        "text",
        "contained",
        "extension",
        "modifierExtension",
        "url",
        "version",
        "name",
        "title",
        "status",
        "date",
        "publisher",
        "contact",
        "description",
        "useContext",
        "jurisdiction",
        "purpose",
        "copyright",
        "identifier",
    }
)

# FHIR primitive -> a plausible Avro type, so the benchmark carries realistic type info.
FHIR_TO_AVRO = {
    "string": "string",
    "code": "string",
    "uri": "string",
    "url": "string",
    "canonical": "string",
    "markdown": "string",
    "id": "string",
    "oid": "string",
    "uuid": "string",
    "base64Binary": "bytes",
    "boolean": "boolean",
    "integer": "int",
    "integer64": "long",
    "positiveInt": "int",
    "unsignedInt": "int",
    "decimal": "double",
    "date": "date",
    "dateTime": "datetime",
    "instant": "timestamp",
    "time": "time",
}


def _load_bundle() -> dict[str, Any]:
    """Read profiles-resources.json, extracting it from the archive if needed."""
    if not PROFILES.exists():
        if not ARCHIVE.exists():
            raise SystemExit(
                f"Neither {PROFILES} nor {ARCHIVE} found.\n"
                f"Download with:\n"
                f"  curl -L -o {ARCHIVE} https://hl7.org/fhir/R5/definitions.json.zip"
            )
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extract("profiles-resources.json", RAW)
    return json.loads(PROFILES.read_text(encoding="utf-8"))


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _snake(segment: str) -> str:
    """`birthDate` -> `birth_date`; matches how a flattener would render the segment."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", segment)
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()


def flatten_path(path: str, is_repeating: dict[str, bool]) -> str:
    """
    Render a FHIR element path as a flattened Avro-style name.

    Segments are snake_cased and joined with "_", except that a "__" boundary is emitted
    where the PARENT element repeats (max == "*"), which is exactly what an Avro flattener
    does for an array of records.

    >>> flatten_path("Patient.contact.name.family", {"Patient.contact": True})
    'patient_contact__name_family'
    """
    parts = path.split(".")
    out = _snake(parts[0])
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        # The RESOURCE ROOT is excluded deliberately. FHIR gives every root element
        # max="*" by convention (a Bundle may carry many of them), which is not the same
        # claim as "this field is a repeating child". Treating it as one marked all 4598
        # names as array boundaries and destroyed the signal the marker exists to carry.
        is_root = i == 1
        sep = "__" if (not is_root and is_repeating.get(parent)) else "_"
        out += sep + _snake(parts[i])
    return out


def _element_type(element: dict[str, Any]) -> str:
    types = element.get("type") or []
    if not types:
        return "unknown"
    code = types[0].get("code", "")
    return FHIR_TO_AVRO.get(code, "record" if code and code[0].isupper() else "string")


def build() -> dict[str, Any]:
    """Extract usable elements from the FHIR bundle."""
    bundle = _load_bundle()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    resources = 0

    for entry in bundle.get("entry", []):
        sd = entry.get("resource") or {}
        if sd.get("resourceType") != "StructureDefinition":
            continue
        if sd.get("kind") != "resource" or sd.get("abstract"):
            continue
        resources += 1

        elements = (sd.get("snapshot") or {}).get("element") or []

        # Which paths repeat -- needed to place the "__" array boundary.
        is_repeating = {e["path"]: e.get("max") == "*" for e in elements if e.get("path")}

        for element in elements:
            path = element.get("path") or ""
            if "." not in path:
                continue  # the resource root itself

            leaf = path.split(".")[-1]
            if leaf in BOILERPLATE:
                continue
            # A boilerplate ancestor makes the whole subtree boilerplate too.
            if any(seg in BOILERPLATE for seg in path.split(".")[1:-1]):
                continue
            if path in seen:
                continue

            short = _clean(element.get("short"))
            definition = _clean(element.get("definition"))
            comment = _clean(element.get("comment"))
            if not short and not definition:
                continue

            seen.add(path)
            rows.append(
                {
                    "path": path,
                    "flat": flatten_path(path, is_repeating),
                    "resource": path.split(".")[0],
                    "short": short,
                    "definition": definition,
                    "comment": comment,
                    "data_type": _element_type(element),
                    "repeats": element.get("max") == "*",
                }
            )

    return {"rows": rows, "resources": resources}


def write_benchmark(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Emit the canonical benchmark files.

    EVERY row becomes a dictionary entry (the retrieval pool). Only comment-bearing rows
    become queries, because the query side needs an independently-authored doc string to
    stay non-degenerate. Retrieving against the full pool keeps the distractor set honest.
    """
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "dictionary.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "id": f"fhir::{r['path']}",
                        # Target side: the glossary term and its definition. `comment` is
                        # deliberately absent here -- it belongs to the query side only.
                        "business_name": r["short"] or r["definition"][:80],
                        "logical_name": "",
                        "description": r["definition"],
                        "data_type": r["data_type"],
                        "domain": r["resource"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    queries = [r for r in rows if r["comment"]]
    with (OUT / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in queries:
            segments = r["flat"].replace("__", "_").split("_")
            f.write(
                json.dumps(
                    {
                        "id": f"q::fhir::{r['path']}",
                        "field_name": r["flat"],
                        "field_path": r["path"],
                        "data_type": r["data_type"],
                        "parent_path": " ".join(segments[:-1]),
                        "gold_id": f"fhir::{r['path']}",
                        # The query-side Avro `doc`. Independently authored from `definition`
                        # (measured Jaccard 0.08), which is what keeps this non-degenerate.
                        "doc": r["comment"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return len(rows), len(queries)


def main() -> None:
    print("Building FHIR R5 benchmark...")
    result = build()
    rows = result["rows"]
    n_entries, n_queries = write_benchmark(rows)

    depths: dict[int, int] = {}
    for r in rows:
        d = r["path"].count(".")
        depths[d] = depths.get(d, 0) + 1
    arrays = sum(1 for r in rows if "__" in r["flat"])

    print(f"  resources            {result['resources']}")
    print(f"  dictionary entries   {n_entries}")
    print(f"  queries (with doc)   {n_queries}")
    print(f"  array boundaries     {arrays} names contain '__'")
    print(f"  nesting depth        {dict(sorted(depths.items()))}")
    print(f"\nWrote -> {OUT}")

    print("\n  Examples:")
    for r in [x for x in rows if x["comment"]][:3]:
        print(f"    query : {r['flat']}")
        print(f"            doc={r['comment'][:70]!r}")
        print(f"    gold  : {r['short']!r}")
        print(f"            def={r['definition'][:70]!r}")
        print()


if __name__ == "__main__":
    main()
