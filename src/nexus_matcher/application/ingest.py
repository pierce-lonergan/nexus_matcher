"""
nexus_matcher.application.ingest | Layer: APPLICATION
One entry point for getting a glossary out of anything and into the index.

## Relationships
# DEPENDS_ON → domain/ports/vector_store :: index target
# DEPENDS_ON → domain/models/entities :: DictionaryEntry
# USED_BY    → presentation/cli :: `nexus-matcher sync`
# USED_BY    → application/use_cases/match_schema :: dictionary loading

## Attributes
# Security: Reads local files and DB connections the caller supplies; no network of its own
# Performance: Re-embeds only rows whose text CHANGED (content-hash diff)
# Reliability: A failed row is reported, not silently dropped

## The two problems this solves

**Reading.** A business glossary lives in a CSV, an Excel export, a Parquet extract, a
JSON dump or a database table, and every one of those needs a different reader with
different quirks. `load_entries()` dispatches on the source so callers never care.

Three of its options exist because the alternative FAILS SILENTLY, which is a different
thing from being inconvenient. `admit=` keeps drafts and retired terms out of the index --
they compete as real terms because they are real terms, and no threshold repairs that.
`value_delimiters=` says which character separates values inside a column, because one
file routinely uses ';' in one column and ',' in another and reading either the other way
round yields one plausible-looking value instead of an error. `metadata_columns=` and
`metadata_max_bytes=` declare and bound the pass-through plane, which is otherwise every
unmapped column of every row, held for the life of the index. `LoadReport` is how a caller
finds out what any of them did -- a filter is invisible from its result.

**Re-indexing.** Glossaries are edited constantly and re-embedding all of them on every
sync is the expensive part -- at ~600 texts/sec, a 100k-entry glossary costs about three
minutes each time, almost all of it recomputing vectors that did not change.

`sync()` hashes the text that actually gets embedded and re-embeds only rows whose hash
moved. A typical daily sync touches a handful of rows, so it drops from minutes to
milliseconds. Crucially the hash covers the EMBEDDED TEXT, not the whole row: changing a
`last_reviewed_by` column must not invalidate a vector, or incremental sync degrades to a
full rebuild the first time someone touches an audit column.

## Three lines

    from nexus_matcher import ingest

    index = ingest.build_index("glossary.xlsx")        # read, embed, index
    report = ingest.sync(index, "glossary.xlsx")       # later: only what changed
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus_matcher.domain.governance import (
    CLASSIFICATION_COLUMN_ALIASES,
    CODE_COLUMN_ALIASES,
    GovernanceVocabulary,
)
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.shared.types.base import DataType, ProtectionLevel

# Column names a real glossary is likely to use, in preference order. Matching is done on
# a normalised key (lowercased, non-alphanumerics stripped) so "Business Name",
# "business_name" and "BUSINESS-NAME" all land on the same field.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "identifier", "termid", "entryid", "key", "code", "uid", "guid"),
    "business_name": (
        "businessname",
        "term",
        "name",
        "businessterm",
        "label",
        "title",
        "attributename",
        "displayname",
        "glossaryterm",
    ),
    "logical_name": (
        "logicalname",
        "technicalname",
        "physicalname",
        "columnname",
        "fieldname",
        "attribute",
        "column",
    ),
    "definition": (
        "definition",
        "description",
        "businessdefinition",
        "meaning",
        "comment",
        "documentation",
        "doc",
        "notes",
        "summary",
    ),
    "data_type": ("datatype", "type", "sqltype", "logicaltype", "format"),
    "domain": (
        "domain",
        "subjectarea",
        "category",
        "businessdomain",
        "topic",
        "table",
        "entity",
        "schema",
    ),
    # The free-text tier, and the CONTROLLED code that derives it. Both alias tuples are
    # imported rather than written out here: `domain.governance` owns what a classification
    # and a protection code ARE, and `problems_with()` has to resolve the same two columns
    # out of a raw row, so a copy in this file would be a second notion of which column is
    # which -- exactly the split that once had the loader path rejecting files the ingest
    # path read without complaint.
    #
    # Ordered as they are deliberately. `map_columns` takes the first unclaimed column per
    # field, so a glossary carrying both "Classification" and "Classification Code" keeps
    # the existing meaning of the first.
    "protection_level": CLASSIFICATION_COLUMN_ALIASES,
    "governance_code": CODE_COLUMN_ALIASES,
}


def _norm_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# =============================================================================
# MULTI-VALUE COLUMNS
# =============================================================================

# Fields whose source column holds SEVERAL values in one cell.
#
# Both are ordered tuples, and neither reaches `to_searchable_text()` -- so populating
# them cannot move a content hash, cannot re-embed a row, and cannot make one process's
# vectors differ from another's.
#
# `synonyms` is the obvious third member and is deliberately absent. It is a `frozenset`,
# and `to_searchable_text()` extends the embedded text with it UNORDERED, so an entry with
# two or more synonyms produces a different string in every interpreter (str hashing is
# seeded per process). Measured: five processes, five different `content_hash` values for
# one entry with four synonyms. Populating it from here would mean a restart re-embeds the
# whole glossary and the vectors themselves stop being reproducible -- both properties this
# package documents and one of them the reason `sync()` exists. `_refuse_unordered_field`
# says so rather than letting a caller find out from a benchmark.
MULTI_VALUE_FIELDS: tuple[str, ...] = ("sample_values", "enum_values")
_UNORDERED_MULTI_VALUE_FIELDS: tuple[str, ...] = ("synonyms",)

# The separator used when a multi-value column is mapped without declaring one. A default
# is a guess, which is exactly the thing that fails silently here -- so the guess is the
# commonest convention and `_wrong_delimiter` checks it against the data rather than
# trusting it.
_DEFAULT_VALUE_DELIMITER = ","

# Separators a glossary export plausibly uses. Order is the tie-break order in
# `_wrong_delimiter`, so it is deliberate rather than incidental.
_DELIMITER_CANDIDATES: tuple[str, ...] = (";", ",", "|", "\t")

# Below this many non-empty cells a column carries no evidence of a convention, and
# refusing on it would fire on fixtures and pilots -- which is how a good check gets
# switched off for good.
_DELIMITER_MIN_CELLS = 3
# The signature of a wrong separator: the declared one splits almost nothing, while
# another one is present nearly everywhere.
_DELIMITER_UNSPLIT_SHARE = 0.95
_DELIMITER_CANDIDATE_SHARE = 0.60


def _split_values(text: str, delimiter: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(delimiter) if part.strip())


def _wrong_delimiter(cells: Sequence[str], delimiter: str) -> str | None:
    """
    The separator a column appears to actually use, when it is not the declared one.

    Returns None when there is no case to answer -- too few cells to judge, or the
    declared separator is doing work.

    Only this shape is detected, and that is a deliberate narrowing. The other pathology
    worth refusing -- a value list shredded into single characters -- cannot arise from
    `str.split` with a non-empty separator at all, and IS refused, exactly, at
    configuration time. A heuristic over the data cannot separate a shredded string from a
    legitimate list of single-character codes: `0,1,2,3` and a shredded `0123` are the same
    bytes. Refusing the empty separator catches the real cause; guessing at the symptom
    would refuse real flag and grade columns.
    """
    if len(cells) < _DELIMITER_MIN_CELLS:
        return None
    unsplit = sum(1 for cell in cells if delimiter not in cell)
    if unsplit / len(cells) < _DELIMITER_UNSPLIT_SHARE:
        return None
    best: str | None = None
    best_share = _DELIMITER_CANDIDATE_SHARE
    for candidate in _DELIMITER_CANDIDATES:
        if candidate == delimiter:
            continue
        share = sum(1 for cell in cells if candidate in cell) / len(cells)
        if share >= best_share:
            best, best_share = candidate, share
    return best


# =============================================================================
# THE PASS-THROUGH METADATA PLANE
# =============================================================================

# Keys the LOADER writes into `source_metadata`, as opposed to columns the caller's
# glossary supplied. Published because whatever emits the plane has to be able to tell the
# two apart, and a second hand-maintained list of them in the presentation layer is how
# they drift.
#
# They are also the keys truncation may never drop: `governance_code_raw` is the ONLY place
# a rejected code token survives, and `governance_problems` is the evidence for whoever
# fixes the source file. Discarding either to make room for a spreadsheet column would
# destroy the reason a row was refused.
METADATA_RESERVED_KEYS: tuple[str, ...] = (
    "governance_raw",
    "governance_code_raw",
    "governance_problems",
    "metadata_truncated",
)
_RESERVED_KEY_SET = frozenset(METADATA_RESERVED_KEYS)

# Per-entry cap on the pass-through plane, in bytes of key + value.
#
# Chosen by measuring, not by rounding. A realistic enterprise pass-through payload --
# twelve short enrichment columns: two identifiers, a code, two flags, five sample values,
# a steward, a review date, a lifecycle token -- costs 330 bytes. The example pack in
# examples/governance/ has a 103-byte median. 1,024 is 3.1x the first and 10x the second.
#
# What the cap is FOR, measured at 100,000 entries with per-row distinct strings
# (tracemalloc, current after construction):
#
#     no metadata          61.0 MB
#     330-byte payload    140.4 MB
#     1,627-byte payload  278.8 MB   <- a whole spreadsheet row, including a rationale
#                                       column, raked in by an undeclared mapping
#
# So the plane is not free even when it is used well, and an undeclared mapping more than
# quadruples what the index costs to hold. The cap refuses the third row and leaves the
# second alone. Pass `metadata_max_bytes=None` to lift it, having measured your own rows.
#
# It is on by default, and that costs measurable time: sizing 30,000 five-key maps takes
# 29.1 ms against 1.8 ms for the lifted cap (median of five, extremes dropped), roughly
# +11% on a 30k-row CSV load. An ASCII fast path -- `len(text)` where `text.isascii()`,
# which is exact -- was implemented and measured at 23.5 ms against 21.8 ms for encoding
# unconditionally, so it is slower for short strings and is not here. The cost is the
# price of the bound, not an oversight.
METADATA_MAX_BYTES = 1024

# Upper bound on what the `metadata_truncated` marker itself costs, reserved out of the
# budget so a truncated entry still fits the cap it was truncated to meet.
_TRUNCATION_MARKER_BYTES = len("metadata_truncated") + 4


def _value_bytes(value: Any) -> int:
    return len(str(value).encode("utf-8"))


def metadata_bytes(metadata: Mapping[str, Any]) -> int:
    """
    Size of a pass-through map, as bytes of key plus rendered value.

    A proxy, and deliberately a cheap and stable one rather than a true memory figure:
    `sys.getsizeof` is interpreter-defined and shifts with string interning, so a cap
    expressed in it would mean different things on different builds. This counts what the
    row actually said, which is the thing a deployment can reason about.
    """
    return sum(_value_bytes(k) + _value_bytes(v) for k, v in metadata.items())


def _bound_metadata(metadata: dict[str, Any], cap: int | None) -> tuple[dict[str, Any], int]:
    """
    Trim a pass-through map to `cap` bytes, largest key first.

    Returns (map, keys_dropped). Dropping the largest first keeps the most keys, and the
    tie-break on key name makes the survivors identical run to run -- the plane must not
    be the thing that introduces non-deterministic output.

    Surviving keys keep their original order, so a truncated map is a subsequence of the
    untruncated one rather than a re-sorted version of it.
    """
    if cap is None:
        return metadata, 0
    sizes = {key: _value_bytes(key) + _value_bytes(value) for key, value in metadata.items()}
    total = sum(sizes.values())
    if total <= cap:
        return metadata, 0

    budget = cap - _TRUNCATION_MARKER_BYTES
    dropped: set[str] = set()
    for key in sorted(
        (k for k in sizes if k not in _RESERVED_KEY_SET),
        key=lambda k: (-sizes[k], k),
    ):
        if total <= budget:
            break
        total -= sizes[key]
        dropped.add(key)

    if not dropped:
        # Everything over the cap was reserved evidence, which is never dropped. The
        # entry keeps it and the cap is exceeded knowingly.
        return metadata, 0

    kept = {key: value for key, value in metadata.items() if key not in dropped}
    kept["metadata_truncated"] = len(dropped)
    return kept, len(dropped)


# =============================================================================
# ROW ADMISSION
# =============================================================================


def _admit_key(value: Any) -> str:
    """
    The form an admitted value is compared in: trimmed and case-folded.

    A trailing space in an export is not a policy decision, and refusing an entire
    glossary over one produces the same outcome as a broken file -- which is the failure
    this whole filter exists to make distinguishable.
    """
    return str("" if value is None else value).strip().casefold()


# How many distinct observed values a refusal quotes. Enough to recognise the vocabulary
# actually in the column, few enough that the message stays readable next to a 105k-row
# glossary's worth of statuses.
_OBSERVED_SHOWN = 5


def map_columns(columns: Sequence[str]) -> dict[str, str]:
    """
    Map source column names onto entry fields.

    Returns {field: source_column}. Only fields that were confidently matched appear, so
    the caller can tell what was found rather than receiving silent empty strings.

    >>> map_columns(["Term", "Business Definition", "Subject Area"])
    {'business_name': 'Term', 'definition': 'Business Definition', 'domain': 'Subject Area'}
    """
    normalised = {_norm_key(c): c for c in columns}
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for field_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            source = normalised.get(alias)
            if source is not None and source not in taken:
                mapping[field_name] = source
                taken.add(source)
                break
    return mapping


# =============================================================================
# READING
# =============================================================================


def _excel_merged_ranges(path: Path, sheet_name: str) -> list[tuple[int, int, int, int]]:
    """
    Merged cell ranges for one sheet as (min_row, min_col, max_row, max_col), 1-based.

    Read straight from the XLSX package rather than through openpyxl, for two reasons:
    a `read_only=True` worksheet does not expose `.merged_cells` at all, and dropping
    read-only mode to obtain it would pull an entire glossary into memory. The
    workbook -> rels -> sheet lookup below is the documented OPC layout, so it does not
    depend on openpyxl internals.

    Returns [] for anything unreadable -- a missing part, a legacy .xls, an encrypted
    package. Merge handling is an enhancement; failing to find merges must never stop a
    file from loading.
    """
    import zipfile
    from xml.etree import ElementTree as ET

    from openpyxl.utils.cell import range_boundaries

    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "p": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    r_id = f"{{{ns['r']}}}id"

    try:
        with zipfile.ZipFile(path) as z:
            book = ET.fromstring(z.read("xl/workbook.xml"))
            rel_id = next(
                (
                    s.get(r_id)
                    for s in book.findall("m:sheets/m:sheet", ns)
                    if s.get("name") == sheet_name
                ),
                None,
            )
            if rel_id is None:
                return []

            rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
            target = next(
                (
                    rel.get("Target")
                    for rel in rels.findall("p:Relationship", ns)
                    if rel.get("Id") == rel_id
                ),
                None,
            )
            if target is None:
                return []

            # Targets are relative to xl/ unless already absolute within the package.
            part = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            raw = z.read(part)

        refs = _merge_refs_from_sheet_xml(raw, ns)

        ranges = []
        for ref in refs:
            if not ref:
                continue
            min_col, min_row, max_col, max_row = range_boundaries(ref)
            ranges.append((min_row, min_col, max_row, max_col))
        return ranges
    except (KeyError, OSError, ET.ParseError, ValueError, zipfile.BadZipFile):
        return []


def _merge_refs_from_sheet_xml(raw: bytes, ns: dict[str, str]) -> list[str]:
    """
    Pull the merged-range refs out of a sheet part without building its element tree.

    A sheet's XML is almost entirely `<c>` cell elements -- 26 MB and ~330k elements for
    a 30k-row glossary -- while `mergeCells` is a handful of bytes near the end. Handing
    the whole part to `ET.fromstring` to reach it cost 455-720 ms MEASURED at 30k rows,
    which was ~15% of the wall-clock of an Excel load and was paid in full even by files
    with no merged cells at all. Slicing the one element out first costs 7-13 ms.

    `iterparse` with `.clear()` was tried and rejected: it avoids RETAINING the tree but
    still tokenises every cell, so it measured 473-511 ms -- no better than the full parse.

    Three paths, in order, chosen so the fast ones can never return a WRONG answer:

    1. The local name `mergeCells` does not appear anywhere in the bytes. No merge element
       can exist under any namespace prefix, so there are no merges. A raw `<` cannot
       occur in XML text or attribute content (it must be escaped `&lt;`), so this scan
       cannot be fooled by a cell that happens to contain the word.
    2. The unprefixed `<mergeCells` start tag is present -- the form Excel, openpyxl and
       every mainstream writer emit. Slice that element out and parse just it. Its
       children carry `ref` as an unprefixed attribute, so the fragment needs none of the
       root's namespace declarations to be read correctly.
    3. Anything else (a namespace-prefixed `<x:mergeCells>`, say) falls back to the
       original full parse. Rare, correct, and only that file pays for it.
    """
    from xml.etree import ElementTree as ET

    if raw.find(b"mergeCells") < 0:
        return []

    start = raw.find(b"<mergeCells")
    if start >= 0:
        end = raw.find(b"</mergeCells>", start)
        if end >= 0:
            fragment = raw[start : end + len(b"</mergeCells>")]
        else:
            # Self-closing `<mergeCells count="0"/>`, i.e. declared but empty.
            fragment = raw[start : raw.find(b">", start) + 1]
        return [child.get("ref", "") for child in ET.fromstring(fragment)]

    sheet = ET.fromstring(raw)
    return [m.get("ref", "") for m in sheet.findall("m:mergeCells/m:mergeCell", ns)]


def _dedupe_header(names: list[str]) -> list[str]:
    """
    Make column names unique, preserving order and first occurrence.

    Rows are built as `{header[i]: value}`, so two columns sharing a name silently
    collapse and the LAST one wins. A glossary with two "Notes" columns would lose the
    first outright; the same happens to a header merged across columns, which reads back
    as the same name twice. Suffixed names simply fail to match an alias, which surfaces
    as "column not found" rather than as wrong data.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _apply_merged_values(
    rows: Iterator[tuple],
    merges: list[tuple[int, int, int, int]],
    first_data_row: int = 2,
) -> Iterator[list]:
    """
    Propagate each merged cell's value across the range it visibly spans.

    Excel stores a merged block's value ONCE, in the top-left cell; every other cell in
    the block reads back as None. For a governance glossary that is a silent
    misclassification, not a cosmetic gap -- a "Restricted" label merged down C2:C3 gives

        Patient SSN -> RESTRICTED
        Patient MRN -> INTERNAL      (blank, so it takes the default)

    while the human author sees one label covering both rows and reasonably believes the
    file says RESTRICTED twice. Downgrading a restricted field is the exact failure this
    library exists to prevent, so the visible value wins.

    Only cells inside a merged range are filled, and only when they are blank; an
    ordinary empty cell stays empty.

    The HEADER row (anything before `first_data_row`) is deliberately left alone. A merge
    there is a layout banner spanning several columns, not a value belonging to each of
    them, and copying it would produce duplicate column names -- which is a worse failure
    than a blank one. Vertical merges in the data are the unambiguous case, and the one
    that carries governance.
    """
    if not merges:
        yield from (list(r) for r in rows)
        return

    starts: dict[int, list[tuple[int, int, int, int]]] = {}
    for rng in merges:
        starts.setdefault(rng[0], []).append(rng)

    # Ranges currently spanning this row, each paired with its anchor value. Held as a
    # list rather than expanded per-cell so that a full-column merge costs one entry
    # instead of a million.
    active: list[tuple[tuple[int, int, int, int], Any]] = []

    for row_no, raw in enumerate(rows, start=1):
        row = list(raw)
        active = [(rng, v) for rng, v in active if rng[2] >= row_no]
        for rng in starts.get(row_no, ()):
            anchor_col = rng[1] - 1
            active.append((rng, row[anchor_col] if anchor_col < len(row) else None))
        # Anchors are still collected above for header-row merges, so a range starting in
        # the header keeps feeding the data rows underneath it.
        if row_no >= first_data_row:
            for rng, value in active:
                if value is None:
                    continue
                for col in range(rng[1], rng[3] + 1):
                    if col - 1 < len(row) and row[col - 1] is None:
                        row[col - 1] = value
        yield row


def _read_tabular(source: str | Path, **kwargs: Any) -> tuple[list[dict], list[str]]:
    """
    Read any tabular file into rows + column names.

    Args:
        source: Path to a csv/tsv/txt/xlsx/xlsm/xls/json/jsonl/parquet file.
        **kwargs: `header_row` (0-based index of the header, for spreadsheets that open
            with a title banner above it), plus per-format options -- `delimiter` and
            `encoding` for delimited text, `sheet` for Excel.
    """
    path = Path(source)
    suffix = path.suffix.lower()
    # Glossary exports very often carry a title//export-date banner above the real header.
    header_row = int(kwargs.pop("header_row", 0) or 0)

    if suffix in (".csv", ".tsv", ".txt"):
        import csv

        delimiter = kwargs.pop("delimiter", "\t" if suffix == ".tsv" else ",")
        # Feed the FILE to csv, not text.splitlines().
        #
        # str.splitlines() breaks on , , -, ,   and   --
        # none of which are record separators in CSV. A vertical tab or U+2028 pasted in
        # from Word therefore split one row into two: the real entry's definition was
        # truncated and a PHANTOM entry was created, embedded, indexed, and returnable as
        # a top-1 match carrying the default INTERNAL classification. splitlines() also
        # destroys legitimate newlines inside quoted fields, mangling multi-line definitions.
        #
        # newline="" is required so the csv module handles quoted line breaks itself.
        # utf-8-sig strips the BOM Excel writes, which otherwise corrupts the first header.
        encoding = kwargs.pop("encoding", "utf-8-sig")
        with path.open(newline="", encoding=encoding) as handle:
            raw = csv.reader(handle, delimiter=delimiter)
            for _ in range(header_row):
                next(raw, None)
            header = _dedupe_header(next(raw, []))
            reader = csv.DictReader(
                handle,
                fieldnames=header,
                delimiter=delimiter,
                # Ragged rows land under a real string key rather than None, which would
                # otherwise become the literal "null" after a JSON round-trip.
                restkey="_extra_columns",
                restval="",
            )
            rows = [dict(r) for r in reader]
            return rows, header

    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Reading Excel needs openpyxl. Install with: pip install "nexus-matcher[loaders]"'
            ) from exc
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[kwargs["sheet"]] if kwargs.get("sheet") else wb.active
        # Merged cells read back as None everywhere except the top-left; see
        # _apply_merged_values for why that silently downgrades a classification.
        merges = _excel_merged_ranges(path, ws.title)
        # Merge fill starts on the row after the header, wherever the header turns out
        # to be -- a banner above it must not be treated as data.
        it = _apply_merged_values(
            ws.iter_rows(values_only=True), merges, first_data_row=header_row + 2
        )
        for _ in range(header_row):
            next(it, None)
        header = _dedupe_header([str(c) if c is not None else "" for c in next(it, [])])
        rows = [
            {header[i]: ("" if v is None else v) for i, v in enumerate(r) if i < len(header)}
            for r in it
        ]
        wb.close()
        return rows, header

    if suffix in (".json", ".jsonl", ".ndjson"):
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Unwrap a common envelope shape.
                for key in ("entries", "terms", "rows", "data", "items"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    data = [data]
        else:
            data = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        rows = [r for r in data if isinstance(r, dict)]
        columns = list({k for r in rows for k in r})
        return rows, columns

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading Parquet needs pyarrow. Install with: pip install pyarrow"
            ) from exc
        table = pq.read_table(path)
        return table.to_pylist(), list(table.column_names)

    raise ValueError(
        f"Unsupported source '{path.suffix}'. Supported: .csv .tsv .xlsx .json .jsonl "
        f".parquet, a SQL connection string, or an iterable of dicts."
    )


def _read_sql(connection: str, query: str) -> tuple[list[dict], list[str]]:
    """Read from any SQLAlchemy-addressable database."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading from a database needs SQLAlchemy. Install with: "
            'pip install "nexus-matcher[loaders]"'
        ) from exc
    engine = create_engine(connection)
    with engine.connect() as conn:
        result = conn.execute(text(query))
        columns = list(result.keys())
        rows = [dict(zip(columns, r, strict=True)) for r in result.fetchall()]
    return rows, columns


def read_source(
    source: str | Path | Iterable[dict],
    query: str | None = None,
    **kwargs: Any,
) -> tuple[list[dict], list[str]]:
    """
    Read ANY supported source into rows plus column names.

    Args:
        source: A file path, a SQLAlchemy connection string, or an iterable of dicts.
        query: SQL to run when `source` is a connection string. Defaults to selecting
            everything from a `glossary` table, which is only a guess -- pass it.
        **kwargs: Reader-specific options (`sheet`, `delimiter`, `encoding`).

    Returns:
        (rows, columns)
    """
    if not isinstance(source, (str, Path)):
        rows = [dict(r) for r in source]
        return rows, list({k for r in rows for k in r})

    text_source = str(source)
    # A connection string, not a path: "postgresql://", "sqlite:///", "mssql+pyodbc://".
    if "://" in text_source and not text_source.startswith("file://"):
        return _read_sql(text_source, query or "SELECT * FROM glossary")

    return _read_tabular(source, **kwargs)


# =============================================================================
# ENTRIES
# =============================================================================


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_governance(
    source: GovernanceVocabulary | str | Path | Mapping[str, Any] | Sequence[Any] | None,
) -> GovernanceVocabulary | None:
    """
    Coerce a `governance=` argument into a vocabulary, or None for "feature off".

    None and `GovernanceVocabulary.empty()` are NOT the same thing and the difference is
    load-bearing. None means no vocabulary was configured, so no code is validated and
    none is stored -- the behaviour every existing caller already has. `empty()` means a
    vocabulary WAS configured and declares nothing, so every code in the glossary is a
    code nobody defined, and strict loading says so instead of quietly indexing them.
    """
    if source is None or isinstance(source, GovernanceVocabulary):
        return source
    return GovernanceVocabulary.from_json(source)


# How many defective rows a refusal quotes before it summarises the rest. Ten is enough to
# see the shape of the problem -- one spelling, one column, one stale export -- and few
# enough that the message stays readable in a terminal and in a startup log.
_DEFECTS_SHOWN = 10


@dataclass
class LoadReport:
    """
    What a load admitted, refused and bounded.

    Exists because a row filter is invisible from its result. A glossary that comes back
    with 8,000 of 105,000 rows looks exactly like a glossary that IS 8,000 rows, and looks
    exactly like a filter aimed at the wrong column -- there is nothing in the returned
    list that distinguishes the three. The counts are the distinction.

    `load_entries(..., report=report)` fills one in place; `build_index` and `sync` attach
    one to the index as `index.load_report`. A report handed to a load is RESET first, so
    it always describes that load rather than accumulating across refreshes.
    """

    rows_read: int = 0
    admitted: int = 0
    refused: int = 0
    # {admission column: rows it refused}. A row refused by two columns is counted against
    # the first one that refused it, so the totals here sum to `refused` rather than
    # over-counting a row that fails everything.
    refused_by_column: dict[str, int] = field(default_factory=dict)
    # Rows with neither a business name nor a definition. Not a filter decision -- such a
    # row cannot be matched to and has always been skipped -- but counted here so it is
    # not mistaken for one.
    skipped_unmatchable: int = 0
    entries: int = 0
    # Entries whose pass-through map hit `metadata_max_bytes`, and the total keys dropped.
    metadata_truncated: int = 0
    metadata_keys_dropped: int = 0

    def reset(self) -> None:
        self.rows_read = 0
        self.admitted = 0
        self.refused = 0
        self.refused_by_column = {}
        self.skipped_unmatchable = 0
        self.entries = 0
        self.metadata_truncated = 0
        self.metadata_keys_dropped = 0

    def __str__(self) -> str:
        by_column = (
            "  (" + ", ".join(f"{c} {n}" for c, n in sorted(self.refused_by_column.items())) + ")"
            if self.refused_by_column
            else ""
        )
        truncated = (
            f"  !{self.metadata_truncated} metadata truncated" if self.metadata_truncated else ""
        )
        return (
            f"{self.rows_read} rows read  {self.admitted} admitted  "
            f"{self.refused} refused{by_column}  "
            f"{self.skipped_unmatchable} unmatchable  ={self.entries} entries{truncated}"
        )


# Reader options `load_entries` forwards. Anything else is a typo, and a typo that reaches
# the reader is silently discarded there -- which would make `admt={...}` a load with no
# filter at all, indistinguishable from a load that was never configured.
_READER_OPTIONS = frozenset({"sheet", "delimiter", "encoding", "header_row"})


def load_entries(
    source: str | Path | Iterable[dict],
    query: str | None = None,
    columns: dict[str, str] | None = None,
    id_prefix: str = "",
    governance: GovernanceVocabulary | str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    governance_strict: bool = True,
    admit: Mapping[str, Collection[str]] | None = None,
    value_delimiters: Mapping[str, str] | None = None,
    delimiter_strict: bool = True,
    metadata_columns: Sequence[str] | None = None,
    metadata_max_bytes: int | None = METADATA_MAX_BYTES,
    report: LoadReport | None = None,
    **kwargs: Any,
) -> list[DictionaryEntry]:
    """
    Read a source and turn it into DictionaryEntry objects.

    Args:
        source: File path, SQLAlchemy connection string, or iterable of dicts.
        query: SQL, when source is a connection string.
        columns: Explicit {field: source_column} overrides. Anything not given is
            inferred from the header via `map_columns`. The multi-value fields
            (`sample_values`, `enum_values`) are ONLY ever mapped explicitly -- inferring
            them would change the text existing glossaries embed.
        id_prefix: Prefix for generated ids, useful when pooling several glossaries.
        governance: The caller's controlled vocabulary -- a `GovernanceVocabulary`, a
            path to its JSON file, or the parsed document. None (the default) leaves
            governance unread: no code is validated, and none is attached.
        governance_strict: REFUSE the load when governance is defective: with a
            vocabulary configured, any row carrying an undefined code or a tier that
            contradicts its own code; with none configured, a source that HAS a
            protection-code column, which nothing can then read. Set False to load anyway
            -- the catalog still wins, so no entry inherits a contradicted tier, and each
            offending row carries its problems in
            `source_metadata['governance_problems']`.
        admit: {column: accepted values} -- only rows whose named column holds one of the
            accepted values become entries, all columns having to agree. Values are
            compared trimmed and case-folded. Without this, a glossary's drafts and
            retired terms are indexed alongside its approved ones, and they compete as
            real terms because they ARE real terms; no threshold repairs that. The counts
            land in `report`, because a filter that quietly drops nine rows in ten is
            indistinguishable from a broken file.
        value_delimiters: {field: separator} for the multi-value fields -- one file
            routinely separates one such column with ';' and another with ',', and reading
            either with the other's separator yields one value per row containing every
            element, which indexes and matches and is simply wrong. Defaults to ',' for a
            mapped multi-value column that declares nothing.
        delimiter_strict: REFUSE the load when a declared separator does not appear in a
            column that another separator plainly does. Set False to load the column as
            the loader read it, having decided it really is single-valued.
        metadata_columns: The DECLARED pass-through allow-list: exactly these source
            columns reach `source_metadata`, whether or not they also feed a domain
            field. Omit it and the existing behaviour stands -- every unmapped column is
            carried -- which is convenient and unbounded in the number of keys.
        metadata_max_bytes: Per-entry cap on the pass-through map, in bytes of key plus
            rendered value; None lifts it. Over the cap, the largest keys are dropped
            (never the loader's own -- see METADATA_RESERVED_KEYS) and the entry carries
            `metadata_truncated` with the count. See METADATA_MAX_BYTES for how the
            default was measured.
        report: Filled in place with what this load admitted, refused and bounded. Reset
            first, so it describes this load and not the last one too.
        **kwargs: Passed to the reader (`sheet`, `delimiter`, `encoding`, `header_row`).
            Note that `delimiter` is the reader's -- the character between COLUMNS. The
            character between values INSIDE a column is `value_delimiters`.

    Returns:
        Entries, skipping rows with no business name AND no definition -- such a row
        cannot be matched to and would only add a distractor.

    Raises:
        ValueError: if neither a business name nor a definition column can be identified,
            which means the mapping is wrong rather than the data being sparse; if `admit`
            or `metadata_columns` names a column the source does not have, or `admit`
            refuses every row; under `delimiter_strict`, if a declared separator
            contradicts the data; or, under `governance_strict`, if any row's governance
            is defective or the source carries a protection-code column with no vocabulary
            to interpret it.
    """
    _check_reader_options(kwargs)
    report = report or LoadReport()
    report.reset()

    rows, header = read_source(source, query=query, **kwargs)
    mapping = {**map_columns(header), **(columns or {})}
    vocabulary = load_governance(governance)

    unordered = [f for f in _UNORDERED_MULTI_VALUE_FIELDS if f in mapping]
    if unordered:
        raise _unordered_field_error(unordered[0], mapping[unordered[0]])

    delimiters = _resolve_delimiters(value_delimiters, mapping)

    if "business_name" not in mapping and "definition" not in mapping:
        raise ValueError(
            f"Could not find a business-name or definition column in {header}.\n"
            f"Pass an explicit mapping, e.g.\n"
            f"    columns={{'business_name': 'Term', 'definition': 'Meaning'}}"
        )

    # A protection-code column that nobody is configured to read.
    #
    # The silence here is circular, which is why neither layer below can catch it: a code
    # is attached only when a vocabulary is configured, and every consumer refuses codes it
    # cannot resolve -- so with NO vocabulary there are no codes, nothing to refuse, and a
    # glossary whose header plainly says `protection_class` yields entries carrying nothing
    # at all, which a consumer cannot distinguish from a glossary that declares no classes.
    #
    # The check belongs here and not in one caller. The HTTP app had it, and paid a second
    # full read of the glossary to run it; `mapping` is already built, so here it is a dict
    # lookup. `governance_strict` is the opt-out, and deliberately the same one as for a
    # defective row: both say "the governance in this file is not being validated and I am
    # loading it anyway".
    if vocabulary is None and governance_strict and mapping.get("governance_code"):
        raise _unread_code_column_error(source, mapping["governance_code"])

    declared_metadata = _resolve_metadata_columns(metadata_columns, header, source)

    admitted_rows = _admit_rows(rows, admit, header, source, report)
    if delimiter_strict:
        _check_delimiters(admitted_rows, delimiters, mapping)

    entries: list[DictionaryEntry] = []
    used_ids: set[str] = set()
    # The defects the refusal will SHOW, capped, plus a count of all of them.
    #
    # This used to be every defect string, formatted and held whatever the caller asked
    # for, though the refusal quotes ten and nothing else ever reads the list. Under
    # `governance_strict=False` -- the escape hatch the refusal below recommends -- not one
    # of them was read at all. Measured over 30,000 rows carrying an undefined code: peak
    # traced allocation 39.5 -> 33.2 MB. Wall clock is a wash (median 172 -> 178 ms over
    # five runs, ranges overlapping), so this buys memory, not speed.
    defects: list[str] = []
    defect_count = 0
    # Whether any row's code was REJECTED, as opposed to merely contradicting its own tier.
    # Only the first kind sends the reader looking for the list of declared codes; on a
    # glossary full of contradictions that list would push the two tiers that actually
    # disagree off the top of the message.
    rejected_a_code = False
    # Hoisted: this is a property of the MAPPING, not of the row. Rebuilding it inside
    # the comprehension below meant one set construction per row -- 30k of them on a 30k
    # glossary, measured at 3x the cost of the whole source_metadata build.
    mapped_columns = set(mapping.values())
    for row_number, row in enumerate(admitted_rows, start=1):
        # `row` is bound as a default argument rather than captured. Capturing the loop
        # variable works only while the closure is called inside the same iteration --
        # true today, and silently wrong the moment anyone defers a call.
        def get(fieldname: str, _row: dict = row) -> str:
            column = mapping.get(fieldname)
            return _as_text(_row.get(column)) if column else ""

        business_name = get("business_name")
        definition = get("definition")
        if not business_name and not definition:
            report.skipped_unmatchable += 1
            continue

        multi: dict[str, tuple[str, ...]] = {
            field_name: _split_values(get(field_name), delimiter)
            for field_name, delimiter in delimiters.items()
        }

        governance_code, problems, rejected = _row_governance(
            vocabulary, row, get("governance_code")
        )
        if problems:
            defect_count += 1
            rejected_a_code = rejected_a_code or rejected
            if governance_strict and len(defects) < _DEFECTS_SHOWN:
                defects.append(
                    f"row {row_number} ({business_name or definition!r}): " + "; ".join(problems)
                )

        entry_id = _entry_id(get("id"), business_name, get("logical_name"), id_prefix, used_ids)
        used_ids.add(entry_id)

        metadata, dropped_keys = _row_metadata(
            row,
            mapped_columns,
            declared_metadata,
            metadata_max_bytes,
            get("protection_level") if mapping.get("protection_level") else None,
            get("governance_code") if mapping.get("governance_code") else None,
            problems,
        )
        if dropped_keys:
            report.metadata_truncated += 1
            report.metadata_keys_dropped += dropped_keys

        entries.append(
            DictionaryEntry(
                id=entry_id,
                business_name=business_name,
                logical_name=get("logical_name"),
                definition=definition,
                data_type=_coerce_type(get("data_type")),
                domain=get("domain"),
                protection_level=_coerce_protection(get("protection_level")),
                governance_code=governance_code,
                sample_values=multi.get("sample_values", ()),
                enum_values=multi.get("enum_values", ()),
                is_enum=bool(multi.get("enum_values")),
                source_metadata=metadata,
            )
        )

    if defect_count and governance_strict:
        # REFUSE the load rather than return the good rows.
        #
        # Returning a partial glossary is the failure mode this check exists to stop: the
        # rows that vanished are exactly the rows whose governance was wrong, so the
        # caller indexes a dictionary that looks healthy, matches a field against it, and
        # inherits nothing where they should have inherited a class. A defective
        # vocabulary reference is a source-data bug and it has to be fixed in the source.
        raise _defective_governance_error(defect_count, defects, vocabulary, rejected_a_code)

    report.entries = len(entries)
    return entries


def _check_reader_options(kwargs: Mapping[str, Any]) -> None:
    """
    Refuse an option `load_entries` does not understand, instead of forwarding it.

    Everything not named in the signature went to the reader, and the reader pops what it
    recognises and never looks at the rest -- so a misspelled option was a silent no-op,
    and the load reported success having done something other than what was asked.
    Recorded as NM-0032, where `sheet_name=` (the pandas spelling) read the wrong sheet of
    a two-sheet workbook and indexed a glossary of retired terms without a word.
    """
    unsupported = sorted(set(kwargs) - _READER_OPTIONS)
    if unsupported:
        raise ValueError(
            f"load_entries does not take {', '.join(repr(k) for k in unsupported)}. "
            f"Reader options are {', '.join(sorted(_READER_OPTIONS))}; anything else is "
            f"discarded by the reader without a word, so a misspelled option would read as "
            f"a load that was never configured."
        )


def _row_metadata(
    row: Mapping[str, Any],
    mapped_columns: set[str],
    declared_metadata: tuple[str, ...] | None,
    cap: int | None,
    governance_raw: str | None,
    governance_code_raw: str | None,
    problems: list[str],
) -> tuple[dict[str, Any], int]:
    """
    One row's pass-through map, bounded. Returns (map, keys dropped to fit the cap).

    `governance_raw` and `governance_code_raw` are None when the source has no such
    column, and a string -- possibly empty -- when it has one. The distinction is the
    point: an empty cell in a classification column is a row nobody classified, which is
    not the same as a glossary with no classification column at all.
    """
    # THE PASS-THROUGH PLANE, and the loader is what decides what enters it.
    #
    # Declared, the allow-list IS the plane: exactly the named columns, in the order
    # they were named, whether or not they also feed a domain field. Declaring a
    # column is a statement that the deployment wants it back, and the protection
    # class is the standing proof that "it is mapped, so you have it" is wrong -- the
    # enum is lossy, which is why `governance_raw` exists.
    #
    # Undeclared, the historical behaviour stands: everything unmapped is preserved.
    # Convenient, and unbounded in the number of keys, which is what
    # `metadata_max_bytes` is for.
    #
    # PLUS the raw governance strings either way. The classification column is usually
    # the whole reason for matching -- an earlier version mapped it (which excluded it
    # from here) and then never used it, so the value a caller most needs was silently
    # dropped. The enum is lossy by design (an org's "Highly Restricted" collapses to
    # RESTRICTED), so the original text is kept alongside it.
    #
    # `governance_code_raw` is the same guarantee for the controlled code, and it is
    # the ONLY place a rejected token survives: it is evidence for whoever fixes the
    # source file, sitting under a key that cannot be mistaken for an accepted class.
    metadata = {
        **(
            {k: v for k, v in row.items() if k not in mapped_columns}
            if declared_metadata is None
            else {c: row[c] for c in declared_metadata if c in row}
        ),
        **({"governance_raw": governance_raw} if governance_raw is not None else {}),
        **({"governance_code_raw": governance_code_raw} if governance_code_raw is not None else {}),
        **({"governance_problems": problems} if problems else {}),
    }
    return _bound_metadata(metadata, cap)


def _row_governance(
    vocabulary: GovernanceVocabulary | None,
    row: Mapping[str, Any],
    code_text: str,
) -> tuple[str | None, list[str], bool]:
    """
    One row's governance, resolved through the caller's vocabulary and NEVER from the row.

    Returns (canonical code, problems, code was rejected).

    The code stored is the CANONICAL one the vocabulary declares, so a legacy spelling is
    stored as the code it maps to and a token the vocabulary does not define is stored as
    nothing at all. Storing an unknown code would leave a field carrying a label nobody
    defined, which reads as governance and is not.

    The third member distinguishes a REJECTED code from a row that merely contradicts its
    own tier: only the first sends a reader looking for the list of declared codes. It is
    derived from the lookup rather than by reading the message text, so rewording a problem
    cannot silently change what a refusal prints.
    """
    if vocabulary is None:
        return None, [], False
    problems = vocabulary.problems_with(row)
    protection_class = vocabulary.get(code_text)
    code = protection_class.code if protection_class is not None else None
    return code, problems, protection_class is None


def _entry_id(
    raw_id: str,
    business_name: str,
    logical_name: str,
    id_prefix: str,
    used_ids: set[str],
) -> str:
    """The entry's id: the source's own if it has one, else derived from content."""
    if raw_id:
        entry_id = f"{id_prefix}{raw_id}"
    else:
        # Derive a STABLE id from identifying content, never from row position.
        #
        # Positional ids ("row-0", "row-1", ...) look harmless and quietly destroy
        # incremental sync: deleting one row of ten renumbers every row after it, so
        # the diff sees 9 updates and 1 removal and re-embeds the whole glossary --
        # measured, 9 of 9 rows re-embedded for a single deletion. Any source without
        # an id column, which is most spreadsheets, would rebuild on every edit.
        key = f"{business_name}{logical_name}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
        entry_id = f"{id_prefix}{digest}"
        # Genuine duplicates (same name AND same technical name) still need distinct
        # ids, so disambiguate deterministically by order of appearance.
        if entry_id in used_ids:
            n = 2
            while f"{entry_id}-{n}" in used_ids:
                n += 1
            entry_id = f"{entry_id}-{n}"
    return entry_id


def _subject_of(source: str | Path | Iterable[dict[str, Any]]) -> str:
    """
    A name for the source, for a refusal a human reads at deployment time.

    "some glossary has a column" sends an operator looking through however many they
    configured. An iterable of dicts has no name, and its repr would be the whole glossary.
    """
    return str(source) if isinstance(source, (str, Path)) else "this glossary"


def _resolve_delimiters(
    value_delimiters: Mapping[str, str] | None,
    mapping: Mapping[str, str],
) -> dict[str, str]:
    """
    {multi-value field: separator}, for the multi-value fields that were actually mapped.

    Refuses a declaration naming something that is not a multi-value field. A declaration
    the loader silently ignores is worse than no declaration: the caller believes a column
    is being split and it is not, and the entry carries one value that looks like data.
    """
    declared = dict(value_delimiters or {})
    unknown = sorted(set(declared) - set(MULTI_VALUE_FIELDS))
    if unknown:
        raise ValueError(
            f"value_delimiters names {', '.join(repr(f) for f in unknown)}, which "
            f"{'are' if len(unknown) > 1 else 'is'} not multi-value. The multi-value "
            f"fields are {', '.join(MULTI_VALUE_FIELDS)}."
        )
    resolved: dict[str, str] = {}
    for field_name in MULTI_VALUE_FIELDS:
        if field_name not in mapping:
            continue
        delimiter = declared.get(field_name, _DEFAULT_VALUE_DELIMITER)
        if not delimiter:
            # The one input that shreds a value into single characters, and the only way
            # `str.split` can produce that outcome at all. Python refuses it too, with a
            # message ("empty separator") that names neither the column nor the field.
            raise ValueError(
                f"value_delimiters[{field_name!r}] is empty. An empty separator splits a "
                f"value into single characters -- str.split refuses it outright -- so a "
                f"column of terms would become a column of letters. Declare the character "
                f"between values, e.g. value_delimiters={{{field_name!r}: ';'}}."
            )
        resolved[field_name] = delimiter
    return resolved


def _unordered_field_error(field_name: str, column: str) -> ValueError:
    """The refusal for mapping a multi-value field whose container has no order."""
    return ValueError(
        f"columns maps {field_name!r} to {column!r}, and this loader cannot populate it.\n"
        f"DictionaryEntry.{field_name} is a frozenset, and DictionaryEntry."
        f"to_searchable_text() emits it unordered -- so an entry with two or more values "
        f"embeds a different string, and therefore hashes and vectorises differently, in "
        f"every process. Measured: five interpreters, five content hashes for one entry "
        f"with four synonyms. Every restart would re-embed the whole glossary, and no two "
        f"builds would agree on a vector.\n"
        f"Map the column to an ordered multi-value field "
        f"({', '.join(MULTI_VALUE_FIELDS)}), or carry it as pass-through metadata with "
        f"metadata_columns=[{column!r}]."
    )


def _resolve_admission(
    admit: Mapping[str, Collection[str]],
    header: Sequence[str],
    source: str | Path | Iterable[dict[str, Any]],
) -> tuple[tuple[str, frozenset[str], tuple[str, ...]], ...]:
    """
    The admission filter, normalised, with both ways of writing it wrong refused here.

    A column the file does not have would refuse every row; an accepted set with nothing
    in it would too. Both produce an empty glossary, which is indistinguishable downstream
    from a glossary of nothing but drafts.

    Each entry is (column, comparison keys, the values AS THE CALLER SPELLED THEM). The
    third member exists only for refusals: quoting a case-folded 'published' back at
    someone who wrote 'Published' invites them to hunt for a bug in their own casing,
    which is the one thing this filter does not care about.
    """
    missing = [column for column in admit if column not in header]
    if missing:
        raise ValueError(
            f"admit names {', '.join(repr(c) for c in missing)}, which "
            f"{_subject_of(source)} does not have. Its columns are "
            f"{', '.join(repr(c) for c in header)}.\n"
            f"Every row would be refused, and an empty glossary reads downstream exactly "
            f"like a glossary with nothing approved in it."
        )
    resolved: list[tuple[str, frozenset[str], tuple[str, ...]]] = []
    for column, values in admit.items():
        accepted = frozenset(_admit_key(v) for v in values)
        if not accepted:
            raise ValueError(
                f"admit[{column!r}] accepts nothing, so every row is refused. Name the "
                f"values that ARE admitted, e.g. admit={{{column!r}: {{'Approved'}}}}."
            )
        resolved.append((column, accepted, tuple(sorted(str(v) for v in values))))
    return tuple(resolved)


def _admit_rows(
    rows: list[dict],
    admit: Mapping[str, Collection[str]] | None,
    header: Sequence[str],
    source: str | Path | Iterable[dict[str, Any]],
    report: LoadReport,
) -> list[dict]:
    """
    The rows that will become entries, and the counts for the ones that will not.

    Deliberately its own pass rather than a `continue` inside the build loop. The
    delimiter check has to judge a column on the cells that will actually be INDEXED -- a
    draft row's punctuation is not evidence about the approved ones -- and the counts have
    to be complete before the first entry is built, because the refusal for "this filter
    admitted nothing" is only useful if it can also say what the column actually held.

    A row that fails several columns is counted against the FIRST one that refused it, so
    the per-column tallies sum to the total rather than over-counting a row that fails
    everything.
    """
    report.rows_read = len(rows)
    if admit is None:
        report.admitted = len(rows)
        return rows

    accepted = _resolve_admission(admit, header, source)
    admitted: list[dict] = []
    for row in rows:
        refused_by = next(
            (
                column
                for column, keys, _spelled in accepted
                if _admit_key(row.get(column)) not in keys
            ),
            None,
        )
        if refused_by is None:
            admitted.append(row)
        else:
            report.refused += 1
            report.refused_by_column[refused_by] = report.refused_by_column.get(refused_by, 0) + 1

    if rows and not admitted:
        raise _nothing_admitted_error(source, rows, accepted)
    report.admitted = len(admitted)
    return admitted


def _check_delimiters(
    rows: Sequence[Mapping[str, Any]],
    delimiters: Mapping[str, str],
    mapping: Mapping[str, str],
) -> None:
    """Refuse a declared separator that the column's own contents contradict."""
    for field_name, delimiter in delimiters.items():
        column = mapping[field_name]
        cells = [text for row in rows if (text := _as_text(row.get(column)))]
        actual = _wrong_delimiter(cells, delimiter)
        if actual is not None:
            raise _wrong_delimiter_error(column, field_name, delimiter, actual, cells)


def _nothing_admitted_error(
    source: str | Path | Iterable[dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    accepted: Sequence[tuple[str, frozenset[str], tuple[str, ...]]],
) -> ValueError:
    """
    The refusal for a filter that admitted no row at all.

    Quotes what the column actually holds, because the bug is almost always that the
    deployment named the neighbouring vocabulary -- "Published" against a file that says
    "Approved" -- and no amount of staring at the configuration reveals that. Only the
    data does.
    """
    lines = []
    for column, _keys, spelled in accepted:
        tally: dict[str, int] = {}
        for row in rows:
            seen = _as_text(row.get(column))
            tally[seen] = tally.get(seen, 0) + 1
        top = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[:_OBSERVED_SHOWN]
        lines.append(
            f"  {column!r} accepts {list(spelled)}; the column holds "
            + ", ".join(f"{value!r} x{count}" for value, count in top)
            + (f" (+{len(tally) - len(top)} more)" if len(tally) > len(top) else "")
        )
    return ValueError(
        f"Row admission refused all {len(rows)} rows of {_subject_of(source)}.\n"
        + "\n".join(lines)
        + "\nEither the accepted values are wrong, or this is not the file you think it "
        "is. Returning an empty glossary would look identical to a glossary with nothing "
        "approved in it."
    )


def _wrong_delimiter_error(
    column: str,
    field_name: str,
    delimiter: str,
    actual: str,
    cells: Sequence[str],
) -> ValueError:
    """
    The refusal for a separator the data contradicts.

    This is the failure the whole per-column delimiter feature exists for: one file
    routinely separates one multi-value column with ';' and another with ',', and reading
    either with the other's separator raises nothing. It yields ONE value per row
    containing every element -- which embeds, indexes, and comes back as a match. There is
    no downstream check that can notice, because the result is well-formed. Only the
    loader is in a position to see it.
    """
    unsplit = sum(1 for cell in cells if delimiter not in cell)
    present = sum(1 for cell in cells if actual in cell)
    return ValueError(
        f"Column {column!r} is mapped to {field_name!r} with delimiter {delimiter!r}, "
        f"which appears in none of {unsplit} of its {len(cells)} non-empty cells -- while "
        f"{actual!r} appears in {present} of them.\n"
        f"Splitting on {delimiter!r} would give one value per row containing every "
        f"element. That indexes and matches and is simply wrong, and nothing downstream "
        f"can tell.\n"
        f"Pass value_delimiters={{{field_name!r}: {actual!r}}}, or delimiter_strict=False "
        f"if this column really does hold one value."
    )


def _resolve_metadata_columns(
    metadata_columns: Sequence[str] | None,
    header: Sequence[str],
    source: str | Path | Iterable[dict[str, Any]],
) -> tuple[str, ...] | None:
    """
    The declared pass-through allow-list, or None for "carry every unmapped column".

    A declared column the source does not have is refused rather than skipped. The
    declaration is the deployment's statement of what it is carrying; one that names
    nothing carries nothing, and the response side has no way to tell that from a column
    that was empty on every row.
    """
    if metadata_columns is None:
        return None
    declared = tuple(metadata_columns)
    missing = [column for column in declared if column not in header]
    if missing:
        raise ValueError(
            f"metadata_columns names {', '.join(repr(c) for c in missing)}, which "
            f"{_subject_of(source)} does not have. Its columns are "
            f"{', '.join(repr(c) for c in header)}.\n"
            f"A declared pass-through column that carries nothing is indistinguishable, "
            f"downstream, from one that was empty on every row."
        )
    return declared


def _unread_code_column_error(
    source: str | Path | Iterable[dict[str, Any]], code_column: str
) -> ValueError:
    """The refusal for a glossary that carries protection codes nobody can interpret."""
    subject = _subject_of(source)
    return ValueError(
        f"{subject} has a protection-code column ({code_column!r}) and no vocabulary to "
        f"interpret it, so every entry would come back carrying no class -- "
        f"indistinguishable, downstream, from a glossary that declares none.\n"
        f"Pass governance= the JSON file that declares those codes, or "
        f"governance_strict=False to read that column as plain metadata."
    )


def _defective_governance_error(
    defect_count: int,
    shown: list[str],
    vocabulary: GovernanceVocabulary | None,
    rejected_a_code: bool,
) -> ValueError:
    """
    The refusal for a glossary whose rows disagree with the caller's vocabulary.

    THE DECLARED CODES ARE NAMED HERE, ONCE. This is the one part of a load that runs
    exactly once, which is the whole reason the list belongs in it: `problems_with()` used
    to interpolate the full list into every rejected row's message, so this refusal reached
    ~70,000 characters -- one list, ten times over -- and a reader had to scroll past nine
    copies of it to reach the tenth defect. `problems_with()` now names the offending token
    and the SIZE of the vocabulary; the vocabulary itself is here. Measured on the same
    30,000-row refusal against a nine-class vocabulary, the message is now 1,991 characters.
    """
    more = f"\n  ... and {defect_count - len(shown)} more" if defect_count > len(shown) else ""
    declared = ""
    if rejected_a_code and vocabulary is not None:
        codes = sorted(vocabulary.codes)
        if codes:
            declared = f"\nThe vocabulary declares {len(codes)} code(s): " + ", ".join(codes)
    return ValueError(
        f"{defect_count} row(s) carry defective governance and the load was refused.\n  "
        + "\n  ".join(shown)
        + more
        + declared
        + "\nFix the source, or pass governance_strict=False to load anyway -- the "
        "catalog still wins, so no entry inherits a contradicted tier, and each "
        "offending row carries its problems in source_metadata['governance_problems']."
    )


_TYPE_WORDS = {
    "string": DataType.STRING,
    "str": DataType.STRING,
    "text": DataType.STRING,
    "varchar": DataType.STRING,
    "char": DataType.STRING,
    "nvarchar": DataType.STRING,
    "int": DataType.INTEGER,
    "integer": DataType.INTEGER,
    "smallint": DataType.INTEGER,
    "bigint": DataType.LONG,
    "long": DataType.LONG,
    "float": DataType.FLOAT,
    "real": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "decimal": DataType.DECIMAL,
    "numeric": DataType.DECIMAL,
    "bool": DataType.BOOLEAN,
    "boolean": DataType.BOOLEAN,
    "bit": DataType.BOOLEAN,
    "date": DataType.DATE,
    "datetime": DataType.TIMESTAMP,
    "timestamp": DataType.TIMESTAMP,
    "uuid": DataType.UUID,
    "json": DataType.JSON,
    "bytes": DataType.BYTES,
}


# Vocabulary an organisation actually uses for classification, mapped onto the enum.
# Deliberately substring-based: real glossaries say "Confidential - Internal Use Only" or
# "PII / Sensitive", not a bare enum name.
_PROTECTION_WORDS: tuple[tuple[str, ProtectionLevel], ...] = (
    ("restricted", ProtectionLevel.RESTRICTED),
    ("secret", ProtectionLevel.RESTRICTED),
    ("highly confidential", ProtectionLevel.RESTRICTED),
    ("pii", ProtectionLevel.PII),
    ("personal", ProtectionLevel.PII),
    ("phi", ProtectionLevel.PII),
    ("sensitive", ProtectionLevel.CONFIDENTIAL),
    ("confidential", ProtectionLevel.CONFIDENTIAL),
    ("internal", ProtectionLevel.INTERNAL),
    ("public", ProtectionLevel.PUBLIC),
    ("open", ProtectionLevel.PUBLIC),
)


# NEGATED forms, checked BEFORE the positive table.
#
# Substring matching alone inverts the meaning of real financial and defence vocabulary:
# "Non-Public" and "NPI - Nonpublic" contain "public", and "Unrestricted" contains
# "restricted", so a naive scan mapped GLBA nonpublic personal information to PUBLIC --
# the weakest level in the enum, and precisely backwards. Word boundaries alone fix
# "unrestricted"; the spaced variants are needed because separator normalisation runs first.
_NEGATED_PROTECTION: tuple[tuple[str, ProtectionLevel], ...] = (
    ("nonpublic", ProtectionLevel.CONFIDENTIAL),
    ("non public", ProtectionLevel.CONFIDENTIAL),
    ("not for public", ProtectionLevel.CONFIDENTIAL),
    ("npi", ProtectionLevel.CONFIDENTIAL),
    ("unrestricted", ProtectionLevel.PUBLIC),
    ("un restricted", ProtectionLevel.PUBLIC),
    ("not pii", ProtectionLevel.INTERNAL),
    ("no pii", ProtectionLevel.INTERNAL),
    ("non pii", ProtectionLevel.INTERNAL),
)


# Built ONCE, at import, rather than per call.
#
# The token list is constant, so the pattern for each entry is constant too -- but
# building it inside the loop meant every row paid `re.escape` plus a cache lookup for
# all 20 tokens, i.e. ~20 pattern rebuilds per row. On a 30k-row glossary that was the
# single most expensive thing this module did to its own data: measured 5.9x slower than
# the precompiled form, and roughly a third of the wall-clock of a whole CSV load.
_SEPARATOR_RUN = re.compile(r"[-_/]+")
_WHITESPACE_RUN = re.compile(r"\s+")
_PROTECTION_PATTERNS: tuple[tuple[re.Pattern[str], ProtectionLevel], ...] = tuple(
    (re.compile(rf"(?<!\w){re.escape(token)}(?!\w)"), level)
    for token, level in (*_NEGATED_PROTECTION, *_PROTECTION_WORDS)
)


def _coerce_protection(value: str) -> ProtectionLevel:
    """
    Map a free-text classification onto ProtectionLevel.

    Two rules, in order:

    1. NEGATIONS first. "Non-Public" is not public and "Unrestricted" is not restricted;
       a plain substring scan gets both exactly backwards.
    2. Then the positive table, strictest first, so "highly confidential" beats
       "confidential" and an ambiguous label resolves to the stronger protection.
       Under-protecting a field is the expensive mistake.

    Matching is on WORD BOUNDARIES after normalising separators, so "un-restricted",
    "un_restricted" and "Unrestricted" behave alike. Unrecognised text falls back to
    INTERNAL rather than PUBLIC, and the raw string is preserved in source_metadata
    regardless, because this enum cannot represent every organisation's taxonomy.

    Order within `_PROTECTION_PATTERNS` is load-bearing and matches the source tables:
    negations first, then strictest-first positives.
    """
    if not value:
        return ProtectionLevel.INTERNAL
    text = _WHITESPACE_RUN.sub(" ", _SEPARATOR_RUN.sub(" ", value.strip().lower()))
    for pattern, level in _PROTECTION_PATTERNS:
        if pattern.search(text):
            return level
    return ProtectionLevel.INTERNAL


def _coerce_type(value: str) -> DataType:
    if not value:
        return DataType.UNKNOWN
    base = str(value).strip().lower().split("(")[0].strip()
    return _TYPE_WORDS.get(base, DataType.UNKNOWN)


# =============================================================================
# INCREMENTAL SYNC
# =============================================================================


def content_hash(entry: DictionaryEntry) -> str:
    """
    Hash of the text that actually gets embedded.

    Deliberately NOT a hash of the whole row. If it covered every column, editing an
    audit field like `last_reviewed_by` would invalidate the vector and the first such
    edit would turn an incremental sync into a full re-embed -- the exact cost this
    exists to avoid. Only fields that change the embedding belong here.

    DERIVED FROM THE EMBEDDED TEXT ITSELF, not from a hand-listed subset of fields.

    It used to join business_name, logical_name and definition -- but
    `to_searchable_text()` ALSO embeds `synonyms`. So editing an entry's synonyms changed
    the text that got encoded while leaving this hash untouched, and `sync()` reported
    success over a vector that no longer matched its entry. A stale vector is the worst
    outcome this function can produce: nothing errors, the report says the row is
    unchanged, and the entry quietly stops matching what it says it matches.

    Two hand-maintained lists of "the fields that matter" cannot be kept in step by
    discipline; they diverge the first time somebody adds a field to one of them. Hashing
    the embedded string means the question "does this field affect the embedding?" has
    exactly one answer, in exactly one place.

    The audit-field guarantee is unchanged and now stronger, because it follows from the
    definition rather than from remembering to exclude things: a column that does not
    reach `to_searchable_text()` cannot reach this hash.
    """
    payload = entry.to_searchable_text()
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class SyncReport:
    """What a sync actually did."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    embedded: int = 0
    # Entries whose protection code MOVED -- appeared, disappeared, or changed to another
    # code -- between the indexed entry and the one just read.
    #
    # Deliberately its own list rather than a contribution to `updated`, and deliberately
    # not counted in `changed`: those two are about vectors, and governance never reaches
    # `to_searchable_text()` (see `content_hash`), so a reclassified row is genuinely
    # "unchanged" as far as embedding work goes. It is not unchanged as far as the caller
    # is concerned, and the one line most callers print was the only place that could say
    # so -- a sync that stripped the code off every entry in the index reported
    # `=30 unchanged` and nothing else.
    governance_changed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated) + len(self.removed)

    def __str__(self) -> str:
        total = self.unchanged + len(self.added) + len(self.updated)
        saved = ""
        if total:
            saved = f"  ({self.unchanged}/{total} reused, {1 - self.embedded / max(total, 1):.0%} skipped)"
        governance = (
            f"  !{len(self.governance_changed)} governance changed"
            if self.governance_changed
            else ""
        )
        return (
            f"+{len(self.added)} added  ~{len(self.updated)} updated  "
            f"-{len(self.removed)} removed  ={self.unchanged} unchanged{governance}{saved}"
        )


def diff_entries(
    previous: dict[str, str],
    entries: Sequence[DictionaryEntry],
) -> tuple[list[DictionaryEntry], list[str], SyncReport]:
    """
    Work out what changed since the last sync.

    Args:
        previous: {entry_id: content_hash} from the previous sync.
        entries: The entries as they are now.

    Returns:
        (entries_needing_embedding, ids_to_remove, report)
    """
    report = SyncReport()
    current: dict[str, str] = {}
    to_embed: list[DictionaryEntry] = []

    for entry in entries:
        digest = content_hash(entry)
        current[entry.id] = digest
        old = previous.get(entry.id)
        if old is None:
            report.added.append(entry.id)
            to_embed.append(entry)
        elif old != digest:
            report.updated.append(entry.id)
            to_embed.append(entry)
        else:
            report.unchanged += 1

    report.removed = [eid for eid in previous if eid not in current]
    report.embedded = len(to_embed)
    return to_embed, report.removed, report


def load_hashes(path: str | Path) -> dict[str, str]:
    """Read the hash manifest written by `save_hashes`. Missing file -> full rebuild."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt manifest must degrade to a full re-embed, never to a crash or -- far
        # worse -- to a partially-updated index that silently disagrees with the source.
        return {}


def save_hashes(path: str | Path, entries: Sequence[DictionaryEntry]) -> None:
    """Write the manifest for the next sync."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({e.id: content_hash(e) for e in entries}, indent=0),
        encoding="utf-8",
    )


# =============================================================================
# THE THREE-LINE API
# =============================================================================


@dataclass
class GlossaryIndex:
    """
    A searchable glossary: entries, their vectors, and the hashes that make sync cheap.

    Produced by `build_index`, refreshed in place by `sync`. Holding the hashes alongside
    the vectors is what lets a later sync re-embed only what moved, without the caller
    having to manage a manifest.
    """

    entries: dict[str, DictionaryEntry] = field(default_factory=dict)
    vectors: Any = None  # np.ndarray, aligned with `order`
    order: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    provider: Any = None
    # Everything `build_index` handed `load_entries`, so `sync` can re-read the source THE
    # SAME WAY. It is the pattern `provider` already uses, for the same reason: an index
    # that cannot say how it was built can only be refreshed by a caller who remembers.
    #
    # `governance` is the member that made this urgent. `sync(index, "glossary.xlsx")` --
    # the call this module's own docstring shows -- re-read the file with no vocabulary,
    # and `sync`'s refresh loop then replaced every entry object with the uncoded one.
    # Measured on a byte-identical file: 27 coded entries out of 30 became 0, and the
    # report said `=30 unchanged`. The refusal gate went with it, which is worse than the
    # loss -- a row whose stated tier contradicts its own code (NM-0028) raised from
    # `build_index` and loaded SILENTLY through `sync`, so the invariant this library
    # exists to enforce was unenforced on the module's documented refresh path.
    #
    # `columns`, `id_prefix`, `sheet`, `delimiter`, `encoding` and `header_row` have the
    # identical hole and are stored for the identical reason. They merely failed loudly by
    # accident -- dropping `columns` changes the derived ids, so the diff reports the whole
    # glossary added and removed instead of quietly re-describing it.
    load_options: dict[str, Any] = field(default_factory=dict)
    # What the most recent load admitted, refused and bounded. Replaced by every `sync`,
    # never accumulated, and deliberately NOT a member of `load_options` -- a mutable
    # report replayed on every refresh would tally the whole history of the index under a
    # name that reads as "this load".
    load_report: LoadReport | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def searchable_text(self, entry: DictionaryEntry) -> str:
        return entry.to_searchable_text()


def _default_provider() -> Any:
    from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
        default_embedding_provider,
    )

    return default_embedding_provider()


def _embed_documents(provider: Any, texts: Sequence[str]):
    """Encode as DOCUMENTS. Never as queries -- BGE is asymmetric and mixing the two
    silently costs accuracy, because the query instruction belongs on one side only."""
    import numpy as np

    if not texts:
        return np.zeros((0, provider.dimension), dtype="float32")
    if hasattr(provider, "embed_documents"):
        return provider.embed_documents(list(texts))
    result = provider.embed(list(texts))  # fallback for the plain port
    if result.is_failure:
        raise RuntimeError(f"Embedding failed: {result.error}")
    return result.unwrap().embeddings


def build_index(
    source: str | Path | Iterable[dict],
    provider: Any = None,
    **kwargs: Any,
) -> GlossaryIndex:
    """
    Read a glossary from anywhere and embed it. One call.

    Args:
        source: File path (.csv/.tsv/.xlsx/.json/.jsonl/.parquet), a SQLAlchemy
            connection string, or an iterable of dicts.
        provider: Embedding provider. Defaults to the bundled offline encoder.
        **kwargs: Passed to `load_entries` -- `query`, `columns`, `id_prefix`,
            `governance`, `governance_strict`, `admit`, `value_delimiters`,
            `delimiter_strict`, `metadata_columns`, `metadata_max_bytes`, and the reader
            options `sheet`, `delimiter`, `encoding` and `header_row`. All of them are
            remembered on the index and reused by `sync`, so a refresh reads the source
            the same way -- same vocabulary, same admission filter, same separators.

            `report` is the one exception: it is per-call and is never remembered, because
            `load_options` is replayed by every refresh.

    Returns:
        A GlossaryIndex ready to search or to `sync`. `index.load_report` says what the
        load admitted, refused and bounded.

    Example:
        index = build_index("glossary.xlsx")
        index = build_index("glossary.xlsx", governance="protection_classes.json")
        index = build_index("postgresql://host/db", query="SELECT * FROM terms")
    """
    import numpy as np

    # Resolve the vocabulary ONCE, here, and remember the resolved object rather than the
    # path or document the caller passed. `sync` re-reads the SOURCE; re-reading the
    # CATALOG underneath it would let an edit to that JSON reclassify half an index under a
    # report that says "unchanged", which is the same silent shape as the loss this bundle
    # closes. A caller who wants the new catalog says so -- `sync(index, src,
    # governance=...)` overrides -- or builds again.
    options = dict(kwargs)
    if "governance" in options:
        options["governance"] = load_governance(options["governance"])
    # The report is per-call and never remembered: `load_options` is replayed by every
    # `sync`, so a report stored in it would be refilled -- under the caller's own object
    # -- on every refresh for the life of the index.
    report = options.pop("report", None) or LoadReport()

    entries = load_entries(source, report=report, **options)
    provider = provider or _default_provider()

    order = [e.id for e in entries]
    vectors = _embed_documents(provider, [e.to_searchable_text() for e in entries])

    return GlossaryIndex(
        entries={e.id: e for e in entries},
        vectors=np.asarray(vectors, dtype="float32"),
        order=order,
        hashes={e.id: content_hash(e) for e in entries},
        provider=provider,
        load_options=options,
        load_report=report,
    )


def sync(
    index: GlossaryIndex,
    source: str | Path | Iterable[dict],
    **kwargs: Any,
) -> SyncReport:
    """
    Refresh an index from its source, re-embedding ONLY what changed.

    Mutates `index` in place and returns what happened.

    Args:
        index: An index from `build_index`.
        source: The same source, re-read.
        **kwargs: Passed to `load_entries`, overriding what `build_index` was given for
            this call only. Anything not passed here is taken from `index.load_options`,
            so a refresh reads the source the way the build did -- same vocabulary, same
            column mapping, same sheet. Governance can still be turned off, with
            `governance=None, governance_strict=False`; omitting a keyword no longer does
            it by accident.

    Returns:
        SyncReport, whose `embedded` count is the number of texts actually encoded and
        whose `governance_changed` lists the entries whose protection code moved.
        `index.load_report` is REPLACED with what this refresh admitted and refused --
        replaced rather than accumulated, so it always describes the current index.

    Example:
        report = sync(index, "glossary.xlsx")
        print(report)   # +3 added  ~1 updated  -0 removed  =996 unchanged
    """
    import numpy as np

    options = {**index.load_options, **kwargs}
    load_report = options.pop("report", None) or LoadReport()
    entries = load_entries(source, report=load_report, **options)
    index.load_report = load_report
    to_embed, removed, report = diff_entries(index.hashes, entries)
    # Before anything is mutated, and only for entries that exist on BOTH sides: an id
    # that is new is reported as added, and one that is gone is reported as removed, so
    # calling either of those a governance change would say the same thing twice.
    report.governance_changed = [
        entry.id
        for entry in entries
        if (indexed := index.entries.get(entry.id)) is not None
        and indexed.governance_code != entry.governance_code
    ]

    for entry_id in removed:
        index.entries.pop(entry_id, None)
        index.hashes.pop(entry_id, None)

    if to_embed:
        new_vectors = _embed_documents(
            index.provider or _default_provider(),
            [e.to_searchable_text() for e in to_embed],
        )
        position = {eid: i for i, eid in enumerate(index.order)}
        appended: list[str] = []
        appended_vectors: list[Any] = []

        for entry, vector in zip(to_embed, new_vectors, strict=True):
            i = position.get(entry.id)
            if i is None:
                appended.append(entry.id)
                appended_vectors.append(vector)
            else:
                index.vectors[i] = vector

        if appended:
            index.order.extend(appended)
            index.vectors = (
                np.vstack([index.vectors, np.asarray(appended_vectors, dtype="float32")])
                if len(index.vectors)
                else np.asarray(appended_vectors, dtype="float32")
            )
    # Refresh EVERY entry object, not only the re-embedded ones.
    #
    # content_hash covers just the embedded text, so a row whose classification changed
    # but whose name and definition did not is reported "unchanged" -- correct for the
    # vector, wrong for the entry. Updating entries only inside the `to_embed` branch
    # meant a governance edit was applied or silently dropped depending on whether some
    # UNRELATED row happened to change in the same sync. That is the worst failure mode
    # this tool has: the index looks healthy and hands back a stale PII classification.
    for entry in entries:
        index.entries[entry.id] = entry

    # Hashes, by contrast, only need rewriting for the rows that MOVED. An unchanged row
    # was classified unchanged by comparing against `index.hashes` in the first place, so
    # the value already stored there is by definition the current one -- recomputing it
    # re-hashed the whole glossary on every sync to arrive back at what was already
    # there. Measured at 30k rows that was ~44 ms of pure waste on a no-change sync,
    # which is the case a daily sync actually hits.
    for entry in to_embed:
        index.hashes[entry.id] = content_hash(entry)

    if removed:
        # Rebuild the alignment between `order` and `vectors`. Dropping rows from a numpy
        # array by index is fiddly enough that doing it in place invites an off-by-one
        # that silently mis-associates every vector after the deletion.
        keep = [i for i, eid in enumerate(index.order) if eid in index.entries]
        index.vectors = index.vectors[keep] if len(index.vectors) else index.vectors
        index.order = [index.order[i] for i in keep]

    return report
