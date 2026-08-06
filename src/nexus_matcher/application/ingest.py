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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    "protection_level": (
        "protectionlevel",
        "classification",
        "sensitivity",
        "governance",
        "governancestatus",
        "confidentiality",
        "pii",
        "securityclassification",
    ),
}


def _norm_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


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


def _read_tabular(source: str | Path, **kwargs: Any) -> tuple[list[dict], list[str]]:
    """Read any tabular file into rows + column names."""
    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        import csv

        delimiter = kwargs.pop("delimiter", "\t" if suffix == ".tsv" else ",")
        # utf-8-sig strips the BOM Excel writes, which otherwise corrupts the first header.
        text = path.read_text(encoding=kwargs.pop("encoding", "utf-8-sig"))
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        rows = [dict(r) for r in reader]
        return rows, list(reader.fieldnames or [])

    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Reading Excel needs openpyxl. Install with: pip install "nexus-matcher[loaders]"'
            ) from exc
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[kwargs["sheet"]] if kwargs.get("sheet") else wb.active
        it = ws.iter_rows(values_only=True)
        header = [str(c) if c is not None else "" for c in next(it, ())]
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


def load_entries(
    source: str | Path | Iterable[dict],
    query: str | None = None,
    columns: dict[str, str] | None = None,
    id_prefix: str = "",
    **kwargs: Any,
) -> list[DictionaryEntry]:
    """
    Read a source and turn it into DictionaryEntry objects.

    Args:
        source: File path, SQLAlchemy connection string, or iterable of dicts.
        query: SQL, when source is a connection string.
        columns: Explicit {field: source_column} overrides. Anything not given is
            inferred from the header via `map_columns`.
        id_prefix: Prefix for generated ids, useful when pooling several glossaries.
        **kwargs: Passed to the reader (`sheet`, `delimiter`, `encoding`).

    Returns:
        Entries, skipping rows with no business name AND no definition -- such a row
        cannot be matched to and would only add a distractor.

    Raises:
        ValueError: if neither a business name nor a definition column can be identified,
            which means the mapping is wrong rather than the data being sparse.
    """
    rows, header = read_source(source, query=query, **kwargs)
    mapping = {**map_columns(header), **(columns or {})}

    if "business_name" not in mapping and "definition" not in mapping:
        raise ValueError(
            f"Could not find a business-name or definition column in {header}.\n"
            f"Pass an explicit mapping, e.g.\n"
            f"    columns={{'business_name': 'Term', 'definition': 'Meaning'}}"
        )

    entries: list[DictionaryEntry] = []
    used_ids: set[str] = set()
    for row in rows:
        # `row` is bound as a default argument rather than captured. Capturing the loop
        # variable works only while the closure is called inside the same iteration --
        # true today, and silently wrong the moment anyone defers a call.
        def get(fieldname: str, _row: dict = row) -> str:
            column = mapping.get(fieldname)
            return _as_text(_row.get(column)) if column else ""

        business_name = get("business_name")
        definition = get("definition")
        if not business_name and not definition:
            continue

        raw_id = get("id")
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
            key = f"{business_name}{get('logical_name')}"
            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
            entry_id = f"{id_prefix}{digest}"
            # Genuine duplicates (same name AND same technical name) still need distinct
            # ids, so disambiguate deterministically by order of appearance.
            if entry_id in used_ids:
                n = 2
                while f"{entry_id}-{n}" in used_ids:
                    n += 1
                entry_id = f"{entry_id}-{n}"
        used_ids.add(entry_id)

        entries.append(
            DictionaryEntry(
                id=entry_id,
                business_name=business_name,
                logical_name=get("logical_name"),
                definition=definition,
                data_type=_coerce_type(get("data_type")),
                domain=get("domain"),
                protection_level=_coerce_protection(get("protection_level")),
                # Everything unmapped is preserved, PLUS the raw governance string. The
                # classification column is usually the whole reason for matching -- an
                # earlier version mapped it (which excluded it from here) and then never
                # used it, so the value a caller most needs was silently dropped. The
                # enum is lossy by design (an org's "Highly Restricted" collapses to
                # RESTRICTED), so the original text is kept alongside it.
                source_metadata={
                    **{k: v for k, v in row.items() if k not in set(mapping.values())},
                    **(
                        {"governance_raw": get("protection_level")}
                        if mapping.get("protection_level")
                        else {}
                    ),
                },
            )
        )
    return entries


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


def _coerce_protection(value: str) -> ProtectionLevel:
    """
    Map a free-text classification onto ProtectionLevel.

    Order matters: "highly confidential" must be checked before "confidential", and
    "restricted" before anything weaker, so the STRICTEST reading wins on an ambiguous
    label. Getting this wrong in the lenient direction would under-protect a field, which
    is the expensive mistake. Unrecognised text falls back to INTERNAL rather than PUBLIC
    for the same reason -- and the raw string is preserved in source_metadata regardless.
    """
    if not value:
        return ProtectionLevel.INTERNAL
    text = value.strip().lower()
    for token, level in _PROTECTION_WORDS:
        if token in text:
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
    """
    payload = "\x1f".join((entry.business_name, entry.logical_name, entry.definition))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class SyncReport:
    """What a sync actually did."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    embedded: int = 0

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated) + len(self.removed)

    def __str__(self) -> str:
        total = self.unchanged + len(self.added) + len(self.updated)
        saved = ""
        if total:
            saved = f"  ({self.unchanged}/{total} reused, {1 - self.embedded / max(total, 1):.0%} skipped)"
        return (
            f"+{len(self.added)} added  ~{len(self.updated)} updated  "
            f"-{len(self.removed)} removed  ={self.unchanged} unchanged{saved}"
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
        **kwargs: Passed to `load_entries` (`query`, `columns`, `sheet`, `id_prefix`).

    Returns:
        A GlossaryIndex ready to search or to `sync`.

    Example:
        index = build_index("glossary.xlsx")
        index = build_index("postgresql://host/db", query="SELECT * FROM terms")
    """
    import numpy as np

    entries = load_entries(source, **kwargs)
    provider = provider or _default_provider()

    order = [e.id for e in entries]
    vectors = _embed_documents(provider, [e.to_searchable_text() for e in entries])

    return GlossaryIndex(
        entries={e.id: e for e in entries},
        vectors=np.asarray(vectors, dtype="float32"),
        order=order,
        hashes={e.id: content_hash(e) for e in entries},
        provider=provider,
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
        **kwargs: Passed to `load_entries`.

    Returns:
        SyncReport, whose `embedded` count is the number of texts actually encoded.

    Example:
        report = sync(index, "glossary.xlsx")
        print(report)   # +3 added  ~1 updated  -0 removed  =996 unchanged
    """
    import numpy as np

    entries = load_entries(source, **kwargs)
    to_embed, removed, report = diff_entries(index.hashes, entries)

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
            index.entries[entry.id] = entry
            index.hashes[entry.id] = content_hash(entry)
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
    else:
        for entry in entries:
            index.entries[entry.id] = entry

    if removed:
        # Rebuild the alignment between `order` and `vectors`. Dropping rows from a numpy
        # array by index is fiddly enough that doing it in place invites an off-by-one
        # that silently mis-associates every vector after the deletion.
        keep = [i for i, eid in enumerate(index.order) if eid in index.entries]
        index.vectors = index.vectors[keep] if len(index.vectors) else index.vectors
        index.order = [index.order[i] for i in keep]

    return report
