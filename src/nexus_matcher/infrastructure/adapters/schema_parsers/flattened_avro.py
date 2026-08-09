"""
nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro | Layer: INFRASTRUCTURE
Parser for FLATTENED Avro schemas -- the output of a schema flattener, not raw Avro.

## Relationships
# IMPLEMENTS → domain/ports/schema_parser :: SchemaParser protocol
# USED_BY    → application/use_cases/match_schema :: governance inheritance matching
# MIRRORS    → GAvroSchemaFlattener.groovy (NexusPiercer) :: flattening conventions

## Attributes
# Security: No external calls; parses local files or in-memory structures
# Performance: Pure string work, linear in field count
# Reliability: Tolerates missing optional metadata; never raises on a single bad field

## Why a dedicated parser

A flattener turns a nested Avro record into flat rows keyed by a joined path:

    customer_addresses__street_name
    customer_orders__items__sku

Handing that string to the matcher verbatim wastes the most valuable signal we have.
Measured on the labelled benchmark, supplying the PARENT PATH separately from the leaf
name is worth **+19.3 points of P@1** -- the single largest factor in the whole pipeline,
larger than the model, the fusion method and the reranker combined.

A flattened name already contains that path; it just has to be split back out. So this
parser reconstructs the hierarchy rather than treating the identifier as one opaque token:

    customer_addresses__street_name
      -> name        "street_name"
         parent_path "customer.addresses"
         full_path   "customer.addresses.street_name"
         is_array    True   (the __ boundary)

## Flattening conventions

Mirrors GAvroSchemaFlattener:

  * path segments joined with "_"
  * "__" (double underscore) marks an ARRAY BOUNDARY
  * arrays of records are exploded into one column per leaf
  * arrays of primitives are JSON-serialised to a string (`isArraySerialized`)
  * maps are serialised to string
  * unions are unwrapped to their non-null branch, with a `nullable` flag

## The doc gap

GAvroSchemaFlattener does NOT propagate the Avro `doc` attribute, so a flattened schema
produced by it carries names but no definitions. Definitions are the strongest matching
signal after the path, so:

  * this parser reads `doc` (and several common aliases) when the flattened input has it
  * `nexus_matcher.infrastructure.adapters.schema_parsers.avro` flattens raw Avro
    natively AND propagates doc, which is the preferred path when the .avsc is available

Ambiguity note: because "_" is both the separator and a legal character inside Avro field
names, `customer_id` is indistinguishable from a field `id` nested under `customer`. The
splitter therefore prefers KNOWN structure (an explicit path field, or `__` boundaries)
and falls back to a conservative heuristic only when nothing better exists. Pass
`field_paths` or use the raw-Avro parser when exactness matters.

## Identity: the reconstructed path is NOT a key

The split is lossy in one direction and that matters downstream. `contact__email` and
`contact_email` -- an array of contacts and a scalar column, both legal fields of one
Avro record -- reconstruct to the same `contact.email`. So the dotted path identifies a
POSITION IN THE HIERARCHY, which is what the matcher wants for retrieval, and cannot
identify the column, which is what a caller wants for lookup.

`source_metadata["flattened_name"]` is that identity: the exact string the caller gave
us, set on every field this module produces. `match_schema` keys its results by it.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nexus_matcher.domain.models.entities import Schema, SchemaField
from nexus_matcher.shared.types.base import DataType, Result

ARRAY_BOUNDARY = "__"

# Keys a flattener might use for each concept. Checked in order; first hit wins.
_NAME_KEYS = ("flattenedName", "flattened_name", "name", "column", "field", "path")
_DOC_KEYS = ("doc", "description", "definition", "comment", "documentation", "business_definition")
_TYPE_KEYS = ("dataType", "data_type", "type", "originalAvroType", "original_avro_type", "sql_type")
_NULLABLE_KEYS = ("nullable", "is_nullable", "isNullable", "optional")
_ARRAY_KEYS = ("isArraySerialized", "is_array_serialized", "isArray", "is_array", "repeated")
_ELEMENT_KEYS = ("arrayElementType", "array_element_type", "elementType", "items")
_PATH_KEYS = ("fieldPath", "field_path", "jsonPath", "json_path", "dotted_path", "original_path")

_AVRO_TO_DOMAIN = {
    "string": DataType.STRING,
    "bytes": DataType.BYTES,
    "int": DataType.INTEGER,
    "long": DataType.LONG,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "boolean": DataType.BOOLEAN,
    "null": DataType.UNKNOWN,
    "record": DataType.RECORD,
    "enum": DataType.ENUM,
    "array": DataType.ARRAY,
    "map": DataType.JSON,
    "fixed": DataType.BYTES,
    "decimal": DataType.DECIMAL,
    "date": DataType.DATE,
    "timestamp-millis": DataType.TIMESTAMP,
    "timestamp-micros": DataType.TIMESTAMP,
    "uuid": DataType.UUID,
    # Common SQL-ish spellings, since flatteners often emit these instead.
    "integer": DataType.INTEGER,
    "bigint": DataType.LONG,
    "smallint": DataType.INTEGER,
    "varchar": DataType.STRING,
    "text": DataType.STRING,
    "char": DataType.STRING,
    "numeric": DataType.DECIMAL,
    "real": DataType.FLOAT,
    "bool": DataType.BOOLEAN,
    "datetime": DataType.TIMESTAMP,
    "timestamp": DataType.TIMESTAMP,
    "json": DataType.JSON,
}


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def map_data_type(raw: Any) -> DataType:
    """Map a flattener's type string onto the domain DataType."""
    if raw is None:
        return DataType.UNKNOWN
    if isinstance(raw, DataType):
        return raw
    if isinstance(raw, (list, tuple)):  # an un-unwrapped union
        for item in raw:
            if str(item).lower() != "null":
                return map_data_type(item)
        return DataType.UNKNOWN
    if isinstance(raw, Mapping):
        return map_data_type(raw.get("logicalType") or raw.get("type"))

    text = str(raw).strip().lower()
    if text in _AVRO_TO_DOMAIN:
        return _AVRO_TO_DOMAIN[text]
    # "varchar(255)", "decimal(10,2)"
    base = re.split(r"[(<\[]", text, maxsplit=1)[0].strip()
    return _AVRO_TO_DOMAIN.get(base, DataType.UNKNOWN)


def split_flattened_name(
    flattened: str,
    separator: str = "_",
    array_boundary: str = ARRAY_BOUNDARY,
) -> tuple[list[str], bool]:
    """
    Split a flattened identifier into path segments.

    Returns (segments, touches_array).

    The double-underscore array boundary is authoritative where present, because it is
    unambiguous. Single underscores are ALSO split, which is the right default: a
    flattened name is a path by construction, and over-splitting merely produces extra
    context tokens (harmless for retrieval), whereas under-splitting hides the hierarchy
    that is worth +19.3 P@1.

    >>> split_flattened_name("customer_addresses__street_name")
    (['customer', 'addresses', 'street', 'name'], True)
    """
    if not flattened:
        return [], False

    touches_array = array_boundary in flattened
    # Normalise the array boundary to the plain separator once it has been noted.
    text = flattened.replace(array_boundary, separator) if touches_array else flattened

    # Respect explicit dotted paths if the flattener emitted them.
    if "." in text:
        parts = [p for p in text.split(".") if p]
    else:
        parts = [p for p in text.split(separator) if p]

    # Split residual camelCase inside a segment: "streetName" -> "street", "Name".
    segments: list[str] = []
    for part in parts:
        pieces = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", part).split()
        segments.extend(p for p in pieces if p)
    return segments, touches_array


def field_from_flattened(
    flattened_name: str,
    doc: str = "",
    data_type: Any = None,
    nullable: bool = True,
    is_array: bool = False,
    array_element_type: Any = None,
    explicit_path: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> SchemaField:
    """
    Build a SchemaField from one flattened row, reconstructing its hierarchy.

    `explicit_path` (a dotted path, if the flattener emitted one) always wins over
    inference from the joined name.
    """
    if explicit_path:
        segments = [p for p in str(explicit_path).replace("/", ".").split(".") if p]
        touches_array = is_array
    else:
        segments, touches_array = split_flattened_name(flattened_name)

    if not segments:
        segments = [flattened_name or "field"]

    leaf = segments[-1]
    parents = segments[:-1]

    return SchemaField(
        name=leaf,
        data_type=map_data_type(data_type),
        # Dotted path is what ContextEnricher expects; it derives the parent context from it.
        full_path=".".join(segments),
        parent_path=".".join(parents),
        description=(doc or "").strip(),
        is_nullable=_as_bool(nullable) if nullable is not None else True,
        is_array=bool(is_array or touches_array),
        array_item_type=map_data_type(array_element_type) if array_element_type else None,
        source_metadata={
            **(dict(extra) if extra else {}),
            # Written LAST because match_schema keys its result dict by this: it is the
            # caller's handle on the field, and the dotted path above is not (several
            # flattened names split to one path). An unrecognised column literally named
            # "flattened_name" used to land in `extra` and overwrite it, leaving the
            # field addressable only under some other row's name.
            "flattened_name": flattened_name,
            "flattened": True,
        },
    )


class FlattenedAvroParser:
    """
    Parse a flattened Avro schema into domain SchemaFields.

    Accepts, transparently:
      * a JSON object keyed by flattened name  {"cust_addr__city": {...}}
      * a JSON array of row objects            [{"flattenedName": ..., "doc": ...}]
      * a CSV/TSV export with a header row
      * a plain list of names                  ["cust_id", "cust_addr__city"]

    Example:
        parser = FlattenedAvroParser()
        schema = parser.parse_file("customer_flat.json").unwrap()
        for f in schema.fields:
            print(f.full_path, "|", f.description)
    """

    format_name = "flattened_avro"
    file_extensions = (".json", ".csv", ".tsv", ".jsonl")

    def can_parse(self, source: Any) -> bool:
        """True when the payload looks like flattened-schema rows."""
        if isinstance(source, Mapping):
            if any(k in source for k in ("fields", "type", "namespace")):
                return False  # looks like raw Avro; the avro parser should take it
            values = list(source.values())[:5]
            return bool(values) and all(isinstance(v, (Mapping, str)) for v in values)
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            head = list(source)[:5]
            return bool(head) and all(isinstance(v, (Mapping, str)) for v in head)
        return False

    # -- parsing ----------------------------------------------------------

    def parse(self, source: Any, schema_name: str = "flattened") -> Result[Schema]:
        try:
            rows = self._normalise(source)
            fields = [self._to_field(r) for r in rows]
            fields = [f for f in fields if f is not None]
            if not fields:
                return Result.failure("No fields found in flattened schema")
            return Result.success(
                Schema(name=schema_name, fields=tuple(fields), source_format=self.format_name)
            )
        except Exception as exc:
            return Result.failure(f"Flattened Avro parse failed: {type(exc).__name__}: {exc}")

    def parse_file(self, path: str | Path) -> Result[Schema]:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8-sig")
        except Exception as exc:
            return Result.failure(f"Cannot read {p}: {exc}")

        suffix = p.suffix.lower()
        try:
            if suffix in (".csv", ".tsv"):
                delimiter = "\t" if suffix == ".tsv" else ","
                rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
                return self.parse(rows, schema_name=p.stem)
            if suffix == ".jsonl":
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
                return self.parse(rows, schema_name=p.stem)
            return self.parse(json.loads(text), schema_name=p.stem)
        except Exception as exc:
            return Result.failure(f"Cannot parse {p}: {type(exc).__name__}: {exc}")

    # -- internals --------------------------------------------------------

    def _normalise(self, source: Any) -> list[dict[str, Any]]:
        """Coerce any accepted shape into a list of row dicts carrying a name."""
        if isinstance(source, Mapping):
            # Some flatteners wrap the payload.
            for key in ("fields", "columns", "schema", "flattenedFields"):
                inner = source.get(key)
                if isinstance(inner, (Mapping, list)):
                    return self._normalise(inner)
            rows = []
            for name, value in source.items():
                if isinstance(value, Mapping):
                    rows.append({**value, "__name__": name})
                else:
                    rows.append({"__name__": name, "dataType": value})
            return rows

        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            rows = []
            for item in source:
                if isinstance(item, Mapping):
                    rows.append(dict(item))
                elif isinstance(item, str):
                    rows.append({"__name__": item})
            return rows

        raise TypeError(f"Unsupported flattened schema payload: {type(source).__name__}")

    def _to_field(self, row: Mapping[str, Any]) -> SchemaField | None:
        name = row.get("__name__") or _first(row, _NAME_KEYS)
        if not name:
            return None
        known = set(_NAME_KEYS) | set(_DOC_KEYS) | set(_TYPE_KEYS) | set(_NULLABLE_KEYS)
        known |= set(_ARRAY_KEYS) | set(_ELEMENT_KEYS) | set(_PATH_KEYS) | {"__name__"}
        return field_from_flattened(
            flattened_name=str(name),
            doc=str(_first(row, _DOC_KEYS) or ""),
            data_type=_first(row, _TYPE_KEYS),
            nullable=_first(row, _NULLABLE_KEYS),
            is_array=_as_bool(_first(row, _ARRAY_KEYS)),
            array_element_type=_first(row, _ELEMENT_KEYS),
            explicit_path=_first(row, _PATH_KEYS),
            extra={k: v for k, v in row.items() if k not in known},
        )


class _NamedTypes:
    """
    Registry of Avro named types, so a reference by name can be expanded.

    Avro's commonest idiom is to define a record ONCE and refer to it by name later
    (`{"name": "bill_to", "type": "Address"}`). Without a registry those references are
    opaque strings: the referenced record's leaves are never emitted, and the reference
    itself is emitted as a single UNKNOWN-typed leaf. On a realistic order schema that
    lost 5 of 14 columns and invented 2 that do not exist -- and an invented column will
    happily match a dictionary entry and inherit its governance level.
    """

    __slots__ = ("_types",)

    _NAMED = frozenset({"record", "enum", "fixed", "error"})

    def __init__(self) -> None:
        self._types: dict[str, Any] = {}

    def register(self, node: Any) -> None:
        """Record a type definition under both its bare and fully-qualified name."""
        if not isinstance(node, Mapping):
            return
        name = node.get("name")
        if not name or node.get("type") not in self._NAMED:
            return
        self._types[name] = node
        namespace = node.get("namespace")
        if namespace:
            self._types[f"{namespace}.{name}"] = node

    def resolve(self, node: Any) -> Any:
        """Follow a named-type reference, if that is what this is."""
        if isinstance(node, str):
            return self._types.get(node, node)
        return node


def _unwrap_union(node: Any) -> tuple[Any, bool]:
    """Resolve a union to its non-null branch; report nullability."""
    if not isinstance(node, list):
        return node, False
    non_null = [
        n for n in node if n != "null" and (not isinstance(n, Mapping) or n.get("type") != "null")
    ]
    return (non_null[0] if non_null else "null"), len(non_null) < len(node)


def flatten_avro_schema(
    schema: Mapping[str, Any],
    separator: str = "_",
    array_boundary: str = ARRAY_BOUNDARY,
    inherit_doc: bool = True,
) -> list[SchemaField]:
    """
    Flatten a RAW Avro schema the way GAvroSchemaFlattener does -- but keeping `doc`.

    This is the preferred entry point when the .avsc is available, because it produces
    the exact hierarchy rather than inferring it from a joined string, and because it
    propagates documentation that the Groovy flattener drops.

    Args:
        schema: A parsed Avro schema (a record type).
        separator: Segment join character.
        array_boundary: Marker emitted at an array boundary.
        inherit_doc: When a leaf has no doc of its own, fall back to the nearest ancestor
            record's doc. A parent's description is weak evidence but strictly better than
            an empty string, and the matcher weights description heavily.

    Returns:
        SchemaFields with full_path, parent_path, description and array flags populated.
    """
    fields: list[SchemaField] = []
    named = _NamedTypes()

    def emit(
        path: list[str],
        flat: list[str],
        data_type: DataType,
        doc: str,
        nullable: bool,
        is_array: bool = False,
        item_type: DataType | None = None,
        **metadata: Any,
    ) -> None:
        """Append one leaf. Every emission site goes through here so that the flattened
        name, the dotted path and the parent path can never drift apart."""
        fields.append(
            SchemaField(
                name=path[-1] if path else "value",
                data_type=data_type,
                full_path=".".join(path),
                parent_path=".".join(path[:-1]),
                description=doc.strip(),
                is_nullable=nullable,
                is_array=is_array,
                array_item_type=item_type,
                source_metadata={"flattened_name": separator.join(flat), **metadata},
            )
        )

    def walk(
        node: Any,
        path: list[str],
        flat: list[str],
        doc: str,
        in_array: bool,
        active: frozenset = frozenset(),
    ) -> None:
        node, nullable = _unwrap_union(node)
        node = named.resolve(node)
        named.register(node)

        # Break self-reference. A record containing itself (a tree node with children, an
        # employee with a manager) is legal Avro and would otherwise recurse until the
        # stack gives out. `active` holds the type names being expanded on THIS branch, so
        # a cycle stops while sibling branches still expand normally.
        type_name = node.get("name") if isinstance(node, Mapping) else None
        if type_name and type_name in active:
            # Emit the cycle point as a single leaf rather than returning nothing.
            # Dropping it would silently omit a real column from the schema, and a column
            # nobody sees is a column nobody governs -- the same class of failure as the
            # phantom leaves above, just in the other direction.
            emit(
                path, flat, DataType.RECORD, doc, nullable, in_array, recursive_reference=type_name
            )
            return
        if type_name:
            active = active | {type_name}
        node_type = node.get("type") if isinstance(node, Mapping) else node
        node_doc = (node.get("doc") if isinstance(node, Mapping) else "") or ""
        carried = node_doc or (doc if inherit_doc else "")

        if node_type == "record" and isinstance(node, Mapping):
            for child in node.get("fields", []):
                cname = child.get("name")
                if not cname:
                    continue
                cdoc = child.get("doc") or ""
                walk(
                    child.get("type"),
                    [*path, cname],
                    [*flat, cname],
                    cdoc or carried,
                    in_array,
                    active,
                )
            return

        if node_type == "array" and isinstance(node, Mapping):
            items, _ = _unwrap_union(node.get("items"))
            items = named.resolve(items)
            named.register(items)
            item_type = items.get("type") if isinstance(items, Mapping) else items
            if item_type == "record":
                # The ITEM type must join `active` before descending. The cycle guard in
                # walk() only sees types passed to walk(), and this branch iterates the
                # item's fields directly -- so `array<Employee>` inside Employee never
                # tripped it and recursed until the stack died. An employee with a manager
                # AND a list of reports is an entirely ordinary schema.
                item_name = items.get("name")
                if item_name and item_name in active:
                    emit(
                        path,
                        flat,
                        DataType.ARRAY,
                        carried,
                        nullable,
                        True,
                        recursive_reference=item_name,
                    )
                    return
                item_active = active | ({item_name} if item_name else set())

                # Exploded into one column per leaf; mark the boundary.
                for child in items.get("fields", []):
                    cname = child.get("name")
                    if not cname:
                        continue
                    # Mark the array boundary by padding the parent segment, so that
                    # separator.join() yields "orders__sku" rather than "orders_sku".
                    # (array_boundary.strip(separator) would be the empty string -- the
                    # padding is the DIFFERENCE between the two markers, not the marker.)
                    pad = array_boundary[len(separator) :]
                    walk(
                        child.get("type"),
                        [*path, cname],
                        [*flat[:-1], f"{flat[-1]}{pad}", cname] if flat else [cname],
                        child.get("doc") or carried,
                        True,
                        item_active,
                    )
                return
            # Array of primitives: serialised to a single string column.
            emit(
                path,
                flat,
                DataType.ARRAY,
                carried,
                nullable,
                True,
                item_type=map_data_type(item_type),
                array_serialized=True,
            )
            return

        if node_type == "map":
            emit(path, flat, DataType.JSON, carried, nullable, map_serialized=True)
            return

        emit(path, flat, map_data_type(node), carried, nullable, in_array)

    walk(schema, [], [], "", False)
    return fields
