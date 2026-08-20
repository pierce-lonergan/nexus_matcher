"""
benchmarks.synthetic.schemas | Layer: BENCHMARK
Artifact 3 of 5: synthetic schemas, in BOTH forms, with their ground truth.

Every schema ships twice -- as a raw `.avsc` and as a pre-flattened field list -- because
those are two different tests. The raw form exercises the flattener; the pre-flattened
form is what a production pipeline actually sends, and testing only the form that is
convenient to generate is how a library ends up correct on an input nobody uses.

The six profiles, and what each one is for
------------------------------------------
  flat-english      readable names, full docs. The ceiling.
  flat-contracted   THE SAME COLUMNS, contracted through the naming standard, no docs.
                    Paired with flat-english field for field, so the abbreviation gap is
                    a paired measurement rather than two independent runs.
  nested-deep       depth 6+, long flattened paths.
  nested-repeated   one leaf name under many parents. The important one -- see below.
  no-doc            no documentation at all, contracted names. A dense query built from
                    nothing still returns a nearest neighbour, with a confidence that
                    looks fine.
  mixed-production  a proportioned mixture of the other five.

Why nested-repeated is the important one
----------------------------------------
It is built from the glossary's deliberately WIDE near-duplicate clusters: one term name
governed separately in N domains. The schema puts that one leaf name under N different
parents, and the correct answer for each occurrence is the cluster member owned by that
parent's domain. So the leaf name carries no information at all and the parent carries
all of it.

That makes it the direct test of cache-key composition. Key the cache on the leaf name
and all N occurrences collapse to one answer; N-1 of them are then wrong, and the failure
is not an exception or a miss -- it is N confidently-wrong results with no symptom
whatsoever except a count nobody has reason to check.

Leak control
------------
A column is never a copy of its term. `_paraphrase` drops and swaps leading qualifiers,
so query text and indexed text share the subject and the class word and diverge on the
rest. This matters because this repository has already published an inflated benchmark
built the other way -- a gold label derived from the field name, making half the corpus a
string-identity task -- and the giveaway was that nobody measured the overlap. `verify.py`
measures it and prints the distribution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .abbreviations import AbbreviationCatalog
from .glossary import Glossary, GlossaryRow
from .pools import Pools
from .truth import DEFAULT_SHARES, TruthClass, TruthRow

# The marker a flattener emits at an array boundary, and the one this pack puts between
# the parent path and the leaf column so that "the leaf name" is unambiguous to a reader.
ARRAY_BOUNDARY = "__"

_DOC_FRAMES: tuple[str, ...] = (
    "Source column supplied by the upstream extract. Holds the {subject} {kind} for each "
    "record in this feed.",
    "Populated by the collecting system at intake. Carries the {subject} {kind} as it was "
    "received.",
    "Feed column. The {subject} {kind} for the record, passed through without transformation.",
    "Extract column carrying the {subject} {kind}. Nullable where the source system did "
    "not supply one.",
)

_RECORD_NOUNS: tuple[str, ...] = (
    "record",
    "detail",
    "block",
    "segment",
    "entry",
    "payload",
    "group",
    "section",
)


@dataclass(frozen=True)
class SchemaProfile:
    """One schema shape. `leaves` scales with the pack's `scale` multiplier."""

    name: str
    leaves: int
    depth: int
    contracted: bool
    doc_share: float
    repeated: bool = False
    mixed: bool = False
    # Names the profile this one mirrors column-for-column, or "" when it stands alone.
    mirrors: str = ""


PROFILES: tuple[SchemaProfile, ...] = (
    SchemaProfile("flat-english", leaves=200, depth=0, contracted=False, doc_share=1.0),
    SchemaProfile(
        "flat-contracted",
        leaves=200,
        depth=0,
        contracted=True,
        doc_share=0.0,
        mirrors="flat-english",
    ),
    SchemaProfile("nested-deep", leaves=800, depth=6, contracted=True, doc_share=0.4),
    SchemaProfile(
        "nested-repeated", leaves=1200, depth=2, contracted=True, doc_share=0.3, repeated=True
    ),
    SchemaProfile("no-doc", leaves=400, depth=1, contracted=True, doc_share=0.0),
    SchemaProfile(
        "mixed-production", leaves=1800, depth=3, contracted=True, doc_share=0.35, mixed=True
    ),
)


@dataclass
class SyntheticSchema:
    """One generated schema in both forms, plus the answers for its fields."""

    name: str
    profile: SchemaProfile
    # The pre-flattened field list: exactly the shape `FlattenedAvroParser` accepts.
    flattened: list[dict[str, Any]] = field(default_factory=list)
    truth: list[TruthRow] = field(default_factory=list)

    def as_avro(self) -> dict[str, Any]:
        """Rebuild the nested `.avsc` this flattened list came from.

        Reconstructed rather than kept alongside, so the two forms cannot disagree: if
        this function and the flattened list ever describe different schemas, the raw form
        is the one that is wrong, and it is derived here from the one a pipeline sends.
        """
        return _to_avro(self.name, self.flattened)


# =============================================================================
# COLUMN DERIVATION
# =============================================================================


def _paraphrase(
    rng: random.Random, pools: Pools, row: GlossaryRow, strength: float
) -> tuple[str, ...]:
    """
    The column a source system would name for this term -- related, never identical.

    The subject and the class word survive (they are what the column IS); the leading
    qualifiers are dropped or swapped. A term of two tokens has no qualifiers to vary and
    comes through as itself, which is honest: some columns really are named after their
    term, and pretending otherwise would understate the ceiling.
    """
    tokens = list(row.tokens)
    if len(tokens) <= 2:
        return tuple(tokens)
    head, tail = tokens[:-2], tokens[-2:]
    kept: list[str] = []
    for token in head:
        draw = rng.random()
        if draw < strength * 0.5:
            continue  # dropped
        if draw < strength:
            # Swapped for a source-system word, which no glossary name carries. Not
            # repeated within one column: `SOURCE_SOURCE_X` is a generator artefact, and
            # a repeated token would double that word's weight in the lexical arm.
            choices = [w for w in pools.source_words if w not in kept]
            if choices:
                kept.append(rng.choice(choices))
            continue
        kept.append(token)
    return tuple(kept + tail)


def _doc(rng: random.Random, subject: str, class_word: str) -> str:
    return rng.choice(_DOC_FRAMES).format(subject=subject.lower(), kind=class_word.lower())


def _render(tokens: tuple[str, ...], catalog: AbbreviationCatalog, contracted: bool) -> str:
    if contracted:
        return "_".join(catalog.contract_tokens(tokens))
    return "_".join(t.lower() for t in tokens)


@dataclass
class _ChainPool:
    """
    The parent paths a schema reuses.

    Chains are drawn from a bounded pool rather than generated per field, because a schema
    where every one of 800 leaves has its own unique six-level ancestry is not a nested
    schema -- it is 800 unrelated single-column tables sharing a file, and the raw `.avsc`
    it reconstructs to is a fan of 800 disjoint branches (measured: 8 MB for 800 leaves
    before this was shared, 10x smaller after). Reuse is also what gives the repeated-leaf
    and cache-key questions their shape: leaves under a SHARED parent are the ordinary
    case, and it has to be present for the pathological case to stand out from it.
    """

    by_domain: dict[str, tuple[tuple[str, ...], ...]]
    neutral: tuple[tuple[str, ...], ...]

    def for_domain(self, rng: random.Random, domain: str) -> tuple[str, ...]:
        chains = self.by_domain.get(domain) or self.neutral
        return rng.choice(chains) if chains else ()

    def any_neutral(self, rng: random.Random) -> tuple[str, ...]:
        return rng.choice(self.neutral) if self.neutral else ()


def _build_chains(
    rng: random.Random,
    pools: Pools,
    catalog: AbbreviationCatalog,
    profile: SchemaProfile,
    chains_per_domain: int = 3,
    neutral_chains: int = 12,
) -> _ChainPool:
    """Record names for this schema, drawn from a small shared pool of segments."""
    if not profile.depth:
        return _ChainPool(by_domain={}, neutral=())

    segments = tuple(
        _render(
            (rng.choice(pools.subjects), rng.choice(_RECORD_NOUNS).capitalize()),
            catalog,
            profile.contracted,
        )
        for _ in range(max(6, profile.depth * 4))
    )

    def tail(n: int) -> tuple[str, ...]:
        return tuple(rng.choice(segments) for _ in range(n))

    by_domain = {
        domain: tuple(
            (_render((domain,), catalog, profile.contracted), *tail(profile.depth - 1))
            for _ in range(chains_per_domain)
        )
        for domain in pools.domains
    }
    return _ChainPool(
        by_domain=by_domain, neutral=tuple(tail(profile.depth) for _ in range(neutral_chains))
    )


def _flat_row(
    flattened_name: str,
    leaf_name: str,
    data_type: str,
    doc: str,
    is_array: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "flattenedName": flattened_name,
        "dataType": data_type,
        "nullable": True,
    }
    if doc:
        row["doc"] = doc
    if is_array:
        row["isArraySerialized"] = True
    # Not a key the parser knows, so it rides through into `source_metadata` untouched.
    # The repeated-leaf experiment needs to group columns the way a cache keyed on the
    # leaf column name would, and reconstructing that by re-parsing the joined string
    # would be measuring this pack's naming convention rather than the matcher.
    row["leafName"] = leaf_name
    return row


# =============================================================================
# THE FOUR TRUTH CLASSES
# =============================================================================


def _exact_field(
    rng: random.Random,
    pools: Pools,
    catalog: AbbreviationCatalog,
    profile: SchemaProfile,
    row: GlossaryRow,
    strength: float,
    chains: _ChainPool,
) -> tuple[dict[str, Any], TruthRow, str]:
    tokens = _paraphrase(rng, pools, row, strength)
    leaf = _render(tokens, catalog, profile.contracted)
    parents = chains.for_domain(rng, row.domain)
    flattened = "_".join(parents) + ARRAY_BOUNDARY + leaf if parents else leaf
    doc = _doc(rng, row.subject, row.class_word) if rng.random() < profile.doc_share else ""
    return (
        _flat_row(flattened, leaf, row.data_type, doc, bool(parents)),
        TruthRow(
            schema=profile.name,
            flattened_name=flattened,
            field_path=".".join((*parents, leaf)),
            data_type=row.data_type,
            truth_class=TruthClass.EXACT,
            correct_ids=(row.id,),
            note="",
        ),
        leaf,
    )


def _ambiguous_field(
    rng: random.Random,
    pools: Pools,
    catalog: AbbreviationCatalog,
    profile: SchemaProfile,
    glossary: Glossary,
    cluster: tuple[str, ...],
    strength: float,
    chains: _ChainPool,
) -> tuple[dict[str, Any], TruthRow, str]:
    """
    A column whose term is genuinely contested.

    The parent path is DOMAIN-NEUTRAL on purpose: every member of the cluster has the same
    name and differs only by the domain that governs it, so with no domain in the path
    every member is defensible and all of them are recorded. Putting a domain in the path
    would turn this into an EXACT row with extra steps.
    """
    members = [glossary.by_id[i] for i in cluster]
    row = members[0]
    tokens = _paraphrase(rng, pools, row, strength)
    leaf = _render(tokens, catalog, profile.contracted)
    parents = chains.any_neutral(rng)
    flattened = "_".join(parents) + ARRAY_BOUNDARY + leaf if parents else leaf
    doc = _doc(rng, row.subject, row.class_word) if rng.random() < profile.doc_share else ""
    return (
        _flat_row(flattened, leaf, row.data_type, doc, bool(parents)),
        TruthRow(
            schema=profile.name,
            flattened_name=flattened,
            field_path=".".join((*parents, leaf)),
            data_type=row.data_type,
            truth_class=TruthClass.AMBIGUOUS,
            correct_ids=tuple(m.id for m in members),
            note="near-duplicate cluster; no domain in the path, so every member stands",
        ),
        leaf,
    )


def _no_match_field(
    rng: random.Random,
    pools: Pools,
    catalog: AbbreviationCatalog,
    profile: SchemaProfile,
    chains: _ChainPool,
) -> tuple[dict[str, Any], TruthRow, str]:
    """A column built entirely from the held-out orphan vocabulary."""
    orphan = rng.choice(pools.orphans)
    class_word = rng.choice(pools.class_words)
    tokens = (orphan, class_word.long)
    if rng.random() < 0.4:
        tokens = (rng.choice(pools.orphans), *tokens)
    leaf = _render(tokens, catalog, profile.contracted)
    parents = chains.any_neutral(rng)
    flattened = "_".join(parents) + ARRAY_BOUNDARY + leaf if parents else leaf
    doc = _doc(rng, orphan, class_word.long) if rng.random() < profile.doc_share else ""
    return (
        _flat_row(flattened, leaf, "string", doc, bool(parents)),
        TruthRow(
            schema=profile.name,
            flattened_name=flattened,
            field_path=".".join((*parents, leaf)),
            data_type="string",
            truth_class=TruthClass.NO_MATCH,
            correct_ids=(),
            note="orphan vocabulary; no glossary row can describe this at any scale",
        ),
        leaf,
    )


def _trap_field(
    rng: random.Random,
    pools: Pools,
    catalog: AbbreviationCatalog,
    profile: SchemaProfile,
    row: GlossaryRow,
    chains: _ChainPool,
) -> tuple[dict[str, Any], TruthRow, str]:
    """
    High lexical overlap, unrelated meaning, and no correct term.

    The term's qualifiers and class word are kept and its SUBJECT is replaced by an
    orphan. The surface still looks like the term -- most tokens are shared -- and the
    thing the column is about does not exist in the glossary. `trap_id` names the term the
    matcher is expected to return, so the measurement is "how often, and how confidently"
    rather than "did something go wrong".
    """
    orphan = rng.choice(pools.orphans)
    tokens = (*row.tokens[:-2], orphan, row.tokens[-1])
    leaf = _render(tokens, catalog, profile.contracted)
    parents = chains.for_domain(rng, row.domain)
    flattened = "_".join(parents) + ARRAY_BOUNDARY + leaf if parents else leaf
    doc = _doc(rng, orphan, row.class_word) if rng.random() < profile.doc_share else ""
    return (
        _flat_row(flattened, leaf, row.data_type, doc, bool(parents)),
        TruthRow(
            schema=profile.name,
            flattened_name=flattened,
            field_path=".".join((*parents, leaf)),
            data_type=row.data_type,
            truth_class=TruthClass.TRAP,
            correct_ids=(),
            trap_id=row.id,
            note="shares the trap term's qualifiers and class word; the subject is orphan",
        ),
        leaf,
    )


# =============================================================================
# THE REPEATED-LEAF CONSTRUCTION
# =============================================================================


def _repeated_leaf_fields(
    rng: random.Random,
    catalog: AbbreviationCatalog,
    profile: SchemaProfile,
    glossary: Glossary,
    repeats_per_domain: int,
) -> tuple[list[dict[str, Any]], list[TruthRow]]:
    """
    One leaf name per wide cluster, emitted under every domain that governs it.

    `repeats_per_domain` > 1 emits the same (leaf, domain) pair under a second record
    path. Those extra occurrences share their answer with the first, which is deliberate:
    a real schema repeats a column both ways, and a fixture where every repetition has a
    different answer would overstate how visible the collapse is.
    """
    rows: list[dict[str, Any]] = []
    truth: list[TruthRow] = []

    for cluster in glossary.wide_clusters():
        members = [glossary.by_id[i] for i in cluster]
        leaf_tokens = members[0].tokens
        leaf = _render(leaf_tokens, catalog, profile.contracted)
        for member in members:
            for repeat in range(repeats_per_domain):
                noun = _RECORD_NOUNS[repeat % len(_RECORD_NOUNS)]
                parents = (
                    _render((member.domain,), catalog, profile.contracted),
                    _render((noun.capitalize(),), catalog, profile.contracted),
                )
                flattened = "_".join(parents) + ARRAY_BOUNDARY + leaf
                doc = (
                    _doc(rng, member.subject, member.class_word)
                    if rng.random() < profile.doc_share
                    else ""
                )
                rows.append(_flat_row(flattened, leaf, member.data_type, doc, True))
                truth.append(
                    TruthRow(
                        schema=profile.name,
                        flattened_name=flattened,
                        field_path=".".join((*parents, leaf)),
                        data_type=member.data_type,
                        truth_class=TruthClass.EXACT,
                        correct_ids=(member.id,),
                        note="repeated leaf; the parent domain is the only disambiguator",
                    )
                )
    return rows, truth


# =============================================================================
# SCHEMA ASSEMBLY
# =============================================================================


def _fill(
    rng: random.Random,
    pools: Pools,
    catalog: AbbreviationCatalog,
    profile: SchemaProfile,
    glossary: Glossary,
    count: int,
    strength: float,
    shares: dict[TruthClass, float],
    mirror_of: SyntheticSchema | None,
) -> tuple[list[dict[str, Any]], list[TruthRow]]:
    """Emit `count` fields in the class proportions `shares` asks for."""
    if mirror_of is not None:
        # Same columns, rendered through the naming standard instead. The truth is copied
        # rather than re-derived, so the pair is comparable field for field.
        rows: list[dict[str, Any]] = []
        truth: list[TruthRow] = []
        for src_row, src_truth in zip(mirror_of.flattened, mirror_of.truth, strict=True):
            tokens = tuple(t for t in src_truth.flattened_name.split("_") if t)
            leaf = "_".join(catalog.contract_tokens(tokens))
            doc = src_row.get("doc", "") if rng.random() < profile.doc_share else ""
            rows.append(_flat_row(leaf, leaf, src_row["dataType"], doc, False))
            truth.append(
                TruthRow(
                    schema=profile.name,
                    flattened_name=leaf,
                    field_path=leaf,
                    data_type=src_truth.data_type,
                    truth_class=src_truth.truth_class,
                    correct_ids=src_truth.correct_ids,
                    trap_id=src_truth.trap_id,
                    note=f"mirrors {mirror_of.name}:{src_truth.flattened_name}",
                )
            )
        return rows, truth

    exact_pool = glossary.unclustered_approved()
    if not exact_pool:
        raise RuntimeError(
            "the glossary has no approved, unclustered rows, so no EXACT truth can be "
            "written; lower near_duplicate_share or non_approved_share"
        )
    clusters = [c for c in glossary.clusters.values() if 2 <= len(c) <= 4]
    if not clusters:  # a tiny glossary may have none in the preferred width
        clusters = [c for c in glossary.clusters.values() if len(c) >= 2]

    plan: list[TruthClass] = []
    for cls, share in shares.items():
        plan.extend([cls] * round(count * share))
    while len(plan) < count:
        plan.append(TruthClass.EXACT)
    plan = plan[:count]
    rng.shuffle(plan)

    chains = _build_chains(rng, pools, catalog, profile)

    rows = []
    truth = []
    for cls in plan:
        if cls is TruthClass.EXACT:
            row, tr, _leaf = _exact_field(
                rng, pools, catalog, profile, rng.choice(exact_pool), strength, chains
            )
        elif cls is TruthClass.AMBIGUOUS and clusters:
            row, tr, _leaf = _ambiguous_field(
                rng, pools, catalog, profile, glossary, rng.choice(clusters), strength, chains
            )
        elif cls is TruthClass.TRAP:
            row, tr, _leaf = _trap_field(
                rng, pools, catalog, profile, rng.choice(exact_pool), chains
            )
        else:
            row, tr, _leaf = _no_match_field(rng, pools, catalog, profile, chains)
        rows.append(row)
        truth.append(tr)
    return rows, truth


def build_schemas(
    pools: Pools,
    glossary: Glossary,
    catalog: AbbreviationCatalog,
    seed: int,
    scale: float = 1.0,
    paraphrase_strength: float = 0.6,
    repeats_per_domain: int = 2,
    shares: dict[TruthClass, float] | None = None,
) -> tuple[SyntheticSchema, ...]:
    """
    Generate every profile, in a fixed order, with the answers.

    Duplicate flattened names are removed WITHIN a schema before it is returned. A
    genuine duplicate column is a real thing and `match_schema` handles it by suffixing
    the key -- but a duplicate produced by the generator would mean two truth rows
    claiming the same handle, and the harness would score one of them against the other's
    answer. The count that survives is reported, so a profile that lost fields to
    collisions cannot look like one that never had them.
    """
    rng = random.Random(seed ^ 0x5EED_0004)
    shares = shares or DEFAULT_SHARES
    built: dict[str, SyntheticSchema] = {}
    out: list[SyntheticSchema] = []

    for profile in PROFILES:
        count = max(1, int(profile.leaves * scale))
        schema = SyntheticSchema(name=profile.name, profile=profile)

        if profile.repeated:
            rows, truth = _repeated_leaf_fields(rng, catalog, profile, glossary, repeats_per_domain)
            remaining = max(0, count - len(rows))
            more_rows, more_truth = _fill(
                rng,
                pools,
                catalog,
                profile,
                glossary,
                remaining,
                paraphrase_strength,
                shares,
                None,
            )
            rows.extend(more_rows)
            truth.extend(more_truth)
        elif profile.mixed:
            rows, truth = [], []
            # Proportions: the mixture a production run actually presents -- mostly
            # nested and contracted, a minority readable, a real slice with no doc.
            for donor, portion in (
                ("nested-deep", 0.40),
                ("no-doc", 0.25),
                ("nested-repeated", 0.15),
                ("flat-contracted", 0.12),
                ("flat-english", 0.08),
            ):
                donor_profile = next(p for p in PROFILES if p.name == donor)
                sub = SchemaProfile(
                    name=profile.name,
                    leaves=int(count * portion),
                    depth=donor_profile.depth,
                    contracted=donor_profile.contracted,
                    doc_share=donor_profile.doc_share,
                )
                sub_rows, sub_truth = _fill(
                    rng,
                    pools,
                    catalog,
                    sub,
                    glossary,
                    sub.leaves,
                    paraphrase_strength,
                    shares,
                    None,
                )
                rows.extend(sub_rows)
                truth.extend(sub_truth)
        else:
            mirror = built.get(profile.mirrors) if profile.mirrors else None
            rows, truth = _fill(
                rng,
                pools,
                catalog,
                profile,
                glossary,
                count,
                paraphrase_strength,
                shares,
                mirror,
            )

        seen: set[str] = set()
        kept_rows: list[dict[str, Any]] = []
        kept_truth: list[TruthRow] = []
        for row, tr in zip(rows, truth, strict=True):
            if row["flattenedName"] in seen:
                continue
            seen.add(row["flattenedName"])
            kept_rows.append(row)
            kept_truth.append(tr)

        schema.flattened = kept_rows
        schema.truth = kept_truth
        built[profile.name] = schema
        out.append(schema)

    return tuple(out)


# =============================================================================
# RAW AVRO RECONSTRUCTION
# =============================================================================


def _avro_type(data_type: str) -> Any:
    return {
        "string": "string",
        "integer": "int",
        "long": "long",
        "decimal": "double",
        "double": "double",
        "boolean": "boolean",
        "date": {"type": "int", "logicalType": "date"},
        "timestamp": {"type": "long", "logicalType": "timestamp-millis"},
    }.get(data_type, "string")


def _to_avro(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Rebuild a nested record from flattened names.

    The `__` boundary becomes an array of records, which is what put it there; every other
    `_` inside the parent portion becomes a nested record level. Leaf names keep their
    underscores, because a flattener joins the parent path to the leaf and does not
    rewrite the leaf itself.
    """
    root: dict[str, Any] = {}
    for row in rows:
        flattened = row["flattenedName"]
        leaf = row.get("leafName") or flattened
        parent_part = flattened[: -(len(leaf) + len(ARRAY_BOUNDARY))] if leaf != flattened else ""
        segments = [s for s in parent_part.split("_") if s]
        node = root
        for segment in segments:
            node = node.setdefault(segment, {})
        node[leaf] = row

    def build(node: dict[str, Any], record_name: str, path: str) -> dict[str, Any]:
        fields = []
        for key, value in node.items():
            if isinstance(value, dict) and "flattenedName" in value:
                entry: dict[str, Any] = {
                    "name": key,
                    "type": ["null", _avro_type(value["dataType"])],
                    "default": None,
                }
                if value.get("doc"):
                    entry["doc"] = value["doc"]
                fields.append(entry)
            else:
                child = build(value, f"{record_name}_{key}", f"{path}.{key}")
                fields.append(
                    {"name": key, "type": {"type": "array", "items": child}, "default": []}
                )
        return {"type": "record", "name": record_name, "fields": fields}

    schema = build(root, name.replace("-", "_"), name)
    schema["namespace"] = "synthetic.pack"
    schema["doc"] = (
        "SYNTHETIC. Generated by benchmarks/synthetic. Every name in this file was "
        "manufactured; it describes no real system."
    )
    return schema
