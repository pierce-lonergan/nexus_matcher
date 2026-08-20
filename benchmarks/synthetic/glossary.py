"""
benchmarks.synthetic.glossary | Layer: BENCHMARK
Artifact 1 of 5: the synthetic glossary.

The row count is the least interesting parameter here. What makes this a test rather
than a pile of strings is the DISTRIBUTION, and each of the five properties below exists
because a matcher behaves differently when it holds and when it does not:

  near-duplicate clusters   terms identical except for the domain that owns them. This is
                            the condition a domain prior exists to resolve; without it a
                            prior has nothing to do and measuring one is theatre.
  class-word distribution   a short head (Identifier / Code / Name / Date) and a long
                            tail. A uniform distribution would make the class word a free
                            discriminator.
  name length               median 4 tokens, tail to 9. Lexical scoring is length
                            sensitive, and a corpus of uniformly 3-token names hides it.
  non-approved share        12% drafts and retired terms, which compete as real terms
                            because they ARE real terms. This is what makes row admission
                            (`load_entries(admit=...)`) a measurable feature rather than
                            a defensive habit.
  definition echoes name    ~20% of definitions restate the name and add nothing. Real
                            glossaries are full of these and they are the rows a
                            description-weighted scorer over-trusts.

What is deliberately EMPTY
--------------------------
`logical_name`. `DictionaryEntry.to_searchable_text()` embeds it, and the technical name
is what the query side is built from -- so populating it would put the query string
inside the indexed document and the corpus would score near ceiling while measuring
string identity. That is the exact leak this repository already shipped once and had to
withdraw (see the OMOP split note in benchmarks/eval_harness.py). The column is present
and blank so that anyone who fills it in has to do so deliberately.

Two delimiters, on purpose
--------------------------
`sample_values` is comma-separated and `enum_values` in the same file is
semicolon-separated. `synonyms` is semicolon-separated as well. Reading either mapped
column with the other's separator yields one value per row containing every element --
which indexes, matches, and is simply wrong, with no error anywhere. `load_entries`
refuses that under `delimiter_strict`; the pack ships the trap so the refusal has
something to fire on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .pools import ClassWord, Pools

# Total tokens in a name -> relative weight. Median 4, tail to 9, which is the shape a
# naming standard produces once qualifiers are allowed to stack.
_NAME_LENGTH_WEIGHTS: dict[int, int] = {2: 8, 3: 20, 4: 30, 5: 20, 6: 10, 7: 6, 8: 4, 9: 2}

_STATUS_APPROVED = "Approved"
_STATUS_DRAFT = "Draft"
_STATUS_RETIRED = "Retired"

# Class word -> the data type a standard would give it. Drives sample values and the
# type-compatibility signal, and gives the type column something better than one value.
_TYPE_FOR_CLASS_WORD: dict[str, str] = {
    "Identifier": "string",
    "Code": "string",
    "Name": "string",
    "Date": "date",
    "Amount": "decimal",
    "Indicator": "boolean",
    "Text": "string",
    "Number": "string",
    "Quantity": "integer",
    "Timestamp": "timestamp",
    "Description": "string",
    "Percent": "decimal",
    "Rate": "decimal",
    "Status": "string",
    "Type": "string",
    "Value": "string",
    "Flag": "boolean",
    "Count": "integer",
    "Address": "string",
    "Duration": "integer",
    "Ratio": "decimal",
    "Score": "decimal",
    "Category": "string",
}

# Ordinary connective prose. Not manufactured, and not from anywhere: these are the
# sentence frames every definition in every glossary is assembled from.
_OPENINGS: tuple[str, ...] = (
    "The value that records",
    "The attribute that captures",
    "The element that identifies",
    "The property that describes",
    "The item that represents",
    "The reference that resolves",
)
_CLAUSES: tuple[str, ...] = (
    "It is captured at the point the record is first created and is not revised afterwards.",
    "It is maintained by the owning team and reviewed on the published governance cycle.",
    "It is populated for every instance and carries no default when the source is silent.",
    "It is derived from the upstream feed and reconciled against the register each period.",
    "It is unique within its owning domain and is not guaranteed unique outside it.",
    "It is optional on intake and becomes mandatory once the record has been confirmed.",
)


@dataclass(frozen=True)
class GlossaryRow:
    """One glossary term, with the generation facts downstream artifacts need."""

    id: str
    name: str
    definition: str
    domain: str
    status: str
    data_type: str
    synonyms: tuple[str, ...]
    sample_values: tuple[str, ...]
    enum_values: tuple[str, ...]
    # Generation metadata. Not written to the CSV -- the CSV is a glossary, and a glossary
    # does not carry the answers.
    tokens: tuple[str, ...] = ()
    subject: str = ""
    class_word: str = ""
    cluster_id: str = ""
    definition_echoes_name: bool = False

    @property
    def is_approved(self) -> bool:
        return self.status == _STATUS_APPROVED


@dataclass
class GlossaryProfile:
    """
    The dials. Defaults reproduce the shape the pack specification asks for; every one of
    them is a parameter because "at what scale does this stop being true?" is the question
    the whole pack exists to answer.
    """

    rows: int = 10_000
    near_duplicate_share: float = 0.15
    cluster_size_min: int = 2
    cluster_size_max: int = 6
    non_approved_share: float = 0.12
    definition_echo_share: float = 0.20
    max_synonyms: int = 3
    max_sample_values: int = 5
    # A handful of clusters far wider than the rest. These are what the repeated-leaf
    # schema is built from: one term name governed separately in `wide_cluster_size`
    # domains means one leaf name whose correct answer is decided ENTIRELY by its parent,
    # which is the fixture cache-key composition is tested against. A cluster differs only
    # by domain, so its size cannot exceed the number of domains -- the value is clamped
    # rather than silently truncated, and the achieved width is reported in the manifest.
    wide_cluster_count: int = 4
    wide_cluster_size: int = 30

    def scaled(self, difficulty: float) -> GlossaryProfile:
        """
        A harder or easier variant of this profile.

        `difficulty` 1.0 is the specification's shape. Above 1.0 the ambiguity knobs move
        together -- more near-duplicates, more drafts competing in the index, more
        tautological definitions -- because those are the three properties that make a
        glossary hard to match against, and turning one without the others produces a
        corpus that is hard in one axis and trivially easy in the rest.
        """
        return GlossaryProfile(
            rows=self.rows,
            near_duplicate_share=min(0.60, self.near_duplicate_share * difficulty),
            cluster_size_min=self.cluster_size_min,
            cluster_size_max=self.cluster_size_max,
            non_approved_share=min(0.50, self.non_approved_share * difficulty),
            definition_echo_share=min(0.60, self.definition_echo_share * difficulty),
            max_synonyms=self.max_synonyms,
            max_sample_values=self.max_sample_values,
            wide_cluster_count=self.wide_cluster_count,
            wide_cluster_size=self.wide_cluster_size,
        )


@dataclass
class Glossary:
    """The generated rows plus the indexes the other generators read."""

    rows: tuple[GlossaryRow, ...]
    profile: GlossaryProfile
    by_id: dict[str, GlossaryRow] = field(default_factory=dict)
    clusters: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {row.id: row for row in self.rows}
        clusters: dict[str, list[str]] = {}
        for row in self.rows:
            if row.cluster_id:
                clusters.setdefault(row.cluster_id, []).append(row.id)
        self.clusters = {k: tuple(v) for k, v in clusters.items()}

    @property
    def approved(self) -> tuple[GlossaryRow, ...]:
        return tuple(r for r in self.rows if r.is_approved)

    def wide_clusters(self) -> tuple[tuple[str, ...], ...]:
        """The deliberately wide near-duplicate clusters, widest first.

        Their ids start with 'W'. Keyed on the id rather than on a recomputed width so
        that a cluster which came out narrower than asked for (fewer domains than
        `wide_cluster_size`) is still the one the repeated-leaf schema uses, rather than
        the schema silently falling back to an ordinary cluster and measuring nothing.
        """
        wide = [(cid, ids) for cid, ids in self.clusters.items() if cid.startswith("W")]
        wide.sort(key=lambda kv: (-len(kv[1]), kv[0]))
        return tuple(ids for _cid, ids in wide)

    def unclustered_approved(self) -> tuple[GlossaryRow, ...]:
        """Approved rows with no near-duplicate twin -- the pool EXACT truth draws from.

        A row inside a cluster cannot carry an unambiguously-correct answer: by
        construction two to six rows describe the same thing under different domains, and
        calling one of them the single right answer would make the AMBIGUOUS class a
        scoring bug rather than a property of the data.
        """
        return tuple(r for r in self.approved if not r.cluster_id)


def _weighted_choice(rng: random.Random, weights: dict[int, int]) -> int:
    total = sum(weights.values())
    pick = rng.randrange(total)
    upto = 0
    for value, weight in weights.items():
        upto += weight
        if pick < upto:
            return value
    return next(iter(weights))


def _pick_class_word(rng: random.Random, class_words: tuple[ClassWord, ...]) -> ClassWord:
    total = sum(cw.weight for cw in class_words)
    pick = rng.randrange(total)
    upto = 0
    for cw in class_words:
        upto += cw.weight
        if pick < upto:
            return cw
    return class_words[0]


def _sample_values(rng: random.Random, data_type: str, count: int) -> tuple[str, ...]:
    """Type-plausible example values. Never comma-bearing: the file uses ',' as one of
    its two multi-value separators and a value containing one would make the trap
    ambiguous instead of sharp."""
    out: list[str] = []
    for _ in range(count):
        if data_type == "date":
            out.append(
                f"20{rng.randrange(10, 26):02d}-{rng.randrange(1, 13):02d}-"
                f"{rng.randrange(1, 29):02d}"
            )
        elif data_type == "timestamp":
            out.append(
                f"20{rng.randrange(10, 26):02d}-{rng.randrange(1, 13):02d}-"
                f"{rng.randrange(1, 29):02d}T{rng.randrange(0, 24):02d}:00:00Z"
            )
        elif data_type in ("decimal",):
            out.append(f"{rng.randrange(0, 100000) / 100:.2f}")
        elif data_type == "integer":
            out.append(str(rng.randrange(0, 5000)))
        elif data_type == "boolean":
            out.append(rng.choice(("Y", "N")))
        else:
            out.append("".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)))
    return tuple(out)


def _definition(
    rng: random.Random,
    name: str,
    subject: str,
    class_word: str,
    domain: str,
    echoes: bool,
) -> str:
    """A definition of 80-250 characters, or a tautological restatement of the name."""
    if echoes:
        text = (
            f"{name}. This term carries the {class_word.lower()} named above and is "
            f"recorded against the {domain} domain."
        )
    else:
        text = (
            f"{rng.choice(_OPENINGS)} the {subject.lower()} {class_word.lower()} for a "
            f"record governed by the {domain} domain. {rng.choice(_CLAUSES)}"
        )
    # The band is a property of the corpus, so clip rather than hope. Truncation lands on
    # a word boundary; a definition ending mid-token would tokenise differently from every
    # other row and become its own accidental signal.
    if len(text) > 250:
        text = text[:250].rsplit(" ", 1)[0]
    while len(text) < 80:
        text += " The definition is maintained in the owning domain's register."
    return text[:250].rsplit(" ", 1)[0] if len(text) > 250 else text


def _name_tokens(
    rng: random.Random,
    pools: Pools,
    subject: str,
    class_word: ClassWord,
) -> tuple[str, ...]:
    total = _weighted_choice(rng, _NAME_LENGTH_WEIGHTS)
    n_subjects = 2 if total >= 5 and rng.random() < 0.35 else 1
    n_qualifiers = max(0, total - 1 - n_subjects)
    qualifiers = rng.sample(pools.qualifiers, min(n_qualifiers, len(pools.qualifiers)))
    subjects = [subject]
    if n_subjects == 2:
        subjects.append(rng.choice(pools.subjects))
    return (*qualifiers, *subjects, class_word.long)


def _synonyms(
    rng: random.Random, pools: Pools, tokens: tuple[str, ...], count: int
) -> tuple[str, ...]:
    """Alternative spellings of the same term: a qualifier dropped, or one swapped."""
    out: list[str] = []
    for _ in range(count):
        parts = list(tokens)
        if len(parts) > 2 and rng.random() < 0.5:
            parts.pop(0)
        elif len(parts) > 2:
            parts[0] = rng.choice(pools.qualifiers)
        candidate = " ".join(parts)
        if candidate not in out:
            out.append(candidate)
    return tuple(out)


def build_glossary(
    pools: Pools,
    profile: GlossaryProfile,
    seed: int,
) -> Glossary:
    """
    Generate the glossary.

    Ids are six-digit zero-padded and assigned in emission order, so a row's id is stable
    for a given (seed, profile) and two runs produce the same file byte for byte.
    """
    rng = random.Random(seed ^ 0x5EED_0001)
    rows: list[GlossaryRow] = []
    next_id = 1

    n_clustered_target = int(profile.rows * profile.near_duplicate_share)
    clustered_emitted = 0
    cluster_index = 0

    # The wide clusters come first, so their ids are stable when the row count changes.
    # That matters more than it looks: the scale experiment holds the query set fixed and
    # grows the glossary, and it can only do that if a term keeps its id across sizes.
    wide_width = min(profile.wide_cluster_size, len(pools.domains))
    for _ in range(profile.wide_cluster_count):
        if len(rows) + wide_width > profile.rows:
            break
        cluster_index += 1
        cluster_id = f"W{cluster_index:05d}"
        subject = rng.choice(pools.subjects)
        class_word = _pick_class_word(rng, pools.class_words)
        tokens = _name_tokens(rng, pools, subject, class_word)
        name = " ".join(tokens)
        data_type = _TYPE_FOR_CLASS_WORD.get(class_word.long, "string")
        for domain in rng.sample(list(pools.domains), wide_width):
            rows.append(
                GlossaryRow(
                    id=f"{next_id:06d}",
                    name=name,
                    # Never echoing and always approved. This cluster is the answer key
                    # for the repeated-leaf fixture, and a member that row admission
                    # removes -- or whose definition says only its own name -- would make
                    # a cache-key measurement report a different feature's effect.
                    definition=_definition(rng, name, subject, class_word.long, domain, False),
                    domain=domain,
                    status=_STATUS_APPROVED,
                    data_type=data_type,
                    synonyms=(),
                    sample_values=_sample_values(rng, data_type, 2),
                    enum_values=("Y", "N", "U") if data_type == "boolean" else (),
                    tokens=tokens,
                    subject=subject,
                    class_word=class_word.long,
                    cluster_id=cluster_id,
                    definition_echoes_name=False,
                )
            )
            next_id += 1
            clustered_emitted += 1

    while len(rows) < profile.rows:
        subject = rng.choice(pools.subjects)
        class_word = _pick_class_word(rng, pools.class_words)
        tokens = _name_tokens(rng, pools, subject, class_word)
        name = " ".join(tokens)
        data_type = _TYPE_FOR_CLASS_WORD.get(class_word.long, "string")

        # A cluster is the SAME name under several domains. Nothing else varies, which is
        # what makes retrieval score alone unable to separate the members.
        in_cluster = clustered_emitted < n_clustered_target
        if in_cluster:
            cluster_index += 1
            cluster_id = f"C{cluster_index:05d}"
            size = rng.randint(profile.cluster_size_min, profile.cluster_size_max)
            size = min(size, profile.rows - len(rows))
            domains = rng.sample(list(pools.domains), min(size, len(pools.domains)))
        else:
            cluster_id = ""
            size = 1
            domains = [rng.choice(pools.domains)]

        for domain in domains:
            echoes = rng.random() < profile.definition_echo_share
            status = _STATUS_APPROVED
            if rng.random() < profile.non_approved_share:
                status = _STATUS_DRAFT if rng.random() < 0.67 else _STATUS_RETIRED

            n_syn = rng.randint(0, profile.max_synonyms)
            n_sample = rng.randint(0, profile.max_sample_values)
            rows.append(
                GlossaryRow(
                    id=f"{next_id:06d}",
                    name=name,
                    definition=_definition(rng, name, subject, class_word.long, domain, echoes),
                    domain=domain,
                    status=status,
                    data_type=data_type,
                    synonyms=_synonyms(rng, pools, tokens, n_syn),
                    sample_values=_sample_values(rng, data_type, n_sample),
                    # A second ordered multi-value column, present on the enumerated
                    # types only -- which is where a real glossary carries one.
                    enum_values=(("Y", "N", "U") if data_type == "boolean" else ()),
                    tokens=tokens,
                    subject=subject,
                    class_word=class_word.long,
                    cluster_id=cluster_id,
                    definition_echoes_name=echoes,
                )
            )
            next_id += 1
            if in_cluster:
                clustered_emitted += 1
            if len(rows) == profile.rows:
                break

    return Glossary(rows=tuple(rows), profile=profile)


# The CSV header, and the mapping a caller has to pass `load_entries` to read it.
#
# `sample_values` and `enum_values` are NEVER inferred by `map_columns` -- inferring a
# multi-value column would change the text an existing glossary embeds -- so a caller who
# wants them has to name them, and naming them is also where the separator is declared.
CSV_HEADER: tuple[str, ...] = (
    "id",
    "business_name",
    "logical_name",
    "definition",
    "data_type",
    "domain",
    "status",
    "synonyms",
    "sample_values",
    "enum_values",
)

COLUMN_MAPPING: dict[str, str] = {
    "id": "id",
    "business_name": "business_name",
    "definition": "definition",
    "data_type": "data_type",
    "domain": "domain",
    "sample_values": "sample_values",
    "enum_values": "enum_values",
}

VALUE_DELIMITERS: dict[str, str] = {"sample_values": ",", "enum_values": ";"}

ADMIT_APPROVED: dict[str, set[str]] = {"status": {_STATUS_APPROVED}}


def glossary_rows_as_dicts(glossary: Glossary) -> list[dict[str, str]]:
    """The rows in the shape `load_entries` reads, without going through a file.

    Used by the experiments that need several glossaries in one process: writing and
    re-reading 100,000 rows three times to vary one flag is a minute of disk for no
    information.
    """
    return [
        {
            "id": row.id,
            "business_name": row.name,
            "logical_name": "",
            "definition": row.definition,
            "data_type": row.data_type,
            "domain": row.domain,
            "status": row.status,
            "synonyms": ";".join(row.synonyms),
            "sample_values": ",".join(row.sample_values),
            "enum_values": ";".join(row.enum_values),
        }
        for row in glossary.rows
    ]
