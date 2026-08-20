# Governed abbreviation expansion

**This capability already ships and was documented nowhere.** If your schemas are written
in an in-house naming standard — `PAX_LGL_NM` rather than `passenger_legal_name` — you can
hand `NexusMatcher` your approved-abbreviation catalog and it will expand the query text
before retrieval. Nothing needs to be added to the library.

It is **off by default and should stay off** unless you have measured it on your own
fields. This page is how to turn it on, what it is worth, and the three reasons the
published numbers overstate it.

Contents:

1. [Why this page is in `docs/guides/`](#why-this-page-is-in-docsguides)
2. [The wiring](#the-wiring)
3. [What it is worth, and what the number is not](#what-it-is-worth-and-what-the-number-is-not)
4. [What is untested](#what-is-untested)
5. [Before you enable it on real data](#before-you-enable-it-on-real-data)

---

## Why this page is in docs/guides/

Three homes were considered and rejected.

**Not `README.md`.** The README is the wheel's `long_description` and therefore the PyPI
landing page. This feature is off by default, requires the caller to supply a catalog the
library deliberately does not ship, and cannot be written honestly in fewer than about
eighty lines because most of those lines are caveats. This repository has already published
a "100% Precision@1" headline that was seventeen hand-written pairs, and a P@1 of 0.715
that was a benchmark leak. A capability whose honest write-up is 20% capability and 80%
caveat does not belong in the shop window; putting it there is how the third one of those
happens.

**Not `docs/GOVERNANCE.md`.** That document is about the caller-supplied controlled
vocabulary of *protection classes* — what a matched field inherits. An approved-abbreviation
catalog is also caller-supplied governed vocabulary, which makes the pull real, but it acts
on *retrieval*, not on classification, and conflating the two would invite a reader to
believe an abbreviation catalog carries governance weight. It does not.

**Not `docs/modules/abbreviation_expansion.md`.** That is a component reference — domain
model, invariants, states. This is a task: *I have an in-house naming standard, what do I
do*. The component doc should link here; it should not become a how-to.

`docs/guides/` is a task-shaped home for a task-shaped question, and this is its first
occupant. Every measured number lives in
[BENCHMARK_REGISTRY.md](../BENCHMARK_REGISTRY.md#exp-governed-abbrev--what-a-caller-supplied-abbreviation-catalog-is-worth)
and is *referenced* from here rather than restated, because this repository still carries
72 recorded cases of the same number drifting apart in two documents.

---

## The wiring

Two objects and one flag. The catalog is a plain `dict[str, str]` of short form to long
form; case does not matter on the key.

```python
from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.domain.services.abbreviation import (
    AbbreviationDictionary,
    AbbreviationExpander,
)
from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import (
    CsvDictionaryLoader,
)
from nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx import (
    default_embedding_provider,
)
from nexus_matcher.infrastructure.adapters.schema_parsers.flattened_avro import (
    FlattenedAvroParser,
)
from nexus_matcher.infrastructure.adapters.sparse_retrievers.bm25 import BM25Retriever
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore

# YOUR approved-abbreviation catalog. This library ships no naming standard, in the same
# way and for the same reason it ships no protection-class taxonomy.
#
# The rows below are FICTIONAL, from the Gravel Bay Ferry Authority example pack. Replace
# all of them. Do not treat this as a starting list.
FERRY_CATALOG = {
    "PAX": "passenger",
    "LGL": "legal",
    "NM": "name",
    "ADDR": "address",
    "MOB": "mobile",
    "CERT": "certificate",
    "ASGN": "assignment",
    "VSL": "vessel",
    "DEP": "departure",
}

expander = AbbreviationExpander(AbbreviationDictionary.from_dict(FERRY_CATALOG))

provider = default_embedding_provider()
matcher = NexusMatcher(
    embedding_provider=provider,
    vector_store=InMemoryVectorStore(
        VectorStoreConfig(collection_name="dictionary", dimension=provider.dimension)
    ),
    sparse_retriever=BM25Retriever(),
    schema_parser_registry={"flattened_avro": FlattenedAvroParser()},
    dictionary_loader_registry={"csv": CsvDictionaryLoader()},
    abbreviation_expander=expander,
    config=MatchingConfig(expand_query_abbreviations=True),
)
matcher.load_dictionary("examples/governance/glossary.csv")
results = matcher.match_schema("your_flattened_schema.json")
```

### from_config cannot reach this feature

`NexusMatcher.from_config()` takes `config` and `governance` and **not**
`abbreviation_expander`. It builds the expander itself, from the bundled generic dictionary,
and there is no setter afterwards. So the one-line factory every other page in this
repository recommends is the one path that cannot turn governed expansion on — which is a
large part of why the capability went undocumented. Assembling the five components by hand,
as above, is currently the only way in.

That is a DX defect, not a design decision. It is recorded here rather than fixed because
fixing it changes a public signature.

### What the expander actually does

`AbbreviationExpander` is an **exact dictionary lookup with passthrough**: one lookup per
token, lowercased; on a miss the token is emitted unchanged. Its docstring states the
invariant — *"Unknown abbreviations pass through unchanged."* It does not guess, and it does
not score.

```
>>> expander.expand("pax lgl nm zzz").expanded
'passenger legal name zzz'
```

`zzz` is not in the catalog and survives untouched. That property is the whole safety
argument: a catalog with 40% coverage costs you the 60% it does not know about, not the
100%.

### What it does to the query, on real data

Running the snippet above against the fictional ferry glossary, with
`expand_query_abbreviations` off and then on:

| Field | Query text, hook off | Query text, hook on |
|---|---|---|
| `bkg_pax__lgl_nm` | `bkg, pax, lgl nm array` | `bkg, passenger, legal name array` |
| `bkg_pax__postal_addr` | `bkg, pax, postal addr array` | `bkg, passenger, postal address array` |
| `crew__cert_no` | `crew, cert no array` | `crew, certificate no array` |
| `crew__shift_asgn` | `crew, shift asgn array` | `crew, shift assignment array` |

Three things to read off that table.

`bkg` and `no` have no catalog row, so they **pass through untouched**. That is the
passthrough guarantee doing its job, not a bug.

`pax` sits in the **parent path**, and it expands. It did not always: expansion used to run
over the whole enriched string, so the comma separating hierarchy levels rode along on the
token in front of it and the expander looked up `"pax,"`, missed, and passed it through —
while the identical token expanded correctly everywhere else in the same query. The token
that silently failed was always a parent-path level, which is the part of the query
carrying the most signal. That is fixed; `_build_query_text` now expands per level.

`array` is not from the expander. The flattened-Avro parser defaults `is_array` to true
when the schema does not say otherwise, and the context enricher appends the word. It is
noise in this example and unrelated to abbreviations.

The cosine similarity to the correct glossary entry rose from 0.57 to 0.68 on the first
field. It **fell**, from 0.74 to 0.69, on the second — expanding `addr` to `address` moved
that query slightly away from an entry whose business name is "Passenger Postal Address".
Both still ranked first. Expansion is not uniformly positive even when every row it fires
is correct.

> **None of those cosine figures is a benchmark.** They come from running the snippet on
> one machine against four hand-picked fields and a thirty-entry glossary. Nothing was
> persisted, so by this repository's own grading they are D-grade and must not be cited. All
> four fields already matched correctly with the hook off, and `final_confidence` did not
> move at all — on a corpus this small the fused retrieval score is min-max normalised to
> 1.0 at rank 1, so confidence saturates and hides the change entirely. That saturation is
> also why you cannot evaluate this feature by eyeballing confidences on a small glossary.
> The measured numbers are in the registry.

---

## What it is worth, and what the number is not

The measured trade lives in
[BENCHMARK_REGISTRY.md → EXP-GOVERNED-ABBREV](../BENCHMARK_REGISTRY.md#exp-governed-abbrev--what-a-caller-supplied-abbreviation-catalog-is-worth),
with the artifact names and the McNemar tests. In summary, and in the order that matters
for a decision:

**1. Abbreviation is the largest single effect measured in this repository.** A fully
abbreviated schema costs about **46.5 points of P@1** on the combined benchmark and **23.4**
on FHIR. For comparison, parent-path context — described everywhere in this repo as the
largest accuracy factor in the pipeline — is worth about **19.9**. If your schemas are
genuinely governed by a naming standard, this is the biggest lever available to you.

**2. Recovery tracks catalog coverage roughly linearly.** With 75%, 50% and 25% of the
catalog rows present, roughly 72%, 46% and 22% of the loss comes back on the combined
corpus (68/38/15 on FHIR). Missing rows cost about what you would guess.

**3. It tolerates staleness much better than absence.** A catalog with 5% of its rows
pointing at the wrong long form still recovers about 91%. At 25% wrong, about 64%.
Break-even against *not expanding at all* sits near **75% wrong**. At 100% wrong the
combined corpus lands about **7 points of P@1 below leaving the abbreviations alone** — so
expansion is a bet on the catalog, not a free hedge.

**4. Appending the expansion instead of substituting it is not free.** Keeping the raw
abbreviation alongside its expansion costs about 9 points against substitution on a
complete catalog, and only wins past 75% wrong. This model dislikes longer queries; the
same effect is measured for appended type words.

### The caveat that matters most

The recovery figures come from a **synthetic** experiment, and they are partly circular.

The abbreviations were generated *from the gold text* by a mechanical rule, and the catalog
is the exact inverse of that rule. So expanding does not recover *meaning* — it reconstructs
the original *string*. Measured directly, with the shipped `AbbreviationExpander` and the
full catalog, the expanded query is caselessly identical to the original for **683 of 688**
queries on the combined corpus and **1556 of 1556** on FHIR. The "100% of the gap
recovered" headline is `f⁻¹(f(x)) == x`. It demonstrates that the plumbing is lossless and
nothing else.

What survives that objection:

- the **coverage curve** and the **wrong-rate curve**, because they vary a real quantity
  (which rows exist, which rows lie) and produce an ordered response;
- the **direction and rough magnitude** of the damage figure, though the abbreviation
  scheme used — devowel to four characters — is harsher than any human naming standard, so
  46.5 points is an upper bound;
- and the fact that the **generic bundled dictionary recovers almost none of it** (4.7% on
  combined, 2.7% on FHIR), which is a real result about catalogs rather than about plumbing.

What does not survive: any claim of the form "governed expansion recovers ~100% of the loss
on your data".

---

## What is untested

**Colliding short forms.** `ST` is *State* in one column and *Street* in the next. The
synthetic catalog is **injective by construction** — one short form, exactly one long form,
guaranteed by the generator. Real approved-abbreviation catalogs are not, and no catalog row
can be correct in both places.

The shipped expander cannot help here even in principle: it does one exact dictionary lookup
per token with no context, so a colliding short form resolves the same way everywhere. This
is the single most likely reason the numbers above would fail to transfer to a real catalog,
and nobody has measured it — not in this repository and not in the library evaluation that
produced these figures.

If your standard has collisions, the honest options are to leave those rows out of the
catalog (they pass through untouched, which is the safe failure), or to resolve them
upstream of the matcher where you know which table you are in.

Also unmeasured: multi-word long forms, non-ASCII short forms, and interaction with
`dictionary_alias_count` (which is off by default for unrelated reasons).

---

## Before you enable it on real data

The public benchmarks cannot answer the question for you. The value, if it exists, is
entirely in your own field names.

1. **Export ~200 real field names and the glossary term each should map to.** This is the
   only non-synthetic measurement available to you.
2. **Measure P@1 with the names as-is, then with the names expanded by hand.** That gap is
   the real prize. If it is small, your schemas are not as abbreviated as you think and the
   rest of this page does not apply.
3. **Measure your catalog's wrong-rate against those fields** — the fraction of rows
   asserting a long form that is wrong *in this schema's context*. That is the number that
   predicts anything. Below 25% wrong keeps roughly two-thirds of the win; near 75% is
   break-even.
4. **Count your collisions**, per the section above.
5. **Use a paired test on the full set**, not a subsample. A 300-query fixture in this
   repository once read a change as −1.33 points while the full corpus read the same change
   as +0.58.
6. **Re-run `benchmarks/exp_calibration.py` afterwards, without exception.** A large P@1
   shift moves the whole score distribution, pushes more candidates over a fixed threshold,
   and can *lower* auto-approve precision. That has already happened twice here. Enabling
   this hook without re-calibrating is how you trade retrieval accuracy for silently worse
   auto-approvals.

The default stays `expand_query_abbreviations = False`. On the bundled generic dictionary
the flag is measurably not worth turning on — see the comment on the field in
`MatchingConfig`, and `docs/API_REFERENCE.md`.
