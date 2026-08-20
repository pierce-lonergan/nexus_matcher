# The synthetic corpus generator

`benchmarks/synthetic/` generates a schema-matching corpus that reproduces the statistical
properties of a large enterprise glossary **without containing one**. It exists because the
accuracy questions that matter most on a big corpus cannot be asked of the public
benchmarks at all, and because the corpora that could answer them are the ones that can
never be shared.

It is a **generator**, not a fixture. Scale and difficulty are dials, and every artifact is
a pure function of `(rows, difficulty, seed)`. Two runs of one spec produce byte-identical
files.

> **Everything it produces is synthetic and says so.** Subject words, domains and
> abbreviations come out of a seeded syllable grammar in `benchmarks/synthetic/pools.py`.
> Nothing was sampled from any glossary. Every emitted file carries a `notice` field
> repeating that, because a corpus that *looks* real is exactly the kind of file someone
> later mistakes for one.

Contents:

1. [Why synthetic, and why generated](#why-synthetic-and-why-generated)
2. [The five artifacts](#the-five-artifacts)
3. [The two traps](#the-two-traps)
4. [Running it](#running-it)
5. [Verification, and what it caught](#verification-and-what-it-caught)
6. [The experiments, and what they measured](#the-experiments-and-what-they-measured)
7. [What this pack does not prove](#what-this-pack-does-not-prove)

---

## Why synthetic, and why generated

`benchmarks/eval_harness.py` measures this library on ~800 labelled pairs from BIRD-SQL and
the OMOP CDM. Those are good benchmarks and they are structurally unable to exercise
several things this library ships:

| Property | Public benchmarks | A large governed glossary |
|---|---|---|
| Dictionary size | hundreds | tens or hundreds of thousands |
| Field naming | readable English | contracted through a naming standard |
| Near-duplicate terms | few | many, separated only by which domain owns them |
| Repeated leaf names | few | one column name under dozens of parents |
| Doc coverage | high | many leaves have none |
| Rows that are not approved | none | a real share of every export |
| Fields with **no** correct term | none | a real share of every schema |
| Reviewer history | none | the primary long-term signal |

A matcher can score well on the first column and fail on the first schema from the second,
because the benchmarks never present a column with no English in it, no documentation, and
forty plausible glossary neighbours.

**Generated rather than committed** for one reason: the interesting question about almost
every threshold in this library is *at what corpus size does this stop being true?* A fixed
file fixes the one variable the questions are about. Nothing is committed to git; the
output lands under `data/`, which this repository already ignores for `data/benchmarks/`
and for the same reasons — large, and reproducible from source.

**Ground truth is free.** The generator makes the term first and the column that should
match it second, so the answer is known by construction, at any scale, with no steward
involved.

---

## The five artifacts

### 1. Glossary (`glossary.csv`)

The row count is the least interesting parameter. Five distributional properties are what
make it a test:

| Property | Default | Why it is there |
|---|---|---|
| Near-duplicate clusters | 15% of rows, in clusters of 2–6 | terms identical except for the domain that owns them. The condition a domain prior exists to resolve. |
| Class-word distribution | short head, long tail | a uniform distribution would make the class word a free discriminator. |
| Name length | median 4 tokens, tail to 9 | lexical scoring is length-sensitive. |
| Not approved | 12% | drafts and retired terms compete as real terms, because they *are* real terms. |
| Definition echoes name | 20% | real glossaries are full of tautological definitions, and they are the rows a description-weighted scorer over-trusts. |

Plus a handful of **deliberately wide** clusters — one term name governed separately in as
many domains as the pack has. Those are the answer key for the repeated-leaf fixture.

`logical_name` is present and **blank**. `DictionaryEntry.to_searchable_text()` embeds it,
and the technical name is what the query side is built from, so filling it in would put the
query string inside the indexed document. That is the leak this repository already
published once and had to withdraw.

### 2. Abbreviation catalog (`abbreviations.json`, `abbreviations-delta.json`)

`expansions` is `{short: long}` and nothing else, because that is exactly what
`AbbreviationDictionary.from_dict` takes. Everything else in the file is metadata about
that map. It covers every token the glossary generator can emit, and it is hard in the five
ways expansion is hard:

- **ambiguous shorts** — one short form, several defensible long forms; the standard picks
  one and the discarded candidates are recorded;
- **multi-word rules** — one token expands to two or three;
- **stopword collisions** — short forms spelled exactly like English function words, which
  any pipeline that strips stopwords before expanding will silently lose;
- **never-expand acronyms** — already atoms; expanding them is the error;
- **a versioned delta** — ~200 mappings that *change*, which is the property a load-time
  configuration file cannot express.

### 3. Schemas (`schemas/*.avsc`, `schemas/*.flattened.json`)

Six profiles, each shipped **twice** — raw Avro and pre-flattened — because a production
pipeline sends the flattened form and testing only the convenient one is how a library ends
up correct on an input nobody uses.

| Profile | Exercises |
|---|---|
| `flat-english` | readable names, full docs — the ceiling |
| `flat-contracted` | the same columns through the naming standard, no docs; paired field-for-field with the above |
| `nested-deep` | depth 6+, long flattened paths |
| `nested-repeated` | one leaf name under dozens of parents — **the important one** |
| `no-doc` | no documentation at all |
| `mixed-production` | a proportioned mixture |

`nested-repeated` is built so the leaf name carries no information and the parent carries
all of it. That makes it the direct test of cache-key composition: key on the leaf name and
every occurrence collapses to one answer.

A generated column is never a copy of its term — qualifiers are dropped and swapped for
source-system words no term carries. `verify` measures the resulting token overlap and
prints the distribution, because "we did not check whether the benchmark was degenerate" is
how the last one got published.

### 4. Ground truth (`truth.csv`)

| Class | Share | Meaning |
|---|---|---|
| `EXACT` | 60% | one unambiguously correct term |
| `AMBIGUOUS` | 20% | a whole near-duplicate cluster, **all** members recorded |
| `NO_MATCH` | 15% | no correct term exists |
| `TRAP` | 5% | high lexical overlap with a real term, unrelated meaning, still no correct term |

The last two are the ones no public benchmark has. Both are constructed so that "no correct
term exists" is a fact about the generator rather than an observation about the data:
`NO_MATCH` columns are built from a held-out orphan vocabulary the glossary generator never
sees, and `TRAP` columns keep a real term's qualifiers and class word and replace its
subject with an orphan. `trap_id` records the term the matcher is expected to fall for.

### 5. Feedback trace (`feedback.jsonl`)

Reviewer verdicts: `APPROVED`, `REJECTED`, and `MANUAL_OVERRIDE` — the reviewer chose a term
**the matcher never proposed**. That third class is the highest-signal record there is: it
says what the right answer was on a field where retrieval did not surface it at all.

Measured against this repository's own stored record shape
(`presentation/api/feedback.py`), the loss is narrower than "a boolean cannot express an
override", and worth stating precisely because the narrow version is fixable:

- approved-versus-rejected survives, in `wasCorrect`;
- "the reviewer chose a different term" survives, because `chosenGovernanceId` and
  `suggestedGovernanceId` are both stored;
- **"the reviewer chose a term that was never proposed" does not.**
  `suggestedGovernanceId` is rank 1 only, so after storage a chosen id that differs from it
  may have been rank 2 or absent from a fifty-candidate list. Those are opposite facts —
  the first says the ranking was nearly right, the second says retrieval missed entirely —
  and nothing downstream can tell them apart.

Every record therefore carries `proposedIds`, and `FeedbackEvent.wire_projection()` returns
what would survive a round trip. The difference between the two is the loss, in bytes.

---

## The two traps

Both fail **silently** if you get them wrong, which is the only reason they are worth
shipping.

### Two multi-value columns, two separators

`sample_values` in the glossary is comma-separated; `enum_values` **in the same file** is
semicolon-separated. Read either with the other's separator and every multi-valued cell
becomes one value containing all of them, which indexes, matches, and is wrong.

`load_entries(delimiter_strict=True)` refuses that, and `benchmarks/exp_row_admission.py`
exercises the refusal in both directions — declared correctly, swapped under strict (must
raise), and swapped with the check off (must succeed, and produce the giant values). A gate
is not live until it has been seen to fail.

### Synthesised vocabulary, never sampled

A generator that reaches into a real glossary for "realistic" nouns produces a corpus
exactly as unshareable as the glossary was, and the leak is invisible — nobody reviewing
100,000 generated rows will notice which few hundred words came from where.

So the content words are manufactured by a seeded syllable grammar. Three pools are
ordinary English on purpose (class words like *Identifier* and *Code*, generic modelling
adjectives, and the connective prose in definitions): the rule is about **provenance**, not
about vocabulary, and manufacturing those would delete the signal the corpus exists to
carry. `verify` counts how many manufactured subject words happen to spell a whole word in
the bundled encoder's vocabulary and prints the number rather than assuming it is small.

---

## Running it

```
python benchmarks/gen_synthetic_pack.py --rows 10000 --verify
python benchmarks/gen_synthetic_pack.py --rows 100000 --out data/synthetic/large
python benchmarks/gen_synthetic_pack.py --rows 5000 --difficulty 2.0 --no-write --verify
```

Exit code `1` means a verification failed; the pack is still written and the findings say
what is wrong. `--difficulty` above 1.0 raises near-duplicates, non-approved rows and
tautological definitions **together**, because those are the three properties that make a
glossary hard to match against and turning one without the others produces a corpus that is
hard in one axis and trivial in the rest.

The generator imports nothing from `nexus_matcher` and uses only the standard library, so
it runs anywhere — including on a machine with no encoder, and inside an organisation that
wants to read what the corpus looks like before anyone runs a matcher over it.

---

## Verification, and what it caught

`benchmarks/synthetic/verify.py` holds the generator to every property this page claims:
the distribution targets, catalog coverage, both traps, the orphan-vocabulary guarantee,
the query/gold overlap, and a two-run checksum comparison for determinism.

It has already earned its place twice, on its first full run:

1. **The orphan pool overlapped the glossary's subjects.** Each word pool was generated
   with its own dedupe set, so a held-out orphan could collide with a subject the glossary
   uses — silently converting a `NO_MATCH` row into one with a perfectly good correct
   answer. Nothing else would have caught it: the truth file would still have said
   `NO_MATCH`, and the matcher would still have been right to disagree with it.
2. **Every nested field had its own unique ancestry**, which is not a nested schema — it is
   hundreds of unrelated single-column tables sharing a file. The raw `.avsc` it
   reconstructed to was 8 MB for 800 leaves, and about ten times smaller once parent chains
   were shared.

---

## The experiments, and what they measured

Four experiment scripts slot into the existing `benchmarks/exp_*.py` pattern and write
JSON into `benchmarks/results/`. Every accuracy comparison is **paired** over the fields
both conditions scored, and tested with the exact McNemar in
`benchmarks/optimization_ledger.py`.

All numbers below are on a 10,000-row synthetic glossary, seed 20260819, shipped
`MatchingConfig`. **They describe this corpus.** They are evidence about mechanisms, not
about anyone's real accuracy.

### The confidence floor, and abstention — `exp_confidence_floor.py`

`MatchingConfig.minimum_achievable_confidence` derives a floor of
`semantic_weight × fusion_alpha` = **0.63**. Over 4,574 fields the floor **held** — nothing
fell below it — and it is **not tight**: the lowest confidence any field actually reached
was **0.7050**, 0.075 above the derived bound.

That slack matters. `get_low_confidence_fields(threshold=…)` refuses a threshold at or below
0.63 on the grounds that it could only ever return `[]`. A caller who takes that refusal at
its word and passes 0.635 still gets `[]` on this corpus — with no refusal, and no
explanation. The number that makes that API honest is the **observed** minimum, not the
derived one.

Abstention, with the shipped `absolute_score_floor=None`:

| Reading | Value |
|---|---|
| Fields with no correct term, by construction | 872 of 4,574 (19.1%) |
| Fields the matcher declined (`FieldDecision.NO_MATCH`) | **0** |
| Fields auto-approved although nothing is correct | 12 |
| Mean confidence, answerable minus unanswerable | **+0.0200** |

Zero abstention is not a defect in itself — it is the documented consequence of shipping no
floor. The +0.02 is the important number: `confidence` is min-max normalised inside one
field's shortlist, so it barely separates a field with a correct answer from a field with
none. That is precisely why `absolute_score_floor` exists, and the script prints the full
sweep of what each candidate value would buy and cost on this corpus.

### Cache-key composition — `exp_repeated_leaf.py`

On the `nested-repeated` schema, where one leaf name appears **56 times** under 56 different
parent paths spanning 28 domains — so 28 distinct terms are correct across those 56
columns, and which one is decided entirely by the parent:

| | Keyed on the field | Keyed on the leaf name |
|---|---|---|
| P@1 on the 254 fields with a repeated leaf | **0.0709** | **0.0354** |
| P@1 over the whole schema (1,180 fields) | 0.1229 | 0.1153 |

Collapsing halves accuracy on the fields it touches (−3.54 points, 95% CI
[−0.0709, −0.0039], p = 0.064), and the whole-schema number understates it by 4.7× — which
is what a run-level metric would have reported.

The failure mode is the point. Collapsing raises nothing, drops no field and lowers no
confidence: the conservation law still holds, 232 answers change, 14 become wrong, and
**57% of those wrong answers arrive at or above `auto_approve_threshold`**, because they are
real confident matches — for a different column.

The pipeline builds **56 distinct query texts** for those 56 columns, because
`ContextEnricher` injects the parent path. An embedding cache keyed on that text is safe;
one keyed on the leaf name is not.

### Row admission — `exp_row_admission.py`

Same corpus, same queries, same seed, differing only in whether
`admit={"status": {"Approved"}}` is passed:

| Reading | Filtered | Unfiltered |
|---|---|---|
| Indexed entries | 8,849 | 10,000 |
| P@1 | 0.1831 | 0.1824 |
| Rank 1 was a **draft or retired** term | — | 364 of 3,394 (**10.7%**) |
| …and auto-approved anyway | — | 6 |

**Read the two rows separately, and do not expect them to agree.** The P@1 delta is
+0.0007, p = 0.90 — a clean null. That is not evidence that admission is worthless, and
"fixing" it by dropping the filter would trade a governance property for retrieval noise.
Removing rows changes the candidate set, so it changes the per-field min-max normalisation
and the lexical arm's document statistics, and survivors reorder; the delta can land either
side of zero.

The number admission is *for* is the 10.7%. No threshold makes that share acceptable,
because the row is not wrong — it is unratified, and only its status column says so.

### Scale — `exp_synthetic_scale.py`

The queries and the gold entries are held fixed and only the number of competing entries
grows. 229 fields, in the pack's own class mixture, against three corpora:

| Indexed entries | P@1 | Auto-approve coverage at the shipped 0.87 | Auto-approve precision | *n* approved |
|---|---|---|---|---|
| 923 | 0.3169 | 0.1223 | 0.7500 | 28 |
| 8,872 | 0.2459 | 0.1004 | 0.9565 | 23 |
| 88,079 | 0.1530 | 0.0568 | 0.8462 | 13 |

**Two of those columns are well estimated and one is not, and the difference decides what
may be claimed.**

P@1 halves — 0.3169 → 0.1530, over 183 answerable fields at every size, same queries, same
answers, only more competition. Coverage at a fixed 0.87 falls by more than half, over all
229 fields. Both are the same statement: **the score distribution moves with the size of
the candidate pool**, so a fixed threshold admits a steadily smaller share of a steadily
harder corpus. A threshold is a claim about a distribution, and this is the distribution
moving.

Auto-approve **precision** is a ratio over the 13–28 fields that cleared the bar, and at
that count it is not a measurement. The script reports it paired, with a bootstrap interval
that resamples queries and recomputes the whole ratio: 88,079 against 923 reads +0.0962
[−0.1701, +0.3377] — an interval wide enough to contain anything, and pointing *up* mostly
because coverage fell. **No precision-transfer claim is supportable from this run.** A
smoke run at 919 → 3,568 entries, where the query set is larger relative to the corpus and
40+ fields clear the bar, reads −0.0285 [−0.1304, +0.0711] — the expected direction, still
not significant.

The limitation is structural rather than a matter of running longer: a 1,000-entry corpus
cannot host a large query set whose answers are all present *and* still be a
needle-in-haystack, because the answers would be most of the corpus. Closing it means
raising the smallest size (and giving up the comparison to a few-hundred-entry
calibration), or widening `--gold-budget` and stating the higher gold density plainly.

---

## What the pack unlocks that is not measured here

The fixtures for these exist; the experiments do not, because each needs a mechanism that
is not in this repository, or is only half in it.

| Experiment | Fixture in the pack | What it needs |
|---|---|---|
| Abbreviation overlay's value | `flat-english` paired field-for-field with `flat-contracted`, plus the catalog | a per-request abbreviation channel |
| Overlay must be per-request | `abbreviations-delta.json` — 200 mappings that change | the same channel, applied mid-run |
| Domain prior's value | the wide near-duplicate clusters; `nested-repeated` | a request-level domain prior |
| Mode selection | `no-doc` | a way to downgrade a content-free field to lexical-only retrieval |
| Approved-pair bypass | `feedback.jsonl`, 5,000 verdicts | feedback read back into ranking; today it is recorded and never read |

The repeated-leaf experiment already sizes the third of these from below: on that fixture
the parent path alone reaches P@1 0.1229 where choosing at random inside a cluster scores
0.0333, and the distance from 0.1229 to 1.0 is disambiguation only the caller holds. No
experiment in this pack can close it from the query side.

---

## What this pack does not prove

Stated plainly so it is not oversold.

- **It does not measure real-world accuracy.** Synthetic data validates *mechanisms*. The
  absolute P@1 numbers above are properties of a manufactured vocabulary and mean nothing
  about any real corpus.
- **It cannot reproduce real semantic subtlety.** Genuine ambiguity between two governed
  terms is richer than any near-duplicate rule, and the syllable grammar carries no meaning
  a dense encoder can exploit beyond shared stems.
- **A pass here is necessary, not sufficient.** It is the fast, cheap, shareable gate that
  runs before the expensive one — never a substitute for a corpus somebody labelled.

Related: [`docs/guides/absolute_score_floor.md`](absolute_score_floor.md) for calibrating a
floor on your own data, and [`docs/guides/governed_abbreviations.md`](governed_abbreviations.md)
for the abbreviation feature this pack's catalog is shaped for.
