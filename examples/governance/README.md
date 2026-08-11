# Governance example pack

> **This data is FICTIONAL.** The Gravel Bay Ferry Authority does not exist, and neither
> does its taxonomy. Every protection code, tier, glossary term and reviewer in this
> directory was invented to demonstrate the feature. It is **not** a template of any
> organisation's catalog and it is not a starting point for yours — the whole point of the
> feature is that *you* supply the vocabulary. Replace all of it.

A runnable end-to-end example of governance inheritance: a caller-supplied controlled
vocabulary, a glossary validated against it, schema fields matched into it, and the
protection class each field would inherit.

Concepts and rationale live in [`docs/GOVERNANCE.md`](../../docs/GOVERNANCE.md). This file
is how to run it.

## The five commands

Run them in order, from the repository root.

```
python examples/governance/validate.py            # 1. the vocabulary and the glossary
python examples/governance/run_pack.py            # 2. match, and write out/results.json
python examples/governance/check_expectations.py  # 3. hold it to fields.json
python examples/governance/calibrate.py           # 4. sweep the threshold over labels.jsonl
python examples/governance/review_queue.py        # 5. the queue a reviewer works
```

Commands 1 and 3 exit non-zero when something is wrong; 2, 4 and 5 are reports. Command 2
writes `examples/governance/out/results.json`, which 3 and 5 read — `out/` is generated,
not source.

Everything prints plain ASCII, uses no box-drawing and no spinner, and produces identical
bytes when stdout is a pipe. That is deliberate: this repository has already shipped a
surface that raised `'charmap' codec can't encode character` on a console using a legacy
Windows code page.

## The files

| File | What it is |
|---|---|
| `protection_classes.json` | The **caller-supplied** controlled vocabulary: 9 codes across 4 tiers, each with its personal-information and direct-identifier flags, plus declared aliases for legacy tokens. |
| `glossary.csv` | 30 governed terms covering every code, plus the uncoded case. One row carries a legacy token; one carries a token the vocabulary declares to be noise. |
| `glossary_invalid.csv` | 5 deliberately broken rows. Each carries an `expected_problem` column — not part of a real glossary — naming the reason the loader must refuse it. |
| `fields.json` | 26 schema fields to match, each with the id and decision the pack expects. Includes one near-miss and two fields nothing in the glossary governs. |
| `labels.jsonl` | 42 labelled query → id pairs for threshold calibration. Names and types only, no descriptions: the harder case, and the one most schemas actually present. |
| `feedback.jsonl` | 6 reviewer accept/reject events, folded back into the queue by command 5. |
| `_pack.py` | Shared plumbing for the five commands. Not part of the library. |

## The finding this pack exists to show

**Read the decision, never the score.**

`final_confidence` is a *rank-relative* number, not a similarity. It is the min-max
normalised fused retrieval score, so the rank-1 candidate sits near `fusion_alpha`
whether the match is excellent or absurd. Measured on this pack, over all 26 fields:

| | confidence |
|---|---|
| lowest rank-1 confidence of any field | 0.8058 |
| highest rank-1 confidence of any field | 0.8958 |
| `lifejacket_locker_inspection_due_date`, which nothing in the glossary governs | **0.8792** |
| `booking_passenger__legal_name_digest`, a correct top-1 that must not be inherited | 0.8292 |

A lifejacket locker inspection date scored **higher** than a correct match, and its top
candidate was *Passenger Date Of Birth* — a `SEALED_RESTRICTED` direct identifier. What
stopped it being auto-approved was not the confidence. It was `min_confidence_gap`: its
margin over the runner-up was 0.0384 against a required 0.10.

Two consequences worth carrying away:

1. **No threshold on `final_confidence` separates a novel field from a governed one on
   this pack.** A `review_threshold` high enough to reject 0.8792 also rejects eight
   fields that are correct.
2. **`REJECT` is unreachable for a top-1 match in the shipped configuration.**
   `final_confidence` has a structural floor of `semantic_weight × fusion_alpha`
   = 0.70 × 0.90 = 0.63, and `review_threshold` is 0.50. Nothing can fall below the floor,
   so nothing can be rejected at rank 1. `REJECT` does appear at lower ranks. So the two
   novel fields in `fields.json` expect `REVIEW`, not `REJECT`, and the safe reading of
   `REVIEW` is "a human must decide", never "probably fine".

This is the same shape as DX-001 (`tests/museum/NM-0027`), where a default cutoff of 0.6
sat below the floor and reported "nothing to review" on every schema ever matched.

### What the calibration sweep shows

Command 4, on this pack's 42 labels over 30 glossary entries:

```
 threshold   coverage  auto-approved   auto-approve precision   wrong
      0.70     0.8810             37                   0.9189       3
      0.80     0.6667             28                   1.0000       0
      0.85     0.4048             17                   1.0000       0
      0.87     0.4048             17                   1.0000       0
      0.90     0.0238              1                   1.0000       0
```

All three wrong auto-approvals at 0.70 are fields nothing in the glossary governs. That is
what the threshold is buying: not better matching, but fewer novel fields silently
inheriting somebody else's protection class.

These numbers calibrate **this pack** and nothing else. `docs/HAZARDS.md` H-002: a
threshold calibrated on one corpus does not transfer, and corpus size is a regime change
rather than a scaling factor.

## What this pack found

Building it surfaced a live defect in the loading path, which command 2 reports as a
`WIRING DEFECT` block:

**`matcher.load_dictionary()` drops the protection code.** It indexes through
`BaseDictionaryLoader._convert_row`, which constructs its `DictionaryEntry` without ever
reading the protection-code column. So every indexed entry carries `governance_code=None`,
every `MatchResult.governance` comes back `None`, and a field the glossary marks
`SEALED_RESTRICTED` and a direct identifier is **auto-approved carrying no class at all**.
`None` there is indistinguishable from "this entry genuinely has no class", which is
exactly why nothing notices — the same failure as `tests/museum/NM-0005`, one layer up, in
the documented path that `nexus-matcher match` uses.

`ingest.load_entries(source, governance=vocabulary)` attaches the code correctly. The pack
falls back to joining against it so the rest of the commands still run, and records
`"governance_source": ["caller_side_join", ...]` on the affected records rather than
producing a silently blank governance report.

## Adapting it

1. Replace `protection_classes.json` with your own vocabulary. Nothing in the library
   defines a code; every code the library will accept comes from that file.
2. Point `glossary.csv` at your glossary, or change the column names — the code column is
   recognised as `governance_code`, `protection_class`, `protection_code`,
   `classification_code` and a few more, and the tier column as `classification`,
   `protection_level`, `sensitivity` and others.
3. Run command 1 first, and keep running it. It is the check that stops a row whose stated
   tier contradicts its own code from ever being indexed.
4. Build your own `fields.json` and `labels.jsonl` from your schemas. Do not carry this
   pack's thresholds over; re-run command 4 against your corpus, at its size.
