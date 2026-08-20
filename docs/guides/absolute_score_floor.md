# Calibrating the absolute score floor

`absolute_score_floor` turns a rank-1 candidate whose raw retrieval score is too low into
a **`NO_MATCH`** verdict on the field. It is the only way this library can say *nothing in
your dictionary answers this column* — `REJECT` cannot, for a structural reason set out
below.

It ships **off**, and this page is not an argument for turning it on with a particular
number. It is the procedure for measuring one, because **a floor is a statement about a
score distribution**, and the distribution belongs to your dictionary, your field names
and your encoder. A number measured somewhere else describes somewhere else.

If you read only one thing: **a floor copied from another deployment is worse than no
floor**, because it looks configured. The worked example below shows a plausible-looking
value that produces zero `NO_MATCH` verdicts on any corpus we have measured — a setting
that is on, documented, monitored, and inert.

Contents:

1. [What the floor is compared against](#what-the-floor-is-compared-against)
2. [Why there is no default](#why-there-is-no-default)
3. [The procedure](#the-procedure)
4. [Worked example: the same glossary, twice, with two different answers](#worked-example-the-same-glossary-twice-with-two-different-answers)
5. [The wobble you must leave room for](#the-wobble-you-must-leave-room-for)
6. [Turning it on, and reading it back](#turning-it-on-and-reading-it-back)
7. [Reading a no-match verdict](#reading-a-no-match-verdict)
8. [When to re-calibrate](#when-to-re-calibrate)

---

## What the floor is compared against

The floor is compared against the **rank-1 candidate's `absoluteScore`** and nothing else.

`absoluteScore` is the raw dense-retrieval score for that candidate, published on every
candidate of every match response. Under the shipped wiring it is a cosine similarity;
`scoring.absoluteScoreMetric` in the response says which metric actually produced it, and
a store that cannot say reports `"unknown"` rather than guessing. It is the same number as
`explain.absoluteCosine`, read once so the two cannot disagree.

**It is not `confidence`, and that is the whole point of it existing.** `confidence` is
min-max normalised inside one field's shortlist, so a rank-1 candidate lands near
`fusion_alpha` whether the match is excellent or absurd. It has a structural floor of
`semantic_weight × fusion_alpha` = **0.63** at the shipped weights, which sits *above*
`review_threshold` = 0.50 — so no setting of `review_threshold` can ever reject a rank-1
match on score alone. Every field comes back at least `REVIEW`, however irrelevant its best
candidate is. That gap is why `NO_MATCH` and this floor exist.

`scoring.comparability` in every match response states this in machine-readable form:
`confidence` is `WITHIN_FIELD`, `absoluteScore` is `ACROSS_FIELDS`. Only the second can be
compared against a fixed number.

---

## Why there is no default

Shipping a number here would be inventing a calibration for somebody else's corpus, which
is the same mistake as shipping a taxonomy. `auto_approve_threshold` at least names the
corpus it was measured on; there is no corpus at all behind this one, and there cannot be:
the library has never seen your dictionary.

This is [HAZARDS.md H-002](../HAZARDS.md) in its most practical form. That hazard is
usually read as *re-measure after a change*. The stronger reading is the one this page
exists for: **two field sets over the same glossary, the same encoder and the same library
can disagree about whether a usable floor exists at all.** The next section measures
exactly that.

One consequence worth stating before the procedure, because it is the failure this page was
written after: the encoder's cosine similarities over short field texts are **compressed
into a narrow, high band**. Nothing in 72 measured fields scored below 0.49. So a floor
that "sounds conservative" — 0.2, 0.3, 0.4 — is not conservative. It is inert. It will
never fire, on any input, and the only evidence you will ever have of that is the
`NO_MATCH` count that stays at zero.

---

## The procedure

You need a **labelled sample of your own fields**, and it must contain both kinds of row:

- fields whose correct glossary entry you know, and
- fields for which **no entry is a correct answer**.

The second kind is the one that does the work and the one everybody omits. Without it you
are measuring how high correct answers score, which tells you where *not* to put a floor
and nothing about where to put one. Twenty of each is enough to see the shape; the pack's
own set is 42 rows, 36 positive and 6 negative.

`examples/governance/labels.jsonl` is the format:

```json
{"flattenedName": "manifest_pax_full_name", "dataType": "string", "expected_id": "GBF-0001"}
{"flattenedName": "wifi_router_firmware_version", "dataType": "string", "expected_id": null}
```

Then:

1. **Match the sample against your real dictionary, through the entry point you will run in
   production.** Which *loader* you use does not matter: `matcher.load_dictionary(...)` and
   the service's own `NEXUS_API_DICTIONARY` bootstrap were driven with the same 26 field
   specs and disagreed on `absoluteScore` **zero times** — same scores to six decimals, same
   rank-1 entry, same governance on 24 of 26. (`confidence` differed on 2 of the 26, by
   exactly 0.01; one more reason the floor is compared against the absolute number.)

   Which **entry point** you use matters a great deal. Sending those same 26 columns as
   `{name, path, doc, type}` to `POST /api/v1/match`, versus handing the pack's own file to
   `matcher.match_schema(...)`, moved `absoluteScore` on **all 26**, by up to **0.037** —
   because the two build different query text out of the same column. Calibrate through the
   entry point you serve, or the floor is measured on a distribution you do not ship.
2. **Record the rank-1 `absoluteScore` for every row**, together with whether rank 1 was
   the correct entry.
3. **Split into two piles.** `positives` = rows whose rank 1 was correct. `negatives` =
   rows labelled `null`, *plus* rows where rank 1 was the wrong entry. A wrong rank 1 is a
   field the floor should ideally catch, so it belongs with the negatives.
4. **Compare `min(positives)` against `max(negatives)`.**
   - If `max(negatives) < min(positives)`, the two are separable. Any floor in that open
     band catches every negative and costs no positive. Take the **midpoint**, not an
     endpoint.
   - If they overlap, **no free choice exists** and a floor is a trade. Sweep it and read
     off, at each value, how many correct answers you lose against how many unmatchable
     fields you catch. Pick the point on that curve your review capacity can absorb.
5. **Subtract the margin from [the next section](#the-wobble-you-must-leave-room-for)**,
   then round *down* to two decimals. Over-inheriting is the expensive error
   ([GOVERNANCE.md rule 5](../GOVERNANCE.md#5-over-inheriting-is-the-expensive-error)), but
   so is a `NO_MATCH` on a field that had a right answer: it sends real work back to a
   queue.

Report both numbers whenever you report the floor — how many correct answers it costs and
how many unmatchable fields it catches. A floor reported with only the second is
[H-001](../HAZARDS.md) wearing a new hat.

---

## Worked example: the same glossary, twice, with two different answers

Everything below was measured on **2026-08-19** against the **bundled int8 ONNX encoder**
and the 30-entry glossary of the fictional
[Gravel Bay Ferry Authority pack](../../examples/governance/README.md). Do not carry these
numbers anywhere. They are here to show the *shape* of the answer and how far it moves.

### Corpus A — 30 described fields, through the live HTTP service

The pack's 26 schema fields (each with a one-line `doc`), plus four fields invented for
this measurement that a ferry glossary cannot possibly answer — a quasar flux index, a
sourdough starter hydration percentage, a guitar pickup impedance, a mitochondrial
haplogroup. Driven through `POST /api/v1/match` against a server started the documented
way, with the floor loaded from `NEXUS_API_MATCHING_CONFIG` on each run:

```
rank-1 absoluteScore
  correct rank-1      n=24  min=0.720624  max=0.976209
  no correct answer   n=6   min=0.496592  max=0.650861
  SEPARATING BAND: (0.650861, 0.720624]
```

```
floor   fieldDecisions                                   NO_MATCH  right answers lost
None    {AUTO_APPROVE: 24, REVIEW: 6}                     0         0
0.30    {AUTO_APPROVE: 24, REVIEW: 6}                     0         0
0.45    {AUTO_APPROVE: 24, REVIEW: 6}                     0         0
0.50    {AUTO_APPROVE: 24, NO_MATCH: 1, REVIEW: 5}        1         0
0.60    {AUTO_APPROVE: 24, NO_MATCH: 5, REVIEW: 1}        5         0
0.66    {AUTO_APPROVE: 24, NO_MATCH: 6}                   6         0
0.72    {AUTO_APPROVE: 24, NO_MATCH: 6}                   6         0
0.75    {AUTO_APPROVE: 23, NO_MATCH: 7}                   7         1
0.80    {AUTO_APPROVE: 22, NO_MATCH: 8}                   8         2
```

Read the top three rows first. **A floor of 0.30 produces zero `NO_MATCH` verdicts. So
does 0.45.** They are below every score the encoder produced on any of the 30 fields, so
they cannot fire — not on a bad match, not on total nonsense. The lowest rank-1
`absoluteScore` seen anywhere in this corpus was 0.4966, on the guitar pickup field.

On this corpus the answer is comfortable: the distributions separate cleanly, and anything
from 0.66 to 0.72 catches all six unanswerable fields while costing none of the 24 correct
ones. The midpoint of the band is **0.685**.

### Corpus B — the pack's own 42 labelled rows, and a gap too thin to stand on

Same glossary, same encoder, same library, **same entry point** — one `POST /api/v1/match`
carrying every row. The only difference is the fields: `examples/governance/labels.jsonl` is
42 bare column names with a data type and **no description** — `pax_dob`, `card_last4`,
`wifi_router_firmware_version` — which is what a real schema extract usually looks like.

```
rank-1 absoluteScore
  correct rank-1      n=34  min=0.617166  max=0.841510
  no correct answer   n=8   min=0.516254  max=0.658960
  worst correct = 0.617166   best incorrect = 0.658960
  NOT SEPARABLE: the distributions overlap.
```

```
floor   correct answers lost   unmatchable fields caught
0.30    0                      0 / 8
0.50    0                      0 / 8
0.55    0                      2 / 8
0.60    0                      5 / 8
0.65    4                      7 / 8
0.70    10                     8 / 8
0.80    24                     8 / 8
```

The eighth negative is `pax_dob`, which ranks the postal-address entry first — a wrong
rank 1 at 0.6590, above 11 of the 34 correct answers. Nothing catches it and keeps them.

**Now the part that matters, and the reason the procedure has step 5.** Ignore `pax_dob`
for a moment and the remaining seven negatives all sit below the worst correct answer:
the highest of them is `lifeboat_drill_attendance` at **0.614483**, and the lowest correct
answer is **0.617166**. A floor of 0.617 would catch **7 of 8 and cost nothing**. It looks
like a free win.

It is not one. The gap it stands in is **0.0027 wide**, and the request-shape wobble
measured in [the next section](#the-wobble-you-must-leave-room-for) is **0.010** — nearly
four times it. A floor placed there is a coin toss re-flipped by how many columns a caller
happens to send in one request. Applying step 5's margin instead — 0.617166 − 0.02 — gives
**0.59**, which still catches 5 of the 8 and still costs nothing, and is a number that
survives being served.

**Nothing changed but the field text, and the answer went from "0.685, and the band is 0.07
wide" to "0.59, and everything above it is an illusion".** Same 30 entries, same encoder,
same weights, same route. That is the entire argument of this page in one comparison, and it
is why the procedure ends in a sweep and a margin rather than in a number.

A deployment looking at Corpus B has three honest options: take 0.59 and accept that three
unanswerable columns still reach a reviewer, spend the effort to get descriptions onto the
fields — the one change that moved this glossary from a 0.0027 gap to a 0.07 band — or leave
the floor off and route on `REVIEW`. The third is the shipped default and is not a failure.

---

## The wobble you must leave room for

`absoluteScore` for one field is **not identical across differently-shaped requests**, and
the size of the movement is worth knowing before you place a floor within 0.01 of anything.

The bundled encoder pads a batch to the length of its longest member, and the padding
changes the query vector. Encoding one text alone, and again in a batch alongside a much
longer text, produced query vectors whose cosine with each other was **0.994260**, with a
largest single-component difference of **1.81e-2**. Batched alongside an *equal-length*
text, the vector was bit-identical.

On the wire, over all 30 fields of Corpus A, scored once one-field-per-request and once as
a single 30-field request:

```
  n=30  min delta=-0.010036  max delta=+0.009325  max|delta| = 0.010036
  fields whose rank-1 ENTRY changed between the two request shapes: 2
```

So:

- **Leave at least 0.02 of margin** between your floor and the nearest score you care
  about. Corpus A's band is 0.070 wide, which absorbs this comfortably; a band narrower
  than about 0.03 is not a band.
- **Calibrate at roughly the request shape you serve.** Measuring one field per request and
  then serving batches of 100 measures a slightly different quantity.
- This does **not** make responses non-deterministic. Two identical requests return
  byte-identical bodies — verified on every capture in this document. The dependence is on
  *what else is in the same request*, not on the run.

---

## Turning it on, and reading it back

**Library:**

```python
from nexus_matcher import MatchingConfig, NexusMatcher

matcher = NexusMatcher.from_config(MatchingConfig(absolute_score_floor=0.685))
```

**Service** — a JSON or TOML file named by `NEXUS_API_MATCHING_CONFIG`:

```json
{"absolute_score_floor": 0.685}
```

```bash
export NEXUS_API_DICTIONARY=examples/governance/glossary.csv
export NEXUS_API_GOVERNANCE=examples/governance/protection_classes.json
export NEXUS_API_MATCHING_CONFIG=matching.json
nexus-matcher api --host 127.0.0.1 --port 8000
```

A misspelled key in that file is **refused, not ignored** — the same standard `FieldSpec`
applies to a misspelled field key. The refusal follows the service's degradation contract
rather than killing the process: the server starts, and both match routes answer **503**
with `NEXUS-1002` and a message naming the option it could not understand. Verified with
`{"absolute_score_flooor": 0.685}`:

```
503  {"error": {"code": "NEXUS-1002", "message": "The matching service is not ready:
     loading NEXUS_API_DICTIONARY failed: ValueError: Unknown matching config option(s)
     in matching.json ...
```

Closing the loop on Corpus A: the procedure's own output was the band
`(0.650861, 0.720624]`, whose midpoint is `0.685`. Loaded back through that file:

```
counts: {'AUTO_APPROVE': 24, 'NO_MATCH': 6}
published scoring.absoluteScoreFloor: 0.685
right answers lost: 0
```

**Reading it back.** The active floor is published on every match response at
`scoring.absoluteScoreFloor`, `null` when none is configured. That is the value a consumer
should quote when it explains a `NO_MATCH`.

It is **also** on `GET /api/v1/status`, as `thresholds.absoluteScoreFloor`, beside
`thresholds.absoluteScoreMetric` which names what the floor is compared against. Both are
read off the same properties the match response reads, so the two surfaces cannot report
different floors for one server. That is the route to use when you want to check what a
deployment has set **without matching anything** — which is also the only way to see it on a
server whose dictionary has not loaded.

> Until 2026-08-20 the status route carried no floor, and this section said so: the only way
> to read one was to send a one-field match. An operator who cannot see the active floor
> cannot tell an emitted `NO_MATCH` from a field the matcher simply had nothing for, so it is
> published there now. `null` on that route means *no floor is configured*, which is the
> opposite convention from the rest of that block, where `null` means "this matcher does not
> expose it".

Which numbers a deployment has moved away from the shipped defaults, and what the shipped
ones were fitted on, is the neighbouring `calibration` block —
[Calibration profiles](calibration_profiles.md).

---

## Reading a no-match verdict

`NO_MATCH` on a field means: **this response carries nothing that field may inherit.** It
arises two ways.

1. **The field came back with no candidates at all.** This happens with **no floor
   configured**, and is worth knowing because "off" does not mean "`NO_MATCH` is
   unreachable". Confirmed directly: `derive_field_decision((), None)` is `NO_MATCH`.
2. **A floor is configured and rank 1 does not clear it** — including the case where rank 1
   carries `absoluteScore: null`, meaning the dense retriever never proposed that
   candidate and so offers no evidence it clears any floor.

The two are not distinguishable from the verdict alone. Read `results[path]` beside it:
empty is case 1, non-empty is case 2.

**The candidates on a `NO_MATCH` field are still returned, and they still carry
`governance`.** They are evidence for a reviewer, not a classification. This is
[GOVERNANCE.md rule 3](../GOVERNANCE.md#3-an-unmatched-field-inherits-nothing) at the wire:
read `fieldDecisions[path]` *first*, and on `NO_MATCH` inherit nothing, however
authoritative `results[path][0].governance` looks. A captured example is in
[GOVERNANCE.md](../GOVERNANCE.md#where-rule-3-is-published-fielddecisions) — a nonsense
field whose rank-1 candidate carries a populated protection class and a `confidence` of
0.82.

`NO_MATCH` is a **per-field** verdict. The per-candidate `decision` keeps its three values
(`AUTO_APPROVE`, `REVIEW`, `REJECT`) and cannot express it; a per-candidate `REJECT` says
"this candidate is below the bar", which is a different claim.

---

## When to re-calibrate

The floor is a property of a score distribution, so anything that moves the distribution
invalidates it:

- **the encoder** — a different model, or the sentence-transformers extra replacing the
  bundled ONNX one;
- **the fusion weights, `fusion_alpha`, or the reranker** being wired in;
- **the query representation** — including turning on abbreviation expansion, or starting
  to send `doc` text where you previously sent bare names. That is the largest single
  difference between Corpus A and Corpus B above, and it was enough to move the answer
  from a free choice to no choice;
- **the dictionary** — new entries, changed definitions, and above all a change of *size*.
  Corpus size is a regime change and not a scaling factor: one option in this library is
  worth +1.9 P@1 at 688 entries and −18.8 at 30,000.

Re-run the procedure. It costs one labelled file and a few seconds of matching, and the
alternative is a governance verdict standing on a number nobody has checked since the
encoder changed.
