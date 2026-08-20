# Calibration profiles

**A threshold is a statement about a score distribution.** The distribution belongs to your
dictionary and your field names. Change either and the same number means something else.

This library ships thresholds fitted on a public benchmark, because it has to ship
something and a number with a corpus behind it is better than a round one without. This
guide is about the two things that follow from that: how to see which numbers a deployment
is actually running, and how to replace them with numbers fitted on your data.

For the *absolute score floor* specifically — a different threshold, on a different
quantity, answering "did anything match at all?" — see
[Calibrating the absolute score floor](absolute_score_floor.md). The two are complementary:
that one decides `NO_MATCH`, these decide `AUTO_APPROVE`.

---

## 1. What the shipped numbers were fitted on

| | |
|---|---|
| Corpus | `bird+omop combined` — BIRD-SQL dev set + OMOP CDM v5.4 field specification |
| Size | **688** labelled fields against a **688**-entry pooled dictionary |
| Splits | bird 361, omop 327 |
| Domains | public relational database schemas; a clinical common data model |
| Field naming | ordinary SQL and CDM column identifiers. The bird split is heavily abbreviated, the omop split is not. Neither is a flattened nested path; neither was contracted by a governed abbreviation standard. |
| Ambiguity | one gold entry per field, drawn from two unrelated domains — a query competes against 687 distractors mostly from a different subject area |
| Fitted by | `benchmarks/exp_calibration.py` |
| Artifact | `benchmarks/results/exp_calibration_combined.json` |

Measured at the shipped `auto_approve_threshold` of **0.87**, on that corpus:

| Metric | Value |
|---|---|
| auto-approve precision | 0.952941 |
| auto-approve coverage | 0.123547 |
| P@1 over the whole corpus | 0.581395 |

The neighbouring point on the same curve, for the trade: at **0.85**, coverage rises to
0.207849 and precision falls to 0.916084. Eight points of coverage for four of precision.

The default targets ~95% auto-approve precision and accepts low coverage, because
auto-approving a wrong mapping is far more expensive than sending a field to a human. If
your cost balance is different, that is a reason to move the number — but move it against
*your* curve, not this one.

**None of this is a claim about your data.** Every figure above is served over HTTP at
`GET /api/v1/status` under `calibration.corpus`, and is checked against the artifact by
`tests/packaging/test_calibration_provenance.py`, so it cannot quietly go stale.

---

## 2. Seeing which profile a deployment is running

```bash
curl -s localhost:8000/api/v1/status | jq '{thresholds, calibration}'
```

```json
{
  "thresholds": {
    "autoApprove": 0.87,
    "review": 0.5,
    "minConfidenceGap": 0.1,
    "resultsPerField": 5,
    "fusionAlpha": 0.9,
    "minimumAchievableConfidence": 0.63,
    "reviewThresholdBelowFloor": true,
    "absoluteScoreFloor": null,
    "absoluteScoreMetric": "cosine"
  },
  "calibration": {
    "defaultsInForce": true,
    "overrides": {},
    "dictionarySizeRatio": 0.043605,
    "warnAboveSizeRatio": 10.0,
    "corpus": { "…": "as above" }
  }
}
```

Read it in this order:

1. **`calibration.overrides`** — empty means this server is running the numbers this library
   shipped. Non-empty names every setting that differs, with its live value, in the
   snake_case spelling you would put in a config file. It is derived from the config
   dataclass's own fields, so a setting nobody remembered to publish still appears here the
   day a deployment changes it.
2. **`thresholds`** — the live numbers. Every member is `null` when the matcher does not
   expose it, never `0.0`; the one exception is `absoluteScoreFloor`, where `null` means *no
   floor is configured*, which is the shipped default.
3. **`calibration.dictionarySizeRatio`** — how many times the calibration corpus your
   dictionary is.

The block is byte-stable: two GETs against one process produce identical bytes, so you can
diff two hosts and see only the difference that matters.

---

## 3. The `UNCALIBRATED_SIZE` warning

`degraded: true` with

```json
{"code": "UNCALIBRATED_SIZE", "message": "This server is running on the shipped default thresholds, which were fitted on 688 fields against a 688-entry dictionary…"}
```

means both of these are true:

* all three decision thresholds — `auto_approve_threshold`, `review_threshold`,
  `min_confidence_gap` — are still the shipped ones, **and**
* your dictionary is more than `calibration.warnAboveSizeRatio` (10) times the calibration
  corpus, so above 6,880 entries.

**It is not a defect report.** Nothing is broken; matching is running exactly as configured.
It says the published auto-approve precision is a fact about a 688-entry public benchmark
and not about your glossary, and that nobody has yet measured what 0.87 buys on your data.

### What it deliberately does not check

**Domain and naming style.** They are the other two dimensions that matter, and comparing
them would need a similarity metric this library has never validated. A warning computed
from an invented metric is wrong in a direction nobody can audit, so instead the corpus's
domains, naming style and ambiguity are *described* on `calibration.corpus` and left for you
to compare by eye. If your fields are contracted identifiers flattened out of nested
records, matched against a large single-industry glossary, the corpus above resembles them
in no dimension at all — and the absence of a warning is not a reassurance.

**Dictionaries smaller than the corpus.** Nothing here has measured that direction. A
surface that reported `degraded` on every demo would teach operators to ignore the field
that exists to catch a silent encoder fallback.

### Why ten

It is the smallest ratio this repository can point at a measurement across. In
`benchmarks/results/exp_alias_scale.json`, dictionary-side alias generation is worth **+1.9**
points of P@1 on a corpus the size of the calibration one, and **-13.7** on a corpus ten
times it — the sign of a retrieval effect inverts across that step, because alias noise
scales with corpus size while signal does not. A tighter ratio would be a guess wearing a
threshold.

---

## 4. Fitting your own

### 4.1 What you need

A **labelled sample of your own fields**: for each, the dictionary entry a governance lead
agrees it should map to. There is no substitute, and this is the expensive part. A few
hundred fields is enough to see the shape of the curve; the shipped defaults were fitted on
688.

Draw the sample from the schemas you actually run, not from the ones that are easy to label.
A sample of unambiguous fields moves the whole curve up and produces a threshold that is too
low for the ambiguous ones — which are the fields the threshold exists for.

### 4.2 Measure the curve

```python
from nexus_matcher import NexusMatcher

matcher = NexusMatcher.from_config(governance="your_protection_classes.json")
matcher.load_dictionary("your_glossary.xlsx")

session = matcher.match_schema_session("your_labelled_schema.avsc")
rank1 = {
    path: candidates[0]
    for path, candidates in session.results.items()
    if candidates
}

# `gold` maps the SAME key `results` is keyed by -- the caller's own field identity --
# to the entry id a governance lead agreed on. Keying it any other way is how a sweep
# ends up measuring string equality against itself.
for threshold in [round(0.50 + 0.01 * step, 2) for step in range(38)]:
    approved = [(path, m) for path, m in rank1.items() if m.final_confidence >= threshold]
    correct = [path for path, m in approved if m.governance_id == gold[path]]
    coverage = len(approved) / len(rank1)
    precision = len(correct) / len(approved) if approved else None
    print(f"{threshold:.2f}  coverage {coverage:.3f}  precision {precision}")
```

`benchmarks/exp_calibration.py` is the same sweep with the reporting and the artifact
writing, and is the script to copy if you want the output in the shape this library's own
artifacts use.

### 4.3 Choose the point, not the number

Pick your precision target first — how often you can afford an auto-approved mapping to be
wrong — then read the threshold off your own curve at that precision. Do not copy 0.87. A
threshold copied from another deployment is a guess wearing a number.

Two properties of the curve are worth knowing before you read it:

* **A rank-1 `confidence` has a structural floor** of `semantic_weight × fusion_alpha` =
  0.63 with the shipped weights, published as `thresholds.minimumAchievableConfidence`. A
  threshold at or below it selects everything and rejects nothing. This is not hypothetical:
  a `get_low_confidence_fields()` default of 0.6 once sat below that floor and answered
  "nothing to review" on every schema ever matched.
* **`review_threshold` cannot express "nothing matched."** Because of that floor, a rank-1
  candidate can never be `REJECT` on score alone. If what you need is a no-match verdict,
  the setting is `absolute_score_floor` and the guide is
  [Calibrating the absolute score floor](absolute_score_floor.md).

### 4.4 Load it

A JSON or TOML file holding any subset of `MatchingConfig`'s fields:

```json
{
  "auto_approve_threshold": 0.82,
  "review_threshold": 0.50,
  "min_confidence_gap": 0.10,
  "absolute_score_floor": 0.34
}
```

```bash
export NEXUS_API_MATCHING_CONFIG=/etc/nexus/matching.json
```

or, from Python, `NexusMatcher.from_config("matching.json")`. A file may wrap the fields in
a `[matching]` table so one project file can hold several sections.

**An unknown key is a startup error, not a warning.** A mistyped `auto_approve_treshold`
would otherwise be dropped in silence and leave you believing you had raised the bar while
the matcher went on approving at 0.87.

### 4.5 Verify it is in force

The failure this step exists for is a config file that is loaded, parsed, and then not
applied — which looks exactly like a config file that works.

```bash
curl -s localhost:8000/api/v1/status | jq '.calibration.overrides, .thresholds.autoApprove'
```

```json
{
  "auto_approve_threshold": 0.82,
  "absolute_score_floor": 0.34
}
0.82
```

`overrides` naming your settings, with your values, is the confirmation. An empty
`overrides` after setting `NEXUS_API_MATCHING_CONFIG` means the file was not read, or that
it restates the shipped defaults.

---

## 5. Re-measure after anything that moves the distribution

The auto-approve threshold is downstream of everything that changes a score. Re-run the
sweep after:

* **an encoder change**, including an accidental one — `encoder.fallbackInForce` on
  `/api/v1/status` is `true` when selection fell through to a lower rung, and a server in
  that state is not scoring with the encoder any published number was measured on;
* **a retrieval change** — `fusion_alpha`, `dense_top_k`, `dictionary_alias_count`, the
  query representation;
* **a dictionary change of any size** — entries added, definitions rewritten, a domain
  merged in;
* **a glossary that grew past an order of magnitude**, which is what `UNCALIBRATED_SIZE`
  is telling you about.

Note the counterintuitive direction, measured twice in this repository during tuning:
**better retrieval LOWERS auto-approve precision at a fixed threshold.** It shifts the score
distribution upward, which pushes more candidates over a fixed bar — including wrong ones. An
improvement that is not followed by a re-fit silently loosens the boundary that decides which
definitions ship.
