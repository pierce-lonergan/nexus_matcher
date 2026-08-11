# Governance inheritance

A caller matches a schema field against a business glossary **in order to inherit that
glossary entry's governance**. The match is the means; the classification is the point.

This document explains the model. A runnable example lives in
[`examples/governance/`](../examples/governance/README.md) — a fictional vocabulary, a
glossary, a schema, a calibration set and five commands.

## The library ships no taxonomy, and that is the design

NexusMatcher defines **no protection codes and no classification tiers**. It has no
built-in notion of what "restricted" means, no list of sensitive categories, and no
default policy. The controlled vocabulary is a JSON file the **caller** supplies; the
library loads it, validates the glossary against it, and passes it through.

That is not a limitation worked around. It is correct twice over:

- **A taxonomy belongs to an organisation.** Codes, tier names, the number of tiers and the
  rules attached to each differ between organisations, and the catalog itself is frequently
  material the organisation does not publish. A library with one organisation's codes
  compiled in is useless to every other one.
- **Baked-in codes are how a customer's own classification policy leaks into a public
  repository.** If the vocabulary is a file the caller owns, there is nothing to leak.

`ProtectionLevel` — the five-value enum on `DictionaryEntry.protection_level` — is
deliberately *not* the governance model. It is a coarse, lossy summary the library has
always had; an organisation's "Highly Restricted" collapses into `RESTRICTED` and the
distinction is gone. Governance is carried separately, in the caller's own vocabulary, and
the raw source string is preserved in `source_metadata` alongside it.

## Five rules

### 1. A code implies a tier. The tier is derived, never free text.

Each entry in the vocabulary declares a `code`, the `classification` that code derives, a
`personal_information` flag and a `direct_identifier` flag. Given a code, the tier follows.

```json
{
  "code": "MANIFEST_NAME",
  "name": "Passenger manifest identity",
  "classification": "SEALED_RESTRICTED",
  "personal_information": true,
  "direct_identifier": true,
  "enhancement": "MASK_IN_LOGS"
}
```

`enhancement` is an optional handling instruction the caller attaches — masking,
tokenisation, a retention rule. The library passes it through untouched and never
interprets it.

### 2. A glossary row whose stated tier contradicts its code is a data defect

Glossaries usually carry both: a code column and a human-readable classification column.
When they disagree, the code wins and the row is **refused**, not corrected in silence.

```
protection_class = MANIFEST_NAME    -> derives SEALED_RESTRICTED
classification   = CREW_ONLY        -> contradicts its own code -> REFUSED
```

Indexing that row would let a field inherit a tier its own code disowns. `problems_with()`
reports it and a strict load refuses the whole file rather than returning the good rows —
because the rows that would vanish are exactly the rows whose governance is wrong, so the
caller ends up with a glossary that looks healthy and inherits nothing where it should
have inherited something.

The catalog also wins over the row for the two flags, and the disagreement is reported so
somebody fixes the source.

### 3. An unmatched field inherits nothing

A field with no match, or whose match was not accepted, carries no class. Not a default
class, not a fallback tier, not the most common tier in the glossary — nothing. A field
whose glossary entry carries no code inherits nothing either, and sits at the vocabulary's
open tier.

The open tier is named by the caller (`open_classification`). Left unset it is the
sentinel `UNCLASSIFIED`, which is deliberately not a word a real taxonomy uses, so an
unconfigured vocabulary cannot be mistaken for a configured one.

### 4. Unknown codes are rejected; legacy tokens are declared

A code the vocabulary does not define is not a class. It is never stored — a field
carrying a label nobody defined reads as governance and is not. The raw token survives
only in `source_metadata['governance_code_raw']`, as evidence for whoever fixes the source.

Legacy spellings are handled by **declaring** them. An alias maps an old token onto a
current code; an alias mapping to `null` declares a token to be noise that must be dropped.
Both are explicit, so "we quietly dropped something" cannot happen unnoticed.

```json
"aliases": {
  "GBF-LEGACY-NAME": "MANIFEST_NAME",
  "n/a": null
}
```

### 5. Over-inheriting is the expensive error

Under-inheriting is loud: a field that should have been protected and was not eventually
shows up in an audit or an incident. Over-inheriting is silent — a field tagged more
closed than it warrants breaks nothing, upsets nobody, and quietly stops a lawful use of
the data. Nobody files a ticket about data they were told they could not use.

So the safe outcome for an ambiguous match is a human, not a guess. In the example pack, a
salted digest of a passenger name retrieves the passenger name entry as its top candidate.
Inheriting it would tag a pseudonym as a `SEALED_RESTRICTED` direct identifier. The match
is not auto-approved, and that is the correct behaviour.

## What a match carries

`MatchResult` carries governance as first-class fields, not as something to fish out of
`source_metadata`:

| Attribute | Meaning |
|---|---|
| `governance_id` | The matched entry's id, which **is** the governance id. Always populated. |
| `governance` | The `ProtectionClass` this match would confer, or `None`, which means one of two things — below. |

`governance` is resolved on **every** candidate, not on rank 1 alone, because the fact that
decides between rank 1 and rank 2 is usually which of the two is a direct identifier.

So `None` carries two meanings, and `decision` does not separate them:

- **The matched entry carries no code**, so it sits at the open tier and has nothing to
  confer.
- **This is the rank-1 candidate and the matcher rejected it.** A rejected top match means
  no entry in this glossary describes the field, and a novel field — the case that most
  needs a human — would otherwise arrive wearing the class of the least-bad candidate.
  `MatchResult` clears the class there, and only there.

**A rejected runner-up keeps the class its entry confers.** The rank qualifier is the rule,
not a detail of it. Counted over `examples/governance/run_pack.py` on the 26-field pack at
`top_k=5`: 79 of the 104 runner-up candidates are `REJECT`, 66 of those name an entry
carrying a real code, and 16 of those are direct identifiers. Clearing on every `REJECT`
blanked all 66, so a reviewer comparing rank 1 against rank 2 read `null` for both of the
things they needed to tell apart — "this entry has no class" and "this entry's class was
withheld" — on the comparison the field exists to make.

The rank-1 clause does not fire at the shipped numbers: the structural floor described in
the next section sits above `review_threshold`, so a top-1 match cannot be rejected. It is
there for a caller who raises `review_threshold` past that floor.

## Reading the decision, not the score

`final_confidence` is a rank-relative number. It is the min-max normalised fused retrieval
score, so the rank-1 candidate lands near `fusion_alpha` whether the match is excellent or
implausible, and it has a structural floor of `semantic_weight × fusion_alpha`.

Two consequences that matter for governance, both demonstrated in the example pack:

- **A high confidence is not evidence of a good match.** On the example pack a field
  nothing governs scored higher than several fields that matched correctly.
- **`REJECT` is unreachable for a top-1 match in the shipped configuration**, because the
  structural floor sits above `review_threshold`. `REVIEW` therefore means "a human must
  decide", never "probably fine".

What separated the bad matches from the good ones on that pack was the **margin over the
runner-up** (`min_confidence_gap`), not the confidence.

## Matching over HTTP

Everything above is reachable over HTTP. `create_app()` registers `POST /api/v1/match`,
`POST /api/v1/match/batch` (identical contract, higher field cap) and `POST /api/v1/feedback`
(appends a reviewer's verdict to an audit log; never fed back into ranking).

Eleven sentences across six documents said otherwise, in the register this repository
reserves for being honest about what is missing — which is exactly why they were believed.
They are retracted, and `tests/packaging/test_documented_routes.py` now fails the build if a
document denies a route `create_app()` registers.

### Starting a server

The vocabulary is caller-supplied and both shipped entry points — `nexus-matcher api` and
`uvicorn ...app:create_app --factory` — call `create_app()` with no arguments, so the two
files reach the server through the environment:

```bash
export NEXUS_API_DICTIONARY=examples/governance/glossary.csv
export NEXUS_API_GOVERNANCE=examples/governance/protection_classes.json
nexus-matcher api --host 127.0.0.1 --port 8000
```

[`examples/governance/serve.sh`](../examples/governance/serve.sh) and
[`serve.ps1`](../examples/governance/serve.ps1) are exactly that, plus
`NEXUS_API_FEEDBACK_PATH` so `/api/v1/feedback` works too. Without
`NEXUS_API_DICTIONARY` the match routes answer **503** naming the setting.

`NEXUS_API_GOVERNANCE` is the one worth checking twice, because forgetting it fails
*quietly*: a server with a glossary and no vocabulary would answer **200** with
`"governance": null` on every field — which is rule 3's "inherits nothing" and a
misconfigured server wearing the same face. So the bootstrap refuses to bring a matcher up
at all in that state. Verified against this pack: the process starts, `/health/ready` goes
503, and every match answers 503 quoting the column it found —

```
examples/governance/glossary.csv has a protection-code column ('protection_class') and no
vocabulary is configured to interpret it, so every match would come back with
governance=null -- indistinguishable, to the caller, from a glossary that carries no
classes at all. Set NEXUS_API_GOVERNANCE to the JSON file that declares those codes, or
remove the column.
```

That check reads the glossary's header against the code-column aliases, so it closes the
common case and not the general one: codes living under a column name the alias list does
not recognise are still invisible to it.

### The request contract, which is not the example pack's field names

A request is `{"fields": [...], "top_k": 5, "explain": false}`, and each field is exactly
these four keys:

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | The column's own name. |
| `path` | no | The caller's identifier for the field, **and the key the response is returned under**. Defaults to `name`. |
| `doc` | no | Column comment or description. Real retrieval signal — an entry with an empty definition gives the encoder very little to work with. |
| `type` | no | Source type name, normalised server-side. Unknown types are accepted. |

**Send a dotted `path`.** The segment before the last dot becomes the retrieval query's
parent context, and that is the single largest accuracy factor measured on this task
(+20 points of P@1 — `benchmarks/results/exp_query_repr_combined.json`). `booking.passenger.legal_name`
is worth more than `legal_name`, and it is also the key you look the answer up under.

**These are not the key names in
[`examples/governance/fields.json`](../examples/governance/fields.json).** That file uses
`flattenedName` and `dataType`, which are the *pack's* input format, not the wire contract.
`FieldSpec` sets `extra="forbid"` on purpose — a misspelled `documentation` silently ignored
would drop the column comment and quietly cost accuracy with nothing to indicate why — so
pasting a row from the pack into a request body is a **422**, and both reviewers who built
against this pack hit exactly that. The body names all three problems:

```json
{"error": {"code": "NEXUS-8004",
  "message": "The request body is not valid. See details.violations for the exact fields and why each was rejected.",
  "details": {"status_code": 422, "violations": [
    {"location": ["body", "fields", "0", "name"], "message": "Field required", "type": "missing"},
    {"location": ["body", "fields", "0", "flattenedName"], "message": "Extra inputs are not permitted", "type": "extra_forbidden"},
    {"location": ["body", "fields", "0", "dataType"], "message": "Extra inputs are not permitted", "type": "extra_forbidden"}]}}}
```

### One request, and what comes back

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/match \
  -H 'Content-Type: application/json' \
  -d '{"fields":[{"name":"legal_name","path":"booking.passenger.legal_name","doc":"Full legal name of the passenger as printed on the sailing manifest.","type":"string"}],"top_k":1}'
```

Against the ferry pack, pretty-printed here — the response is one line, and two identical
requests produce identical bytes:

```json
{
  "results": {
    "booking.passenger.legal_name": [
      {
        "rank": 1,
        "governanceId": "GBF-0001",
        "businessName": "Passenger Legal Name",
        "definition": "The full legal name of a ticketed passenger as printed on the Gravel Bay sailing manifest.",
        "domain": "Passenger",
        "governance": {
          "code": "MANIFEST_NAME",
          "name": "Passenger manifest identity",
          "classification": "SEALED_RESTRICTED",
          "personalInformation": true,
          "directIdentifier": true
        },
        "confidence": 0.904167,
        "decision": "AUTO_APPROVE"
      }
    ]
  }
}
```

`results` carries **one key per input field, keyed by that field's own `path`, in the order
sent** — a field nothing matched gets `[]`, never a missing key. That is rule 3 as a wire
contract: a field cannot silently vanish from the map and inherit nothing unnoticed.

`governance` is the protection class the matched entry confers, drawn from *your*
`protection_classes.json`; `null` means the entry carries no code and sits at the open tier.
`code`, `name` and `classification` are your vocabulary's own strings — the library defines
none of them and does not type them as a closed set.

Do not diff against `confidence`. It is the rank-relative number described above, it will
move with any retrieval change (H-001), and `decision` is what carries the verdict. The
value shown is what this pack produced on the day it was written, not a promise.

## Two hazards this feature sits inside

Both are in [`docs/HAZARDS.md`](HAZARDS.md), and both apply directly here.

**H-001 — better retrieval lowers auto-approve precision at a fixed threshold.** Improving
retrieval lifts the whole score distribution, so more candidates cross a fixed bar,
including wrong ones. The metric that decides whether a class is applied *without a human*
gets worse while the headline gets better. It has happened three times in this repository.
Never report a retrieval number for a governance change without the decision number beside
it.

**H-002 — a threshold calibrated on one corpus does not transfer.** Corpus size is a
regime change, not a scaling factor: one option in this library measures +1.9 P@1 at 688
entries and −18.8 at 30,000. Calibrate against your own glossary, at its size, and
re-calibrate after any change to the encoder, the fusion weights, the query representation
or the glossary itself.

## Getting started

Copy [`examples/governance/`](../examples/governance/README.md), replace
`protection_classes.json` with your own vocabulary, point the glossary at yours, and run
the five commands. The example's vocabulary is fictional — an invented ferry operator —
precisely so that nobody mistakes it for a taxonomy worth adopting.
