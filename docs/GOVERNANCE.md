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
| `governance` | The `ProtectionClass` this match would confer, or `None` when the entry has no code and therefore sits at the open tier. |

A rejected match carries no class: `MatchResult` clears `governance` when the decision is
`REJECT`, so a refused match cannot confer anything.

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
