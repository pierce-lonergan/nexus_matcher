# Governance as access control

**Audience:** you are wiring NexusMatcher's output into something that decides who or what
may read a column — RBAC roles on a warehouse, a masking policy, a row filter, a catalog
that grants access.

`governance.code` is [an access-control
class](../GOVERNANCE.md#governancecode--an-access-control-class-not-a-label): your own
description of how protected a data element is, of the kind an organisation writes in order
to attach read permissions to columns. This library defines none of them and enforces
nothing. It resolves your code against your vocabulary and hands the class back, and
everything that class *does* is done by the code you are about to write.

This page is that code, stated as a procedure, plus the checks that go with it.

---

## The one-line version

**Never treat a missing classification as a missing restriction.** Absence of a class is
"I could not classify this", and the safe reading of that is your **most restrictive**
class, not your least.

---

## Before you write anything: two constants you have to choose

Both are yours. The library ships neither and will refuse to invent them.

| Constant | Where it comes from | If you skip it |
|---|---|---|
| **Your most restrictive class** | The last element of `vocabulary.tiersMostOpenFirst`, if your vocabulary declares a ladder. Otherwise a constant your deployment names. | Every branch below that says "most restrictive" has no value to use, and the tempting fallback — sort the tiers alphabetically — puts `CONFIDENTIAL` above `PUBLIC`. |
| **What your open tier permits** | Your policy's reading of `vocabulary.openClassification`. | You will read "open tier" as "everyone", which is a policy decision the library never made. In this repository's example pack the open tier is `OPEN_DECK`; in yours it may still exclude contractors and third-party services. |

Declaring the ladder is worth doing for a second reason: when `tiers_most_open_first` is
present, the loader checks it against the classes in the same file and refuses a ladder with
a rung missing under a class that stands on it. An undeclared ladder is unchecked.

---

## The procedure

Once per **response**, before any field:

```
0.  response.vocabulary.openClassification == "UNCLASSIFIED"
        -> REFUSE THE WHOLE RESPONSE. That is the library's sentinel for "no vocabulary
           is configured on this server", so nothing in the body is a classification --
           including candidates whose glossary rows carry codes. Fix the deployment.
           Do not classify. STOP.
```

Then, for one field, keyed by the `path` you sent:

```
verdict = response.fieldDecisions[path]

1.  verdict == "NO_MATCH"
        -> most restrictive class; queue for a human; STOP.
           The candidates in results[path] may carry a full protection class.
           They are evidence for the reviewer, not a classification.

2.  verdict in ("REVIEW", "REJECT")
        -> most restrictive class; queue for a human; STOP.

3.  verdict == "AUTO_APPROVE"
        -> candidate = response.results[path][0]

4.      candidate.governance is not null
            -> that class. Permissions from candidate.governance.code.
               candidate.governance.enhancement, when non-null, is your own
               handling instruction for this class -- masking, tokenisation,
               a retention rule -- and it travels with the class for exactly
               this decision.

5.      candidate.governance is null
            -> the entry carries no code, so the field sits at the tier named by
               response.vocabulary.openClassification. Step 0 has already ruled
               out the sentinel, so this is one of YOUR tiers.
               Resolve it against your policy.
```

Nothing in that procedure reads `confidence`, and step 3 is the only one that reads
`results` at all.

**Step 0 is a precondition, not an edge case, and it does not fold into step 5.** By the
time you are looking at a null class the two are indistinguishable: an unconfigured server
returns the same entry, the same confidence, the same `AUTO_APPROVE`, the same `null` and
the same `fieldDecisions` verdict as a correctly-configured one whose matched entry carries
no code. [The capture is in
GOVERNANCE.md](../GOVERNANCE.md#the-five-shapes-and-the-member-that-tells-them-apart), and
in it the field is a passenger's legal name.

### Why `fieldDecisions` and not `results[path][0].decision`

`decision` on a candidate is a statement about that candidate. `fieldDecisions[path]` is
the statement about the **column**, and it is the only place `NO_MATCH` can be expressed —
`MatchDecision` has three values and `FieldDecision` has four, deliberately, because
widening the candidate enum would have sent a fourth value down a wire whose existing
clients deserialise three.

### Why `REVIEW` locks the column

At the shipped thresholds this is the common verdict — auto-approve fires on about **12% of
fields** on this repository's 688-pair benchmark — so the great majority of your columns
sit at your most restrictive class waiting for somebody. That cost is real, and it is the
right way round, because it is the reversible direction. The way to reduce it is
calibration on your own corpus — [the absolute score floor](absolute_score_floor.md) and
[calibration profiles](calibration_profiles.md) — not a looser default, and never a default
this library picks for a corpus it has never seen.

---

## The five nulls, and how each one reads

`governance: null` is produced by five different situations. The library keeps them
distinguishable and publishes the member that separates each one. All five below were
captured from live servers.

| # | Shape | Meaning | Distinguished by |
|---|---|---|---|
| 0 | `vocabulary.openClassification == "UNCLASSIFIED"` | **No vocabulary on this server.** Nothing in the response is a classification. | `vocabulary.openClassification`, and nothing else |
| 1 | `fieldDecisions[path] == "NO_MATCH"` (candidate class may be **populated**) | Nothing here to inherit. | `fieldDecisions` |
| 2 | `governance: null`, `rank: 1`, `decision: "REJECT"` | The class was **withheld**. The entry may well have one. | `decision`, rank 1 only |
| 3 | `governance: null`, `provenance: "RETRIEVAL"`, `decision` not `REJECT` | Entry carries no code: the **open tier**. | `vocabulary.openClassification` |
| 4 | `governance: null`, `provenance: "APPROVED_PAIR"` | Open tier, and a named reviewer decided this field. Retrieval did not run. | `provenance` |

**Shapes 0 and 3 are identical at every per-candidate key.** That is the whole argument for
step 0, and the capture proving it is
[in GOVERNANCE.md](../GOVERNANCE.md#the-five-shapes-and-the-member-that-tells-them-apart).

### Shape 2 is the dangerous one

Captured today, `review_threshold` raised to 0.95 (which is what it takes to make a rank-1
`REJECT` reachable at all), `top_k: 2`, one field the ferry glossary has no term for.
`definition` and `sourceMetadata` elided:

```json
{
  "results": {
    "telemetry.quasar_flux_index": [
      {
        "rank": 1,
        "governanceId": "GBF-0022",
        "businessName": "Vessel Heading Degrees",
        "domain": "Voyage",
        "governance": null,
        "confidence": 0.833333,
        "decision": "REJECT",
        "absoluteScore": 0.60913,
        "provenance": "RETRIEVAL"
      },
      {
        "rank": 2,
        "governanceId": "GBF-0020",
        "businessName": "Vessel Engine Temperature",
        "domain": "Voyage",
        "governance": {
          "code": "VESSEL_TELEMETRY",
          "name": "Vessel operational telemetry",
          "classification": "CREW_ONLY",
          "personalInformation": false,
          "directIdentifier": false,
          "enhancement": null
        },
        "confidence": 0.773643,
        "decision": "REJECT",
        "absoluteScore": 0.59356,
        "provenance": "RETRIEVAL"
      }
    ]
  },
  "fieldDecisions": {"telemetry.quasar_flux_index": "REJECT"},
  "vocabulary": {
    "openClassification": "OPEN_DECK",
    "tiersMostOpenFirst": ["OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"]
  }
}
```

Rank 1 is `null`; rank 2 is `CREW_ONLY`; both entries are in the same domain of the same
glossary. Reading rank 1's `null` as "unrestricted" makes a crew telemetry column
world-readable on the strength of the library **declining to answer**. The procedure never
gets there: `fieldDecisions` is `REJECT`, so step 2 fires.

### Shape 4, so `provenance` is not a mystery

A field a reviewer approved is answered from the recorded verdict and retrieval is skipped
for it. Captured today from a server with an `ApprovedPairBypass` attached — a deployment
opts into that by passing `feedback_consumer=` to the `NexusMatcher` constructor, and the
shipped wiring attaches none:

```json
{
  "results": {
    "sailing_route_code": [
      {
        "rank": 1,
        "governanceId": "GBF-0028",
        "businessName": "Sailing Route Code",
        "definition": "The short code that identifies a scheduled route between two terminals.",
        "domain": "Published",
        "governance": null,
        "confidence": 1.0,
        "decision": "AUTO_APPROVE",
        "absoluteScore": null,
        "sourceMetadata": {"values": {}, "droppedKeyCount": 0, "renderedKeys": []},
        "provenance": "APPROVED_PAIR"
      }
    ]
  },
  "vocabulary": {
    "openClassification": "OPEN_DECK",
    "tiersMostOpenFirst": ["OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"]
  },
  "fieldDecisions": {"sailing_route_code": "AUTO_APPROVE"}
}
```

This `null` is the strongest one in the set: a named reviewer looked at this column, chose
this entry, and that entry declares no protection code. The field sits at `OPEN_DECK`
because a human said so.

`provenance` is also the reason not to infer any of this from `confidence`. That `1.0` is
not a very good match; the five default scoring weights sum to exactly 1.0 and every signal
caps at 1.0, so ordinary retrieval reaches 1.0 too. `absoluteScore` is `null` and `explain`
is absent because nothing measured this candidate.

---

## On the JVM, this is one call

The procedure above is written out longhand because the rule has to be stated somewhere a
reader can check it against the server's behaviour. If you are calling from Java, do not
hand-write it: `clients/java` implements exactly this recipe, and a hand-written copy is a
second implementation of a security decision.

```java
FieldGovernance governance = response.governanceFor(path);

if (governance.outcome().maySafelyApply()) {
    applyClass(governance.classification());     // CONFERRED or OPEN_TIER
} else if (governance.outcome().needsAHuman()) {
    routeToReview(path, governance);             // the three withheld outcomes
} else {
    haltAndPageAnOperator(path, governance);     // the service could not classify at all
}
```

`GovernanceOutcome` has seven values, and the split is the same one this document argues
for — two that permit writing something, four that do not, and one that says the response
itself cannot be trusted:

| Outcome | Corresponds to | May apply? |
| --- | --- | --- |
| `CONFERRED` | the matched entry carries a code | yes |
| `OPEN_TIER` | the entry carries none, and the vocabulary names the tier it sits at | yes |
| `WITHHELD_PENDING_REVIEW` | the verdict was `REVIEW` | no — a human |
| `WITHHELD_REJECTED_TOP_MATCH` | rank 1 was `REJECT`, so it confers nothing | no — a human |
| `WITHHELD_NO_MATCH` | nothing cleared the absolute floor | no — a human |
| `UNCLASSIFIABLE_NO_VOCABULARY` | the service has no vocabulary configured | no — **an operator** |
| `UNREADABLE` | the response is not shaped as this contract requires | no — **an operator** |

Two of those deserve emphasis, because they are the ones a hand-written check tends to
miss.

`UNCLASSIFIABLE_NO_VOCABULARY` is step 0 of the procedure, and it is a **response-level**
condition, not a per-field one. A service with no vocabulary returns fields that are
byte-identical to correctly-classified open-tier fields at every per-candidate key — the
only difference is in the response's own `vocabulary` block. Read it once per response,
before you look at any field. That is why the outcome exists rather than being folded into
`OPEN_TIER`: the follow-up is a deployment fix, not a review queue.

`maySafelyApply()` is deliberately the only affirmative test. There is no
`isWithheld()` — a caller who has to enumerate the negative cases in order to get the
positive one has already been given the chance to miss one, and missing one here means
writing an unrestricted class onto a column nobody classified.

One divergence to be aware of, and it is deliberate: this document distinguishes a
reviewer-approved answer from an open-tier answer, because they arrive by different routes
and `provenance` tells you which. `GovernanceOutcome` does not — an approved pair carrying
a real code is `CONFERRED`, and one carrying none is `OPEN_TIER`. The outcome answers "what
may I write", and by that question the route does not matter. Read `provenance` when you
care about the route; read `outcome()` when you care about the permission.

## Three checks to run on your own deployment

Each one catches a failure this repository has actually produced.

**1. Assert that some fields come back with a class.** A response where *every* field is
`governance: null` is well-formed, and it is what both a correctly uncoded glossary and a
governance pipeline that silently stopped working look like. Count coded entries at load
(`27 of 30` on this repository's example pack) and fail your own pipeline if that count is
zero while your glossary has a protection-code column. This is exactly the defect
[NM-0033](../../CHANGELOG.md) was, and it is invisible from the response alone.

**2. Assert `vocabulary.openClassification` is not `UNCLASSIFIED` at deploy time, not
only in the client.** Step 0 of the procedure is the client-side guard; this is the same
assertion run once against a live `/api/v1/match` when the service comes up, so the state
is caught by whoever can fix it rather than by whoever is trying to classify. The HTTP
bootstrap already refuses to start in the common form of the misconfiguration — a glossary
with a recognised protection-code column and no `NEXUS_API_GOVERNANCE` — but that check
reads the glossary header against a list of known code-column aliases, so a code column
under a name the aliases do not know still comes up green. That is exactly the state [the capture in
GOVERNANCE.md](../GOVERNANCE.md#the-five-shapes-and-the-member-that-tells-them-apart) was
taken from.

**3. Assert your ids survive your own storage.** Round-trip one `governanceId` from a match
response, through whatever column and serialiser you keep it in, back into
`POST /api/v1/lookup`, and assert `missing` is empty. It is opaque and it is a
[string](../API_REFERENCE.md#governanceid-is-an-opaque-string-and-lookup-is-where-that-becomes-visible);
a `bigint` column or a numeric-coercing JSON parser turns `0000123` into `123`, and lookup
answers **200** with `null` rather than failing loudly.

---

## What this library will not do for you

- **Rank your tiers.** `SEALED_RESTRICTED` is not "above" `OPEN_DECK` to any code here.
  The only ordering that exists is the one your vocabulary declares.
- **Pick a most restrictive class.** That constant is yours, above.
- **Grant, revoke or enforce anything.** There is no permission model in this package.
- **Invent a floor, a threshold or a default class.** [H-002](../HAZARDS.md#h-002--a-threshold-calibrated-on-one-corpus-does-not-transfer):
  a threshold calibrated on one corpus does not transfer, and every one of these numbers is
  a measurement on your glossary.

## See also

- [Governance inheritance](../GOVERNANCE.md) — the model, the five rules, and the fail-open
  hazard argued in full.
- [API reference](../API_REFERENCE.md#the-matching-response) — every key on the wire.
- [Calibrating the absolute score floor](absolute_score_floor.md) — how to make `NO_MATCH`
  reachable on your own corpus, and the measurement showing a plausible-sounding floor can
  produce none at all.
