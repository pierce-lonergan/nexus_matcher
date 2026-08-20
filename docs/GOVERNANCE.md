# Governance inheritance

A caller matches a schema field against a business glossary **in order to inherit that
glossary entry's governance**. The match is the means; the classification is the point.

This document explains the model. A runnable example lives in
[`examples/governance/`](../examples/governance/README.md) — a fictional vocabulary, a
glossary, a schema, a calibration set and five commands.

Two members carry the whole contract, and both are defined in the next section:
`governanceId`, which is how you join a match back to your own system, and
`governance.code`, which is **an access-control class** — a description of how protected a
data element is, of the kind an organisation writes in order to decide who or what may read
a column. Because the second one is security-relevant, the single most important section in
this file is [the fail-open
hazard](#the-fail-open-hazard-a-null-class-is-not-no-restriction): a `governance: null`
read as "no restriction" is a column made world-readable.

## The two definitions this document rests on

An enterprise adoption review listed both members below as documentation **blockers**: the
library published them, a consumer had to build against them, and nothing in this
repository said in one stable place what either of them **is**. Both are answered here.
The second one changes the register of everything after it.

### governanceId — your own identifier for the entry, carried through unchanged

`governanceId` is **the id of the matched glossary entry, in your id space**. It is the
`id` column of the row you loaded, echoed back. The library mints nothing and owns nothing
here: this is the handle you use to **join a match back to your own system** — to the
glossary row, its steward, the ticket, the catalog record — and a handle you cannot join
on is not a handle.

It is **opaque**, and it is typed **`string`**.

Opaque is a promise with four parts:

- The library never **parses** it. There is no integer path, no UUID path, no prefix
  convention, no check digit.
- The library never **normalises** it. The single transformation anything in this library
  applies to an id is the loader stripping whitespace from around the cell.
- The library never **compares it numerically**. Indexing, `POST /api/v1/lookup`,
  `expected_governance_id` on the retrieval diagnostic, an approved pair's stored id —
  every one of them compares the exact bytes.
- The library never **interprets** it. Whatever your id encodes — a sequence number, a
  system-of-record prefix, a domain letter — means nothing here and is yours to read.

`string` is what the schema says, not a convenience: `governanceId` is `"type": "string"`
on both `MatchCandidateView` and `LookupEntryView` in a live `/openapi.json`, the `ids`
array on `POST /api/v1/lookup` is `"items": {"type": "string"}`, and the generated Java
client binds it as `String`.

Two things in this library can put an id where your source had none, and both are yours to
switch on: `id_prefix` on a load concatenates a prefix **you** chose, and a row carrying no
id at all is given a stable content digest rather than a row number — a row-number id
renumbers the rest of your glossary the first time a row is deleted.

#### The trap, because "just a number" is the case where this bites

"A number from 1 to 10000000" is an ordinary id scheme, and a deployment that writes those
numbers zero-padded is exactly where `string` stops being a formality. Three ids that are
the same *number* are three different *entries*. Captured today, one CSV through
`load_entries`:

```
  id column        entry id
  ---------        --------
  "  0000123  " -> '0000123'            surrounding whitespace stripped, padding kept
  "123"         -> '123'
  "0123"        -> '0123'
  ""            -> 'ad77dbe304e1d05a'   no id in the row, so a content digest
```

A padded id survives to the wire byte for byte. Against a 30-entry glossary whose ids are
zero-padded seven-digit numbers, captured today from a live server and pasted unedited
except for pretty-printing (`top_k: 1`, the response's other blocks omitted):

```json
{
  "results": {
    "booking.passenger.legal_name": [
      {
        "rank": 1,
        "governanceId": "0000123",
        "businessName": "Passenger Legal Name",
        "definition": "The full legal name of a ticketed passenger as printed on the Gravel Bay sailing manifest.",
        "domain": "Passenger",
        "governance": {
          "code": "MANIFEST_NAME",
          "name": "Passenger manifest identity",
          "classification": "SEALED_RESTRICTED",
          "personalInformation": true,
          "directIdentifier": true,
          "enhancement": "MASK_IN_LOGS"
        },
        "confidence": 0.904167,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.784332,
        "sourceMetadata": {
          "values": {"personal_information": "yes", "direct_identifier": "yes"},
          "droppedKeyCount": 0,
          "renderedKeys": []
        },
        "provenance": "RETRIEVAL"
      }
    ]
  }
}
```

**And the unpadded form does not resolve it.** `POST /api/v1/lookup` with
`{"ids": ["0000123", "123"]}` against that same server, captured today and unedited:

```json
{
  "results": {
    "0000123": {
      "governanceId": "0000123",
      "businessName": "Passenger Legal Name",
      "definition": "The full legal name of a ticketed passenger as printed on the Gravel Bay sailing manifest.",
      "domain": "Passenger",
      "governance": {
        "code": "MANIFEST_NAME",
        "name": "Passenger manifest identity",
        "classification": "SEALED_RESTRICTED",
        "personalInformation": true,
        "directIdentifier": true,
        "enhancement": "MASK_IN_LOGS"
      },
      "sourceMetadata": {
        "values": {"personal_information": "yes", "direct_identifier": "yes"},
        "droppedKeyCount": 0,
        "renderedKeys": []
      }
    },
    "123": null
  },
  "missing": ["123"],
  "vocabulary": {
    "openClassification": "OPEN_DECK",
    "tiersMostOpenFirst": ["OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"]
  }
}
```

`"123"` is in `missing`. There is no silent numeric equivalence, and that is the correct
behaviour: the alternative is a library that decides two of your ids are one id.

So a consumer that round-trips `governanceId` through an integer — `int()`, a `bigint`
column, a JSON parser that coerces numeric-looking strings — has changed the key. It asks
for `123`, is told the entry is missing, and ends up with no governed term for a field that
matched perfectly well. The worse version is a glossary in which both forms exist, which an
int-parse silently merges; [the API reference carries that
capture](API_REFERENCE.md#resolving-an-id-post-apiv1lookup), where `0000123` and `123` are
two different terms carrying two different protection classes.

Store it as text, join on it as text, log it as text.

### governance.code — an access-control class, not a label

`governance.code` is **a deployment's own description of how protected a data element is,
of the kind an organisation writes in order to attach read permissions to a column: which
roles, which services and which people may read it.**

That is a different kind of thing from a business domain or a data type, and the difference
is why this document needed rewriting rather than extending. A code is not a label a
catalog displays. It is an input to a decision about **who or what may read data**, and
this documentation from here on treats it as security-relevant rather than descriptive.

Three things follow, and only the third is new:

- **The library still defines none of them.** Your `protection_classes.json` is the only
  source of codes, tiers, flags and handling instructions; this library ships no taxonomy
  and types none of those members as a closed set. Nothing here changes that, and the next
  section is why it is right twice over.
- **The library still enforces nothing.** There is no permission here, no grant, no ACL,
  no connection to a catalog. It resolves your code against your vocabulary and hands the
  class back. Everything a code *does* is done by whatever you wire it into.
- **So the library has to be honest about what a wrong answer costs.** A metadata field
  that comes back wrong is a wrong label. An access-control class that comes back wrong —
  or missing — is a wrong permission, and the direction it fails in is
  [the fail-open hazard](#the-fail-open-hazard-a-null-class-is-not-no-restriction),
  which is the most important section in this file.

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
interprets it, and it travels all the way to the caller: it is the only member of a class
that says what to **do** with a field rather than what the field **is**, and whoever is
deciding how to protect a column is exactly who needs it. `null` is a normal value — five
of the nine classes in the example pack declare it, meaning the tier is the whole
instruction.

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

**Where a consumer reads this rule.** "The match was not accepted" is a verdict about the
*field*, and the field is where it is published:

| Reading from | Key | Note |
|---|---|---|
| HTTP | `fieldDecisions[path]` | One verdict per column, same keys and order as `results`. |
| Python | `session.field_decisions()[path]` | From `match_schema_session`; a method, not a property. Returns `FieldDecision`. |

Both take four values — `AUTO_APPROVE`, `REVIEW`, `REJECT` and **`NO_MATCH`** — and
`NO_MATCH` is this rule stated on the wire: *this response carries nothing this field may
inherit*. It is a **per-field** verdict and has no per-candidate equivalent: the `decision`
on a candidate has only the first three values, and its `REJECT` says "this candidate is
below the bar", which is a different claim.

The candidates on a `NO_MATCH` field are still returned and still carry their class. They
are evidence for a reviewer, not a classification — **read `fieldDecisions[path]` first, and
on `NO_MATCH` inherit nothing regardless of what `results[path][0].governance` says.** A
captured example, and the reason `REJECT` alone cannot carry this,
are [under `fieldDecisions` below](#where-rule-3-is-published-fielddecisions).

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

## Ranking tiers is your business, and you can say so

Nothing here ranks classifications. `SEALED_RESTRICTED` is not "above" `OPEN_DECK` to any
code in this library; they are two strings out of your file, and a library that ordered
them would be shipping a taxonomy through the back door.

That is also why the [fail-open recipe](#the-safe-recipe) says "your most restrictive
class" and never names one: the only ordering that exists anywhere in this system is the
one your vocabulary declares below, and its last element is the tier that recipe means.

A vocabulary may declare its **own** ladder, and the example pack does:

```json
"tiers_most_open_first": ["OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"]
```

The key is **optional**: a vocabulary without it loads exactly as before and reports no
ordering — treat tiers as incomparable there, never as alphabetical, which sorts
`CONFIDENTIAL` above `PUBLIC`. When it *is* present it is **checked against the classes in
the same file**. A tier some class derives, or your declared `open_classification`, that is
missing from the list refuses the load, with a message naming both the class and the list.

That check is your file against itself. The list is yours, the tiers in it are yours, and
the library still supplies none — it only refuses a ladder with a rung missing under a
class that is standing on it. A tier you declare that no class currently uses is fine; a
policy is allowed rungs this vocabulary does not reach.

The key had shipped in the example pack while being read by no code at all, so anyone who
started their vocabulary by copying that file — which is what the pack is for — inherited a
no-op that looked like a rule. Enforcing it is what makes declaring it worth anything, and
it is what puts the ordering on the wire (`vocabulary.tiersMostOpenFirst`, below).

## What a match carries

`MatchResult` carries governance as first-class fields, not as something to fish out of
`source_metadata`:

| Attribute | Meaning |
|---|---|
| `governance_id` | The matched entry's id, which **is** the governance id. Always populated, always a `str`, [never parsed or normalised](#governanceid--your-own-identifier-for-the-entry-carried-through-unchanged). |
| `governance` | The `ProtectionClass` this match would confer, or `None`, which means one of two things — below. |

`governance` is resolved on **every** candidate, not on rank 1 alone, because the fact that
decides between rank 1 and rank 2 is usually which of the two is a direct identifier.

So `None` carries two meanings, and `decision` does not separate them:

- **The matched entry carries no code**, so it sits at the open tier and has nothing to
  confer. Over HTTP the response names that tier itself, in `vocabulary.openClassification`
  — otherwise "the open tier" is an answer the caller cannot resolve without your file.
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

The rank-1 clause does not fire at the shipped numbers: the structural floor described two
sections down sits above `review_threshold`, so a top-1 match cannot be rejected. It is
there for a caller who raises `review_threshold` past that floor.

Neither of those two meanings is "unrestricted", and the next section is about what happens
when a consumer reads them that way.

## The fail-open hazard: a null class is not "no restriction"

This is the section to read if you are wiring `governance.code` into read permissions, and
it is the reason the definition above had to be written down.

**`governance: null` says nothing about permissions.** It is not "unrestricted", not
"public", not "safe". It is the library saying *this response carries no class for you to
inherit here* — and there are five different situations that produce that sentence, none of
which is a grant.

The failure mode is mechanical and it is easy to write by accident:

```python
# WRONG. Do not ship this.
cls = candidate["governance"]
if cls is None:
    grant_read(column, roles=EVERYONE)     # "no class, no rule, no restriction"
else:
    grant_read(column, roles=roles_for(cls["code"]))
```

A field that comes back `null` becomes **world-readable**. Not unlabelled — readable. And
the mapping is one to one with the number of columns the matcher had nothing confident to
say about, which is the population most likely to contain the columns nobody has looked at
yet.

### Why the direction matters more than the rate

An access-control error in the closed direction is loud and recoverable. Somebody cannot
read a table, they raise a ticket, you look at it, you fix it, and the data was never
exposed while you were deciding.

An error in the open direction is silent in both directions at once: nothing fails, nobody
is blocked, no ticket is filed, and the only party who finds out is whoever reads the data.
Nobody reports being able to read something. So the rate is not the interesting number —
one column of ten thousand is a full exposure of that column, and it will sit there.

The safe reading of "I could not classify this" is therefore your **most restrictive**
class, not your least.

**This is rule 5, not a contradiction of it.** Rule 5 says over-inheriting is the expensive
error, because a field tagged more closed than it warrants quietly stops a lawful use of
the data and nobody files a ticket about that either. Both are true, and they resolve the
same way: rule 5's actual instruction is that *the safe outcome for an ambiguous match is a
human, not a guess*. Fail-open says the same thing from the other side — `null` is not a
guess you are entitled to make. The restrictive class is the **interim** state while the
human decides, chosen because it is the reversible one. A column that was world-readable
for a fortnight cannot be made un-read.

### The five shapes, and the member that tells them apart

The library does not collapse these into one value, and it publishes exactly what each one
needs. All five were captured from live servers today; the three below are the ones a
consumer gets wrong, and the full set is in
[the guide](guides/governance_as_access_control.md#the-five-nulls-and-how-each-one-reads).

| Shape | What it means | Read this |
|---|---|---|
| `vocabulary.openClassification` is **`UNCLASSIFIED`** | **No vocabulary is configured on this server.** Nothing in the response is a classification — not even for entries whose glossary rows carry codes. | `vocabulary.openClassification`, and **nothing else** |
| `fieldDecisions[path]` is **`NO_MATCH`** | This response carries nothing the field may inherit. `results[path][0].governance` may be fully populated; it is evidence for a reviewer, not a classification. | `fieldDecisions` |
| `governance: null` on **rank 1** with `decision: "REJECT"` | The class was **withheld** by the matcher. The entry itself may well carry one — in the capture below, its runner-up out of the same domain carries `CREW_ONLY`. **Not** the open tier. | `decision`, on rank 1 only |
| `governance: null`, `provenance: "RETRIEVAL"`, `decision` not `REJECT` | The matched entry carries no code, so the field sits at your **open tier** — a real tier with a name, which is not the same as no rule. | `vocabulary.openClassification` |
| `governance: null`, `provenance: "APPROVED_PAIR"` | The same open-tier answer, except a named reviewer decided this field and retrieval did not run for it. | `provenance` |

**Rows one and four are identical candidate for candidate.** Same entry, same confidence,
same `decision`, same `null` class, same `fieldDecisions` verdict. The only thing anywhere
in the response that separates "this term is published by policy" from "this server was
never given a vocabulary" is the response-level `vocabulary` block — which is why it rides
on every match rather than sitting on an endpoint somebody has to know to call.

Captured today from a server pointed at the ferry glossary with its protection-code column
renamed to something the bootstrap check does not recognise, and no vocabulary configured:

```json
{
  "results": {
    "booking.passenger.legal_name": [
      {
        "rank": 1,
        "governanceId": "GBF-0001",
        "businessName": "Passenger Legal Name",
        "domain": "Passenger",
        "governance": null,
        "confidence": 0.904167,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.784332,
        "sourceMetadata": {
          "values": {
            "zzz_local_tag": "GBF-LEGACY-NAME",
            "personal_information": "yes",
            "direct_identifier": "yes"
          },
          "droppedKeyCount": 0,
          "renderedKeys": []
        },
        "provenance": "RETRIEVAL"
      }
    ]
  },
  "vocabulary": {
    "openClassification": "UNCLASSIFIED",
    "tiersMostOpenFirst": []
  },
  "fieldDecisions": {
    "booking.passenger.legal_name": "AUTO_APPROVE"
  }
}
```

*(`definition` elided.)*

That is the pack's `SEALED_RESTRICTED` direct-identifier entry — a passenger's legal name —
arriving `AUTO_APPROVE` with no class, and the unread code is visible sitting in
`sourceMetadata` as `GBF-LEGACY-NAME`. **`openClassification` is `UNCLASSIFIED` and
`tiersMostOpenFirst` is empty, and that is the whole of the difference.** Check it once per
response, before you look at a single field.

`UNCLASSIFIED` is this library's own sentinel and deliberately not a word a real taxonomy
uses. The common form of this misconfiguration — a glossary with a recognised
protection-code column and no `NEXUS_API_GOVERNANCE` — is refused at startup, which is why
the capture above had to rename the column to reach the state at all. That refusal reads
the glossary header against the code-column aliases, so it closes the common case and not
the general one, and the client-side check is what closes the rest.

The next row is the one worth seeing after that, because the null there is actively
misleading.
Captured today from a server with `review_threshold` raised to 0.95 — which is what it
takes to make a rank-1 `REJECT` reachable at all, [see below](#reading-the-decision-not-the-score) —
sending one field the ferry glossary cannot answer, at `top_k: 2`:

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

*(`definition` and `sourceMetadata` elided from both candidates; nothing else is edited.)*

Rank 1 is `null` and rank 2 is `CREW_ONLY` — **on two entries out of the same domain of the
same glossary**. The `null` on rank 1 is not a statement that the term is open. It is the
matcher declining to confer a class it rejected, and the runner-up in the same body shows
you what class the neighbourhood carries. A consumer reading `results[path][0].governance is
None` as "unrestricted" has just made a crew telemetry column world-readable on the
strength of the library withholding an answer.

And the inverse trap is in the same family: a field whose `fieldDecisions` verdict is
`NO_MATCH` comes back with a **fully populated** `governance` block on its candidates.
Captured today from a server with `absolute_score_floor` set to 0.70, two fields:

```json
{
  "results": {
    "timetable.route_cd": [
      {
        "rank": 1,
        "governanceId": "GBF-0028",
        "businessName": "Sailing Route Code",
        "governance": null,
        "confidence": 0.925,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.926131,
        "provenance": "RETRIEVAL"
      }
    ],
    "telemetry.quasar_flux_index": [
      {
        "rank": 1,
        "governanceId": "GBF-0022",
        "businessName": "Vessel Heading Degrees",
        "governance": {
          "code": "VESSEL_TELEMETRY",
          "name": "Vessel operational telemetry",
          "classification": "CREW_ONLY",
          "personalInformation": false,
          "directIdentifier": false,
          "enhancement": null
        },
        "confidence": 0.833333,
        "decision": "REVIEW",
        "absoluteScore": 0.600246,
        "provenance": "RETRIEVAL"
      }
    ]
  },
  "vocabulary": {
    "openClassification": "OPEN_DECK",
    "tiersMostOpenFirst": ["OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"]
  },
  "fieldDecisions": {
    "timetable.route_cd": "AUTO_APPROVE",
    "telemetry.quasar_flux_index": "NO_MATCH"
  },
  "scoring": {"confidenceFloor": 0.63, "absoluteScoreFloor": 0.7, "absoluteScoreMetric": "cosine"}
}
```

*(`definition`, `domain` and `sourceMetadata` elided from both candidates.)*

Read only `results` and you get both errors in one response: `timetable.route_cd` looks
unclassified when it is `OPEN_DECK`, and `telemetry.quasar_flux_index` looks `CREW_ONLY`
when the correct answer is that the deployment has no term for it. Read `fieldDecisions`
first and both come out right.

### The safe recipe

Seven steps, in this order. It fits in one function and every value it reads is in the
response you already have. Step 0 is per **response**; the rest are per field.

```
0.  vocabulary.openClassification == "UNCLASSIFIED"
        -> REFUSE THE WHOLE RESPONSE. No vocabulary is configured on that server, so no
           null in it means anything. Fix the deployment; classify nothing. Stop.

1.  verdict = fieldDecisions[path]
2.  verdict == NO_MATCH         -> most restrictive class. Queue for a human. Stop.
3.  verdict in (REVIEW, REJECT) -> most restrictive class. Queue for a human. Stop.
4.  verdict == AUTO_APPROVE     -> take candidate = results[path][0]
5.  candidate.governance is not null -> that class. Permissions from .code.
6.  candidate.governance is null     -> the field sits at vocabulary.openClassification.
                                        Resolve THAT tier against your policy.
```

Four notes, each of which is a thing somebody has got wrong:

- **Step 0 cannot be folded into step 6.** By the time you are looking at a null class you
  cannot tell those two cases apart — the capture above is the proof — and a check written
  inside the null branch reads as an edge case rather than as the precondition it is.
- **Step 6 is not "grant everyone".** `openClassification` names one of *your* tiers. In
  the pack above it is `OPEN_DECK`; in your vocabulary it might be a tier that still
  excludes contractors and third-party services. Whether the open tier is world-readable
  is a question for your policy, and it is a question — not a default.
- **Never derive a permission from `confidence`.** It is min-max normalised inside one
  field, its rank-1 value has a structural floor of 0.63 published as
  `scoring.confidenceFloor`, and a field that matches nothing still scores well above it —
  0.833 in both captures above, on a column the ferry glossary has no term for.
- **The library will not pick your most restrictive class for you.** It ranks nothing, by
  design. The only ordering that exists is the one *your* vocabulary declares, in
  `vocabulary.tiersMostOpenFirst`; the last element is your most closed tier. An empty list
  means your file declares no order, and the correct handling then is a constant your
  deployment names — never an alphabetical sort, which puts `CONFIDENTIAL` above `PUBLIC`.

The step-by-step version, with the deployment checks that go with it, is
[Governance as access control](guides/governance_as_access_control.md).

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

**The number that *is* comparable across fields is `absolute_cosine`** —
`absoluteScore` on the wire, `ScoreBreakdown.absolute_cosine` in Python. It is the raw
retrieval score, before any per-field normalisation, so it has no structural floor and can
be compared against a constant. Setting `MatchingConfig.absolute_score_floor` turns a
rank-1 below that constant into `NO_MATCH`, which is the missing verdict this section is
otherwise describing the absence of. It ships **off**, and choosing a value for it is a
measurement on your own corpus, not a number to copy:
[docs/guides/absolute_score_floor.md](guides/absolute_score_floor.md).

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

A request is `{"fields": [...], "top_k": 5, "explain": false}`, optionally carrying
`signals`, `contrast`, `consistency` and `consistency_qualifier_segments` — all defaulted
server-side and all documented in
[API_REFERENCE.md](API_REFERENCE.md#the-matching-request-body). Each field is these four
keys, plus an optional `signals` map of its own:

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

**The strictness stops at the field.** An unrecognised key on the request *envelope* —
alongside `fields`, `top_k` and `explain` — is **ignored**, not rejected. The two rules
disagree on purpose, because the two mistakes cost different things. Inside a field, a
silently ignored key is lost retrieval signal and worse matches with nothing to show for
it. On the envelope, it is a version skew: under `extra="forbid"` a caller sending a key a
newer server understands gets a 422 from an older one, so the first optional request field
this endpoint ever gains would be a breaking change needing a coordinated deploy of your
pipeline. What you give up is bounded and visible in the response you already have — a
misspelled `top_k` is silently the default `5`, a misspelled `explain` silently `false`,
and neither can change a classification. `fields` is required, so misspelling *that* is
still a 422.

### One request, and what comes back

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/match \
  -H 'Content-Type: application/json' \
  -d '{"fields":[{"name":"legal_name","path":"booking.passenger.legal_name","doc":"Full legal name of the passenger as printed on the sailing manifest.","type":"string"}],"top_k":1}'
```

Against the ferry pack, captured from a live app on 2026-08-19 and pasted here unedited
except for pretty-printing. The response is one line, ASCII-only, and two identical
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
          "directIdentifier": true,
          "enhancement": "MASK_IN_LOGS"
        },
        "confidence": 0.904167,
        "decision": "AUTO_APPROVE",
        "absoluteScore": 0.784332,
        "sourceMetadata": {
          "values": {
            "personal_information": "yes",
            "direct_identifier": "yes"
          },
          "droppedKeyCount": 0,
          "renderedKeys": []
        }
      }
    ]
  },
  "vocabulary": {
    "openClassification": "OPEN_DECK",
    "tiersMostOpenFirst": [
      "OPEN_DECK",
      "CREW_ONLY",
      "BRIDGE_SENSITIVE",
      "SEALED_RESTRICTED"
    ]
  },
  "fieldDecisions": {
    "booking.passenger.legal_name": "AUTO_APPROVE"
  },
  "scoring": {
    "confidenceFloor": 0.63,
    "absoluteScoreFloor": null,
    "absoluteScoreMetric": "cosine",
    "absoluteScorePooledOverAliases": false,
    "thresholdableAcrossFields": [
      "absoluteScore",
      "explain.absoluteCosine",
      "explain.scores.lexical",
      "explain.scores.editDistance",
      "explain.scores.type",
      "explain.scores.domain"
    ],
    "comparabilityScopesNarrowestFirst": [
      "WITHIN_FIELD",
      "ACROSS_FIELDS",
      "ACROSS_RUNS"
    ],
    "comparability": {
      "confidence": "WITHIN_FIELD",
      "absoluteScore": "ACROSS_FIELDS",
      "explain.absoluteCosine": "ACROSS_FIELDS",
      "explain.scores.fusedRetrieval": "WITHIN_FIELD",
      "explain.scores.lexical": "ACROSS_FIELDS",
      "explain.scores.editDistance": "ACROSS_FIELDS",
      "explain.scores.type": "ACROSS_FIELDS",
      "explain.scores.domain": "ACROSS_FIELDS"
    }
  }
}
```

`results` carries **one key per input field, keyed by that field's own `path`, in the order
sent** — a field nothing matched gets `[]`, never a missing key. That is rule 3 as a wire
contract: a field cannot silently vanish from the map and inherit nothing unnoticed.

`governance` is the protection class the matched entry confers, drawn from *your*
`protection_classes.json`. `code`, `name`, `classification` and `enhancement` are your
vocabulary's own strings — the library defines none of them and does not type any of them
as a closed set, so a generated client gets `String` and not an enum of somebody else's
tiers. `enhancement` is `null` on a class that declares none.

`sourceMetadata` is your glossary's own enrichment columns for the matched entry, carried
through untouched. It is not governance and nothing in the library reads it.

### Where rule 3 is published: fieldDecisions

```json
"fieldDecisions": {
  "booking.passenger.legal_name": "AUTO_APPROVE"
}
```

Same keys as `results`, same order, **one verdict per column** — and this is the value that
decides whether the field inherits anything. `results[path][0].decision` is a statement
about a *candidate*.

Four values: `AUTO_APPROVE`, `REVIEW`, `REJECT` and **`NO_MATCH`**, and the fourth exists
only here. `NO_MATCH` means *this response carries nothing this field may inherit* — either
the field came back with no candidates at all, or an
[absolute score floor](guides/absolute_score_floor.md) is configured and rank 1 does not
clear it.

**On `NO_MATCH`, inherit nothing — however authoritative `results[path][0].governance`
looks.** The candidates are still returned and still carry their class, because a reviewer
needs to see what the matcher found. They are evidence, not a classification. Captured from
a live server with `absolute_score_floor` set to 0.70, sending one field the ferry glossary
answers and one it cannot:

```json
{
  "results": {
    "telemetry.quasar_flux_index": [
      {
        "rank": 1,
        "governanceId": "GBF-0022",
        "businessName": "Vessel Heading Degrees",
        "definition": "The compass heading a vessel reported at the last telemetry ping.",
        "domain": "Voyage",
        "governance": {
          "code": "VESSEL_TELEMETRY",
          "name": "Vessel operational telemetry",
          "classification": "CREW_ONLY",
          "personalInformation": false,
          "directIdentifier": false,
          "enhancement": null
        },
        "confidence": 0.823333,
        "decision": "REVIEW",
        "absoluteScore": 0.586716,
        "sourceMetadata": {
          "values": {
            "personal_information": "no",
            "direct_identifier": "no"
          },
          "droppedKeyCount": 0,
          "renderedKeys": []
        }
      }
    ]
  },
  "fieldDecisions": {
    "booking.passenger.legal_name": "AUTO_APPROVE",
    "telemetry.quasar_flux_index": "NO_MATCH"
  },
  "scoring": {
    "absoluteScoreFloor": 0.7,
    "absoluteScoreMetric": "cosine"
  }
}
```

A consumer reading only `results` would tag a quasar telemetry column `CREW_ONLY` on a
candidate with a confidence of 0.82. That is rule 5 — over-inheriting is the expensive
error — happening silently, and `fieldDecisions` is the key that stops it.

**`REJECT` is unreachable for a rank-1 match at the shipped numbers**, which is why the
floor exists. `confidence` is min-max normalised inside one field's shortlist, so it has a
structural floor of `semantic_weight × fusion_alpha` = 0.63 (published as
`scoring.confidenceFloor`) sitting above `review_threshold` = 0.50. No lowering of
`review_threshold` recovers "nothing matched"; the floor moves with the weights and the
threshold does not move with it. `absolute_score_floor` is compared against
`absoluteScore` — the raw retrieval score, which has no such floor — and it ships **off**,
because a floor is a statement about a score distribution and the distribution belongs to
your glossary. [Measuring one](guides/absolute_score_floor.md).

### `vocabulary`, and why a `null` needs it

`governance: null` means one of the two things listed under [What a match
carries](#what-a-match-carries), and the first of them — *the entry carries no code, so it
sits at the open tier* — is not readable from the response unless the response says which
tier that is. It did not. `open_classification` was reachable nowhere over HTTP: not on a
route, and the string appeared nowhere in `/openapi.json`. A Java client receiving `null`
had to open a JSON file sitting on the server to find out what it had been told.

So every match response carries the two facts needed to read its own nulls:

| Key | Meaning |
|---|---|
| `openClassification` | The tier a field with no protection code sits at. `UNCLASSIFIED` is the library's sentinel and means no vocabulary is configured. |
| `tiersMostOpenFirst` | Your declared ladder, in your order. `[]` when your vocabulary declares none. |

It rides on the response rather than on a `/vocabulary` endpoint because this body is a
governance artifact: it gets pasted into a ticket and diffed, and the person reading that
ticket cannot be expected to know there is a second call to make. It is constant per
deployment and costs 134 bytes on the response above.

It is **not** a dump of your catalog. Your codes, names and classes are in your own file;
this is the minimum needed to interpret the response it arrives with.

Do not diff against `confidence`. It is the rank-relative number described above, it will
move with any retrieval change (H-001), and `decision` is what carries the verdict. The
value shown is what this pack produced on the day it was written, not a promise.

## Two hazards this feature sits inside

Both are in [`docs/HAZARDS.md`](HAZARDS.md), and both apply directly here. The fail-open
hazard above is a third of the same kind — a property of this problem domain that will keep
producing wrong lines — and it is **not yet in that ledger**, which is stated here rather
than left implicit, because a ledger that is quietly incomplete is the failure H-008 is
about.

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

This applies with full force to `absolute_score_floor`, which is why the library ships no
value for it. Measured against this repository's own ferry pack, the *same* 30-entry
glossary, the *same* encoder and the *same* route gave a 0.07-wide band of free choice for
one field set and a 0.0027-wide illusion of one for another — the two differed only in
whether the fields carried descriptions. Both measurements, and the procedure that produced
them, are in [docs/guides/absolute_score_floor.md](guides/absolute_score_floor.md).

## Getting started

Copy [`examples/governance/`](../examples/governance/README.md), replace
`protection_classes.json` with your own vocabulary, point the glossary at yours, and run
the five commands. The example's vocabulary is fictional — an invented ferry operator —
precisely so that nobody mistakes it for a taxonomy worth adopting.
