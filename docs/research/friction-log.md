# Friction log — using the published library on messy input

**2026-08-10, 00:11–00:23 EDT.** `nexus-matcher==2.0.1` from PyPI, installed into a clean
venv at `C:\nmxf\venv` (Python 3.13.4, Windows 11). Never the repo tree. Every finding below
was produced first-hand against the wheel a stranger downloads.

The input was built to be awkward on purpose, because clean input has already been measured
and told us what it was going to tell us:

| File | Shape |
|---|---|
| `orders.avsc` | nested Avro four levels below the root (`cust.addrs[].geo.lat`) — records inside arrays inside records, nullable unions, enums, a map, `decimal`/`timestamp-micros`/`date` logical types, `doc` on some fields and not others |
| `warehouse_ddl.sql` | Snowflake `GET_DDL` export — 3 tables + a view, `create or replace TRANSIENT TABLE`, `NUMBER(38,0)`, `TIMESTAMP_NTZ(9)`, `VARIANT`, `ARRAY`, `WITH MASKING POLICY`, 17 column `COMMENT` clauses |
| `glossary_export.xlsx` | 42 terms as a steward actually sends them — a title banner in row 1, a blank spacer, the real header in row 3, mixed casing (`customer identifier`, `CUSTOMER FULL NAME`), trailing spaces, an em dash, and 3 junk rows |

**From the first documented command (00:14:24) to the last proof (00:23:09): under 9 minutes.**
That is fast, and the speed is misleading. Every blocker below was cleared by reading
`site-packages`, and a user who cannot `grep` an installed package spends a different
afternoon. The honest metric is not minutes; it is **eight times I had to open the source, and
zero times the answer was in the README or in an error message.**

The eight: `ColumnMapping`'s real import path · that `header_row` exists at all ·
`load_dictionary`'s signature · that `NexusMatcher` has no route for pre-loaded entries · which
parsers `from_config()` actually registers · which token `_CREATE_TABLE_RE` rejects ·
the `min_confidence_gap` rule behind an inverted decision · and that two protection-level
normalisers exist.

---

## The timeline

Timestamps are real and every one is checkable against the scripts named at the bottom. The
order is chronological with one exception: the falsification run at 00:35:32 is filed next to
the claim it attacks rather than at the end, because that is where it is worth reading.

### 00:11:22 → 00:11:47 · install, 25 s, no friction — because I cheated

I created the venv at `C:\nmxf` rather than under `Documents\GitHub\...`. That was a
deliberate dodge of DX-004 (`WinError 206`, path too long). It worked, and it worked because
I already knew. A stranger picks their project directory and finds out after downloading 38
packages. **Not our bug; still our first impression.**

### 00:14:24 · first documented command. 1 second to a dead end.

```
$ nexus-matcher match orders.avsc -d glossary_export.xlsx
Error: Read 49 row(s) from glossary_export.xlsx but produced no entries -- the
columns were not recognised (first error: Row 1: Missing business name). Pass
an explicit column_mapping=ColumnMapping(...).
```

Three separate problems in one message, and I only found the third by disproving the first two.

**(a) The remedy names a symbol you cannot import.** `from nexus_matcher import ColumnMapping`
→ `ImportError`. It is not in `__all__` (00:14:38). It lives at
`nexus_matcher.domain.ports.dictionary_loader.ColumnMapping` — four packages deep, behind a
`domain.ports` path that reads private. I found it by grepping site-packages (00:14:46). The
README has the same gap: it says "Pass `column_mapping=ColumnMapping(...)` to override the
detection" and never shows the import.

**(b) The remedy cannot be applied from the surface that recommends it.** `nexus-matcher
match --help` has no `--column-mapping`, no `--header-row`, no `--sheet`. The CLI's only
advice for a CLI failure is "write Python instead."

**(c) The diagnosis is wrong.** I took the advice anyway and passed a fully explicit
`ColumnMapping` naming every column my file actually has (00:15:24):

```
EXPLICIT MAPPING -> FAILED: Read 49 row(s) ... the columns were not recognised
                    ... Pass an explicit column_mapping=ColumnMapping(...).
```

Byte-identical message, including the advice to do the thing that had just failed. My columns
were never the problem. My header was on row 3:

```
header_row=0 header -> ['ACME Enterprise Business Glossary — export 2026-08-07 (owner: d.ramos)',
                        '', '_1', '_2', '_3', '_4', '_5']
header_row=2 header -> ['ID','Business Name','Logical Name','Definition','Data Type',
                        'Subject Area','Classification']   | data rows: 47
```

The error described a cause that was not present and prescribed a fix that could not work.
**Two and a half minutes of wall clock and four source-file reads trace back to this one
message** — and only because I could grep the installed package. The information needed to
write a correct message was already in hand: the loader had the header it read, and printing it
would have made the problem self-evident in one line.

### 00:15:03 · the fix exists, was written for this exact file, and is unreachable

`ExcelDictionaryLoader._load_rows` documents:

> `header_row`: 0-based row holding the headers (default: 0), **for exports that open with a
> title banner above the real header**

Someone met this file before me and built the escape hatch. It is reachable from neither
user-facing surface:

- `NexusMatcher.load_dictionary(source, column_mapping, source_type)` — no `**options`. It
  calls `loader.load(path, column_mapping)` and drops everything else on the floor.
- The CLI has no flag.
- `NexusMatcher`'s entire public surface is six names — `from_config`, `load_dictionary`,
  `match_schema`, `match_schema_session`, `dictionary_size`, `is_ready`. There is **no public
  way to hand the matcher entries you loaded yourself.**

`header_row` is also absent from the documented `**kwargs` of both ingest functions —
`load_entries` lists "`sheet`, `delimiter`, `encoding`", `build_index` lists "`query`,
`columns`, `sheet`, `id_prefix`". It works. Nothing says so. I learned it existed by reading
`excel.py`.

**Workaround #1**, written at 00:16:57, and the shape of it is the finding:

```python
m = NexusMatcher.from_config()
entries = load_entries("glossary_export.xlsx", header_row=2)
m._index_dictionary(entries)          # private; no public route exists
```

### 00:16:17 · three junk rows became three dictionary entries, silently

`load_entries(..., header_row=2)` → **45 entries** from 42 terms:

```
id='3edf6679453b0059'  name='--- FINANCE SECTION (added by j.okafor 2025-11) ---'
id='GL-XXX'            name='TBD'   def='awaiting steward sign-off'
id='5c143b4a082dd699'  name='End of export. 42 approved terms, 1 pending.'
```

`load_entries` returns a bare `list` — no statistics object, no warnings channel. The loader
path does keep `LoadStatistics`, and on the same file it reported `total_rows=47 valid=45
errors=2 skipped=0`, `warnings=[]`. The two blank rows were counted as *errors*; the three
section markers were counted as *valid entries*. Nothing distinguishes "a term" from "a note
the steward left in column B". They are now distractors competing for every field.

### 00:17:32 → 00:23:09 · the governance payload. This is the one that matters.

The stated use case is "so the object inherits that entry's classification." I checked what
gets inherited.

`ProtectionLevel.from_string` — exported in `__all__`, called by
`BaseDictionaryLoader._convert_row`, which is what `load_dictionary` runs — does an exact
`member.value == level_upper` comparison and falls through to `INTERNAL`:

```
from_string('PII'              ) -> PII
from_string('RESTRICTED'       ) -> RESTRICTED
from_string('Restricted - PII' ) -> INTERNAL      <-- downgrade
from_string('Restricted - PCI' ) -> INTERNAL      <-- downgrade
from_string(''                 ) -> INTERNAL
from_string('banana'           ) -> INTERNAL
```

An unparseable classification is indistinguishable from a deliberate `INTERNAL`.

The *other* documented ingest path does it properly. `application.ingest._coerce_protection`
handles negations, matches on word boundaries, resolves ambiguity to the **stronger** level —
its docstring says "Under-protecting a field is the expensive mistake" — and preserves the
original string as `source_metadata['governance_raw']` because "the value a caller most needs
was silently dropped" in an earlier version.

Same file, two documented paths, different answers (00:17:59):

```
entries where the two paths DISAGREE on protection level: 2
  GL-003  Email Address                     ingest=RESTRICTED   loader=INTERNAL
  GL-029  Payment Card Number - Last Four   ingest=RESTRICTED   loader=INTERNAL
```

Then the end-to-end run at 00:23:09 — **no workarounds, no private access, a clean CSV, the
README quickstart verbatim**:

```
load_dictionary: total=45 valid=45 errors=0 warnings=[]

  pay.pan_last4    -> Payment Card Number - Last Four       protection=INTERNAL    AUTO_APPROVE
  cust.eml         -> Email Address                         protection=INTERNAL    AUTO_APPROVE
  cust.ssn_last4   -> national identity number - last four  protection=RESTRICTED  AUTO_APPROVE
```

The card number and the email address are matched **correctly**, auto-approved with **no human
in the loop**, and stamped **INTERNAL** on a glossary that says `Restricted - PCI` and
`Restricted - PII`. The load reports zero errors and zero warnings.

### 00:35:32 · I tried to falsify this, and it got worse

The claim above is "the normaliser causes the downgrade." The way to break it is to change
**only** the classification string and see whether the loader path still disagrees. If a
different literal fixes it, nothing about my file is responsible:

```
Payment Card Number - Last Four:        Street Address Line One:
   'Restricted - PCI'    -> INTERNAL       'Confidential'         -> CONFIDENTIAL
   'RESTRICTED'          -> RESTRICTED     'Highly Confidential'  -> INTERNAL
```

The claim survived — one literal, one difference, everything else held constant. And the
control case exposed a sharper defect than the one I was testing:

**Making a classification stricter makes the parsed protection weaker.** `Confidential` parses
to `CONFIDENTIAL`. `Highly Confidential` parses to `INTERNAL`. A steward who tightens a label
loosens the field, silently, and the tighter the wording the further it falls. This is the
precise inversion `_coerce_protection`'s docstring says it was written to prevent — *"strictest
first, so 'highly confidential' beats 'confidential'"* — and the path the README tells you to
use does the opposite of the thing the package already knows to do.

This is NM-0005's class — a silent governance failure — and the correct implementation is
already in the package, thirty files away, called by the other path. It is the same shape as
the bug `detect_column_mapping`'s own docstring warns about: *"two independent notions of
'which column is the business name' is how the loader path ended up rejecting files the ingest
path read without complaint."* Same disease, different column, still live.

### 00:18:42 / 00:22:40 · both machine-readable formats drop the answer

`-f json`, the documented governance output, for `pay.pan_last4`:

```json
"dictionary_entry": {"id":"GL-029","business_name":"Payment Card Number - Last Four",
                     "logical_name":"pan_l4","data_type":"string"}
```

No `protection_level`. `-f csv` is worse — `field_path,rank,business_name,logical_name,
confidence,decision` — it drops the entry `id` too, so you cannot even join back to the
glossary to recover the classification by key.

I also measured how much the emitted `scores` block actually says. Across 43 rank-1 matches:

| Component | Distribution |
|---|---|
| `semantic` | **1.0 on 31 of 43** (72%), then 0.9 ×4, and 7 one-off values |
| `lexical` | 1.0 ×29, 0.0 ×12 — effectively a boolean |
| `type` | 1.0 ×21, 0.5 ×11, 0.8 ×6, 0.0 ×5 |

`semantic` is the min-max-normalised fused retrieval score, so rank 1 gets 1.0 almost by
definition. It is a statement about position in a list, printed in the column an auditor reads
as similarity. Seven fields share confidence `0.925` exactly. The JSON also carries 3 of the 5
components, so the emitted numbers cannot reproduce the emitted confidence.

### 00:19:16 → 00:21:06 · the Snowflake DDL

```
$ nexus-matcher match warehouse_ddl.sql -d glossary_clean.csv
Error: No parser found for extension .sql
```

The CLI's own `--help` says `Path to schema file (Avro, JSON Schema, SQL DDL)`. `from_config()`
registers `avro` and `flattened_avro` only. `SqlDdlParser` and `JsonSchemaParser` ship in the
wheel, unregistered. The README's remedy is "pass the others explicitly via
`schema_parser_registry`" — a constructor argument, and the README's own advice two sections
earlier is *not* to use the constructor, because it makes you supply an embedding provider, a
vector store and a sparse retriever by hand. The remedy costs you `from_config()` entirely.

**Workaround #3:** `m._schema_parsers["sql_ddl"] = SqlDdlParser()`. Private, again.

Then (00:20:03):

```
ValueError: Invalid DDL: No CREATE TABLE statement found
```

— on a file containing three `CREATE TABLE` statements. Isolated at 00:20:41:

```
can_parse  True   CREATE TABLE T                     can_parse  False  CREATE OR REPLACE TABLE T
can_parse  True   CREATE TEMPORARY TABLE T           can_parse  False  create or replace TRANSIENT TABLE T
can_parse  True   CREATE TABLE IF NOT EXISTS T       can_parse  False  CREATE EXTERNAL TABLE T
                                                     can_parse  False  CREATE OR REPLACE ICEBERG TABLE T
```

`_CREATE_TABLE_RE` allows `GLOBAL|LOCAL`, `TEMP(ORARY)`, `UNLOGGED`, `IF NOT EXISTS` — the
PostgreSQL vocabulary. It does not allow `OR REPLACE`, which is what Snowflake's `GET_DDL`
emits, what BigQuery and Databricks emit, and what dbt writes for every table materialization.
The error blames the absence of `CREATE TABLE` rather than naming the token it choked on.

**Workaround #4:** a regex over the file to delete `OR REPLACE` and `TRANSIENT`. It then
parses, and two more things go wrong quietly (00:21:06):

```
schema name='DIM_CUSTOMER'  fields=17          (the file declares 3 tables)
fields carrying a description: 0 / 17          (the DDL has 17 COMMENT clauses)
```

**Every column `COMMENT` is dropped.** `COMMENT 'Unique identifier assigned to a customer
account'` is the business definition, sitting in the DDL, free. The README says definitions are
the strongest signal after the path and worth +19.3 P@1. The DDL parser reads the only
definition source a DDL file has and discards it.

**`match_schema` sees only the first table.** `parse_all()` finds all three (17 + 16 + 9 = 42
fields); `match_schema` calls `parse()`, gets `DIM_CUSTOMER`, matches 17 fields and reports
success. 25 columns — including `PAN_LAST4` and `ACCT_BAL_AFTER` in `FCT_PAYMENT` — were never
classified, and nothing said so. The only summary any surface prints is a count of fields
matched, so a partial parse and a complete one are indistinguishable from the output.

### 00:21:46 · a higher confidence got the worse decision

From the DDL run:

```
field             conf  runner-up   margin  decision
FULL_NM         0.9036     0.5725   0.3311  AUTO_APPROVE
EFF_FROM_TS     0.9061     0.8433   0.0628  REVIEW
```

0.906 → REVIEW. 0.904 → AUTO_APPROVE. (The Avro run has the same shape: `cust.addrs[].pc`
scores 0.875, clears the 0.87 bar, and goes to REVIEW.)

The cause is correct and is good design: `min_confidence_gap = 0.1` blocks auto-approval when
rank 1 and rank 2 are close, because a near-tie is exactly what a human should adjudicate.
The defect is that **nothing in any output says so.** `MatchResult` exposes `decision`,
`final_confidence`, `rank`, `score_breakdown` — no margin, no reason. The JSON has no reason
field. A reviewer opening a queue sorted by confidence sees two adjacent rows with inverted
decisions and no explanation, and the mechanism that produced it is documented nowhere in the
README. The rule is derivable from rank-2 in the JSON, but only if you already know it exists.

While there: `MatchingConfig().auto_approve_threshold` is **0.87**. The README's Limitations
section says "0.85 is calibrated on this benchmark"; the Quickstart says "every score sits
under the 0.87 bar". The README states two different values for its own default.

### 00:22:27 · the review list, and who is on it

```
get_low_confidence_fields() default (0.6) -> 0 fields
  threshold=0.87 -> 15      top-1 confidence: min 0.707 max 0.983
  threshold=0.93 -> 37      below auto_approve 0.87: 15 of 43
```

DX-001 reproduced independently, on different input, in a different domain. The API whose name
answers "which of these should I not trust?" returns nothing while 15 of 43 sit under the bar.

Two more things nothing warns about:

```
glossary entries claimed by more than one schema field: 5
  'customer identifier' x5: ['cust','cust.cid','cust.dob','cust.addrs[].cty','cust.addrs[].geo']
  'Payment Method'      x5: ['pay','pay.mthd','pay.attempts','pay.attempts[].seq','pay.attempts[].ts']
  'Street Address Line One' x2   'Ordered Quantity' x2   'Tax Amount' x2
```

**One glossary term claimed by five fields is a signal, and it is not surfaced.** Three of the
five `customer identifier` claims are wrong — `cust.dob` is Date of Birth, `cust.addrs[].cty`
is City Name, and both terms exist in the glossary. The parent path is doing it: `cust.` is
worth +19.3 P@1 when the parent segment is context, and it is a magnet when the parent segment
is *itself a glossary term*. That is the documented strength producing the observed failure,
and I have not seen it named anywhere.

**Six of the sixteen duplicate claims are container nodes, not columns.** `cust`, `pay`,
`lines`, `cust.addrs`, `lines[].tax`, `pay.attempts` are records and arrays — they hold no
data and cannot carry a classification. The Avro parser emits them as matchable fields
alongside the leaves, and `lines -> Ordered Quantity 0.707` is what that looks like. There is
no leaf-only flag and no `is_container` on `SchemaField`.

### 00:18:31 · the number that should worry us most

```
Summary: 27/43 fields auto-approved (62.8%)
```

The README: "at the default threshold it fires on **~12% of fields** and is 95.3% precise."
On my 42-term glossary the same default fires on **62.8%**. Nothing in the run says the
threshold was calibrated somewhere else, on a 793-entry corpus, in two unrelated domains.
H-002 is not a theoretical hazard here — it is the default behaviour on the first realistic
glossary I pointed it at. A 42-entry glossary is a small candidate pool; the rank-normalised
score rises accordingly; the fixed bar does not move. A governance lead running this on a
departmental glossary auto-approves five times more than the documentation led them to expect,
and two of those auto-approvals are the downgraded PCI and PII columns from 00:23:09.

### What worked, and it is not a short list

- **The nested Avro parser is genuinely good.** 43 fields out of a four-level schema with unions,
  arrays, enums, maps and logical types, first try, no configuration. `cust.addrs[].geo.lat ->
  Latitude`, `cust.addrs[].subdiv -> STATE OR PROVINCE CODE 0.983`. Array boundaries are marked,
  unions unwrap, parent paths are preserved. This is the part that is hard and it is done.
- **Mixed casing and trailing whitespace were non-issues.** `CUSTOMER FULL NAME` matched at
  0.979; `'Email Address '` was stripped on load; the em dash in GL-031 survived intact.
- **The encoding matrix holds.** Identical table output under `cp437`, `cp850`, `cp1252`, piped
  to a non-TTY, exit 0 every time. NM-0001's fix is real. (Docstrings still are not: printing
  `nexus_matcher.ingest.__doc__` under `cp1252` still raises `UnicodeEncodeError` on `→`.
  DX-006, live.)
- **Determinism held.** Identical results across repeated runs and across processes.
- **Error messages, where they are right, are excellent.** "No parser found for extension .sql"
  is exactly what happened. The complaint is never about tone; it is about the two messages
  that describe a cause that was not there.

---

## What I would fix, in this order

1. **`load_dictionary` must use `_coerce_protection`.** The README path silently downgrades
   `Restricted - PCI` to `INTERNAL` and auto-approves it, and downgrades `Highly Confidential`
   below `Confidential`. One function call, already written and already correct thirty files
   away. Everything else on this page is ergonomics; this one mislabels a card number.
2. **Emit `protection_level` and `governance_raw` in the JSON and CSV outputs.** The stated use
   case is unachievable from either documented machine-readable surface.
3. **`CREATE OR REPLACE`, `TRANSIENT`, `EXTERNAL`** in `_CREATE_TABLE_RE`, and **keep column
   `COMMENT`** as `SchemaField.description`. One alternation and one capture; the second is
   worth more accuracy on DDL input than any model change.
4. **`match_schema` on a multi-table DDL must not silently match one table.** Use `parse_all`,
   or refuse and say why.
5. **Let the two user-facing surfaces reach `header_row`/`sheet`.** A `**options` passthrough on
   `load_dictionary` and a `--header-row` flag. Then make the error name it — a message that
   prints the header it actually found (as `load_entries` does) turns 11 minutes into 30
   seconds.
6. **Export `ColumnMapping` from the top level**, or stop naming it in errors and READMEs.
7. **Say why a decision was made.** A `reason` on `MatchResult` and in the JSON —
   `below_threshold`, `ambiguous_margin_0.063`, `auto` — so 0.906/REVIEW next to 0.904/AUTO
   is legible.
8. **Warn on the things that are silent:** rows that produced no definition and no id pattern,
   one glossary term claimed by N fields, and an auto-approve rate wildly outside the
   calibrated band. Each is a one-line warning against a silent wrong answer.

**Reproduction.** `C:\nmxf\work\` holds `orders.avsc`, `warehouse_ddl.sql`,
`make_glossary.py`, `probe1.py`–`probe9.py`, `run1.py`–`run3.py` and `falsify.py`, each named
for the timestamped step above. Nothing in this log is second-hand, and the one claim heavy
enough to carry a fix was attacked before it was written down.

**Lane note.** This file is documentation. It changes nothing in `src/`, `tests/` or
`scripts/`, and lands no gate. Items 1–4 above are the ones that need a failing test and a
museum entry, and each belongs to whoever owns those trees.
