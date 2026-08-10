# Competitive scan — 2026-08-10

Two sections, deliberately. Everything that is neither a gap we must close nor an advantage a
server-based product structurally cannot copy is noise, and is omitted.

Scope of what was actually read, so nothing below is inference from a product name:
[Collibra Unified Data Classification](https://productresources.collibra.com/docs/collibra/latest/Content/Catalog/DataClassification/UnifiedDataClassification/co_about-data-classification.htm)
and its
[auto accept/reject configuration](https://productresources.collibra.com/docs/collibra/latest/Content/Catalog/DataClassification/ta_configure-automatic-acceptance-rejection.htm);
[Alation's Lexicon/ALLIE glossary entries](https://docs.alation.com/en/latest/welcome/Glossary/index.html);
[Atlan's Asset-term link app](https://docs.atlan.com/product/capabilities/governance/glossary/how-tos/automate-term-asset-links);
[OpenMetadata Auto-Classification](https://docs.open-metadata.org/latest/how-to-guides/data-governance/classification/auto-classification)
and its [deployment requirements](https://docs.open-metadata.org/latest/quick-start/local-docker-deployment);
[DataHub's ingestion Classification module](https://docs.datahub.com/docs/metadata-ingestion/docs/dev_guides/classification),
[glossary term propagation](https://docs.datahub.com/docs/automations/glossary-term-propagation)
and [DataHub Cloud AI](https://datahub.com/products/ai-data-management/);
[acryl-datahub-classify on PyPI](https://pypi.org/project/acryl-datahub-classify/);
[Valentine on PyPI](https://pypi.org/project/valentine/) and its
[README](https://github.com/delftdata/valentine);
[Magneto (VLDB '25)](https://arxiv.org/abs/2412.08194);
[Matchmaker](https://arxiv.org/abs/2410.24105); [Schemora](https://arxiv.org/pdf/2507.14376).
Products I did not open are not discussed.

---

## 1. Table stakes we lack

Capabilities that shipped competitors treat as baseline and we do not have at all. Each is
stated with the source that establishes it as baseline, and with the observation from
`friction-log.md` that shows what its absence costs.

**A deterministic rule layer under the semantic one.** OpenMetadata's auto-classification runs
a *Column Name Scanner* — regex over names for emails, SSNs, bank accounts — before, and
independently of, any model. Collibra's data classes are built from rules over metadata and
over values. Both let a steward assert a fact and have it hold. We have no override mechanism
of any kind: no "any column matching `*_sk` is a surrogate key," no "`pan_last4` is always
PCI." The friction log's `CUST_SK -> customer identifier, 0.9328, AUTO_APPROVE` is a
one-character-away semantic collision that any rule layer would have caught and that no
threshold can fix, because it is *confident* and wrong. A pure-semantic system with no
deterministic floor is worse than `grep` on the cases `grep` gets right.

**Instance evidence, even as an option.** Collibra samples up to 1,000 rows and needs at least
6 values to classify a column. OpenMetadata runs NLP entity recognition over sample data.
DataHub's classifier took `sample_size: 100`. Valentine ships three instance-based matchers
alongside its schema-based ones. We read metadata and nothing else. For `col_17 VARCHAR(9)`
with no comment and no parent path, metadata-only has no signal at all, and this is not rare —
the README already concedes 87% of BIRD entries have empty descriptions. Section 2 argues that
*never requiring* instances is a genuine differentiator; that argument only survives if the
instance path exists as an opt-in for the users who can grant it. Right now the choice is
made for them.

**Propagation and cross-schema consistency.** DataHub propagates glossary terms down
column-level lineage and across sibling assets, so one adjudicated decision covers every
downstream copy. We classify each schema in isolation. The friction log measured the
consequence inside a *single* schema: `customer identifier` was claimed by five different
fields, three of them wrong, with no warning that one term had been claimed five times. Across
twelve tables carrying the same physical column, we produce twelve independent guesses and no
mechanism notices they disagree. This is the largest structural gap on the list, and it is not
an accuracy problem — the individual scores can all be right and the estate still be
inconsistent.

**A decision store and a feedback path.** Collibra's suggestions are accepted or rejected by a
steward, and the accept/reject thresholds are configuration a steward owns (the documented
example accepts at ≥75%, rejects at ≤49%). The assignment persists in the platform. Our
`MatchingSession` exposes `session_id`, `results` and `total_duration_ms`, and then the process
exits. There is no record of what a human decided, no way to re-run a schema and see only what
changed, and therefore no way for the second run to be cheaper than the first. Governance work
is iterative and we model it as a batch job.

**Reading the warehouse's own dialect.** Every platform ingests from Snowflake, BigQuery and
Databricks directly. We accept a file, which is fine — but the single most common way to
obtain a schema from Snowflake is `GET_DDL`, and its output begins `create or replace
TRANSIENT TABLE`, which our parser rejects with "No CREATE TABLE statement found." A file-based
tool that cannot read the file the source system emits has not made a scoping decision; it has
a bug. Same for the 17 column `COMMENT` clauses it then discards.

**Note on what is *not* on this list.** Nobody I read publishes an accuracy number, so
"competitive accuracy" is not table stakes — it is unmeasurable in this market. See §2.

---

## 2. Differentiators a library has and a platform structurally cannot

Structural means: the platform would have to stop being a platform. Not "hasn't built it yet."

**We can run where the data is not allowed to go.** This is the sharpest one, and it is the
inverse of a table-stakes gap. Collibra pulls up to 1,000 real rows into the Edge cache to
classify a column. OpenMetadata's entity-recognition arm requires sample data ingestion.
DataHub's classifier required sample values. Their accuracy is *built on* reading the data. A
metadata-only matcher reads a `.avsc` or a `GET_DDL` output and never touches a row — so it
runs against a schema in a data room, a vendor's proposed contract, a regulated extract nobody
may sample, or a table that does not exist yet. A catalog cannot follow us there, because
without the sample its classifier is reduced to the regex layer. Their strongest signal is
precisely the one we are structurally forbidden to want, and that asymmetry is permanent.

**We can run before the data lands.** OpenMetadata needs 6 GiB of RAM, 4 vCPUs, a server, a
Postgres or MySQL, an Elasticsearch and an Airflow to start. Every platform on this list
classifies assets that have already been ingested into it. A 22.6 MB wheel with the encoder
inside it and no network call on first run is a function, and a function fits where a
deployment does not: in a dbt pre-hook, in a PR check on a schema diff, in a CI job that fails
the build when a new column matches a RESTRICTED term and has no owner. Classification at
design time is a different product from classification at catalog time, and only one of the
two can be delivered by something you deploy.

**Our answer is pinnable, theirs is not.** `nexus-matcher==2.0.1` produces byte-identical
decisions today and in 2028; the friction log confirmed determinism across processes. A SaaS
classifier's model changes underneath the customer, deliberately and without a version they can
hold. When an auditor asks why a column was labelled INTERNAL in March, the platform customer
cannot re-run March. This is not a nicety in a regulated estate — it is the difference between
a defensible control and an anecdote. A vendor could version their model; they cannot give the
customer the artifact, which is the part that matters.

**We can hand over the calibration harness; they can only hand over the knob.** Collibra
exposes accept/reject thresholds. OpenMetadata exposes `confidence`, 0–100, default 80. None of
them can tell a customer what those numbers should be on that customer's estate, because
deriving them requires a labelled sample from the customer, and getting one means the customer
shipping labelled sensitive metadata to the vendor. A library runs on the customer's machine,
so it can ship the harness itself and let the number be derived locally. The friction log makes
the case that this is mandatory rather than nice: the documented ~12% auto-approve rate became
**62.8%** on the first realistic 42-term glossary, with the default threshold unchanged. A
threshold shipped as a constant is wrong for almost everyone; a threshold *derived on your
corpus, offline* is a thing only a library can offer.

**We can be a component of the platform.** A library composes into DataHub's ingestion, into an
Atlan workflow, into a dbt package. Platforms compete for the center of the diagram and
therefore cannot be a step inside a rival's diagram. The relevant precedent is on this list:
DataHub's built-in classifier `DataHubClassifier` was **removed** because it depended on
`acryl-datahub-classify`, now published with a deprecation notice and `Development Status :: 7
- Inactive`. The offline classification component in the largest open-source catalog is
abandoned, and the replacement is a Cloud AI feature. That is a vacancy in a place a library
can stand and a platform cannot.

### Does anyone publish honest accuracy numbers, or a "when not to use this" page?

**Accuracy: academia yes, vendors no, and it is not close.** The research line publishes freely
and precisely. Magneto reports MRR **0.866 ± 0.083** on the GDC benchmark and **0.939–0.971**
across fabricated Valentine datasets, alongside the cost of getting there — 545–589 s for the
LLM reranker against 11–33 s for its bipartite one, with ISResMat and Unicorn failing to
complete GDC beyond 100 columns at all. Valentine exists specifically to make such comparisons
reproducible. On the vendor side I found confidence *controls* — Collibra's 75/49 example,
OpenMetadata's default 80 — and **not one precision or recall figure against any named corpus,
from any of the five products.**

None of those numbers is comparable to our 0.581 P@1, and saying so is the point. Magneto's own
results contain the demonstration: the classical matchers score near-perfect MRR (~1.0) on
human-curated Valentine datasets and are beaten badly on GDC. Same algorithms, same metric,
opposite verdicts, because the corpus changed. That is H-002 published in a venue. Any number
without a named, downloadable corpus is decoration.

**"When not to use this": nobody.** Valentine's README has no limitations section. The Collibra,
OpenMetadata, DataHub and Atlan docs I read have none. The established convention for this kind
of disclosure — intended use, out-of-scope use, disaggregated metrics — is the model card, and
it has been adopted for *models*, not for data-governance products. Our README's Limitations
section, which names the machine, the two domains, the English-only measurement, the 793-entry
corpus and the corpus-specific threshold, appears to have no counterpart in this market.

**Is that a moat or a liability? On its own it is a liability, and it is worth converting.**

Honesty is free to copy and asymmetric in the copying. A competitor can publish a number next
week, choose the corpus that flatters them, and beat 0.581 in every RFP grid — and per the
Magneto evidence, the corpus choice alone is worth the entire gap. Meanwhile the buyer who
reads our Limitations page learns that roughly half of an abbreviation-heavy schema will not
have the right answer at rank 1, and the competing page tells them nothing at all. In a
side-by-side, disclosed weakness loses to undisclosed weakness every time. That is the liability,
and it is real, and pretending otherwise is how a good practice gets abandoned after it costs a
deal.

It converts into a moat at exactly one point: when the artifact stops being a *claim* and
becomes a *procedure the buyer can run on their own data*. "Trust our 0.581" is a liability.
"Here is the harness; label 200 of your own columns, run it offline this afternoon, and get
*your* number — for us and, since the input is just a labelled CSV, for anyone else you are
evaluating" is a moat, for the structural reason in this section: a platform cannot offer it
without the buyer exporting labelled sensitive metadata to the vendor. We turn the honesty into
a tool, and the tool is the thing that cannot be copied by a product that lives on a server.

That reframes the "when not to use this" page too. It is currently a virtue signal. It should be
a *qualification filter* with a decision procedure attached — no parent path and no description
on your fields means abbreviation-only matching, which is the measured worst case; a 40-term
glossary means the auto-approve default is calibrated for someone else's corpus and will fire
five times more often than documented. Both of those are things a reader can check about their
own input in a minute, and both send some readers away. Sending the wrong buyer away before they
run it is cheaper than being wrong for them afterwards, and — given that library selection is
increasingly done by an agent reading the package page rather than a human skimming it — a
machine-readable statement of when the tool does not apply is a distribution feature, not
modesty. Which makes DX-007, the PyPI page failing to render for a programmatic reader, a
commercial defect rather than a cosmetic one.
