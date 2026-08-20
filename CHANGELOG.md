# Changelog

All notable changes to NexusMatcher are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every performance or accuracy number below names the artifact in `benchmarks/results/`
that it came from. Numbers without an artifact are not stated. A one-off measurement taken
while making a change — a profile, an RSS reading, the length of a refusal message — names
its method instead: it is evidence that a defect was real, not a claim about how well this
library matches.

---

## [Unreleased]

Nothing yet. 2.1.0 has not been published, so work lands under it until it is.

---

## [2.1.0] - 2026-08-10

**Prepared, not published.** PyPI has never served a 2.1.0. `dist/` holds a locally built
`nexus_matcher-2.1.0-py3-none-any.whl`, which is what led one review to open it, find the
defects below still present in it, and conclude the release had shipped with them. It is a
build artifact, not a release; everything in this section is unreleased work.

### Security

- **A comment naming the organisation and business unit whose data model the default
  domain hierarchy was sketched from has been removed.** It sat above
  `DEFAULT_HIERARCHY_DATA` in `domain/services/domain_hierarchy.py` and shipped **inside
  the 2.0.0 and 2.0.1 wheels**. Two further copies were in
  `docs/modules/domain_hierarchy.md` and four in an integration test. `DEFAULT_HIERARCHY_DATA`
  itself is unchanged — the names in it are ordinary industry vocabulary — but it is now
  documented for what it is: an illustrative default that callers should replace via
  `DomainHierarchy.from_dict()`.

  **2.0.0 and 2.0.1 have been DELETED from PyPI, not yanked**, so those files are gone from
  the index rather than merely hidden from resolvers. An earlier revision of this entry said
  they remained available "unless yanked" and advised pinning 2.1.0 or later; it was written
  before the deletion, and both halves of it are now wrong — nothing to pin away from, and
  nothing published to pin to.

  **1.0.0 is still on PyPI and still carries the string**: three occurrences in the wheel,
  nineteen in the sdist. One of the three is in `cli/main.py` and appears in no later
  wheel — an internal repository URL that `nexus-matcher info` printed to whoever ran it.
  2.0.1 retracted that URL, and
  `tests/unit/presentation/test_cli_regressions.py::test_info_points_at_the_real_repository`
  is what keeps it retracted. Deleting 1.0.0 as well is a decision that has not been taken,
  so anyone holding it in a lock file or a vendored mirror still has the string, and the
  sdist is the worse half of the two.

- **Two gates now enforce this mechanically**, because the rule previously existed only
  as an intention. `tests/meta/test_no_confidential_terms.py` scans every tracked file on
  each commit, and `scripts/release_preflight.py` scans the built wheel before upload —
  the tree and the artifact are different scopes, and it was the artifact that mattered.
  The blocklist is stored as salted digests rather than words, so the gate can recognise
  a term the repository never contains; `scripts/add_confidential_term.py` extends it
  without the term reaching your shell history. Recorded as **NM-0029**.

- **Oversized request bodies are refused before they are buffered or parsed.**
  `MatchService` applied `max_fields` only after FastAPI had read the whole body and built
  every `FieldSpec`, so the parse sat outside admission control and outside the deadline.
  Measured against real uvicorn, watching RSS on the server process: a 198 MiB body cost
  +808 MiB before its 413, and eight concurrent ones cost +3.5 GiB with **zero** 503s —
  `BoundedWorkPool` never saw them, so on a memory-capped container the outcome is not a
  413, it is an OOMKill with nothing shedding first. A pure-ASGI `BodySizeLimitMiddleware`
  counts bytes as they arrive and refuses the moment the cap is crossed, with or without
  a `Content-Length`; registered last so it is outermost, which is the only position where
  it gets `receive` before the body has been streamed through. The cap is **derived** from
  `max_batch_fields` and `FieldSpec`'s own `max_length`s (×4 for UTF-8) rather than typed
  as a literal, so raising the field cap raises the byte cap with it — 9.42 MiB at the
  shipped 250-field default, against 2.36 MB for the largest batch a client can legitimately
  send. `NEXUS_API_MAX_BODY_BYTES` overrides it and is **refused at startup** if it falls
  below what the field caps admit, because a 413 that contradicts the schema a client
  generated from is worse than no cap. Callers already receiving 413 now receive it earlier,
  with `details.limit_bytes` and `details.observed_bytes`.

  **A refused body within twice the cap is now read and discarded before the 413 is sent**,
  bounded at 2 × `max_bytes` (19.7 MB at the default) and 2.0 s of wall time, whichever
  comes first. Outside those bounds nothing is read, exactly as before. This costs read
  bandwidth and up to two seconds of connection time per refusal, and it buys the 413
  actually arriving: refusing a body the client is still writing closes the socket with
  bytes unread, which is an RST, and an RST discards the response already sent — the caller
  sees a transport error instead of the readable 413 telling it to re-chunk.

  Measured against the Java client's own transport on a pooled connection, a mis-chunked
  batch 1.06× over the cap: **34/40 readable before, 120/120 after**. A partial drain was
  tried and is *worse than doing nothing* (375/400 at 256 KiB against 392/400 draining
  nothing) — the client refills the kernel buffer faster than a bounded read empties it, so
  the bytes are still unread at close and the bandwidth bought nothing. Only reaching the
  end of the body removes the reset, so the rule is drain fully or not at all, decided
  before the first read from the declared length.

  Memory is unaffected, which is the whole point: draining discards each chunk, so retained
  bytes stay O(chunk). Sustained 15 rounds × 16 concurrent × 18.5 MiB — 4.44 GiB of body —
  moved server RSS 175 → 178 MiB, against the +3.5 GiB this middleware was written to stop.

  **Residual, stated because it is easy to overstate this fix:** a body beyond twice the cap
  is still refused unread and still loses its 413 about 15% of the time on a pooled
  connection. The fix moves the threshold so realistic mis-chunked batches fall inside it;
  it does not make an arbitrarily large body safe to send. Treat a transport failure on a
  large request body as a probable 413 and re-chunk rather than retry.

  **That 413 now carries `X-Request-ID` and `X-Response-Time-Ms`, and the request is logged
  under the id the caller is holding.** Being outermost is what makes the byte cap work, and
  it is also what put this response outside `request_id_middleware`, which stamps both
  headers on everything it wraps — so the 413 previously carried neither, missing for exactly
  the caller most likely to need it: the one sending oversized batches and trying to work out
  which of them was refused. Both headers are stamped by the middleware itself now, and every
  status this service answers was driven on a live app to check it: 200, 201, 404, 405, 413,
  422, 500, 503 and 504 all carry both. A caller-supplied `X-Request-ID` wins over a minted
  one, so `X-Request-ID: caller-abc` is the id the 413 comes back under.

  The streaming path was previously worse than a missing header, and that half a caller could
  not have worked around. Driving a chunked body past the cap, the client got a **413 with no
  id** while the server wrote one `http_request` line reading **`status_code: 400`** under a
  correlation id the client had never seen — FastAPI turning the disconnect into its own
  response, which `guarded_send` then suppressed. Two records of one request, agreeing on
  neither the status nor the id, which is not a trail anyone can join. The id is now minted
  here with the same recipe `app.py` uses and **injected into the ASGI scope** before the app
  is called, so `request_id_middleware` adopts it rather than minting a second one. Measured
  on a chunked refusal today: the 413 the client holds and the server's `http_request` line
  carry the same id. The statuses still differ — that log line is the disconnect and still
  reads 400 — so join the two records on the id, not on the status.

- **The published OpenAPI no longer advertises authentication that nothing implements.**
  The FastAPI `description` — which ships in `/openapi.json` and therefore into every
  generated client — offered an `X-API-Key` header enabled by `NEXUS_API_KEY`. No code has
  ever read that variable: with it set, `POST /api/v1/match` answered 200 and a real
  protection class to a request carrying no header, and `components.securitySchemes` was
  absent. The module header claimed rate limiting on the same evidence, and `RateLimitError`
  is raised nowhere. Both claims are retracted and replaced by an explicit sentence that the
  service ships unauthenticated and belongs behind the operator's own gateway. Deleting the
  paragraph was half the fix; stating the position is the half that stops it being invented
  again, which is the job `docs/API_REFERENCE.md`'s "Not implemented" table already does.

### Changed — BREAKING

- **`tiers_most_open_first` is now enforced, where it was previously read by no code.** The
  key appeared in the shipped example vocabulary and in nothing else — an adopter copying
  the pack inherited a declaration that did nothing. It now validates the caller's file
  against **itself**: every tier a class derives must appear in the list, the declared
  `open_classification` must appear, no tier may appear twice (compared after
  normalisation, so `"Sealed "` and `"sealed"` are one rung), and a bare string is refused
  rather than read one letter per rung. A vocabulary that omits a tier its own classes use
  now fails to load, naming both sides. This is validation of the caller's own declared
  ordering, not a taxonomy this library supplies — it still supplies none.

  Two deliberate non-rules: a declared tier no class currently uses stays legal, because a
  policy ladder may have unreached rungs; and the `UNCLASSIFIED` sentinel is exempt, so
  this library's own default cannot fail this library's own check.

  Fixing this surfaced a second defect the review had not found: `tiers_most_open_first`
  was missing from the reserved-key set, so a `{code: {...}}`-shaped document that also
  declared the ordering was refused outright with *"no protection classes found"* — a
  message pointing at the classes, which were fine.

- **`MatchRequest` no longer rejects unknown top-level keys.** `extra="forbid"` on the
  envelope made v1 non-extensible in both directions: a caller sending a key a newer
  server understands got a 422 from an older one. `FieldSpec` stays strict — the typo
  argument holds where the field names are, and relaxing it has a documented accuracy
  cost. A misspelled `fields` is still a `missing` error, and an unknown key *inside* a
  field is still a 422.

- **`ScoreBreakdown.semantic_score` is renamed to `fused_retrieval_score`.** It never was
  a semantic similarity. It is the min-max-normalised *fused retrieval* score, rescaled
  over the candidates retrieved for one field, so it is rank-relative: the rank-1
  candidate lands at or just above `fusion_alpha` for essentially every field, which is
  why the recurring value was 0.9 — **0.9 is `fusion_alpha`**. Move `fusion_alpha` to 0.5
  and the rank-1 floor moves to ~0.45; a cosine would not move at all. Telling an auditor
  "semantic score 0.9" claimed 90% similarity and delivered "ranked first among N
  candidates". `semantic_score` still works for both reading and construction and emits a
  `DeprecationWarning`; **it is removed in 3.0.**
- **`MatchingSession.get_low_confidence_fields(threshold=...)` now defaults to `None`**,
  meaning "was not auto-approved", and **raises `ValueError` for a threshold at or below
  the structural floor** instead of returning an empty list. Callers relying on the old
  `0.6` default were receiving `[]` unconditionally — see NM-0027.
- **`nexus-matcher match -f json` changes shape.** The `dictionary_entry` object gains
  `protection_level`, `definition` and `domain`; `scores` gains `edit_distance` and
  `domain`, and its `semantic` key is renamed **`fused_retrieval`** to match the domain
  model above — a key named "semantic" carrying a rank-relative number is the same claim
  the rename exists to stop making. Each match record also gains a `weights` object.
  Object keys are now emitted in sorted order, so the field paths are listed
  alphabetically rather than in schema order, and the document ends with a newline.
  Finally, when `-f json` or `-f csv` is used **without** `--output`, progress and the
  summary line move to **stderr**, because stdout is carrying the payload — see NM-0026.
  A script reading `["scores"]["semantic"]`, or relying on schema ordering, needs updating.
- **A rejected RUNNER-UP now carries its protection class; only a rejected rank 1 carries
  none.** This is the only change in this release that alters values already on the wire,
  so plainly: **a client treating `governance: null` as "do not apply" will now see
  populated `code`, `name` and `classification` at ranks 2 and below.**
  `MatchResult.__post_init__` cleared `governance` for any `REJECT`, but `REJECT` is a
  per-*candidate* verdict — every rank is compared against `review_threshold` — while the
  justification for clearing it is a per-*field* one: a novel field must not arrive
  carrying the class of the least-bad candidate. A field inherits from rank 1 only.
  Measured over the 26-field Gravel Bay Ferry Authority pack at `top_k=5`, 66 of 104
  runner-up candidates came back with no class although the indexed entry carried a real
  code, 16 of them direct identifiers; only 13 of the nulls were genuinely uncoded, and
  `decision` could not tell the two apart because both read `REJECT`. The rule is now
  `rank == 1 and decision == REJECT`, and `GovernanceView`'s published description states
  both cases, because the string `"REJECT"` had appeared nowhere in `/openapi.json`.

  The rank-1 guard is **latent at the shipped defaults**: `review_threshold` is 0.50 and a
  rank-1 confidence cannot fall below `semantic_weight × fusion_alpha` = 0.63, so no top
  match can be rejected without moving one of those two numbers. It stays because that
  floor is a consequence of tunable settings rather than a law — it fires for a caller who
  raises the threshold past it — and
  `test_the_rank_one_guard_is_latent_at_the_shipped_thresholds` pins the relation, so the
  day a default moves the guard out of latency is a day the suite says so.
  `examples/governance/_pack.py` carried its own copy of the old unqualified strip and was
  updated in lockstep; had it not been, the shipped audit artifact would disagree with the
  service it exists to audit.
- **404, 405 and both health 503s move from `{"detail": …}` to the `{"error": {…}}`
  envelope. Anything parsing `detail` on those statuses breaks.** `errors.py` had said
  since it was written that raising `HTTPException` "would give the same service two error
  shapes, which is the sort of thing a client library ends up handling with a string test",
  and that is exactly what shipped: a census against a live app found every `/api/v1/*`
  failure in one shape and those four in the other, because 404 and 405 are raised by the
  router and the health probes raise `HTTPException` on purpose. One handler registered on
  `starlette.exceptions.HTTPException` now renders them all. It forwards `headers`, so the
  405 still carries `Allow: POST` — the obvious `JSONResponse(status_code, content)` drops
  it, which would have traded a client-library annoyance for a protocol defect. The 422
  envelope is unchanged.
- **CORS is closed by default. Browser callers break; a server-to-server caller sends no
  `Origin` and is unaffected.** It previously ran with `allow_origins=["*"]` and
  `allow_credentials=True` behind a `# Configure in production` comment, which is not a
  configuration — it is a note asking somebody to edit the source of an installed package.
  Measured: a preflight from any origin came back with that origin reflected and
  `access-control-allow-credentials: true`, so any page anywhere could read the response of
  an endpoint that asks for no credentials, and could `POST /api/v1/feedback`, which fsyncs
  a reviewer verdict into the audit trail. `NEXUS_API_CORS_ORIGINS` (comma-separated) names
  the origins that may use a browser; empty — the default — mounts no `CORSMiddleware` at
  all rather than mounting one that answers nothing. `NEXUS_API_CORS_ALLOW_CREDENTIALS=true`
  together with `*` is **refused at startup**, because Starlette reflects the requesting
  origin instead of sending `*` when credentials are on, so "allow every origin" silently
  becomes "allow *this* origin, with cookies" — a policy nobody can read off the settings
  that produced it. `allow_methods` is now `GET, OPTIONS, POST` instead of `*`.
- **`/health/ready` answers 503 when no dictionary is configured, and a
  health-and-introspection deployment must now set `NEXUS_API_MATCHING_OPTIONAL=true`.**
  The `matcher` component was registered only when a dictionary loaded, and `check_ready()`
  is `all()` over what is registered — so a deployment whose `NEXUS_API_DICTIONARY` was
  absent, empty, or misspelled registered nothing, answered `/health/ready` 200, and 503'd
  every `POST /api/v1/match`. A *broken* dictionary went red correctly; a *missing* one did
  not, and missing is what a Helm value that fails to resolve produces. `matcher` is now
  registered unconditionally, `all(())` no longer counts as ready, and the 503 body carries
  the component map instead of the bare string "Service not ready", which told an operator
  nothing about which component was red. The opt-out is inverted from the obvious
  `NEXUS_API_REQUIRE_MATCHER` on purpose: a knob whose default is the unsafe value protects
  only the deployments whose operator remembered to set it, which are not the misconfigured
  ones.

  None of this would have been visible to any deployment this repository documents:
  `docker/Dockerfile`, `docker/docker-compose.yml` and both probes in `docs/DEPLOYMENT.md`
  all targeted `/health`, which returns 200 with `status="degraded"` even when the
  dictionary failed to load, so `curl -f` passed. They now target `/health/ready` and
  `/health/live`.
- **A vocabulary file with `null`, or any non-string, in `code` or `classification` now
  refuses to load.** `str()` ran before the emptiness test and `str(None)` is `"None"`, so
  neither guard fired. `"code": null` produced the code `'None'`, which then matched every
  glossary cell spelling "none" or "N/A-as-none" — a field asserting it has *no* code
  inherited that class, and `problems_with()` returned `[]`. `"classification": null`
  derived the literal tier `'None'`, which is a `str` and therefore shipped over the wire as
  a tier nobody declared; worse, honest rows stating their real tier then read as
  self-contradictions, so `governance_strict` refused the whole load blaming the glossary
  for a defect in the catalog. `GovernanceVocabulary(...)` refuses the same two fields
  directly, so the invariant does not depend on going through `from_json`.

  **`null` stays meaningful everywhere it was documented as meaningful** — `name`,
  `enhancement`, `open_classification`, and an alias target, which is how a caller declares
  a token droppable noise. The shipped ferry pack, with `"enhancement": null` on five of its
  nine classes and `"n/a": null` and `"tbc": null` in its alias map, loads unchanged; a
  single "reject null" rule applied to the obvious field list would have rejected this
  repository's own example.
- **A vocabulary declaring one alias token twice, pointing at two different things, now
  refuses to load — NM-0031.** Duplicate *codes* have raised since the class was written;
  duplicate *aliases* were overwritten silently at both build sites, and class-level aliases
  always beat the top-level map because the map was seeded first — so the file's most
  explicit statement was the one that lost. A restricted class and an open class both
  claiming one legacy spelling loaded the restricted rows as the **open** tier, with
  `problems_with() == []` and a strict load not refused: NM-0005's harm, a field losing the
  class it should have inherited, reached through the catalog instead of through the row. A
  `null` "drop this token" declaration was promoted into a real class the same way, which
  this module's own docstring promised could not happen unnoticed. Exact redeclaration of
  the same mapping still loads — a rule that could not tell a restatement from a conflict
  gets worked around by deleting the restatement that documented the intent.
- **A class-level `aliases` given as a bare string now raises.** `"aliases": "LEGACY-METER"`
  is iterable, so the loop declared `L`, `E`, `G`, `A`, `C`, `Y` and `M` as aliases of that
  class. Single-character aliases match nothing a glossary carries, so the spelling the
  caller was trying to declare stayed unknown and every row using it was refused — for a
  reason that appeared nowhere in their file.
- **Protection codes and tiers are Unicode-normalised (NFC) before comparison, so an NFC
  and an NFD spelling are one class rather than two.** `_norm_code` strips
  non-alphanumerics and a combining accent is not alphanumeric, so an accented code had two
  normalised forms — the composed one an editor types, the decomposed one macOS and several
  exporters emit — that render identically everywhere a human reads them. A catalog
  declaring both loaded as two classes with two tiers and showed the same word twice in
  `codes`, and which tier a row inherited depended on which byte sequence its exporter
  chose. It bites harder on tiers, because `casefold()` does not normalise: a decomposed
  accent in a row against a composed one in the catalog read as a row contradicting itself,
  which at the default `governance_strict` refuses the entire glossary over two words that
  look the same.
- **`load_entries` and `build_index` now refuse a source that has a protection-code column
  when no `governance=` vocabulary is passed.** The silence was circular, which is why
  neither layer below could catch it: a code is attached only when a vocabulary is
  configured, and every consumer refuses codes it cannot resolve — so with no vocabulary
  there were no codes, nothing to refuse, and a glossary whose header plainly said
  `protection_class` produced entries carrying nothing at all, indistinguishable downstream
  from a glossary that declares no classes. `governance_strict=False` is the opt-out and
  reads the column as plain metadata. The HTTP app had this check and paid a second full
  read of the glossary to run it; in the loader the column mapping is already built, so it
  is a dict lookup.
- **A glossary column named `PII` is read as the personal-information FLAG, not as the
  free-text classification tier, so it no longer populates
  `DictionaryEntry.protection_level` and moves to `source_metadata`.** `"pii"` came over
  with the rest of `COLUMN_ALIASES["protection_level"]`, where it was harmless — as a tier
  alias it was just another place somebody might write a tier. Under the derivation
  invariant the same membership asserts that a row under a `PII` column claims its *tier* is
  "Yes", so the header `term,definition,governance_code,PII` refused every coded row with a
  contradiction the row had never stated, and whether the load survived depended on whether
  some other tier column happened to be present to mask it. The silent half is why it had to
  move rather than simply be dropped: when an earlier tier alias *was* present, `PII` lost
  the tier race and was then read by nothing, so a row claiming personal information against
  a class declaring `personal_information: false` reported no disagreement at all — while
  `personal_data`, `is_personal_data` and `contains_personal_data` all reported it, and
  `docs/GOVERNANCE.md` rule 2 promises that disagreement is reported. `is_pii` is recognised
  alongside it.

### Added

- **`governance.enhancement` now crosses the wire**, appended as the sixth key of the
  governance object so existing key order is unchanged. It is the caller's own instruction
  for how to protect a field — the thing a consumer most needs after learning the field is
  sensitive — and it was resolved internally and then dropped before serialisation. `null`
  for a class that declares none, which five of the nine example classes do; read with
  `getattr`, not the drift path, because routing it through drift would make this repo's
  own example pack a 500.

- **Responses carry a `vocabulary` block** on both match routes:
  `{"openClassification": ..., "tiersMostOpenFirst": [...]}`. Without it a
  `"governance": null` is uninterpretable — a consumer cannot tell which tier an uncoded
  field sits at, and a Java service calling this API has no second copy of the vocabulary
  file to consult. Both values are the caller's own, echoed back.

- **Governance inheritance over HTTP.** `create_app()` registers `POST /api/v1/match`,
  `POST /api/v1/match/batch` (the same service and projection at a higher field cap) and
  `POST /api/v1/feedback`. The response is one entry per input field, keyed by the caller's
  own `path` and in the order sent, and every candidate carries the protection class its
  glossary entry confers — resolved through the caller's own vocabulary, which this library
  still does not define and does not type as a closed set.

  Both shipped entry points call `create_app()` with no arguments, so the wiring is
  environment-driven: `NEXUS_API_DICTIONARY` and `NEXUS_API_GOVERNANCE` are required for
  matching, `NEXUS_API_FEEDBACK_PATH` for recording. `NEXUS_API_MATCHING_CONFIG`,
  `NEXUS_API_DEADLINE_SECONDS`, `NEXUS_API_MAX_WORKERS`, `NEXUS_API_MAX_QUEUED`,
  `NEXUS_API_MAX_FIELDS`, `NEXUS_API_MAX_BATCH_FIELDS` and `NEXUS_API_MAX_BODY_BYTES` tune
  the rest; `NEXUS_API_CORS_ORIGINS`, `NEXUS_API_CORS_ALLOW_CREDENTIALS` and
  `NEXUS_API_MATCHING_OPTIONAL` are documented under Changed — BREAKING above, because each
  of them changes a default. The wire contract is in
  [docs/GOVERNANCE.md](docs/GOVERNANCE.md#matching-over-http); note that the field keys are
  `name`/`path`/`doc`/`type` and **not** the `flattenedName`/`dataType` spellings used by
  `examples/governance/fields.json`, which `extra="forbid"` turns into a 422.

- **A typed error DTO and a `decision` enum in the published OpenAPI.** A regenerated client
  gains both types — source-breaking for the generated client, not for the wire.
  `ErrorResponse` sat unused in `app.py` with `error: dict[str, Any]`, which renders as
  `{"type": "object", "additionalProperties": true}`, so attaching it unchanged would have
  bought a generated client nothing; it now carries a typed `ErrorDetail` of `code`,
  `message` and `details`, and is published on 413, 422, 500, 503 and 504 for both match
  routes, on 422, 500 and 503 for `/api/v1/feedback`, and on the **503 of `/health/ready` and
  `/health/startup`** — **15 published failures across 5 routes**, counted off a live
  `/openapi.json`, every one of them `ErrorResponse`. One service, one spec, so a generated
  client does not get a typed DTO for every way a match can fail and a bare `Map` for every
  way a verdict can. 500 was absent from those tables entirely, although three exception
  classes return it on the match routes and an `OSError` on the feedback append
  returns it there, and it is the status a client is least able to guess. `details` stays a
  free-form map
  on purpose: its keys legitimately vary by failure — `limit`, `violations`,
  `deadline_seconds`, `duplicate_paths` — so pinning them would either be a lie or force
  every new failure mode to change the published schema. `MatchCandidateView.decision` is
  typed as the library's own `MatchDecision`, so the published
  `["AUTO_APPROVE", "REVIEW", "REJECT"]` and the value the endpoint emits are the same
  object by construction rather than a second copy of the list. `explain` becomes
  `ExplainView` with open `scores` and `weights` maps rather than five named fields, because
  the verifier it mirrors is written to survive a sixth weighted signal this schema would
  otherwise contradict.

  The two health probes are the reason the count moved from three routes to five. Both raise
  their 503 through `HTTPException` and neither declared it, so a client generated from the
  spec saw a readiness probe and a startup probe that **could only succeed** — for
  `/health/ready` that is the answer the route exists to give, and for `/health/startup` it
  is the only answer it has before the lifespan finishes. Each now publishes it with a
  description that separates the two: readiness carries `details.components` naming the
  component that is red, the startup probe has no map to carry because nothing has reported
  in yet, and both render through the same handler as every other failure, so they are the
  same DTO rather than a second shape. `/health` and `/health/live` still publish no failure
  at all, and that is correct rather than an omission — `/health` answers 200 with
  `status: "degraded"` when a component is red, and `/health/live` returns a constant, which
  is exactly why `curl -f /health` passed on a deployment whose dictionary had not loaded.

- **`examples/governance/serve.sh` and `serve.ps1`** — the three exports and the start
  command, over the example pack, so the endpoint can be reached in one line.

- **`ScoreBreakdown.absolute_cosine`** — the raw, un-normalised dense-retrieval score for
  the candidate. Under the shipped wiring this is the actual cosine similarity, and it is
  the only score in the breakdown that is comparable *across* fields. It was computed on
  every match and discarded, so "how similar were they really?" had no answer anywhere in
  the API.
- **`MatchingConfig.minimum_achievable_confidence`** and
  **`NexusMatcher.minimum_achievable_confidence`** — the structural floor of
  `final_confidence`, `semantic_weight * fusion_alpha` (**0.63** as shipped), or `None`
  when a reranker makes the bound unsound. `MatchingSession` carries the value that
  produced it. The floor was folklore; nothing computed it, so no threshold could be
  checked against it.
- **`GovernanceVocabulary` and `ProtectionClass`** — a caller-supplied controlled
  vocabulary of protection codes, loaded from a JSON file the caller owns. **This library
  ships no taxonomy**: one organisation's codes baked into a library are useless to every
  other organisation. `DictionaryEntry` gains `governance_code`, and `MatchResult` gains
  `governance_id` and `governance` — populated on every candidate, not only rank 1,
  because a reviewer choosing between rank 1 and rank 2 usually decides on the class.
  `ingest.load_entries(governance=…)` validates every row against the vocabulary: a code
  the vocabulary does not define is rejected rather than stored, and a row whose stated
  tier contradicts its own code **refuses the load** — the tier is derived from the code,
  never read from the row (**NM-0028**). A `REJECT` at **rank 1** confers no class, so a
  novel field can never inherit the classification of a top candidate the matcher rejected;
  a rejected runner-up keeps its class, because nothing inherits from rank 2 and the class
  is what lets a reviewer see that rank 1 is a direct identifier and rank 2 is not. See
  Changed — BREAKING above for what that qualifier corrected.

### Fixed

- **NM-0033 — `from_config(governance=…)` accepted a controlled vocabulary and
  `load_dictionary` never applied it, so every indexed entry carried no protection code and
  every match came back with `governance: null` — indistinguishable from a glossary that
  declares no classes.** The `DictionaryLoader` port hands back finished `DictionaryEntry`
  objects and `ColumnMapping` has no field for a protection-code column, so neither shipped
  loader ever read one; the vocabulary was consulted only at match time, where there was no
  stored code left to resolve. Measured on `examples/governance/glossary.csv` against the
  pack's own vocabulary: 30 entries indexed, **0** carrying a code before the fix and **27**
  after — the remaining three rows declare none. The library's own HTTP app had already
  worked around it by calling `ingest.load_entries` and the private `_index_dictionary`, and
  the example pack printed a wiring-defect banner and rescued the class with a caller-side
  join; both descriptions were accurate and neither was a gate. `load_dictionary` now
  attaches the vocabulary's canonical code after the loader runs, refuses a self-contradicting
  row with the same message `load_entries` gives for the same file, and refuses a
  protection-code column it has no vocabulary to interpret. `governance_strict=False` is the
  documented opt-out, and `tests/museum/NM-0033/test_nm_0033.py` pins the gate.

- **NM-0032 — `load_entries` forwarded any unrecognised keyword to the reader, which
  discarded it in silence, so a misspelled option loaded a different glossary than the one
  asked for.** The reader's signature is `**kwargs` and it keeps only the four options it
  knows, so everything else went nowhere and nothing said so. `sheet_name=` is the pandas
  spelling and the obvious thing to type; the option this loader takes is `sheet=`.
  Measured on a two-sheet workbook: `load_entries(book.xlsx, sheet_name="Approved")`
  returned the **Retired** sheet's rows — a glossary of retired terms indexed, matched
  against and inherited from, under a load report that said the load was healthy. It
  needed no new option to reach and had been there since the reader was written. Every
  ingest test passed the options it meant, so no test ever asked what happens to one that
  is misspelled. `load_entries` now refuses a keyword it does not understand and names the
  ones that exist; `read_source` underneath stays permissive, and
  `tests/museum/NM-0032/test_nm_0032.py` pins that split so the refusal cannot drift down
  a layer.

- **NM-0030 — `sync()` stripped the protection code off every entry and reported the rows
  unchanged.** `GlossaryIndex` remembered the `provider` and nothing else, and `sync`'s own
  docstring example is `sync(index, "glossary.xlsx")` with no `governance=`, so the refresh
  re-read the source with no vocabulary — and the refresh loop added specifically to stop
  stale governance then replaced every entry object with the uncoded one, making the loss
  total rather than partial. Measured on a byte-identical file: 27 coded entries of 30
  became 0, and the report said `=30 unchanged`. Worse than the loss, `sync` **bypassed the
  refusal gate**: a row whose stated tier contradicts its own code raised from
  `load_entries` and loaded silently through `sync`, so NM-0028's invariant — the thing this
  library exists to enforce — was unenforced on the module's own documented refresh path.

  `GlossaryIndex` now stores the whole load-kwargs bundle (vocabulary, `governance_strict`,
  `columns`, `id_prefix`, `sheet`, `delimiter`, `encoding`, `header_row`) and `sync` reuses
  it unless the caller overrides, which is the pattern `provider` already used. The
  *resolved* vocabulary is remembered, not the path: re-reading the catalog underneath a
  sync would let an edit to that JSON reclassify half an index under a report saying
  "unchanged", which is the same silent shape. `SyncReport` gains **`governance_changed`**,
  the ids whose code appeared, disappeared or moved, and the summary line reports it —
  governance never reaches `to_searchable_text()`, so a reclassified row is genuinely
  unchanged as far as embedding work goes and was invisible in the one line most callers
  print. A steward blanking a code cell is still a legitimate edit and is reported, not
  refused. It escaped because 24 tests called `build_index` and not one of them passed
  `governance=`, so every sync test in the suite ran over a glossary with no codes to lose.

- **NM-0027 — the review queue was empty because the bar was under the floor.**
  `MatchingSession.get_low_confidence_fields()` defaulted to a threshold of `0.6`, and
  `final_confidence` has a structural floor of **0.63** (`semantic_weight` 0.70 ×
  `fusion_alpha` 0.90, because the fused retrieval score is min-max normalised per field
  so the rank-1 candidate always lands at or above `fusion_alpha`). The one API whose name
  answers "which of these should I not trust?" therefore returned an empty list on every
  schema ever matched. Measured on a 6-field schema: default → 0 flagged, 0.87 → 6
  flagged, actual top-1 confidences 0.730–0.755, six of six below the auto-approve bar. A
  governance lead was told there was nothing to review on a schema where nothing was
  trustworthy — the same class as NM-0005. The default is now "was not auto-approved",
  which also catches the confident-but-ambiguous near-tie a numeric threshold clears; a
  field that matched *nothing* is now flagged instead of skipped; and the floor is
  exposed so the next threshold can be checked against a number.

- **NM-0025 — the documented machine-readable output dropped the governance payload.**
  `nexus-matcher match -f json` emitted the dictionary entry as
  `{id, business_name, logical_name, data_type}`. No `protection_level`. The stated use
  case of this library is that a matched field inherits that entry's classification, and
  the one field that use case rests on was absent from the only interface a script can
  consume. The `scores` block carried three of the five components and no weights, so the
  emitted numbers could not reproduce the emitted confidence and an auditor could not check
  the arithmetic from the file. A fresh-eyes agent given a real governance task abandoned
  the CLI and rebuilt on the Python API (DX-002). The payload now carries the
  classification, the definition and the domain, all five components, and the weights that
  produced the total; the writer recomputes the weighted sum before emitting and **refuses
  to write a document whose own numbers do not close**. It escaped because the CLI tests
  stubbed the dictionary entry with exactly the four attributes the writer read — a stub
  shaped like the defect cannot see it — so those doubles are real domain objects now.

- **NM-0026 — the machine-readable output was not machine-readable when redirected.**
  `match … -f json > results.json` wrote a spinner frame before the document and the
  summary line after it, because Rich's `Progress` and `rich.print` both default to stdout
  and nothing in the command had ever said otherwise. The most obvious way anyone would
  script this CLI produced a file `json.loads` rejects on its first character. Status moves
  to stderr whenever stdout is carrying a payload, and stays exactly where it was for the
  human `table` format and for `--output`. It escaped because every CLI test asserted that
  a substring *appeared* in the output and none had ever parsed it — `"Summary" in output`
  passes just as happily when the summary is sitting inside a JSON document.

- **NM-0024 — a vector went stale while `sync` reported the row unchanged.** `content_hash`
  hashed a hand-written list of three fields, but `DictionaryEntry.to_searchable_text()`
  embedded those three *and* `synonyms`. Editing an entry's synonyms changed the text that
  got encoded while leaving the hash untouched, so the row was skipped and its stored vector
  silently stopped matching it — no error, and the report said "unchanged". The hash is now
  derived *from* the embedded text, so the two cannot drift; a field that never reaches
  `to_searchable_text()` still cannot invalidate a vector, which is the guarantee that keeps
  an audit-column edit from turning every incremental sync into a full re-embed.

- **NM-0020 — ranking depended on `PYTHONHASHSEED`.** Fusion iterated a set, so the order
  of tied candidates changed with how the interpreter was started. With a measured margin
  of 0.0024 cosine between the correct entry and the nearest wrong one, ties are the normal
  case here, not an edge case — so this decided real answers. Ranking is now total and
  seed-independent, verified across four seeds in separate interpreters.
- **NM-0023 — CI linted only part of the repository.** `ruff` ran over
  `src/nexus_matcher` and `tests` only; `scripts/` and `benchmarks/` were covered by
  nobody and 33 errors accumulated there unseen. All four directories are linted now, with
  a named, dated exemption for six rules on nine frozen experiment scripts so that *new*
  benchmark code is fully covered. A scope detector fails the build if the list ever
  narrows again — and it immediately caught its own directory, `tests/museum`, which no CI
  job was running.

- **The service reported a version it was not.** `GET /`, `GET /health`, the OpenAPI
  `info.version` and every structured log line carried the literal `"2.0.0"` — four
  independent copies, in `app.py` and `shared/logging.py` — while `__version__` had moved to
  2.1.0. `pyproject.toml` sets `dynamic = ["version"]` against `__init__.py`, so
  `__version__` IS what the wheel is built with and every other spelling is a copy that
  drifts. Publishing this release would have shipped a service identifying itself, on every
  surface an operator greps, as a version that has been **deleted from PyPI** — and
  `docs/API_REFERENCE.md`'s route table printed the wrong value as if it were the contract.
  All four now resolve through one cached `service_version()`, the document prints no literal
  at all, and `test_no_service_surface_hardcodes_a_version_string` fails the build on the
  shape of the mistake rather than on its four instances, because a fifth copy on a new
  surface would drift exactly as these did.

- **`scripts/publish.sh` and `scripts/publish.ps1` ran `twine check` and nothing else.**
  `scripts/release_preflight.py` — which installs the built wheel into a clean venv with no
  extras, runs every console script for real including under a cp437 codepage, resolves every
  `__all__` name and entry point, and drives an end-to-end match offline — was wired into
  `.github/workflows/publish.yml` alone. Two paths to PyPI, one gate, and the ungated one is
  the path a human takes under time pressure when CI is red or slow. `twine check` reads the
  metadata: it cannot see a console script installed without its dependencies, which is
  exactly what 2.0.0 shipped. Both scripts now run the preflight after the build and **before
  the upload confirmation**, on `dist/*.whl` — the artifact about to be uploaded, not a second
  one built for the occasion, because a preflight that builds its own wheel proves something
  about a file nobody publishes. A non-zero exit aborts; in PowerShell that means an explicit
  `$LASTEXITCODE` test, since `$ErrorActionPreference = "Stop"` does not stop the script on a
  native executable's exit code and the upload would otherwise have run straight past a
  printed "NOT FIT TO PUBLISH". Both scripts also refuse to proceed when `dist/` holds
  anything other than exactly one wheel, so the gate cannot be satisfied by checking the
  wrong file. Verified on both shells, all four paths: preflight passes, preflight fails, no
  wheel, two wheels.

### Fixed — documentation

- **Three published wire keys and a fourth verdict value were documented nowhere a caller
  reads.** `absoluteScore`, `fieldDecisions`, `scoring` and `NO_MATCH` appeared in zero of
  `README.md`, `QUICKSTART.md`, `docs/API_REFERENCE.md` and `docs/GOVERNANCE.md`, and both
  the QUICKSTART and GOVERNANCE worked examples printed a response body missing all of
  them — under the words "real output" and "two identical requests produce identical
  bytes". `tests/packaging/test_documented_routes.py` checks **paths**, so a route can gain
  keys, gain a schema and gain an enum member with nothing failing.

  `docs/API_REFERENCE.md` now names all four top-level response keys in wire order and
  documents every member of the candidate, `fieldDecisions`, `scoring`, the status body and
  the retrieval diagnostic, key by key. `docs/GOVERNANCE.md` rule 3 — *an unmatched field
  inherits nothing* — now says **where** a consumer reads that rule (`fieldDecisions[path]`
  over HTTP, `session.field_decisions()` in Python) and carries a captured `NO_MATCH`
  response whose rank-1 candidate holds a populated `CREW_ONLY` class and a confidence of
  0.82, which is the shape of the mistake the key exists to prevent. `QUICKSTART.md` and
  `README.md` state the four keys and the read order.

  **Every payload in those four documents was re-captured from a live app** rather than
  edited, including the two that were stale, and each was checked for byte-identity across
  two identical requests and for being ASCII-only. Measuring the gap that let this ship: a
  script asserting that every property of every published response schema, and every value
  of every published enum, appears in at least one of those four documents reported **37**
  undocumented names at `HEAD` and **0** after this change. It is a script and not a test
  because this lane does not own `tests/`; see the note under **Planned**.

- **The `absolute_score_floor` figure quoted for the new `NO_MATCH` verdict was a
  stand-in-encoder number and would never have fired.** The reported floor of 0.30, derived
  from an unmatchable field scoring 0.123, came from a fixture that substitutes a
  bag-of-tokens provider for the encoder. Re-measured on the **shipped bundled int8 ONNX
  encoder** against the 30-entry Gravel Bay glossary, driven through a live
  `POST /api/v1/match` with the floor loaded from `NEXUS_API_MATCHING_CONFIG` on each run:
  the lowest rank-1 `absoluteScore` produced by any of 72 fields across two field sets was
  **0.4966**, so **a floor of 0.30 — or 0.40, or 0.45 — produces zero `NO_MATCH` verdicts
  and cannot fire on any input.** A caller who read 0.30 as a starting point would have
  configured a floor that is on, monitored, and inert.

  No number replaces it, because there is nothing to replace it with: a floor is a statement
  about a score distribution and the distribution belongs to the caller's glossary. The new
  [`docs/guides/absolute_score_floor.md`](docs/guides/absolute_score_floor.md) is a
  **procedure** — build a labelled sample that includes fields with no correct answer, split
  the rank-1 absolute scores into two piles, compare `max(negatives)` against
  `min(positives)`, and take the midpoint if they separate or sweep the trade if they do
  not.

  Its worked example is the argument for the procedure existing. The **same** glossary,
  encoder, library and route over two different field sets gave two different answers. 26
  described fields plus four invented unmatchable ones separate cleanly,
  `(0.650861, 0.720624]` — a 0.07-wide band whose midpoint, 0.685, catches all six
  unanswerable fields and costs none of the 24 correct ones. The pack's own 42-row
  `labels.jsonl`, which is bare column names with no descriptions, **overlaps**: its
  apparent gap is **0.0027** wide (0.614483 to 0.617166), which is under a third of the
  request-shape wobble below, so the defensible floor is 0.59 — catching 5 of 8 rather than
  7 of 8. Nothing changed but the field text.

  The guide also records a hazard nothing in the tree had measured: **`absoluteScore` for
  one field moves with what else is in the same request.** The bundled encoder pads a batch
  to its longest member, so the same text encoded alone and again beside a longer sibling
  gives query vectors whose cosine with each other is 0.994260. Over 30 fields scored once
  one-per-request and once as a single 30-field request, `max|delta|` was **0.010036** and
  two fields changed which entry ranked first. Responses stay byte-identical for identical
  requests — the dependence is on request *shape*, not on the run — but a floor placed
  within 0.02 of anything it cares about is not calibrated. `docs/API_REFERENCE.md` gains
  the `absolute_score_floor` row its `MatchingConfig` table was missing, and
  `docs/GOVERNANCE.md`'s H-002 note now points at the two-corpus measurement.

- **Eleven sentences across six documents denied that the matching endpoint existed**, in
  this repository's own register for being honest about what is missing, while
  `create_app()` had been registering all three routes: `README.md` ×2 — and README is the
  PyPI long_description, so a wheel built from that tree would have carried the false
  sentence to the package page — `QUICKSTART.md` ×2, `docs/API_REFERENCE.md`,
  `docs/ARCHITECTURE.md` ×3, `docs/DEPLOYMENT.md`, `docs/PROJECT_STATE.md` ×2.
  `grep -rn "api/v1/match" --include=*.md .` returned zero hits.

  All are retracted, and `tests/packaging/test_documented_routes.py` now fails the build in
  **both** directions — a registered route absent from the four route tables, and any
  tracked markdown denying a route that is registered. It was watched red on the docs as
  they stood. Two of the eleven are why it flattens emphasis and line wrapping before
  matching: one wrote its negation inside bold markers, one wrapped across two lines of a
  blockquote, and both survived a plain grep — which is why two independent reviews reported
  different counts of the same defect. It also caught two attempts, while this change was
  being written, to retract a denial by quoting it — a quoted denial reads identically to
  the original once the formatting is gone. A third, in the past tense, slipped past the
  patterns and was found by hand; the gate is a floor, not a proofreader.

  The 2.0.x retraction table below is untouched: it is a true statement about what 2.0.0
  claimed, and rewriting a released section to agree with today's router would be falsifying
  the record. So the gate scans this file from the `[Unreleased]` heading through the newest
  version section and stops at the second heading down — the two places a claim about
  *today* can live, since a version that has not been published is not history yet. An
  earlier revision of this paragraph described a window that stopped at the FIRST version
  heading, which was right while `[Unreleased]` held the staged work and silently wrong the
  moment that work was collected under a 2.1.0 heading to cut a release. Re-applying that
  superseded rule to this file measures the damage: **86 characters** — the two-line note
  under `[Unreleased]` and nothing else — so a denial written anywhere in the release being
  staged passed the gate. The window now runs from that note down to the 2.0.1 heading,
  which is the whole of this release and none of the history below it; both bounds are
  asserted, in both directions, by
  `test_the_changelog_scan_covers_the_staged_release_and_stops_before_history`.

  Two spellings in this prose are deliberate rather than stylistic. The `[Unreleased]`
  heading is written without its marker, and no line of this file starts with a hash-hash-
  space except a version heading: the window opens on a line-anchored match for that heading
  and closes at the next line starting that way, so a wrapped paragraph beginning with those
  three characters would hand the gate a heading that is not one and truncate the window
  there. The anchoring is itself a fix — the first cut located the window with a plain
  substring search, and a mutation deleting the real heading stayed green because the search
  landed on a sentence quoting it.

- **`docs/DEPLOYMENT.md` §2 documented five API environment variables that no code reads**
  (`NEXUS_API_HOST`, `_PORT`, `_WORKERS`, `_TIMEOUT`, `_CORS_ORIGINS`) and none of the ones
  `create_app()` actually reads. An operator configuring a deployment from that list got a
  server that 503s every match. Host, port and worker count are flags on
  `nexus-matcher api`, not environment variables. §9 "Security Hardening" was worse than
  useless for the same reason: it told operators to set `NEXUS_API_CORS_ORIGINS` in a
  JSON-list syntax nothing parsed, and three `NEXUS_RATE_LIMIT_*` variables for a mechanism
  that does not exist, so hardening by the book left CORS wide open.

  That list was then short by one, which is the same defect at a tenth the size:
  `NEXUS_API_MAX_BODY_BYTES` appeared in **no** document, and §2 counted "the nine variables
  `create_app()` actually reads" plus "three more" against a real set of **thirteen**. The
  count is now measured rather than counted by eye — `create_app()` is handed an environment
  mapping that records every key it looks up and driven through a full startup, because three
  of the thirteen are read inside the lifespan handler and one only when another is non-empty,
  so no grep of the module header gets this right. The missing variable is the one that can
  refuse to start: §2 now documents the derived default, the floor it is validated against,
  the refusal verbatim, and the two ways out that the refusal names.

- **`docs/API_REFERENCE.md` documented no 413 and no 504**, though both are published on both
  match routes, and its feedback row named only the 503 while 422 and the new 500 are
  published there too. Those are the statuses a generated client is least able to guess: the
  413 is what an adopter's chunking branch keys on, and the 504 is the one that ends a long
  request. The document now carries the whole published set — 15 failures across 5 routes,
  enumerated from a live `/openapi.json` rather than from memory — with the error code and
  the `details` keys each one actually carries, every row driven on a live app rather than
  read off the source. It also says which statuses are answered but *not* published, and why
  404 and 405 cannot be: they are raised by the router, so no path object in the spec owns
  them.

- **`docs/ARCHITECTURE.md` §4.1** was labelled a design target and drew a flow with a
  schema-parsing step and two cache steps that the endpoint does not perform. It now
  describes the shipped path, including the admission/deadline boundary and the
  conservation checks that refuse a short response.

### Added — measurement

- **A real, labelled benchmark.** `benchmarks/datasets/build_benchmarks.py` builds 793
  query→entry pairs from BIRD-SQL dev `database_descriptions` (361) and the OHDSI OMOP
  CDM v5.4 field-level spec (432). Dictionary entries are indexed on **business name and
  definition only** — the source system's technical column name is deliberately excluded,
  so nothing can be solved by string identity.
- **`benchmarks/eval_pipeline.py`** — end-to-end evaluation that drives the real
  `NexusMatcher` orchestrator, not a hand-rolled cosine loop. Artifact:
  `eval_pipeline_combined.json`.

  | Metric | combined | bird | omop |
  |---|---|---|---|
  | P@1 | 0.700 | 0.490 | 0.819 |

  Also: P@5 0.888, MRR@10 0.781, Recall@10 0.919, 652 fields/sec, 1.76 s index build
  for 793 entries, on CPU.
- **Ablation and calibration experiments**, each writing its own artifact:
  `exp_query_repr.py`, `exp_fusion.py`, `exp_calibration.py`, `exp_rerank.py`.
- **`tests/unit/test_regression_guards.py`** — tests that fail when accuracy-destroying
  changes are made, rather than only when APIs break.

### Fixed — accuracy defects found by the new benchmark

- **`AbbreviationExpander` destroyed enriched queries.** It collapsed multi-word
  natural-language queries into a single camelCase mega-token. The production path was
  measuring dense P@1 0.309 and BM25 P@1 0.005, with **787 of 793 queries returning zero
  BM25 hits** — the sparse arm was contributing essentially nothing. After the fix:
  dense P@1 0.636, BM25 P@1 0.531, zero zero-hit queries.
- **Missing BGE query-instruction prefix.** BGE retrieval models are trained with an
  instruction on the query side only. Adding it asymmetrically (queries prefixed,
  documents not) was worth +5.3 points of P@1.
- **Loading a second dictionary left the first one's vectors searchable.**
  `_dictionary_entries` was replaced but the vector store was only ever upserted into,
  producing silent misses and matches against unresolvable entries. The store is now
  cleared of previously indexed ids first.
- **Failed sparse index builds were silent.** `SparseRetriever.index()` returned a
  `Result` that was discarded, so a failure left the matcher running dense-only with no
  indication. It now raises.
- **`match_schema_session()` parsed the source twice**, doubling parse cost and risking
  a mismatch between the returned schema and the results computed from it.

### Changed — defaults, all measurement-driven

- **Encoder `batch_size` default 512 → 32**, hoisted out of four repeated signatures into
  one named `DEFAULT_BATCH_SIZE`. Artifact: `exp_encoder_batch_size.json`.

  **512 was never a cap.** Batches are assembled against a 4096-token budget, and on the
  full FHIR corpus no batch ever reaches 512 rows — so 512, 1024 and 4096 produce
  byte-identical batch plans (65 batches, widest 315 rows, `batches_capped_by_rows: 0`)
  and byte-identical embeddings. Whatever the number was chosen to do, it had stopped
  doing it when token-budget batching landed.

  Cost, interleaved best-of-3 against a noise band calibrated on identical code in the
  same session, on an idle machine per H-007: **32 is 5.6% faster at one intra-op thread
  (band 0.7%) and 38.6% faster at the shipped eight (band 6.2%)** — outside the band in
  both regimes, which is the H-003 requirement that the win not be a thread artifact.
  A third run at 36–88% CPU busy produced bands of 15–48% and is retained in the artifact
  as `cost_UNMEASURABLE_busy_machine` rather than averaged in; it certifies nothing.

  **No accuracy claim is made in either direction.** Paired exact McNemar against 512 over
  1556 queries: every batch size is inconclusive (32: 45 gained, 48 lost, p=0.84). int8
  inference is genuinely not batch-invariant — 886 of 1556 queries change rank between 32
  and 512 — but the movement is symmetric.

  Two numbers this project previously believed, corrected by re-measuring rather than
  re-quoting: the padding ratio at 32 vs 512 is **1.0151 vs 1.0282**, a 1.29% gap, not the
  1.048 vs 2.230 recorded earlier — that 2.2x belongs to the pre-token-budget fixed-window
  batching and did not survive it, so padding is *not* the mechanism behind the speedup.
  And "512 is ~11.9% slower" had no artifact anywhere in the repo; the direction was right,
  the magnitude is machine- and thread-dependent across 4.8–38.6%.

  32 rather than 16 because 16 is 1.7 points better at one thread and 8.8 points worse at
  the thread count that ships. That tiebreak is the weakest link in this decision and is
  labelled as such where the constant lives; both values beat 512 cleanly.

- **Query text now includes the parent path.** `satscores sname` instead of `sname`.
  Worth **+20.1 points of P@1** (0.491 → 0.691) — the largest single accuracy factor in
  the pipeline. Artifact: `exp_query_repr_combined.json`.
- **Scalar type words are no longer appended to queries.** Adding "text field" to the
  query *cost* 2.1 points of P@1. Now off by default.
- **Fusion is linear min-max with `fusion_alpha = 0.90`,** not RRF. Measured on the
  combined benchmark: linear dense=0.9 → 0.7024, dense-only → 0.6910, RRF k=60 → 0.6103.
  **RRF was the worst method measured and worse than not fusing at all.** Artifact:
  `exp_fusion_combined.json`.
- **`auto_approve_threshold` raised from 0.75 to 0.85.** At 0.75 the auto-approved slice
  was only 86.3% precise. At 0.85 it is 94.7% precise over 42.7% coverage. Auto-approving
  a wrong mapping costs more than sending a field to review. Artifact:
  `exp_calibration_combined.json`.

### Changed — performance

- `InMemoryVectorStore` no longer re-normalises the entire corpus matrix on every query
  (this also removed a large per-query allocation).
- Edit distance now uses `rapidfuzz` instead of a pure-Python DP loop; results are
  bit-identical.
- `_match_fields` embeds all query strings in one batched call rather than one per field.

  These three are micro-benchmarks without committed artifacts — see the "unarchived"
  section of [docs/BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md). So are the three
  below, which are timings of this change rather than of the matcher.

- **`problems_with()` no longer names the whole vocabulary in every per-row message, so the
  text of `source_metadata['governance_problems']` entries has changed.**
  `", ".join(sorted(self.codes))` sat inside the branch that runs once per defective *row*,
  and `codes` builds a fresh frozenset on every access. Measured over 30,000 unknown-code
  rows: 50 ms at 9 classes and 804 ms at 800, against a valid-code path flat at ~35 ms.
  Caching the joined string does not fix it — interpolating a 4,000-character list into
  30,000 f-strings copies it 30,000 times — and caching does not touch the larger cost at
  all: under `governance_strict=False`, the escape hatch the refusal message itself
  recommends, every one of those strings is retained, traced at 127.1 MB holding one
  distinct value. The message now names the offending token and the *count* of declared
  codes; the list itself is printed once, in the refusal `load_entries` already builds,
  which fell from roughly 70,000 characters to 1,991. The phrase "no vocabulary is
  configured" is unchanged, because "configured nothing" and "configured permissively" are
  different answers and that message is the only place a reader learns which one they have.
- **The governance column-resolution cache is keyed on the row's governance columns rather
  than on the whole row.** CSV and Excel hand every row the same key tuple, so the whole-row
  key worked and hid the problem; JSON, JSONL and iterable-of-dicts sources hand each row
  back verbatim, so a sparse exporter produces one cache entry per *combination of present
  columns* — twelve optional columns is 4,096 of them against a 64-wide LRU, which is a
  cliff rather than a slope. Measured over 30,000 rows: 64 shapes 36.5 ms, 65 shapes
  137.9 ms, 4,096 shapes 170.3 ms; now flat at ~45 ms. Filtering is not free, and that is
  the trade: about +18% on the uniform CSV shape that was already fine. The key stays a
  tuple and not a frozenset, because two spellings can normalise alike and set iteration
  order resolved the governance column three different ways across six processes under
  `PYTHONHASHSEED=random` — the same file refused on one run and accepted on the next.
- **Both match handlers return a `DeterministicJSONResponse` instead of a `dict`,** which
  short-circuits FastAPI's `serialize_response` and the `jsonable_encoder` walk inside it.
  Measured on a 250-field explain payload: 28.6 ms of encoding against 3.4 ms of rendering,
  the largest single item by internal time at 58,843 calls per request — and all of it on
  the event loop, so it was `/health/live` latency too. Bodies are byte-identical, verified
  across twelve shapes. Two deltas come with it, both deliberate: `jsonable_encoder` was a
  lenient net (`datetime` to ISO, `Decimal` to float, `set` to list, none of which
  `json.dumps` accepts), so a future non-primitive leaf is now a loud 500 rather than a
  silent coercion; and a response header set by a future `Depends(...)` would be lost, which
  is inert today because `src/` contains no `Depends(` at all.

### Removed

- **The duplicate `nexus_matcher_src/` tree** (127 files). `src/nexus_matcher/` is
  canonical.
- **The stray second readme** (`README (1).md`). There is one `README.md`.
- **Broken plugin entry points.** `pyproject.toml` declared entry points for
  `csv_headers`, `database`, `faiss` and `openai` modules that do not exist; a broken
  entry point makes plugin discovery raise at import time for every consumer of the
  package. Remaining entry points were each verified importable.
- **`vector_store` and `cache` are gone from the `/health/ready` component map.** Both were
  set `True` inside a `try:` whose body was a comment saying the check would go here, so no
  failure could reach the `except:` and neither could ever be `False`. A component that
  reports `True` unconditionally is not a check, it is a claim, and this map is read by
  rollout gates. `docs/API_REFERENCE.md` documented them as hardcoded, which is why their
  removal is recorded here rather than treated as an implementation detail. The map is now
  `api`, `config` and `matcher`, and `matcher` is a real check — see Changed — BREAKING
  above. The two come back when something actually probes Qdrant and Redis.

### Documentation — retractions

The following claims appeared in the README, this changelog, the package docstring and
`docs/ENHANCEMENT_JOURNEY.md`. They were false and have been removed.

| Retracted claim | What is actually true |
|---|---|
| **"100% Precision@1"** | Came from `benchmarks/suite_008_combined.py`, which **never calls `NexusMatcher`** — it computes raw cosine similarity over 17 hand-written source fields against a 20-entry hand-written target set, and got 17/17. Measured end-to-end P@1 is **0.700**. |
| **"1.68× INT8 speedup"** | Not in any artifact. `suite_002_real_20251209_162836.json` measures 1.27× at batch 32 (the batch size the claim cited), ranging 1.26×–2.93× across batch sizes, on a machine without VNNI. |
| **"3.07% accuracy loss" from INT8** | No accuracy figure was ever recorded. The artifact carries `accuracy_pass: false` and `overall_pass: false`. |
| **"56.99% cache hit rate", "99.3% cost reduction"** cited as VALIDATED | `benchmarks/suite_004_cache_performance.py` and `suite_004b_semantic_cache.py` write **no artifact at all**. One cited run ID, `run_20251209_062xxx`, is a literal placeholder. Also: a cache's hit rate is a property of the workload, which in this case was a synthetic 60%-repetition query pattern. |
| **"86x faster reranking" / "93.7x"** | The same measurement compared two ways — cold 274.0 ms avg vs warm 2.93 ms avg (93.6×) or vs warm 3.17 ms p95 (86×), at 100 candidates. It is a **latency** result for pre-computing document token embeddings, and the same artifact shows MaxSim did not change the top-5 ranking at all on its sample. It is not evidence of an accuracy gain. |
| **10 documented REST endpoints** | 2.0.0's app served health and introspection routes and nothing more. Matching over HTTP arrives in this release — see Added above — and the rest of that list is still absent: no dictionary CRUD, no cache routes, no `/metrics`, no API-key auth and no rate limiting. See [docs/API_REFERENCE.md](docs/API_REFERENCE.md). |
| **"Prometheus metrics endpoint"** listed under Added in 2.0.0 | A `PrometheusMetrics` backend class exists; no route exposes it. |
| **Default model `all-MiniLM-L6-v2`** | The shipped default in `SentenceTransformersProvider` is `BAAI/bge-base-en-v1.5`. The published benchmark uses `BAAI/bge-small-en-v1.5`. |
| **YAML / environment configuration of matching** | `NexusMatcher.from_config()` accepts a `MatchingConfig` or a JSON/TOML file, but the `NEXUS_*` settings classes are still consumed only by the logging setup. There is no YAML path and no environment-variable control of matching behaviour. |
| **Test count "433 tests"** | Current measured state: 551 passed, 0 failed, 35 skipped (skips are uninstalled optional dependencies). Line coverage 60% against a configured gate of 80%. |

### Planned

- GPU measurement. Every number in this repository is CPU-only, single machine.
- Accuracy measurement at catalogue scale; the benchmark corpus is ~1,200 entries.
- Non-English schema matching. All measurement to date is English.
- Authentication in-process, if it is ever wanted: as a router dependency on `/api/v1/*`
  comparing with `hmac.compare_digest` and declared as a security scheme so it reaches
  `/openapi.json`. Until then the service is unauthenticated and says so — see Security
  above.
- **A Python twin of the Java client-drift gate**, asserting that every property of every
  published response schema and every value of every published enum appears in at least one
  of `README.md`, `QUICKSTART.md`, `docs/API_REFERENCE.md` and `docs/GOVERNANCE.md`.
  `tests/packaging/test_java_client_contract.py` already does this against a generated
  client, so the *wire* cannot drift from the *client* unnoticed; nothing does it against
  the *documents*, and `test_documented_routes.py` checks paths only. That is the exact hole
  `absoluteScore`, `fieldDecisions`, `scoring` and `NO_MATCH` fell through. Run as a script
  over this tree it reports 37 undocumented names at `HEAD` and 0 now, so it is both
  non-vacuous and currently satisfiable. It is not committed here because it belongs in
  `tests/`, which this change does not own; the design note, including the two judgement
  calls it needs (which documents count as caller-facing, and how to declare a key
  deliberately undocumented) is in the change's report.

  "An HTTP matching endpoint" used to head this list. It is delivered in this release, so
  it is gone from it rather than annotated: a Planned list that keeps items it shipped is
  the same shape as the documentation defect recorded under Fixed — documentation above.

---

## [2.0.1] - 2026-08-09

Fixes from a verification sweep of the published 2.0.0 wheel — installed into a clean
environment and driven through the documented quickstart. Packaging, CLI and
documentation only. The matching pipeline is untouched and no measured number in this
file moves.

### Fixed

- **`match` and `sync` crashed with `UnicodeEncodeError` on non-UTF-8 Windows consoles.**
  Rich's default spinner animates with Braille code points that cp437, cp850 and cp1252
  all refuse to encode, and its `no_wrap` truncation adds U+2026, so the only two commands
  that do real work died with a bare codec error and exit 1 — after the matching had been
  paid for, and with nothing to suggest that a terminal or a field name was the problem.
  The CLI now picks decorations the console can encode and escapes what is left, keeping
  the user's code page rather than forcing UTF-8 onto it.
- **The `nexus-matcher` console script was installed without typer or rich.** The entry
  point is declared unconditionally but its dependencies sat in the `cli` extra, so a
  plain `pip install nexus-matcher` put a command on `PATH` that could not start. Both
  moved into the core dependencies; `cli` is kept as a name so existing pins still
  resolve. Its `typer[all]` marker is also gone — typer has published no extras since
  0.12, so that asked for something which does not exist and pip warned about it on every
  install.
- **`--output/-o` was silently ignored when `--format` was left at its default.** The
  default was `table` and the table branch had no write path, so
  `nexus-matcher match schema.avsc -d dictionary.csv -o results.json` printed to stdout,
  wrote no file and exited 0 — a scripted run could not tell that it had produced nothing.
  The format is now inferred from the `--output` extension when it is not given.
- **`match_schema` silently dropped fields whose dotted paths collided.** Results are
  keyed by field path, and where two fields produced the same path the later one replaced
  the earlier. The displaced field then never appeared in the results and so never
  received the governance classification of the entry it would have matched; the returned
  mapping was simply shorter than the schema, with nothing to say which fields were gone.
- **Flattened Avro results were keyed by names the caller had not supplied.** The keys in
  the returned mapping did not match the flattened field names passed in, so looking a
  result up by the name you provided missed.
- **`create_app` was listed in `__all__` but needs the `api` extra.** `from nexus_matcher
  import *` therefore raised `ImportError` on any install without `[api]`, including the
  bare install the README documents as the complete pipeline.
- **The `Documentation` project URL 404'd, and the README's relative links were dead on
  PyPI.** README.md is the PyPI `long_description`, where a relative markdown link
  resolves against pypi.org rather than against the repository: twelve of them — including
  every entry in the README's own Documentation section — went nowhere for anyone arriving
  from the package page. They are absolute GitHub URLs now. One dead in-page anchor went
  with them; `#known-limits` named no heading in the file and scrolled nowhere.

---

## [2.0.0] - 2025-12-09

Complete rewrite from a procedural single-file implementation to a hexagonal
(ports and adapters) architecture.

> **The performance table originally published with this release was not valid.** See
> the retractions above. The entries below have had unsupported numbers removed.

### Added

- Hexagonal architecture: domain / application / infrastructure / presentation layers,
  dependency-injection container, plugin system via entry points.
- Three-stage matching pipeline: retrieval → optional reranking → multi-signal scoring
  with a decision policy (`AUTO_APPROVE` / `REVIEW` / `REJECT`).
- Schema parsers: Avro, JSON Schema, SQL DDL.
- Dictionary loaders: Excel, CSV.
- Vector stores: in-memory, Qdrant, HNSW.
- Sparse retrieval: BM25.
- Rerankers: cross-encoder and ColBERT MaxSim. Both optional, off by default.
- Caches: L1 LRU in-memory, Redis, content-addressed semantic cache. Implemented and
  unit-tested; not exercised by the accuracy benchmark.
- Incremental update manager with BLAKE3 content hashing.
- Learned type projections and a graph matcher. Experimental.
- INT8 quantized embedding provider via ONNX Runtime.
- REST API with health and readiness probes; CLI with Typer; Python library API.

### Changed

- Configuration moved to YAML with environment-variable overrides — **note that this
  affects logging only; the matching pipeline does not read it.**
- Test suite expanded substantially.

### Removed

- Legacy single-file implementation.
- OpenAI embedding provider.

### Security

- Non-root Docker container.
- CORS is currently configured with `allow_origins=["*"]`; narrow it before exposing the
  service.

---

## [1.0.0] - 2025-10-15

- Initial release: basic semantic schema matching, Excel dictionary loader, Avro parser,
  sentence-transformers embeddings, cosine similarity scoring.
- Single-threaded, no caching. No labelled benchmark existed at this point, so the
  accuracy of this release is unknown; earlier claims of "~85%" are unsupported.

---

## Upgrade guide: 1.x → 2.x

**Import paths changed.**

```python
# Old
from schema_matcher import match_schema

# New
from nexus_matcher import NexusMatcher
matcher = NexusMatcher.from_config()      # NOT NexusMatcher()
matcher.load_dictionary("dictionary.csv")
results = matcher.match_schema("schema.avsc")
```

`NexusMatcher()` with no arguments raises `TypeError` — `embedding_provider` and
`vector_store` are required. `from_config()` supplies the defaults.

**Result structure changed.**

```python
# Old
result = {"field": "matched_entry", "score": 0.95}

# New
results: dict[str, tuple[MatchResult, ...]]     # keyed by SchemaField.full_path
top = results["Customer.email"][0]
top.dictionary_entry.business_name
top.final_confidence          # float in [0, 1]
top.decision                  # MatchDecision enum (str-backed)
top.score_breakdown           # per-signal components
```

---

## Links

- [Repository](https://github.com/pierce-lonergan/nexus_matcher)
- [Issues](https://github.com/pierce-lonergan/nexus_matcher/issues)
- [Benchmark registry](docs/BENCHMARK_REGISTRY.md)
