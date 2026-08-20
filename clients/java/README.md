# nexus-matcher-client

Java client for the nexus_matcher governance matching API.

```xml
<dependency>
  <groupId>io.github.pierce-lonergan</groupId>
  <artifactId>nexus-matcher-client</artifactId>
  <version>2.1.0</version>
</dependency>
```

Java 17. One runtime dependency (`jackson-databind`); transport is `java.net.http.HttpClient`
from the JDK. The version tracks the Python package's.

## What this is, and what it deliberately is not

**Transport and types.** The Python service is the one implementation of the matching and
governance semantics. This client carries requests there and decodes what comes back. It does
not re-decide a match, does not apply a confidence threshold, and does not rank two
classifications against each other — for that last one it hands you
`Vocabulary.tiersMostOpenFirst()` and stops, because the ordering belongs to your vocabulary and
this library ships no taxonomy at all.

Nothing in your controlled vocabulary is typed as a Java enum. `code`, `name`, `classification`,
`enhancement`, `tier`, `absoluteScoreMetric`, a status warning's `code` and every key and value of
`sourceMetadata` are open and will stay open. The only Java enums here are `MatchDecision` and
`FieldDecision`, which are the library's own vocabularies and are published as closed sets in
`/openapi.json` — and they behave differently on an unknown value. See
[Two enums, two answers](#two-enums-two-answers-about-an-unknown-value).

## Using it

```java
NexusMatcherClient client = NexusMatcherClient.builder("http://127.0.0.1:8000")
        .timeout(Duration.ofSeconds(30))   // keep ABOVE the server's own deadline
        .maxRetries(2)                     // 503 only
        .build();

String path = "booking.passenger.legal_name";
MatchResponse response = client.match(List.of(
        FieldSpec.of("legal_name", path,
                     "Full legal name as printed on the sailing manifest.", "string")));

// The FIELD verdict first. It is the authority for a column; a candidate's own decision is
// about that candidate, and a NO_MATCH field still arrives with candidates that look fine.
switch (response.verdictFor(path).map(FieldVerdict::decision).orElse(FieldDecision.UNKNOWN)) {
    case AUTO_APPROVE -> response.inheritableGovernanceFor(path).ifPresentOrElse(
            governance -> apply(governance),
            ()         -> applyOpenTier(response.vocabulary().openClassification()));
    case REVIEW, REJECT, NO_MATCH, UNKNOWN -> sendToAHuman(response.candidatesFor(path));
}
```

The client is immutable and thread-safe; build one and share it.

### Routes

| method | route | returns |
|---|---|---|
| `match` / `matchBatch` | `POST /api/v1/match`, `/match/batch` | `MatchResponse` |
| `lookup` | `POST /api/v1/lookup` | `LookupResponse` |
| `status` | `GET /api/v1/status` | `ServiceStatus` |
| `diagnoseRetrieval` | `POST /api/v1/diag/retrieval` | `RetrievalDiagnostic` |
| `submitFeedback` | `POST /api/v1/feedback` | `FeedbackReceipt` |
| `health` / `readiness` | `GET /health`, `/health/ready` | `HealthStatus` / `Readiness` |

`lookup(String)` goes over the **POST** route rather than the service's single-id
`GET /api/v1/lookup/{id}`. A dictionary id is an opaque string of yours that may contain a slash,
a space or a non-ASCII character, and putting one in a path segment means percent-encoding it
against a catch-all route where an encoded `%2F` and a raw `/` mean the same thing to the server
and different things to every proxy in between. The POST route takes the identical string in a
JSON body and answers the identical DTO. The GET form is still there for a terminal.

### Field keys

Exactly `name`, `path`, `doc`, `type`. They are **not** the `flattenedName` / `dataType`
spellings in `examples/governance/fields.json` — that is the example pack's own input format, and
pasting a row from it into a request is a 422. The server sets `extra="forbid"` on a field spec on
purpose, so a misspelled `doc` is a loud failure rather than silently dropped retrieval signal.

Send a **dotted** `path`. The segment before the last dot becomes the retrieval query's parent
context, which is the largest single accuracy factor measured on this task, and it is also the key
you look the answer up under.

### The two meanings of a null class

`governance` is null in two cases and they mean opposite things, so the client does not make you
test for null:

| `governanceStatus()` | on the wire | what it means |
|---|---|---|
| `CONFERRED` | an object | this class applies |
| `OPEN_TIER` | `null` | the matched entry carries no code: governed, **openly**, at `vocabulary().openClassification()` |
| `WITHHELD_REJECTED_TOP_MATCH` | `null`, at rank 1, decision `REJECT` | no entry describes this field. It inherits **nothing** — do not fall back to the runner-up |

A **rejected runner-up keeps its class**, and that is not a bug: nothing inherits from a runner-up,
and the class is exactly what lets a reviewer see that rank 1 is a direct identifier and rank 2 is
not.

`governanceStatus()` is a reading of the rule the server publishes, from `rank`, `decision` and
`governance` on the response. It decides nothing: when a class is withheld it still hands back
nothing, it only says which kind of nothing.

### Read `decision()`, never `confidence()`

`confidence` is rank-relative — a min-max normalised fused retrieval score with a structural floor
— so a high number is not evidence of a good match, and on the example pack a field nothing
governs scores higher than several that matched correctly. `REVIEW` means "a human must decide",
never "probably fine". Do not diff against `confidence` either; it moves with any retrieval change.

### One verdict per column: `fieldDecisions`

`decision` on a candidate is about **that candidate**. The value that goes into a metadata sheet is
the field verdict, and the server publishes it rather than leaving every client to reconstruct the
roll-up rule differently:

```java
FieldVerdict verdict = response.verdictFor(path).orElseThrow();
verdict.decision();     // AUTO_APPROVE | REVIEW | REJECT | NO_MATCH | UNKNOWN
verdict.wireValue();    // the server's own string, verbatim
```

`NO_MATCH` is the state a per-candidate verdict cannot express: rank-1 `confidence` has a
structural floor (`scoring.confidenceFloor`, 0.63 shipped) that sits **above** the review threshold
(0.50), so rank 1 can never be `REJECT` on score alone and every field would otherwise come back at
least `REVIEW` however irrelevant its best candidate is.

**A `NO_MATCH` field still carries candidates.** The server chose that over an empty list on
purpose — the candidates are evidence for the reviewer who now has to decide — so:

* do **not** find no-match fields by testing `candidatesFor(path).isEmpty()`; you will find none;
* do **not** read `results[path].get(0).governance()` on one. Against the shipped pack, a
  `NO_MATCH` field's rank 1 arrives at confidence 0.82 with a populated protection class and a
  per-candidate verdict of `REVIEW`. Applying it classifies a column from an entry the server has
  just told you describes nothing.

`response.inheritableGovernanceFor(path)` reads the verdict and the candidate together and is empty
for everything but `AUTO_APPROVE`.

### `absoluteScore`: the one number comparable across fields

Present on **every** candidate, not gated behind `explain`. It is the raw dense-retrieval score,
with no normalisation and no floor, and `scoring.thresholdableAcrossFields` names it as the number
you may legitimately compare against a constant across different columns — which `confidence`
is not.

**`null` is not zero.** Null means the dense retriever never returned that candidate at all; it
reached the shortlist through the lexical arm alone. Zero would mean "measured, and as far from the
query as this metric goes", which is a very different claim, and on a cosine metric a very bad
score. The component is therefore a boxed `Double`, deliberately: a primitive would let Jackson
bind the absent number to `0.0` in silence and a caller filtering `>= floor` would drop the
candidate as *failed* rather than as *unmeasured*. Prefer `absoluteScoreValue()`, which is an
`OptionalDouble` with no null to forget, or `hasAbsoluteScore()`. There is no
`absoluteScoreOrZero()` and there should not be.

Read `scoring.absoluteScoreMetric()` before treating it as a cosine, and
`scoring.absoluteScorePooledOverAliases()` before treating it as a similarity to the entry's own
text. A floor measured on a deployment where either differs does not transfer.

### `sourceMetadata`: your own columns, carried and not read

Every candidate and every looked-up entry carries the deployment's own enrichment columns — a
steward, a review date, an upstream identifier — passed through the pipeline untouched.

```java
SourceMetadata plane = candidate.sourceMetadata();
plane.value("steward");        // Optional<Object>
plane.isComplete();            // false => the loader trimmed this entry to its size cap
plane.wasRendered("review_date");  // true => this is the TEXT FORM of a non-JSON source value
```

`values()` is a `Map<String, Object>` and must never become a typed record: both the keys and the
values are your vocabulary, and typing either would compile one organisation's spreadsheet into
this artifact. Nothing in it was read by the server — no score, ranking, threshold or verdict
depends on it — so nothing in it can justify a classification either.

Check `isComplete()` before treating the map as a record of the glossary row. When it is false, an
absent key may have been *dropped to fit the cap* rather than *not populated*, and those are
different conclusions about your glossary.

### Two enums, two answers about an unknown value

Both closed sets in this client are the library's own vocabulary. They behave differently when a
newer server sends a value this build has never heard of, and the difference is deliberate:

| | unknown value | why |
|---|---|---|
| `MatchDecision` (per candidate) | **throws**, naming the value | the service has committed to freezing it. Putting `NO_MATCH` on a *new* enum instead of widening this one **is** that commitment |
| `FieldDecision` (per field) | becomes `UNKNOWN`, keeps the raw string, grants nothing | this is the vocabulary that grows — it was born by growing |

Binding `FieldDecision` closed would re-create, on the new field, the exact deserialisation break
the new field was invented to avoid. The blast radius also differs by the batch size: a `decision`
sits inside one candidate of one field, while a field verdict sits in a map with one entry per
field — up to 250 on the batch route — so refusing the whole body over one unrecognised verdict
would discard 249 that decoded perfectly.

Degrading here is not a guess, because `UNKNOWN` is not usable as an answer:

* `maySafelyInherit()` is false for it, and `inheritableGovernanceFor()` returns empty;
* `FieldVerdict.wireValue()` hands back the exact string the server sent, so the new value can be
  named in a ticket rather than merely counted;
* `response.pathsWithUnknownVerdicts()` lists the columns affected;
* the client never maps it onto the nearest value it does know. That silent reinterpretation — a
  new `APPROVE_WITH_CONDITIONS` quietly becoming an auto-approval — is the failure this seam exists
  to prevent, and it is a different thing from degrading loudly.

`tests/packaging/test_java_client_contract.py` asserts that `UNKNOWN` is **not** a value the
service publishes. The day it becomes one, that gate goes red — because at that moment the client's
sentinel would start absorbing a real server verdict.

### Errors

Every failure is a `NexusMatcherException` (unchecked) carrying the server's error code, message,
`details` map, HTTP status and the `X-Request-ID` — so a Java stack trace joins to a server log
line. Branch on the status and the code, never on the message.

| | | |
|---|---|---|
| `NexusMatcherRequestException` | 400 / 422 | `violations()`, `resultsPerFieldCap()`, `duplicatePaths()`. Never retried |
| `PayloadTooLargeException` | 413 | `suggestedChunkSize()`, plus `limitBytes()` / `observedBytes()` / `source()` on the byte-cap path |
| `ServiceUnavailableException` | 503 | Retryable. `isConfigurationProblem()` says when retrying will not help |
| `DeadlineExceededException` | 504 | `deadlineSeconds()`. Retryable **once at most** |
| `NexusMatcherServerException` | 500 | Not retried. No field was classified — treat the request as unanswered |
| `NexusMatcherTransportException` | — | No response at all. `httpStatus()` is 0 |
| `NexusMatcherProtocolException` | — | A response arrived that this client cannot understand |

Re-chunking a 413:

```java
try {
    client.match(fields);
} catch (PayloadTooLargeException e) {
    int size = e.suggestedChunkSize().orElse(fields.size() / 2);
    for (List<FieldSpec> chunk : FieldSpec.chunk(fields, size)) {
        client.match(chunk);
    }
}
```

`suggestedChunkSize()` is empty on the **byte-cap** path, because nothing on that path counted the
fields — the request was refused from `Content-Length` before the body was read. Fall back to a
size of your own, as above; this client will not invent a number the server did not send.

#### An oversized body could arrive as a transport failure instead of a 413 — fixed server-side

**This is fixed in the server, not in this client.** Against a server carrying the fix, an
oversized body inside the drain budget produces a readable 413 in **400 of 400** requests on each
of four arms — `Content-Length` and chunked, warm pooled connection and fresh. Nothing in this
client changed, and nothing needs to.

What it was: the server refused from `Content-Length` and closed without reading the body while
the client was still writing it. Closing a socket that still has unread bytes sends RST rather
than FIN, and an RST discards whatever the peer had already buffered — including the 413. It
surfaced as:

```
NexusMatcherTransportException: ... fixed content-length: 403, bytes received: 0
```

— headers in, body gone. Measured at 5 of 40 oversized requests (~12%) from this client on a
reused keep-alive connection, and at 8 of 400 from a raw-socket probe on both reused *and* fresh
connections; the 0 of 12 that made a fresh connection look immune was small-n, not a different
mechanism. The server now reads and discards a bounded remainder of a refused body before
answering, so its close is a clean FIN and the response survives.

**One residual case, and it is the one you are least likely to hit.** The server only drains what
it can finish draining — bodies up to twice its byte cap (~19.7 MB against the ~9.87 MB cap). A
body far beyond that is still refused without being read, because a drain that cannot finish
would spend the bandwidth and lose the 413 anyway; that is deliberate, and it is what keeps eight
concurrent 200 MB bodies costing the server no memory. So a *wildly* oversized request can still
come back as a transport failure. Measured with this client's own transport on a pooled
connection: **3 of 20 at 198 MB and 6 of 40 just past the budget, ~15% either way**, against
9 of 40 before the fix. The fix moves the THRESHOLD so realistic mis-chunked batches fall
inside it; it does not make an arbitrarily large body safe to send.
The advice is unchanged and cheap: **treat `NexusMatcherTransportException` on a large request as
a probable 413 and re-chunk rather than retrying the same payload.** That is also the right
behaviour against an older server that predates the fix.

`HttpRequest.Builder.expectContinue(true)` also removed the symptom (0 of 40) and remains
deliberately **not** used: against this server on JDK 17 the client then *hangs* — the 413 arrives
instead of the `100 Continue` the JDK is waiting for, `send` never returns, and the request
timeout does not fire (measured 654 s against a 30 s timeout). A hang defeats every timeout and
fallback an adopter has, which is the exact failure the service's own `errors.py` was written to
prevent. The server-side drain carries its own deadline for the same reason, so a client that
stops sending mid-body is answered rather than waited on indefinitely.

### Retries

`RetryPolicy` is injectable; `RetryPolicy.none()` turns retrying off entirely. The shipped policy
retries **503 only**, twice, with exponential backoff and equal jitter, capped at 5 s.

4xx is never retried — the same request produces the same refusal. 500 is never retried — the
server has already said nothing was classified. **504 is not retried by default**, and at most
once when you opt in with `ExponentialBackoffRetryPolicy.builder().retryDeadlineExceeded(true)`:
the server does not stop the work its deadline fired on, so the timed-out match is still running
and still holding its admission permit, and an immediate retry adds a second copy of exactly the
work that could not finish.

Every attempt of one logical request uses one correlation id, so a retried request reads as one
request in the server's log rather than as three unrelated failures.

### Timeouts

Keep `timeout(...)` **above** the server's own `NEXUS_API_DEADLINE_SECONDS` (25 s by default).
The server's deadline exists so a slow match ends in a 504 rather than a hang; if the client
timeout is the shorter one, the client always gives up first and never sees the 504 — which is the
hang the whole arrangement was built to avoid, reintroduced by an off-by-one in seconds.

### Additive server changes

Unknown response keys are ignored, on every DTO, with `@JsonIgnoreProperties(ignoreUnknown = true)`
rather than only by a flag on the client's own mapper — so a stricter mapper you pass in still
works. This is not hypothetical: `vocabulary` and `governance.enhancement` were both added to this
contract while this client was being written.

Unknown *values* in a closed set are handled per set, not by one rule — see
[Two enums, two answers](#two-enums-two-answers-about-an-unknown-value).

## Building and testing

```
mvn test        # 73 unit tests. No service, no network
mvn verify      # + 57 integration tests against a REAL service
```

`mvn verify` needs four running fixtures, because four of the behaviours worth pinning are
properties of a server's configuration rather than of a request:

| | | |
|---|---|---|
| `http://127.0.0.1:8000` | the example pack, loaded | everything normal |
| `http://127.0.0.1:8001` | no dictionary at all | every match is a real 503 |
| `http://127.0.0.1:8002` | the pack, `NEXUS_API_DEADLINE_SECONDS=0.001` | every match is a real 504 |
| `http://127.0.0.1:8003` | the pack, `NEXUS_API_MATCHING_CONFIG=clients/java/fixture-absolute-floor.json` | `NO_MATCH` is reachable |

From the repository root:

```
./clients/java/serve-fixtures.sh        # or .\clients\java\serve-fixtures.ps1
```

Override the URLs with `-Dnexus.matcher.baseUrl=`, `-Dnexus.matcher.unavailableBaseUrl=`,
`-Dnexus.matcher.deadlineBaseUrl=`, `-Dnexus.matcher.floorBaseUrl=`.

**8003 exists because `NO_MATCH` is otherwise untestable against a live server.** The verdict is
earned two ways: a field that came back with no candidates at all, or a configured
`absolute_score_floor` that rank 1 fails. The library ships no floor and will not invent one — a
floor is a statement about a score distribution, and the distribution belongs to a glossary the
library has never seen — and a 30-entry glossary never returns zero candidates. So without this
fixture the whole verdict could only be tested against a response body somebody typed, which is the
half of a hand-written client that goes stale silently.

`fixture-absolute-floor.json` sets **0.65**, and that number is a test fixture, not a recommended
default. It is the middle of a gap that was measured rather than guessed: sending all 26 fields of
`examples/governance/fields.json` at `top_k=1` to the bundled encoder over the example glossary,

| | n | rank-1 `absoluteScore` |
|---|---|---|
| fields the pack declares a correct answer for | 24 | min **0.7139** |
| fields the pack declares `expected_id: null` | 2 | max **0.5839** |

so 0.65 sits clear of both sides with ~0.06 of margin either way. Reproduce it by matching that
file and reading `absoluteScore` — it takes one request.

**None of that transfers.** A floor is a statement about a score distribution, and the distribution
belongs to a glossary, an encoder and a metric. Calibrate against your own corpus, and read
`scoring.absoluteScoreMetric` and `scoring.absoluteScorePooledOverAliases` before comparing against
a floor measured anywhere else — the same number under `dot` or `euclidean`, or over an index that
pools aliases, is not the same number.

**The integration tests do not mock the service and do not skip when it is missing.** A skipped
integration test is a green build that proved nothing, and this contract moved twice while the
client was being written — exactly the drift a mock cannot see. A missing fixture fails the build
and names the script above.

### The captured bodies

The unit tests decode response bodies **captured verbatim** from a running service, in
`src/test/resources/captured/`. They are captures rather than hand-written expectations, which is
the difference between testing the contract and testing what the author believed it was.

**Regenerate them; never edit them.** With the fixtures running, from the repository root:

```
./clients/java/capture-fixtures.sh
git diff --stat clients/java/src/test/resources/captured
```

The script re-derives every byte of every file, and it exists so that "the fixture disagrees with
the test" can only ever be resolved the honest way. A body somebody typed to make a test pass tests
the author's belief about the contract — the exact belief the fixture was there to check. A diff
after running it is the service having changed; read it before you accept it.

`match-response-no-match.json` is the one capture taken from 8003 rather than 8000.

### One case the shipped configuration cannot reach — and how to reach it anyway

`WITHHELD_REJECTED_TOP_MATCH` — a rejected rank-1 candidate — is unreachable against the **shipped
thresholds**. `final_confidence` has a structural floor of `semantic_weight × fusion_alpha` = 0.63
while `review_threshold` is 0.50, so no top match can fall below the bar that would reject it. The
server says as much in `domain/models/entities.py`, calling the clause latent and reachable for a
caller who raises `review_threshold` past that floor.

That does **not** mean it takes somebody else's deployment to see. `review_threshold` is settable
on a local fixture through `NEXUS_API_MATCHING_CONFIG`, and the case has been provoked and the
client's mapping confirmed end to end against a real server. Port **8004** below, because 8003 is
now the shipped `NO_MATCH` fixture:

```bash
cat > /tmp/reject.json <<'JSON'
{"auto_approve_threshold": 0.995, "review_threshold": 0.99,
 "min_confidence_gap": 0.10, "results_per_field": 5}
JSON

NEXUS_API_DICTIONARY=examples/governance/glossary.csv \
NEXUS_API_GOVERNANCE=examples/governance/protection_classes.json \
NEXUS_API_MATCHING_CONFIG=/tmp/reject.json \
  .venv/Scripts/python -m uvicorn nexus_matcher.presentation.api.app:create_app \
  --factory --host 127.0.0.1 --port 8004
```

Matching `terminal_name` at `top_k=3` against that server returns, and the client reads it as:

| rank | decision | confidence | `governanceStatus()` | class |
|---|---|---|---|---|
| 1 | `REJECT` | 0.9173 | `WITHHELD_REJECTED_TOP_MATCH` | *withheld* |
| 2 | `REJECT` | 0.5375 | `CONFERRED` | `SEALED_RESTRICTED` |
| 3 | `REJECT` | 0.4253 | `CONFERRED` | `CREW_ONLY` |

That is the whole rule in one response: a rejected **rank 1** confers nothing, a rejected
**runner-up** keeps its class, and the two nulls stay distinguishable.

The shipped suite does not start that fifth fixture. `GovernanceNullsIT` instead asserts against
the live pack that no rank-1 candidate is `REJECT`, and fails with an explanation if that ever
stops being true; `MatchResponseDecodingTest` pins the client's decoding from a body labelled in
place as hand-built. If you are changing anything in this area, run the fixture above rather than
trusting either — it is ten seconds of work and it exercises the real server.

`NO_MATCH` used to sit in this same category and no longer does. 8003 makes it reachable, and
`FieldDecisionIT` asserts it against a live server from both sides: the same field is `REVIEW` with
no floor configured and `NO_MATCH` with one, from the same pack, the same glossary and the same
encoder, differing by one configuration key.

### If the integration tests cannot start an HTTP client

On some managed Windows hosts every Java program fails at `Selector.open()` with:

```
java.io.IOException: Unable to establish loopback connection
Caused by: java.net.SocketException: Invalid argument: connect
    at sun.nio.ch.UnixDomainSockets.connect0(Native Method)
```

`HttpClient` needs an internal wakeup pipe, the JDK builds it from an AF_UNIX socket pair when it
can, and endpoint-protection software on such hosts blocks the AF_UNIX *connect* while leaving
*bind* working — so the JDK's own fallback to TCP never triggers. It is not specific to this
client; `Selector.open()` alone reproduces it.

Pointing the JDK's AF_UNIX temp directory at a path that does not exist makes the *bind* fail
instead, which is the failure the JDK does fall back from — so the wakeup pipe is built over TCP
loopback, which is what the JDK did for years before AF_UNIX support landed.

**This is on by default**, in the `nexus.matcher.itJvmArgs` build property:

```xml
<nexus.matcher.itJvmArgs>-Dfile.encoding=UTF-8 -Djdk.net.unixdomain.tmpdir=nexus-no-af-unix-here</nexus.matcher.itJvmArgs>
```

It was originally off, on the reasoning that the build should not quietly depend on a workaround.
That reasoning does not survive contact with where the value actually goes: it reaches only the
forked **test** JVM, through failsafe's `argLine`, so no published artifact can depend on it — and
leaving it off bought nothing except a `mvn verify` that was red out of the box on the machine the
client is developed on. A red default build hides the workaround from nobody; it just blocks.

Turn it off with:

```
mvn verify -Dnexus.matcher.itJvmArgs=-Dfile.encoding=UTF-8
```

The value is a **relative** directory name and must stay one. An absolute path built from
`${project.build.directory}` was tried first and broke the build outright: failsafe passes
`argLine` to the fork unquoted, and a checkout under a path containing a space (`C:\Users\Firstname
Lastname\...`) splits the argument, so the fork dies with *"The forked VM terminated without
properly saying goodbye"* before a single test runs. A bare name has no space to split on and
resolves against the fork's working directory, where it equally does not exist — which is all the
workaround needs.

## Not in this client

- **No authentication.** The service ships unauthenticated: it declares no security scheme,
  implements no API-key or OAuth check, and reads no credential. This client sends none —
  deliberately, because an earlier revision of the service's own description offered an
  `X-API-Key` header that no code ever checked. Put it behind your gateway.
- **No `threshold` request parameter.** `/openapi.json` publishes `fields`, `top_k` and `explain`
  and nothing else. The server now ignores unrecognised top-level request keys rather than
  refusing them, so sending one would be silently discarded.
- **No client-side field caps.** `max_fields`, `max_batch_fields`, the lookup id-length bound and
  the string-length bounds are per-deployment settings; a copy of them compiled in here would
  refuse requests a tuned server accepts and go stale silently the first time an operator raised
  one. Read them from `status().limits()` and chunk against the answer. The two checks
  `LookupRequest` *does* make locally — a blank id and a duplicated id — are structural rather than
  configured: the response is a map keyed by the id, so no deployment can answer either, and
  mirroring them cannot go stale.
- **No absolute-score floor, and no suggestion of one.** `NO_MATCH` on a stock server means "this
  field came back with no candidates". Turning it into "rank 1 was not good enough" takes a floor,
  and the floor is yours to measure — see the fixture note above for what measuring it looks like
  and for why the fixture's own 0.65 is not a recommendation.
- **No opinion about `sourceMetadata`.** The pass-through plane is carried and never read. Nothing
  in this client branches on a key or a value in it, and nothing should: the moment it did, this
  artifact would be one specific enterprise's client wearing a generic name.
- **No feedback-driven improvement.** `submitFeedback` writes to an audit trail and nothing else.
  Training on that signal was measured on the server's own benchmark and *lost* accuracy, so no
  accuracy claim is made for it here either.
