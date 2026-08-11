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

Nothing in your controlled vocabulary is typed as a Java enum. `code`, `name`, `classification`
and `enhancement` are open `String`s and will stay open. The one enum here is `MatchDecision`,
which is the library's own vocabulary and is published as a closed set in `/openapi.json`.

## Using it

```java
NexusMatcherClient client = NexusMatcherClient.builder("http://127.0.0.1:8000")
        .timeout(Duration.ofSeconds(30))   // keep ABOVE the server's own deadline
        .maxRetries(2)                     // 503 only
        .build();

MatchResponse response = client.match(List.of(
        FieldSpec.of("legal_name", "booking.passenger.legal_name",
                     "Full legal name as printed on the sailing manifest.", "string")));

MatchCandidate top = response.topCandidateFor("booking.passenger.legal_name").orElseThrow();
switch (top.governanceStatus()) {
    case CONFERRED                   -> apply(top.governance());
    case OPEN_TIER                   -> applyOpenTier(response.vocabulary().openClassification());
    case WITHHELD_REJECTED_TOP_MATCH -> sendToAHuman();
}
```

The client is immutable and thread-safe; build one and share it.

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

Unknown *values* in a closed set are refused loudly. A `decision` this client does not know
decides whether a class is applied without a human, and mapping it onto the nearest known one
would be a silent lie.

## Building and testing

```
mvn test        # 38 unit tests. No service, no network
mvn verify      # + 33 integration tests against a REAL service
```

`mvn verify` needs three running fixtures, because three of the behaviours worth pinning are
properties of a server's configuration rather than of a request:

| | | |
|---|---|---|
| `http://127.0.0.1:8000` | the example pack, loaded | everything normal |
| `http://127.0.0.1:8001` | no dictionary at all | every match is a real 503 |
| `http://127.0.0.1:8002` | the pack, `NEXUS_API_DEADLINE_SECONDS=0.001` | every match is a real 504 |

From the repository root:

```
./clients/java/serve-fixtures.sh        # or .\clients\java\serve-fixtures.ps1
```

Override the URLs with `-Dnexus.matcher.baseUrl=`, `-Dnexus.matcher.unavailableBaseUrl=`,
`-Dnexus.matcher.deadlineBaseUrl=`.

**The integration tests do not mock the service and do not skip when it is missing.** A skipped
integration test is a green build that proved nothing, and this contract moved twice while the
client was being written — exactly the drift a mock cannot see. A missing fixture fails the build
and names the script above.

The unit tests decode response bodies **captured verbatim** from a running service, in
`src/test/resources/captured/`. They are captures rather than hand-written expectations, which is
the difference between testing the contract and testing what the author believed it was.

### One case the shipped configuration cannot reach — and how to reach it anyway

`WITHHELD_REJECTED_TOP_MATCH` — a rejected rank-1 candidate — is unreachable against the **shipped
thresholds**. `final_confidence` has a structural floor of `semantic_weight × fusion_alpha` = 0.63
while `review_threshold` is 0.50, so no top match can fall below the bar that would reject it. The
server says as much in `domain/models/entities.py`, calling the clause latent and reachable for a
caller who raises `review_threshold` past that floor.

That does **not** mean it takes somebody else's deployment to see. `review_threshold` is settable
on a local fixture through `NEXUS_API_MATCHING_CONFIG`, and the case has been provoked and the
client's mapping confirmed end to end against a real server:

```bash
cat > /tmp/reject.json <<'JSON'
{"auto_approve_threshold": 0.995, "review_threshold": 0.99,
 "min_confidence_gap": 0.10, "results_per_field": 5}
JSON

NEXUS_API_DICTIONARY=examples/governance/glossary.csv \
NEXUS_API_GOVERNANCE=examples/governance/protection_classes.json \
NEXUS_API_MATCHING_CONFIG=/tmp/reject.json \
  .venv/Scripts/python -m uvicorn nexus_matcher.presentation.api.app:create_app \
  --factory --host 127.0.0.1 --port 8003
```

Matching `terminal_name` at `top_k=3` against that server returns, and the client reads it as:

| rank | decision | confidence | `governanceStatus()` | class |
|---|---|---|---|---|
| 1 | `REJECT` | 0.9173 | `WITHHELD_REJECTED_TOP_MATCH` | *withheld* |
| 2 | `REJECT` | 0.5375 | `CONFERRED` | `SEALED_RESTRICTED` |
| 3 | `REJECT` | 0.4253 | `CONFERRED` | `CREW_ONLY` |

That is the whole rule in one response: a rejected **rank 1** confers nothing, a rejected
**runner-up** keeps its class, and the two nulls stay distinguishable.

The shipped suite does not start that fourth fixture. `GovernanceNullsIT` instead asserts against
the live pack that no rank-1 candidate is `REJECT`, and fails with an explanation if that ever
stops being true; `MatchResponseDecodingTest` pins the client's decoding from a body labelled in
place as hand-built. If you are changing anything in this area, run the fixture above rather than
trusting either — it is ten seconds of work and it exercises the real server.

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
- **No client-side field caps.** `max_fields`, `max_batch_fields` and the string-length bounds are
  per-deployment settings; a copy of them compiled in here would refuse requests a tuned server
  accepts and go stale silently the first time an operator raised one.
- **No feedback-driven improvement.** `submitFeedback` writes to an audit trail and nothing else.
  Training on that signal was measured on the server's own benchmark and *lost* accuracy, so no
  accuracy claim is made for it here either.
