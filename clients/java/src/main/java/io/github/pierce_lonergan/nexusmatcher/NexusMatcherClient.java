package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherException;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherProtocolException;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherTransportException;
import io.github.pierce_lonergan.nexusmatcher.model.Feedback;
import io.github.pierce_lonergan.nexusmatcher.model.FeedbackReceipt;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.HealthStatus;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.Readiness;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Supplier;

/**
 * Java client for the nexus_matcher governance matching API.
 *
 * <p><strong>Transport and types, nothing else.</strong> The Python service is the one
 * implementation of the matching and governance semantics; this artifact carries requests there and
 * decodes what comes back. It does not re-decide a match, does not apply a threshold, and does not
 * rank two classifications -- for that last one it hands you
 * {@link io.github.pierce_lonergan.nexusmatcher.model.Vocabulary#tiersMostOpenFirst()} and stops.
 *
 * <pre>{@code
 * NexusMatcherClient client = NexusMatcherClient.builder("http://127.0.0.1:8000")
 *         .timeout(Duration.ofSeconds(30))
 *         .maxRetries(2)
 *         .build();
 *
 * MatchResponse response = client.match(List.of(
 *         FieldSpec.of("legal_name", "booking.passenger.legal_name",
 *                      "Full legal name as printed on the sailing manifest.", "string")));
 *
 * response.topCandidateFor("booking.passenger.legal_name").ifPresent(candidate -> {
 *     switch (candidate.governanceStatus()) {
 *         case CONFERRED -> apply(candidate.governance());
 *         case OPEN_TIER -> applyOpenTier(response.vocabulary().openClassification());
 *         case WITHHELD_REJECTED_TOP_MATCH -> sendToAHuman();
 *     }
 * });
 * }</pre>
 *
 * <p>Instances are immutable and safe to share between threads; the underlying
 * {@link HttpClient} is designed to be reused and building one per call is the usual way a JDK HTTP
 * integration ends up leaking threads.
 *
 * <p><strong>This service ships unauthenticated.</strong> It declares no security scheme and reads
 * no credential, so this client sends none -- deliberately, because an earlier revision of the
 * service's own description offered an API-key header that no code ever checked. Put it behind your
 * gateway.
 */
public final class NexusMatcherClient {

    private static final String MATCH_PATH = "/api/v1/match";
    private static final String MATCH_BATCH_PATH = "/api/v1/match/batch";
    private static final String FEEDBACK_PATH = "/api/v1/feedback";
    private static final String HEALTH_PATH = "/health";
    private static final String READY_PATH = "/health/ready";

    private static final String REQUEST_ID_HEADER = "X-Request-ID";
    private static final String RESPONSE_TIME_HEADER = "X-Response-Time-Ms";

    private final String baseUrl;
    private final HttpClient httpClient;
    private final ObjectMapper mapper;
    private final Duration timeout;
    private final RetryPolicy retryPolicy;
    private final Supplier<String> requestIdSupplier;
    private final Sleeper sleeper;

    private NexusMatcherClient(Builder builder) {
        this.baseUrl = builder.baseUrl;
        this.mapper = builder.mapper;
        this.timeout = builder.timeout;
        this.retryPolicy = builder.retryPolicy;
        this.requestIdSupplier = builder.requestIdSupplier;
        this.sleeper = builder.sleeper;
        this.httpClient = builder.httpClient != null
                ? builder.httpClient
                : HttpClient.newBuilder()
                        .connectTimeout(builder.connectTimeout)
                        // HTTP/1.1, not the JDK default of 2. The service is uvicorn behind
                        // whatever gateway an adopter puts in front of it, and the JDK's HTTP/2
                        // upgrade negotiation buys nothing here while adding a failure mode to
                        // debug. Nothing in this contract needs multiplexing.
                        .version(HttpClient.Version.HTTP_1_1)
                        .build();
    }

    /** Start building a client against a service base URL, e.g. {@code http://127.0.0.1:8000}. */
    public static Builder builder(String baseUrl) {
        return new Builder(baseUrl);
    }

    // ---------------------------------------------------------------------------------------
    // Matching
    // ---------------------------------------------------------------------------------------

    /**
     * Match fields, taking the server's defaults for {@code top_k} and {@code explain}.
     *
     * <p>Field cap {@code NEXUS_API_MAX_FIELDS} (100 by default). Over it the server answers 413
     * and {@link io.github.pierce_lonergan.nexusmatcher.error.PayloadTooLargeException#suggestedChunkSize()}
     * tells you what to re-chunk to; {@link #matchBatch(List)} has the higher cap.
     */
    public MatchResponse match(List<FieldSpec> fields) {
        return match(MatchRequest.of(fields));
    }

    /** Match fields with {@code top_k} and {@code explain} set. */
    public MatchResponse match(MatchRequest request) {
        return match(request, null);
    }

    /**
     * Match fields under a correlation id you choose.
     *
     * <p>The id you pass wins over a generated one and is echoed on the response, on every log line
     * the server writes for it, and on any exception -- which is the whole point of being allowed to
     * choose it.
     */
    public MatchResponse match(MatchRequest request, String requestId) {
        return postMatch(MATCH_PATH, request, requestId);
    }

    /**
     * Match a chunk of fields against the higher field cap: {@code NEXUS_API_MAX_BATCH_FIELDS},
     * 250 by default.
     *
     * <p>Identical contract to {@link #match(MatchRequest)} -- one implementation behind two caps,
     * so the semantics cannot drift apart.
     */
    public MatchResponse matchBatch(List<FieldSpec> fields) {
        return matchBatch(MatchRequest.of(fields));
    }

    /** Match a chunk of fields, with the knobs set. */
    public MatchResponse matchBatch(MatchRequest request) {
        return matchBatch(request, null);
    }

    /** Match a chunk of fields under a correlation id you choose. */
    public MatchResponse matchBatch(MatchRequest request, String requestId) {
        return postMatch(MATCH_BATCH_PATH, request, requestId);
    }

    private MatchResponse postMatch(String path, MatchRequest request, String requestId) {
        Objects.requireNonNull(request, "request");
        Exchange exchange = send("POST", path, request, requestId, 200);
        MatchResponse body = decode(exchange, MatchResponse.class);
        return body.withTransport(exchange.requestId(), exchange.responseTimeMs());
    }

    // ---------------------------------------------------------------------------------------
    // Feedback
    // ---------------------------------------------------------------------------------------

    /**
     * Record a reviewer's verdict in the server's append-only audit trail.
     *
     * <p>Returns the stored record rather than {@code void}: it carries the server's own
     * {@code receivedAt}, which is what the trail is ordered by and the only evidence the caller has
     * of what was actually written. The 201 comes back only after the append and its fsync have
     * returned, so a receipt in hand means the line is on disk.
     *
     * <p>This does <strong>not</strong> improve later matches. The server makes no such claim --
     * training on this signal was measured on its own benchmark and lost accuracy -- so treat it as
     * an audit trail, which is what it is.
     *
     * <p>Answers 503 when {@code NEXUS_API_FEEDBACK_PATH} is unset on the server, and 500 if the
     * append itself failed, in which case <em>nothing was recorded</em>: do not treat the verdict
     * as filed.
     */
    public FeedbackReceipt submitFeedback(Feedback feedback) {
        return submitFeedback(feedback, null);
    }

    /** Record a verdict under a correlation id you choose. */
    public FeedbackReceipt submitFeedback(Feedback feedback, String requestId) {
        Objects.requireNonNull(feedback, "feedback");
        Exchange exchange = send("POST", FEEDBACK_PATH, feedback, requestId, 201);
        return decode(exchange, FeedbackReceipt.class);
    }

    // ---------------------------------------------------------------------------------------
    // Health
    // ---------------------------------------------------------------------------------------

    /**
     * {@code GET /health}.
     *
     * <p>Cannot fail: it answers 200 with {@code status: "degraded"} when a component is red, so do
     * not gate a rollout on it. Use {@link #readiness()}.
     */
    public HealthStatus health() {
        return decode(send("GET", HEALTH_PATH, null, null, 200), HealthStatus.class);
    }

    /**
     * {@code GET /health/ready}, including when the answer is no.
     *
     * <p>A 503 here is not an exception, it is the answer: it comes back as a {@link Readiness} with
     * {@link Readiness#ready()} false and the component map the server put in the error body, so a
     * caller can see WHICH component is red without a try/catch. Anything else -- unreachable, a
     * 500, an unreadable body -- still throws.
     */
    public Readiness readiness() {
        Exchange exchange = send("GET", READY_PATH, null, null, 200, 503);
        if (exchange.status() == 503) {
            Map<String, Object> details = errorDetails(exchange);
            Object components = details.get("components");
            Map<String, Boolean> map = components instanceof Map<?, ?>
                    ? mapper.convertValue(components, new TypeReference<Map<String, Boolean>>() {})
                    : Map.of();
            return new Readiness(false, null, map);
        }
        return decode(exchange, Readiness.class);
    }

    /** The base URL this client talks to, without a trailing slash. */
    public String baseUrl() {
        return baseUrl;
    }

    // ---------------------------------------------------------------------------------------
    // The exchange
    // ---------------------------------------------------------------------------------------

    private Exchange send(
            String method,
            String path,
            Object body,
            String callerRequestId,
            int... acceptedStatuses) {

        // One id for the whole retry sequence, not one per attempt. Every attempt of the same
        // logical request lands in the server log under the same id, which is what makes "this
        // succeeded on the third try" readable there rather than three unrelated lines.
        String requestId = callerRequestId != null ? callerRequestId : requestIdSupplier.get();
        byte[] payload = body == null ? null : serialise(body, requestId);

        int attemptsMade = 0;
        while (true) {
            attemptsMade++;
            try {
                return attempt(method, path, payload, requestId, acceptedStatuses);
            } catch (NexusMatcherException failure) {
                Optional<Duration> delay = retryPolicy.nextDelay(failure, attemptsMade);
                if (delay.isEmpty()) {
                    throw failure;
                }
                sleeper.sleep(delay.get(), failure);
            }
        }
    }

    private Exchange attempt(
            String method,
            String path,
            byte[] payload,
            String requestId,
            int[] acceptedStatuses) {

        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(timeout)
                .header("Accept", "application/json")
                .header(REQUEST_ID_HEADER, requestId);

        if (payload == null) {
            builder.method(method, HttpRequest.BodyPublishers.noBody());
        } else {
            builder.header("Content-Type", "application/json")
                    .method(method, HttpRequest.BodyPublishers.ofByteArray(payload));
            // DO NOT add `.expectContinue(true)` here. It is the obvious fix for the byte-cap race
            // documented on RequestFailureIT#byteCapIsRefusedReadably, it does remove that race
            // (measured 0 losses in 40 against 5 in 40 without it), and against THIS server on JDK
            // 17 it then hangs: the server answers the 413 from Content-Length alone and closes
            // without ever sending `100 Continue`, the JDK is left holding a body it was told to
            // wait to send, and `HttpClient.send` never returns. The request timeout does not fire
            // -- measured at 654 s on a 30 s timeout, and three bounded runs at 180 s each timed
            // out rather than failing. A hang defeats every control an adopter has (the connect
            // succeeded, the read never returned, no fallback fires), which is the exact failure
            // the service's own errors.py was written to prevent. The server side has since
            // changed -- it drains a refused body within twice its cap, so the 413 survives --
            // which removes the reason anyone would reach for expectContinue. The hang it
            // causes is unchanged, so this note stays.

        }

        HttpResponse<String> response;
        try {
            response = httpClient.send(
                    builder.build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException exc) {
            throw new NexusMatcherTransportException(
                    "no response from " + baseUrl + path + ": " + exc, requestId, exc);
        } catch (InterruptedException exc) {
            // Restore the flag. Swallowing it leaves a thread that has been asked to stop looking
            // to every layer above like one that has not.
            Thread.currentThread().interrupt();
            throw new NexusMatcherTransportException(
                    "interrupted while calling " + baseUrl + path, requestId, exc);
        }

        // The server's own id wins if it sent one back -- it does, on every status including the
        // 413 its outermost middleware answers -- because that is the id in its logs.
        String effectiveId = response.headers().firstValue(REQUEST_ID_HEADER).orElse(requestId);
        Double responseTimeMs = response.headers()
                .firstValue(RESPONSE_TIME_HEADER)
                .map(NexusMatcherClient::parseDouble)
                .orElse(null);

        Exchange exchange =
                new Exchange(response.statusCode(), response.body(), effectiveId, responseTimeMs);

        for (int accepted : acceptedStatuses) {
            if (exchange.status() == accepted) {
                return exchange;
            }
        }
        throw toException(exchange);
    }

    private byte[] serialise(Object body, String requestId) {
        try {
            return mapper.writeValueAsBytes(body);
        } catch (IOException exc) {
            throw new NexusMatcherProtocolException(
                    "could not serialise " + body.getClass().getSimpleName(), 0, requestId, exc);
        }
    }

    private <T> T decode(Exchange exchange, Class<T> type) {
        try {
            return mapper.readValue(exchange.body(), type);
        } catch (IOException | IllegalArgumentException exc) {
            // IllegalArgumentException covers a MatchDecision this client does not know: Jackson
            // wraps the enum's own refusal, and it reaches here as a value it cannot bind. Either
            // way the client and the service disagree about what they are speaking.
            throw new NexusMatcherProtocolException(
                    "could not read a " + type.getSimpleName() + " from the response: " + exc,
                    exchange.status(),
                    exchange.requestId(),
                    exc);
        }
    }

    /**
     * Turn a non-accepted response into the right typed exception.
     *
     * <p>Every documented failure arrives in one envelope, so this reads one shape. When it cannot
     * -- a gateway's own HTML 502, a truncated body -- the status still decides the class, because
     * the status alone is enough to know whether to retry, and losing that because a proxy answered
     * in HTML would be the worst possible trade.
     */
    private NexusMatcherException toException(Exchange exchange) {
        String code = null;
        String message = null;
        Map<String, Object> details = Map.of();

        JsonNode error = errorNode(exchange);
        if (error != null) {
            code = text(error, "code");
            message = text(error, "message");
            JsonNode detailNode = error.get("details");
            if (detailNode != null && detailNode.isObject()) {
                details = mapper.convertValue(
                        detailNode, new TypeReference<Map<String, Object>>() {});
            }
        }
        if (message == null || message.isEmpty()) {
            message = "HTTP " + exchange.status() + " from the matching service: "
                    + snippet(exchange.body());
        }
        return NexusMatcherException.of(
                exchange.status(), code, message, details, exchange.requestId());
    }

    private Map<String, Object> errorDetails(Exchange exchange) {
        JsonNode error = errorNode(exchange);
        JsonNode details = error == null ? null : error.get("details");
        if (details == null || !details.isObject()) {
            return Map.of();
        }
        return mapper.convertValue(details, new TypeReference<Map<String, Object>>() {});
    }

    private JsonNode errorNode(Exchange exchange) {
        if (exchange.body() == null || exchange.body().isEmpty()) {
            return null;
        }
        try {
            JsonNode root = mapper.readTree(exchange.body());
            JsonNode error = root.get("error");
            return error != null && error.isObject() ? error : null;
        } catch (IOException exc) {
            return null;
        }
    }

    private static String text(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }

    private static String snippet(String body) {
        if (body == null || body.isEmpty()) {
            return "(empty body)";
        }
        return body.length() <= 300 ? body : body.substring(0, 300) + "...";
    }

    private static Double parseDouble(String raw) {
        try {
            return Double.valueOf(raw);
        } catch (NumberFormatException exc) {
            return null;
        }
    }

    /** One HTTP round trip, reduced to the four things anything downstream needs. */
    private record Exchange(int status, String body, String requestId, Double responseTimeMs) {
    }

    /**
     * The wait between retries. An interface only so a test can run the backoff without spending
     * the wall-clock time it describes; there is one real implementation.
     */
    interface Sleeper {
        void sleep(Duration duration, NexusMatcherException failure);

        static Sleeper real() {
            return (duration, failure) -> {
                try {
                    Thread.sleep(duration.toMillis());
                } catch (InterruptedException exc) {
                    Thread.currentThread().interrupt();
                    throw new NexusMatcherTransportException(
                            "interrupted while backing off after " + failure.httpStatus(),
                            failure.requestId().orElse(null),
                            exc);
                }
            };
        }
    }

    /** Builder for {@link NexusMatcherClient}. */
    public static final class Builder {
        private final String baseUrl;
        private Duration timeout = Duration.ofSeconds(30);
        private Duration connectTimeout = Duration.ofSeconds(5);
        private RetryPolicy retryPolicy = RetryPolicy.defaultPolicy();
        private ObjectMapper mapper = defaultMapper();
        private HttpClient httpClient;
        private Sleeper sleeper = Sleeper.real();
        private Supplier<String> requestIdSupplier =
                () -> UUID.randomUUID().toString().substring(0, 8);

        private Builder(String baseUrl) {
            Objects.requireNonNull(baseUrl, "baseUrl");
            String trimmed = baseUrl.strip();
            if (trimmed.isEmpty()) {
                throw new IllegalArgumentException("baseUrl must not be blank");
            }
            while (trimmed.endsWith("/")) {
                trimmed = trimmed.substring(0, trimmed.length() - 1);
            }
            this.baseUrl = trimmed;
        }

        /**
         * How long to wait for one response before giving up locally.
         *
         * <p>Keep this ABOVE the server's own deadline ({@code NEXUS_API_DEADLINE_SECONDS}, 25 s by
         * default). The server's deadline exists so that a slow match ends in a 504 rather than a
         * hang; if this timeout is the shorter one, the client always gives up first and never sees
         * the 504 -- which is the hang the whole arrangement was built to avoid, reintroduced by an
         * off-by-one in seconds. Default 30 s, matching the adopter's read timeout the server was
         * calibrated against.
         */
        public Builder timeout(Duration value) {
            this.timeout = requirePositive(value, "timeout");
            return this;
        }

        /** How long to wait for the TCP connection. Default 5 s. */
        public Builder connectTimeout(Duration value) {
            this.connectTimeout = requirePositive(value, "connectTimeout");
            return this;
        }

        /**
         * Retries for a 503, with the shipped backoff. Convenience for
         * {@code retryPolicy(ExponentialBackoffRetryPolicy.builder().maxRetries(n).build())}, and it
         * REPLACES any policy already set. 0 disables retrying.
         */
        public Builder maxRetries(int value) {
            this.retryPolicy = ExponentialBackoffRetryPolicy.builder().maxRetries(value).build();
            return this;
        }

        /** The whole retry policy. {@link RetryPolicy#none()} turns retrying off. */
        public Builder retryPolicy(RetryPolicy value) {
            this.retryPolicy = Objects.requireNonNull(value, "retryPolicy");
            return this;
        }

        /**
         * Where correlation ids come from when the caller does not supply one per call. Default is
         * eight hex characters of a UUID -- the same recipe the server uses when it mints one, so
         * ids from both ends look alike in one log.
         */
        public Builder requestIdSupplier(Supplier<String> value) {
            this.requestIdSupplier = Objects.requireNonNull(value, "requestIdSupplier");
            return this;
        }

        /**
         * A Jackson mapper of your own.
         *
         * <p>Whatever you pass, unknown properties must not be fatal: this service adds response
         * keys additively -- it gained two while this client was being written -- and a client that
         * throws on the next one turns an additive server change into an outage. The default mapper
         * has {@code FAIL_ON_UNKNOWN_PROPERTIES} off, and every DTO also carries
         * {@code @JsonIgnoreProperties(ignoreUnknown = true)} so that a stricter mapper still works.
         */
        public Builder objectMapper(ObjectMapper value) {
            this.mapper = Objects.requireNonNull(value, "objectMapper");
            return this;
        }

        /** An {@link HttpClient} of your own -- a proxy, an executor, a configured SSL context. */
        public Builder httpClient(HttpClient value) {
            this.httpClient = Objects.requireNonNull(value, "httpClient");
            return this;
        }

        /** Test seam: run the backoff without spending it. */
        Builder sleeper(Sleeper value) {
            this.sleeper = Objects.requireNonNull(value, "sleeper");
            return this;
        }

        public NexusMatcherClient build() {
            return new NexusMatcherClient(this);
        }

        private static ObjectMapper defaultMapper() {
            return new ObjectMapper()
                    .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        }

        private static Duration requirePositive(Duration value, String name) {
            Objects.requireNonNull(value, name);
            if (value.isNegative() || value.isZero()) {
                throw new IllegalArgumentException(name + " must be > 0, got " + value);
            }
            return value;
        }
    }
}
