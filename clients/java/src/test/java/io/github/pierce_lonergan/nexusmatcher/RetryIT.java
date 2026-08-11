package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.ServiceUnavailableException;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The retry loop, watched against a server that really answers 503.
 *
 * <p>The fixture is a nexus_matcher process started with no {@code NEXUS_API_DICTIONARY}, so every
 * match is a genuine {@code MatcherUnavailableError} rather than a status somebody stubbed. The
 * backoff is observed through the injected sleeper rather than spent, so the timing assertions are
 * exact and the suite does not pay for them.
 */
class RetryIT {

    private static String base;

    @BeforeAll
    static void connect() {
        base = LiveService.unavailable();
    }

    private static List<FieldSpec> oneField() {
        return List.of(FieldSpec.of("terminal_name", "t.name", "The terminal's public name.",
                "string"));
    }

    @Test
    @DisplayName("a 503 is retried up to the configured cap, then thrown")
    void serviceUnavailableIsRetriedThenThrown() {
        CopyOnWriteArrayList<Duration> waits = new CopyOnWriteArrayList<>();
        NexusMatcherClient client = NexusMatcherClient.builder(base)
                .maxRetries(2)
                .sleeper((duration, failure) -> waits.add(duration))
                .build();

        ServiceUnavailableException failure = assertThrows(
                ServiceUnavailableException.class, () -> client.match(oneField()));

        assertEquals(503, failure.httpStatus());
        assertEquals(
                2,
                waits.size(),
                "two retries after the first attempt, then the failure reaches the caller");
        assertTrue(
                waits.get(1).compareTo(waits.get(0)) > 0,
                "the second wait must be longer than the first: " + waits);
        assertTrue(failure.requestId().isPresent());
    }

    @Test
    @DisplayName("a 503 that will never clear says so, so a caller can stop backing off")
    void configurationProblemIsDistinguishable() {
        NexusMatcherClient client =
                NexusMatcherClient.builder(base).retryPolicy(RetryPolicy.none()).build();

        ServiceUnavailableException failure = assertThrows(
                ServiceUnavailableException.class, () -> client.match(oneField()));

        assertEquals("NEXUS-1002", failure.errorCode().orElseThrow());
        assertTrue(
                failure.isConfigurationProblem(),
                "this server has no dictionary and never will until an operator sets one; "
                        + "retrying it forever is a client burning a fallback window");
        assertTrue(
                failure.reason().orElseThrow().contains("NEXUS_API_DICTIONARY"),
                "and the message names the setting to change: " + failure.getMessage());
        assertTrue(failure.capacity().isEmpty(), "this is not shed load");
    }

    @Test
    @DisplayName("RetryPolicy.none() sends exactly one attempt")
    void retryingCanBeTurnedOff() {
        CopyOnWriteArrayList<Duration> waits = new CopyOnWriteArrayList<>();
        NexusMatcherClient client = NexusMatcherClient.builder(base)
                .retryPolicy(RetryPolicy.none())
                .sleeper((duration, failure) -> waits.add(duration))
                .build();

        assertThrows(ServiceUnavailableException.class, () -> client.match(oneField()));

        assertEquals(0, waits.size());
    }

    @Test
    @DisplayName("every attempt of one logical request carries the same correlation id")
    void oneRequestIdAcrossTheWholeRetrySequence() {
        CopyOnWriteArrayList<String> ids = new CopyOnWriteArrayList<>();
        NexusMatcherClient client = NexusMatcherClient.builder(base)
                .maxRetries(2)
                .sleeper((duration, failure) -> ids.add(failure.requestId().orElse("(none)")))
                .build();

        ServiceUnavailableException failure = assertThrows(
                ServiceUnavailableException.class,
                () -> client.match(
                        io.github.pierce_lonergan.nexusmatcher.model.MatchRequest.of(oneField()),
                        "java-client-retry-0001"));

        assertEquals(
                List.of("java-client-retry-0001", "java-client-retry-0001"),
                ids,
                "three attempts under one id read as one request in the server's log; three ids "
                        + "read as three unrelated failures");
        assertEquals("java-client-retry-0001", failure.requestId().orElseThrow());
    }

    @Test
    @DisplayName("feedback on a server with no feedback file is the same typed 503")
    void feedbackWithoutAFileIs503() {
        NexusMatcherClient client =
                NexusMatcherClient.builder(base).retryPolicy(RetryPolicy.none()).build();

        ServiceUnavailableException failure = assertThrows(
                ServiceUnavailableException.class,
                () -> client.submitFeedback(
                        io.github.pierce_lonergan.nexusmatcher.model.Feedback.of(
                                "t.name", "GBF-0027", true, "java-client-it",
                                java.time.Instant.parse("2026-08-11T09:00:00Z"))));

        assertEquals(
                "NEXUS_API_FEEDBACK_PATH",
                failure.setting().orElseThrow(),
                "the route exists and answers 503 naming the setting, rather than 404 -- which "
                        + "would tell a client the endpoint is not in this build");
    }
}
