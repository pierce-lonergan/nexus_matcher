package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The retry policy as arithmetic, with no HTTP anywhere near it.
 *
 * <p>The policy decides "again or not, and after how long" from a status and an attempt count, so
 * it can be tested exactly, with no clock and no service. Whether a real 503 actually reaches it is
 * a different claim, and {@code RetryIT} makes that one against a real server.
 */
class RetryPolicyTest {

    private static NexusMatcherException failure(int status) {
        return NexusMatcherException.of(status, "NEXUS-TEST", "test", Map.of(), "test-0001");
    }

    /** A jitter source pinned to the top of its range, so the delays below are exact. */
    private static Random fixedJitter() {
        return new Random() {
            @Override
            public double nextDouble() {
                return 1.0;
            }
        };
    }

    @Test
    @DisplayName("503 is retried, and the delay doubles")
    void backoffIsExponential() {
        RetryPolicy policy = ExponentialBackoffRetryPolicy.builder()
                .maxRetries(3)
                .baseDelay(Duration.ofMillis(100))
                .maxDelay(Duration.ofSeconds(10))
                .random(fixedJitter())
                .build();

        List<Long> delays = new ArrayList<>();
        for (int attempt = 1; attempt <= 4; attempt++) {
            policy.nextDelay(failure(503), attempt).ifPresent(d -> delays.add(d.toMillis()));
        }

        assertEquals(
                List.of(100L, 200L, 400L),
                delays,
                "three retries after the first attempt, doubling, then nothing");
    }

    @Test
    @DisplayName("the delay is capped, so a client never waits an unbounded time")
    void backoffIsCapped() {
        RetryPolicy policy = ExponentialBackoffRetryPolicy.builder()
                .maxRetries(10)
                .baseDelay(Duration.ofMillis(500))
                .maxDelay(Duration.ofSeconds(2))
                .random(fixedJitter())
                .build();

        for (int attempt = 1; attempt <= 10; attempt++) {
            Optional<Duration> delay = policy.nextDelay(failure(503), attempt);
            delay.ifPresent(d -> assertTrue(
                    d.compareTo(Duration.ofSeconds(2)) <= 0,
                    "no single delay may exceed maxDelay, got " + d));
        }
    }

    @Test
    @DisplayName("jitter spreads the retries, within half the computed delay and the whole of it")
    void jitterIsBoundedAndActuallyVaries() {
        RetryPolicy policy = ExponentialBackoffRetryPolicy.builder()
                .maxRetries(1)
                .baseDelay(Duration.ofMillis(1000))
                .maxDelay(Duration.ofSeconds(10))
                .random(new Random(20260811L))
                .build();

        List<Long> seen = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            long millis = policy.nextDelay(failure(503), 1).orElseThrow().toMillis();
            assertTrue(millis >= 500 && millis <= 1000, "equal jitter stays in [d/2, d]: " + millis);
            seen.add(millis);
        }
        assertTrue(
                seen.stream().distinct().count() > 50,
                "without real spread, every client shed by one overload retries in the same "
                        + "millisecond and the second wave is as synchronised as the first");
    }

    @Test
    @DisplayName("a 4xx is never retried, whatever the policy is configured to allow")
    void clientErrorsAreNeverRetried() {
        RetryPolicy policy =
                ExponentialBackoffRetryPolicy.builder().maxRetries(10).build();

        for (int status : new int[] {400, 413, 422}) {
            assertTrue(
                    policy.nextDelay(failure(status), 1).isEmpty(),
                    status + " must not be retried: the same request produces the same refusal");
        }
    }

    @Test
    @DisplayName("a 500 is not retried either: the server already said nothing was classified")
    void serverErrorsAreNotRetried() {
        RetryPolicy policy = ExponentialBackoffRetryPolicy.builder().maxRetries(5).build();
        assertTrue(policy.nextDelay(failure(500), 1).isEmpty());
    }

    @Test
    @DisplayName("504 is not retried by default, and at most once when opted in")
    void deadlineExceededIsOptInAndCappedAtOne() {
        RetryPolicy shipped = ExponentialBackoffRetryPolicy.builder().maxRetries(5).build();
        assertTrue(
                shipped.nextDelay(failure(504), 1).isEmpty(),
                "the server spent its deadline on work that is STILL RUNNING; an immediate retry "
                        + "doubles the load that caused the timeout");
        assertFalse(
                ((ExponentialBackoffRetryPolicy) shipped).retriesDeadlineExceeded());

        RetryPolicy optedIn = ExponentialBackoffRetryPolicy.builder()
                .maxRetries(5)
                .retryDeadlineExceeded(true)
                .build();
        assertTrue(optedIn.nextDelay(failure(504), 1).isPresent(), "one retry is allowed");
        assertTrue(
                optedIn.nextDelay(failure(504), 2).isEmpty(),
                "and only one, however high maxRetries is set");
    }

    @Test
    @DisplayName("none() disables retrying entirely")
    void noneNeverRetries() {
        assertTrue(RetryPolicy.none().nextDelay(failure(503), 1).isEmpty());
    }

    @Test
    @DisplayName("maxRetries(0) disables retrying")
    void zeroMaxRetriesDisablesRetrying() {
        RetryPolicy policy = ExponentialBackoffRetryPolicy.builder().maxRetries(0).build();
        assertTrue(policy.nextDelay(failure(503), 1).isEmpty());
    }

    @Test
    @DisplayName("the shipped default retries a 503 twice and nothing else")
    void shippedDefaults() {
        RetryPolicy policy = RetryPolicy.defaultPolicy();

        assertTrue(policy.nextDelay(failure(503), 1).isPresent());
        assertTrue(policy.nextDelay(failure(503), 2).isPresent());
        assertTrue(policy.nextDelay(failure(503), 3).isEmpty());
        assertTrue(policy.nextDelay(failure(504), 1).isEmpty());
        assertTrue(policy.nextDelay(failure(422), 1).isEmpty());
    }

    @Test
    @DisplayName("an impossible configuration is refused at build time, not under load")
    void configurationIsValidatedEagerly() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ExponentialBackoffRetryPolicy.builder().maxRetries(-1).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> ExponentialBackoffRetryPolicy.builder()
                        .baseDelay(Duration.ofSeconds(10))
                        .maxDelay(Duration.ofSeconds(1))
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> ExponentialBackoffRetryPolicy.builder().baseDelay(Duration.ZERO).build());
    }
}
