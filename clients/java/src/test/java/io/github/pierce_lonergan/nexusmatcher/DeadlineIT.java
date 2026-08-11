package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.DeadlineExceededException;
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
 * The 504 path, against a server whose deadline is short enough that every match trips it.
 *
 * <p>The fixture is the same example pack served with {@code NEXUS_API_DEADLINE_SECONDS=0.001}. A
 * real match takes single-digit milliseconds, so the deadline fires every time and the response is
 * a genuine {@code DeadlineExceededError} from the server's own limiter.
 */
class DeadlineIT {

    private static String base;

    @BeforeAll
    static void connect() {
        base = LiveService.deadline();
    }

    private static List<FieldSpec> oneField() {
        return List.of(FieldSpec.of(
                "legal_name",
                "booking.passenger.legal_name",
                "Full legal name of the passenger as printed on the sailing manifest.",
                "string"));
    }

    @Test
    @DisplayName("the server answers 504 rather than hanging, and says what its budget was")
    void deadlineIsAnAnswerRatherThanAHang() {
        NexusMatcherClient client = NexusMatcherClient.builder(base)
                .timeout(Duration.ofSeconds(10))
                .build();

        DeadlineExceededException failure =
                assertThrows(DeadlineExceededException.class, () -> client.match(oneField()));

        assertEquals(504, failure.httpStatus());
        assertEquals("NEXUS-6002", failure.errorCode().orElseThrow());
        assertTrue(
                failure.deadlineSeconds().orElseThrow() > 0,
                "an operator reading a client's log can tell a too-tight server deadline from a "
                        + "genuinely slow match from this number alone");
        assertTrue(failure.requestId().isPresent());
    }

    @Test
    @DisplayName("the shipped policy does not retry a 504")
    void deadlineIsNotRetriedByDefault() {
        CopyOnWriteArrayList<Duration> waits = new CopyOnWriteArrayList<>();
        NexusMatcherClient client = NexusMatcherClient.builder(base)
                .maxRetries(3)
                .sleeper((duration, failure) -> waits.add(duration))
                .build();

        assertThrows(DeadlineExceededException.class, () -> client.match(oneField()));

        assertEquals(
                0,
                waits.size(),
                "the timed-out match is still running on the server and still holding its "
                        + "admission permit; an immediate retry adds a second copy of exactly the "
                        + "work that could not finish");
    }

    @Test
    @DisplayName("opting in retries a 504 exactly once, however high maxRetries is set")
    void optingInRetriesExactlyOnce() {
        CopyOnWriteArrayList<Duration> waits = new CopyOnWriteArrayList<>();
        NexusMatcherClient client = NexusMatcherClient.builder(base)
                .retryPolicy(ExponentialBackoffRetryPolicy.builder()
                        .maxRetries(5)
                        .retryDeadlineExceeded(true)
                        .build())
                .sleeper((duration, failure) -> waits.add(duration))
                .build();

        assertThrows(DeadlineExceededException.class, () -> client.match(oneField()));

        assertEquals(1, waits.size(), "one retry, and the cap is the failure's own, not the policy's");
    }

    @Test
    @DisplayName("a 504 still carries the caller's correlation id")
    void deadlineCarriesTheRequestId() {
        NexusMatcherClient client =
                NexusMatcherClient.builder(base).retryPolicy(RetryPolicy.none()).build();

        DeadlineExceededException failure = assertThrows(
                DeadlineExceededException.class,
                () -> client.match(
                        io.github.pierce_lonergan.nexusmatcher.model.MatchRequest.of(oneField()),
                        "java-client-deadline-0001"));

        assertEquals("java-client-deadline-0001", failure.requestId().orElseThrow());
    }
}
