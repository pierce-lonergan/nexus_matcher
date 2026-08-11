package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherRequestException;
import io.github.pierce_lonergan.nexusmatcher.model.Feedback;
import io.github.pierce_lonergan.nexusmatcher.model.FeedbackReceipt;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Recording a reviewer's verdict, against a running service with a feedback file configured. */
class FeedbackIT {

    private static NexusMatcherClient client;

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
    }

    @Test
    @DisplayName("a verdict is recorded and echoed back with the server's own receivedAt")
    void verdictIsRecorded() {
        String clientTimestamp = "2026-08-11T09:00:00Z";
        Feedback feedback = new Feedback(
                "booking.passenger.legal_name",
                "Full legal name of the passenger as printed on the sailing manifest.",
                "GBF-0001",
                "GBF-0002",
                true,
                "java-client-it",
                clientTimestamp);

        FeedbackReceipt receipt = client.submitFeedback(feedback);

        assertTrue(receipt.recorded());
        assertEquals("booking.passenger.legal_name", receipt.storedField().orElseThrow());
        assertEquals("GBF-0001", receipt.record().get("chosenGovernanceId"));
        assertEquals("GBF-0002", receipt.record().get("suggestedGovernanceId"));
        assertEquals(clientTimestamp, receipt.record().get("ts"));

        String receivedAt = receipt.receivedAt().orElseThrow();
        assertNotEquals(
                clientTimestamp,
                receivedAt,
                "the server stamps its own arrival time and that is the field to order the audit "
                        + "trail by; a client clock is not evidence about when a review happened");
        assertTrue(receivedAt.startsWith("20"));
    }

    @Test
    @DisplayName("an Instant-stamped verdict round-trips")
    void instantStampedVerdictRoundTrips() {
        Instant now = Instant.parse("2026-08-11T09:15:30Z");

        FeedbackReceipt receipt = client.submitFeedback(
                Feedback.of("sailing.route_code", "GBF-0028", false, "java-client-it", now)
                        .withDoc("Short code identifying a scheduled route."));

        assertTrue(receipt.recorded());
        assertEquals(now.toString(), receipt.record().get("ts"));
        assertEquals(Boolean.FALSE, receipt.record().get("wasCorrect"));
    }

    @Test
    @DisplayName("a malformed verdict is a 422 naming what is missing, and is not retried")
    void malformedVerdictIs422() {
        // Built by hand rather than through the record's own constructor: the record already
        // refuses a null reviewer locally, so the only way to see the SERVER's refusal is to send
        // a body it will reject for a different reason. An empty reviewer passes the client's
        // null check and fails the server's min_length.
        NexusMatcherRequestException failure = assertThrows(
                NexusMatcherRequestException.class,
                () -> client.submitFeedback(new Feedback(
                        "sailing.route_code", null, "GBF-0028", null, true, "",
                        "2026-08-11T09:00:00Z")));

        assertEquals(422, failure.httpStatus());
        assertTrue(
                failure.violations().stream()
                        .anyMatch(violation -> violation.location().contains("reviewer")),
                "the body names the offending field: " + failure.violations());
    }
}
