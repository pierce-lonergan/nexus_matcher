package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherRequestException;
import io.github.pierce_lonergan.nexusmatcher.model.Feedback;
import io.github.pierce_lonergan.nexusmatcher.model.FeedbackReceipt;
import io.github.pierce_lonergan.nexusmatcher.model.ReviewVerdict;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
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
                clientTimestamp,
                null);

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
    @DisplayName("an unchanged pre-widening body now stores a ninth key, and it is verdict: null")
    void anUnchangedBodyStoresAnExplicitNullVerdict() {
        // The SAME shape a build before `verdict` existed sent. It is still 201 and still stores
        // the same eight values -- but the stored record, and the appended trail line, now carry
        // a ninth key. Additive and defensible; not "nothing changed for an unchanged request".
        // A trail-consuming script asserting an exact key set breaks on it, so it is asserted
        // here against a live service and recorded in CHANGELOG.md.
        FeedbackReceipt receipt = client.submitFeedback(
                Feedback.of(
                        "vessel.telemetry.fuel_level_pct",
                        "GBF-0021",
                        true,
                        "java-client-it",
                        Instant.parse("2026-08-11T09:20:00Z")));

        assertTrue(receipt.recorded());
        assertTrue(
                receipt.record().containsKey("verdict"),
                "the key is written with a null value rather than omitted, so a record predating "
                        + "the member and a reviewer who gave none read identically");
        assertNull(receipt.record().get("verdict"));
        assertTrue(receipt.storedVerdict().isEmpty());
        assertEquals(
                9,
                receipt.record().size(),
                "eight values plus the appended verdict. Pinned deliberately: this is the "
                        + "audit format, and the CHANGELOG says a consumer asserting an exact "
                        + "key set breaks when it grows -- so THIS is the place that is "
                        + "supposed to go red when it does, rather than a stranger's script.");
    }

    @Test
    @DisplayName("MANUAL_OVERRIDE round-trips: the verdict a boolean could not hold")
    void manualOverrideRoundTrips() {
        FeedbackReceipt receipt = client.submitFeedback(
                Feedback.of(
                        "galley.stock.reorder_quantity",
                        "GBF-0030",
                        true,
                        "java-client-it",
                        Instant.parse("2026-08-11T09:25:00Z"))
                        .withVerdict(ReviewVerdict.MANUAL_OVERRIDE));

        assertTrue(receipt.recorded());
        assertTrue(receipt.storedVerdict().orElseThrow().isManualOverride());
        assertEquals(
                Boolean.FALSE,
                receipt.record().get("wasCorrect"),
                "withVerdict set the boolean the verdict requires; the server refuses the other "
                        + "pairing rather than reconciling it");
    }

    // A contradictory verdict cannot be SENT from here: the record refuses to build one, which is
    // the point of the refusal. So the check that the client's copy of that rule still matches the
    // server's own lives where it can be made -- tests/packaging/test_java_client_contract.py asks
    // the real server model which pairs it accepts and which it refuses, in both directions.

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
                        "2026-08-11T09:00:00Z", null)));

        assertEquals(422, failure.httpStatus());
        assertTrue(
                failure.violations().stream()
                        .anyMatch(violation -> violation.location().contains("reviewer")),
                "the body names the offending field: " + failure.violations());
    }
}
