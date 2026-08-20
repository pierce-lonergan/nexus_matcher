package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.Feedback;
import io.github.pierce_lonergan.nexusmatcher.model.FeedbackReceipt;
import io.github.pierce_lonergan.nexusmatcher.model.ReviewDecision;
import io.github.pierce_lonergan.nexusmatcher.model.ReviewVerdict;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code verdict}: the member a boolean could not hold, and the open binding it is given.
 *
 * <p>The service publishes this vocabulary INLINE rather than as a schema component -- three values
 * on the property, no named type -- which is the server declining to hand generated clients a
 * closed enum that breaks on a fourth value. The tests below are what makes that decision real on
 * this side of the wire rather than a comment claiming it is:
 * {@link #anUnknownVerdictFromANewerServerDoesNotBreakThisBuild()} is the one that would fail if
 * {@link ReviewVerdict} were ever bound closed.
 */
class ReviewVerdictTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private static Feedback base(boolean wasCorrect) {
        return Feedback.of(
                "sailing.route_code",
                "GBF-0028",
                wasCorrect,
                "a.reviewer",
                Instant.parse("2026-08-11T09:00:00Z"));
    }

    // =============================================================================
    // THE OPEN BINDING
    // =============================================================================

    @Test
    @DisplayName("an unknown verdict from a newer server does not break this build")
    void anUnknownVerdictFromANewerServerDoesNotBreakThisBuild() {
        // Deliberately NOT a captured fixture, and it cannot be one: no server that exists today
        // can produce this value, which is exactly why the behaviour needs a test. The literal is
        // a hypothetical fourth verdict, not a copy of anything the service publishes.
        ReviewVerdict decoded = ReviewVerdict.fromWire("APPROVED_WITH_CONDITIONS");

        assertEquals(ReviewDecision.UNKNOWN, decoded.decision());
        assertFalse(decoded.isKnown());
        assertEquals(
                "APPROVED_WITH_CONDITIONS",
                decoded.wireValue(),
                "the value has to survive, or an operator can count unknown verdicts but cannot "
                        + "name one in a ticket");
        assertFalse(
                decoded.isManualOverride(),
                "an unknown value must never be answered as one of the values this build knows; "
                        + "silently mapping onto the nearest known verdict is the failure the "
                        + "sentinel exists to prevent");
    }

    @Test
    @DisplayName("the three published values decode to themselves")
    void thePublishedValuesDecode() {
        assertEquals(ReviewDecision.APPROVED, ReviewVerdict.fromWire("APPROVED").decision());
        assertEquals(ReviewDecision.REJECTED, ReviewVerdict.fromWire("REJECTED").decision());
        assertEquals(
                ReviewDecision.MANUAL_OVERRIDE,
                ReviewVerdict.fromWire("MANUAL_OVERRIDE").decision());
        assertTrue(ReviewVerdict.fromWire("MANUAL_OVERRIDE").isManualOverride());
    }

    @Test
    @DisplayName("the literal string UNKNOWN from a server is a value, not this client's sentinel")
    void theLiteralUnknownIsNotMatchedByName() {
        ReviewVerdict decoded = ReviewVerdict.fromWire("UNKNOWN");

        assertEquals(ReviewDecision.UNKNOWN, decoded.decision());
        assertEquals(
                "UNKNOWN",
                decoded.wireValue(),
                "a server that started publishing the literal UNKNOWN would have it read as "
                        + "'this client did not understand you', which is a different claim. The "
                        + "raw value is kept so the two are still distinguishable, and "
                        + "tests/packaging/test_java_client_contract.py fails if the service ever "
                        + "publishes it");
    }

    @Test
    @DisplayName("a verdict serialises back to the string it arrived as, known or not")
    void aVerdictSerialisesToItsWireString() throws Exception {
        assertEquals(
                "\"MANUAL_OVERRIDE\"",
                mapper.writeValueAsString(ReviewVerdict.MANUAL_OVERRIDE));
        assertEquals(
                "\"A_FUTURE_VERDICT\"",
                mapper.writeValueAsString(ReviewVerdict.fromWire("A_FUTURE_VERDICT")),
                "a decoded response must re-encode to what the server said, not to what this "
                        + "build could name");
    }

    @Test
    @DisplayName("the sentinel cannot be sent, because the string UNKNOWN would be a 422")
    void theSentinelCannotBeSent() {
        assertThrows(
                IllegalArgumentException.class, () -> ReviewVerdict.of(ReviewDecision.UNKNOWN));
        assertThrows(
                IllegalArgumentException.class,
                () -> base(false).withVerdict(new ReviewVerdict(ReviewDecision.UNKNOWN, "X")));
    }

    // =============================================================================
    // THE AGREEMENT RULE, REFUSED LOCALLY
    // =============================================================================

    @Test
    @DisplayName("a verdict that contradicts wasCorrect is refused here, not discovered as a 422")
    void aContradictoryVerdictIsRefusedLocally() {
        // Structural, not configurable: no deployment can record a trail line that argues with
        // itself, so there is nothing to be gained by finding this out over the network -- and a
        // 422 on this route costs the reviewer the verdict they just gave.
        // tests/packaging/test_java_client_contract.py pins this rule against the real server
        // model, in both directions, so the copy cannot drift from the original.
        assertThrows(
                IllegalArgumentException.class,
                () -> new Feedback(
                        "t.a", null, "GBF-0001", null, false, "r", "2026-08-11T09:00:00Z",
                        ReviewVerdict.APPROVED));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Feedback(
                        "t.a", null, "GBF-0001", null, true, "r", "2026-08-11T09:00:00Z",
                        ReviewVerdict.MANUAL_OVERRIDE));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Feedback(
                        "t.a", null, "GBF-0001", null, true, "r", "2026-08-11T09:00:00Z",
                        ReviewVerdict.REJECTED));
    }

    @Test
    @DisplayName("withVerdict sets the boolean from the verdict rather than letting them disagree")
    void withVerdictSetsTheBooleanItRequires() {
        Feedback approved = base(false).withVerdict(ReviewVerdict.APPROVED);
        assertTrue(approved.wasCorrect());
        assertEquals(ReviewDecision.APPROVED, approved.verdict().decision());

        Feedback overridden = base(true).withVerdict(ReviewVerdict.MANUAL_OVERRIDE);
        assertFalse(overridden.wasCorrect());
        assertTrue(overridden.verdict().isManualOverride());

        Feedback cleared = overridden.withVerdict(null);
        assertTrue(cleared.verdictValue().isEmpty());
        assertFalse(
                cleared.wasCorrect(),
                "clearing the verdict leaves the boolean alone: it is not derived from the "
                        + "verdict and was never deprecated");
    }

    @Test
    @DisplayName("every published pairing is accepted, so this refusal is not simply a ban")
    void everyPublishedPairingIsAccepted() {
        assertEquals(
                ReviewDecision.APPROVED, base(true).withVerdict(ReviewVerdict.APPROVED)
                        .verdict().decision());
        assertEquals(
                ReviewDecision.REJECTED, base(true).withVerdict(ReviewVerdict.REJECTED)
                        .verdict().decision());
        assertEquals(
                ReviewDecision.MANUAL_OVERRIDE,
                base(true).withVerdict(ReviewVerdict.MANUAL_OVERRIDE).verdict().decision());
    }

    // =============================================================================
    // ON THE WIRE
    // =============================================================================

    @Test
    @DisplayName("an absent verdict is omitted, and a present one is the last key")
    void theVerdictIsOmittedWhenAbsentAndAppendedWhenPresent() throws Exception {
        JsonNode without = mapper.valueToTree(base(true));
        assertFalse(
                without.has("verdict"),
                "an absent optional is omitted, never sent as an explicit null: the server writes "
                        + "the null into the record itself, and a client that sends one is "
                        + "asserting a shape it was not asked for");

        JsonNode with = mapper.valueToTree(base(false).withVerdict(ReviewVerdict.MANUAL_OVERRIDE));
        List<String> keys = new ArrayList<>();
        with.fieldNames().forEachRemaining(keys::add);
        assertEquals(
                "verdict",
                keys.get(keys.size() - 1),
                "appended rather than placed beside wasCorrect, matching the server's own record "
                        + "order, so a trail spanning an upgrade diffs cleanly");
        assertEquals("MANUAL_OVERRIDE", with.get("verdict").asText());
        assertFalse(with.get("wasCorrect").asBoolean());
    }

    // =============================================================================
    // THE RECEIPT
    // =============================================================================

    @Test
    @DisplayName("the stored record carries the verdict, decoded")
    void theStoredVerdictIsReadable() throws Exception {
        FeedbackReceipt receipt = mapper.readValue(
                Fixtures.captured("feedback-receipt-verdict.json"), FeedbackReceipt.class);

        assertTrue(receipt.recorded());
        assertEquals("sailing.route_code", receipt.storedField().orElseThrow());
        assertTrue(receipt.storedVerdict().orElseThrow().isManualOverride());
        assertEquals("MANUAL_OVERRIDE", receipt.storedVerdict().orElseThrow().wireValue());
    }

    @Test
    @DisplayName("a request that sent no verdict is stored with the key present and null")
    void anUnsentVerdictIsStoredAsAnExplicitNull() throws Exception {
        // This capture is the SAME request body a build before `verdict` existed would have sent,
        // and the record it comes back with is no longer the same: it carries a ninth key. The
        // response and the appended trail line both gained `"verdict": null`. That is additive,
        // and a tolerant reader is fine -- but a trail-consuming script asserting an exact key set
        // is not, which is why the fact is asserted here and recorded in CHANGELOG.md rather than
        // reported as "nothing changed for an unchanged request".
        FeedbackReceipt receipt = mapper.readValue(
                Fixtures.captured("feedback-receipt.json"), FeedbackReceipt.class);

        assertTrue(
                receipt.record().containsKey("verdict"),
                "the server writes the key with a null value rather than omitting it");
        assertTrue(
                receipt.storedVerdict().isEmpty(),
                "empty means the reviewer gave no verdict, which is what every record written "
                        + "before the member existed reads as");
        assertEquals(
                9,
                receipt.record().size(),
                "eight values plus the appended verdict. Pinned deliberately: this is the "
                        + "audit format, and the CHANGELOG says a consumer asserting an exact "
                        + "key set breaks when it grows -- so THIS is the place that is "
                        + "supposed to go red when it does, rather than a stranger's script.");
    }
}
