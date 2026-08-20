package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.EncoderStatus;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.ServiceStatus;
import io.github.pierce_lonergan.nexusmatcher.model.Thresholds;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@code GET /api/v1/status}, against servers in both the healthy and the useless state. */
class StatusIT {

    @Test
    @DisplayName("a loaded server reports itself fit for a bulk run, and names what it loaded")
    void aLoadedServerIsFitForABulkRun() {
        ServiceStatus status = NexusMatcherClient.builder(LiveService.matching()).build().status();

        assertTrue(status.ready());
        assertFalse(status.degraded());
        assertTrue(status.fitForBulkRun());
        assertEquals(List.of(), status.warningCodes());

        assertTrue(status.dictionary().entries().orElseThrow() > 0);
        EncoderStatus encoder = status.encoderValue().orElseThrow();
        assertFalse(
                encoder.fallbackInForce(),
                "the fixture is meant to run on the bundled encoder; a fallback here means the "
                        + "OTHER integration tests are measuring a different model than they "
                        + "claim, which is the six-hour silent failure this field exists for");
    }

    @Test
    @DisplayName("a server with no dictionary answers 200 and says it is degraded")
    void anUnusableServerStillAnswersAndSaysWhy() {
        // Deliberately 200, not 503. A diagnostic that fails when things are broken is a
        // diagnostic nobody can use -- and this is the surface an operator reaches for precisely
        // when something is already wrong.
        ServiceStatus status =
                NexusMatcherClient.builder(LiveService.unavailable()).build().status();

        assertFalse(status.ready());
        assertTrue(status.degraded());
        assertFalse(status.fitForBulkRun());
        assertTrue(
                status.hasWarning(ServiceStatus.NO_DICTIONARY),
                "the codes are what a caller branches on: " + status.warningCodes());
        assertTrue(
                status.thresholdsValue().isEmpty(),
                "and thresholds are absent rather than defaulted -- reporting numbers that are "
                        + "not in force would be a wrong answer, not a missing one");
        assertTrue(status.encoderValue().isEmpty());
    }

    @Test
    @DisplayName("the limits it publishes are the limits it enforces")
    void publishedLimitsAreTheEnforcedOnes() {
        // The whole reason FieldSpec mirrors none of the server's caps is that they are
        // per-deployment. That is only safe if a client can ask -- so this checks the answer is
        // usable, by sending exactly the number of fields the server says it will take.
        NexusMatcherClient client = NexusMatcherClient.builder(LiveService.matching()).build();
        int cap = client.status().limits().fieldCap(false);

        List<FieldSpec> atTheCap = java.util.stream.IntStream.range(0, cap)
                .mapToObj(i -> FieldSpec.of("col_" + i, "wide.table.col_" + i))
                .toList();

        MatchResponse response = client.match(MatchRequest.of(atTheCap, 1));
        assertEquals(cap, response.results().size());
    }

    @Test
    @DisplayName("the deadline it publishes is the one a client timeout has to sit above")
    void theDeadlineIsReadableBeforeChoosingATimeout() {
        ServiceStatus status = NexusMatcherClient.builder(LiveService.matching()).build().status();
        double deadline = status.limits().deadlineSeconds();

        assertTrue(deadline > 0);
        assertTrue(
                deadline < 30.0,
                "the client's default timeout is 30 s and has to be the LONGER of the two, or it "
                        + "gives up first and the server's 504 is never seen -- which is the hang "
                        + "the deadline exists to prevent, reintroduced by an off-by-one. Server "
                        + "deadline is " + deadline + " s.");
    }

    @Test
    @DisplayName("two identical status calls produce identical answers")
    void statusIsByteStable() {
        NexusMatcherClient client = NexusMatcherClient.builder(LiveService.matching()).build();

        ServiceStatus first = client.status();
        ServiceStatus second = client.status();

        assertEquals(
                first,
                second,
                "nothing here is read from a clock or from live load at request time -- the "
                        + "in-flight count is deliberately absent -- which is what lets an "
                        + "operator diff two hosts and see only the difference that matters");
    }

    @Test
    @DisplayName("the confidence floor on a match response agrees with the one on status")
    void theTwoPublishedFloorsAgree() {
        // The same number reaches a caller through two different routes, computed by two different
        // code paths. If they ever disagreed, a client would get a different answer about what its
        // confidences can mean depending on which endpoint it asked.
        NexusMatcherClient client = NexusMatcherClient.builder(LiveService.matching()).build();

        Thresholds thresholds = client.status().thresholdsValue().orElseThrow();
        MatchResponse response = client.match(MatchRequest.of(
                List.of(FieldSpec.of("legal_name", "booking.passenger.legal_name",
                        "Full legal name of the passenger as printed on the sailing manifest.",
                        "string")),
                1));

        assertEquals(
                thresholds.confidenceFloorValue().orElseThrow(),
                response.scoringValue().orElseThrow().confidenceFloorValue().orElseThrow(),
                1e-9,
                "status.thresholds.minimumAchievableConfidence and "
                        + "match.scoring.confidenceFloor are the same claim about the same server");
    }
}
