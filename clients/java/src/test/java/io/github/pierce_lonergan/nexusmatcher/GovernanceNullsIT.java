package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.GovernanceStatus;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The two nulls, against the live pack.
 *
 * <p>{@code governance} is null in two documented cases and a client must not collapse them. One of
 * the two is reachable against the shipped configuration and is exercised here for real. The other
 * is not, and this class says so with an assertion rather than with a comment -- see
 * {@link #rejectedRankOneIsUnreachableOnTheShippedConfiguration()}.
 */
class GovernanceNullsIT {

    private static NexusMatcherClient client;

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
    }

    @Test
    @DisplayName("null case 1: an entry with no protection code sits at the open tier")
    void uncodedEntryConfersTheOpenTier() {
        MatchResponse response = client.match(MatchRequest.of(
                List.of(FieldSpec.of(
                        "route_code",
                        "sailing.route_code",
                        "Short code identifying a scheduled route between two terminals.",
                        "string")),
                5));

        MatchCandidate top = response.topCandidateFor("sailing.route_code").orElseThrow();

        assertEquals("GBF-0028", top.governanceId(),
                "the pack's one glossary row with an empty protection-code column");
        assertNull(top.governance());
        assertEquals(
                GovernanceStatus.OPEN_TIER,
                top.governanceStatus(),
                "this field IS governed, as open; treating it as 'unknown' would send a "
                        + "published route code to a reviewer for no reason");
        assertEquals(
                "OPEN_DECK",
                response.vocabulary().openClassification(),
                "and the response says which tier that is, so the caller does not need the "
                        + "server's vocabulary file to read its own answer");
    }

    @Test
    @DisplayName("null case 2 is NOT null: a REJECTED runner-up keeps the class it confers")
    void rejectedRunnerUpKeepsItsClass() {
        MatchResponse response = client.match(MatchRequest.of(
                List.of(FieldSpec.of(
                        "route_code",
                        "sailing.route_code",
                        "Short code identifying a scheduled route between two terminals.",
                        "string")),
                5));

        List<MatchCandidate> rejectedRunnerUps = response.candidatesFor("sailing.route_code")
                .stream()
                .filter(candidate -> candidate.rank() > 1)
                .filter(candidate -> candidate.decision() == MatchDecision.REJECT)
                .toList();

        assertFalse(
                rejectedRunnerUps.isEmpty(),
                "this field's runner-ups are REJECT on the shipped pack; if that stopped being "
                        + "true the case below is no longer being tested and this test must be "
                        + "re-aimed rather than deleted");

        for (MatchCandidate runnerUp : rejectedRunnerUps) {
            assertNotNull(
                    runnerUp.governance(),
                    "rank " + runnerUp.rank() + " was REJECTED and still carries its class: "
                            + "nothing inherits from a runner-up, and the class is what lets a "
                            + "reviewer see that rank 1 is a direct identifier and rank 2 is not");
            assertEquals(GovernanceStatus.CONFERRED, runnerUp.governanceStatus());
        }
    }

    @Test
    @DisplayName("a rejected rank 1 cannot be provoked against this pack, and this is why")
    void rejectedRankOneIsUnreachableOnTheShippedConfiguration() {
        // The clause exists -- MatchResult clears the class on a rejected rank 1 -- but it cannot
        // fire at the shipped numbers: final_confidence has a structural floor of
        // semantic_weight * fusion_alpha = 0.70 * 0.90 = 0.63, and review_threshold is 0.50, so no
        // top match can fall below the bar that would reject it. Provoking it needs a server
        // reconfigured with review_threshold above 0.63, which is a change to somebody else's
        // deployment rather than something this client can arrange.
        //
        // So rather than fake it, this asserts the reason: across a spread of fields including two
        // the glossary genuinely does not describe, no rank-1 candidate is REJECT. The client-side
        // half -- what the decoder does when such a body does arrive -- is pinned in
        // MatchResponseDecodingTest#rejectedTopMatchWithholdsItsClass.
        List<FieldSpec> fields = List.of(
                FieldSpec.of("legal_name", "booking.passenger.legal_name",
                        "Full legal name of the passenger as printed on the sailing manifest.",
                        "string"),
                FieldSpec.of("lifejacket_locker_inspection_due",
                        "vessel.safety.lifejacket_locker_inspection_due",
                        "Date the lifejacket locker is next due for inspection.", "date"),
                FieldSpec.of("galley_stock_reorder_quantity", "galley.stock.reorder_quantity",
                        "How many units of a galley line to reorder.", "integer"),
                FieldSpec.of("name_digest", "booking.passenger.legal_name_digest",
                        "Salted digest of the passenger legal name.", "string"),
                FieldSpec.of("route_code", "sailing.route_code",
                        "Short code identifying a scheduled route between two terminals.",
                        "string"));

        MatchResponse response = client.match(MatchRequest.of(fields, 5));

        List<String> rejectedAtRankOne = new ArrayList<>();
        for (String path : response.paths()) {
            response.topCandidateFor(path)
                    .filter(candidate -> candidate.decision() == MatchDecision.REJECT)
                    .ifPresent(candidate -> rejectedAtRankOne.add(path));
        }

        assertTrue(
                rejectedAtRankOne.isEmpty(),
                "a rank-1 REJECT appeared on the shipped configuration (" + rejectedAtRankOne
                        + "). That is not a client failure: it means the server's thresholds have "
                        + "moved and the case IS now provokable, so this test should become a "
                        + "real one for WITHHELD_REJECTED_TOP_MATCH.");
    }

    @Test
    @DisplayName("a REVIEW is not a soft approval, and a high confidence is not evidence")
    void reviewIsNotAnApproval() {
        // The pack exists to show this: a salted digest of a passenger name retrieves the passenger
        // name entry as its top candidate, and inheriting it would tag a pseudonym as a
        // SEALED_RESTRICTED direct identifier. The correct behaviour is that it is not
        // auto-approved -- and the client must not read the confidence as permission.
        MatchResponse response = client.match(MatchRequest.of(
                List.of(FieldSpec.of(
                        "name_digest",
                        "booking.passenger.legal_name_digest",
                        "Salted digest of the passenger legal name, for join keys only.",
                        "string")),
                1));

        MatchCandidate top =
                response.topCandidateFor("booking.passenger.legal_name_digest").orElseThrow();

        assertEquals(
                MatchDecision.REVIEW,
                top.decision(),
                "over-inheriting is the silent, expensive error: nobody files a ticket about "
                        + "data they were told they could not use");
        assertTrue(
                top.confidence() > 0.6,
                "and it scores high while doing it (" + top.confidence() + "), which is exactly "
                        + "why a client must branch on decision() and never on confidence()");
    }
}
