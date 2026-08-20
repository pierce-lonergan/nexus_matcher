package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.FieldGovernance;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.GovernanceOutcome;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.Vocabulary;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link MatchResponse#governanceFor(String)} against real services.
 *
 * <p>Four of the seven outcomes are reachable from the two fixture servers this suite already
 * starts, and this class provokes all four -- three of them in ONE request, so the discrimination
 * is shown to be per-COLUMN rather than a property of a server somebody configured to produce it.
 *
 * <p>The fourth needs the second server: {@link GovernanceOutcome#WITHHELD_NO_MATCH} requires a
 * configured absolute-score floor, the library ships none, and {@code LiveService.floor()} is the
 * fixture that sets one. The SAME field spec goes to both servers, so a column that is
 * {@code WITHHELD_NO_MATCH} on one and something else on the other differs by the server's
 * configuration alone.
 *
 * <p>The remaining three are unreachable here and are pinned in {@code FieldGovernanceTest}, each
 * with the reason: a rank-1 REJECT needs a review threshold above the structural confidence floor,
 * an unconfigured vocabulary needs a glossary carrying no protection codes, and an unreadable
 * verdict needs a server newer than this build.
 */
class FieldGovernanceIT {

    private static NexusMatcherClient client;
    private static NexusMatcherClient floored;

    /** A column the pack classifies confidently. */
    private static final FieldSpec CONFERRED_FIELD = FieldSpec.of(
            "legal_name",
            "booking.passenger.legal_name",
            "Full legal name of the passenger as printed on the sailing manifest.",
            "string");

    /** A column whose best match is the pack's one row with an empty protection-code cell. */
    private static final FieldSpec OPEN_FIELD = FieldSpec.of(
            "route_code",
            "sailing.route_code",
            "Short code identifying a scheduled route between two terminals.",
            "string");

    /**
     * A salted digest of a passenger name. It retrieves the passenger-name entry confidently and
     * must not be auto-approved onto it -- inheriting there would tag a pseudonym as a direct
     * identifier. The pack exists partly to demonstrate this.
     */
    private static final FieldSpec REVIEW_FIELD = FieldSpec.of(
            "name_digest",
            "booking.passenger.legal_name_digest",
            "Salted digest of the passenger legal name, for join keys only.",
            "string");

    /** A column nothing in the pack describes, and the one the floor fixture rejects. */
    private static final FieldSpec HOPELESS_FIELD = FieldSpec.of(
            "galley_stock_reorder_quantity",
            "galley.stock.reorder_quantity",
            "How many units of a galley line to reorder.",
            "integer");

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
        floored = NexusMatcherClient.builder(LiveService.floor()).build();
    }

    @Test
    @DisplayName("three outcomes in one response, so the discrimination is per column")
    void oneRequestProducesThreeDifferentOutcomes() {
        MatchResponse response = client.match(
                MatchRequest.of(List.of(CONFERRED_FIELD, OPEN_FIELD, REVIEW_FIELD), 5));

        Map<GovernanceOutcome, String> seen = new EnumMap<>(GovernanceOutcome.class);
        for (String path : response.paths()) {
            seen.put(response.governanceFor(path).outcome(), path);
        }

        assertEquals(
                3,
                seen.size(),
                "the three columns produced " + seen + ". They are meant to land on three "
                        + "different outcomes in a single response; if they no longer do, the "
                        + "pack's scores have moved and these specs need re-aiming rather than "
                        + "the assertion relaxing.");

        assertEquals(
                GovernanceOutcome.CONFERRED,
                response.governanceFor(CONFERRED_FIELD.path()).outcome());
        assertEquals(
                GovernanceOutcome.OPEN_TIER,
                response.governanceFor(OPEN_FIELD.path()).outcome());
        assertEquals(
                GovernanceOutcome.WITHHELD_PENDING_REVIEW,
                response.governanceFor(REVIEW_FIELD.path()).outcome(),
                "a high confidence is not permission. This column scores well against the "
                        + "passenger-name entry and inheriting it would classify a pseudonym as a "
                        + "direct identifier.");
    }

    @Test
    @DisplayName("CONFERRED hands back the class the server resolved, unaltered")
    void aConferredClassIsTheServersOwn() {
        MatchResponse response = client.match(MatchRequest.of(List.of(CONFERRED_FIELD), 1));
        String path = CONFERRED_FIELD.path();

        FieldGovernance governance = response.governanceFor(path);

        assertEquals(GovernanceOutcome.CONFERRED, governance.outcome());
        assertTrue(governance.maySafelyApply());
        assertEquals(
                response.topCandidateFor(path).orElseThrow().governance(),
                governance.conferred().orElseThrow(),
                "this client re-decides nothing: the class handed back is the object the server "
                        + "sent on rank 1, not a reconstruction of it");
    }

    @Test
    @DisplayName("OPEN_TIER names the deployment's own tier, never the sentinel")
    void anOpenTierIsResolvedFromTheResponseItArrivedOn() {
        MatchResponse response = client.match(MatchRequest.of(List.of(OPEN_FIELD), 5));

        FieldGovernance governance = response.governanceFor(OPEN_FIELD.path());

        assertEquals(GovernanceOutcome.OPEN_TIER, governance.outcome());
        assertEquals("OPEN_DECK", governance.openTier().orElseThrow());
        assertTrue(
                response.vocabulary().isConfigured(),
                "this outcome is only correct on a deployment that HAS a vocabulary; the same "
                        + "wire shape from one that does not is UNCLASSIFIABLE_NO_VOCABULARY");
        assertNotEquals(
                Vocabulary.UNCONFIGURED_OPEN_CLASSIFICATION, governance.openTier().orElseThrow());
    }

    @Test
    @DisplayName("WITHHELD_NO_MATCH: the same column, and only the server's floor changed")
    void aConfiguredFloorTurnsAnOrdinaryColumnIntoOneThatInheritsNothing() {
        MatchRequest request = MatchRequest.of(List.of(HOPELESS_FIELD), 5);
        String path = HOPELESS_FIELD.path();

        MatchResponse withoutFloor = client.match(request);
        MatchResponse withFloor = floored.match(request);

        assertNotEquals(
                GovernanceOutcome.WITHHELD_NO_MATCH,
                withoutFloor.governanceFor(path).outcome(),
                "the stock server ships no absolute-score floor and must not invent one; a "
                        + "NO_MATCH here would mean the library started unclassifying columns on a "
                        + "threshold nobody chose");

        FieldGovernance governance = withFloor.governanceFor(path);
        assertEquals(GovernanceOutcome.WITHHELD_NO_MATCH, governance.outcome());
        assertFalse(governance.maySafelyApply());
        assertTrue(governance.needsAHuman());
        assertTrue(governance.conferred().isEmpty());
        assertTrue(governance.openTier().isEmpty());

        assertFalse(
                withFloor.candidatesFor(path).isEmpty(),
                "the candidates survive a NO_MATCH -- they are the evidence for whoever now has "
                        + "to decide whether the glossary needs a new term");
    }

    @Test
    @DisplayName("no outcome anywhere in a normal batch is a permission this client invented")
    void everyOutcomeIsBackedByTheServersOwnVerdict() {
        MatchResponse response = client.match(MatchRequest.of(
                List.of(CONFERRED_FIELD, OPEN_FIELD, REVIEW_FIELD, HOPELESS_FIELD), 5));

        for (String path : response.paths()) {
            FieldGovernance governance = response.governanceFor(path);
            if (!governance.maySafelyApply()) {
                continue;
            }
            assertTrue(
                    response.verdictFor(path).orElseThrow().maySafelyInherit(),
                    path + " may be classified from this response while the server's own verdict "
                            + "for it is " + response.verdictFor(path).orElseThrow().wireValue()
                            + ". This client holds no second opinion about inheritance; an outcome "
                            + "that permits applying a class without an AUTO_APPROVE behind it is "
                            + "this client deciding governance, which it must not do.");
        }
    }
}
