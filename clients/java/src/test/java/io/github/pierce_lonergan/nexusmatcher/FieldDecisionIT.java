package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.FieldDecision;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.ScoringContract;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The per-field verdict, against two live servers that differ by one configuration key.
 *
 * <p>{@link LiveService#matching()} ships no absolute-score floor, so every field there comes back
 * with a verdict rolled up from rank 1. {@link LiveService#floor()} is the same pack, the same
 * encoder and the same glossary with a floor configured, so a field the glossary does not describe
 * comes back {@link FieldDecision#NO_MATCH}. Running both is what makes NO_MATCH a demonstrated
 * server behaviour here rather than a shape somebody typed into a fixture.
 */
class FieldDecisionIT {

    private static NexusMatcherClient client;
    private static NexusMatcherClient floorClient;

    /** A field the ferry glossary genuinely does not describe. */
    private static FieldSpec undescribed() {
        return FieldSpec.of(
                "lifejacket_locker_inspection_due",
                "vessel.safety.lifejacket_locker_inspection_due",
                "Date the lifejacket locker is next due for inspection.",
                "date");
    }

    /** A field it does. */
    private static FieldSpec described() {
        return FieldSpec.of(
                "legal_name",
                "booking.passenger.legal_name",
                "Full legal name of the passenger as printed on the sailing manifest.",
                "string");
    }

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
        floorClient = NexusMatcherClient.builder(LiveService.floor()).build();
    }

    @Test
    @DisplayName("every field that was sent gets a verdict, keyed and ordered like results")
    void everyFieldGetsAVerdict() {
        List<FieldSpec> fields = List.of(described(), undescribed());
        MatchResponse response = client.match(MatchRequest.of(fields, 3));

        assertEquals(
                fields.stream().map(FieldSpec::responseKey).toList(),
                List.copyOf(response.fieldDecisions().keySet()),
                "fieldDecisions is the conservation law's second half: same keys, same order. A "
                        + "column missing from it is a column with no verdict and no error.");
        for (String path : response.paths()) {
            assertTrue(
                    response.verdictFor(path).orElseThrow().isKnown(),
                    path + " came back with a verdict this client build does not know. That is "
                            + "not a bug in this test -- it means the server has added a value and "
                            + "the client needs updating; the value is "
                            + response.verdictFor(path).orElseThrow().wireValue());
        }
        assertEquals(List.of(), response.pathsWithUnknownVerdicts());
    }

    @Test
    @DisplayName("with no floor configured, an undescribed field is still REVIEW, not NO_MATCH")
    void withoutAFloorNothingIsNoMatch() {
        // This is the gap NO_MATCH was invented for, shown rather than described. The glossary has
        // no entry for a lifejacket locker inspection date, and the shipped configuration cannot
        // say so: min-max normalisation puts the best of a hopeless shortlist above the structural
        // floor, so the field comes back REVIEW at a healthy-looking confidence.
        MatchResponse response = client.match(MatchRequest.of(List.of(undescribed()), 3));
        String path = undescribed().responseKey();

        assertEquals(
                FieldDecision.REVIEW,
                response.verdictFor(path).orElseThrow().decision(),
                "if this ever becomes NO_MATCH on the stock fixture, the server has started "
                        + "shipping a floor and the claim below needs re-checking rather than "
                        + "this assertion needing deleting");
        assertTrue(
                response.scoringValue().orElseThrow().absoluteScoreFloorValue().isEmpty(),
                "and the response says why: no floor is configured, so NO_MATCH could only come "
                        + "from a field with no candidates at all");
        assertFalse(response.candidatesFor(path).isEmpty());
    }

    @Test
    @DisplayName("with a floor configured, the same field is NO_MATCH -- and keeps its candidates")
    void withAFloorTheSameFieldIsNoMatchAndStillCarriesCandidates() {
        MatchResponse response = floorClient.match(MatchRequest.of(List.of(undescribed()), 3));
        String path = undescribed().responseKey();

        assertEquals(FieldDecision.NO_MATCH, response.verdictFor(path).orElseThrow().decision());
        assertTrue(response.verdictFor(path).orElseThrow().isNoMatch());

        List<MatchCandidate> candidates = response.candidatesFor(path);
        assertFalse(
                candidates.isEmpty(),
                "the server returns the candidates anyway -- they are evidence for the reviewer, "
                        + "and an empty list would have thrown that evidence away. A client model "
                        + "that assumed no-match means no candidates would be wrong on a live "
                        + "server, not just on a fixture.");

        MatchCandidate top = candidates.get(0);
        assertNotNull(
                top.governance(),
                "and rank 1 confers a real class, which a client reading rank 1 instead of the "
                        + "verdict would apply to a column nothing in the glossary describes");
        assertEquals(MatchDecision.REVIEW, top.decision());
        assertTrue(
                response.inheritableGovernanceFor(path).isEmpty(),
                "so the field inherits nothing");

        ScoringContract scoring = response.scoringValue().orElseThrow();
        double floor = scoring.absoluteScoreFloorValue().orElseThrow();
        assertTrue(
                top.absoluteScoreValue().orElseThrow() < floor,
                "rank 1's absolute score must sit below the configured floor for this verdict to "
                        + "mean what it says; got " + top.absoluteScore() + " against " + floor);
    }

    @Test
    @DisplayName("a floor does not sweep up the fields the glossary really does describe")
    void aFloorLeavesTheDescribedFieldAlone() {
        MatchResponse response =
                floorClient.match(MatchRequest.of(List.of(described(), undescribed()), 3));

        assertEquals(
                FieldDecision.AUTO_APPROVE,
                response.verdictFor(described().responseKey()).orElseThrow().decision());
        assertEquals(
                "MANIFEST_NAME",
                response.inheritableGovernanceFor(described().responseKey())
                        .orElseThrow()
                        .code());
        assertEquals(
                FieldDecision.NO_MATCH,
                response.verdictFor(undescribed().responseKey()).orElseThrow().decision(),
                "one response, one floor, two different answers -- which is the whole point of "
                        + "the floor being a per-field test rather than a global switch");
    }

    @Test
    @DisplayName("absoluteScore separates the two fields where confidence does not")
    void absoluteScoreSeesTheDifferenceConfidenceCannot() {
        MatchResponse response = client.match(MatchRequest.of(List.of(described(), undescribed()), 1));

        MatchCandidate good =
                response.topCandidateFor(described().responseKey()).orElseThrow();
        MatchCandidate bad =
                response.topCandidateFor(undescribed().responseKey()).orElseThrow();

        double confidenceFloor =
                response.scoringValue().orElseThrow().confidenceFloorValue().orElseThrow();
        assertTrue(
                bad.confidence() > confidenceFloor,
                "the field nothing describes still clears the structural confidence floor ("
                        + bad.confidence() + " > " + confidenceFloor + "), which is exactly why "
                        + "a confidence threshold cannot find it");
        assertTrue(
                good.absoluteScoreValue().orElseThrow() > bad.absoluteScoreValue().orElseThrow(),
                "and the absolute score can: " + good.absoluteScore() + " against "
                        + bad.absoluteScore() + ". It is the number the scoring block declares "
                        + "comparable ACROSS fields, and this is what that buys.");
        assertTrue(
                response.scoringValue().orElseThrow().comparableAcrossFields("absoluteScore"),
                "and the server says so itself rather than leaving the client to assume it");
    }
}
