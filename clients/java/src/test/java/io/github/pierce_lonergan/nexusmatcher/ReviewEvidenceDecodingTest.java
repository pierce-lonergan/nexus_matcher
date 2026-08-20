package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.Agreement;
import io.github.pierce_lonergan.nexusmatcher.model.ConceptGroup;
import io.github.pierce_lonergan.nexusmatcher.model.ConsistencyReport;
import io.github.pierce_lonergan.nexusmatcher.model.Contrast;
import io.github.pierce_lonergan.nexusmatcher.model.ContrastReport;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.Separation;
import io.github.pierce_lonergan.nexusmatcher.model.SignalDifference;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The two review-evidence blocks, against bodies captured verbatim from a running service.
 *
 * <p>Three fixtures, and the three of them together are the test:
 *
 * <ul>
 *   <li>{@code match-response.json} asked for neither block, so both must decode to null. That is
 *       the additive property from this client's side of the wire -- a caller who does not ask is
 *       handed the same object they were handed before either block existed.
 *   <li>{@code match-response-evidence.json} asked for both at the SERVER'S DEFAULT grouping, where
 *       the consistency report finds nothing. An empty report is the shipped behaviour, and a test
 *       that only ever saw a populated one would be testing a setting nobody runs.
 *   <li>{@code match-response-evidence-leaf-key.json} asked for the loose grouping, where the two
 *       columns named {@code name} become one "concept" and their different answers are reported as
 *       a contradiction. They are a ferry terminal and a passenger. The finding is a collision, it
 *       is the failure {@link ConsistencyReport}'s javadoc warns about, and it is here because a
 *       documented hazard with no fixture behind it is one nobody can check.
 * </ul>
 *
 * <p>Plain {@link ObjectMapper}, like {@link MatchResponseDecodingTest}: this is a test of the
 * records' own annotations, not of the mapper the client happens to configure.
 */
class ReviewEvidenceDecodingTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private MatchResponse decode(String fixture) throws Exception {
        return mapper.readValue(Fixtures.captured(fixture), MatchResponse.class);
    }

    // =============================================================================
    // ADDITIVE: a caller who asks for nothing gets nothing new
    // =============================================================================

    @Test
    @DisplayName("a response that asked for neither block carries neither, and says so with null")
    void bothBlocksAreAbsentWhenNeitherWasAskedFor() throws Exception {
        MatchResponse response = decode("match-response.json");

        assertNull(response.contrast());
        assertNull(response.consistency());
        assertTrue(response.contrastValue().isEmpty());
        assertTrue(response.consistencyValue().isEmpty());
        assertTrue(
                response.contrastFor("booking.passenger.legal_name").isEmpty(),
                "no block means no contrast for any path, and the convenience accessor must not "
                        + "invent one");

        // The half that makes the above worth asserting: everything else still decoded.
        assertFalse(response.results().isEmpty());
        assertNotNull(response.vocabulary());
    }

    // =============================================================================
    // CONTRAST
    // =============================================================================

    @Test
    @DisplayName("every input path is a key of the contrast, in the order it was sent")
    void everyPathIsPresentInTheContrast() throws Exception {
        MatchResponse response = decode("match-response-evidence.json");

        ContrastReport report = response.contrastValue().orElseThrow();
        assertEquals(
                List.of("published.terminal.name", "booking.passenger.name", "sailing.route_code"),
                List.copyOf(report.paths()),
                "the contrast is keyed and ordered exactly like results; losing that order "
                        + "re-associates a contrast with the wrong column");
        assertEquals(List.copyOf(response.paths()), List.copyOf(report.paths()));
    }

    @Test
    @DisplayName("the arithmetic closes: the weighted differences sum to the confidence gap")
    void theContrastArithmeticCloses() throws Exception {
        MatchResponse response = decode("match-response-evidence.json");
        ContrastReport report = response.contrastValue().orElseThrow();

        // The server verifies this before answering and refuses rather than send a contrast that
        // does not close, so re-running it here is a check that the CLIENT decoded the numbers it
        // was sent -- a component silently bound to the wrong wire name would show up as a sum
        // that no longer reaches the gap.
        //
        // The tolerance is one order of magnitude above the published resolution, and that is the
        // server's own tolerance rather than a number chosen here: both operands of every delta
        // are rounded to the resolution before being subtracted, so the sum of five of them can
        // legitimately sit a few units of the last place away from a separately rounded gap. The
        // captured body is exactly such a case -- 0.285436 against 0.285435.
        double tolerance = report.resolution() * 10.0;
        assertTrue(report.resolution() > 0.0, "a resolution of zero would make this vacuous");

        for (String path : report.paths()) {
            Contrast contrast = report.contrastFor(path).orElseThrow();
            double summed = contrast.signals().stream()
                    .mapToDouble(SignalDifference::weightedDelta)
                    .sum();
            assertEquals(contrast.signalGap(), summed, tolerance,
                    path + ": the per-signal weighted deltas must sum to signalGap");
            assertEquals(contrast.confidenceGap(), contrast.signalGap(), tolerance,
                    path + ": signalGap and confidenceGap are the same margin reached two ways");
            assertEquals(
                    contrast.topConfidence() - contrast.runnerUpConfidence(),
                    contrast.confidenceGap(),
                    tolerance,
                    path + ": the gap is the subtraction a reader would do by hand");
        }
    }

    @Test
    @DisplayName("an empty decidingSignals is a real answer, not a missing one")
    void noSingleSignalCarriedTheMargin() throws Exception {
        MatchResponse response = decode("match-response-evidence.json");
        Contrast contrast = response.contrastFor("published.terminal.name").orElseThrow();

        assertTrue(contrast.isSeparated());
        assertFalse(contrast.isTied());
        assertEquals("fusedRetrieval", contrast.largestDifferenceValue().orElseThrow());

        assertTrue(contrast.decidingSignals().isEmpty());
        assertFalse(
                contrast.hasDecidingSignal(),
                "removing the largest signal's contribution still leaves rank 1 ahead here, so no "
                        + "ONE signal decided it. That is an answer -- reading it as 'the server "
                        + "did not say' is the misreading this accessor exists to prevent");
        assertTrue(contrast.signals().stream().noneMatch(SignalDifference::deciding));
    }

    @Test
    @DisplayName("a signal below the resolution is never separating and never a cause")
    void signalsThatDoNotDifferAreNotCauses() throws Exception {
        MatchResponse response = decode("match-response-evidence.json");
        Contrast contrast = response.contrastFor("published.terminal.name").orElseThrow();

        SignalDifference tied = contrast.signal("type").orElseThrow();
        assertEquals(0.0, tied.delta(), 0.0);
        assertFalse(tied.separating());
        assertFalse(tied.deciding());

        assertTrue(
                contrast.separatingSignals().stream().noneMatch(each -> each.delta() == 0.0),
                "a signal with no difference cannot be one of the signals that separated the two");
        assertTrue(
                contrast.separatingSignals().size() < contrast.signals().size(),
                "this fixture must contain both kinds, or the filter is untested");
    }

    @Test
    @DisplayName("the two facts that come from the entries rather than from any signal")
    void governanceAndDomainDifferencesAreCarried() throws Exception {
        MatchResponse response = decode("match-response-evidence.json");
        Contrast contrast = response.contrastFor("published.terminal.name").orElseThrow();

        assertEquals("GBF-0027", contrast.topGovernanceId());
        assertEquals("GBF-0001", contrast.runnerUpGovernanceId());
        assertTrue(contrast.governanceDiffers());
        assertTrue(contrast.domainDiffers());
    }

    @Test
    @DisplayName("the contrast's own numbers carry the scope they may be compared over")
    void comparabilityIsCarriedAndReadable() throws Exception {
        MatchResponse response = decode("match-response-evidence.json");
        ContrastReport report = response.contrastValue().orElseThrow();

        assertEquals(
                "WITHIN_FIELD",
                report.confidenceGapScope().orElseThrow(),
                "a difference is no more comparable than its operands, and confidence is "
                        + "within-field; a fixed cut point on a gap across a schema means nothing");
        assertEquals("WITHIN_FIELD", report.signalScope("fusedRetrieval").orElseThrow());
        assertEquals("ACROSS_FIELDS", report.signalScope("lexical").orElseThrow());
        assertTrue(
                report.signalScope("aSignalThisServerDoesNotWeight").isEmpty(),
                "an undeclared number must not be compared with anything, and empty is how that "
                        + "is said");
    }

    // =============================================================================
    // CONSISTENCY
    // =============================================================================

    @Test
    @DisplayName("at the server's default grouping the report finds nothing, and that is shipped")
    void theDefaultGroupingReportsNothing() throws Exception {
        MatchResponse response = decode("match-response-evidence.json");
        ConsistencyReport report = response.consistencyValue().orElseThrow();

        assertEquals(
                1,
                report.qualifierSegments().orElseThrow(),
                "the server's default is 1 -- a leaf groups only with a leaf under the same "
                        + "declared parent. If this is ever 0 the default moved to the key that "
                        + "was measured at 0.0233 precision, and ConsistencyReport's javadoc, "
                        + "which tells a reader 1 is the default, is now wrong");
        assertEquals(0, report.groupsFound());
        assertEquals(0, report.fieldsGrouped());
        assertEquals(0, report.groupsDisagreeing());
        assertTrue(report.groups().isEmpty());
        assertTrue(report.disagreeingGroups().isEmpty());
        assertFalse(report.promotionApplied());

        assertTrue(report.includeDataType().orElseThrow());
        assertFalse(report.orderSensitive().orElseThrow());
        assertEquals(2, report.minGroupSize().orElseThrow());
    }

    @Test
    @DisplayName("the loose grouping reports a DISAGREE, and it is a collision")
    void theLooseGroupingManufacturesADisagreement() throws Exception {
        MatchResponse response = decode("match-response-evidence-leaf-key.json");
        ConsistencyReport report = response.consistencyValue().orElseThrow();

        assertEquals(0, report.qualifierSegments().orElseThrow());
        assertEquals(1, report.groupsFound());
        assertEquals(2, report.fieldsGrouped());
        assertEquals(1, report.groupsDisagreeing());
        assertFalse(
                report.promotionApplied(),
                "the block changed nothing in results or fieldDecisions, and says so rather than "
                        + "leaving a consumer to infer it");

        ConceptGroup group = report.disagreeingGroups().get(0);
        assertTrue(group.disagrees());
        assertFalse(group.agrees());
        assertFalse(group.isUndecided());
        assertEquals(
                List.of("published.terminal.name", "booking.passenger.name"),
                group.fields());
        assertEquals("GBF-0027", group.answerFor("published.terminal.name").orElseThrow());
        assertEquals("GBF-0001", group.answerFor("booking.passenger.name").orElseThrow());

        // The reader's own test for a collision, run on a real collision. A ferry terminal's name
        // and a passenger's name are not one business concept; they share four letters. Every
        // column that answered gave a DIFFERENT answer, which is what distinguishes "these columns
        // were merged by mistake" from "these columns are one concept and the matcher was
        // inconsistent about them".
        assertEquals(2, group.answeredCount());
        assertEquals(2, group.distinctAnswers());
        assertEquals(
                group.answeredCount(),
                group.distinctAnswers(),
                "distinctAnswers equal to answeredCount is the collision signature "
                        + "ConsistencyReport's javadoc tells a reader to check first");

        assertTrue(
                group.majorityAnswer().isEmpty(),
                "no answer holds a plurality when every member gave a different one");
        assertEquals(0, group.majorityCount());
    }

    @Test
    @DisplayName("a column that shares its concept with nothing is not reported at all")
    void anUngroupedColumnIsAbsentFromTheReport() throws Exception {
        MatchResponse response = decode("match-response-evidence-leaf-key.json");
        ConsistencyReport report = response.consistencyValue().orElseThrow();

        assertTrue(
                report.groupFor("sailing.route_code").isEmpty(),
                "it cannot disagree with anyone, so there is nothing to say about it");
        assertTrue(report.groupFor("published.terminal.name").isPresent());
        assertTrue(
                response.paths().contains("sailing.route_code"),
                "it was still matched and still came back; absence from the report is not "
                        + "absence from the response");
    }

    @Test
    @DisplayName("the concept key is a grouping artifact and is carried as one")
    void theConceptKeyIsCarriedVerbatim() throws Exception {
        MatchResponse response = decode("match-response-evidence-leaf-key.json");
        ConceptGroup group = response.consistencyValue().orElseThrow().groups().get(0);

        assertEquals("|name|name|string", group.concept());
    }

    // =============================================================================
    // THE TWO VOCABULARIES THE SERVER CLOSED AND THIS CLIENT DID NOT
    // =============================================================================

    @Test
    @DisplayName("the live values decode, and an unknown one costs nothing")
    void theEvidenceVocabulariesDegradeRatherThanRefuse() throws Exception {
        MatchResponse live = decode("match-response-evidence.json");
        assertEquals(
                Separation.SEPARATED,
                live.contrastFor("published.terminal.name").orElseThrow().separationValue());
        assertEquals(
                Agreement.DISAGREE,
                decode("match-response-evidence-leaf-key.json")
                        .consistencyValue().orElseThrow().groups().get(0).agreementValue());

        // A body the current service cannot produce -- so it is written here rather than
        // captured, and the two literals are hypothetical future values, not copies of anything
        // published. The service publishes both vocabularies as CLOSED components; this client
        // binds them open anyway, because one unrecognised word describing why a runner-up lost
        // must not cost a caller every field in a 250-field batch. That claim is only worth
        // making if it is exercised.
        String futureBody = """
                {"results": {"t.a": []},
                 "contrast": {"resolution": 1e-06, "comparability": {}, "fields": {"t.a": {
                    "topGovernanceId": "A", "runnerUpGovernanceId": "B",
                    "topConfidence": 0.9, "runnerUpConfidence": 0.8,
                    "confidenceGap": 0.1, "signalGap": 0.1,
                    "separation": "NEARLY_TIED", "largestDifference": null,
                    "decidingSignals": [], "governanceDiffers": false,
                    "domainDiffers": false, "signals": []}}},
                 "consistency": {"grouping": {}, "groupsFound": 1, "fieldsGrouped": 2,
                    "groupsDisagreeing": 0, "promotionApplied": false, "groups": [{
                    "concept": "|x|x|string", "fields": ["t.a"], "answers": {"t.a": "A"},
                    "distinctAnswers": 1, "agreement": "AGREE_WITH_RESERVATIONS",
                    "majorityGovernanceId": "A", "majorityCount": 1}]}}
                """;

        MatchResponse future = mapper.readValue(futureBody, MatchResponse.class);

        Contrast contrast = future.contrastFor("t.a").orElseThrow();
        assertEquals(Separation.UNKNOWN, contrast.separationValue());
        assertEquals(
                "NEARLY_TIED",
                contrast.separation(),
                "the server's own word survives, so it can be named in a ticket rather than "
                        + "merely counted");
        assertFalse(contrast.isTied());
        assertFalse(
                contrast.isSeparated(),
                "an unread value must not be answered as either; both predicates say no and the "
                        + "raw string is where the caller looks");

        ConceptGroup group = future.consistencyValue().orElseThrow().groups().get(0);
        assertEquals(Agreement.UNKNOWN, group.agreementValue());
        assertEquals("AGREE_WITH_RESERVATIONS", group.agreement());
        assertFalse(
                group.agrees(),
                "an unrecognised agreement must never become a quiet AGREE -- that is how a new "
                        + "value silently reads as 'these columns confirmed each other'");
        assertFalse(group.disagrees());
        assertFalse(group.isUndecided());

        // And the point of all of it: everything else in the response still decoded.
        assertTrue(future.paths().contains("t.a"));
        assertEquals("A", contrast.topGovernanceId());
    }
}
