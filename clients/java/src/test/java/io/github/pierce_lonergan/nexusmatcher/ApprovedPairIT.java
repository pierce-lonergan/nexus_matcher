package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchProvenance;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code provenance}, against two REAL servers that differ only in whether a reviewer's verdict is
 * attached.
 *
 * <p>The decoding test beside this one proves what the client makes of a captured body. This
 * proves that the body is what a live service actually sends, and -- the part a capture cannot
 * show -- that the difference between the two values is a property of the SERVER'S CONFIGURATION
 * and not of the request. The same field spec goes to both ports; one answers from retrieval and
 * one answers from a human's decision.
 *
 * <p>{@code clients/java/fixture_approved_pair_app.py} is the fifth fixture. It exists because the
 * service ships no feedback consumer -- {@code create_app()} builds none and {@code NexusMatcher()}
 * takes {@code feedback_consumer=None}, which its own source documents as a measured decision
 * rather than an unfinished wire-up -- so {@link MatchProvenance#APPROVED_PAIR} is unreachable on
 * every server this package starts.
 */
class ApprovedPairIT {

    /** The field {@code clients/java/fixture-approved-pairs.jsonl} carries a standing verdict for. */
    private static final String REVIEWED_PATH = "booking.passenger.legal_name";

    /**
     * A column whose retrieval confidence against the shipped pack is exactly 1.0.
     *
     * <p>All five signals are maximal: the name IS the glossary's own {@code logical_name} for
     * GBF-0027, so editDistance and lexical are 1.0, {@code type} is declared, the {@code domain}
     * signal names the entry's own domain, and the entry is rank 1 in both retrieval arms.
     */
    private static final String CONTROL_PATH = "published.terminal_nm";

    private static NexusMatcherClient bypassing;
    private static NexusMatcherClient matching;

    @BeforeAll
    static void connect() {
        bypassing = NexusMatcherClient.builder(LiveService.approvedPair()).build();
        matching = NexusMatcherClient.builder(LiveService.matching()).build();
    }

    private static MatchRequest bothFields() {
        FieldSpec reviewed = FieldSpec.of(
                "legal_name",
                REVIEWED_PATH,
                "Full legal name of the passenger as printed on the sailing manifest.",
                "string");
        FieldSpec control = new FieldSpec(
                "terminal_nm",
                CONTROL_PATH,
                "The public name of a Gravel Bay ferry terminal.",
                "string",
                Map.of("domain", "Published"));
        return MatchRequest.of(List.of(reviewed, control), 3).withExplain(true);
    }

    @Test
    @DisplayName("the same field is APPROVED_PAIR on one server and RETRIEVAL on the other")
    void provenanceIsAPropertyOfTheServerNotTheRequest() {
        MatchRequest request = bothFields();

        MatchCandidate decided =
                bypassing.match(request).topCandidateFor(REVIEWED_PATH).orElseThrow();
        MatchCandidate scored =
                matching.match(request).topCandidateFor(REVIEWED_PATH).orElseThrow();

        assertEquals(MatchProvenance.APPROVED_PAIR, decided.provenanceValue());
        assertEquals(MatchProvenance.RETRIEVAL, scored.provenanceValue());

        assertEquals(
                scored.governanceId(),
                decided.governanceId(),
                "the reviewer approved the entry retrieval finds for this column anyway, which is "
                        + "deliberate: a different id would leave the two answers distinguishable "
                        + "without reading provenance at all, and this comparison would prove "
                        + "nothing about the member it exists for");
        assertEquals(MatchDecision.AUTO_APPROVE, decided.decision());
        assertEquals(MatchDecision.AUTO_APPROVE, scored.decision());
    }

    @Test
    @DisplayName("confidence 1.0 is reachable by ORDINARY RETRIEVAL against the shipped pack")
    void oneIsNotOutsideTheScorersRange() {
        MatchCandidate control =
                matching.match(bothFields()).topCandidateFor(CONTROL_PATH).orElseThrow();

        assertEquals(
                1.0,
                control.confidence(),
                0.0,
                "the five default weights sum to exactly 1.0 and every signal is attainable at "
                        + "1.0. A magic confidence of 1.0 was once documented as identifying a "
                        + "bypass because the scorer supposedly could not produce it; this is a "
                        + "live server producing it, with nothing bypassed.");
        assertEquals(MatchDecision.AUTO_APPROVE, control.decision());
        assertEquals(MatchProvenance.RETRIEVAL, control.provenanceValue());
        assertTrue(control.wasScored());
        assertFalse(
                control.decidedByAReviewer(),
                "so (confidence 1.0, AUTO_APPROVE) is not a discriminator, and any client using "
                        + "it as one is reading a very good match as a human's decision");
        assertTrue(control.hasAbsoluteScore(), "it was measured, so the number is here");
    }

    @Test
    @DisplayName("a bypassed field is answered alone, unscored, and unexplained")
    void whatABypassedFieldActuallyCarries() {
        MatchResponse response = bypassing.match(bothFields());

        List<MatchCandidate> candidates = response.candidatesFor(REVIEWED_PATH);
        assertEquals(
                1,
                candidates.size(),
                "top_k was 3. Ranks 2 and beyond would have to come from retrieval, and "
                        + "retrieval did not run for this field.");

        MatchCandidate decided = candidates.get(0);
        assertTrue(decided.decidedByAReviewer());
        assertFalse(decided.wasScored());
        assertEquals(1.0, decided.confidence(), 0.0);

        assertNull(
                decided.absoluteScore(),
                "absent because nothing measured it. Zero would be a measurement, and on a "
                        + "cosine metric a very bad one, about a term a human chose.");
        assertTrue(decided.absoluteScoreValue().isEmpty());
        assertTrue(
                decided.explainValue().isEmpty(),
                "explain WAS requested. The block promises sum(scores * weights) == confidence "
                        + "and a candidate with no components cannot keep it, so it is left out "
                        + "rather than filled with five measurements nobody took.");

        // The same response, the field that was matched: everything the bypassed one lacks.
        MatchCandidate control = response.topCandidateFor(CONTROL_PATH).orElseThrow();
        assertTrue(control.wasScored());
        assertTrue(control.hasAbsoluteScore());
        assertTrue(control.explainValue().isPresent());
    }

    @Test
    @DisplayName("the rest of the batch is untouched: every other candidate is RETRIEVAL")
    void theBypassDoesNotLeakIntoTheOtherFields() {
        MatchResponse response = bypassing.match(bothFields());

        for (MatchCandidate candidate : response.candidatesFor(CONTROL_PATH)) {
            assertEquals(
                    MatchProvenance.RETRIEVAL,
                    candidate.provenanceValue(),
                    CONTROL_PATH + " rank " + candidate.rank());
        }
        assertEquals(
                3,
                response.candidatesFor(CONTROL_PATH).size(),
                "a bypassed field in the batch must not shorten the shortlist of one that "
                        + "was matched");
    }

    @Test
    @DisplayName("the field verdict is still the authority on inheritance, on both servers")
    void inheritanceIsUnchangedByProvenance() {
        MatchResponse bypassed = bypassing.match(bothFields());
        MatchResponse retrieved = matching.match(bothFields());

        assertEquals(
                "MANIFEST_NAME",
                bypassed.inheritableGovernanceFor(REVIEWED_PATH).orElseThrow().code());
        assertEquals(
                "MANIFEST_NAME",
                retrieved.inheritableGovernanceFor(REVIEWED_PATH).orElseThrow().code(),
                "the same class either way. provenance says WHO decided, and this client does "
                        + "not turn that into a second opinion about whether the field may "
                        + "inherit -- fieldDecisions owns that rule and the server publishes it.");
    }
}
