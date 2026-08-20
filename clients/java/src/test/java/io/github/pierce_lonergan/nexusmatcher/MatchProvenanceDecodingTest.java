package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchProvenance;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code provenance}: where a candidate's answer came from, against real captured bodies.
 *
 * <p>The fixture this class leans on is
 * {@code captured/match-response-approved-pair.json}, taken verbatim from the fifth fixture
 * server -- the example pack with one reviewer verdict standing. That server exists because the
 * service ships no feedback consumer, so on every other port every candidate is
 * {@link MatchProvenance#RETRIEVAL} and half this vocabulary could only ever have been tested
 * against a body somebody typed. See {@code clients/java/fixture_approved_pair_app.py}.
 *
 * <p>As elsewhere in this directory, a PLAIN {@link ObjectMapper} is used rather than the one the
 * client configures, so what is under test is the DTO's own annotations.
 */
class MatchProvenanceDecodingTest {

    private static final String APPROVED_PAIR_PATH = "booking.passenger.legal_name";
    private static final String RETRIEVED_PATH = "published.terminal_nm";

    private final ObjectMapper mapper = new ObjectMapper();

    private MatchResponse approvedPairCapture() throws Exception {
        return mapper.readValue(
                Fixtures.captured("match-response-approved-pair.json"), MatchResponse.class);
    }

    @Test
    @DisplayName("the pair a client was told to read -- confidence 1.0 and AUTO_APPROVE -- "
            + "separates nothing, and this capture is the evidence")
    void confidenceAndDecisionAreIdenticalOnBothProvenances() throws Exception {
        MatchResponse response = approvedPairCapture();

        MatchCandidate reviewerDecided = response.topCandidateFor(APPROVED_PAIR_PATH).orElseThrow();
        MatchCandidate retrieved = response.topCandidateFor(RETRIEVED_PATH).orElseThrow();

        // Not "close to 1.0" and not "above a threshold". Exactly 1.0, on both, from one live
        // response -- because the five default weights sum to exactly 1.0 and every signal is
        // attainable at 1.0, so the scorer's range INCLUDES the value that was documented as
        // being outside it. `published.terminal_nm` IS the glossary's own logical_name for
        // GBF-0027, which is what drives editDistance to 1.0 alongside the other four.
        assertEquals(1.0, reviewerDecided.confidence(), 0.0);
        assertEquals(1.0, retrieved.confidence(), 0.0);
        assertEquals(MatchDecision.AUTO_APPROVE, reviewerDecided.decision());
        assertEquals(MatchDecision.AUTO_APPROVE, retrieved.decision());

        assertNotEquals(
                reviewerDecided.provenanceValue(),
                retrieved.provenanceValue(),
                "two candidates identical on (confidence, decision) and NOT identical in where "
                        + "they came from: that is the whole reason provenance is a member rather "
                        + "than an inference from those two numbers");
    }

    @Test
    @DisplayName("a candidate a reviewer decided says so, and says it as a value")
    void anApprovedPairIsStatedNotInferred() throws Exception {
        MatchCandidate candidate =
                approvedPairCapture().topCandidateFor(APPROVED_PAIR_PATH).orElseThrow();

        assertEquals("APPROVED_PAIR", candidate.provenance());
        assertEquals(MatchProvenance.APPROVED_PAIR, candidate.provenanceValue());
        assertTrue(candidate.decidedByAReviewer());
        assertFalse(candidate.wasScored());
        assertTrue(candidate.provenanceValue().isKnown());
    }

    @Test
    @DisplayName("a scored candidate says RETRIEVAL, on every rank, not only the winner")
    void retrievedCandidatesSayRetrieval() throws Exception {
        MatchResponse response = approvedPairCapture();

        List<MatchCandidate> retrieved = response.candidatesFor(RETRIEVED_PATH);
        assertEquals(3, retrieved.size(), "top_k was 3 and retrieval ran for this field");
        for (MatchCandidate candidate : retrieved) {
            assertEquals(
                    MatchProvenance.RETRIEVAL,
                    candidate.provenanceValue(),
                    "rank " + candidate.rank() + " of " + RETRIEVED_PATH);
            assertTrue(candidate.wasScored());
            assertFalse(candidate.decidedByAReviewer());
        }
    }

    @Test
    @DisplayName("an approved pair has no absoluteScore and no explain: ABSENT, never 0.0")
    void whatNobodyMeasuredIsAbsentRatherThanZero() throws Exception {
        MatchResponse response = approvedPairCapture();
        MatchCandidate reviewerDecided = response.topCandidateFor(APPROVED_PAIR_PATH).orElseThrow();

        assertNull(reviewerDecided.absoluteScore());
        assertFalse(reviewerDecided.hasAbsoluteScore());
        assertTrue(reviewerDecided.absoluteScoreValue().isEmpty());
        assertNotEquals(
                0.0,
                reviewerDecided.absoluteScoreValue().orElse(-1.0),
                "nothing scored this candidate, so there is no number. Zero would be a "
                        + "measurement -- and on a cosine metric, a very bad one -- which is a "
                        + "claim this response is not making about a term a human chose.");

        assertTrue(
                reviewerDecided.explainValue().isEmpty(),
                "explain was ASKED FOR in this capture and is still absent here: the block "
                        + "promises sum(scores * weights) == confidence, and a candidate with no "
                        + "components cannot keep it. Five zeroes would close the arithmetic by "
                        + "publishing measurements nobody took.");
        assertTrue(
                response.topCandidateFor(RETRIEVED_PATH).orElseThrow().explainValue().isPresent(),
                "and the scored candidate in the SAME response does carry one, so the absence "
                        + "above is about the candidate rather than about the request");

        assertFalse(
                reviewerDecided.wasScored(),
                "wasScored() is what tells the two reasons for a null absoluteScore apart: "
                        + "'the dense arm did not return this entry' and 'nothing measured it'");
    }

    @Test
    @DisplayName("retrieval was skipped, not merely outranked: one candidate where top_k asked "
            + "for three")
    void aBypassedFieldCarriesTheHumansAnswerAlone() throws Exception {
        MatchResponse response = approvedPairCapture();

        assertEquals(
                1,
                response.candidatesFor(APPROVED_PAIR_PATH).size(),
                "ranks 2 and beyond would have to come from retrieval, and retrieval did not "
                        + "run. Inventing runner-ups for a human's answer would present a "
                        + "shortlist nobody produced.");
        assertEquals(
                3,
                response.candidatesFor(RETRIEVED_PATH).size(),
                "the same request, the same top_k, the field that was matched");
    }

    @Test
    @DisplayName("every candidate of an ordinary response is RETRIEVAL")
    void aStockServerAnswersEverythingByRetrieval() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        for (String path : response.paths()) {
            for (MatchCandidate candidate : response.candidatesFor(path)) {
                assertEquals(
                        MatchProvenance.RETRIEVAL,
                        candidate.provenanceValue(),
                        path + " rank " + candidate.rank() + ": the service attaches no feedback "
                                + "consumer on any server it starts, so nothing here can be a "
                                + "human's answer");
            }
        }
    }

    @Test
    @DisplayName("a provenance from a newer server degrades to UNKNOWN and claims nothing")
    void anUnknownProvenanceDoesNotBecomeEitherAnswer() throws Exception {
        // Hand-built on purpose: no server publishes a third value today, and the whole question
        // is what this build does on the day one does. The trap it guards is a closed Java enum,
        // which refuses an unknown constant on decode -- so ONE new word on ONE runner-up would
        // cost every verdict, class and candidate in a response carrying up to 250 fields.
        String body = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":0.9,\
                "decision":"AUTO_APPROVE","absoluteScore":0.7,"provenance":"RULE_ENGINE"}]},\
                "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":[]}}
                """;

        MatchCandidate candidate =
                mapper.readValue(body, MatchResponse.class).topCandidateFor("t.a").orElseThrow();

        assertEquals(MatchProvenance.UNKNOWN, candidate.provenanceValue());
        assertFalse(candidate.provenanceValue().isKnown());
        assertEquals(
                "RULE_ENGINE",
                candidate.provenance(),
                "the server's own word survives, so an operator can name it in a ticket rather "
                        + "than count anonymous unknowns");

        assertFalse(
                candidate.decidedByAReviewer(),
                "an unread value must never become a quiet 'a human approved this'");
        assertFalse(
                candidate.wasScored(),
                "and it must not become a quiet 'the pipeline measured this' either -- both "
                        + "answers are no, and provenance() is where you go to find out which");
    }

    @Test
    @DisplayName("a server predating the member sends nothing, and nothing is not RETRIEVAL")
    void anAbsentProvenanceIsNotDefaultedToRetrieval() throws Exception {
        // The bypass existed before this member did -- it was identifiable only by a magic
        // confidence, which is the defect the member closes -- so an older server's silence is
        // genuinely uninformative about where its answer came from. Defaulting it to RETRIEVAL
        // would put the false claim back, one version skew along.
        String body = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":1.0,\
                "decision":"AUTO_APPROVE","absoluteScore":null}]},\
                "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":[]}}
                """;

        MatchCandidate candidate =
                mapper.readValue(body, MatchResponse.class).topCandidateFor("t.a").orElseThrow();

        assertNull(candidate.provenance(), "the key was not on the wire at all");
        assertEquals(MatchProvenance.UNKNOWN, candidate.provenanceValue());
        assertFalse(candidate.decidedByAReviewer());
        assertFalse(candidate.wasScored());
    }

    @Test
    @DisplayName("UNKNOWN is a client-side sentinel and is not matched off the wire by name")
    void theSentinelIsNotReachableFromTheWire() {
        assertEquals(MatchProvenance.RETRIEVAL, MatchProvenance.fromWire("RETRIEVAL"));
        assertEquals(MatchProvenance.APPROVED_PAIR, MatchProvenance.fromWire("APPROVED_PAIR"));
        assertEquals(MatchProvenance.UNKNOWN, MatchProvenance.fromWire(null));
        assertEquals(MatchProvenance.UNKNOWN, MatchProvenance.fromWire("retrieval"));

        // The sharp one. A server that started publishing the literal string "UNKNOWN" would be
        // saying something; reading it as "this client did not understand you" is a different
        // claim, and it is the claim that would silently absorb a real value.
        assertEquals(
                MatchProvenance.UNKNOWN,
                MatchProvenance.fromWire("UNKNOWN"),
                "it still lands on the sentinel -- there is nowhere else for it to go -- but it "
                        + "gets there by NOT matching, and provenance() still carries the string. "
                        + "tests/packaging/test_java_client_contract.py fails the build if the "
                        + "service ever publishes this value, which is the real defence.");
        assertFalse(MatchProvenance.UNKNOWN.isKnown());
        assertTrue(MatchProvenance.RETRIEVAL.isKnown());
        assertTrue(MatchProvenance.APPROVED_PAIR.isKnown());
    }

    @Test
    @DisplayName("provenance does not decide inheritance; the field verdict still does")
    void inheritanceStillReadsTheFieldVerdict() throws Exception {
        MatchResponse response = approvedPairCapture();

        // Both fields are AUTO_APPROVE on this deployment, so both inherit -- the reviewer-decided
        // one is not privileged and the retrieved one is not held back. This client holds no
        // second opinion about the server's roll-up rule, and decidedByAReviewer() is deliberately
        // not a licence to apply a class: it answers WHO decided, which is the fact no other
        // member carries.
        assertEquals(
                "MANIFEST_NAME",
                response.inheritableGovernanceFor(APPROVED_PAIR_PATH).orElseThrow().code());
        assertEquals(
                "TIDE_TABLE",
                response.inheritableGovernanceFor(RETRIEVED_PATH).orElseThrow().code());
        assertTrue(
                response.topCandidateFor(APPROVED_PAIR_PATH).orElseThrow().decidedByAReviewer());
        assertFalse(response.topCandidateFor(RETRIEVED_PATH).orElseThrow().decidedByAReviewer());
    }
}
