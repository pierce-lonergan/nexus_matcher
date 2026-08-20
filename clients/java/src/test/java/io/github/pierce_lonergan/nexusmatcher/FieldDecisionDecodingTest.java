package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.FieldDecision;
import io.github.pierce_lonergan.nexusmatcher.model.FieldVerdict;
import io.github.pierce_lonergan.nexusmatcher.model.Governance;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The per-field verdict, and what this client does with one it has never heard of.
 *
 * <p>This is the file that argues the asymmetry. {@link MatchDecision} refuses an unknown value and
 * loses the whole response; {@link FieldDecision} accepts one, names it, and refuses to act on it.
 * Both behaviours are asserted here, side by side, because the interesting claim is not either one
 * alone -- it is that the two are different on purpose.
 *
 * <p>The unknown-value bodies are hand-built and say so, for the reason
 * {@code MatchResponseDecodingTest#rejectedTopMatchWithholdsItsClass} gives for its own: no server
 * that exists sends a fifth verdict, and the whole point of the behaviour is what happens the day
 * one does. What can be tested against a real service -- that the four known values decode, and
 * that NO_MATCH really does arrive with candidates -- is tested against real captures and in
 * {@link FieldDecisionIT} against a live one.
 */
class FieldDecisionDecodingTest {

    private final ObjectMapper mapper = new ObjectMapper();

    /** A response body with one field, whose verdict is whatever is passed in. */
    private static String bodyWithVerdict(String verdict) {
        return """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":{"code":"TIDE_TABLE",\
                "name":"Published sailing and tide data","classification":"OPEN_DECK",\
                "personalInformation":false,"directIdentifier":false,"enhancement":null},\
                "confidence":0.9,"decision":"AUTO_APPROVE","absoluteScore":0.88}]},\
                "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":[]},\
                "fieldDecisions":{"t.a":"%s"}}
                """.formatted(verdict);
    }

    @Test
    @DisplayName("the four published verdicts decode to their constants and keep their wire string")
    void publishedVerdictsDecode() throws Exception {
        for (FieldDecision expected : List.of(
                FieldDecision.AUTO_APPROVE,
                FieldDecision.REVIEW,
                FieldDecision.REJECT,
                FieldDecision.NO_MATCH)) {

            MatchResponse response =
                    mapper.readValue(bodyWithVerdict(expected.name()), MatchResponse.class);
            FieldVerdict verdict = response.verdictFor("t.a").orElseThrow();

            assertEquals(expected, verdict.decision());
            assertEquals(expected.name(), verdict.wireValue());
            assertTrue(verdict.isKnown());
        }
    }

    @Test
    @DisplayName("a fifth verdict from a newer server does not cost the rest of the response")
    void unknownVerdictDegradesRatherThanFailing() throws Exception {
        // NOT a capture: no server sends this, which is exactly why the behaviour needs pinning
        // before one does. The name is invented and deliberately plausible -- the dangerous
        // unknown value is the one a client is tempted to map onto AUTO_APPROVE.
        MatchResponse response =
                mapper.readValue(bodyWithVerdict("APPROVE_WITH_CONDITIONS"), MatchResponse.class);

        FieldVerdict verdict = response.verdictFor("t.a").orElseThrow();

        assertEquals(
                FieldDecision.UNKNOWN,
                verdict.decision(),
                "an unrecognised verdict must not be guessed at; UNKNOWN is the absence of an "
                        + "answer, not the nearest one");
        assertEquals(
                "APPROVE_WITH_CONDITIONS",
                verdict.wireValue(),
                "the raw string is the only thing that tells an operator WHICH new verdict "
                        + "arrived, and therefore what to search the release notes for. Losing it "
                        + "turns forty unreadable columns into forty identical mysteries.");
        assertFalse(verdict.isKnown());
        assertEquals(List.of("t.a"), response.pathsWithUnknownVerdicts());

        // And the rest of the response survived, which is the whole reason for degrading.
        MatchCandidate top = response.topCandidateFor("t.a").orElseThrow();
        assertEquals("GBF-0027", top.governanceId());
        assertEquals(MatchDecision.AUTO_APPROVE, top.decision());
        assertNotNull(top.governance());
    }

    @Test
    @DisplayName("an unknown verdict is inert: it grants nothing, however good rank 1 looks")
    void unknownVerdictGrantsNothing() throws Exception {
        MatchResponse response =
                mapper.readValue(bodyWithVerdict("APPROVE_WITH_CONDITIONS"), MatchResponse.class);

        assertFalse(response.verdictFor("t.a").orElseThrow().maySafelyInherit());
        assertTrue(
                response.inheritableGovernanceFor("t.a").isEmpty(),
                "rank 1 here is AUTO_APPROVE at confidence 0.9 with a populated class, and the "
                        + "field verdict is a word this client cannot read. Inheriting would be "
                        + "acting on a server instruction nobody in this build has understood.");

        // The evidence is still reachable -- degrading loses the verdict, not the response.
        Governance rankOne = response.topCandidateFor("t.a").orElseThrow().governance();
        assertEquals("TIDE_TABLE", rankOne.code());
    }

    @Test
    @DisplayName("the literal string UNKNOWN from a server is NOT read as 'this client is lost'")
    void serverSentUnknownIsItsOwnValue() throws Exception {
        // The sentinel's one sharp edge. If a future server ever publishes the value UNKNOWN, this
        // client must not silently absorb it into its own "I could not read that" state: those are
        // different claims, and conflating them would hide a real verdict behind a client-side
        // excuse. So the decoder never matches the sentinel by name, and the raw string is what
        // distinguishes the two cases.
        MatchResponse response = mapper.readValue(bodyWithVerdict("UNKNOWN"), MatchResponse.class);
        FieldVerdict verdict = response.verdictFor("t.a").orElseThrow();

        assertEquals(FieldDecision.UNKNOWN, verdict.decision());
        assertEquals("UNKNOWN", verdict.wireValue());
        assertFalse(verdict.maySafelyInherit());
    }

    @Test
    @DisplayName("an unknown CANDIDATE decision still fails hard, and that is the deliberate half")
    void unknownMatchDecisionStillFailsLoudly() {
        String body = bodyWithVerdict("REVIEW").replace("\"AUTO_APPROVE\"", "\"ESCALATE\"");

        Exception failure =
                assertThrows(Exception.class, () -> mapper.readValue(body, MatchResponse.class));

        assertTrue(
                failure.toString().contains("ESCALATE"),
                "MatchDecision is the vocabulary the service has committed to FREEZING -- adding "
                        + "NO_MATCH to a separate enum rather than to this one is that commitment "
                        + "in action -- so a fifth value here means the contract moved under this "
                        + "client, not that the client is behind. Got: " + failure);
    }

    @Test
    @DisplayName("a verdict round-trips to the string it arrived as, known or not")
    void verdictsRoundTripToTheirWireValue() throws Exception {
        assertEquals("\"NO_MATCH\"", mapper.writeValueAsString(FieldVerdict.fromWire("NO_MATCH")));
        assertEquals(
                "\"APPROVE_WITH_CONDITIONS\"",
                mapper.writeValueAsString(FieldVerdict.fromWire("APPROVE_WITH_CONDITIONS")),
                "re-encoding an unrecognised verdict as the sentinel's own name would put a word "
                        + "the server never said into whatever this response is written to next");
    }

    @Test
    @DisplayName("only AUTO_APPROVE permits inheritance, and each 'no' keeps its own identity")
    void onlyAutoApproveGrants() {
        assertTrue(FieldDecision.AUTO_APPROVE.maySafelyInherit());
        for (FieldDecision refusing : List.of(
                FieldDecision.REVIEW,
                FieldDecision.REJECT,
                FieldDecision.NO_MATCH,
                FieldDecision.UNKNOWN)) {
            assertFalse(refusing.maySafelyInherit(), refusing + " must not grant inheritance");
        }
        assertTrue(FieldDecision.NO_MATCH.isKnown(), "NO_MATCH is an answer; UNKNOWN is not");
        assertFalse(FieldDecision.UNKNOWN.isKnown());
    }

    @Test
    @DisplayName("a server that sends no fieldDecisions gets no invented ones")
    void absentFieldDecisionsAreNotDefaultedFromRankOne() throws Exception {
        // What an older server's body looks like. The temptation is to fall back to rank 1's own
        // decision, which would be a verdict this client made up and attributed to the server.
        String body = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":0.9,\
                "decision":"AUTO_APPROVE"}]},"vocabulary":{"openClassification":"OPEN_DECK",\
                "tiersMostOpenFirst":[]}}
                """;
        MatchResponse response = mapper.readValue(body, MatchResponse.class);

        assertTrue(response.fieldDecisions().isEmpty());
        assertTrue(response.verdictFor("t.a").isEmpty());
        assertTrue(
                response.inheritableGovernanceFor("t.a").isEmpty(),
                "no verdict means no permission, not permission by default");
        assertTrue(response.scoringValue().isEmpty());
        assertTrue(response.pathsWithUnknownVerdicts().isEmpty());
    }
}
