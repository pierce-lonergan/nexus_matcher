package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.Explain;
import io.github.pierce_lonergan.nexusmatcher.model.Governance;
import io.github.pierce_lonergan.nexusmatcher.model.GovernanceStatus;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Decoding, against response bodies captured verbatim from a running service.
 *
 * <p>Deliberately uses a PLAIN {@link ObjectMapper} rather than the one the client configures. That
 * makes this a test of the DTOs' own annotations: an adopter who passes their own strict mapper
 * through {@code objectMapper(...)} has to get the same result, and if the only thing keeping
 * unknown keys survivable were a flag on the client's private mapper, this test would fail.
 */
class MatchResponseDecodingTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("results keep the order the fields were sent in")
    void resultsPreserveRequestOrder() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        assertEquals(
                List.of("booking.passenger.legal_name", "sailing.route_code"),
                List.copyOf(response.paths()),
                "the response is keyed by the caller's own path, in the order sent; a decoder "
                        + "that loses that order silently re-associates governance with columns");
    }

    @Test
    @DisplayName("a matched entry's protection class arrives whole, enhancement included")
    void governanceIsCarriedWhole() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        MatchCandidate top = response.topCandidateFor("booking.passenger.legal_name").orElseThrow();
        assertEquals(1, top.rank());
        assertEquals("GBF-0001", top.governanceId());
        assertEquals(MatchDecision.AUTO_APPROVE, top.decision());
        assertEquals(GovernanceStatus.CONFERRED, top.governanceStatus());

        Governance governance = top.governance();
        assertNotNull(governance);
        assertEquals("MANIFEST_NAME", governance.code());
        assertEquals("Passenger manifest identity", governance.name());
        assertEquals("SEALED_RESTRICTED", governance.classification());
        assertTrue(governance.personalInformation());
        assertTrue(governance.directIdentifier());
        assertEquals(
                "MASK_IN_LOGS",
                governance.enhancement(),
                "the handling instruction is the only member that says what to DO with the field");
    }

    @Test
    @DisplayName("an entry with no protection code decodes as OPEN_TIER, not as a gap")
    void uncodedEntryIsOpenTier() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        MatchCandidate top = response.topCandidateFor("sailing.route_code").orElseThrow();
        assertEquals("GBF-0028", top.governanceId());
        assertNull(top.governance());
        assertEquals(MatchDecision.AUTO_APPROVE, top.decision());
        assertEquals(GovernanceStatus.OPEN_TIER, top.governanceStatus());
        assertTrue(top.governanceValue().isEmpty());

        assertEquals(
                "OPEN_DECK",
                response.vocabulary().openClassification(),
                "the response has to say what its own nulls mean; without this the caller needs "
                        + "the server's vocabulary file to read the answer");
        assertTrue(response.vocabulary().isConfigured());
    }

    @Test
    @DisplayName("a REJECTED runner-up keeps the class its entry confers")
    void rejectedRunnerUpKeepsItsClass() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        List<MatchCandidate> candidates = response.candidatesFor("sailing.route_code");
        MatchCandidate runnerUp = candidates.get(1);

        assertEquals(2, runnerUp.rank());
        assertEquals(MatchDecision.REJECT, runnerUp.decision());
        assertNotNull(
                runnerUp.governance(),
                "nothing inherits from a runner-up, and the class is exactly what lets a reviewer "
                        + "see that rank 1 is a direct identifier and rank 2 is not");
        assertEquals("TIDE_TABLE", runnerUp.governance().code());
        assertEquals(GovernanceStatus.CONFERRED, runnerUp.governanceStatus());
    }

    @Test
    @DisplayName("a REJECTED rank 1 confers nothing, and that is a different null")
    void rejectedTopMatchWithholdsItsClass() throws Exception {
        // NOT a capture. The shipped configuration cannot produce this: final_confidence has a
        // structural floor of semantic_weight * fusion_alpha = 0.63 while review_threshold is
        // 0.50, so no rank-1 candidate can fall below the bar that would reject it. The server
        // documents the clause as reachable only for a caller who raises review_threshold past
        // that floor, and GovernanceNullsIT asserts against the live pack that it does not occur
        // there. This body is therefore hand-built, from the captured shape, to pin what the
        // CLIENT does with it -- which is the half that can be tested without reconfiguring
        // somebody else's server.
        String body = """
                {"results":{"vessel.novel_column":[
                  {"rank":1,"governanceId":"GBF-0028","businessName":"Sailing Route Code",
                   "definition":"The short code that identifies a scheduled route between two \
                terminals.","domain":"Published","governance":null,"confidence":0.41,
                   "decision":"REJECT"}]},
                 "vocabulary":{"openClassification":"OPEN_DECK",
                   "tiersMostOpenFirst":["OPEN_DECK","CREW_ONLY","BRIDGE_SENSITIVE",\
                "SEALED_RESTRICTED"]}}
                """;

        MatchResponse response = mapper.readValue(body, MatchResponse.class);
        MatchCandidate top = response.topCandidateFor("vessel.novel_column").orElseThrow();

        assertEquals(
                GovernanceStatus.WITHHELD_REJECTED_TOP_MATCH,
                top.governanceStatus(),
                "a rejected top match means no entry describes this field; reading it as the open "
                        + "tier would file a novel column as 'governed, openly'");
        assertTrue(top.governanceValue().isEmpty());
    }

    @Test
    @DisplayName("explain reproduces the confidence it arrived with")
    void explainArithmeticCloses() throws Exception {
        MatchResponse response = mapper.readValue(
                Fixtures.captured("match-response-explain.json"), MatchResponse.class);

        MatchCandidate top = response.topCandidateFor("published.terminal_name").orElseThrow();
        Explain explain = top.explainValue().orElseThrow();

        assertEquals(
                top.confidence(),
                explain.recomputedConfidence(),
                1e-5,
                "sum(scores * weights), clamped, is the promise explain exists to keep");
        assertEquals(0.714917, explain.absoluteCosineValue().orElseThrow(), 1e-9);
        assertEquals(1.0, explain.score("fusedRetrieval").orElseThrow(), 1e-9);
    }

    @Test
    @DisplayName("explain is absent, not null-filled, when it was not asked for")
    void explainIsAbsentUnlessRequested() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        assertTrue(response.topCandidateFor("sailing.route_code").orElseThrow()
                .explainValue().isEmpty());
    }

    @Test
    @DisplayName("an unknown response key is ignored; an unknown decision is not")
    void additiveKeysSurviveButUnknownDecisionsDoNot() throws Exception {
        // The additive half is not hypothetical: `vocabulary` and `governance.enhancement` were
        // both added to this contract while this client was being written, and a client that
        // threw on either would have been an outage caused by a server improving.
        String withNewKey = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":0.9,\
                "decision":"AUTO_APPROVE","somethingAddedLater":{"x":1}}]},\
                "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":[],\
                "alsoNew":true},"topLevelAddedLater":7}
                """;
        MatchResponse response = mapper.readValue(withNewKey, MatchResponse.class);
        assertEquals(1, response.topCandidateFor("t.a").orElseThrow().rank());

        String withNewDecision = withNewKey.replace("\"AUTO_APPROVE\"", "\"ESCALATE\"");
        Exception failure = assertThrows(
                Exception.class, () -> mapper.readValue(withNewDecision, MatchResponse.class));
        assertTrue(
                failure.toString().contains("ESCALATE"),
                "a decision this client does not know decides whether a class is applied without "
                        + "a human; mapping it onto the nearest known one would be a silent lie. "
                        + "Got: " + failure);
    }

    @Test
    @DisplayName("a field that matched nothing is an empty list, and asking is safe")
    void emptyCandidateListIsNotAMissingKey() throws Exception {
        String body = """
                {"results":{"t.nothing":[]},"vocabulary":{"openClassification":"OPEN_DECK",\
                "tiersMostOpenFirst":[]}}
                """;
        MatchResponse response = mapper.readValue(body, MatchResponse.class);

        assertTrue(response.paths().contains("t.nothing"));
        assertTrue(response.candidatesFor("t.nothing").isEmpty());
        assertTrue(response.topCandidateFor("t.nothing").isEmpty());
    }

    @Test
    @DisplayName("an unconfigured vocabulary reports itself, rather than looking like a tier")
    void unconfiguredVocabularyIsVisible() throws Exception {
        String body = """
                {"results":{"t.a":[]},"vocabulary":{"openClassification":"UNCLASSIFIED",\
                "tiersMostOpenFirst":[]}}
                """;
        MatchResponse response = mapper.readValue(body, MatchResponse.class);

        assertFalse(
                response.vocabulary().isConfigured(),
                "UNCLASSIFIED is the library's sentinel for 'no vocabulary loaded'; a caller "
                        + "applying it as a tier would be applying a word no taxonomy defines");
        assertTrue(response.vocabulary().openness("OPEN_DECK").isEmpty());
    }

    @Test
    @DisplayName("results are unmodifiable, so a decoded governance artifact cannot be edited")
    void resultsAreUnmodifiable() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        assertThrows(
                UnsupportedOperationException.class,
                () -> response.results().put("injected", List.of()));
    }
}
