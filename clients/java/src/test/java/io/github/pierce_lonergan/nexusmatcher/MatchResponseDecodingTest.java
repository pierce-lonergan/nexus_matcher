package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.Explain;
import io.github.pierce_lonergan.nexusmatcher.model.FieldDecision;
import io.github.pierce_lonergan.nexusmatcher.model.Governance;
import io.github.pierce_lonergan.nexusmatcher.model.GovernanceStatus;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.ScoringContract;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
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
                   "decision":"REJECT","absoluteScore":0.21}]},
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
        assertEquals(0.891667, explain.absoluteCosineValue().orElseThrow(), 1e-9);
        assertEquals(1.0, explain.score("fusedRetrieval").orElseThrow(), 1e-9);

        assertEquals(
                explain.absoluteCosineValue().orElseThrow(),
                top.absoluteScoreValue().orElseThrow(),
                0.0,
                "the server promises these are the SAME number -- absoluteScore is the one a new "
                        + "client should read, and absoluteCosine is kept for clients already "
                        + "reading it. Exact equality, not a tolerance: two renderings of one "
                        + "value cannot legitimately differ at all.");
    }

    @Test
    @DisplayName("absoluteScore is on every candidate without asking, and it is not the confidence")
    void absoluteScoreIsPresentWithoutExplain() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);

        for (String path : response.paths()) {
            for (MatchCandidate candidate : response.candidatesFor(path)) {
                assertTrue(
                        candidate.hasAbsoluteScore(),
                        path + " rank " + candidate.rank() + " carries no absoluteScore; it is "
                                + "published on every candidate and is not gated behind explain");
                assertTrue(candidate.explainValue().isEmpty(), "explain was not requested");
            }
        }

        // The reason the member was promoted out of `explain`: it disagrees with `confidence`, and
        // the disagreement is the information. The worst candidate of sailing.route_code scores
        // well above its own confidence, because confidence is min-max normalised inside one
        // field's shortlist and this number is not normalised at all.
        List<MatchCandidate> routeCode = response.candidatesFor("sailing.route_code");
        MatchCandidate worst = routeCode.get(routeCode.size() - 1);
        assertTrue(
                worst.absoluteScoreValue().orElseThrow() > worst.confidence(),
                "expected the raw score to sit above the normalised confidence for this field's "
                        + "worst candidate; got absolute=" + worst.absoluteScore()
                        + " confidence=" + worst.confidence());
    }

    @Test
    @DisplayName("a null absoluteScore stays null: it means 'not measured', never 0.0")
    void nullAbsoluteScoreIsNotZero() throws Exception {
        // Hand-built, and the reason is worth stating: on the shipped wiring the dense arm returns
        // every candidate, so a null absoluteScore cannot be provoked against the example pack. It
        // is reachable on any deployment whose lexical arm surfaces an entry the dense arm did
        // not, and the damage it would do is silent -- which is why the CLIENT half is pinned here
        // rather than left until the day it happens. The trap is a primitive `double` component:
        // Jackson binds a JSON null to 0.0 without a word, and 0.0 on a cosine metric is a real
        // and very bad score, so a caller filtering `>= floor` would discard the candidate as
        // failed rather than as unmeasured.
        String body = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":0.9,\
                "decision":"AUTO_APPROVE","absoluteScore":null}]},\
                "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":[]}}
                """;
        MatchCandidate candidate =
                mapper.readValue(body, MatchResponse.class).topCandidateFor("t.a").orElseThrow();

        assertNull(candidate.absoluteScore());
        assertFalse(candidate.hasAbsoluteScore());
        assertTrue(candidate.absoluteScoreValue().isEmpty());
        assertNotEquals(
                0.0,
                candidate.absoluteScoreValue().orElse(-1.0),
                "an unmeasured score must not arrive as zero; zero means 'measured, and as far "
                        + "from the query as this metric goes'");
    }

    @Test
    @DisplayName("the scoring block says which numbers may be compared across fields")
    void scoringContractIsCarried() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);
        ScoringContract scoring = response.scoringValue().orElseThrow();

        assertTrue(
                scoring.comparableAcrossFields("absoluteScore"),
                "this is the point of the block: absoluteScore is the cross-field number");
        assertFalse(
                scoring.comparableAcrossFields("confidence"),
                "and confidence is not, however much it looks like a quality score");
        assertEquals("ACROSS_FIELDS", scoring.scopeOf("absoluteScore").orElseThrow());
        assertEquals("WITHIN_FIELD", scoring.scopeOf("confidence").orElseThrow());
        assertTrue(
                scoring.scopeWidth("ACROSS_FIELDS").orElseThrow()
                        > scoring.scopeWidth("WITHIN_FIELD").orElseThrow(),
                "the ordering rides on the response so a client need not hard-code it");

        assertTrue(scoring.absoluteScoreIsCosine());
        assertFalse(scoring.absoluteScorePooledOverAliases());
        assertEquals(0.63, scoring.confidenceFloorValue().orElseThrow(), 1e-9);
        assertTrue(
                scoring.absoluteScoreFloorValue().isEmpty(),
                "the library ships no absolute floor and will not invent one; empty here means "
                        + "NO_MATCH can only come from a field with no candidates at all");
    }

    @Test
    @DisplayName("a NO_MATCH field arrives WITH candidates, and they must not be inherited")
    void noMatchStillCarriesCandidates() throws Exception {
        // A real capture, from the fixture server that configures an absolute-score floor. Every
        // part of this case is a trap for a client that reads rank 1 instead of the field verdict.
        MatchResponse response = mapper.readValue(
                Fixtures.captured("match-response-no-match.json"), MatchResponse.class);
        String path = "vessel.safety.lifejacket_locker_inspection_due";

        assertEquals(FieldDecision.NO_MATCH, response.verdictFor(path).orElseThrow().decision());

        List<MatchCandidate> candidates = response.candidatesFor(path);
        assertFalse(
                candidates.isEmpty(),
                "the server deliberately did NOT return an empty list -- the candidates are "
                        + "evidence for the reviewer who now has to decide. A client that looks "
                        + "for no-match fields by testing for an empty list finds none of them.");

        MatchCandidate top = candidates.get(0);
        assertNotNull(
                top.governance(),
                "and rank 1 carries a real protection class, which is what makes reading it "
                        + "instead of the field verdict such an easy mistake");
        assertEquals(MatchDecision.REVIEW, top.decision());
        assertTrue(
                top.confidence() > 0.8,
                "at confidence " + top.confidence() + ", too -- min-max normalisation puts the "
                        + "best of a hopeless shortlist near the top, which is exactly why "
                        + "confidence cannot express 'nothing matched'");

        assertTrue(
                response.inheritableGovernanceFor(path).isEmpty(),
                "so the field inherits NOTHING, however good rank 1 looks");

        // The absolute score is the number that knows better, and the response says so.
        ScoringContract scoring = response.scoringValue().orElseThrow();
        double floor = scoring.absoluteScoreFloorValue().orElseThrow();
        assertTrue(
                top.absoluteScoreValue().orElseThrow() < floor,
                "rank 1's absolute score should sit below the configured floor; got "
                        + top.absoluteScore() + " against " + floor);

        // And the field the glossary really does describe, in the same response, is unaffected.
        String matched = "booking.passenger.legal_name";
        assertEquals(
                FieldDecision.AUTO_APPROVE, response.verdictFor(matched).orElseThrow().decision());
        assertEquals(
                "MANIFEST_NAME",
                response.inheritableGovernanceFor(matched).orElseThrow().code());
    }

    @Test
    @DisplayName("an AUTO_APPROVE field at the open tier inherits nothing, and is still governed")
    void anApprovedOpenTierFieldIsGovernedOpenly() throws Exception {
        // The overloaded-empty case, from a real capture. sailing.route_code is AUTO_APPROVE and
        // its rank-1 entry carries no protection code, so there is genuinely no class to apply --
        // but the field is governed, AS OPEN, and the correct action is to file it at
        // vocabulary().openClassification() rather than to send it to a human. Empty from
        // inheritableGovernanceFor means "no class from this response"; only the verdict says
        // which kind of no.
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);
        String path = "sailing.route_code";

        assertEquals(
                FieldDecision.AUTO_APPROVE, response.verdictFor(path).orElseThrow().decision());
        assertTrue(response.inheritableGovernanceFor(path).isEmpty());
        assertEquals(
                GovernanceStatus.OPEN_TIER,
                response.topCandidateFor(path).orElseThrow().governanceStatus(),
                "inside an AUTO_APPROVE branch this is the only way empty can happen, because "
                        + "an approved field has a rank 1 and a rank 1 that confers nothing sits "
                        + "at the open tier");
        assertEquals("OPEN_DECK", response.vocabulary().openClassification());
    }

    @Test
    @DisplayName("fieldDecisions is keyed and ordered exactly like results")
    void fieldDecisionsMirrorResults() throws Exception {
        MatchResponse response = mapper.readValue(
                Fixtures.captured("match-response-no-match.json"), MatchResponse.class);

        assertEquals(
                List.copyOf(response.paths()),
                List.copyOf(response.fieldDecisions().keySet()),
                "the two maps carry the same keys in the same order -- the conservation law this "
                        + "endpoint is built around, one member over. A verdict map short one key "
                        + "is a column with no verdict and nothing saying so.");
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
        // both added to this contract while this client was being written, and `absoluteScore`,
        // `fieldDecisions` and `scoring` arrived after it shipped. A client that threw on any of
        // them would have been an outage caused by a server improving.
        String withNewKey = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":0.9,\
                "decision":"AUTO_APPROVE","absoluteScore":0.7,"somethingAddedLater":{"x":1}}]},\
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
    @DisplayName("results and fieldDecisions are unmodifiable, so the artifact cannot be edited")
    void decodedMapsAreUnmodifiable() throws Exception {
        MatchResponse response = mapper.readValue(
                Fixtures.captured("match-response-no-match.json"), MatchResponse.class);

        assertThrows(
                UnsupportedOperationException.class,
                () -> response.results().put("injected", List.of()));
        assertThrows(
                UnsupportedOperationException.class,
                () -> response.fieldDecisions().remove(
                        "vessel.safety.lifejacket_locker_inspection_due"),
                "removing a NO_MATCH verdict from a decoded response would leave the candidates "
                        + "behind with nothing saying not to inherit them");
    }
}
