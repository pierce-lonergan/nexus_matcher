package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.FieldGovernance;
import io.github.pierce_lonergan.nexusmatcher.model.GovernanceOutcome;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.Vocabulary;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.EnumSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link MatchResponse#governanceFor(String)}: the four reasons a column gets no protection class,
 * told apart from the response alone.
 *
 * <p>A consumer maps {@code governance.code} onto who may read a column, so what the ABSENCE of a
 * code means decides whether somebody sees data they should not. There are four absences, they
 * demand three different follow-ups, and the cheap reading -- "no class, so it is open" -- is
 * correct for exactly one of them.
 *
 * <p>Three of the seven outcomes come off bodies captured verbatim from a running service. Three
 * cannot: they need a server configured in a way no shipped fixture is, and the fourth needs a
 * server newer than this build. Those four are hand-built here, each with the reason it has to be,
 * following the precedent {@code MatchResponseDecodingTest#rejectedTopMatchWithholdsItsClass} set
 * for the same situation. The reachable ones are also driven against a real service in
 * {@code FieldGovernanceIT}, which is where the claim that these bodies are the right SHAPE is
 * tested.
 *
 * <p>A plain {@link ObjectMapper}, like every other decoding test here, so this exercises the DTOs'
 * own annotations rather than a flag on the client's private mapper.
 */
class FieldGovernanceTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private MatchResponse captured(String name) throws Exception {
        return mapper.readValue(Fixtures.captured(name), MatchResponse.class);
    }

    private MatchResponse decoded(String body) throws Exception {
        return mapper.readValue(body, MatchResponse.class);
    }

    // =========================================================================
    // THE THREE THAT COME OFF A REAL CAPTURE
    // =========================================================================

    @Test
    @DisplayName("CONFERRED: an AUTO_APPROVE column whose rank 1 carries a class")
    void aConferredClassIsHandedStraightBack() throws Exception {
        FieldGovernance governance =
                captured("match-response.json").governanceFor("booking.passenger.legal_name");

        assertEquals(GovernanceOutcome.CONFERRED, governance.outcome());
        assertEquals("MANIFEST_NAME", governance.conferred().orElseThrow().code());
        assertTrue(governance.maySafelyApply());
        assertFalse(governance.needsAHuman());
        assertTrue(
                governance.openTier().isEmpty(),
                "openTier() is for the OPEN_TIER outcome alone; a value here would let a caller "
                        + "read a tier name off a column that has a real class");
    }

    @Test
    @DisplayName("OPEN_TIER: the entry carries no code, and the response names the tier")
    void anUncodedEntryYieldsTheDeploymentsOwnOpenTier() throws Exception {
        MatchResponse response = captured("match-response.json");
        FieldGovernance governance = response.governanceFor("sailing.route_code");

        assertEquals(GovernanceOutcome.OPEN_TIER, governance.outcome());
        assertEquals(
                "OPEN_DECK",
                governance.openTier().orElseThrow(),
                "the tier comes from the response's own vocabulary block, so a caller never needs "
                        + "the server's vocabulary FILE to act on this outcome");
        assertEquals(response.vocabulary().openClassification(), governance.openTier().orElseThrow());
        assertTrue(governance.maySafelyApply());
        assertTrue(
                governance.conferred().isEmpty(),
                "there is no class to confer here -- the tier is the whole answer");
    }

    @Test
    @DisplayName("WITHHELD_NO_MATCH: nothing cleared the floor, and rank 1 still carries a class")
    void aNoMatchColumnInheritsNothingHoweverGoodItsCandidatesLook() throws Exception {
        MatchResponse response = captured("match-response-no-match.json");
        String path = "vessel.safety.lifejacket_locker_inspection_due";

        FieldGovernance governance = response.governanceFor(path);

        assertEquals(GovernanceOutcome.WITHHELD_NO_MATCH, governance.outcome());
        assertFalse(governance.maySafelyApply());
        assertTrue(governance.needsAHuman());
        assertTrue(governance.conferred().isEmpty());
        assertTrue(governance.openTier().isEmpty());

        // The trap, stated against the same response: the candidate DOES carry a class, and a
        // client reading rank 1 directly would apply it.
        assertFalse(response.candidatesFor(path).isEmpty());
        assertEquals(
                "HARBOUR_TERMS",
                response.candidatesFor(path).get(0).governance().code(),
                "a NO_MATCH column arrives with a populated rank-1 class on purpose -- the "
                        + "candidates are evidence for a reviewer. If this stopped being true the "
                        + "outcome above would be trivially safe and this test would prove less.");
    }

    // =========================================================================
    // THE FOUR THAT NO SHIPPED FIXTURE CAN PRODUCE
    // =========================================================================

    @Test
    @DisplayName("WITHHELD_REJECTED_TOP_MATCH: nothing in the glossary describes the column")
    void aRejectedTopMatchConfersNothing() throws Exception {
        // NOT a capture, and it cannot be one. A rank-1 REJECT needs review_threshold above the
        // structural rank-1 confidence floor of semantic_weight * fusion_alpha = 0.63, and the
        // shipped review_threshold is 0.50 -- GovernanceNullsIT asserts against the live pack that
        // no rank 1 is REJECT there. Hand-built from the captured shape, exactly as
        // MatchResponseDecodingTest#rejectedTopMatchWithholdsItsClass is and for the same reason.
        String body = """
                {"results":{"vessel.novel_column":[
                  {"rank":1,"governanceId":"GBF-0028","businessName":"Sailing Route Code",
                   "definition":"The short code that identifies a scheduled route.",
                   "domain":"Published","governance":null,"confidence":0.41,
                   "decision":"REJECT","absoluteScore":0.21}]},
                 "fieldDecisions":{"vessel.novel_column":"REJECT"},
                 "vocabulary":{"openClassification":"OPEN_DECK",
                   "tiersMostOpenFirst":["OPEN_DECK","CREW_ONLY"]}}
                """;

        FieldGovernance governance = decoded(body).governanceFor("vessel.novel_column");

        assertEquals(GovernanceOutcome.WITHHELD_REJECTED_TOP_MATCH, governance.outcome());
        assertFalse(governance.maySafelyApply());
        assertTrue(governance.needsAHuman());
        assertTrue(
                governance.openTier().isEmpty(),
                "the deployment HAS an open tier and this column is not at it. Handing OPEN_DECK "
                        + "back here would file a column the server cannot identify as 'governed, "
                        + "openly', which is the fail-open this outcome exists to prevent.");
    }

    @Test
    @DisplayName("UNCLASSIFIABLE_NO_VOCABULARY: no vocabulary loaded, so nothing was classified")
    void anUnconfiguredDeploymentIsNotAnOpenTier() throws Exception {
        // NOT a capture. Every fixture server this suite starts is loaded with the example pack's
        // protection_classes.json, and it has to be: NexusMatcher refuses to index an entry whose
        // protection code its vocabulary cannot resolve, so an unconfigured deployment can only
        // exist over a glossary that carries no codes either -- a sixth fixture server with its own
        // glossary file. This body is what such a server sends, which is the SAME body as the
        // OPEN_TIER capture above with one string changed.
        String body = """
                {"results":{"sailing.route_code":[
                  {"rank":1,"governanceId":"GBF-0028","businessName":"Sailing Route Code",
                   "definition":"The short code that identifies a scheduled route.",
                   "domain":"Published","governance":null,"confidence":0.83,
                   "decision":"AUTO_APPROVE","absoluteScore":0.77}]},
                 "fieldDecisions":{"sailing.route_code":"AUTO_APPROVE"},
                 "vocabulary":{"openClassification":"UNCLASSIFIED","tiersMostOpenFirst":[]}}
                """;

        MatchResponse response = decoded(body);
        FieldGovernance governance = response.governanceFor("sailing.route_code");

        assertEquals(GovernanceOutcome.UNCLASSIFIABLE_NO_VOCABULARY, governance.outcome());
        assertFalse(
                governance.maySafelyApply(),
                "nothing on this server is classified, so there is no classification to apply. "
                        + "The follow-up is an operator wiring a vocabulary.");
        assertFalse(
                governance.needsAHuman(),
                "and it is not a review either: routing every column on a misconfigured server to "
                        + "a review queue buries one configuration defect under a schema of tickets");
        assertTrue(
                governance.openTier().isEmpty(),
                "UNCLASSIFIED is this library's sentinel for 'no vocabulary configured'. Handing "
                        + "it back as a tier name is how it ends up written into somebody's "
                        + "permission model as though a person had chosen it.");
        assertFalse(response.vocabulary().isConfigured());
    }

    @Test
    @DisplayName("the open tier and an unconfigured server differ ONLY in the vocabulary block")
    void theTwoBodiesAreIdenticalExceptForOneString() throws Exception {
        // The whole reason governanceFor() consults the vocabulary at all, shown rather than
        // argued. These two bodies are the same bytes but for `openClassification` and the tier
        // list -- same candidate, same id, same null class, same AUTO_APPROVE verdict -- and they
        // mean opposite things. Nothing inside `results` or `fieldDecisions` can separate them.
        String candidate = """
                {"results":{"sailing.route_code":[
                  {"rank":1,"governanceId":"GBF-0028","businessName":"Sailing Route Code",
                   "definition":"The short code that identifies a scheduled route.",
                   "domain":"Published","governance":null,"confidence":0.83,
                   "decision":"AUTO_APPROVE","absoluteScore":0.77}]},
                 "fieldDecisions":{"sailing.route_code":"AUTO_APPROVE"},
                 "vocabulary":""";
        MatchResponse configured = decoded(candidate
                + "{\"openClassification\":\"OPEN_DECK\",\"tiersMostOpenFirst\":[\"OPEN_DECK\"]}}");
        MatchResponse unconfigured = decoded(candidate
                + "{\"openClassification\":\"UNCLASSIFIED\",\"tiersMostOpenFirst\":[]}}");

        assertEquals(
                configured.results(),
                unconfigured.results(),
                "the two responses agree candidate for candidate; if that ever stops being true, "
                        + "re-derive what separates these cases before trusting the claim below");
        assertEquals(configured.fieldDecisions(), unconfigured.fieldDecisions());
        assertNotEquals(configured.vocabulary(), unconfigured.vocabulary());

        assertEquals(
                GovernanceOutcome.OPEN_TIER,
                configured.governanceFor("sailing.route_code").outcome());
        assertEquals(
                GovernanceOutcome.UNCLASSIFIABLE_NO_VOCABULARY,
                unconfigured.governanceFor("sailing.route_code").outcome());
    }

    @Test
    @DisplayName("the reading this helper replaces would have applied the sentinel as a tier")
    void theUnsafeReadingIsTheOneThatUsedToBeShorter() throws Exception {
        // Not a test of the server. A test of WHY this helper exists: the two-call reading that
        // was previously the obvious one -- and that the client's own javadoc example showed --
        // produces a tier name on the unconfigured deployment, and that name is the sentinel.
        String body = """
                {"results":{"sailing.route_code":[
                  {"rank":1,"governanceId":"GBF-0028","businessName":"Sailing Route Code",
                   "definition":"The short code that identifies a scheduled route.",
                   "domain":"Published","governance":null,"confidence":0.83,
                   "decision":"AUTO_APPROVE","absoluteScore":0.77}]},
                 "fieldDecisions":{"sailing.route_code":"AUTO_APPROVE"},
                 "vocabulary":{"openClassification":"UNCLASSIFIED","tiersMostOpenFirst":[]}}
                """;
        MatchResponse response = decoded(body);

        String whatTheOldReadingWouldApply = response
                .inheritableGovernanceFor("sailing.route_code")
                .map(governance -> governance.classification())
                .orElseGet(() -> response.vocabulary().openClassification());

        assertEquals(
                Vocabulary.UNCONFIGURED_OPEN_CLASSIFICATION,
                whatTheOldReadingWouldApply,
                "the short reading writes the library's 'no vocabulary configured' sentinel into "
                        + "the caller's permission model as though it were a tier somebody chose");
        assertFalse(response.governanceFor("sailing.route_code").maySafelyApply());
    }

    @Test
    @DisplayName("UNREADABLE: a verdict from a newer server, and a path that was never sent")
    void anUnreadableVerdictNeverBecomesAnApproval() throws Exception {
        // The first half cannot be captured by construction: it is a value a FUTURE server sends.
        // tests/packaging/test_java_client_contract.py asserts the service publishes no such
        // verdict today, which is what makes this a forward-compatibility test rather than a
        // description of a body that exists.
        String body = """
                {"results":{"sailing.route_code":[
                  {"rank":1,"governanceId":"GBF-0028","businessName":"Sailing Route Code",
                   "definition":"The short code that identifies a scheduled route.",
                   "domain":"Published","governance":null,"confidence":0.83,
                   "decision":"AUTO_APPROVE","absoluteScore":0.77}]},
                 "fieldDecisions":{"sailing.route_code":"APPROVE_WITH_CONDITIONS"},
                 "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":["OPEN_DECK"]}}
                """;
        MatchResponse response = decoded(body);

        FieldGovernance unknownVerdict = response.governanceFor("sailing.route_code");
        assertEquals(GovernanceOutcome.UNREADABLE, unknownVerdict.outcome());
        assertFalse(
                unknownVerdict.maySafelyApply(),
                "a verdict this build cannot read must never become a quiet approval -- that is "
                        + "how a new 'APPROVE_WITH_CONDITIONS' turns into an unconditional one");
        assertTrue(unknownVerdict.needsAHuman());
        assertEquals(
                "APPROVE_WITH_CONDITIONS",
                response.verdictFor("sailing.route_code").orElseThrow().wireValue(),
                "and the value the server sent is still readable, so the column can be named in a "
                        + "ticket rather than merely counted");

        FieldGovernance neverSent = response.governanceFor("some.path.nobody.asked.about");
        assertEquals(GovernanceOutcome.UNREADABLE, neverSent.outcome());
        assertFalse(neverSent.maySafelyApply());
    }

    @Test
    @DisplayName("a server predating fieldDecisions gets no manufactured verdict")
    void anOlderServerIsUnreadableRatherThanApproved() throws Exception {
        // `fieldDecisions` was appended to this response; a server older than it sends the key at
        // all. The client will not reconstruct a governance verdict the server did not send, so
        // every column on such a response is UNREADABLE -- loudly unusable rather than quietly
        // approved off rank 1's own decision.
        String body = """
                {"results":{"sailing.route_code":[
                  {"rank":1,"governanceId":"GBF-0028","businessName":"Sailing Route Code",
                   "definition":"The short code that identifies a scheduled route.",
                   "domain":"Published","governance":null,"confidence":0.83,
                   "decision":"AUTO_APPROVE"}]},
                 "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":["OPEN_DECK"]}}
                """;

        FieldGovernance governance = decoded(body).governanceFor("sailing.route_code");

        assertEquals(GovernanceOutcome.UNREADABLE, governance.outcome());
        assertFalse(
                governance.maySafelyApply(),
                "rank 1 says AUTO_APPROVE, and reading that as a column verdict is exactly the "
                        + "roll-up the server publishes fieldDecisions to own");
    }

    // =========================================================================
    // THE ENUM ITSELF
    // =========================================================================

    @Test
    @DisplayName("exactly two outcomes permit applying a classification, and they are named")
    void onlyTwoOutcomesAreAPermission() {
        Set<GovernanceOutcome> permitting = EnumSet.noneOf(GovernanceOutcome.class);
        for (GovernanceOutcome outcome : GovernanceOutcome.values()) {
            if (outcome.maySafelyApply()) {
                permitting.add(outcome);
            }
        }

        assertEquals(
                EnumSet.of(GovernanceOutcome.CONFERRED, GovernanceOutcome.OPEN_TIER),
                permitting,
                "an outcome added to this enum defaults to 'no' by construction, and adding one "
                        + "that answers yes is a decision that has to be made here rather than "
                        + "inherited. If this failed because a constant was ADDED, the question to "
                        + "answer is whether a caller may write a classification from it.");
    }

    @Test
    @DisplayName("every outcome is either a permission, a review, or an operator's problem")
    void noOutcomeFallsBetweenTheThreeFollowUps() {
        for (GovernanceOutcome outcome : GovernanceOutcome.values()) {
            boolean apply = outcome.maySafelyApply();
            boolean human = outcome.needsAHuman();
            boolean operator = outcome == GovernanceOutcome.UNCLASSIFIABLE_NO_VOCABULARY;

            assertEquals(
                    1,
                    (apply ? 1 : 0) + (human ? 1 : 0) + (operator ? 1 : 0),
                    outcome + " maps to " + (apply ? "apply " : "") + (human ? "human " : "")
                            + (operator ? "operator" : "")
                            + ". Every outcome must map to exactly one follow-up: zero leaves a "
                            + "caller with a value and no action, and two leaves them choosing.");
        }
    }
}
