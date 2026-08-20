package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.ConceptGroup;
import io.github.pierce_lonergan.nexusmatcher.model.ConsistencyReport;
import io.github.pierce_lonergan.nexusmatcher.model.Contrast;
import io.github.pierce_lonergan.nexusmatcher.model.ContrastReport;
import io.github.pierce_lonergan.nexusmatcher.model.Explain;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.Governance;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchDecision;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.SignalDifference;
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
 * Matching, against a running service loaded with the repository's example pack.
 *
 * <p>All field names, comments and expected ids below come from
 * {@code examples/governance/} -- an invented ferry operator, shipped precisely so that nobody
 * mistakes it for a taxonomy worth adopting.
 */
class MatchIT {

    private static NexusMatcherClient client;

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
    }

    private static FieldSpec legalName() {
        return FieldSpec.of(
                "legal_name",
                "booking.passenger.legal_name",
                "Full legal name of the passenger as printed on the sailing manifest.",
                "string");
    }

    @Test
    @DisplayName("a match returns the real protection class the glossary entry confers")
    void matchReturnsRealGovernance() {
        MatchResponse response = client.match(List.of(legalName()));

        MatchCandidate top =
                response.topCandidateFor("booking.passenger.legal_name").orElseThrow();
        assertEquals("GBF-0001", top.governanceId());
        assertEquals("Passenger Legal Name", top.businessName());
        assertEquals(MatchDecision.AUTO_APPROVE, top.decision());

        Governance governance = top.governance();
        assertNotNull(governance, "this entry carries a code, so a class must arrive");
        assertEquals("MANIFEST_NAME", governance.code());
        assertEquals("SEALED_RESTRICTED", governance.classification());
        assertTrue(governance.personalInformation());
        assertTrue(governance.directIdentifier());
        assertEquals("MASK_IN_LOGS", governance.enhancement());

        assertEquals("OPEN_DECK", response.vocabulary().openClassification());
        assertEquals(
                List.of("OPEN_DECK", "CREW_ONLY", "BRIDGE_SENSITIVE", "SEALED_RESTRICTED"),
                response.vocabulary().tiersMostOpenFirst());
    }

    @Test
    @DisplayName("every field sent comes back, under its own path, in the order sent")
    void conservationAndOrderHold() {
        List<FieldSpec> fields = List.of(
                FieldSpec.of("gate_scan_ts", "terminal.gate.scan_timestamp",
                        "The moment a boarding pass was scanned at the gate.", "timestamp"),
                legalName(),
                FieldSpec.of("fuel_pct", "vessel.telemetry.fuel_level_pct",
                        "Percentage of usable fuel remaining in the vessel tanks.", "float"),
                FieldSpec.of("galley_reorder_qty", "galley.stock.reorder_quantity",
                        "How many units of a galley line to reorder.", "integer"));

        MatchResponse response = client.match(MatchRequest.of(fields, 3));

        assertEquals(
                fields.stream().map(FieldSpec::responseKey).toList(),
                List.copyOf(response.paths()),
                "the map is keyed by the caller's own path in the order sent; a field that "
                        + "vanished from it would inherit nothing and nothing would say so");
        for (String path : response.paths()) {
            assertTrue(
                    response.candidatesFor(path).size() <= 3,
                    path + " returned more candidates than top_k asked for");
        }
    }

    @Test
    @DisplayName("a field given no path comes back under its name, as the server documents")
    void omittedPathFallsBackToName() {
        MatchResponse response = client.match(List.of(FieldSpec.of("terminal_name")));

        assertEquals(List.of("terminal_name"), List.copyOf(response.paths()));
    }

    @Test
    @DisplayName("explain arrives and its arithmetic closes against the confidence sent with it")
    void explainReproducesTheConfidence() {
        MatchResponse response = client.match(
                MatchRequest.of(List.of(legalName()), 1).withExplain(true));

        MatchCandidate top =
                response.topCandidateFor("booking.passenger.legal_name").orElseThrow();
        Explain explain = top.explainValue().orElseThrow();

        assertEquals(explain.scores().keySet(), explain.weights().keySet());
        assertEquals(
                top.confidence(),
                explain.recomputedConfidence(),
                1e-5,
                "the server verifies this before sending; if it fails here the client has "
                        + "mangled the numbers on the way in");
    }

    @Test
    @DisplayName("the batch route takes more fields than /match does, on one contract")
    void batchTakesTheHigherCap() {
        List<FieldSpec> fields = new ArrayList<>();
        for (int i = 0; i < 150; i++) {
            fields.add(FieldSpec.of("col_" + i, "wide.table.col_" + i,
                    "Column " + i + " of a wide extract.", "string"));
        }

        MatchResponse response = client.matchBatch(MatchRequest.of(fields, 1));

        assertEquals(150, response.results().size());
        assertEquals("wide.table.col_0", List.copyOf(response.paths()).get(0));
        assertEquals("wide.table.col_149", List.copyOf(response.paths()).get(149));
        assertNotNull(response.vocabulary(), "the batch route carries the vocabulary too");
    }

    @Test
    @DisplayName("a caller-supplied correlation id wins and comes back on the response")
    void callerSuppliedRequestIdIsPropagated() {
        MatchResponse response =
                client.match(MatchRequest.of(List.of(legalName())), "java-client-it-0001");

        assertEquals(
                "java-client-it-0001",
                response.requestId(),
                "this is the id in the server's log for this request; without it a Java stack "
                        + "trace and a server log line cannot be joined");
        assertTrue(response.responseTime().orElse(-1) >= 0);
    }

    @Test
    @DisplayName("two identical requests produce identical results")
    void matchingIsDeterministic() {
        MatchRequest request = MatchRequest.of(List.of(legalName()), 5);

        MatchResponse first = client.match(request);
        MatchResponse second = client.match(request);

        assertEquals(first.results(), second.results());
        assertEquals(first.vocabulary(), second.vocabulary());
    }

    // =============================================================================
    // THE REVIEW-EVIDENCE BLOCKS, against a live server
    // =============================================================================
    //
    // The decoding tests read captured bodies. These three read a running service, and the
    // property they are here for is the one a capture cannot show: that asking for the blocks
    // changes NOTHING ELSE, and that the server's defaults are what this client's javadoc says
    // they are. A default that moves in the service and not in the javadoc is exactly the drift
    // tests/packaging/test_java_client_contract.py keeps a gate on for the SHAPE of the wire,
    // and the shape is not the only thing a client documents.

    /** Two columns that share the leaf `name` and are not the same concept. */
    private static List<FieldSpec> twoColumnsNamedName() {
        return List.of(
                FieldSpec.of("name", "published.terminal.name",
                        "The public name of a Gravel Bay ferry terminal.", "string"),
                FieldSpec.of("name", "booking.passenger.name",
                        "Full legal name of the passenger as printed on the sailing manifest.",
                        "string"));
    }

    @Test
    @DisplayName("asking for the evidence blocks changes nothing else in the response")
    void theEvidenceBlocksAreAdditive() {
        MatchRequest plain = MatchRequest.of(twoColumnsNamedName(), 2);

        MatchResponse without = client.match(plain);
        MatchResponse with = client.match(plain.withContrast(true).withConsistency(true));

        assertNull(without.contrast(), "not asked for, so not sent");
        assertNull(without.consistency());
        assertNotNull(with.contrast(), "asked for, so sent");
        assertNotNull(with.consistency());

        assertEquals(
                without.results(),
                with.results(),
                "the evidence blocks are reporting-only; a candidate that moved because a "
                        + "reviewer asked why it won would make the answer depend on the question");
        assertEquals(without.fieldDecisions(), with.fieldDecisions());
        assertEquals(without.vocabulary(), with.vocabulary());
        assertFalse(
                with.consistency().promotionApplied(),
                "and the server says so machine-readably rather than leaving it to be inferred");
    }

    @Test
    @DisplayName("the contrast closes against the confidences sent beside it")
    void theContrastArithmeticClosesLive() {
        MatchResponse response = client.match(
                MatchRequest.of(twoColumnsNamedName(), 2).withContrast(true));

        ContrastReport report = response.contrastValue().orElseThrow();
        assertEquals(
                List.copyOf(response.paths()),
                List.copyOf(report.paths()),
                "every input path is a key of the contrast, in the order sent");

        for (String path : report.paths()) {
            Contrast contrast = report.contrastFor(path).orElseThrow();
            double summed = contrast.signals().stream()
                    .mapToDouble(SignalDifference::weightedDelta)
                    .sum();
            // One order of magnitude above the published resolution: that is the server's own
            // tolerance, and it is there because both operands of every delta are rounded before
            // being subtracted. See ReviewEvidenceDecodingTest for the worked case.
            assertEquals(contrast.signalGap(), summed, report.resolution() * 10.0, path);
            assertEquals(
                    contrast.topConfidence(),
                    response.topCandidateFor(path).orElseThrow().confidence(),
                    0.0,
                    path + ": the contrast is about the candidates in THIS response, and its "
                            + "top confidence is the one rank 1 carries");
        }
    }

    @Test
    @DisplayName("the shipped grouping default reports nothing; the loose key manufactures a find")
    void theGroupingDialBehavesAsDocumented() {
        MatchRequest asking = MatchRequest.of(twoColumnsNamedName(), 2).withConsistency(true);

        ConsistencyReport shipped = client.match(asking).consistencyValue().orElseThrow();
        assertEquals(
                1,
                shipped.qualifierSegments().orElseThrow(),
                "ConsistencyReport's javadoc tells a reader the default is 1 and that it reports "
                        + "nothing. If the server's default moves, that javadoc is wrong and this "
                        + "is where a reader finds out");
        assertEquals(
                0,
                shipped.groupsFound(),
                "these two columns hang off different records, so at the shipped default they "
                        + "are not one concept and nothing is reported");

        ConsistencyReport loose = client.match(asking.withConsistencyQualifierSegments(0))
                .consistencyValue()
                .orElseThrow();
        assertEquals(0, loose.qualifierSegments().orElseThrow());
        assertEquals(1, loose.groupsFound());
        assertEquals(1, loose.groupsDisagreeing());

        ConceptGroup collision = loose.disagreeingGroups().get(0);
        assertEquals(
                collision.answeredCount(),
                collision.distinctAnswers(),
                "a ferry terminal and a passenger are not one concept. Every column that "
                        + "answered gave a different answer, which is the collision signature, "
                        + "not a matcher that contradicted itself");
    }
}
