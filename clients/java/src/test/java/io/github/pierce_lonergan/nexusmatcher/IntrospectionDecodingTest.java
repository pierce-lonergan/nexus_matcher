package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.EncoderStatus;
import io.github.pierce_lonergan.nexusmatcher.model.ExpectedPlacement;
import io.github.pierce_lonergan.nexusmatcher.model.RetrievalChannel;
import io.github.pierce_lonergan.nexusmatcher.model.RetrievalDiagnostic;
import io.github.pierce_lonergan.nexusmatcher.model.ServiceStatus;
import io.github.pierce_lonergan.nexusmatcher.model.Thresholds;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@code /api/v1/status} and {@code /api/v1/diag/retrieval}, decoded from captured bodies. */
class IntrospectionDecodingTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("status reports what is loaded and whether it is degraded")
    void statusReportsTheLoadedState() throws Exception {
        ServiceStatus status =
                mapper.readValue(Fixtures.captured("status.json"), ServiceStatus.class);

        assertTrue(status.ready());
        assertFalse(status.degraded());
        assertTrue(status.fitForBulkRun());
        assertEquals(List.of(), status.warningCodes());
        assertFalse(status.hasWarning(ServiceStatus.FALLBACK_ENCODER));

        assertEquals(30, status.dictionary().entries().orElseThrow());
        assertTrue(status.dictionary().sourceValue().orElseThrow().endsWith("glossary.csv"));

        EncoderStatus encoder = status.encoderValue().orElseThrow();
        assertEquals("bundled", encoder.tier());
        assertTrue(encoder.bundledEncoderAvailable());
        assertFalse(
                encoder.fallbackInForce(),
                "this is the field the surface exists for: true means retrieval silently fell "
                        + "through to a lower rung and every result is quietly worse");
        assertEquals(384, encoder.embeddingDimension().orElseThrow());

        assertEquals(250, status.limits().maxBatchFields());
        assertEquals(100, status.limits().fieldCap(false));
        assertEquals(250, status.limits().fieldCap(true));
        assertTrue(
                status.limits().deadlineSeconds() > 0,
                "a client timeout has to be set ABOVE this or the 504 is never seen");
    }

    @Test
    @DisplayName("the thresholds block publishes the floor problem as arithmetic")
    void thresholdsPublishTheFloorProblem() throws Exception {
        ServiceStatus status =
                mapper.readValue(Fixtures.captured("status.json"), ServiceStatus.class);
        Thresholds thresholds = status.thresholdsValue().orElseThrow();

        assertEquals(0.87, thresholds.autoApproveValue().orElseThrow(), 1e-9);
        assertEquals(0.5, thresholds.reviewValue().orElseThrow(), 1e-9);
        assertEquals(0.63, thresholds.confidenceFloorValue().orElseThrow(), 1e-9);
        assertEquals(5, thresholds.resultsPerFieldValue().orElseThrow());

        assertEquals(
                Boolean.TRUE,
                thresholds.reviewThresholdBelowFloor(),
                "0.50 sits below the 0.63 structural floor on a stock deployment, so no rank-1 "
                        + "candidate can fall below review on score alone. That is not a defect "
                        + "on this server -- it is the fact that made a separate NO_MATCH verdict "
                        + "necessary, published as arithmetic instead of as a warning.");
        assertFalse(
                status.degraded(),
                "and it is deliberately NOT a warning: a status surface that reports every stock "
                        + "install as degraded teaches operators to ignore the field");
    }

    @Test
    @DisplayName("the diagnostic reports what the encoder actually saw")
    void diagnosticReportsTheQueryText() throws Exception {
        RetrievalDiagnostic diagnostic = mapper.readValue(
                Fixtures.captured("retrieval-diagnostic.json"), RetrievalDiagnostic.class);

        assertEquals("telemetry.quasar_flux_index", diagnostic.field().get("path"));
        assertTrue(
                diagnostic.queryText().startsWith("telemetry quasar flux index"),
                "the field's parent path is injected into the query before encoding, so THIS is "
                        + "the string the encoder saw -- not the column name. Got: "
                        + diagnostic.queryText());
        assertFalse(diagnostic.rerankerWired());

        RetrievalChannel dense = diagnostic.dense().orElseThrow();
        assertTrue(dense.available());
        assertTrue(dense.unavailableBecause().isEmpty());
        assertEquals(30, dense.returnedCount().orElseThrow());
        assertEquals(
                3,
                dense.candidates().size(),
                "returned() counts the FULL result and candidates() is the display truncation; "
                        + "the two disagreeing is the contract, not a defect");
        assertEquals(1, dense.candidates().get(0).rank());
        assertTrue(dense.shows("GBF-0022"));

        assertTrue(
                diagnostic.sparse().orElseThrow().candidates().get(0).score()
                        > diagnostic.dense().orElseThrow().candidates().get(0).score(),
                "the sparse channel's raw score is on its own scale -- these two numbers are not "
                        + "comparable, and this assertion exists to make that concrete rather "
                        + "than to claim sparse retrieval is better");
    }

    @Test
    @DisplayName("naming an expected entry turns 'it missed' into a diagnosis")
    void expectedPlacementSeparatesTheTwoDiagnoses() throws Exception {
        RetrievalDiagnostic diagnostic = mapper.readValue(
                Fixtures.captured("retrieval-diagnostic.json"), RetrievalDiagnostic.class);
        ExpectedPlacement expected = diagnostic.expectedValue().orElseThrow();

        assertEquals("GBF-0022", expected.governanceId());
        assertTrue(
                expected.inDictionary(),
                "in the dictionary AND retrieved is a scoring problem; not in the dictionary is a "
                        + "glossary problem, and no amount of threshold tuning fixes the second");
        assertTrue(expected.retrievedByAnyChannel());
        assertEquals(1, expected.rankIn(RetrievalDiagnostic.DENSE).orElseThrow());
        assertTrue(diagnostic.diagnosis().orElseThrow().contains("GBF-0022"));
    }

    @Test
    @DisplayName("a channel that did not return the expected entry reports null, not rank 0")
    void anAbsentRankIsEmptyRatherThanZero() throws Exception {
        // Hand-built: the captured field's expected entry is rank 1 in every channel, so the null
        // arm of rankByChannel has no live example here. The distinction matters because rank 0
        // does not exist in a 1-based ranking -- a client that read a missing rank as 0 would
        // report the entry as having been retrieved FIRST.
        String body = """
                {"field":{"name":"a"},"queryText":"a","encoderModel":null,"rerankerWired":false,\
                "channels":{},"expected":{"governanceId":"GBF-0001","inDictionary":true,\
                "rankByChannel":{"dense":null,"sparse":12}}}
                """;
        ExpectedPlacement expected = mapper.readValue(body, RetrievalDiagnostic.class)
                .expectedValue()
                .orElseThrow();

        assertTrue(expected.rankIn("dense").isEmpty());
        assertEquals(12, expected.rankIn("sparse").orElseThrow());
        assertTrue(expected.retrievedByAnyChannel());
    }

    @Test
    @DisplayName("an entry in the dictionary that no channel retrieved says exactly that")
    void nothingRetrievedIsItsOwnDiagnosis() throws Exception {
        String body = """
                {"field":{"name":"a"},"queryText":"a","encoderModel":null,"rerankerWired":false,\
                "channels":{},"expected":{"governanceId":"GBF-0001","inDictionary":true,\
                "rankByChannel":{"dense":null,"sparse":null,"fused":null}}}
                """;
        RetrievalDiagnostic diagnostic = mapper.readValue(body, RetrievalDiagnostic.class);

        assertFalse(diagnostic.expectedValue().orElseThrow().retrievedByAnyChannel());
        assertTrue(
                diagnostic.diagnosis().orElseThrow().contains("no channel retrieved it"),
                "scoring never saw it, so tuning the scoring weights would change nothing. Got: "
                        + diagnostic.diagnosis().orElseThrow());
    }
}
