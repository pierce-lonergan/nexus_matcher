package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.ExpectedPlacement;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.RetrievalCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.RetrievalChannel;
import io.github.pierce_lonergan.nexusmatcher.model.RetrievalDiagnostic;
import io.github.pierce_lonergan.nexusmatcher.model.RetrievalDiagnosticRequest;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@code POST /api/v1/diag/retrieval}, against a running service loaded with the example pack. */
class RetrievalDiagnosticIT {

    private static NexusMatcherClient client;

    private static FieldSpec undescribed() {
        return FieldSpec.of(
                "lifejacket_locker_inspection_due",
                "vessel.safety.lifejacket_locker_inspection_due",
                "Date the lifejacket locker is next due for inspection.",
                "date");
    }

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
    }

    @Test
    @DisplayName("the diagnostic shows the query the encoder actually saw, not the column name")
    void theQueryTextIsTheThingThatWasEncoded() {
        RetrievalDiagnostic diagnostic =
                client.diagnoseRetrieval(RetrievalDiagnosticRequest.of(undescribed()));

        assertFalse(diagnostic.queryText().isBlank());
        assertTrue(
                diagnostic.queryText().contains("lifejacket"),
                "the query is built from the field's own words: " + diagnostic.queryText());
        assertTrue(
                diagnostic.queryText().contains("safety"),
                "and from its PARENT PATH, which is the largest single accuracy factor measured "
                        + "on this task -- and the reason a flat column name matches worse. Got: "
                        + diagnostic.queryText());
        assertEquals(
                "vessel.safety.lifejacket_locker_inspection_due", diagnostic.field().get("path"));
    }

    @Test
    @DisplayName("every candidate a real match returns appears in the fused channel")
    void everyMatchedCandidateWasRetrievedFirst() {
        // The invariant that keeps this route honest. Matching can only score what retrieval
        // returned, so every candidate on a match response must appear in the fused list this
        // route reports for the same field. It holds for any scoring change and fails the moment
        // this route stops driving the same pipeline -- which is the way a diagnostic goes wrong:
        // not by breaking, but by quietly describing something else.
        FieldSpec field = undescribed();
        MatchResponse matched = client.match(MatchRequest.of(List.of(field), 5));
        RetrievalDiagnostic diagnostic =
                client.diagnoseRetrieval(RetrievalDiagnosticRequest.of(field).withTopK(100));

        RetrievalChannel fused = diagnostic.fused().orElseThrow();
        List<String> fusedIds =
                fused.candidates().stream().map(RetrievalCandidate::governanceId).toList();

        for (MatchCandidate candidate : matched.candidatesFor(field.responseKey())) {
            assertTrue(
                    fusedIds.contains(candidate.governanceId()),
                    candidate.governanceId() + " came back from /match but is absent from the "
                            + "fused retrieval list this route reports. Matching can only score "
                            + "what retrieval returned, so the diagnostic is now describing a "
                            + "different pipeline. Fused list: " + fusedIds);
        }
    }

    @Test
    @DisplayName("naming an expected entry that exists reports where it landed")
    void anExpectedEntryInTheDictionaryReportsItsRank() {
        RetrievalDiagnostic diagnostic = client.diagnoseRetrieval(
                RetrievalDiagnosticRequest.of(undescribed(), "GBF-0001").withTopK(5));

        ExpectedPlacement expected = diagnostic.expectedValue().orElseThrow();
        assertEquals("GBF-0001", expected.governanceId());
        assertTrue(
                expected.inDictionary(),
                "GBF-0001 is the pack's passenger-name entry and is definitely indexed");
        assertTrue(
                diagnostic.diagnosis().orElseThrow().contains("GBF-0001"),
                "the one-line diagnosis names the entry that was asked about");
    }

    @Test
    @DisplayName("an entry that is not in the dictionary is a different diagnosis, and says so")
    void anEntryThatWasNeverIndexedIsAGlossaryProblem() {
        RetrievalDiagnostic diagnostic = client.diagnoseRetrieval(
                RetrievalDiagnosticRequest.of(undescribed(), "GBF-NOT-A-REAL-ID"));

        ExpectedPlacement expected = diagnostic.expectedValue().orElseThrow();
        assertFalse(expected.inDictionary());
        assertFalse(expected.retrievedByAnyChannel());
        assertTrue(
                diagnostic.diagnosis().orElseThrow().contains("not in the loaded dictionary"),
                "'retrieved at rank 34' is a scoring problem and 'never indexed' is a glossary "
                        + "problem; confusing them wastes a day. Got: "
                        + diagnostic.diagnosis().orElseThrow());
    }

    @Test
    @DisplayName("asking for no expected entry asks a smaller question, and gets no answer to it")
    void withoutAnExpectedIdThereIsNoPlacement() {
        RetrievalDiagnostic diagnostic =
                client.diagnoseRetrieval(RetrievalDiagnosticRequest.of(undescribed()));

        assertTrue(diagnostic.expectedValue().isEmpty());
        assertTrue(
                diagnostic.diagnosis().isEmpty(),
                "no expected entry means no question to answer, and inventing one would be this "
                        + "client having an opinion about which entry was right");
    }

    @Test
    @DisplayName("ranks are over the full channel result, so top_k changes the view not the answer")
    void rankIsComputedOverTheFullResultRatherThanTheTruncation() {
        // If ranks were computed over the displayed slice, asking for fewer rows would change the
        // rank reported for the expected entry -- and an entry at rank 26 would come back as
        // "absent" from a top_k of 3, which is the wrong diagnosis entirely.
        RetrievalDiagnosticRequest base =
                RetrievalDiagnosticRequest.of(undescribed(), "GBF-0022");

        ExpectedPlacement narrow =
                client.diagnoseRetrieval(base.withTopK(1)).expectedValue().orElseThrow();
        ExpectedPlacement wide =
                client.diagnoseRetrieval(base.withTopK(20)).expectedValue().orElseThrow();

        assertEquals(
                narrow.rankByChannel(),
                wide.rankByChannel(),
                "the expected entry's rank must not depend on how many rows were displayed");
        assertEquals(
                1,
                client.diagnoseRetrieval(base.withTopK(1)).dense().orElseThrow().candidates().size(),
                "while the display truncation does what it says");
    }
}
