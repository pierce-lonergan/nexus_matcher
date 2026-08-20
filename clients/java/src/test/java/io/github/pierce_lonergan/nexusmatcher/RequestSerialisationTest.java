package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.Feedback;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.LookupRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.RetrievalDiagnosticRequest;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What goes ON the wire.
 *
 * <p>The server sets {@code extra="forbid"} on a field spec, so a single stray key is a 422 for the
 * whole request. That makes "which keys does this client emit" a correctness property rather than
 * a formatting preference, and these tests read the emitted JSON rather than trusting the
 * annotations.
 */
class RequestSerialisationTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("a field spec emits exactly the four contract keys, and omits the absent ones")
    void fieldSpecEmitsOnlyContractKeys() throws Exception {
        JsonNode full = mapper.valueToTree(
                FieldSpec.of("legal_name", "booking.passenger.legal_name", "A comment.", "string"));

        List<String> keys = new ArrayList<>();
        full.fieldNames().forEachRemaining(keys::add);
        assertEquals(List.of("name", "path", "doc", "type"), keys);

        JsonNode minimal = mapper.valueToTree(FieldSpec.of("legal_name"));
        List<String> minimalKeys = new ArrayList<>();
        minimal.fieldNames().forEachRemaining(minimalKeys::add);
        assertEquals(
                List.of("name"),
                minimalKeys,
                "an absent optional must be omitted, not sent as null: the server types these as "
                        + "strings with defaults and would refuse an explicit null");
    }

    @Test
    @DisplayName("the request knobs are snake_case, because the request half of the contract is")
    void requestKnobsAreSnakeCase() throws Exception {
        JsonNode node = mapper.valueToTree(
                MatchRequest.of(List.of(FieldSpec.of("a", "t.a"))).withTopK(3).withExplain(true));

        assertTrue(node.has("fields"));
        assertEquals(3, node.get("top_k").asInt());
        assertTrue(node.get("explain").asBoolean());
        assertFalse(node.has("topK"), "the wire name is top_k; camelCase is the RESPONSE half");
    }

    @Test
    @DisplayName("unset knobs are omitted, so the server's own defaults apply")
    void unsetKnobsAreOmitted() throws Exception {
        JsonNode node = mapper.valueToTree(MatchRequest.of(List.of(FieldSpec.of("a"))));

        assertFalse(node.has("top_k"));
        assertFalse(node.has("explain"));
        assertFalse(node.has("contrast"));
        assertFalse(node.has("consistency"));
        assertFalse(
                node.has("consistency_qualifier_segments"),
                "a client that sent its own copy of every default would pin this artifact to the "
                        + "defaults of the server it was written against. Whether a deployment "
                        + "sends the evidence blocks is the deployment's decision");
    }

    @Test
    @DisplayName("the evidence knobs are snake_case too, including the long one")
    void evidenceKnobsSpellTheWireNames() throws Exception {
        JsonNode node = mapper.valueToTree(
                MatchRequest.of(List.of(FieldSpec.of("a", "t.a")))
                        .withContrast(true)
                        .withConsistency(true)
                        .withConsistencyQualifierSegments(0));

        assertTrue(node.get("contrast").asBoolean());
        assertTrue(node.get("consistency").asBoolean());
        assertEquals(0, node.get("consistency_qualifier_segments").asInt());
        assertFalse(
                node.has("consistencyQualifierSegments"),
                "the request half of the contract is snake_case; camelCase is the RESPONSE half");
    }

    @Test
    @DisplayName("asking for a grouping policy does not silently ask for the report")
    void theGroupingDialDoesNotTurnTheReportOn() throws Exception {
        JsonNode node = mapper.valueToTree(
                MatchRequest.of(List.of(FieldSpec.of("a"))).withConsistencyQualifierSegments(0));

        assertEquals(0, node.get("consistency_qualifier_segments").asInt());
        assertFalse(
                node.has("consistency"),
                "the knob and the block are separate on the wire, and a request that never asks "
                        + "for the report should not start receiving one because a policy was set");
    }

    @Test
    @DisplayName("every wither keeps every other knob, so a re-chunked 413 asks the same question")
    void withersPreserveEveryOtherKnob() {
        // The failure this pins is silent and specific: a `with*` that rebuilds the record and
        // passes null for a knob it was not changing. The caller then gets a request that is
        // subtly not the one they built -- and the case that matters most is withFields, which is
        // what the retry loop calls to re-chunk a 413. Half a batch answered under different
        // knobs than the other half is not a batch anybody can read.
        MatchRequest full = MatchRequest.of(List.of(FieldSpec.of("a", "t.a")))
                .withTopK(3)
                .withExplain(true)
                .withSignals(Map.of("domain", "Passenger"))
                .withContrast(true)
                .withConsistency(true)
                .withConsistencyQualifierSegments(2);

        List<MatchRequest> rebuilt = List.of(
                full.withTopK(4),
                full.withExplain(false),
                full.withContrast(false),
                full.withConsistency(false),
                full.withConsistencyQualifierSegments(1),
                full.withFields(List.of(FieldSpec.of("b", "t.b"))));

        for (MatchRequest request : rebuilt) {
            assertNotNull(request.topK(), "top_k was dropped by a wither that does not own it");
            assertNotNull(request.explain(), "explain was dropped");
            assertNotNull(request.signals(), "signals was dropped");
            assertNotNull(request.contrast(), "contrast was dropped");
            assertNotNull(request.consistency(), "consistency was dropped");
            assertNotNull(
                    request.consistencyQualifierSegments(),
                    "consistency_qualifier_segments was dropped");
        }

        MatchRequest rechunked = full.withFields(List.of(FieldSpec.of("b", "t.b")));
        assertEquals(3, rechunked.topK());
        assertEquals(Map.of("domain", "Passenger"), rechunked.signals());
        assertEquals(2, rechunked.consistencyQualifierSegments());
    }

    @Test
    @DisplayName("a feedback record uses the wire name `field`, not `fieldPath`")
    void feedbackUsesTheAliasedWireName() throws Exception {
        JsonNode node = mapper.valueToTree(new Feedback(
                "booking.passenger.legal_name",
                "A comment.",
                "GBF-0001",
                null,
                true,
                "a.reviewer",
                "2026-08-11T09:00:00Z",
                null));

        assertEquals("booking.passenger.legal_name", node.get("field").asText());
        assertFalse(node.has("fieldPath"));
        assertFalse(
                node.has("suggestedGovernanceId"),
                "an absent suggestion is omitted; the server records it as null itself");
    }

    @Test
    @DisplayName("the response key is the path when there is one, and the name when there is not")
    void responseKeyMirrorsTheServersOwnFallback() {
        assertEquals("t.a", FieldSpec.of("a", "t.a").responseKey());
        assertEquals("a", FieldSpec.of("a").responseKey());
    }

    @Test
    @DisplayName("chunking preserves order, which is what the response is keyed by")
    void chunkingPreservesOrder() {
        List<FieldSpec> fields = new ArrayList<>();
        for (int i = 0; i < 250; i++) {
            fields.add(FieldSpec.of("c" + i, "t.c" + i));
        }

        List<List<FieldSpec>> chunks = FieldSpec.chunk(fields, 100);

        assertEquals(3, chunks.size());
        assertEquals(100, chunks.get(0).size());
        assertEquals(50, chunks.get(2).size());
        List<FieldSpec> flattened = chunks.stream().flatMap(List::stream).toList();
        assertEquals(fields, flattened);
    }

    @Test
    @DisplayName("obviously unusable input is refused here rather than at the server")
    void structurallyImpossibleInputIsRefusedLocally() {
        assertThrows(IllegalArgumentException.class, () -> FieldSpec.of("  "));
        assertThrows(NullPointerException.class, () -> FieldSpec.of(null));
        assertThrows(IllegalArgumentException.class, () -> MatchRequest.of(List.of()));
        assertThrows(IllegalArgumentException.class, () -> FieldSpec.chunk(List.of(), 0));
    }

    @Test
    @DisplayName("the diagnostic request is snake_case too, and omits the knobs it was not given")
    void diagnosticRequestSpellsTheWireNames() throws Exception {
        // This envelope is extra="forbid" on the server -- unlike MatchRequest -- so a wrong key
        // here is a 422 rather than a silently defaulted knob, and an explicit null on `top_k`
        // is refused because the server types it as an int with a default.
        JsonNode full = mapper.valueToTree(
                RetrievalDiagnosticRequest.of(FieldSpec.of("a", "t.a"), "GBF-0001").withTopK(3));

        List<String> keys = new ArrayList<>();
        full.fieldNames().forEachRemaining(keys::add);
        assertEquals(List.of("field", "expected_governance_id", "top_k"), keys);
        assertFalse(full.has("expectedGovernanceId"), "the wire name is snake_case here");

        JsonNode minimal = mapper.valueToTree(RetrievalDiagnosticRequest.of(FieldSpec.of("a")));
        List<String> minimalKeys = new ArrayList<>();
        minimal.fieldNames().forEachRemaining(minimalKeys::add);
        assertEquals(List.of("field"), minimalKeys);
    }

    @Test
    @DisplayName("a lookup request emits its ids and refuses the two the server cannot answer")
    void lookupRequestEmitsIdsAndRefusesTheUnanswerable() throws Exception {
        JsonNode node = mapper.valueToTree(LookupRequest.of(List.of("GBF-0001", "GBF-0028")));

        List<String> keys = new ArrayList<>();
        node.fieldNames().forEachRemaining(keys::add);
        assertEquals(List.of("ids"), keys);
        assertEquals("GBF-0001", node.get("ids").get(0).asText());

        // Refused locally because both are STRUCTURAL, not configured: the response is a map
        // keyed by the id, so no deployment can answer a blank one or two of the same one. The
        // length cap is a server number and is deliberately not mirrored -- see LookupIT.
        assertThrows(IllegalArgumentException.class, () -> LookupRequest.of(List.of()));
        assertThrows(IllegalArgumentException.class, () -> LookupRequest.of(List.of(" ")));
        assertThrows(
                IllegalArgumentException.class,
                () -> LookupRequest.of(List.of("GBF-0001", "GBF-0001")));
    }

    @Test
    @DisplayName("the server's configurable caps are NOT mirrored here")
    void configurableCapsAreLeftToTheServer() {
        List<FieldSpec> overTheDefaultCap = new ArrayList<>();
        for (int i = 0; i < 5000; i++) {
            overTheDefaultCap.add(FieldSpec.of("c" + i, "t.c" + i));
        }

        // Builds without complaint, on purpose. `max_fields`, `max_batch_fields`, the string
        // length bounds and `results_per_field` are all per-deployment settings; a copy of them
        // compiled into this artifact would refuse requests a tuned server accepts, and would go
        // stale silently the first time an operator raised one. The server's 413 names its own
        // number, and PayloadTooLargeException.suggestedChunkSize() hands it back.
        MatchRequest request = MatchRequest.of(overTheDefaultCap);
        assertEquals(5000, request.fields().size());
    }
}
