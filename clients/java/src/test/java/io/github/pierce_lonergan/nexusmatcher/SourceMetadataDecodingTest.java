package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.LookupEntry;
import io.github.pierce_lonergan.nexusmatcher.model.LookupResponse;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import io.github.pierce_lonergan.nexusmatcher.model.SourceMetadata;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The pass-through plane: the deployment's own enrichment columns, decoded.
 *
 * <p>The client's whole job here is to carry it and to type NOTHING about it, so most of what is
 * asserted below is that a value survives rather than that it is understood. The one place this
 * client is allowed an opinion is the shape of the wrapper -- what {@code droppedKeyCount} and
 * {@code renderedKeys} mean for a caller who is about to read a value out of the map.
 */
class SourceMetadataDecodingTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("a candidate carries its entry's enrichment columns, unread and in order")
    void aCandidateCarriesTheEntrysColumns() throws Exception {
        MatchResponse response =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);
        MatchCandidate top = response.topCandidateFor("booking.passenger.legal_name").orElseThrow();
        SourceMetadata metadata = top.sourceMetadata();

        assertFalse(metadata.isEmpty());
        assertTrue(metadata.isComplete(), "nothing was dropped from this entry");
        assertEquals(
                List.copyOf(metadata.keys()),
                List.copyOf(metadata.values().keySet()),
                "the key order is the loader's own and is part of what pass-through means");
        assertEquals("yes", metadata.text("personal_information").orElseThrow());
        assertTrue(
                metadata.value("no_such_column").isEmpty(),
                "asking for a column this deployment does not carry is safe and empty");
    }

    @Test
    @DisplayName("a looked-up entry carries the identical object a matched candidate does")
    void bothPlanesAgreeAboutOneGlossaryRow() throws Exception {
        MatchResponse matched =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);
        LookupResponse resolved =
                mapper.readValue(Fixtures.captured("lookup-response.json"), LookupResponse.class);

        MatchCandidate candidate =
                matched.topCandidateFor("booking.passenger.legal_name").orElseThrow();
        LookupEntry entry = resolved.entryFor("GBF-0001").orElseThrow();

        assertEquals("GBF-0001", candidate.governanceId());
        assertEquals(
                candidate.sourceMetadata(),
                entry.sourceMetadata(),
                "the server renders both planes with one function over one entry, so a pipeline "
                        + "that resolves some columns by id and matches the rest must not get two "
                        + "different answers about the same glossary row");
    }

    @Test
    @DisplayName("an entry with no spare columns is an empty map, not a missing member")
    void anEntryWithNoColumnsIsEmptyRatherThanAbsent() throws Exception {
        String body = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":0.9,\
                "decision":"AUTO_APPROVE","absoluteScore":0.7,\
                "sourceMetadata":{"values":{},"droppedKeyCount":0,"renderedKeys":[]}}]},\
                "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":[]}}
                """;
        MatchCandidate candidate =
                mapper.readValue(body, MatchResponse.class).topCandidateFor("t.a").orElseThrow();

        assertTrue(candidate.sourceMetadata().isEmpty());
        assertTrue(candidate.sourceMetadata().isComplete());
    }

    @Test
    @DisplayName("a server that predates the plane gets an empty one, never a null to trip over")
    void anOlderServerYieldsAnEmptyPlane() throws Exception {
        String body = """
                {"results":{"t.a":[{"rank":1,"governanceId":"GBF-0027","businessName":"Terminal \
                Name","definition":"d","domain":"Published","governance":null,"confidence":0.9,\
                "decision":"AUTO_APPROVE"}]},"vocabulary":{"openClassification":"OPEN_DECK",\
                "tiersMostOpenFirst":[]}}
                """;
        MatchCandidate candidate =
                mapper.readValue(body, MatchResponse.class).topCandidateFor("t.a").orElseThrow();

        assertNotNull(
                candidate.sourceMetadata(),
                "the member is defaulted rather than left null: a caller should not have to "
                        + "null-check something every current server sends, and 'no columns' is "
                        + "the same answer whether the server is old or the entry is bare");
        assertTrue(candidate.sourceMetadata().isEmpty());
    }

    @Test
    @DisplayName("a trimmed plane says so, and a caller can tell before reading a value out of it")
    void aTrimmedPlaneAnnouncesItself() throws Exception {
        // droppedKeyCount is this bound reaching a consumer for the first time. Before it, an HTTP
        // caller could not tell a whole plane from a trimmed one -- so an absent key read as "the
        // deployment does not populate that column" when it actually meant "the loader ran out of
        // room". Those are very different conclusions to draw about a glossary.
        String body = """
                {"values":{"steward":"harbourmaster"},"droppedKeyCount":4,"renderedKeys":[]}
                """;
        SourceMetadata metadata = mapper.readValue(body, SourceMetadata.class);

        assertFalse(
                metadata.isComplete(),
                "four keys of this row did not fit; the map is a bounded subsequence and this "
                        + "response is not the place to read that row from");
        assertEquals(4, metadata.droppedKeyCount());
        assertTrue(metadata.value("review_date").isEmpty());
    }

    @Test
    @DisplayName("a value the source could not express as JSON is named, not silently coerced")
    void renderedValuesAreNamed() throws Exception {
        String body = """
                {"values":{"review_date":"2026-08-19 00:00:00","owner":"ops"},\
                "droppedKeyCount":0,"renderedKeys":["review_date"]}
                """;
        SourceMetadata metadata = mapper.readValue(body, SourceMetadata.class);

        assertTrue(
                metadata.wasRendered("review_date"),
                "this string is the TEXT FORM of a date cell, not a date the source held as "
                        + "text; parsing it back is a guess about the source system's format");
        assertFalse(metadata.wasRendered("owner"), "and this one really is what the cell held");
    }

    @Test
    @DisplayName("open values survive whatever the source held, including a null")
    void valuesAreOpenAndSurviveTheirTypes() throws Exception {
        // Both sides of this map are the deployment's own vocabulary, so the binding is
        // Map<String, Object> and stays one. A JSON or Parquet glossary can hold any of these in
        // one cell, and a client that narrowed the value type would refuse a body the service
        // legitimately sends.
        String body = """
                {"values":{"text":"a","number":7,"decimalish":1.5,"flag":true,"blank":null,\
                "list":[1,2],"nested":{"k":"v"}},"droppedKeyCount":0,"renderedKeys":[]}
                """;
        SourceMetadata metadata = mapper.readValue(body, SourceMetadata.class);

        assertEquals("a", metadata.value("text").orElseThrow());
        assertEquals(7, metadata.value("number").orElseThrow());
        assertEquals(Boolean.TRUE, metadata.value("flag").orElseThrow());
        assertEquals(List.of(1, 2), metadata.value("list").orElseThrow());
        assertEquals(Map.of("k", "v"), metadata.value("nested").orElseThrow());

        assertTrue(
                metadata.values().containsKey("blank"),
                "a blank cell is a PRESENT key with a null value, and the decoder must keep it: "
                        + "Map.copyOf would have thrown here, turning one empty spare column into "
                        + "a failure to read the whole response");
        assertTrue(metadata.value("blank").isEmpty());
    }

    @Test
    @DisplayName("the decoded plane is unmodifiable, like the rest of the artifact")
    void theDecodedPlaneIsUnmodifiable() throws Exception {
        LookupResponse resolved =
                mapper.readValue(Fixtures.captured("lookup-response.json"), LookupResponse.class);
        SourceMetadata metadata = resolved.entryFor("GBF-0001").orElseThrow().sourceMetadata();

        assertThrows(
                UnsupportedOperationException.class,
                () -> metadata.values().put("injected", "x"));
        assertThrows(
                UnsupportedOperationException.class, () -> metadata.renderedKeys().add("injected"));
    }
}
