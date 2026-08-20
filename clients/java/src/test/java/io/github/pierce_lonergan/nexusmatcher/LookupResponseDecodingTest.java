package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.GovernanceStatus;
import io.github.pierce_lonergan.nexusmatcher.model.LookupEntry;
import io.github.pierce_lonergan.nexusmatcher.model.LookupRequest;
import io.github.pierce_lonergan.nexusmatcher.model.LookupResponse;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The lookup plane, decoded from a captured body, with a plain mapper. */
class LookupResponseDecodingTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private LookupResponse captured() throws Exception {
        return mapper.readValue(Fixtures.captured("lookup-response.json"), LookupResponse.class);
    }

    @Test
    @DisplayName("a missing id is a null in the map, not a dropped key")
    void aMissingIdIsANullRatherThanAnAbsence() throws Exception {
        LookupResponse response = captured();

        assertEquals(
                List.of("GBF-0001", "GBF-0028", "GBF-NOT-A-REAL-ID"),
                List.copyOf(response.ids()),
                "every requested id comes back exactly once in the order sent; a partial list "
                        + "would make the caller's own key vanish with nothing saying so");
        assertTrue(response.results().containsKey("GBF-NOT-A-REAL-ID"));
        assertTrue(response.entryFor("GBF-NOT-A-REAL-ID").isEmpty());
        assertEquals(List.of("GBF-NOT-A-REAL-ID"), response.missing());
        assertFalse(response.allResolved());
        assertTrue(response.isMissing("GBF-NOT-A-REAL-ID"));
        assertFalse(
                response.isMissing("GBF-9999"),
                "an id that was never sent is not 'missing'; the two are different questions and "
                        + "only one of them means the glossary has changed");
    }

    @Test
    @DisplayName("a resolved entry carries the same enrichment a match candidate does")
    void resolvedEntryCarriesTheCandidateSurface() throws Exception {
        LookupEntry entry = captured().entryFor("GBF-0001").orElseThrow();

        assertEquals("GBF-0001", entry.governanceId());
        assertEquals("Passenger Legal Name", entry.businessName());
        assertEquals("Passenger", entry.domain());
        assertEquals(GovernanceStatus.CONFERRED, entry.governanceStatus());
        assertEquals("MANIFEST_NAME", entry.governance().code());
        assertEquals("SEALED_RESTRICTED", entry.governance().classification());
        assertEquals(
                "MASK_IN_LOGS",
                entry.governance().enhancement(),
                "the handling instruction has to survive the lookup plane too, or an operator "
                        + "resolving an id they already hold gets less than one who guessed at it");
    }

    @Test
    @DisplayName("an uncoded entry is the open tier here too, and only that")
    void uncodedEntryIsOpenTier() throws Exception {
        LookupResponse response = captured();
        LookupEntry entry = response.entryFor("GBF-0028").orElseThrow();

        assertTrue(entry.governanceValue().isEmpty());
        assertEquals(
                GovernanceStatus.OPEN_TIER,
                entry.governanceStatus(),
                "a lookup makes no match, so there is no rejected rank 1 and nothing can be "
                        + "WITHHELD -- a null class here has exactly one meaning");
        assertEquals("OPEN_DECK", response.vocabulary().openClassification());
    }

    @Test
    @DisplayName("the request refuses locally what the server refuses remotely")
    void requestRefusesBlankAndDuplicateIds() {
        assertThrows(IllegalArgumentException.class, () -> LookupRequest.of(List.of()));
        assertThrows(IllegalArgumentException.class, () -> LookupRequest.of(List.of("  ")));
        assertThrows(
                IllegalArgumentException.class,
                () -> LookupRequest.of(List.of("GBF-0001", "GBF-0001")),
                "the response is a map keyed by these strings; a duplicate cannot be answered "
                        + "twice and the server refuses it. Both checks are structural rather "
                        + "than configured, which is why mirroring them cannot go stale.");

        assertEquals(List.of("GBF-0001"), LookupRequest.of("GBF-0001").ids());
    }

    @Test
    @DisplayName("the decoded map is unmodifiable and keeps its nulls")
    void decodedMapIsUnmodifiableAndKeepsNulls() throws Exception {
        LookupResponse response = captured();

        // Map.copyOf would have thrown on the null value during decoding, which would turn the
        // answer this endpoint exists to give -- "no such entry" -- into a crash.
        assertTrue(response.results().containsKey("GBF-NOT-A-REAL-ID"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> response.results().put("injected", null));
    }
}
