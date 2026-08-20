package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherRequestException;
import io.github.pierce_lonergan.nexusmatcher.error.ServiceUnavailableException;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.LookupEntry;
import io.github.pierce_lonergan.nexusmatcher.model.LookupResponse;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The lookup plane, against a running service loaded with the example pack. */
class LookupIT {

    private static NexusMatcherClient client;

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
    }

    @Test
    @DisplayName("a known id resolves exactly, with no score to misread")
    void aKnownIdResolvesExactly() {
        LookupEntry entry = client.lookup("GBF-0001").entryFor("GBF-0001").orElseThrow();

        assertEquals("GBF-0001", entry.governanceId());
        assertEquals("Passenger Legal Name", entry.businessName());
        assertEquals("MANIFEST_NAME", entry.governance().code());
        assertEquals("SEALED_RESTRICTED", entry.governance().classification());
        assertEquals("MASK_IN_LOGS", entry.governance().enhancement());
    }

    @Test
    @DisplayName("an id the dictionary does not carry is a 200 with a null, never a 404")
    void anUnknownIdIsAnAnswerRatherThanAFailure() {
        LookupResponse response =
                client.lookup(List.of("GBF-0001", "GBF-NOT-A-REAL-ID", "GBF-0028"));

        assertEquals(
                List.of("GBF-0001", "GBF-NOT-A-REAL-ID", "GBF-0028"),
                List.copyOf(response.ids()),
                "every id comes back exactly once in the order sent, resolved or not");
        assertTrue(response.entryFor("GBF-NOT-A-REAL-ID").isEmpty());
        assertEquals(List.of("GBF-NOT-A-REAL-ID"), response.missing());
        assertFalse(response.allResolved());
        assertTrue(
                response.entryFor("GBF-0028").orElseThrow().governanceValue().isEmpty(),
                "and the pack's uncoded row still resolves -- 'no class' is not 'no entry'");
    }

    @Test
    @DisplayName("a looked-up entry agrees with the matched candidate for the same id")
    void lookupAndMatchAgreeAboutTheSameEntry() {
        // The claim the lookup plane makes is that it returns the SAME enrichment surface a
        // candidate carries, rendered by the same server-side code. If the two ever disagreed, a
        // pipeline that resolved known ids and matched unknown ones would classify one column two
        // ways depending on which door it came through.
        MatchCandidate matched = client.match(MatchRequest.of(
                        List.of(FieldSpec.of(
                                "legal_name",
                                "booking.passenger.legal_name",
                                "Full legal name of the passenger as printed on the sailing "
                                        + "manifest.",
                                "string")),
                        1))
                .topCandidateFor("booking.passenger.legal_name")
                .orElseThrow();

        LookupEntry resolved =
                client.lookup(matched.governanceId()).entryFor(matched.governanceId()).orElseThrow();

        assertEquals(matched.governanceId(), resolved.governanceId());
        assertEquals(matched.businessName(), resolved.businessName());
        assertEquals(matched.definition(), resolved.definition());
        assertEquals(matched.domain(), resolved.domain());
        assertEquals(
                matched.governance(),
                resolved.governance(),
                "the whole protection class, member for member -- these are rendered by one "
                        + "function on the server precisely so this comparison cannot fail");
        assertEquals(
                matched.sourceMetadata(),
                resolved.sourceMetadata(),
                "and the deployment's own pass-through columns too. The service renders both "
                        + "planes with one function over one entry, so this is a fact by "
                        + "construction; asserting it against a live server is what turns that "
                        + "claim into something a refactor can break loudly.");
    }

    @Test
    @DisplayName("a resolved entry carries the deployment's own columns, uninterpreted")
    void aResolvedEntryCarriesThePassThroughPlane() {
        LookupEntry entry = client.lookup("GBF-0001").entryFor("GBF-0001").orElseThrow();

        assertFalse(
                entry.sourceMetadata().isEmpty(),
                "the example pack's glossary has spare columns and they should reach a caller; "
                        + "an empty plane here means the pass-through stopped at the index again");
        assertTrue(
                entry.sourceMetadata().isComplete(),
                "and nothing in this pack is large enough for the loader to trim");
    }

    @Test
    @DisplayName("a duplicate id never leaves this process, because the map cannot hold two")
    void aDuplicateIdIsRefusedLocally() {
        assertThrows(
                IllegalArgumentException.class,
                () -> client.lookup(List.of("GBF-0001", "GBF-0001")));
    }

    @Test
    @DisplayName("the id length cap is the SERVER's, and reaches the caller as a typed 422")
    void anOversizedIdIsARequestFailureFromTheServer() {
        // LookupRequest refuses a blank or a duplicated id locally, because both are
        // structural -- the response is keyed by the id and no deployment can answer either. It
        // deliberately does NOT mirror the LENGTH cap, which is a number on the server, and this
        // test is that decision being exercised rather than asserted: an oversized id leaves this
        // process, the server refuses it, and the refusal arrives as a class the caller can catch.
        // Mirroring the cap here would have refused it locally against a number that could go
        // stale the moment the server tuned its own.
        String oversized = "GBF-" + "x".repeat(2000);

        NexusMatcherRequestException failure =
                assertThrows(NexusMatcherRequestException.class, () -> client.lookup(oversized));

        assertEquals(422, failure.httpStatus());
        assertTrue(
                failure.details().containsKey("limit_chars"),
                "and the failure names the cap it applied, so an adapter can chunk or truncate "
                        + "against the server's own number: " + failure.details());
    }

    @Test
    @DisplayName("a server with no dictionary cannot resolve anything, and says so as a 503")
    void aDictionaryLessServerAnswers503() {
        NexusMatcherClient unavailable =
                NexusMatcherClient.builder(LiveService.unavailable()).retryPolicy(RetryPolicy.none())
                        .build();

        ServiceUnavailableException failure = assertThrows(
                ServiceUnavailableException.class, () -> unavailable.lookup("GBF-0001"));

        assertEquals(503, failure.httpStatus());
    }
}
