package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.LookupResponse;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The id round trip, against a real service.
 *
 * <p>{@code GovernanceIdOpacityTest} pins what this client's decoder does with a numeric-looking
 * id. This pins the other half: that such an id survives being SERIALISED by this client, carried
 * over HTTP, and echoed back -- which is where a {@code long} component, a normalising interceptor
 * or a JSON library configured to coerce would actually show up.
 *
 * <p>The example pack's ids are {@code GBF-nnnn}, so none of the numeric ids below resolves. That
 * is the point rather than a limitation: a lookup that misses still echoes every id it was given,
 * verbatim and in the order sent, so the round trip is observable without needing somebody else's
 * glossary. What a numeric id RESOLVES to is a property of a deployment this suite does not have,
 * and it is pinned against a real matcher on the Python side
 * ({@code tests/unit/presentation/api/test_governance_id_opacity.py}).
 */
class GovernanceIdOpacityIT {

    private static NexusMatcherClient client;

    /** 2^53 + 1 and 2^53: two distinct ids that are one {@code double}. */
    private static final String BIG = "9007199254740993";
    private static final String BIG_DOUBLE_TWIN = "9007199254740992";

    /**
     * Every id spelling worth carrying across the wire, in an order that is neither ascending nor
     * descending under any reading of them.
     */
    private static final List<String> PROBE_IDS =
            List.of("10000000", "0000123", BIG, "123", "1", BIG_DOUBLE_TWIN, "007");

    @BeforeAll
    static void connect() {
        client = NexusMatcherClient.builder(LiveService.matching()).build();
    }

    @Test
    @DisplayName("every numeric-looking id is echoed verbatim, in the order it was sent")
    void idsSurviveTheRoundTripCharacterForCharacter() {
        LookupResponse response = client.lookup(PROBE_IDS);

        assertEquals(
                PROBE_IDS,
                List.copyOf(response.ids()),
                "the response is keyed by the caller's own id strings, in the order sent. A "
                        + "reordering here means something on one side of this exchange is "
                        + "comparing ids rather than merely carrying them.");
        assertEquals(
                PROBE_IDS,
                response.missing(),
                "none of these is in the example pack, so all of them miss -- and `missing` names "
                        + "them in the order sent, in their exact spelling");
        for (String id : PROBE_IDS) {
            assertTrue(
                    response.isMissing(id),
                    id + " did not come back as sent. The zero-padded and unpadded spellings are "
                            + "different keys and must stay different all the way through.");
        }
    }

    @Test
    @DisplayName("a padded id and its unpadded form are two ids, end to end")
    void paddingIsNotStrippedAnywhereAlongThePath() {
        LookupResponse response = client.lookup(List.of("0000123", "123", "007", "7"));

        assertEquals(
                List.of("0000123", "123", "007", "7"),
                List.copyOf(response.ids()),
                "four ids were sent and four must come back. Two keys collapsing into one is what "
                        + "a numeric normalisation looks like from here, and it would take a "
                        + "column's classification with it.");
        assertEquals(4, response.results().size());
    }

    @Test
    @DisplayName("two ids one double cannot tell apart stay two ids across the wire")
    void theDoubleCollidingPairIsNotCollapsed() {
        assertEquals(
                Double.parseDouble(BIG),
                Double.parseDouble(BIG_DOUBLE_TWIN),
                "the pair this test is built on is no longer double-equal, so it has stopped "
                        + "demonstrating the failure it names");
        assertNotEquals(BIG, BIG_DOUBLE_TWIN);

        LookupResponse response = client.lookup(List.of(BIG, BIG_DOUBLE_TWIN));

        assertEquals(List.of(BIG, BIG_DOUBLE_TWIN), List.copyOf(response.ids()));
        assertEquals(List.of(BIG, BIG_DOUBLE_TWIN), response.missing());
    }

    @Test
    @DisplayName("a real id from a match resolves through a lookup unchanged")
    void theIdAMatchReturnsIsTheIdALookupTakes() {
        // The control. The three tests above are built on ids that MISS, so a service that had
        // stopped resolving anything at all would satisfy them. This is the one that says the
        // round trip works when the id is real -- and it is the pipeline shape that matters: an id
        // read off a candidate, sent straight back, resolving to the same row.
        MatchCandidate top = client
                .match(MatchRequest.of(
                        List.of(FieldSpec.of(
                                "legal_name",
                                "booking.passenger.legal_name",
                                "Full legal name of the passenger as printed on the manifest.",
                                "string")),
                        1))
                .topCandidateFor("booking.passenger.legal_name")
                .orElseThrow();

        LookupResponse resolved = client.lookup(top.governanceId());

        assertEquals(
                top.governanceId(),
                resolved.entryFor(top.governanceId()).orElseThrow().governanceId(),
                "the id is echoed from the ENTRY rather than from the request, so a dictionary "
                        + "whose id column disagreed with its key would be visible here");
        assertEquals(
                top.businessName(),
                resolved.entryFor(top.governanceId()).orElseThrow().businessName());
        assertTrue(resolved.allResolved());
    }
}
