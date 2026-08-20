package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.model.LookupEntry;
import io.github.pierce_lonergan.nexusmatcher.model.LookupResponse;
import io.github.pierce_lonergan.nexusmatcher.model.MatchCandidate;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.RecordComponent;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code governanceId} is an OPAQUE STRING on this side of the wire too.
 *
 * <p>The service's answer to "what is a governanceId" is that it is the dictionary entry's own id,
 * carried verbatim. A deployment whose ids happen to be numerals -- and there is one that describes
 * them as "just a number from 1 to 10000000" -- is exactly the deployment where a Java consumer is
 * tempted to bind them as a {@code long}, normalise them, or compare them numerically. Each of
 * those loses something, and loses it silently:
 *
 * <ul>
 *   <li>a zero-padded id parsed to a number comes back a DIFFERENT id, so the row it resolves is
 *       the wrong row or no row;
 *   <li>an id above 2^53 does not survive a {@code double}, and JSON in a language with no integer
 *       type gives you a {@code double} whether you asked for one or not. {@code 9007199254740993}
 *       and {@code 9007199254740992} are two ids and one {@code double}, and inheriting the wrong
 *       one of those two rows means inheriting the wrong protection class;
 *   <li>ordering by id reorders a response whose order is the caller's own request order.
 * </ul>
 *
 * <p>So this pins the component's TYPE, the decoder's fidelity, and the absence of any numeric
 * reading -- on the captured bodies where it can, and on hand-built ones where the shipped example
 * pack cannot supply the id in question. {@code GovernanceIdOpacityIT} does the round trip against
 * a real service.
 */
class GovernanceIdOpacityTest {

    private final ObjectMapper mapper = new ObjectMapper();

    /** 2^53 + 1 and 2^53: two distinct ids that are one {@code double}. */
    private static final String BIG = "9007199254740993";
    private static final String BIG_DOUBLE_TWIN = "9007199254740992";

    // =========================================================================
    // THE TYPE
    // =========================================================================

    @Test
    @DisplayName("both DTOs bind governanceId as String, on the record itself")
    void theComponentIsAString() {
        assertSame(String.class, componentType(MatchCandidate.class, "governanceId"));
        assertSame(String.class, componentType(LookupEntry.class, "governanceId"));
    }

    private static Class<?> componentType(Class<?> record, String name) {
        for (RecordComponent component : record.getRecordComponents()) {
            if (component.getName().equals(name)) {
                return component.getType();
            }
        }
        throw new AssertionError(
                record.getSimpleName() + " has no component named " + name
                        + ". If it was renamed, every caller reading the id is now reading "
                        + "something else, and this test is the only thing that says so.");
    }

    // =========================================================================
    // FIDELITY, ON A REAL CAPTURE
    // =========================================================================

    @Test
    @DisplayName("a captured id decodes to the exact characters the service sent")
    void aCapturedIdSurvivesDecodingUnchanged() throws Exception {
        MatchResponse matched =
                mapper.readValue(Fixtures.captured("match-response.json"), MatchResponse.class);
        LookupResponse looked =
                mapper.readValue(Fixtures.captured("lookup-response.json"), LookupResponse.class);

        String fromMatch =
                matched.topCandidateFor("booking.passenger.legal_name").orElseThrow().governanceId();
        assertEquals("GBF-0001", fromMatch);
        assertEquals(
                fromMatch,
                looked.entryFor("GBF-0001").orElseThrow().governanceId(),
                "the id a match returns is the id a lookup takes; that is what lets a pipeline "
                        + "resolve some columns by id and match the rest without the two halves "
                        + "disagreeing about one glossary row");
    }

    // =========================================================================
    // THE NUMERIC IDS THE SHIPPED PACK CANNOT SUPPLY
    // =========================================================================

    @Test
    @DisplayName("numeric-looking ids decode to their exact characters, padding intact")
    void numericLookingIdsAreNotNumbers() throws Exception {
        // NOT a capture. The example pack's ids are GBF-nnnn, so no fixture server can serve a
        // numeric one, and the deployment whose ids ARE numerals is somebody else's. The BODY is
        // the captured shape with the ids substituted; what is under test is this client's
        // decoder, which is the half that can be tested without their glossary.
        MatchResponse response = mapper.readValue(
                """
                {"results":{"ledger.account_ref":[
                  {"rank":1,"governanceId":"0000123","businessName":"Account Reference",
                   "definition":"Reference for a customer account.","domain":"Ledger",
                   "governance":null,"confidence":0.83,"decision":"AUTO_APPROVE",
                   "absoluteScore":0.77},
                  {"rank":2,"governanceId":"1","businessName":"Account Number",
                   "definition":"Number identifying an account.","domain":"Ledger",
                   "governance":null,"confidence":0.51,"decision":"REVIEW",
                   "absoluteScore":0.44},
                  {"rank":3,"governanceId":"%s","businessName":"Ledger Line Key",
                   "definition":"Key of one ledger line.","domain":"Ledger",
                   "governance":null,"confidence":0.44,"decision":"REJECT",
                   "absoluteScore":0.31},
                  {"rank":4,"governanceId":"10000000","businessName":"Ledger Batch Key",
                   "definition":"Key of one posting batch.","domain":"Ledger",
                   "governance":null,"confidence":0.41,"decision":"REJECT",
                   "absoluteScore":0.29}]},
                 "fieldDecisions":{"ledger.account_ref":"AUTO_APPROVE"},
                 "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":["OPEN_DECK"]}}
                """.formatted(BIG),
                MatchResponse.class);

        List<String> ids = response.candidatesFor("ledger.account_ref").stream()
                .map(MatchCandidate::governanceId)
                .toList();

        assertEquals(List.of("0000123", "1", BIG, "10000000"), ids);
        assertEquals(
                "0000123",
                ids.get(0),
                "the padding is part of the id. A component bound as a long, or an id normalised "
                        + "on the way in, would make this '123' -- a key the glossary does not have");
        assertNotEquals("123", ids.get(0));
    }

    @Test
    @DisplayName("an id above 2^53 keeps every digit a double would lose")
    void aLargeIdSurvivesWhereADoubleWouldNot() throws Exception {
        LookupResponse response = mapper.readValue(
                """
                {"results":{"%s":{"governanceId":"%s","businessName":"Ledger Line Key",
                   "definition":"Key of one ledger line.","domain":"Ledger","governance":null,
                   "sourceMetadata":{"values":{},"droppedKeyCount":0,"renderedKeys":[]}},
                  "%s":{"governanceId":"%s","businessName":"Ledger Line Predecessor",
                   "definition":"Key of the preceding ledger line.","domain":"Ledger",
                   "governance":null,
                   "sourceMetadata":{"values":{},"droppedKeyCount":0,"renderedKeys":[]}}},
                 "missing":[],
                 "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":["OPEN_DECK"]}}
                """.formatted(BIG, BIG, BIG_DOUBLE_TWIN, BIG_DOUBLE_TWIN),
                LookupResponse.class);

        // The hazard, stated in Java so this test says out loud why these two values were chosen.
        assertEquals(
                Double.parseDouble(BIG),
                Double.parseDouble(BIG_DOUBLE_TWIN),
                "2^53+1 and 2^53 are no longer the same double, so this test has stopped "
                        + "demonstrating the failure it names");
        assertNotEquals(BIG, BIG_DOUBLE_TWIN);

        assertEquals(BIG, response.entryFor(BIG).orElseThrow().governanceId());
        assertEquals(
                BIG_DOUBLE_TWIN, response.entryFor(BIG_DOUBLE_TWIN).orElseThrow().governanceId());
        assertNotEquals(
                response.entryFor(BIG).orElseThrow().businessName(),
                response.entryFor(BIG_DOUBLE_TWIN).orElseThrow().businessName(),
                "these are two glossary rows. A consumer that routed their ids through a double "
                        + "would hold one, and would classify one column from the other's entry.");
        assertEquals(List.of(BIG, BIG_DOUBLE_TWIN), List.copyOf(response.ids()));
    }

    // =========================================================================
    // NOTHING ORDERS BY THE ID
    // =========================================================================

    @Test
    @DisplayName("candidates keep the server's rank order, which is not id order")
    void theClientDoesNotReorderByTheId() throws Exception {
        MatchResponse response = mapper.readValue(
                """
                {"results":{"ledger.account_ref":[
                  {"rank":1,"governanceId":"10000000","businessName":"Ledger Batch Key",
                   "definition":"Key of one posting batch.","domain":"Ledger","governance":null,
                   "confidence":0.83,"decision":"AUTO_APPROVE","absoluteScore":0.77},
                  {"rank":2,"governanceId":"0000123","businessName":"Account Reference",
                   "definition":"Reference for a customer account.","domain":"Ledger",
                   "governance":null,"confidence":0.51,"decision":"REVIEW","absoluteScore":0.44},
                  {"rank":3,"governanceId":"1","businessName":"Account Number",
                   "definition":"Number identifying an account.","domain":"Ledger",
                   "governance":null,"confidence":0.44,"decision":"REJECT","absoluteScore":0.31}]},
                 "fieldDecisions":{"ledger.account_ref":"AUTO_APPROVE"},
                 "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":["OPEN_DECK"]}}
                """,
                MatchResponse.class);

        List<String> ids = response.candidatesFor("ledger.account_ref").stream()
                .map(MatchCandidate::governanceId)
                .toList();

        assertEquals(List.of("10000000", "0000123", "1"), ids);
        assertNotEquals(ids.stream().sorted().toList(), ids, "the probe order is now sorted");
        assertEquals(
                "10000000",
                response.topCandidateFor("ledger.account_ref").orElseThrow().governanceId(),
                "rank 1 is the server's rank 1. Sorting or tie-breaking on the id would hand a "
                        + "different entry's protection class to whoever reads the top candidate.");
    }

    @Test
    @DisplayName("a lookup's results and missing list keep the order the ids were sent in")
    void lookupOrderIsTheCallersOwn() throws Exception {
        LookupResponse response = mapper.readValue(
                """
                {"results":{"10000000":null,"0000123":null,"1":null},
                 "missing":["10000000","0000123","1"],
                 "vocabulary":{"openClassification":"OPEN_DECK","tiersMostOpenFirst":["OPEN_DECK"]}}
                """,
                LookupResponse.class);

        assertEquals(List.of("10000000", "0000123", "1"), List.copyOf(response.ids()));
        assertEquals(List.of("10000000", "0000123", "1"), response.missing());
        assertTrue(
                response.entryFor("123").isEmpty(),
                "'123' is not '0000123'; a client that normalised ids would resolve one from the "
                        + "other and there is no error anywhere when it does");
    }
}
