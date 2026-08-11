package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherException;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherRequestException;
import io.github.pierce_lonergan.nexusmatcher.error.PayloadTooLargeException;
import io.github.pierce_lonergan.nexusmatcher.model.FieldSpec;
import io.github.pierce_lonergan.nexusmatcher.model.MatchRequest;
import io.github.pierce_lonergan.nexusmatcher.model.MatchResponse;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Every documented refusal a caller can provoke, against a running service. */
class RequestFailureIT {

    private static String base;
    private static NexusMatcherClient client;

    @BeforeAll
    static void connect() {
        base = LiveService.matching();
        client = NexusMatcherClient.builder(base).build();
    }

    /**
     * A client whose field spec serialises with the EXAMPLE PACK's key names instead of the wire
     * contract's.
     *
     * <p>Not a mock of anything -- it is a real request to the real server, sent in the exact shape
     * a reviewer gets by pasting a row out of {@code examples/governance/fields.json}. That is the
     * mistake the server's own documentation records two reviewers making, so it is the 422 worth
     * proving the client reports usefully.
     */
    private static NexusMatcherClient clientSendingPackSpellings() {
        SimpleModule module = new SimpleModule();
        module.addSerializer(new StdSerializer<>(FieldSpec.class) {
            @Override
            public void serialize(
                    FieldSpec value, JsonGenerator generator, SerializerProvider provider)
                    throws IOException {
                generator.writeStartObject();
                generator.writeStringField("flattenedName", value.responseKey());
                generator.writeStringField("doc", value.doc() == null ? "" : value.doc());
                generator.writeStringField("dataType", value.type() == null ? "" : value.type());
                generator.writeEndObject();
            }
        });
        return NexusMatcherClient.builder(base)
                .objectMapper(new ObjectMapper().registerModule(module))
                .build();
    }

    @Test
    @DisplayName("an unknown field key is a 422 that names every offending key")
    void unknownFieldKeyIs422() {
        NexusMatcherRequestException failure = assertThrows(
                NexusMatcherRequestException.class,
                () -> clientSendingPackSpellings().match(
                        List.of(FieldSpec.of("legal_name", "booking_passenger__legal_name",
                                "Full legal name of the passenger.", "string"))));

        assertEquals(422, failure.httpStatus());
        assertEquals("NEXUS-8004", failure.errorCode().orElseThrow());
        assertFalse(failure.isRetryable());
        assertTrue(failure.requestId().isPresent(), "a 422 must still be joinable to a log line");

        List<String> offending = failure.violations().stream()
                .filter(violation -> "extra_forbidden".equals(violation.type()))
                .map(violation -> violation.location().get(violation.location().size() - 1))
                .toList();
        assertEquals(List.of("flattenedName", "dataType"), offending);
    }

    @Test
    @DisplayName("top_k above the server's cap is a 422 carrying the cap to ask for")
    void topKAboveCapIs422() {
        NexusMatcherRequestException failure = assertThrows(
                NexusMatcherRequestException.class,
                () -> client.match(
                        MatchRequest.of(List.of(FieldSpec.of("terminal_name", "t.name")), 50)));

        int cap = failure.resultsPerFieldCap().orElseThrow();
        assertTrue(cap >= 1);

        // And the number the server gave is usable as-is, which is the point of carrying it.
        MatchResponse retried = client.match(
                MatchRequest.of(List.of(FieldSpec.of("terminal_name", "t.name")), cap));
        assertTrue(retried.candidatesFor("t.name").size() <= cap);
    }

    @Test
    @DisplayName("two fields under one path is a 422, not a silently shorter map")
    void duplicatePathsAre422() {
        NexusMatcherRequestException failure = assertThrows(
                NexusMatcherRequestException.class,
                () -> client.match(List.of(
                        FieldSpec.of("a", "t.same"),
                        FieldSpec.of("b", "t.same"))));

        assertEquals(List.of("t.same"), failure.duplicatePaths());
    }

    @Test
    @DisplayName("over the field cap is a 413, and the re-chunked request then succeeds")
    void fieldCapIs413AndRechunkingWorks() {
        List<FieldSpec> tooMany = new ArrayList<>();
        for (int i = 0; i < 101; i++) {
            tooMany.add(FieldSpec.of("c" + i, "t.c" + i, "Column " + i + ".", "string"));
        }

        PayloadTooLargeException failure =
                assertThrows(PayloadTooLargeException.class, () -> client.match(tooMany));

        assertEquals(413, failure.httpStatus());
        assertEquals(101, failure.observedFields().orElseThrow());
        assertFalse(failure.isByteCap());
        assertFalse(failure.isRetryable());

        int chunkSize = failure.suggestedChunkSize().orElseThrow();
        List<List<FieldSpec>> chunks = FieldSpec.chunk(tooMany, chunkSize);
        int seen = 0;
        for (List<FieldSpec> chunk : chunks) {
            seen += client.match(MatchRequest.of(chunk, 1)).results().size();
        }
        assertEquals(
                tooMany.size(),
                seen,
                "the whole point of carrying the limit is that the next attempt works");
    }

    @Test
    @DisplayName("a body over the byte cap comes back as a readable 413, not a socket reset")
    void byteCapIsRefusedReadably() {
        // This body is 1.06x the cap, which is INSIDE the server's drain budget, so the 413 is
        // reliable here: measured 120/120 on a pooled connection after the server-side fix,
        // against 34/40 before it.
        //
        // The history matters, because the failure mode comes back for a big enough body.
        // Refusing a body the client is still writing closes the socket with bytes unread, and
        // that is an RST, and an RST discards the 413 already sent. It presents as
        //     IOException: fixed content-length: 403, bytes received: 0
        // -- status line and headers in, body gone -- so the caller loses the two numbers the
        // failure exists to carry. The server now reads and discards a refused body up to 2x the
        // cap before answering, which is why a realistic mis-chunked batch like this one is safe.
        //
        // A body BEYOND twice the cap is still refused unread and still loses its 413 about 15%
        // of the time. Keep this test inside the budget; a bigger body would be testing the
        // residual, not the contract.
        //
        // `HttpRequest.Builder.expectContinue(true)` also removes the race and must NOT be
        // adopted: against this server on JDK 17 the client then HANGS, because the 413 arrives
        // instead of the `100 Continue` the JDK is waiting for and `send` never returns --
        // measured at 654 s against a 30 s request timeout that never fired. See the note at the
        // send site in NexusMatcherClient#attempt.
        List<FieldSpec> huge = new ArrayList<>();
        String longDoc = "x".repeat(8000);
        for (int i = 0; i < 1300; i++) {
            huge.add(FieldSpec.of("c" + i, "t.c" + i, longDoc, "string"));
        }

        PayloadTooLargeException failure = assertThrows(
                PayloadTooLargeException.class,
                () -> client.matchBatch(MatchRequest.of(huge, 1)),
                "an oversized body inside the server's drain budget must come back as a readable "
                        + "413. A transport failure here is a REGRESSION -- it was measured "
                        + "120/120 readable. Check the server's drain bounds before blaming the "
                        + "network.");

        assertTrue(failure.isByteCap());
        assertEquals("content-length", failure.source().orElseThrow());
        assertTrue(failure.limitBytes().orElseThrow() > 0);
        assertTrue(
                failure.observedBytes().orElseThrow() > failure.limitBytes().orElseThrow());
        assertTrue(
                failure.suggestedChunkSize().isEmpty(),
                "nothing counted the fields, so the server has no chunk size to give and this "
                        + "client will not invent one");
        assertFalse(
                failure.isRetryable(), "an oversized body is the same size on the second attempt");
        assertTrue(failure.requestId().isPresent(),
                "the 413 is answered by middleware OUTSIDE the request-id middleware, and it "
                        + "stamps the header itself precisely so this stays joinable");
    }

    @Test
    @DisplayName("a base URL pointing at the wrong place fails typed, not as a parse error")
    void wrongBaseUrlIsATypedFailure() {
        NexusMatcherClient misconfigured =
                NexusMatcherClient.builder(base + "/not-the-service").build();

        NexusMatcherException failure = assertThrows(
                NexusMatcherException.class,
                () -> misconfigured.match(List.of(FieldSpec.of("terminal_name", "t.name"))));

        assertEquals(404, failure.httpStatus());
        assertEquals(
                "NEXUS-8004",
                failure.errorCode().orElseThrow(),
                "404 is answered in the same one error envelope as everything else, so a client "
                        + "never has to branch on the SHAPE of an error before reading it");
        assertFalse(failure.isRetryable());
    }
}
