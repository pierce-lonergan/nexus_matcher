package io.github.pierce_lonergan.nexusmatcher;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.pierce_lonergan.nexusmatcher.error.DeadlineExceededException;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherException;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherRequestException;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherServerException;
import io.github.pierce_lonergan.nexusmatcher.error.PayloadTooLargeException;
import io.github.pierce_lonergan.nexusmatcher.error.ServiceUnavailableException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The status-to-exception mapping, driven by real error bodies.
 *
 * <p>The bodies for 422 and 413 are captures from a running service. The 503 and 504 bodies are the
 * ones the server's own source builds, reproduced here so the typed accessors are pinned without a
 * second fixture server; both are also exercised end to end against real servers in
 * {@code RetryIT} and {@code DeadlineIT}, which is where the claim that they arrive at all is
 * made.
 */
class ErrorEnvelopeMappingTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private NexusMatcherException fromEnvelope(int status, String body) {
        try {
            JsonNode error = mapper.readTree(body).get("error");
            Map<String, Object> details = error.get("details") == null
                    ? Map.of()
                    : mapper.convertValue(
                            error.get("details"), new TypeReference<Map<String, Object>>() {});
            return NexusMatcherException.of(
                    status,
                    error.get("code").asText(),
                    error.get("message").asText(),
                    details,
                    "test-0001");
        } catch (Exception exc) {
            throw new AssertionError(exc);
        }
    }

    @Test
    @DisplayName("422 from an unknown field key names the offending key, and is never retried")
    void unknownFieldKeyIsARequestFailure() {
        NexusMatcherException failure =
                fromEnvelope(422, Fixtures.captured("error-422-unknown-field-key.json"));

        NexusMatcherRequestException request =
                assertInstanceOf(NexusMatcherRequestException.class, failure);
        assertEquals("NEXUS-8004", request.errorCode().orElseThrow());
        assertEquals("test-0001", request.requestId().orElseThrow());
        assertFalse(request.isRetryable(), "the same malformed body produces the same 422");

        List<String> offending = request.violations().stream()
                .filter(violation -> "extra_forbidden".equals(violation.type()))
                .map(violation -> violation.location().get(violation.location().size() - 1))
                .toList();
        assertEquals(
                List.of("flattenedName", "dataType"),
                offending,
                "this is the exact mistake two reviewers made: the pack's fields.json spellings "
                        + "are not the wire contract, and the body says so");
    }

    @Test
    @DisplayName("422 from top_k over the cap carries the cap to ask for instead")
    void topKOverCapCarriesTheCap() {
        NexusMatcherRequestException failure = assertInstanceOf(
                NexusMatcherRequestException.class,
                fromEnvelope(422, Fixtures.captured("error-422-top-k-cap.json")));

        assertEquals(5, failure.resultsPerFieldCap().orElseThrow());
        assertTrue(failure.violations().isEmpty());
    }

    @Test
    @DisplayName("422 from duplicate paths names the paths that collided")
    void duplicatePathsAreNamed() {
        NexusMatcherRequestException failure = assertInstanceOf(
                NexusMatcherRequestException.class,
                fromEnvelope(422, Fixtures.captured("error-422-duplicate-paths.json")));

        assertEquals(List.of("t.a"), failure.duplicatePaths());
    }

    @Test
    @DisplayName("413 from the field cap hands back the chunk size to re-send at")
    void fieldCapIsRechunkable() {
        PayloadTooLargeException failure = assertInstanceOf(
                PayloadTooLargeException.class,
                fromEnvelope(413, Fixtures.captured("error-413-field-cap.json")));

        assertEquals(101, failure.observedFields().orElseThrow());
        assertEquals(100, failure.limitFields().orElseThrow());
        assertEquals(100, failure.suggestedChunkSize().orElseThrow());
        assertFalse(failure.isByteCap());
        assertTrue(failure.limitBytes().isEmpty());
        assertFalse(failure.isRetryable());
    }

    @Test
    @DisplayName("413 from the byte cap carries bytes, not fields, and says so")
    void byteCapCarriesBytes() {
        // The body `body_limit.py` builds. Field counts are absent on this path on purpose:
        // nothing ever parsed the body to count them, so the client must not pretend to know.
        String body = """
                {"error":{"code":"NEXUS-8004","message":"The request body is larger than this \
                server's limit of 9437184 bytes, and was refused without being parsed.",\
                "details":{"limit_bytes":9437184,"observed_bytes":11534336,\
                "source":"content-length","status_code":413}}}
                """;
        PayloadTooLargeException failure =
                assertInstanceOf(PayloadTooLargeException.class, fromEnvelope(413, body));

        assertTrue(failure.isByteCap());
        assertEquals(9437184L, failure.limitBytes().orElseThrow());
        assertEquals(11534336L, failure.observedBytes().orElseThrow());
        assertEquals("content-length", failure.source().orElseThrow());
        assertTrue(
                failure.suggestedChunkSize().isEmpty(),
                "the server has no field count to give here, and inventing one would be a guess "
                        + "presented as the server's answer");
    }

    @Test
    @DisplayName("503 is retryable, and says when retrying will not help")
    void serviceUnavailableIsRetryable() {
        String shed = """
                {"error":{"code":"NEXUS-8000","message":"The matching service is at capacity (36 \
                requests in flight or queued) and shed this request rather than queueing it \
                without limit.","details":{"capacity":36,"in_flight":36,"status_code":503}}}
                """;
        ServiceUnavailableException load =
                assertInstanceOf(ServiceUnavailableException.class, fromEnvelope(503, shed));
        assertTrue(load.isRetryable());
        assertEquals(36, load.capacity().orElseThrow());
        assertEquals(36, load.inFlight().orElseThrow());
        assertFalse(load.isConfigurationProblem());

        String unconfigured = """
                {"error":{"code":"NEXUS-1002","message":"The matching service is not ready: no \
                dictionary is configured.","details":{"reason":"no dictionary is configured.",\
                "status_code":503}}}
                """;
        ServiceUnavailableException config = assertInstanceOf(
                ServiceUnavailableException.class, fromEnvelope(503, unconfigured));
        assertTrue(config.isRetryable());
        assertTrue(
                config.isConfigurationProblem(),
                "both are 503 and only one gets better by waiting; the caller needs to be able "
                        + "to tell an operator to fix something instead of backing off forever");
    }

    @Test
    @DisplayName("504 carries the server's budget and allows exactly one retry")
    void deadlineExceededAllowsOneRetry() {
        String body = """
                {"error":{"code":"NEXUS-6002","message":"Matching did not finish within this \
                server's deadline of 25.0s, so the request was ended rather than left hanging.",\
                "details":{"deadline_seconds":25.0,"status_code":504}}}
                """;
        DeadlineExceededException failure =
                assertInstanceOf(DeadlineExceededException.class, fromEnvelope(504, body));

        assertEquals(25.0, failure.deadlineSeconds().orElseThrow(), 1e-9);
        assertTrue(failure.isRetryable());
        assertEquals(
                1,
                failure.maxSafeRetries(),
                "the timed-out match is still running server-side and still holding its permit, "
                        + "so a second retry adds load to the condition that caused the first");
    }

    @Test
    @DisplayName("500 says the request was not partially answered")
    void serverFailureIsNotRetried() {
        String body = """
                {"error":{"code":"NEXUS-6000","message":"3 fields were sent and 2 came back.",\
                "details":{"fields_in":3,"results_out":2,"status_code":500}}}
                """;
        NexusMatcherServerException failure =
                assertInstanceOf(NexusMatcherServerException.class, fromEnvelope(500, body));

        assertEquals(3, failure.fieldsIn().orElseThrow());
        assertEquals(2, failure.resultsOut().orElseThrow());
        assertFalse(failure.isRetryable());
    }

    @Test
    @DisplayName("an unrecognised status is not assumed safe to retry")
    void unknownStatusIsNotRetryable() {
        NexusMatcherException failure =
                NexusMatcherException.of(429, null, "slow down", Map.of(), "test-0001");

        assertEquals(NexusMatcherException.class, failure.getClass());
        assertFalse(failure.isRetryable());
        assertEquals(429, failure.httpStatus());
    }
}
