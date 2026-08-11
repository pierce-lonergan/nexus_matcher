package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * The server failed, or refused a response it could not trust: HTTP 500.
 *
 * <p><strong>No field was classified.</strong> Treat the request as unanswered, never as a partial
 * answer -- that is the server's own wording and it means what it says.
 *
 * <p>Not retried. Three of the server's failure classes land here and none of them get better by
 * being asked again: the matcher raised ({@code NEXUS-6000}, with {@link #cause()}), this layer
 * caught a field that went in and did not come back out ({@code NEXUS-6000}, with
 * {@link #fieldsIn()} and {@link #resultsOut()}), or another layer has drifted from what the HTTP
 * surface reads ({@code NEXUS-1003}). The last two are defects to report, not conditions to wait
 * out.
 */
public class NexusMatcherServerException extends NexusMatcherException {

    @Serial
    private static final long serialVersionUID = 1L;

    NexusMatcherServerException(
            String message, String errorCode, Map<String, Object> details, String requestId) {
        super(message, 500, errorCode, details, requestId, null);
    }

    /** The exception type the matcher raised, when the server named one. */
    public Optional<String> cause() {
        return stringDetail("cause");
    }

    /** Fields sent, on a conservation failure. */
    public OptionalInt fieldsIn() {
        return intDetail("fields_in");
    }

    /** Results that came back, on a conservation failure. Fewer than {@link #fieldsIn()}. */
    public OptionalInt resultsOut() {
        return intDetail("results_out");
    }
}
