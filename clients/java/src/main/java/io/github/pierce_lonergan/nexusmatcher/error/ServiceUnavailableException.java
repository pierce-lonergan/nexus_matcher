package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * The server will not serve this request right now: HTTP 503. Retryable.
 *
 * <p>Two conditions answer 503 and they want different things from the caller:
 *
 * <ul>
 *   <li><em>Shed load.</em> The bounded in-flight queue is full, so the request was refused rather
 *       than queued without limit. {@link #capacity()} and {@link #inFlight()} are populated. This
 *       is the one retrying actually helps -- back off and come back.
 *   <li><em>Nothing loaded.</em> No dictionary is configured, or feedback recording is not, so the
 *       route cannot answer at all. {@link #reason()} or {@link #setting()} names what an operator
 *       must change. Retrying will fail identically until somebody does.
 * </ul>
 *
 * <p>{@link #isRetryable()} is true for both, because the client cannot tell a misconfigured server
 * from a busy one <em>at the status code</em>, and the retry policy is capped anyway -- three
 * attempts against a server with no dictionary costs a few hundred milliseconds and then reports
 * the operator's message. Use {@link #setting()} to decide whether retrying is worth it at all.
 */
public class ServiceUnavailableException extends NexusMatcherException {

    @Serial
    private static final long serialVersionUID = 1L;

    ServiceUnavailableException(
            String message, String errorCode, Map<String, Object> details, String requestId) {
        super(message, 503, errorCode, details, requestId, null);
    }

    @Override
    public boolean isRetryable() {
        return true;
    }

    @Override
    public int maxSafeRetries() {
        return Integer.MAX_VALUE;
    }

    /** How much work the server admits at once, on the shed-load path. */
    public OptionalInt capacity() {
        return intDetail("capacity");
    }

    /** How much was in flight when this request was shed. */
    public OptionalInt inFlight() {
        return intDetail("in_flight");
    }

    /** Why the server is not ready, when it is a configuration problem rather than load. */
    public Optional<String> reason() {
        return stringDetail("reason");
    }

    /** The environment variable an operator must set, when the server named one. */
    public Optional<String> setting() {
        return stringDetail("setting");
    }

    /**
     * Whether this 503 is a configuration problem rather than shed load -- in which case retrying
     * will not help and the message is for an operator, not for a backoff loop.
     */
    public boolean isConfigurationProblem() {
        return details().containsKey("reason") || details().containsKey("setting");
    }
}
