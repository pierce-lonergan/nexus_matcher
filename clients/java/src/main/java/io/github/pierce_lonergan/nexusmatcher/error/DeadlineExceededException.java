package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * The server's own deadline fired before matching finished: HTTP 504.
 *
 * <p>Retryable, but <strong>ONCE AT MOST</strong>, and that cap is not caution -- it is arithmetic.
 * The server does not kill the worker when the deadline fires: CPU-bound Python cannot be
 * interrupted, so the match that timed out is <em>still running</em> and still holding its
 * admission permit. An immediate retry therefore adds a second copy of the same work to a server
 * that has just demonstrated it could not finish the first, and the in-flight count that produced
 * the 504 goes up rather than down. Retry twice and the third request meets a 503, because
 * admission has filled with work nobody is waiting for any more.
 *
 * <p>So {@link #maxSafeRetries()} is 1, and the default retry policy does not retry this at all --
 * see {@code ExponentialBackoffRetryPolicy.Builder#retryDeadlineExceeded(boolean)} to opt in. The
 * better response is usually {@link #deadlineSeconds()}: send fewer fields per request, or ask the
 * operator to raise {@code NEXUS_API_DEADLINE_SECONDS}.
 *
 * <p>The deadline promises a RESPONSE, not a stop. Whether the work completed server-side after
 * you gave up is not observable from here, and no field was classified as far as this caller is
 * concerned -- treat the request as unanswered.
 */
public class DeadlineExceededException extends NexusMatcherException {

    @Serial
    private static final long serialVersionUID = 1L;

    DeadlineExceededException(
            String message, String errorCode, Map<String, Object> details, String requestId) {
        super(message, 504, errorCode, details, requestId, null);
    }

    @Override
    public boolean isRetryable() {
        return true;
    }

    @Override
    public int maxSafeRetries() {
        return 1;
    }

    /**
     * The server-side budget that fired, in seconds. An operator reading a client's log can tell a
     * too-tight server deadline from a genuinely slow match from this number alone.
     */
    public OptionalDouble deadlineSeconds() {
        return doubleDetail("deadline_seconds");
    }
}
