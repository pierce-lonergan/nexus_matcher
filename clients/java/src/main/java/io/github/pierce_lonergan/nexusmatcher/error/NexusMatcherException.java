package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Base of every failure this client raises, carrying the five things needed to act on one.
 *
 * <p>The service answers every documented failure in one envelope --
 * <code>{"error": {code, message, details}}</code> -- and this class is that envelope plus the two
 * facts the envelope does not contain: the HTTP status, and the {@code X-Request-ID} the exchange
 * was answered under. The request id is on <em>every</em> exception on purpose: it is what joins a
 * Java stack trace to the server log line for the same request, and the server stamps it on every
 * response including the ones its own middleware answers.
 *
 * <p><strong>Branch on {@link #httpStatus()} and {@link #errorCode()}, never on the message.</strong>
 * The wording is addressed to whoever reads the log and is explicitly not part of the contract.
 *
 * <p>Unchecked, deliberately. Every call would otherwise carry a five-way catch including
 * {@code health()}; the JDK's own {@code HttpClient} throws a checked {@code IOException} that this
 * client wraps precisely so that a caller can choose where to handle failure rather than being
 * told.
 */
public class NexusMatcherException extends RuntimeException {

    @Serial
    private static final long serialVersionUID = 1L;

    private final int httpStatus;
    private final String errorCode;
    private final Map<String, Object> details;
    private final String requestId;

    protected NexusMatcherException(
            String message,
            int httpStatus,
            String errorCode,
            Map<String, Object> details,
            String requestId,
            Throwable cause) {
        super(message, cause);
        this.httpStatus = httpStatus;
        this.errorCode = errorCode;
        this.details = details == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(details));
        this.requestId = requestId;
    }

    /**
     * The HTTP status, or 0 when no response was received at all (see
     * {@link NexusMatcherTransportException}).
     */
    public int httpStatus() {
        return httpStatus;
    }

    /**
     * The server's stable machine-readable code, e.g. {@code NEXUS-8004}. Empty when the failure
     * happened before a response body could be read.
     */
    public Optional<String> errorCode() {
        return Optional.ofNullable(errorCode);
    }

    /**
     * The failure-specific context, always carrying {@code status_code} when it came from the
     * server. Free-form by design: the keys legitimately differ per failure, which is why the
     * typed accessors live on the subclasses that know which keys their failure carries.
     */
    public Map<String, Object> details() {
        return details;
    }

    /** The {@code X-Request-ID} this exchange was answered under. */
    public Optional<String> requestId() {
        return Optional.ofNullable(requestId);
    }

    /**
     * Whether retrying this exact request could plausibly succeed.
     *
     * <p>False on the base and on every 4xx. Retrying a malformed request just sends the same
     * malformed request again, and doing it three times with backoff turns one loud failure into
     * three slow ones.
     */
    public boolean isRetryable() {
        return false;
    }

    /**
     * How many times a caller may retry at most, when {@link #isRetryable()} is true. Zero
     * otherwise, and deliberately 1 for {@link DeadlineExceededException}.
     */
    public int maxSafeRetries() {
        return 0;
    }

    // -----------------------------------------------------------------------------------------
    // Reading `details`, which is an open map by contract
    // -----------------------------------------------------------------------------------------

    /** One raw entry from {@link #details()}. */
    protected Optional<Object> detail(String key) {
        return Optional.ofNullable(details.get(key));
    }

    protected OptionalInt intDetail(String key) {
        Object value = details.get(key);
        return value instanceof Number number
                ? OptionalInt.of(number.intValue())
                : OptionalInt.empty();
    }

    protected OptionalLong longDetail(String key) {
        Object value = details.get(key);
        return value instanceof Number number
                ? OptionalLong.of(number.longValue())
                : OptionalLong.empty();
    }

    protected OptionalDouble doubleDetail(String key) {
        Object value = details.get(key);
        return value instanceof Number number
                ? OptionalDouble.of(number.doubleValue())
                : OptionalDouble.empty();
    }

    protected Optional<String> stringDetail(String key) {
        Object value = details.get(key);
        return value == null ? Optional.empty() : Optional.of(String.valueOf(value));
    }

    /**
     * Build the right subclass for a status the server answered with.
     *
     * <p>The whole status-to-class mapping lives here, in one visible place, so that adding a
     * failure mode is one edit rather than a hunt. Anything unrecognised becomes a plain
     * {@code NexusMatcherException} -- not a retryable one -- because a status this client has
     * never seen is not one it can claim is safe to send again.
     */
    public static NexusMatcherException of(
            int httpStatus,
            String errorCode,
            String message,
            Map<String, Object> details,
            String requestId) {
        return switch (httpStatus) {
            case 400, 422 ->
                    new NexusMatcherRequestException(message, httpStatus, errorCode, details, requestId);
            case 413 -> new PayloadTooLargeException(message, errorCode, details, requestId);
            case 500 -> new NexusMatcherServerException(message, errorCode, details, requestId);
            case 503 -> new ServiceUnavailableException(message, errorCode, details, requestId);
            case 504 -> new DeadlineExceededException(message, errorCode, details, requestId);
            default ->
                    new NexusMatcherException(message, httpStatus, errorCode, details, requestId, null);
        };
    }
}
