package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * The request was too big for this server: HTTP 413. Re-chunk and send again.
 *
 * <p><strong>Two different refusals arrive under this one status</strong>, and they carry different
 * keys, so a caller who reads only one of them will find the other empty:
 *
 * <ul>
 *   <li><em>Too many fields.</em> The route's own field cap, refused after the body parsed.
 *       {@link #observedFields()} and {@link #limitFields()} are populated.
 *   <li><em>Too many bytes.</em> The raw-body cap, refused by middleware <em>before</em> the body
 *       was read at all. {@link #observedBytes()}, {@link #limitBytes()} and {@link #source()} are
 *       populated, and the field counts are not -- nothing ever parsed the body to count them.
 * </ul>
 *
 * <p>{@link #suggestedChunkSize()} answers the only question a caller actually has, and is empty on
 * the byte-cap path because the server does not know how many fields would have fitted. Feed it to
 * {@code FieldSpec.chunk(fields, size)}.
 *
 * <p><strong>Never retried</strong> as-is: the same body is the same size. The byte-cap refusal
 * also closes the connection without reading the rest of the body -- deliberately, so the bytes are
 * never pulled off the wire -- which some HTTP clients surface as a connection reset instead of a
 * 413. See {@code NexusMatcherTransportException} and the module README for what that looks like.
 */
public class PayloadTooLargeException extends NexusMatcherException {

    @Serial
    private static final long serialVersionUID = 1L;

    PayloadTooLargeException(
            String message, String errorCode, Map<String, Object> details, String requestId) {
        super(message, 413, errorCode, details, requestId, null);
    }

    /** The route's field cap, on the too-many-fields path. */
    public OptionalInt limitFields() {
        return intDetail("limit");
    }

    /** How many fields were sent, on the too-many-fields path. */
    public OptionalInt observedFields() {
        return intDetail("fields");
    }

    /** The server's raw-body byte cap, on the byte-cap path. */
    public OptionalLong limitBytes() {
        return longDetail("limit_bytes");
    }

    /**
     * The body size that tripped the byte cap: what the client declared in {@code Content-Length},
     * or what was counted on the wire. {@link #source()} says which.
     */
    public OptionalLong observedBytes() {
        return longDetail("observed_bytes");
    }

    /**
     * {@code content-length} or {@code stream} -- which measurement refused the body. Worth
     * reading when a proxy between you and the server rewrites {@code Content-Length}.
     */
    public Optional<String> source() {
        return stringDetail("source");
    }

    /** Whether this 413 came from the raw-body byte cap rather than the field cap. */
    public boolean isByteCap() {
        return details().containsKey("limit_bytes");
    }

    /**
     * The largest number of fields to put in the next request, when the server said.
     *
     * <p>Empty on the byte-cap path: the body was refused before anything counted its fields, so
     * the server has no number to give and this client will not invent one. Halve your chunk size
     * and try again.
     */
    public OptionalInt suggestedChunkSize() {
        return limitFields();
    }
}
