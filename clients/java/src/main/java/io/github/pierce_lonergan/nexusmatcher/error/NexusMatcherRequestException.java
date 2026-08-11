package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * The request was malformed or could not be answered as sent: HTTP 400 or 422.
 *
 * <p><strong>Never retried.</strong> Sending the same malformed body again produces the same 422,
 * and the retry policy refuses to.
 *
 * <p>The two cases worth branching on both arrive here. A body-validation failure carries
 * {@link #violations()}, naming the offending field -- which is how you find out you sent
 * {@code flattenedName} instead of {@code name}. A {@code top_k} above the server's configured
 * {@code results_per_field} carries {@link #resultsPerFieldCap()} instead, and duplicate paths
 * carry {@link #duplicatePaths()}.
 */
public class NexusMatcherRequestException extends NexusMatcherException {

    @Serial
    private static final long serialVersionUID = 1L;

    NexusMatcherRequestException(
            String message,
            int httpStatus,
            String errorCode,
            Map<String, Object> details,
            String requestId) {
        super(message, httpStatus, errorCode, details, requestId, null);
    }

    /**
     * One rejected input, as the server's validator described it.
     *
     * @param location the path into the request body, e.g. {@code [body, fields, 0, name]}
     * @param message  why it was rejected
     * @param type     the validator's own type token, e.g. {@code extra_forbidden}
     */
    public record Violation(List<String> location, String message, String type) {
        public Violation {
            location = location == null ? List.of() : List.copyOf(location);
        }

        /** The location as a dotted path, for a log line. */
        public String path() {
            return String.join(".", location);
        }
    }

    /**
     * The fields the server refused and why, empty when this 422 was not a body-validation
     * failure.
     */
    @SuppressWarnings("unchecked")
    public List<Violation> violations() {
        Object raw = details().get("violations");
        if (!(raw instanceof List<?> entries)) {
            return List.of();
        }
        List<Violation> violations = new ArrayList<>(entries.size());
        for (Object entry : entries) {
            if (entry instanceof Map<?, ?> map) {
                Object location = map.get("location");
                List<String> parts = new ArrayList<>();
                if (location instanceof List<?> rawParts) {
                    for (Object part : rawParts) {
                        parts.add(String.valueOf(part));
                    }
                }
                violations.add(new Violation(
                        parts,
                        map.get("message") == null ? "" : String.valueOf(map.get("message")),
                        map.get("type") == null ? "" : String.valueOf(map.get("type"))));
            }
        }
        return List.copyOf(violations);
    }

    /**
     * The server's cap on {@code top_k}, when this failure was a {@code top_k} over it. Ask for at
     * most this many candidates.
     */
    public OptionalInt resultsPerFieldCap() {
        return intDetail("results_per_field");
    }

    /**
     * The paths sent more than once, when that is what was refused. The response is keyed by path,
     * so two fields under one key would leave one of them with no governance and no error -- which
     * is why this is a refusal rather than a silent collapse.
     */
    public List<String> duplicatePaths() {
        Object raw = details().get("duplicate_paths");
        if (!(raw instanceof List<?> entries)) {
            return List.of();
        }
        return entries.stream().map(String::valueOf).toList();
    }
}
