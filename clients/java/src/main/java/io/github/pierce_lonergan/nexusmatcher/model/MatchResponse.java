package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Set;

/**
 * One match response: the candidates for every field that was sent, plus what a null class means
 * on this deployment.
 *
 * <p><strong>Every field sent comes back, under the caller's own {@code path}, in the order sent.</strong>
 * A field nothing matched gets an empty list, never a missing key -- the server refuses to answer
 * at all rather than return a map that is short one entry. {@link #results()} is insertion-ordered
 * to preserve that, so iterating it walks the fields in request order.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MatchResponse(

        /** Candidates per field, keyed by the caller's own path, in the order sent. */
        Map<String, List<MatchCandidate>> results,

        /** What a {@link GovernanceStatus#OPEN_TIER} null means on the server that answered. */
        Vocabulary vocabulary,

        /** The {@code X-Request-ID} this exchange was answered under. Null only if the server
         *  omitted the header, which it does not. */
        String requestId,

        /** The server's own {@code X-Response-Time-Ms}, null when the header was absent. */
        Double responseTimeMs) {

    /**
     * Bind the wire body. Transport metadata is attached afterwards by
     * {@link #withTransport(String, Double)}, because it arrives in headers rather than in the
     * body and a DTO that could be populated from the body would be a DTO a body could lie about.
     */
    @JsonCreator
    public static MatchResponse fromBody(
            @JsonProperty("results") Map<String, List<MatchCandidate>> results,
            @JsonProperty("vocabulary") Vocabulary vocabulary) {
        return new MatchResponse(results, vocabulary, null, null);
    }

    public MatchResponse {
        // LinkedHashMap, not Map.copyOf: Map.copyOf does not promise iteration order, and the
        // order IS the contract here -- it is the order the caller sent their fields in.
        results = results == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(results));
    }

    /** This response with the correlation headers from the exchange that carried it. */
    public MatchResponse withTransport(String newRequestId, Double newResponseTimeMs) {
        return new MatchResponse(results, vocabulary, newRequestId, newResponseTimeMs);
    }

    /** The paths that came back, in the order they were sent. */
    public Set<String> paths() {
        return results.keySet();
    }

    /**
     * The candidates for one path, in rank order.
     *
     * <p>An empty list means the field was matched and nothing was found. A path that was never
     * sent also gives an empty list -- use {@link #paths()} if you need to tell those apart, though
     * the server guarantees every path you sent is present.
     */
    public List<MatchCandidate> candidatesFor(String path) {
        return results.getOrDefault(path, List.of());
    }

    /** The rank-1 candidate for one path, empty when nothing matched it. */
    public Optional<MatchCandidate> topCandidateFor(String path) {
        List<MatchCandidate> candidates = candidatesFor(path);
        return candidates.isEmpty() ? Optional.empty() : Optional.of(candidates.get(0));
    }

    /** {@link #responseTimeMs()} without the null. */
    public OptionalDouble responseTime() {
        return responseTimeMs == null ? OptionalDouble.empty() : OptionalDouble.of(responseTimeMs);
    }
}
