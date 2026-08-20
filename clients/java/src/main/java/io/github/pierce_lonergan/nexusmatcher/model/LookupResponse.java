package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * One answer per requested id, plus what a null class means on the server that answered.
 *
 * <p><strong>Every requested id appears exactly once, in the order it was sent.</strong> An id the
 * dictionary does not carry maps to {@code null} and is named again in {@link #missing()}; it is
 * never simply omitted. A partial list would be the same defect the match plane refuses -- the
 * caller's own key vanishes and the only symptom is a count nobody has reason to check.
 *
 * <p><strong>Not-found is a 200.</strong> A missing entry does not throw and does not arrive as a
 * 404: on this service a 404 means the route does not exist, and confusing "that term was retired"
 * with "you called a path that does not exist" leads to a wrong conclusion about the glossary
 * rather than about the URL. So both answers are in the body, and both are values here.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record LookupResponse(

        /** The caller's own ids in the order sent, each mapped to its entry or to an explicit
         *  null. Never a partial list. */
        Map<String, LookupEntry> results,

        /** Exactly the ids whose {@link #results()} value is null, in the order sent. Derived by
         *  the server from that map in one pass, so the two cannot disagree. */
        List<String> missing,

        /** What a null {@link LookupEntry#governance()} means on this deployment. */
        Vocabulary vocabulary) {

    @JsonCreator
    public static LookupResponse fromBody(
            @JsonProperty("results") Map<String, LookupEntry> results,
            @JsonProperty("missing") List<String> missing,
            @JsonProperty("vocabulary") Vocabulary vocabulary) {
        return new LookupResponse(results, missing, vocabulary);
    }

    public LookupResponse {
        // LinkedHashMap wrapped unmodifiable, NOT Map.copyOf, for two reasons that both matter
        // here: the order is the caller's own request order, and Map.copyOf throws on a null
        // VALUE. A null value is the documented way this response says "no such entry", so
        // copying through Map.copyOf would turn the answer this endpoint exists to give into a
        // NullPointerException during decoding.
        results = results == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(results));
        missing = missing == null ? List.of() : List.copyOf(missing);
    }

    /** The ids that came back, in the order they were sent. */
    public Set<String> ids() {
        return results.keySet();
    }

    /**
     * The entry for one id, empty when the dictionary does not carry it.
     *
     * <p>Empty is also what an id that was never requested gives, which is why {@link #ids()}
     * exists; the server guarantees every id you sent is present.
     */
    public Optional<LookupEntry> entryFor(String id) {
        return Optional.ofNullable(results.get(id));
    }

    /** Whether every requested id resolved. One field read rather than a walk over the map. */
    public boolean allResolved() {
        return missing.isEmpty();
    }

    /**
     * Whether this id was requested and did not resolve.
     *
     * <p>Different from {@code entryFor(id).isEmpty()}, which is also true for an id that was never
     * sent -- the distinction between "your glossary does not have this" and "you did not ask".
     */
    public boolean isMissing(String id) {
        return missing.contains(id);
    }
}
