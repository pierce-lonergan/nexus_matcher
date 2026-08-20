package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * A batch of dictionary ids to resolve exactly.
 *
 * <p>Lookup is not matching. There is no scoring, no ranking, no confidence and no decision, because
 * a hit is exact or it is absent -- so a caller who already HAS the id should never send it through
 * fuzzy matching, which can rank the wrong entry first and costs an encoder call to answer a
 * question a dictionary already answers.
 *
 * <p><strong>Ids must be distinct.</strong> The response is a map keyed by these strings, so a
 * repeated id is a 422 rather than a map quietly shorter than the list that was sent. This record
 * refuses it locally too -- see the constructor -- because that particular mistake is worth catching
 * before it costs a round trip, and because the caller's own de-duplication is the fix either way.
 *
 * <p>The server's id cap for this route is the same number as its batch field cap; read it from
 * {@link ServiceLimits#maxBatchFields()} rather than assuming 250.
 */
public record LookupRequest(

        /** The ids to resolve -- the same identifier a candidate returns as
         *  {@link MatchCandidate#governanceId()}. At least one, and all distinct. */
        @JsonProperty("ids") List<String> ids) {

    public LookupRequest {
        Objects.requireNonNull(ids, "ids");
        if (ids.isEmpty()) {
            throw new IllegalArgumentException(
                    "a lookup needs at least one id; the server refuses an empty list");
        }
        // Structural, not configured -- the same standard FieldSpec applies to itself. A blank id
        // and a repeated id are both refused by the server because the response is KEYED by the id,
        // and neither depends on a limit an operator can tune, so mirroring them here cannot go
        // stale. The length cap is a configured number and is deliberately NOT mirrored.
        Set<String> seen = new LinkedHashSet<>();
        for (String id : ids) {
            Objects.requireNonNull(id, "ids must not contain null");
            if (id.isBlank()) {
                throw new IllegalArgumentException(
                        "a lookup id must not be blank; it is the key the answer comes back under");
            }
            if (!seen.add(id)) {
                throw new IllegalArgumentException(
                        "duplicate lookup id " + id + "; the response is a map keyed by these "
                                + "strings and cannot hold two answers for one key");
            }
        }
        ids = List.copyOf(ids);
    }

    /** Resolve one id. The single-id route answers the identical body with one key. */
    public static LookupRequest of(String id) {
        return new LookupRequest(List.of(id));
    }

    /** Resolve several ids. */
    public static LookupRequest of(List<String> ids) {
        return new LookupRequest(ids);
    }
}
