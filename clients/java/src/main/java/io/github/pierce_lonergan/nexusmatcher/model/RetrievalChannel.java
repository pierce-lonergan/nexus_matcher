package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;

/** What one retrieval channel returned, and what it was asked for. */
@JsonIgnoreProperties(ignoreUnknown = true)
public record RetrievalChannel(

        /** False when this channel is not wired, or could not run. {@link #detail()} says which. */
        @JsonProperty("available") boolean available,

        /** Why the channel is unavailable, or null when it ran normally. */
        @JsonProperty("detail") String detail,

        /** The depth the channel was searched to. */
        @JsonProperty("requestedTopK") Integer requestedTopK,

        /** How many candidates it returned, BEFORE the display truncation -- so this can exceed
         *  {@code candidates().size()} and that is not a defect. */
        @JsonProperty("returned") Integer returned,

        /** The top candidates, truncated to the request's {@code top_k} for display. */
        @JsonProperty("candidates") List<RetrievalCandidate> candidates) {

    @JsonCreator
    public RetrievalChannel {
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
    }

    /** Why this channel is unavailable, empty when it ran normally. */
    public Optional<String> unavailableBecause() {
        return available ? Optional.empty() : Optional.ofNullable(detail);
    }

    /** {@link #returned()} without the null. */
    public OptionalInt returnedCount() {
        return returned == null ? OptionalInt.empty() : OptionalInt.of(returned);
    }

    /** Whether this channel returned an entry at all, at any rank within the displayed slice. */
    public boolean shows(String governanceId) {
        return candidates.stream()
                .anyMatch(candidate -> candidate.governanceId().equals(governanceId));
    }
}
