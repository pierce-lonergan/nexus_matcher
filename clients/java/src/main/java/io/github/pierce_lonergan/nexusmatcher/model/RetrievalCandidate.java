package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Optional;

/**
 * One candidate as a retrieval channel returned it, before any scoring.
 *
 * <p>This is NOT a {@link MatchCandidate}. It has no governance, no decision and no confidence,
 * because none of those exists yet at this point in the pipeline -- and the missing members are the
 * honest signal that {@link #rank()} here is not a final rank.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record RetrievalCandidate(

        /** 1-based position within this channel, before any scoring. */
        @JsonProperty("rank") int rank,

        /** The entry's id. */
        @JsonProperty("governanceId") String governanceId,

        /** The entry's business name, or null when the channel returned an id the dictionary does
         *  not carry -- which is itself a finding. */
        @JsonProperty("businessName") String businessName,

        /**
         * This channel's RAW score.
         *
         * <p><strong>The channels are not comparable with each other.</strong> {@code dense} is a
         * cosine similarity, {@code sparse} is the retriever's own lexical score on its own scale,
         * and {@code fused} is a min-max normalised weighted sum. A number here means something only
         * against other numbers from the same channel.
         */
        @JsonProperty("score") double score) {

    @JsonCreator
    public RetrievalCandidate {
    }

    /** {@link #businessName()} as an {@link Optional}; empty means the id is not in the dictionary. */
    public Optional<String> name() {
        return Optional.ofNullable(businessName);
    }
}
