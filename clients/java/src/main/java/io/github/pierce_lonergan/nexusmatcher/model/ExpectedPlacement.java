package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalInt;

/**
 * Where the entry the caller named actually landed.
 *
 * <p><strong>{@link #inDictionary()} false is the answer</strong>, and it ends the investigation: a
 * field cannot match an entry that was never indexed, and no amount of threshold tuning fixes it.
 * Check it before reading a single rank.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ExpectedPlacement(

        /** The id the caller named. */
        @JsonProperty("governanceId") String governanceId,

        /** Whether this id exists in the loaded dictionary at all. */
        @JsonProperty("inDictionary") boolean inDictionary,

        /** 1-based rank in each channel's FULL result, or null where that channel did not return
         *  it. Keys are the channel names. */
        @JsonProperty("rankByChannel") Map<String, Integer> rankByChannel) {

    @JsonCreator
    public ExpectedPlacement {
        // LinkedHashMap rather than Map.copyOf: a null value is the documented way this map says
        // "that channel did not return it", and Map.copyOf refuses null values outright.
        rankByChannel = rankByChannel == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(rankByChannel));
    }

    /** The expected entry's rank in one channel, empty when that channel did not return it. */
    public OptionalInt rankIn(String channel) {
        Integer rank = rankByChannel.get(channel);
        return rank == null ? OptionalInt.empty() : OptionalInt.of(rank);
    }

    /** Whether any channel returned the expected entry at all. */
    public boolean retrievedByAnyChannel() {
        return rankByChannel.values().stream().anyMatch(rank -> rank != null);
    }
}
