package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.OptionalInt;

/**
 * The two facts about the server's loaded vocabulary that a match response cannot be READ without.
 *
 * <p>It rides on the response rather than sitting behind an endpoint, because the response is a
 * governance artifact that gets pasted into a ticket -- and an artifact whose {@code null} means
 * "ask a second system" is not one.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Vocabulary(

        /**
         * The tier a field with no protection code sits at, in the caller's own vocabulary. This is
         * what {@link GovernanceStatus#OPEN_TIER} MEANS on this deployment.
         *
         * <p>{@code UNCLASSIFIED} is the library's sentinel and indicates that no vocabulary is
         * configured -- deliberately not a word a real taxonomy uses, so it cannot be mistaken for
         * a real tier. {@link #isConfigured()} tests for it.
         */
        @JsonProperty("openClassification") String openClassification,

        /**
         * The caller's declared tier ordering, most open first.
         *
         * <p>The only thing that can rank two classifications against each other. Empty when the
         * vocabulary declares no ordering: treat tiers as incomparable there, never as
         * alphabetical.
         */
        @JsonProperty("tiersMostOpenFirst") List<String> tiersMostOpenFirst) {

    /**
     * The sentinel the library uses when no vocabulary is configured. Matches
     * {@code OPEN_CLASSIFICATION} in {@code nexus_matcher.domain.governance}.
     */
    public static final String UNCONFIGURED_OPEN_CLASSIFICATION = "UNCLASSIFIED";

    @JsonCreator
    public Vocabulary {
        tiersMostOpenFirst = tiersMostOpenFirst == null
                ? List.of()
                : List.copyOf(tiersMostOpenFirst);
    }

    /** Whether the server has a caller-supplied vocabulary loaded, rather than the sentinel. */
    public boolean isConfigured() {
        return !UNCONFIGURED_OPEN_CLASSIFICATION.equals(openClassification);
    }

    /**
     * Where a tier sits in the declared ordering, 0 being the most open.
     *
     * <p>A lookup in the list the caller declared, nothing more. Empty when this deployment
     * declared no ordering or does not know the tier -- in which case the two tiers are
     * incomparable, and this client will not invent an order for them.
     */
    public OptionalInt openness(String classification) {
        int index = tiersMostOpenFirst.indexOf(classification);
        return index < 0 ? OptionalInt.empty() : OptionalInt.of(index);
    }
}
