package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Optional;

/**
 * One candidate glossary entry for one schema field.
 *
 * <p>Ranks are 1-based and every rank is judged independently, so a {@link MatchDecision#REJECT}
 * at rank 3 says nothing about rank 1.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MatchCandidate(

        /** 1-based position among this field's candidates. */
        @JsonProperty("rank") int rank,

        /** The matched entry's id, which IS the governance id. Always populated. */
        @JsonProperty("governanceId") String governanceId,

        /** The matched entry's business name. */
        @JsonProperty("businessName") String businessName,

        /** The matched entry's definition. */
        @JsonProperty("definition") String definition,

        /** The matched entry's domain. */
        @JsonProperty("domain") String domain,

        /**
         * The protection class this candidate confers, or {@code null} -- and {@code null} means
         * one of TWO different things.
         *
         * <p>Call {@link #governanceStatus()} rather than testing this for null. The two cases are
         * {@link GovernanceStatus#OPEN_TIER} (the matched entry carries no code, so it sits at the
         * vocabulary's open tier and there is genuinely nothing to confer) and
         * {@link GovernanceStatus#WITHHELD_REJECTED_TOP_MATCH} (rank 1 was rejected, so no entry in
         * the glossary describes this field and it inherits nothing at all). Collapsing them loses
         * the difference between "governed, openly" and "we do not know".
         *
         * <p>A REJECTED RUNNER-UP KEEPS ITS CLASS. Nothing inherits from a runner-up, and the class
         * is what lets a reviewer see that rank 1 is a direct identifier and rank 2 is not -- so
         * this is non-null on a rejected rank 2, and that is not a bug.
         */
        @JsonProperty("governance") Governance governance,

        /**
         * The server's confidence, rank-relative.
         *
         * <p>Do not threshold on it and do not diff against it. It is a min-max normalised fused
         * retrieval score with a structural floor, so a high number is not evidence of a good
         * match -- {@link #decision()} is what carries the verdict. The server's own documentation
         * makes this point twice.
         */
        @JsonProperty("confidence") double confidence,

        /** The verdict. Read this, not {@link #confidence()}. */
        @JsonProperty("decision") MatchDecision decision,

        /** Score components and weights, present only when the request asked to explain. */
        @JsonProperty("explain") Explain explain) {

    @JsonCreator
    public MatchCandidate {
    }

    /**
     * Which of the three governance states this candidate is in, so that the two meanings of a
     * null {@code governance} are visible in the type system rather than only in a document.
     *
     * <p>Derived from what the response already says -- {@code governance}, {@code rank} and
     * {@code decision} -- against the rule the server publishes. It decides nothing: when the class
     * is withheld this still hands back nothing, it just says which kind of nothing.
     */
    public GovernanceStatus governanceStatus() {
        if (governance != null) {
            return GovernanceStatus.CONFERRED;
        }
        if (rank == 1 && decision == MatchDecision.REJECT) {
            return GovernanceStatus.WITHHELD_REJECTED_TOP_MATCH;
        }
        return GovernanceStatus.OPEN_TIER;
    }

    /**
     * The class this candidate confers, empty when it confers none.
     *
     * <p>Empty is ambiguous by design -- it is both nulls at once -- so use
     * {@link #governanceStatus()} when the difference matters, which is whenever you are about to
     * apply a classification.
     */
    public Optional<Governance> governanceValue() {
        return Optional.ofNullable(governance);
    }

    /** The {@link #explain()} block, empty when the request did not ask for one. */
    public Optional<Explain> explainValue() {
        return Optional.ofNullable(explain);
    }
}
