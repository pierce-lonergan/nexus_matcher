package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * The numbers in force on the live matcher, read off its own configuration rather than off the
 * shipped defaults -- so a tuned deployment reports ITS numbers.
 *
 * <p><strong>Every member is nullable and null means exactly one thing: this matcher does not expose
 * that setting.</strong> It is never a defaulted number, and the boxed types here are load-bearing
 * for the same reason {@link MatchCandidate#absoluteScore()}'s is. A primitive {@code double} would
 * bind an unreported {@code autoApprove} to 0.0, which tells an operator that everything
 * auto-approves -- the most expensive wrong answer this block could give, and indistinguishable
 * from a deployment that really did configure 0.0.
 *
 * <p>Note what is NOT here: the absolute-score floor. It is published on a match response's
 * {@link ScoringContract#absoluteScoreFloor()} instead, so a caller who wants to know whether
 * {@link FieldDecision#NO_MATCH} can fire on score has to read it from there.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Thresholds(

        /** The confidence at or above which rank 1 is AUTO_APPROVE. */
        @JsonProperty("autoApprove") Double autoApprove,

        /** The confidence below which a candidate is REJECT. */
        @JsonProperty("review") Double review,

        /** The minimum gap between rank 1 and rank 2 for an auto-approval. */
        @JsonProperty("minConfidenceGap") Double minConfidenceGap,

        /** The server's default candidates per field, and the cap a request's {@code top_k}
         *  may not exceed. */
        @JsonProperty("resultsPerField") Integer resultsPerField,

        /** The dense/sparse fusion weight. */
        @JsonProperty("fusionAlpha") Double fusionAlpha,

        /**
         * The lowest confidence a RANK-1 candidate can structurally carry on this configuration.
         *
         * <p>Null when a reranker is wired: a reranker replaces the fused score and the derivation
         * lapses, and a bound that quietly does not hold is worse than no bound.
         */
        @JsonProperty("minimumAchievableConfidence") Double minimumAchievableConfidence,

        /**
         * Exactly {@code review < minimumAchievableConfidence}, and nothing more.
         *
         * <p><strong>True on a stock deployment</strong>, and it is not a defect -- it is the
         * property that made {@link FieldDecision#NO_MATCH} necessary. When true, no rank-1
         * candidate can fall below the review threshold on score alone, so no field is ever sent to
         * review BY SCORE and a confidence filter set anywhere at or below the floor selects
         * nothing however bad the matches are. Null when either side is null.
         */
        @JsonProperty("reviewThresholdBelowFloor") Boolean reviewThresholdBelowFloor) {

    @JsonCreator
    public Thresholds {
    }

    /** {@link #autoApprove()} without the null. Empty means the matcher does not expose it. */
    public OptionalDouble autoApproveValue() {
        return autoApprove == null ? OptionalDouble.empty() : OptionalDouble.of(autoApprove);
    }

    /** {@link #review()} without the null. */
    public OptionalDouble reviewValue() {
        return review == null ? OptionalDouble.empty() : OptionalDouble.of(review);
    }

    /** {@link #resultsPerField()} without the null. This is the cap on a request's {@code top_k}. */
    public OptionalInt resultsPerFieldValue() {
        return resultsPerField == null ? OptionalInt.empty() : OptionalInt.of(resultsPerField);
    }

    /** {@link #minimumAchievableConfidence()} without the null. Empty when a reranker is wired. */
    public OptionalDouble confidenceFloorValue() {
        return minimumAchievableConfidence == null
                ? OptionalDouble.empty()
                : OptionalDouble.of(minimumAchievableConfidence);
    }
}
