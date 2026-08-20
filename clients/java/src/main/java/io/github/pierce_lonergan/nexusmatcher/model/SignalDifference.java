package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * One weighted signal's contribution to the margin between rank 1 and rank 2.
 *
 * <p>{@link #delta()} is exactly the subtraction a reviewer would do by hand on the two
 * candidates' {@code explain.scores} entries: the server rounds both operands to the published
 * precision <em>before</em> subtracting, so redoing it from the response gives this number rather
 * than one that disagrees in the last place.
 *
 * <p><strong>{@link #separating()} and {@link #deciding()} are different questions and neither
 * implies the other.</strong> Separating means the two scores differ by more than
 * {@link ContrastReport#resolution()} -- a difference the reviewer can actually see in the numbers
 * they are holding. Deciding means the arithmetic one: removing this signal's contribution would
 * leave rank 2 level with or ahead of rank 1. A wide margin routinely has several separating
 * signals and no deciding one, and that is a real answer -- no single signal carried it.
 *
 * @see Contrast the pair this difference belongs to
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record SignalDifference(

        /** The signal's name -- the same key it carries in {@code explain.scores} and
         *  {@code explain.weights}, so the three can be read together. */
        @JsonProperty("signal") String signal,

        /** Rank 1's score for this signal, at the published precision. */
        @JsonProperty("topScore") double topScore,

        /** Rank 2's score for this signal, at the published precision. */
        @JsonProperty("runnerUpScore") double runnerUpScore,

        /** {@code topScore - runnerUpScore}. Negative where rank 2 won this signal. */
        @JsonProperty("delta") double delta,

        /** The live matcher's weight for this signal, as {@code explain.weights} reports it. */
        @JsonProperty("weight") double weight,

        /**
         * {@code delta * weight}: this signal's share of the confidence gap.
         *
         * <p>The shares sum to {@link Contrast#confidenceGap()} and the service refuses to answer
         * at all rather than send a contrast where they do not, so this is arithmetic a caller may
         * re-run as a check rather than a number it has to take on trust.
         */
        @JsonProperty("weightedDelta") double weightedDelta,

        /**
         * False when the two scores differ by no more than {@link ContrastReport#resolution()}.
         *
         * <p>A signal that is not separating is never named as a cause. A reason invisible in the
         * artifact the reviewer is holding is an invented one.
         */
        @JsonProperty("separating") boolean separating,

        /**
         * True when removing this signal's contribution would leave rank 2 level with or ahead of
         * rank 1.
         *
         * <p>Arithmetic, not judgement. It can be true of none of the signals, and is always false
         * on a {@link Contrast#isTied()} contrast, where nothing decided the order.
         */
        @JsonProperty("deciding") boolean deciding) {

    /** Whether rank 2 scored higher than rank 1 on this signal. A separating negative delta. */
    public boolean wonByRunnerUp() {
        return separating && delta < 0.0;
    }
}
