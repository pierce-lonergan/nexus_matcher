package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Optional;

/**
 * Rank 1 against rank 2 for one field: what separated them, and what decided it.
 *
 * <p><strong>This is not {@link Explain}.</strong> An explain block reports why the winner scored
 * what it did, using weights that are the same for every candidate and are already published. The
 * question a reviewer looking at a surprising match actually has is "why not the other one", and
 * that is a subtraction between two candidates rather than a description of one.
 *
 * <h2>What is deliberately not claimed</h2>
 *
 * <p>A difference at or below {@link ContrastReport#resolution()} is never reported as separating
 * and can never be named as a cause. When the whole margin is at or below it, {@link #isTied()} is
 * true, {@link #largestDifference()} is null and {@link #decidingSignals()} is empty: the order
 * between the two came from the matcher's own sort, and dressing a sort order up as a finding is
 * how a review surface starts producing reasons that are not reasons. The per-signal differences
 * are still reported on a tie, because two signals that disagree and cancel is exactly the case
 * worth seeing.
 *
 * <h2>The two facts that are not about scoring at all</h2>
 *
 * <p>{@link #governanceDiffers()} and {@link #domainDiffers()} come from the two glossary ENTRIES
 * rather than from any signal, and they are usually what settles a review: that rank 1 carries a
 * direct-identifier class and rank 2 does not is the deciding fact far more often than a
 * fourth-decimal score difference is. They are read from the entries' own codes, so a rank-1
 * {@link MatchDecision#REJECT} -- which confers no class by design -- does not read as "these two
 * are classified differently" when they are not.
 *
 * <h2>{@link #separation()} keeps the server's own string</h2>
 *
 * <p>The service publishes {@code separation} as a closed schema component, and this record still
 * holds it as a {@link String} with {@link Separation} beside it as the thing you switch on. See
 * {@link Separation} for why: refusing a whole response over one unrecognised word describing why
 * a runner-up lost would discard up to 250 fields' worth of answers. {@link #isTied()} and
 * {@link #isSeparated()} are both false for a value a newer server might add -- read
 * {@link #separationValue()} or {@link #separation()} when you need to tell that apart.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Contrast(

        /** The governance id of rank 1 -- the candidate the field's answer came from. */
        @JsonProperty("topGovernanceId") String topGovernanceId,

        /** The governance id of rank 2, the candidate this contrast is against. */
        @JsonProperty("runnerUpGovernanceId") String runnerUpGovernanceId,

        /** Rank 1's confidence, at the published precision. */
        @JsonProperty("topConfidence") double topConfidence,

        /** Rank 2's confidence, at the published precision. */
        @JsonProperty("runnerUpConfidence") double runnerUpConfidence,

        /**
         * {@code topConfidence - runnerUpConfidence}.
         *
         * <p>Comparable WITHIN this field only, because {@code confidence} is -- see
         * {@link ContrastReport#confidenceGapScope()} and {@link ScoringContract#scopeOf(String)}.
         * A difference is no more comparable than its operands, so a fixed cut point on this
         * number across a schema means nothing.
         */
        @JsonProperty("confidenceGap") double confidenceGap,

        /**
         * The same margin reached the other way: the sum of every {@link SignalDifference#weightedDelta()}.
         *
         * <p>Published so the arithmetic can be checked from the response alone. The service
         * verifies the two against each other and refuses to answer rather than send a contrast
         * that does not close.
         */
        @JsonProperty("signalGap") double signalGap,

        /** {@code SEPARATED} or {@code TIED}, as the server spelled it.
         *  {@link #separationValue()} is the typed reading. See the type javadoc. */
        @JsonProperty("separation") String separation,

        /**
         * The separating signal with the largest weighted difference -- the headline answer to
         * "what separated these two". Null on a tie, and null when no signal differs by more than
         * the resolution.
         */
        @JsonProperty("largestDifference") String largestDifference,

        /**
         * Every signal whose removal would leave rank 2 level with or ahead of rank 1.
         *
         * <p><strong>Empty is a real answer, and the common one on a wide margin:</strong> it means
         * no single signal carried the margin, not that the server declined to look. Always empty
         * on a tie.
         */
        @JsonProperty("decidingSignals") List<String> decidingSignals,

        /** Whether the two glossary entries carry different protection codes. */
        @JsonProperty("governanceDiffers") boolean governanceDiffers,

        /** Whether the two glossary entries declare different domains. */
        @JsonProperty("domainDiffers") boolean domainDiffers,

        /**
         * One entry per weighted signal, LARGEST WEIGHTED DIFFERENCE FIRST, with ties broken by the
         * order the signals are declared in -- so two identical requests order this list
         * identically and a diff between two runs is a real change.
         */
        @JsonProperty("signals") List<SignalDifference> signals) {

    @JsonCreator
    public Contrast {
        decidingSignals = decidingSignals == null ? List.of() : List.copyOf(decidingSignals);
        signals = signals == null ? List.of() : List.copyOf(signals);
    }

    /** {@link #separation()} as a value you can switch on. {@link Separation#UNKNOWN} if this
     *  build does not know the server's word for it. */
    public Separation separationValue() {
        return Separation.fromWire(separation);
    }

    /**
     * Whether the two candidates are level in every number this response publishes.
     *
     * <p>False for a {@link #separation()} value this build does not know, so do not read the
     * negation as "these two are separated" -- use {@link #isSeparated()} for that, and read
     * {@link #separationValue()} when both answer false.
     */
    public boolean isTied() {
        return separationValue() == Separation.TIED;
    }

    /** Whether the margin exceeded {@link ContrastReport#resolution()}. False on an unknown value. */
    public boolean isSeparated() {
        return separationValue() == Separation.SEPARATED;
    }

    /** {@link #largestDifference()} without the null. Empty on a tie, and when nothing separated. */
    public Optional<String> largestDifferenceValue() {
        return Optional.ofNullable(largestDifference);
    }

    /**
     * Whether any single signal carried the margin.
     *
     * <p>False is the common answer on a wide margin and means no ONE signal decided it -- never
     * that the contrast is missing. {@link #signals()} still carries every difference.
     */
    public boolean hasDecidingSignal() {
        return !decidingSignals.isEmpty();
    }

    /** One signal's difference by name, empty when this response carries no such signal. */
    public Optional<SignalDifference> signal(String name) {
        return signals.stream().filter(each -> each.signal().equals(name)).findFirst();
    }

    /** Only the signals that differ by more than the resolution, in the order the server sent. */
    public List<SignalDifference> separatingSignals() {
        return signals.stream().filter(SignalDifference::separating).toList();
    }
}
