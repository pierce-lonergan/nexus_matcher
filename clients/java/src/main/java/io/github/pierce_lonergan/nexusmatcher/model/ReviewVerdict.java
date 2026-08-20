package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Objects;

/**
 * A reviewer's verdict on one match: what this client made of the value, and the exact string.
 *
 * <p>The same shape as {@link FieldVerdict}, for the same reason and by the same decision -- see
 * {@link ReviewDecision} for why {@code verdict} is an OPEN vocabulary while
 * {@link MatchDecision} is closed. A bare {@link ReviewDecision} loses the value when a newer
 * server sends a fourth one, and a bare {@link String} takes away the compiler and leaves a caller
 * one typo from a branch that never fires.
 *
 * <p>Serialised as the string it was built from, so a verdict that round-trips through this client
 * reaches the server as what the server said, not as what this build could name.
 *
 * <h2>On the request side it can only ever be a value this build knows</h2>
 *
 * <p>{@link #fromWire(String)} degrades, because it decodes what a server sent. The three constants
 * below and {@link #of(ReviewDecision)} are what a caller CONSTRUCTS from, and
 * {@link #of(ReviewDecision)} refuses {@link ReviewDecision#UNKNOWN}: sending the literal string
 * {@code UNKNOWN} to a server whose vocabulary does not contain it is a 422, and a 422 on this
 * route costs the reviewer their verdict.
 */
public record ReviewVerdict(

        /** What this client made of the value. {@link ReviewDecision#UNKNOWN} if nothing. */
        ReviewDecision decision,

        /** The wire string, verbatim. Never null. */
        String wireValue) {

    /** The reviewer accepted the matcher's suggestion. Pairs with {@code wasCorrect: true}. */
    public static final ReviewVerdict APPROVED = of(ReviewDecision.APPROVED);

    /** Nothing in the glossary governs this field. Pairs with {@code wasCorrect: false}. */
    public static final ReviewVerdict REJECTED = of(ReviewDecision.REJECTED);

    /** The reviewer chose a term the matcher never proposed. Pairs with {@code wasCorrect: false}. */
    public static final ReviewVerdict MANUAL_OVERRIDE = of(ReviewDecision.MANUAL_OVERRIDE);

    public ReviewVerdict {
        Objects.requireNonNull(decision, "decision");
        Objects.requireNonNull(wireValue, "wireValue");
    }

    /**
     * A verdict this client is prepared to SEND.
     *
     * @throws IllegalArgumentException on {@link ReviewDecision#UNKNOWN}, which is a decode
     *     sentinel and not something a reviewer can decide
     */
    public static ReviewVerdict of(ReviewDecision decision) {
        Objects.requireNonNull(decision, "decision");
        if (decision == ReviewDecision.UNKNOWN) {
            throw new IllegalArgumentException(
                    "ReviewDecision.UNKNOWN is this client's sentinel for a verdict it could not "
                            + "read, not a verdict it can send. The server publishes APPROVED, "
                            + "REJECTED and MANUAL_OVERRIDE and would answer 422.");
        }
        return new ReviewVerdict(decision, decision.name());
    }

    /**
     * Read one wire value. Never throws.
     *
     * <p>Degrades to {@link ReviewDecision#UNKNOWN} rather than refusing, exactly as
     * {@link FieldVerdict#fromWire(String)} does, because the service publishes this vocabulary
     * inline precisely so that adding a value to it does not break clients already generated.
     */
    @JsonCreator
    public static ReviewVerdict fromWire(String value) {
        Objects.requireNonNull(value, "value");
        for (ReviewDecision candidate : ReviewDecision.values()) {
            // UNKNOWN is a client-side sentinel and is not on the wire. Matching it by name would
            // let a server that started publishing the literal string "UNKNOWN" have it read as
            // "this client did not understand you", which is a different claim entirely.
            if (candidate != ReviewDecision.UNKNOWN && candidate.name().equals(value)) {
                return new ReviewVerdict(candidate, value);
            }
        }
        return new ReviewVerdict(ReviewDecision.UNKNOWN, value);
    }

    /** The wire string. What this record serialises to. */
    @JsonValue
    public String wireValue() {
        return wireValue;
    }

    /** Whether this client understood the value. False means read {@link #wireValue()}. */
    public boolean isKnown() {
        return decision.isKnown();
    }

    /** Whether this is the verdict that says retrieval MISSED rather than mis-ranked. */
    public boolean isManualOverride() {
        return decision == ReviewDecision.MANUAL_OVERRIDE;
    }
}
