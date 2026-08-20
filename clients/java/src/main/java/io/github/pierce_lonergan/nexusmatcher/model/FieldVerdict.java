package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Objects;

/**
 * One entry of {@code fieldDecisions}: the verdict this client understood, and the exact string the
 * server sent.
 *
 * <p>Both halves, because either alone is wrong. A bare {@link FieldDecision} loses the value when
 * a newer server sends a fifth one -- forty columns come back {@link FieldDecision#UNKNOWN} and an
 * operator cannot tell whether that is one new verdict or three, nor what to search the release
 * notes for. A bare {@code String} gives that back and takes away the compiler: a caller writes
 * {@code "AUTO_APPROVE".equals(v)} and is one typo from a branch that never fires, which is the
 * string test this whole client exists to remove.
 *
 * <p>So the enum is the thing you switch on and the string is the thing you log. The map on
 * {@link MatchResponse#fieldDecisions()} is keyed exactly like {@code results} and in the same
 * order, so the two can be walked together.
 *
 * <p>Decoded from a JSON string, and serialised back to the same string it arrived as -- including
 * an unrecognised one, so a decoded response re-encodes to what the server said rather than to what
 * this build could name.
 */
public record FieldVerdict(

        /** What this client made of the server's value. {@link FieldDecision#UNKNOWN} if nothing. */
        FieldDecision decision,

        /** The server's own string, verbatim. Never null. */
        String wireValue) {

    public FieldVerdict {
        Objects.requireNonNull(decision, "decision");
        Objects.requireNonNull(wireValue, "wireValue");
    }

    /**
     * Read one wire value. Never throws.
     *
     * <p>See {@link FieldDecision} for why this degrades where {@link MatchDecision#fromWire(String)}
     * refuses. In short: the server has committed to freezing {@code decision} and has already
     * demonstrated that it grows THIS vocabulary, one unknown value here would cost every other
     * field's verdict in the same response, and {@link FieldDecision#UNKNOWN} cannot be mistaken for
     * an answer.
     */
    @JsonCreator
    public static FieldVerdict fromWire(String value) {
        Objects.requireNonNull(value, "value");
        for (FieldDecision candidate : FieldDecision.values()) {
            // UNKNOWN is a client-side sentinel and is not on the wire. Matching it by name would
            // let a server that started publishing the literal string "UNKNOWN" have it read as
            // "this client did not understand you", which is a different claim entirely.
            if (candidate != FieldDecision.UNKNOWN && candidate.name().equals(value)) {
                return new FieldVerdict(candidate, value);
            }
        }
        return new FieldVerdict(FieldDecision.UNKNOWN, value);
    }

    /** The wire string. What this record serialises to. */
    @JsonValue
    public String wireValue() {
        return wireValue;
    }

    /** Whether this client understood the server's value. False means read {@link #wireValue()}. */
    public boolean isKnown() {
        return decision.isKnown();
    }

    /** Whether this verdict permits inheriting rank 1's class. True for AUTO_APPROVE alone. */
    public boolean maySafelyInherit() {
        return decision.maySafelyInherit();
    }

    /** Whether this field's verdict is {@link FieldDecision#NO_MATCH}. */
    public boolean isNoMatch() {
        return decision == FieldDecision.NO_MATCH;
    }
}
