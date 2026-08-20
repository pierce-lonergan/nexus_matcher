package io.github.pierce_lonergan.nexusmatcher.model;

/**
 * Whether a {@link ConceptGroup}'s members were given the same answer.
 *
 * <p>Published by the service as a closed schema component and bound OPEN here, for the reason
 * {@link Separation} sets out: a consistency report covers every field in the request, and refusing
 * a whole response over one unrecognised word describing a group would cost a caller every verdict
 * in the batch. {@link ConceptGroup#agreement()} carries the server's own string.
 *
 * <p><strong>{@link #DISAGREE} is not a defect report about the matcher.</strong> It says these
 * columns were GROUPED and then answered differently, and it is only as good as the grouping.
 * {@link ConsistencyReport} carries the measurement, and the check to run first is
 * {@link ConceptGroup#distinctAnswers()} against {@link ConceptGroup#answeredCount()}.
 */
public enum Agreement {

    /** Two or more members answered and all gave the same governance id. */
    AGREE,

    /**
     * Two or more members answered and they did not agree.
     *
     * <p>Read {@link ConsistencyReport} before acting on one. At the loose grouping key, on a
     * schema built from repeated leaf names, every finding of this kind the service produced was a
     * collision of distinct concepts rather than a contradiction.
     */
    DISAGREE,

    /**
     * Fewer than two members answered at all, so there was nothing to compare.
     *
     * <p>Deliberately not {@link #AGREE}: one answer and five blanks is not five columns
     * confirming each other. A null in {@link ConceptGroup#answers()} is silence, not assent.
     */
    UNDECIDED,

    /**
     * A value a newer server sent that this build does not know. <strong>Never on the wire.</strong>
     *
     * <p>Not an answer, and in particular not a quiet {@link #AGREE}:
     * {@link ConceptGroup#disagrees()} and {@link ConceptGroup#agrees()} are both false for it.
     */
    UNKNOWN;

    /** Read one wire value. Never throws; an unrecognised value is {@link #UNKNOWN}. */
    public static Agreement fromWire(String value) {
        for (Agreement candidate : values()) {
            if (candidate != UNKNOWN && candidate.name().equals(value)) {
                return candidate;
            }
        }
        return UNKNOWN;
    }

    /** Whether this client understood the server's value. False only for {@link #UNKNOWN}. */
    public boolean isKnown() {
        return this != UNKNOWN;
    }
}
