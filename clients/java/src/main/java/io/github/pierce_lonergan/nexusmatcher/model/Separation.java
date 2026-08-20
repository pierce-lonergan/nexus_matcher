package io.github.pierce_lonergan.nexusmatcher.model;

/**
 * Whether the two candidates in a {@link Contrast} are actually apart.
 *
 * <p>The service publishes this as a closed schema component, exactly like
 * {@link MatchDecision}. This client still binds it OPEN, and the reason is the same one
 * {@link FieldDecision} gives at length: a closed vocabulary is a promise about what the server
 * will SEND, and an open binding is a decision about what happens if that promise is ever revised.
 * The two questions have different answers here for one reason -- <strong>blast radius</strong>.
 *
 * <p>{@link MatchDecision} is the answer, and one candidate's worth of it. A {@code separation} sits
 * inside a contrast, and a contrast block carries one entry for every field in the request -- up to
 * 250 on the batch route. Refusing the whole body over one unrecognised value would discard 249
 * fields' verdicts, candidates and governance, to protect a reader from a word describing why a
 * runner-up lost. That is not a trade worth making for commentary on the answer.
 *
 * <p>So the record keeps the server's own string on {@link Contrast#separation()} and this enum is
 * what you switch on. {@link #UNKNOWN} is not a value the service publishes, and
 * {@code tests/packaging/test_java_client_contract.py} fails if it ever starts.
 */
public enum Separation {

    /** The margin exceeds {@link ContrastReport#resolution()}: the two really are apart. */
    SEPARATED,

    /**
     * The margin does not exceed the resolution.
     *
     * <p>The two candidates are level in every number this response publishes, and the ordering
     * between them came from the matcher's own sort. Nothing is named as a cause on a tie --
     * dressing a sort order up as a finding is how a review surface starts producing reasons that
     * are not reasons.
     */
    TIED,

    /**
     * A value a newer server sent that this build does not know. <strong>Never on the wire.</strong>
     *
     * <p>Not an answer. {@link Contrast#separation()} carries the string the server actually sent.
     */
    UNKNOWN;

    /** Read one wire value. Never throws; an unrecognised value is {@link #UNKNOWN}. */
    public static Separation fromWire(String value) {
        for (Separation candidate : values()) {
            // UNKNOWN is a client-side sentinel and is not on the wire; matching it by name would
            // let a server publishing the literal "UNKNOWN" be read as "this client did not
            // understand you", which is a different claim.
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
