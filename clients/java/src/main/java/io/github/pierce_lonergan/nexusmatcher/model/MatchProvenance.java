package io.github.pierce_lonergan.nexusmatcher.model;

/**
 * Where a candidate's answer came from: the pipeline scored it, or a human decided it.
 *
 * <h2>Why this exists, stated plainly because it replaces a claim that was false</h2>
 *
 * <p>A candidate a reviewer had already decided used to be identifiable only by inference:
 * {@code confidence} was set to 1.0 and the service's own source asserted that value was outside
 * the range its scorer could produce. <strong>It is not.</strong> The five default scoring weights
 * sum to exactly 1.0 and every signal is attainable at 1.0, so ordinary retrieval reaches 1.0
 * whenever all five are maximal.
 *
 * <p>That is not a theoretical corner. {@code src/test/resources/captured/match-response-approved-pair.json}
 * is a real capture from a running service over the repository's own example pack, and it carries
 * two candidates that agree on {@code (confidence 1.0, decision AUTO_APPROVE)} -- one scored by
 * retrieval, one decided by a reviewer. A client reading those two members cannot tell them apart.
 * This one is the member that can, and a VALUE cannot collide with a score the way a magic number
 * can.
 *
 * <h2>Bound OPEN, with a sentinel, like every other vocabulary in this client except one</h2>
 *
 * <p>The service publishes this as a closed schema component, exactly as it publishes
 * {@link MatchDecision}, {@link FieldDecision}, {@link Separation} and {@link Agreement}. Of those
 * four, only {@code MatchDecision} is bound closed here. The rule this client applies is the one
 * {@link FieldDecision} sets out at length: a closed vocabulary is a promise about what the server
 * will SEND, and an open binding is a decision about what happens if that promise is ever revised.
 * The deciding question is blast radius.
 *
 * <p>A {@code decision} is one candidate of one field, and refusing it refuses an answer.
 * {@code provenance} rides on EVERY candidate of EVERY field -- up to 250 fields on the batch
 * route, several candidates each -- so a closed binding would let one unrecognised word on one
 * runner-up throw away every verdict, every protection class and every candidate in the response.
 * An operator who can read the rest and must escalate one candidate is strictly better off than an
 * operator who can read none.
 *
 * <p>Degrading is safe here for the same reason it is safe on {@link FieldDecision}:
 * {@link #UNKNOWN} is not usable as an answer. Both {@link MatchCandidate#decidedByAReviewer()} and
 * {@link MatchCandidate#wasScored()} answer {@code false} for it -- an unread value must never
 * become a quiet "a human approved this", and it must not become a quiet "the pipeline measured
 * this" either. {@link MatchCandidate#provenance()} keeps the exact string the server sent, so the
 * value can be named in a ticket rather than merely counted.
 *
 * @see MatchCandidate#provenanceValue()
 */
public enum MatchProvenance {

    /**
     * The pipeline retrieved and scored this candidate.
     *
     * <p>Its {@code confidence} is a measurement, {@link MatchCandidate#absoluteScore()} is present
     * unless the dense arm did not return it, and an {@code explain} block is available if the
     * request asked for one.
     */
    RETRIEVAL,

    /**
     * A reviewer decided this pair and matching was skipped for the field.
     *
     * <p>The field comes back with exactly ONE candidate however large {@code top_k} was: ranks 2
     * and beyond would have to come from retrieval, and retrieval did not run. Its
     * {@code confidence} is 1.0 because a human decided it and no measurement was taken that could
     * argue with them -- <strong>not</strong> because 1.0 is a sentinel.
     *
     * <p>{@link MatchCandidate#absoluteScore()} is {@code null} and {@code explain} is absent, and
     * both are absent because <strong>nothing measured it</strong>, never because the measurement
     * came out at zero. See {@link MatchCandidate#wasScored()}.
     *
     * <p>Reachable only on a deployment that has attached a feedback consumer. The service ships
     * none, so on a stock server every candidate is {@link #RETRIEVAL}.
     */
    APPROVED_PAIR,

    /**
     * A provenance a newer server sent that this build does not know.
     * <strong>Never on the wire.</strong>
     *
     * <p>Not an answer, and not a third kind of answer either: it means the server said where this
     * candidate came from and this build cannot read it, so nothing may be concluded about the
     * candidate's numbers or about who stood behind them.
     * {@link MatchCandidate#provenance()} names the value the server actually sent.
     *
     * <p>Also what a response from a server predating the member reads as, because that server
     * sends no value at all. Those two cases are different and this client does not pretend to
     * separate them here -- {@link MatchCandidate#provenance()} is {@code null} for the second and
     * a real string for the first. Neither may be defaulted to {@link #RETRIEVAL}: the bypass
     * existed before the member did, so an older server's silence is genuinely uninformative.
     *
     * <p>Deliberately not published by the service, and
     * {@code tests/packaging/test_java_client_contract.py} asserts that it is not: the moment a
     * real server starts sending the string {@code UNKNOWN}, this constant would stop meaning
     * "unrecognised" and start silently absorbing a real provenance.
     */
    UNKNOWN;

    /**
     * Read one wire value. Never throws; an unrecognised or absent value is {@link #UNKNOWN}.
     *
     * @param value the server's own string, or {@code null} from a server predating the member
     */
    public static MatchProvenance fromWire(String value) {
        for (MatchProvenance candidate : values()) {
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
