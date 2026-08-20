package io.github.pierce_lonergan.nexusmatcher.model;

import java.util.Optional;

/**
 * What a reviewer did with one match: the vocabulary of {@link Feedback#verdict()}.
 *
 * <h2>Why this exists at all, when {@code wasCorrect} is already a boolean</h2>
 *
 * <p>A boolean has two states and the vocabulary has three. The one it cannot express is the
 * reviewer who chose a term <strong>the matcher never proposed</strong> -- not rank 2, not rank 20:
 * absent from the candidate list entirely. Collapsed into {@code false}, that record is
 * byte-identical to "the top match was wrong and I took the third one", and those are opposite
 * diagnoses. The second says the answer was retrieved and mis-ranked, which weights, fusion or a
 * reranker can fix. The first says the answer was never retrieved, which no amount of re-ranking a
 * list that never contained it will ever fix. A trail that stores both as {@code false} cannot
 * count either.
 *
 * <h2>This enum is OPEN, and it is open on purpose rather than by omission</h2>
 *
 * <p>The service publishes {@code verdict} <strong>inline</strong> on the property -- three values
 * as documentation -- and deliberately does NOT publish it as a named schema component. That is the
 * server declining to hand every generated client a closed type that breaks on the day a fourth
 * value is added, and binding it closed here would throw that decision away on this client's side
 * of the wire.
 *
 * <p>So this follows {@link FieldDecision} rather than {@link MatchDecision}: an unrecognised value
 * degrades to {@link #UNKNOWN} and {@link ReviewVerdict#wireValue()} keeps the exact string the
 * server sent, so the value can be named in a ticket rather than merely counted. Nothing here maps
 * an unknown value onto the nearest known one, which is the failure worth fearing.
 *
 * <h2>The fourth server-side value that is deliberately absent</h2>
 *
 * <p>The service's own domain vocabulary carries {@code UNSPECIFIED} -- what a record written
 * before {@code verdict} existed reads as. It is not offered on the wire and it is not offered
 * here, because it is not something a reviewer can decide. A verdict that was never given is an
 * absent {@link Feedback#verdict()}, never a value asserting an absence.
 *
 * @see ReviewVerdict the value type that carries this alongside the server's own string
 */
public enum ReviewDecision {

    /**
     * The reviewer accepted the matcher's suggestion. Requires {@code wasCorrect: true}.
     */
    APPROVED,

    /**
     * Nothing in the glossary governs this field. Requires {@code wasCorrect: false}.
     *
     * <p>Not "rank 1 was wrong": that is what {@code wasCorrect: false} with no verdict has always
     * said. This is the stronger claim that the glossary has no answer for the column.
     */
    REJECTED,

    /**
     * The reviewer chose a term that was <strong>not in the candidate list</strong>. Requires
     * {@code wasCorrect: false}.
     *
     * <p>The highest-signal record the trail can hold, because it says retrieval MISSED rather than
     * mis-ranked -- the one failure re-ranking cannot fix.
     */
    MANUAL_OVERRIDE,

    /**
     * A verdict a newer server sent that this client does not know. <strong>Never on the
     * wire.</strong>
     *
     * <p>Not an answer, and not sendable: {@link Feedback} refuses to carry it, because submitting
     * the literal string {@code UNKNOWN} to a server whose vocabulary does not contain it is a 422
     * that costs the reviewer their verdict. {@link ReviewVerdict#wireValue()} names what the
     * server actually sent.
     *
     * <p>Deliberately not published by the service, and
     * {@code tests/packaging/test_java_client_contract.py} asserts that it is not: the moment a
     * real server publishes {@code UNKNOWN}, this constant would stop meaning "unrecognised" and
     * start silently absorbing a real verdict.
     */
    UNKNOWN;

    /** Whether this client understood the server's value. False only for {@link #UNKNOWN}. */
    public boolean isKnown() {
        return this != UNKNOWN;
    }

    /**
     * The {@code wasCorrect} this verdict must accompany, empty for {@link #UNKNOWN}.
     *
     * <p>The server refuses the disagreement rather than reconciling it -- an {@code APPROVED}
     * beside {@code wasCorrect: false} would put a record in the audit trail that argues with
     * itself, and a trail that contradicts itself cannot be cited. This is that rule, readable
     * before the request is sent.
     */
    public Optional<Boolean> requiredWasCorrect() {
        return switch (this) {
            case APPROVED -> Optional.of(Boolean.TRUE);
            case REJECTED, MANUAL_OVERRIDE -> Optional.of(Boolean.FALSE);
            case UNKNOWN -> Optional.empty();
        };
    }
}
