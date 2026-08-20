package io.github.pierce_lonergan.nexusmatcher.model;

/**
 * What a FIELD may be classified as, once every rule the server published has been read in the
 * order the server states them.
 *
 * <p>{@link GovernanceStatus} answers the same shape of question about ONE CANDIDATE. This answers
 * it about the COLUMN, which is the question a consumer mapping {@code governance.code} onto read
 * permissions is actually asking, and the two are not the same: a candidate can carry a perfectly
 * real protection class on a field the server has just said inherits nothing.
 *
 * <h2>Why this exists when {@link MatchResponse#inheritableGovernanceFor(String)} already did</h2>
 *
 * <p>That method returns an {@link java.util.Optional}, and its own documentation says the empty
 * case is overloaded. A caller who does not read that paragraph writes
 * {@code inheritableGovernanceFor(path).orElseGet(this::openTier)} -- and has just granted the open
 * tier's read access to a rejected match, to a {@code NO_MATCH}, and to a deployment that never
 * loaded a vocabulary. The safe reading was available and the unsafe one was shorter. This type is
 * the same correction {@link FieldVerdict} made for {@code fieldDecisions} and
 * {@link MatchProvenance} made for "who decided this": the discrimination is too important to be
 * rediscovered by every caller, so it is given a name and the name is the shortest thing to reach
 * for.
 *
 * <p><strong>The two values that permit applying something are {@link #CONFERRED} and
 * {@link #OPEN_TIER}, and {@link #maySafelyApply()} is the single test for it.</strong> Everything
 * else is a "no", and the constants are distinct because the four nos need four different
 * follow-ups -- a human, a human, a human, and an operator.
 *
 * <h2>The one it is easiest to get wrong</h2>
 *
 * <p>{@link #UNCLASSIFIABLE_NO_VOCABULARY} is the reason this enum is worth its weight. A
 * deployment that loaded no controlled vocabulary answers matches perfectly well: every candidate
 * comes back, every verdict is ordinary, and every {@code governance} is null. Candidate for
 * candidate that response is IDENTICAL to one where the matched entry simply carries no protection
 * code -- the genuine open tier. The only thing that separates them is
 * {@link Vocabulary#openClassification()}, which is the library's {@code UNCLASSIFIED} sentinel in
 * the first case and the caller's own tier name in the second.
 *
 * <p>So a client that reads a null class as "the open tier" without checking the vocabulary block
 * grants the most permissive reading on the strength of a configuration nobody completed, and
 * nothing anywhere reports it. Failing open is silent: no one files a ticket about data they were
 * allowed to see.
 */
public enum GovernanceOutcome {

    /**
     * The field may take the protection class on {@link FieldGovernance#conferred()}.
     *
     * <p>The server's own verdict for this column is {@code AUTO_APPROVE} and its rank-1 candidate
     * carries a class.
     */
    CONFERRED,

    /**
     * The field is governed, AS OPEN. Apply the tier {@link FieldGovernance#openTier()} names.
     *
     * <p>A real answer and not a gap: the matched entry carries no protection code, which on the
     * caller's own vocabulary means the open tier. Sending this to a reviewer spends a human hour
     * on a column that is published by policy.
     *
     * <p>Only reachable on a deployment whose vocabulary IS configured -- see
     * {@link #UNCLASSIFIABLE_NO_VOCABULARY}, which is the same wire shape and a different answer.
     */
    OPEN_TIER,

    /**
     * A human must decide this column. The server's verdict is {@code REVIEW}.
     *
     * <p>Never read as "probably fine". The rank-1 candidate may carry a class and may carry a high
     * confidence; neither is permission. {@link MatchResponse#candidatesFor(String)} is the
     * evidence to put in front of the person who now has to look.
     */
    WITHHELD_PENDING_REVIEW,

    /**
     * Nothing in the glossary describes this column: the rank-1 candidate was REJECTED.
     *
     * <p>The class is withheld even when the matched entry carries one, so this is not "the entry
     * had no code" -- it is "we do not know", which is the case that most needs a human. Named the
     * same as {@link GovernanceStatus#WITHHELD_REJECTED_TOP_MATCH} because it is the same fact,
     * read at the field level rather than at the candidate level.
     */
    WITHHELD_REJECTED_TOP_MATCH,

    /**
     * The response carries nothing this column may inherit from: the verdict is
     * {@link FieldDecision#NO_MATCH}.
     *
     * <p><strong>The candidates are still there, and rank 1 can carry a real class.</strong> That
     * is the trap this outcome exists to close. Earned either by a field that came back with no
     * candidates at all, or by a deployment that configured
     * {@link ScoringContract#absoluteScoreFloor()} and a rank 1 that does not clear it.
     */
    WITHHELD_NO_MATCH,

    /**
     * Nothing could be classified: the server has no controlled vocabulary loaded.
     *
     * <p>A statement about the DEPLOYMENT, not about the column. Every field on such a server comes
     * back with a null class, and reading those nulls as the open tier is the fail-open this enum
     * was added for. The follow-up is an operator, not a reviewer: the glossary needs its
     * {@code protection_classes} file wired before any answer here means anything.
     *
     * <p>Also reported for a server old enough not to send the {@code vocabulary} block at all. It
     * is a different cause and the same position: this client cannot say what a null class means
     * on that deployment, so it will not guess.
     */
    UNCLASSIFIABLE_NO_VOCABULARY,

    /**
     * This client could not read the server's answer for this column. <strong>Never a
     * verdict.</strong>
     *
     * <p>Three causes, all of which mean the same thing to a caller -- apply nothing, escalate:
     * a newer server sent a verdict this build does not know
     * ({@link FieldVerdict#wireValue()} names it); the response carried no verdict for this path,
     * either because the path was never sent or because the server predates
     * {@code fieldDecisions}; or the verdict says {@code AUTO_APPROVE} and there is no rank-1
     * candidate to take a class from, which is a response contradicting itself.
     */
    UNREADABLE;

    /**
     * Whether a classification may be written into a per-column decision from this outcome.
     *
     * <p>True for {@link #CONFERRED} and {@link #OPEN_TIER} alone. Every other constant is a "no",
     * including {@link #UNREADABLE} -- a value this build cannot read never becomes a quiet
     * approval -- and including {@link #UNCLASSIFIABLE_NO_VOCABULARY}, which is a "no" about the
     * server rather than about the column.
     */
    public boolean maySafelyApply() {
        return this == CONFERRED || this == OPEN_TIER;
    }

    /**
     * Whether the follow-up is a person looking at this column.
     *
     * <p>True for the three withheld verdicts and for {@link #UNREADABLE}. False for
     * {@link #UNCLASSIFIABLE_NO_VOCABULARY}, deliberately: that one is fixed by an operator wiring
     * a vocabulary, and routing every column on a misconfigured server to a review queue buries
     * one configuration defect under a schema's worth of tickets.
     */
    public boolean needsAHuman() {
        return this == WITHHELD_PENDING_REVIEW
                || this == WITHHELD_REJECTED_TOP_MATCH
                || this == WITHHELD_NO_MATCH
                || this == UNREADABLE;
    }
}
