package io.github.pierce_lonergan.nexusmatcher.model;

import java.util.Objects;
import java.util.Optional;

/**
 * The whole governance answer for ONE column, in one object: which of the
 * {@link GovernanceOutcome outcomes} applies, and the value to write when one may be written.
 *
 * <p>Built by {@link MatchResponse#governanceFor(String)} by reading the rules the server
 * publishes, in the order the server states them. It decides nothing of its own: every branch below
 * is a documented server rule, and where the server withholds a class this hands back nothing too
 * -- it only says WHICH kind of nothing, and what to do about it.
 *
 * <h2>How to use it, and why it is shaped like this</h2>
 *
 * <p>Two accessors return a value, and each is empty on every outcome but its own. That is
 * deliberate: there is no single {@code classification()} that folds {@link #conferred()} and
 * {@link #openTier()} together, because a caller who had one would stop branching, and the branch
 * is the entire point.
 *
 * <pre>{@code
 * FieldGovernance governance = response.governanceFor(path);
 * switch (governance.outcome()) {
 *     case CONFERRED -> apply(governance.conferred().orElseThrow());
 *     case OPEN_TIER -> applyOpenTier(governance.openTier().orElseThrow());
 *     case WITHHELD_PENDING_REVIEW, WITHHELD_REJECTED_TOP_MATCH, WITHHELD_NO_MATCH, UNREADABLE ->
 *             sendToAHuman(response.candidatesFor(path));
 *     case UNCLASSIFIABLE_NO_VOCABULARY -> failTheRun(
 *             "the server has no controlled vocabulary loaded; nothing it returns is classified");
 * }
 * }</pre>
 *
 * <p>{@link GovernanceOutcome#maySafelyApply()} is the one-line form for callers that only need
 * "may I write something", and {@link GovernanceOutcome#needsAHuman()} for callers routing a review
 * queue. Neither collapses the outcomes; both are readings of them.
 *
 * <h2>What this replaces</h2>
 *
 * <p>The reading it makes cheap used to take three lookups and a paragraph of documentation to get
 * right: {@code verdictFor(path)}, then {@code inheritableGovernanceFor(path)}, then
 * {@code vocabulary().isConfigured()} before daring to treat an empty result as the open tier. Each
 * of those is still here and still correct. What changed is which reading is shorter -- the unsafe
 * one now takes more typing than the safe one, which is the only version of this that survives
 * contact with a caller in a hurry.
 */
public record FieldGovernance(

        /** Which of the seven outcomes this column is in. Never null. */
        GovernanceOutcome outcome,

        /** The class to apply, or null on every outcome but {@link GovernanceOutcome#CONFERRED}. */
        Governance conferredClass,

        /**
         * The deployment's own name for its open tier, or null on every outcome but
         * {@link GovernanceOutcome#OPEN_TIER}.
         *
         * <p>Never the {@code UNCLASSIFIED} sentinel: a response carrying the sentinel is
         * {@link GovernanceOutcome#UNCLASSIFIABLE_NO_VOCABULARY} and reaches this record with a
         * null here, so the sentinel cannot be applied as though it were a tier somebody declared.
         */
        String openTierName) {

    public FieldGovernance {
        Objects.requireNonNull(outcome, "outcome");
    }

    /** The class to apply. Present only on {@link GovernanceOutcome#CONFERRED}. */
    public Optional<Governance> conferred() {
        return Optional.ofNullable(conferredClass);
    }

    /**
     * The caller's own open-tier name to apply. Present only on
     * {@link GovernanceOutcome#OPEN_TIER}.
     */
    public Optional<String> openTier() {
        return Optional.ofNullable(openTierName);
    }

    /** {@link GovernanceOutcome#maySafelyApply()} for this column. */
    public boolean maySafelyApply() {
        return outcome.maySafelyApply();
    }

    /** {@link GovernanceOutcome#needsAHuman()} for this column. */
    public boolean needsAHuman() {
        return outcome.needsAHuman();
    }
}
