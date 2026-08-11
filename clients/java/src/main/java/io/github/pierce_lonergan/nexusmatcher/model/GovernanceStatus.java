package io.github.pierce_lonergan.nexusmatcher.model;

/**
 * Which of the three states a candidate's {@code governance} member is in.
 *
 * <p>This type exists because {@code null} on the wire means two different things and a Java
 * developer must not have to read {@code docs/GOVERNANCE.md} to find out which one they are
 * holding. It is a <em>reading</em> of the published contract, not a second opinion about
 * governance: the server has already decided, and this only labels which of its two documented
 * nulls arrived.
 *
 * <p>The contract it reads is the one published on {@code GovernanceView} in
 * {@code presentation/api/schemas.py} and restated in {@code docs/GOVERNANCE.md}: governance is
 * null when the matched entry carries no code, and separately when the candidate is rank 1 and its
 * decision is {@code REJECT}. A rejected <em>runner-up</em> keeps its class.
 */
public enum GovernanceStatus {

    /**
     * A class is present. {@link MatchCandidate#governance()} is non-null.
     */
    CONFERRED,

    /**
     * No class, because the matched entry carries no protection code: it sits at the vocabulary's
     * open tier, which {@link Vocabulary#openClassification()} on the same response names.
     *
     * <p>This is a real answer, not a gap. The field is governed -- as open.
     */
    OPEN_TIER,

    /**
     * No class, because this is the rank-1 candidate and the server rejected it: no entry in the
     * glossary describes this field, so there is nothing for it to inherit.
     *
     * <p>The field inherits <strong>nothing</strong>. Do not fall back to the runner-up's class,
     * and do not read this as the open tier -- it is "we do not know", which is the case that most
     * needs a human.
     *
     * <p>Honest limit: when a rank-1 candidate is rejected, the response cannot also tell you
     * whether that entry happened to carry no code either. The server clears the class there
     * regardless, and either way the candidate confers nothing, so the two are not distinguishable
     * and do not need to be.
     */
    WITHHELD_REJECTED_TOP_MATCH
}
