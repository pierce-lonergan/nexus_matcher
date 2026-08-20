package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Optional;
import java.util.OptionalDouble;

/**
 * One candidate glossary entry for one schema field.
 *
 * <p>Ranks are 1-based and every rank is judged independently, so a {@link MatchDecision#REJECT}
 * at rank 3 says nothing about rank 1.
 *
 * <h2>Two questions this type answers, and they are not the same question</h2>
 *
 * <p><strong>"May this field inherit a protection class?"</strong> is a FIELD-level question and
 * the server owns the rule: read {@link MatchResponse#verdictFor(String)}, and
 * {@link MatchResponse#inheritableGovernanceFor(String)} applies it. Nothing on this type answers
 * it, deliberately -- a per-candidate second opinion about inheritance is how every client ends up
 * with a different rule.
 *
 * <p><strong>"Who decided this candidate?"</strong> is a CANDIDATE-level question and it is
 * answered here, by {@link #provenanceValue()} and its two named readings
 * {@link #decidedByAReviewer()} and {@link #wasScored()}. <strong>Do not answer it from
 * {@link #confidence()}.</strong> A retrieved candidate can legitimately reach 1.0 with a
 * {@link MatchDecision#AUTO_APPROVE} verdict -- there is a capture of one -- so the pair
 * {@code (confidence 1.0, AUTO_APPROVE)} identifies nothing at all. See {@link MatchProvenance}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MatchCandidate(

        /** 1-based position among this field's candidates. */
        @JsonProperty("rank") int rank,

        /** The matched entry's id, which IS the governance id. Always populated. */
        @JsonProperty("governanceId") String governanceId,

        /** The matched entry's business name. */
        @JsonProperty("businessName") String businessName,

        /** The matched entry's definition. */
        @JsonProperty("definition") String definition,

        /** The matched entry's domain. */
        @JsonProperty("domain") String domain,

        /**
         * The protection class this candidate confers, or {@code null} -- and {@code null} means
         * one of TWO different things.
         *
         * <p>Call {@link #governanceStatus()} rather than testing this for null. The two cases are
         * {@link GovernanceStatus#OPEN_TIER} (the matched entry carries no code, so it sits at the
         * vocabulary's open tier and there is genuinely nothing to confer) and
         * {@link GovernanceStatus#WITHHELD_REJECTED_TOP_MATCH} (rank 1 was rejected, so no entry in
         * the glossary describes this field and it inherits nothing at all). Collapsing them loses
         * the difference between "governed, openly" and "we do not know".
         *
         * <p>A REJECTED RUNNER-UP KEEPS ITS CLASS. Nothing inherits from a runner-up, and the class
         * is what lets a reviewer see that rank 1 is a direct identifier and rank 2 is not -- so
         * this is non-null on a rejected rank 2, and that is not a bug.
         */
        @JsonProperty("governance") Governance governance,

        /**
         * The server's confidence, rank-relative.
         *
         * <p>Do not threshold on it and do not diff against it. It is a min-max normalised fused
         * retrieval score with a structural floor, so a high number is not evidence of a good
         * match -- {@link #decision()} is what carries the verdict. The server's own documentation
         * makes this point twice.
         *
         * <p><strong>1.0 is not a sentinel and it does not mean a human decided this.</strong> The
         * five default scoring weights sum to exactly 1.0 and every signal is attainable at 1.0,
         * so ordinary retrieval reaches 1.0 whenever all five are maximal, and the captured
         * fixtures contain one that does. Read {@link #provenanceValue()} for where the answer
         * came from; this number cannot say.
         */
        @JsonProperty("confidence") double confidence,

        /**
         * The verdict for THIS CANDIDATE. Read it, not {@link #confidence()}.
         *
         * <p>For the one verdict per COLUMN -- the value that goes into a metadata sheet -- read
         * {@link MatchResponse#verdictFor(String)} instead. This one cannot express "nothing
         * matched"; that is what {@link FieldDecision#NO_MATCH} is for.
         */
        @JsonProperty("decision") MatchDecision decision,

        /**
         * The raw dense-retrieval score: the ONLY number on a candidate comparable ACROSS fields,
         * and the one that says whether the shortlist was any good.
         *
         * <p>{@link #confidence()} is min-max normalised within one field's shortlist, so its rank-1
         * floor is structural ({@link ScoringContract#confidenceFloor()}, 0.63 on the shipped
         * wiring) and the best of a hopeless set still scores above it. This number has no such
         * floor. Present on every candidate; it is not gated behind {@code explain}, and it is the
         * same number {@link Explain#absoluteCosine()} carries.
         *
         * <p><strong>{@code null} is not zero, and it now has TWO causes.</strong> Either the dense
         * retriever never returned this candidate -- it reached the shortlist through the lexical
         * arm alone -- or nothing scored it at all, because a reviewer decided it and matching was
         * skipped ({@link MatchProvenance#APPROVED_PAIR}). {@link #wasScored()} tells the two
         * apart. Both are "not measured"; neither is a measurement that came out at zero. Zero
         * would mean "measured, and orthogonal to the query", which is a claim this response is not
         * making, and on a cosine metric it is a claim about the far end of the range. That is why
         * this component is a boxed {@link Double} and
         * not a {@code double}: a primitive would let Jackson bind the absent number to 0.0 in
         * silence, and a caller filtering {@code absoluteScore() >= floor} would then read "we did
         * not measure it" as "it failed", quietly dropping candidates the lexical arm found. Prefer
         * {@link #absoluteScoreValue()}, which has no null to forget.
         *
         * <p>Read {@link ScoringContract#absoluteScoreMetric()} before treating it as a cosine, and
         * {@link ScoringContract#absoluteScorePooledOverAliases()} before treating it as a
         * similarity to the entry's own text.
         */
        @JsonProperty("absoluteScore") Double absoluteScore,

        /**
         * The deployment's own enrichment columns for the MATCHED ENTRY, carried through the
         * pipeline and never interpreted.
         *
         * <p>A fact about the ENTRY, not about the match: the identical object comes back from a
         * lookup of the same id, so a looked-up entry and a matched one can go down one code path.
         * Present on every candidate, with an empty {@link SourceMetadata#values()} when the entry
         * carries none -- so {@link #sourceMetadata()} itself is non-null on any current server.
         */
        @JsonProperty("sourceMetadata") SourceMetadata sourceMetadata,

        /**
         * Where this candidate came from, as the server spelled it. {@link #provenanceValue()} is
         * the typed reading and is what you switch on.
         *
         * <p>{@code RETRIEVAL} means the pipeline scored it; {@code APPROVED_PAIR} means a reviewer
         * decided it and matching was skipped. <strong>Read this, not {@link #confidence()}, to
         * tell the two apart</strong> -- a retrieved candidate legitimately reaches a confidence of
         * 1.0, so the number is not a discriminator. {@link MatchProvenance} gives the whole
         * argument.
         *
         * <p>Held as a {@link String} with {@link MatchProvenance} beside it, which is the same
         * shape {@link Contrast#separation()} and {@link ConceptGroup#agreement()} use and for the
         * same reason: the vocabulary is bound OPEN, so a value a newer server adds degrades to
         * {@link MatchProvenance#UNKNOWN} instead of costing the whole response, and the string the
         * server actually sent is still here to be logged.
         *
         * <p>The service sends it on every candidate and never sends null. It is {@code null} only
         * from a server predating the member, and that null must NOT be read as {@code RETRIEVAL}:
         * the bypass existed before this member did, so an older server's silence says nothing
         * about where its answer came from. {@link #provenanceValue()} reports
         * {@link MatchProvenance#UNKNOWN} there, and both named readings answer false.
         */
        @JsonProperty("provenance") String provenance,

        /**
         * Score components and weights, present only when the request asked to explain.
         *
         * <p>Also absent -- whatever the request asked for -- on a candidate a reviewer decided.
         * The block promises {@code sum(scores * weights) == confidence}, and a candidate nothing
         * measured has no components to keep that promise with; emitting five zeroes so the
         * arithmetic closed would publish measurements nobody took into fields the scoring
         * contract declares comparable ACROSS fields. Absent, not zero-filled, for the same reason
         * {@link #absoluteScore()} is null rather than 0.0.
         */
        @JsonProperty("explain") Explain explain) {

    @JsonCreator
    public MatchCandidate {
        // Defaulted rather than left null so a caller never has to null-check a member the
        // service always sends. A server predating the pass-through plane sends no key at all,
        // and an entry with no enrichment columns sends an empty one; both mean "no columns here"
        // to a consumer, and the empty object is the honest shape for both.
        sourceMetadata = sourceMetadata == null ? SourceMetadata.empty() : sourceMetadata;
    }

    /**
     * Which of the three governance states this candidate is in, so that the two meanings of a
     * null {@code governance} are visible in the type system rather than only in a document.
     *
     * <p>Derived from what the response already says -- {@code governance}, {@code rank} and
     * {@code decision} -- against the rule the server publishes. It decides nothing: when the class
     * is withheld this still hands back nothing, it just says which kind of nothing.
     */
    public GovernanceStatus governanceStatus() {
        if (governance != null) {
            return GovernanceStatus.CONFERRED;
        }
        if (rank == 1 && decision == MatchDecision.REJECT) {
            return GovernanceStatus.WITHHELD_REJECTED_TOP_MATCH;
        }
        return GovernanceStatus.OPEN_TIER;
    }

    /**
     * The class this candidate confers, empty when it confers none.
     *
     * <p>Empty is ambiguous by design -- it is both nulls at once -- so use
     * {@link #governanceStatus()} when the difference matters, which is whenever you are about to
     * apply a classification.
     */
    public Optional<Governance> governanceValue() {
        return Optional.ofNullable(governance);
    }

    /** The {@link #explain()} block, empty when the request did not ask for one. */
    public Optional<Explain> explainValue() {
        return Optional.ofNullable(explain);
    }

    /**
     * {@link #absoluteScore()} without the null.
     *
     * <p>Empty means the dense arm did not return this candidate -- NOT that it scored zero. There
     * is deliberately no {@code absoluteScoreOrZero()} here: the whole reason this accessor exists
     * is that the two are different facts, and a convenience that collapsed them would put the
     * defect back one method along.
     */
    public OptionalDouble absoluteScoreValue() {
        return absoluteScore == null
                ? OptionalDouble.empty()
                : OptionalDouble.of(absoluteScore);
    }

    /**
     * Whether this candidate has a dense score at all.
     *
     * <p>False is a fact about MEASUREMENT, not about quality, and it has two causes. On a
     * {@link MatchProvenance#RETRIEVAL} candidate it means the dense arm did not return this entry
     * -- it reached the shortlist through the lexical arm alone. On a
     * {@link MatchProvenance#APPROVED_PAIR} candidate it means nothing scored it, because a human
     * decided the field and retrieval never ran. {@link #wasScored()} separates them. Either way
     * the candidate cannot be compared against an absolute floor in either direction.
     */
    public boolean hasAbsoluteScore() {
        return absoluteScore != null;
    }

    /**
     * {@link #provenance()} as a value you can switch on.
     *
     * <p>{@link MatchProvenance#UNKNOWN} when this build does not recognise the server's word, and
     * when the server sent none at all. Never null.
     */
    public MatchProvenance provenanceValue() {
        return MatchProvenance.fromWire(provenance);
    }

    /**
     * Whether a HUMAN decided this candidate, rather than the pipeline scoring it.
     *
     * <p>This is the discrimination {@code provenance} exists for, given a name so that every
     * caller does not have to rediscover it -- and the reason it needs a name is that the obvious
     * test was wrong. A bypassed candidate carries {@code confidence} 1.0 and
     * {@link MatchDecision#AUTO_APPROVE}, that pair was once documented as identifying one, and
     * ordinary retrieval reaches it: the captured fixtures contain a scored candidate with exactly
     * those two values. Anything that branches on "did a person stand behind this answer" --
     * counting review coverage, deciding whether to re-review, labelling a lineage record --
     * belongs on this method and not on the numbers.
     *
     * <p><strong>It is not a licence to inherit.</strong> Whether the FIELD may take rank 1's
     * protection class is the server's rule, published as {@code fieldDecisions} and applied by
     * {@link MatchResponse#inheritableGovernanceFor(String)}; this client does not hold a second
     * opinion about it, and a reviewer's approval does not override a verdict the server sent.
     * What this answers is who decided, which is a different fact and the one no other member
     * carries.
     *
     * <p>False for {@link MatchProvenance#UNKNOWN}, so a value this build cannot read never
     * becomes a quiet "a human approved it".
     */
    public boolean decidedByAReviewer() {
        return provenanceValue() == MatchProvenance.APPROVED_PAIR;
    }

    /**
     * Whether this candidate's numbers are a measurement at all.
     *
     * <p>True only for {@link MatchProvenance#RETRIEVAL}. False means nothing scored this
     * candidate, so {@link #absoluteScore()} is absent and {@link #explain()} is absent BECAUSE
     * NOTHING MEASURED THEM -- not because a measurement came out at zero -- and
     * {@link #confidence()} is a human's decision rendered as a number rather than a score. Do not
     * compare it against a threshold, and do not compare it against another candidate's.
     *
     * <p>False for {@link MatchProvenance#UNKNOWN} too: a candidate whose provenance this build
     * cannot read is not one whose numbers it may treat as measurements.
     */
    public boolean wasScored() {
        return provenanceValue() == MatchProvenance.RETRIEVAL;
    }
}
