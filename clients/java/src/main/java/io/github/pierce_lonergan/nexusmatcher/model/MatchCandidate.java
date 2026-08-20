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
         * <p><strong>{@code null} is not zero.</strong> Null means the dense retriever never
         * returned this candidate at all -- it reached the shortlist through the lexical arm alone
         * -- so there is no dense score to report. Zero would mean "measured, and orthogonal to the
         * query", which is a claim this response is not making, and on a cosine metric it is a claim
         * about the far end of the range. That is why this component is a boxed {@link Double} and
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

        /** Score components and weights, present only when the request asked to explain. */
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
     * Whether the dense retriever returned this candidate at all.
     *
     * <p>False is a fact about RETRIEVAL, not about quality: the candidate reached the shortlist
     * through the lexical arm alone, so it has no dense score and cannot be compared against an
     * absolute floor in either direction.
     */
    public boolean hasAbsoluteScore() {
        return absoluteScore != null;
    }
}
