package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Set;

/**
 * One match response: the candidates for every field that was sent, plus what a null class means
 * on this deployment.
 *
 * <p><strong>Every field sent comes back, under the caller's own {@code path}, in the order sent.</strong>
 * A field nothing matched gets an empty list, never a missing key -- the server refuses to answer
 * at all rather than return a map that is short one entry. {@link #results()} is insertion-ordered
 * to preserve that, so iterating it walks the fields in request order.
 *
 * <p><strong>{@link #fieldDecisions()} is the answer, {@link #results()} is the evidence.</strong>
 * The per-column verdict a consumer writes into a metadata sheet is here, published rather than left
 * to each client to derive from rank 1. Reading rank 1's own {@code decision} instead is the mistake
 * this member exists to stop, and it is not a small one: a {@link FieldDecision#NO_MATCH} field
 * still comes back with a full candidate list whose rank 1 can carry a populated protection class
 * and a REVIEW verdict. {@link #inheritableGovernanceFor(String)} reads the two together.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record MatchResponse(

        /** Candidates per field, keyed by the caller's own path, in the order sent. */
        Map<String, List<MatchCandidate>> results,

        /** What a {@link GovernanceStatus#OPEN_TIER} null means on the server that answered. */
        Vocabulary vocabulary,

        /**
         * ONE verdict per field, keyed and ordered exactly like {@link #results()}.
         *
         * <p>The value that goes into a per-column decision. See {@link FieldDecision} for what each
         * verdict claims, and for why an unrecognised one from a newer server degrades here instead
         * of failing the whole response the way an unrecognised {@link MatchDecision} does.
         *
         * <p>Empty on a response from a server predating this member, which is why
         * {@link #verdictFor(String)} returns an {@link java.util.Optional} rather than inventing
         * one: this client will not manufacture a governance verdict the server did not send.
         */
        Map<String, FieldVerdict> fieldDecisions,

        /**
         * What the numbers in this response mean -- which of them may be compared across fields,
         * what metric produced {@link MatchCandidate#absoluteScore()}, and what floors are in force.
         *
         * <p>Null on a response from a server predating this block.
         */
        ScoringContract scoring,

        /**
         * Rank 1 against rank 2 for every field, or null when this response carries no contrast.
         *
         * <p>Null both on a request that did not ask for it -- {@link MatchRequest#withContrast(boolean)}
         * -- and on a server predating the block; the two are indistinguishable here and the
         * difference does not matter to a reader, because in both cases there is no contrast to
         * read. {@link #contrastValue()} folds the null.
         */
        ContrastReport contrast,

        /**
         * Which columns the server grouped as one concept and whether they agree, or null when this
         * response carries no consistency report.
         *
         * <p>Reporting only: nothing in {@link #results()} or {@link #fieldDecisions()} was changed
         * by it, and {@link ConsistencyReport#promotionApplied()} states that machine-readably.
         * <strong>Read {@link ConsistencyReport}'s measured limits before acting on a finding.</strong>
         */
        ConsistencyReport consistency,

        /** The {@code X-Request-ID} this exchange was answered under. Null only if the server
         *  omitted the header, which it does not. */
        String requestId,

        /** The server's own {@code X-Response-Time-Ms}, null when the header was absent. */
        Double responseTimeMs) {

    /**
     * Bind the wire body. Transport metadata is attached afterwards by
     * {@link #withTransport(String, Double)}, because it arrives in headers rather than in the
     * body and a DTO that could be populated from the body would be a DTO a body could lie about.
     */
    @JsonCreator
    public static MatchResponse fromBody(
            @JsonProperty("results") Map<String, List<MatchCandidate>> results,
            @JsonProperty("vocabulary") Vocabulary vocabulary,
            @JsonProperty("fieldDecisions") Map<String, FieldVerdict> fieldDecisions,
            @JsonProperty("scoring") ScoringContract scoring,
            @JsonProperty("contrast") ContrastReport contrast,
            @JsonProperty("consistency") ConsistencyReport consistency) {
        return new MatchResponse(
                results, vocabulary, fieldDecisions, scoring, contrast, consistency, null, null);
    }

    public MatchResponse {
        // LinkedHashMap, not Map.copyOf: Map.copyOf does not promise iteration order, and the
        // order IS the contract here -- it is the order the caller sent their fields in. The same
        // holds for fieldDecisions, which the server keys and orders identically.
        results = results == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(results));
        fieldDecisions = fieldDecisions == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(fieldDecisions));
    }

    /** This response with the correlation headers from the exchange that carried it. */
    public MatchResponse withTransport(String newRequestId, Double newResponseTimeMs) {
        return new MatchResponse(
                results, vocabulary, fieldDecisions, scoring, contrast, consistency,
                newRequestId, newResponseTimeMs);
    }

    /** The paths that came back, in the order they were sent. */
    public Set<String> paths() {
        return results.keySet();
    }

    /**
     * The candidates for one path, in rank order.
     *
     * <p>An empty list means the field was matched and nothing was found. A path that was never
     * sent also gives an empty list -- use {@link #paths()} if you need to tell those apart, though
     * the server guarantees every path you sent is present.
     */
    public List<MatchCandidate> candidatesFor(String path) {
        return results.getOrDefault(path, List.of());
    }

    /** The rank-1 candidate for one path, empty when nothing matched it. */
    public Optional<MatchCandidate> topCandidateFor(String path) {
        List<MatchCandidate> candidates = candidatesFor(path);
        return candidates.isEmpty() ? Optional.empty() : Optional.of(candidates.get(0));
    }

    /** {@link #responseTimeMs()} without the null. */
    public OptionalDouble responseTime() {
        return responseTimeMs == null ? OptionalDouble.empty() : OptionalDouble.of(responseTimeMs);
    }

    /**
     * The one verdict for one field, empty when the server sent none for it.
     *
     * <p>Empty on a server predating {@code fieldDecisions}, and on a path that was never sent.
     * Deliberately not defaulted from rank 1: deriving a per-column verdict is the rule the server
     * publishes this member to own, and a client that quietly reconstructed it would be back to
     * every client guessing differently -- which is the state this member was added to end.
     */
    public Optional<FieldVerdict> verdictFor(String path) {
        return Optional.ofNullable(fieldDecisions.get(path));
    }

    /**
     * The class this FIELD may inherit, empty when it may inherit none.
     *
     * <p>A reading of two published rules together, in the order the server states them, not a
     * third opinion of this client's own:
     *
     * <ol>
     *   <li>read {@code fieldDecisions[path]} first -- it is the field-level authority;
     *   <li>on anything but {@link FieldDecision#AUTO_APPROVE}, inherit nothing.
     * </ol>
     *
     * <p>The case worth spelling out is {@link FieldDecision#NO_MATCH}, because it is the one where
     * the naive reading looks right. The field comes back with candidates -- the server rejected the
     * empty-list design on purpose, so a reviewer can see what was considered -- and rank 1 can
     * carry a real protection class with a per-candidate verdict of REVIEW or even AUTO_APPROVE.
     * Inheriting it would classify a column from an entry the server has just said describes
     * nothing. This method returns empty there, and {@link #candidatesFor(String)} still hands back
     * every candidate for the human who now has to look.
     *
     * <p><strong>{@link #governanceFor(String)} is the accessor to reach for.</strong> It answers
     * this question and the two that have to be asked alongside it, in one call and without an
     * overloaded empty. This method is kept because callers already read it, and because it is the
     * narrower question: "may the field take rank 1's class", with no opinion about what to do when
     * it may not.
     *
     * <p><strong>Empty is overloaded here, so branch on the verdict first and use this inside the
     * branch.</strong> It is what you get for every verdict that is not {@code AUTO_APPROVE}, AND
     * for an {@code AUTO_APPROVE} field whose rank-1 entry carries no protection code. Those need
     * different handling and this method cannot tell you which you have:
     *
     * <pre>{@code
     * switch (response.verdictFor(path).map(FieldVerdict::decision).orElse(FieldDecision.UNKNOWN)) {
     *     case AUTO_APPROVE -> response.inheritableGovernanceFor(path).ifPresentOrElse(
     *             this::apply,
     *             // Empty HERE is the open tier -- but ONLY once the vocabulary block has been
     *             // checked. A deployment that loaded no vocabulary returns this exact shape for
     *             // every field, and applying its UNCLASSIFIED sentinel as a tier grants the most
     *             // permissive reading on the strength of a configuration nobody completed.
     *             () -> {
     *                 if (!response.vocabulary().isConfigured()) {
     *                     throw new IllegalStateException("server has no vocabulary loaded");
     *                 }
     *                 applyOpenTier(response.vocabulary().openClassification());
     *             });
     *     case REVIEW, REJECT, NO_MATCH, UNKNOWN -> sendToAHuman(response.candidatesFor(path));
     * }
     * }</pre>
     *
     * <p>All of which is what {@link #governanceFor(String)} does for you.
     *
     * <p>Outside an {@code AUTO_APPROVE} branch, empty means "do not apply a class from this
     * response" and must never be read as "apply the open tier". See
     * {@link MatchCandidate#governanceStatus()} when the difference between a candidate's two
     * nulls is what you need.
     */
    public Optional<Governance> inheritableGovernanceFor(String path) {
        Optional<FieldVerdict> verdict = verdictFor(path);
        if (verdict.isEmpty() || !verdict.get().maySafelyInherit()) {
            return Optional.empty();
        }
        return topCandidateFor(path).flatMap(MatchCandidate::governanceValue);
    }

    /**
     * THE governance answer for one column: what may be applied to it, or which kind of nothing.
     *
     * <p>The reading a consumer mapping {@code governance.code} onto read permissions needs, made
     * shorter than the wrong reading. Three published rules are applied here, in the order the
     * server states them, and none of them is this client's own opinion:
     *
     * <ol>
     *   <li>{@code fieldDecisions[path]} is the field-level authority. Anything but
     *       {@code AUTO_APPROVE} confers nothing, and the constant says which "nothing" it is --
     *       {@link FieldDecision#NO_MATCH} in particular still arrives with a full candidate list
     *       whose rank 1 can carry a real class.
     *   <li>On {@code AUTO_APPROVE}, rank 1's own {@code governance} is the class, when it has one.
     *   <li>When it has none, that is the open tier -- <strong>but only if this deployment has a
     *       vocabulary at all.</strong> {@link Vocabulary#isConfigured()} is what separates "the
     *       matched entry carries no code" from "nothing on this server was ever classified", and
     *       those two responses are otherwise identical, field for field and null for null.
     * </ol>
     *
     * <p>Step 3 is the one that earns this method. It is a check every caller has to make, on a
     * member on the far side of the response from the field they are reading, to avoid granting the
     * most permissive classification to a server that classified nothing. Left to each caller, it
     * is the check that gets skipped.
     *
     * <p><strong>The vocabulary is consulted after the verdict, not before.</strong> A
     * {@code REVIEW} field on an unconfigured deployment reports
     * {@link GovernanceOutcome#WITHHELD_PENDING_REVIEW} rather than
     * {@link GovernanceOutcome#UNCLASSIFIABLE_NO_VOCABULARY}: the verdict is a fact about that
     * column and is worth keeping, and the deployment-wide question has a deployment-wide answer in
     * {@code vocabulary().isConfigured()}, which is one call per response rather than one per
     * field. The unconfigured outcome is reported exactly where it changes what a caller would do
     * -- where the answer would otherwise have been "apply the open tier".
     *
     * <p>A path that was never sent reports {@link GovernanceOutcome#UNREADABLE}, the same as a
     * verdict this build cannot read: in both cases this response says nothing about that column.
     */
    public FieldGovernance governanceFor(String path) {
        Optional<FieldVerdict> verdict = verdictFor(path);
        if (verdict.isEmpty() || !verdict.get().isKnown()) {
            return new FieldGovernance(GovernanceOutcome.UNREADABLE, null, null);
        }

        switch (verdict.get().decision()) {
            case REVIEW:
                return new FieldGovernance(GovernanceOutcome.WITHHELD_PENDING_REVIEW, null, null);
            case REJECT:
                return new FieldGovernance(
                        GovernanceOutcome.WITHHELD_REJECTED_TOP_MATCH, null, null);
            case NO_MATCH:
                return new FieldGovernance(GovernanceOutcome.WITHHELD_NO_MATCH, null, null);
            case AUTO_APPROVE:
                break;
            default:
                // UNKNOWN is filtered by isKnown() above; anything else is a constant added to
                // FieldDecision without a branch here, which must not fall through to "apply".
                return new FieldGovernance(GovernanceOutcome.UNREADABLE, null, null);
        }

        Optional<MatchCandidate> top = topCandidateFor(path);
        if (top.isEmpty()) {
            // AUTO_APPROVE with nothing to approve. The server does not send this -- a field with
            // no candidates is NO_MATCH -- so it is a response contradicting itself, and the safe
            // reading of a self-contradicting response is that it cannot be read.
            return new FieldGovernance(GovernanceOutcome.UNREADABLE, null, null);
        }

        Governance conferred = top.get().governance();
        if (conferred != null) {
            // A class that ARRIVED is a class, whatever the vocabulary block says: the server
            // resolved it, and nothing here needs to know the tier ordering to pass it on.
            return new FieldGovernance(GovernanceOutcome.CONFERRED, conferred, null);
        }

        if (vocabulary == null || !vocabulary.isConfigured()) {
            return new FieldGovernance(
                    GovernanceOutcome.UNCLASSIFIABLE_NO_VOCABULARY, null, null);
        }
        return new FieldGovernance(
                GovernanceOutcome.OPEN_TIER, null, vocabulary.openClassification());
    }

    /**
     * The paths whose verdict this client did not recognise, in the order sent.
     *
     * <p>Empty on every server this build knows about. Non-empty means a newer server has added a
     * verdict, those fields need a human, and this artifact needs an upgrade --
     * {@link FieldVerdict#wireValue()} on each names the value to look up.
     */
    public List<String> pathsWithUnknownVerdicts() {
        return fieldDecisions.entrySet().stream()
                .filter(entry -> !entry.getValue().isKnown())
                .map(Map.Entry::getKey)
                .toList();
    }

    /** {@link #scoring()} as an {@link Optional}; empty on a server predating the block. */
    public Optional<ScoringContract> scoringValue() {
        return Optional.ofNullable(scoring);
    }

    /** {@link #contrast()} as an {@link Optional}; empty when this response carries no contrast. */
    public Optional<ContrastReport> contrastValue() {
        return Optional.ofNullable(contrast);
    }

    /**
     * {@link #consistency()} as an {@link Optional}; empty when this response carries no report.
     *
     * <p>Empty is not "everything is consistent". It means nothing was checked.
     */
    public Optional<ConsistencyReport> consistencyValue() {
        return Optional.ofNullable(consistency);
    }

    /**
     * The contrast for one field, empty when there is no contrast block or no runner-up.
     *
     * <p>A convenience over {@code contrastValue().flatMap(r -> r.contrastFor(path))}, and the same
     * three meanings collapse into empty: no block was asked for, the field had one candidate, or
     * the path was never sent. Use {@link #contrastValue()} when you need to tell them apart.
     */
    public Optional<Contrast> contrastFor(String path) {
        return contrastValue().flatMap(report -> report.contrastFor(path));
    }
}
