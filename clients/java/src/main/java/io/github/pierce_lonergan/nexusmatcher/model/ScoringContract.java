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
import java.util.OptionalInt;

/**
 * What each number in this response MEANS, shipped in the response that carries them.
 *
 * <p>The same idea as {@link Vocabulary} one type over, and deliberately the same shape: that block
 * ships the tier ordering so a client need not hard-code somebody's taxonomy, and this one ships the
 * scale contract so a client need not read the server's source to learn whether a number may be
 * compared against a constant.
 *
 * <p><strong>It settles a contradiction, and the contradiction is the reason to read it.</strong>
 * The service documents {@link MatchCandidate#confidence()} as rank-relative and says not to
 * threshold on it, and then ships an auto-approve threshold of 0.87, which is a threshold on it.
 * The resolution, in machine-readable form: {@code confidence} is comparable WITHIN one field, and
 * {@link MatchCandidate#absoluteScore()} is the number comparable ACROSS fields.
 *
 * <p>Nothing here is calibrated. None of these numbers behaves like a probability that a match is
 * correct, and this block says so by declaring no number {@code ACROSS_RUNS}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ScoringContract(

        /**
         * The lowest {@code confidence} a rank-1 candidate can structurally carry on this server,
         * or null when the derivation does not hold or did not check out for this response.
         *
         * <p><strong>Never set a confidence threshold at or below it.</strong> It selects nothing
         * however bad the matches are -- a filter that reports "nothing to review" on a schema where
         * nothing is trustworthy. The service shipped exactly that defect once.
         */
        @JsonProperty("confidenceFloor") Double confidenceFloor,

        /**
         * The {@link MatchCandidate#absoluteScore()} beneath which this server reports a field as
         * {@link FieldDecision#NO_MATCH}, or null when no floor is configured -- which is the
         * default, because the service ships none and will not invent one.
         *
         * <p>While it is null, {@link FieldDecision#NO_MATCH} can only mean "the field came back
         * with no candidates at all".
         */
        @JsonProperty("absoluteScoreFloor") Double absoluteScoreFloor,

        /**
         * The distance metric the server's vector store declares, so a client never has to ASSUME
         * {@link MatchCandidate#absoluteScore()} is a cosine.
         *
         * <p>Open string, from the store. {@code cosine} under the shipped wiring; a deployment
         * supplying its own store may report {@code dot} or {@code euclidean}, in which case the
         * number is monotone in similarity but is neither bounded nor a cosine. {@code unknown}
         * means the store declares nothing, which is NOT a synonym for cosine.
         */
        @JsonProperty("absoluteScoreMetric") String absoluteScoreMetric,

        /**
         * True when the server indexes fabricated technical spellings of each dictionary entry, in
         * which case {@link MatchCandidate#absoluteScore()} is the best score over an entry's
         * spellings rather than the similarity to the entry's own text.
         *
         * <p>A floor measured on a deployment where this was the other value does not transfer.
         */
        @JsonProperty("absoluteScorePooledOverAliases") boolean absoluteScorePooledOverAliases,

        /**
         * The response numbers a client may legitimately compare against a CONSTANT across
         * different fields. Everything absent is comparable only within one field, or not at all.
         */
        @JsonProperty("thresholdableAcrossFields") List<String> thresholdableAcrossFields,

        /**
         * The scale vocabulary, narrowest first, so a client can rank two scopes without hard-coding
         * the order. A wider scope implies every narrower one.
         *
         * <p>Open strings on purpose, exactly like {@link Vocabulary#tiersMostOpenFirst()}: the
         * ordering rides on the response so that nothing here has to know the words.
         */
        @JsonProperty("comparabilityScopesNarrowestFirst")
        List<String> comparabilityScopesNarrowestFirst,

        /**
         * One entry per numeric field this response can carry, keyed by its path in the body
         * ({@code confidence}, {@code absoluteScore}, {@code explain.scores.lexical}, ...), naming
         * the WIDEST scope over which two of its values may be compared.
         *
         * <p>A null value means this server declares no scope for that number, and an undeclared
         * number must not be compared with anything.
         */
        @JsonProperty("comparability") Map<String, String> comparability) {

    @JsonCreator
    public ScoringContract {
        thresholdableAcrossFields = thresholdableAcrossFields == null
                ? List.of()
                : List.copyOf(thresholdableAcrossFields);
        comparabilityScopesNarrowestFirst = comparabilityScopesNarrowestFirst == null
                ? List.of()
                : List.copyOf(comparabilityScopesNarrowestFirst);
        // LinkedHashMap rather than Map.copyOf: the entries arrive in the order the server lists
        // them and a caller rendering this into a report should see that order. Also Map.copyOf
        // refuses null VALUES, and null is a documented value here -- "this server declares no
        // scope for that number" -- so copying through it would turn a legal body into a crash.
        comparability = comparability == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(comparability));
    }

    /** {@link #confidenceFloor()} without the null. Empty means the server did not publish one. */
    public OptionalDouble confidenceFloorValue() {
        return confidenceFloor == null ? OptionalDouble.empty() : OptionalDouble.of(confidenceFloor);
    }

    /** {@link #absoluteScoreFloor()} without the null. Empty means no floor is configured. */
    public OptionalDouble absoluteScoreFloorValue() {
        return absoluteScoreFloor == null
                ? OptionalDouble.empty()
                : OptionalDouble.of(absoluteScoreFloor);
    }

    /**
     * Whether two values of this response number, taken from DIFFERENT fields, may be compared --
     * and therefore whether a fixed cut point across a schema means anything.
     *
     * <p>A lookup in the list the server published, nothing more. {@code "absoluteScore"} is true
     * under the shipped wiring and {@code "confidence"} is false, which is the one question most
     * callers have.
     */
    public boolean comparableAcrossFields(String responseNumber) {
        return thresholdableAcrossFields.contains(responseNumber);
    }

    /** The declared scope for one response number, empty when this server declares none. */
    public Optional<String> scopeOf(String responseNumber) {
        return Optional.ofNullable(comparability.get(responseNumber));
    }

    /**
     * Where a scope sits in the declared ordering, 0 being the narrowest.
     *
     * <p>Empty when this server did not declare the scope. Like {@link Vocabulary#openness(String)},
     * this reads the ordering the response carried and invents none: comparing two scopes the
     * server did not name is not something this client is entitled to do.
     */
    public OptionalInt scopeWidth(String scope) {
        int index = comparabilityScopesNarrowestFirst.indexOf(scope);
        return index < 0 ? OptionalInt.empty() : OptionalInt.of(index);
    }

    /**
     * Whether {@link MatchCandidate#absoluteScore()} really is a cosine similarity on this server.
     *
     * <p>Check it before comparing an absolute score against a floor that was measured somewhere
     * else. A floor chosen for a cosine is meaningless under {@code dot} or {@code euclidean}, and
     * {@code unknown} means the store said nothing rather than that it said cosine.
     */
    public boolean absoluteScoreIsCosine() {
        return "cosine".equals(absoluteScoreMetric);
    }
}
