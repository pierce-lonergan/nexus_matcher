package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Everything needed to recompute a candidate's {@link MatchCandidate#confidence()} from the
 * response alone. Present only when the request asked for it.
 *
 * <p>{@code scores} and {@code weights} are open maps rather than named fields, and that is the
 * server's decision, not a shortcut here: it publishes them open so that a sixth weighted signal
 * does not make the schema false. They carry the same keys, and the server verifies
 * {@code sum(scores[k] * weights[k])} against the confidence it emits before sending -- so
 * {@link #recomputedConfidence()} is a check you can run, not a number you have to trust.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Explain(

        /** One entry per weighted signal. {@code fusedRetrieval} is normalised within this field's
         *  shortlist, so 0.9 means "ranked first here", not "90% similar". */
        @JsonProperty("scores") Map<String, Double> scores,

        /** The weights of the live matcher that produced these confidences, not the shipped
         *  defaults. */
        @JsonProperty("weights") Map<String, Double> weights,

        /**
         * Dense cosine similarity: the only number here comparable ACROSS fields. Null when the
         * dense retriever did not return this candidate at all.
         */
        @JsonProperty("absoluteCosine") Double absoluteCosine) {

    @JsonCreator
    public Explain {
        scores = scores == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(scores));
        weights = weights == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(weights));
    }

    /** {@link #absoluteCosine()} without the null. Empty means the dense arm did not return it. */
    public OptionalDouble absoluteCosineValue() {
        return absoluteCosine == null ? OptionalDouble.empty() : OptionalDouble.of(absoluteCosine);
    }

    /**
     * The weighted sum of the emitted components, clamped to [0, 1] exactly as the server clamps.
     *
     * <p>Arithmetic on numbers the response already carries -- not a re-scoring, and not a second
     * opinion about the match. Comparing it against {@link MatchCandidate#confidence()} is how an
     * auditor checks that the explanation explains the number it arrived with.
     */
    public double recomputedConfidence() {
        double total = 0.0;
        for (Map.Entry<String, Double> weight : weights.entrySet()) {
            Double score = scores.get(weight.getKey());
            if (score != null) {
                total += score * weight.getValue();
            }
        }
        return Math.min(Math.max(total, 0.0), 1.0);
    }

    /** One component's score, if the server emitted it under that key. */
    public Optional<Double> score(String component) {
        return Optional.ofNullable(scores.get(component));
    }
}
