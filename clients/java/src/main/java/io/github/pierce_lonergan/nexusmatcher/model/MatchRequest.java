package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A batch of fields to match, plus the two knobs the endpoint accepts.
 *
 * <p>There are exactly two: {@code top_k} and {@code explain}. A {@code threshold} is <em>not</em>
 * part of this contract -- {@code /openapi.json} publishes {@code fields}, {@code top_k} and
 * {@code explain} and nothing else, and the server now ignores unrecognised top-level keys rather
 * than refusing them, so sending one would be silently discarded rather than reported. Confidence
 * is rank-relative anyway; thresholding on it client-side is the mistake the server's own
 * documentation spends a section warning against.
 *
 * <p>Both request keys are snake_case while every response key is camelCase. That asymmetry is the
 * agreed wire contract, not an oversight, and this record spells both sides exactly.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"fields", "top_k", "explain"})
public record MatchRequest(

        /** The fields to match. At least one. */
        @JsonProperty("fields") List<FieldSpec> fields,

        /** Candidates per field. Null sends nothing and takes the server's default (5). A value
         *  above the server's configured {@code results_per_field} is a 422 naming the cap. */
        @JsonProperty("top_k") Integer topK,

        /** Ask for the score components and weights behind each confidence. Null takes the
         *  server's default (false). */
        @JsonProperty("explain") Boolean explain,

        /** Request-level query signals, same vocabulary as {@link FieldSpec#signals()}.
         *  A field-level signal of the same key wins; the two merge KEY BY KEY, so a
         *  request-level overlay and a field-level entity coexist. Null sends nothing. */
        @JsonProperty("signals") Map<String, Object> signals) {

    public MatchRequest {
        Objects.requireNonNull(fields, "fields");
        if (fields.isEmpty()) {
            throw new IllegalArgumentException(
                    "a match request needs at least one field; the server refuses an empty list");
        }
        fields = List.copyOf(fields);
    }

    /** Fields only, taking the server's defaults for {@code top_k} and {@code explain}. */
    public static MatchRequest of(List<FieldSpec> fields) {
        return new MatchRequest(fields, null, null, null);
    }

    /** Fields and a candidate count. */
    public static MatchRequest of(List<FieldSpec> fields, int topK) {
        return new MatchRequest(fields, topK, null, null);
    }

    /** This request asking for {@code topK} candidates per field. */
    public MatchRequest withTopK(int newTopK) {
        return new MatchRequest(fields, newTopK, explain, null);
    }

    /** This request asking the server to explain each confidence. */
    public MatchRequest withExplain(boolean newExplain) {
        return new MatchRequest(fields, topK, newExplain, null);
    }

    /** This request over a different field list, keeping the knobs. Used when re-chunking a 413. */
    public MatchRequest withFields(List<FieldSpec> newFields) {
        return new MatchRequest(newFields, topK, explain, null);
    }
}
