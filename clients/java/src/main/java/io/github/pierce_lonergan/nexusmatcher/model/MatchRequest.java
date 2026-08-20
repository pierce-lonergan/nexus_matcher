package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A batch of fields to match, plus the knobs the endpoint accepts.
 *
 * <p>A {@code threshold} is <em>not</em> one of them and never has been -- {@code /openapi.json}
 * does not publish it, and the server ignores unrecognised top-level keys rather than refusing
 * them, so sending one would be silently discarded rather than reported. Confidence is
 * rank-relative anyway; thresholding on it client-side is the mistake the server's own
 * documentation spends a section warning against.
 *
 * <p>Both request keys are snake_case while every response key is camelCase. That asymmetry is the
 * agreed wire contract, not an oversight, and this record spells both sides exactly --
 * {@code consistency_qualifier_segments} included.
 *
 * <h2>Every knob is nullable, and null means "take the server's default"</h2>
 *
 * <p>A null knob is OMITTED from the emitted JSON rather than sent as an explicit null. That is the
 * only way a client can decline to have an opinion: the server types these with defaults, and a
 * client that sent its own copy of every default would pin this artifact to the defaults of the
 * server it was written against. {@link #contrast()} and {@link #consistency()} are the two where
 * that matters most, because whether a deployment sends those blocks by default is the deployment's
 * decision and not this client's.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
    "fields", "top_k", "explain", "signals", "contrast", "consistency",
    "consistency_qualifier_segments"
})
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
        @JsonProperty("signals") Map<String, Object> signals,

        /**
         * Ask for {@link MatchResponse#contrast()}: rank 1 against rank 2 for every field.
         *
         * <p>Independent of {@link #explain()} -- the contrast carries the differences it needs, so
         * a caller can have the comparison without a weight breakdown on every candidate. Null
         * takes the server's default. Additive: asking for it appends a key to the response and
         * changes nothing already in it.
         */
        @JsonProperty("contrast") Boolean contrast,

        /**
         * Ask for {@link MatchResponse#consistency()}: which columns look like one concept, and
         * whether they were given one answer.
         *
         * <p>Reporting only -- nothing in {@code results} or {@code fieldDecisions} changes,
         * whatever it finds. Null takes the server's default, which is OFF.
         *
         * <p><strong>Read {@link ConsistencyReport} before you turn this on.</strong> The feature
         * is off by default because the grouping behind it was measured and the measurement came
         * back negative; at the server's default {@link #consistencyQualifierSegments()} it
         * reports nothing at all on the corpus it was measured against, and at the looser setting
         * it reported four groups of which four were collisions.
         */
        @JsonProperty("consistency") Boolean consistency,

        /**
         * How many of a column's nearest DECLARED path segments join its leaf in the concept key.
         * Null takes the server's default, which is 1.
         *
         * <p>1 means two columns are one concept only when they share a leaf AND the record they
         * hang off. 0 is the leaf alone -- the loosest key, which finds every repetition and also
         * merges distinct concepts that share a column name. Segments are boundaries you declared
         * (dots, or the {@code __} array boundary), never single underscores, and the key is built
         * from your {@code path} rather than from {@link FieldSpec#name()}.
         *
         * <p><strong>Neither direction is free and the dial has no good setting on a repeated-leaf
         * schema.</strong> Across the whole published policy space the best precision reached by
         * any policy that reports anything at all on that shape was 0.0235 -- see
         * {@link ConsistencyReport} for the numbers and how they were searched for. Measure it on
         * your own schemas before moving it.
         *
         * <p>The upper bound is derived by the server from its own {@code path} length limit and
         * is deliberately not mirrored here, exactly as the field and batch caps are not: a copy
         * compiled into this artifact would go stale the first time the bound moved. A value above
         * the deepest path in your request is inert rather than refused; a negative one is a 422.
         */
        @JsonProperty("consistency_qualifier_segments") Integer consistencyQualifierSegments) {

    public MatchRequest {
        Objects.requireNonNull(fields, "fields");
        if (fields.isEmpty()) {
            throw new IllegalArgumentException(
                    "a match request needs at least one field; the server refuses an empty list");
        }
        fields = List.copyOf(fields);
    }

    /** Fields only, taking the server's defaults for every knob. */
    public static MatchRequest of(List<FieldSpec> fields) {
        return new MatchRequest(fields, null, null, null, null, null, null);
    }

    /** Fields and a candidate count. */
    public static MatchRequest of(List<FieldSpec> fields, int topK) {
        return new MatchRequest(fields, topK, null, null, null, null, null);
    }

    /** This request asking for {@code topK} candidates per field. */
    public MatchRequest withTopK(int newTopK) {
        return new MatchRequest(
                fields, newTopK, explain, signals, contrast, consistency,
                consistencyQualifierSegments);
    }

    /** This request asking the server to explain each confidence. */
    public MatchRequest withExplain(boolean newExplain) {
        return new MatchRequest(
                fields, topK, newExplain, signals, contrast, consistency,
                consistencyQualifierSegments);
    }

    /** This request carrying request-level query signals. Null clears them. */
    public MatchRequest withSignals(Map<String, Object> newSignals) {
        return new MatchRequest(
                fields, topK, explain, newSignals, contrast, consistency,
                consistencyQualifierSegments);
    }

    /** This request asking for {@link MatchResponse#contrast()}. */
    public MatchRequest withContrast(boolean newContrast) {
        return new MatchRequest(
                fields, topK, explain, signals, newContrast, consistency,
                consistencyQualifierSegments);
    }

    /** This request asking for {@link MatchResponse#consistency()}. */
    public MatchRequest withConsistency(boolean newConsistency) {
        return new MatchRequest(
                fields, topK, explain, signals, contrast, newConsistency,
                consistencyQualifierSegments);
    }

    /**
     * This request grouping on the leaf plus {@code segments} of its declared path.
     *
     * <p>Does NOT turn {@link #consistency()} on: the knob and the block are separate on the wire,
     * and setting a grouping policy for a request that never asks for the report should not
     * silently start sending one.
     */
    public MatchRequest withConsistencyQualifierSegments(int segments) {
        return new MatchRequest(
                fields, topK, explain, signals, contrast, consistency, segments);
    }

    /** This request over a different field list, keeping every knob. Used when re-chunking a 413. */
    public MatchRequest withFields(List<FieldSpec> newFields) {
        return new MatchRequest(
                newFields, topK, explain, signals, contrast, consistency,
                consistencyQualifierSegments);
    }
}
