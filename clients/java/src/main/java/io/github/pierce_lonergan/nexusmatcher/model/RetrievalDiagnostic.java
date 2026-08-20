package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Why a field retrieved what it retrieved.
 *
 * <p><strong>Retrieval only.</strong> This is not the ranking {@code /api/v1/match} produces and
 * must not be read as one. The server runs the retrieval half of matching and stops before the
 * five-signal scoring pass and before the decision layer, so a candidate's position here is not its
 * final rank -- lexical, edit distance, type and domain signals still reorder the fused list, and
 * when {@link #rerankerWired()} is true a reranker replaces it outright, which is exactly what that
 * flag is for.
 *
 * <p>The two questions it answers that nothing else can: what the field's text BECAME before it
 * reached the encoder ({@link #queryText()}), and where the entry you expected actually landed
 * ({@link #expected()}).
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record RetrievalDiagnostic(

        /** The field as the server received it, echoed so the artifact is self-describing. Kept as
         *  an open map rather than a {@link FieldSpec}: it is a transcript of the request, and a
         *  transcript that refused an unexpected key would be a worse transcript. */
        @JsonProperty("field") Map<String, String> field,

        /**
         * What the field BECAME before retrieval -- parent-path context injected, abbreviations
         * expanded if that is enabled.
         *
         * <p>This string, not the field name, is what the encoder saw. It is usually the first place
         * a bad match explains itself.
         */
        @JsonProperty("queryText") String queryText,

        /** The model that encoded the query, or null when the server reports none. */
        @JsonProperty("encoderModel") String encoderModel,

        /** When true, matching replaces the fused order with a reranker's, so {@link #fused()}
         *  below is the INPUT to reranking rather than the order matching used. */
        @JsonProperty("rerankerWired") boolean rerankerWired,

        /** {@code dense}, {@code sparse} and {@code fused}, in that order. An open map because the
         *  channel set belongs to the server's wiring, not to this client. */
        @JsonProperty("channels") Map<String, RetrievalChannel> channels,

        /** Where the expected entry landed. Null unless the request named one. */
        @JsonProperty("expected") ExpectedPlacement expected) {

    /** The dense (embedding) channel's name. */
    public static final String DENSE = "dense";

    /** The sparse (lexical) channel's name. */
    public static final String SPARSE = "sparse";

    /** The fused channel's name: the weighted combination the scoring pass is handed. */
    public static final String FUSED = "fused";

    @JsonCreator
    public RetrievalDiagnostic {
        field = field == null ? Map.of() : Collections.unmodifiableMap(new LinkedHashMap<>(field));
        channels = channels == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(channels));
    }

    /** One channel by name, empty when this server did not report it. */
    public Optional<RetrievalChannel> channel(String name) {
        return Optional.ofNullable(channels.get(name));
    }

    /** The dense channel, empty when this server did not report one. */
    public Optional<RetrievalChannel> dense() {
        return channel(DENSE);
    }

    /** The sparse channel, empty when this server did not report one. */
    public Optional<RetrievalChannel> sparse() {
        return channel(SPARSE);
    }

    /** The fused channel, empty when this server did not report one. */
    public Optional<RetrievalChannel> fused() {
        return channel(FUSED);
    }

    /** {@link #expected()} as an {@link Optional}; empty when the request named no expected id. */
    public Optional<ExpectedPlacement> expectedValue() {
        return Optional.ofNullable(expected);
    }

    /**
     * The one-line diagnosis, when the request named an expected entry.
     *
     * <p>A rendering of what the response already says, for a log line or a ticket -- it computes
     * nothing the caller could not read off {@link #expected()} themselves, and it decides nothing.
     * Empty when no expected entry was named, because then there is no question to answer.
     */
    public Optional<String> diagnosis() {
        return expectedValue().map(placement -> {
            if (!placement.inDictionary()) {
                return placement.governanceId() + " is not in the loaded dictionary: this field "
                        + "cannot match it at any threshold. That is a glossary problem, not a "
                        + "scoring one.";
            }
            if (!placement.retrievedByAnyChannel()) {
                return placement.governanceId() + " is in the dictionary but no channel retrieved "
                        + "it, so scoring never saw it.";
            }
            return placement.governanceId() + " was retrieved at " + placement.rankByChannel()
                    + " (1-based, over each channel's full result).";
        });
    }
}
