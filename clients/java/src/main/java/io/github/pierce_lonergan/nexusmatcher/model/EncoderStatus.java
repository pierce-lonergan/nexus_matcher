package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Optional;
import java.util.OptionalInt;

/**
 * Which encoder retrieval is actually running on, and whether that is the intended one.
 *
 * <p>{@link #fallbackInForce()} is the field this whole surface exists for, and the reason is a real
 * incident rather than a hypothetical: the adopting pipeline lost an entire bulk run to a silent
 * encoder fallback. Matching ran, answered 200 on every field, and produced quietly worse results
 * for six hours. Nothing in a liveness probe can catch that, because nothing was down.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record EncoderStatus(

        /** The provider class actually in use. */
        @JsonProperty("provider") String provider,

        /** The model it reports, or null if it reports none. */
        @JsonProperty("modelName") String modelName,

        /** Embedding dimension, or null if not reported. */
        @JsonProperty("dimension") Integer dimension,

        /** Rung of the server's selection ladder: {@code bundled}, {@code transformer},
         *  {@code static}, or {@code custom} for a provider from outside the library. An open
         *  string, for the same reason {@link StatusWarning#code()} is one. */
        @JsonProperty("tier") String tier,

        /** Whether the ladder's first choice could have been used in this install at all. */
        @JsonProperty("bundledEncoderAvailable") boolean bundledEncoderAvailable,

        /**
         * True when encoder selection FELL THROUGH to a lower rung because the first choice was
         * unavailable here -- not when an operator deliberately wired a different provider.
         *
         * <p>Treat it as a reason to stop. Results will still come back, and they will be worse.
         */
        @JsonProperty("fallbackInForce") boolean fallbackInForce) {

    @JsonCreator
    public EncoderStatus {
    }

    /** {@link #modelName()} as an {@link Optional}. */
    public Optional<String> model() {
        return Optional.ofNullable(modelName);
    }

    /** {@link #dimension()} without the null. */
    public OptionalInt embeddingDimension() {
        return dimension == null ? OptionalInt.empty() : OptionalInt.of(dimension);
    }
}
