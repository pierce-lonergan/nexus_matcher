package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * The caps a client has to chunk against, so an adapter can READ them instead of guessing.
 *
 * <p>This is why {@link FieldSpec} deliberately mirrors none of them: they are per-deployment
 * numbers, a copy compiled into this artifact would refuse requests a tuned server accepts, and it
 * would go stale in silence. Ask the server once at startup and chunk against the answer.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ServiceLimits(

        /** Field cap for {@code POST /api/v1/match}. */
        @JsonProperty("maxFields") int maxFields,

        /** Field cap for {@code POST /api/v1/match/batch}. Also the ID cap for
         *  {@code POST /api/v1/lookup}. */
        @JsonProperty("maxBatchFields") int maxBatchFields,

        /** Raw request body cap in bytes, enforced before parsing -- so it is what a 413 is
         *  measured against, not the field count. */
        @JsonProperty("bodyByteCap") long bodyByteCap,

        /**
         * The server's own deadline before it answers 504.
         *
         * <p>Keep the client timeout ABOVE this. If the client gives up first it never sees the
         * 504, which is the hang the server-side deadline exists to prevent, reintroduced by an
         * off-by-one in seconds.
         */
        @JsonProperty("deadlineSeconds") double deadlineSeconds,

        /** Requests that may be admitted-and-unfinished before the server sheds with 503. The live
         *  in-flight count is deliberately not reported: it would make two identical requests
         *  produce different bytes. */
        @JsonProperty("capacity") int capacity) {

    @JsonCreator
    public ServiceLimits {
    }

    /** The field cap for the route {@code matchBatch} uses when {@code batch} is true. */
    public int fieldCap(boolean batch) {
        return batch ? maxBatchFields : maxFields;
    }
}
