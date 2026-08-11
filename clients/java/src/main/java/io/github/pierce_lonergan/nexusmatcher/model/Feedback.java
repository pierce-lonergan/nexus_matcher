package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.time.Instant;
import java.util.Objects;

/**
 * A reviewer's verdict on one match, for the append-only audit trail.
 *
 * <p><strong>Recorded only.</strong> This is not fed back into ranking, and that is a measured
 * decision rather than an unfinished feature: fine-tuning on exactly this signal was benchmarked
 * on the server's own corpus and LOST accuracy. Submitting feedback will not make the next match
 * better, and the server makes no such claim; what it buys is an audit trail of who decided what,
 * when, and against which governance id.
 *
 * <p>The wire key for the field path is {@code field}, not {@code fieldPath} -- the server aliases
 * it, and this record spells the wire name.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
    "field", "doc", "chosenGovernanceId", "suggestedGovernanceId", "wasCorrect", "reviewer", "ts"
})
public record Feedback(

        /** The field this verdict is about; the same path the match response was keyed by. */
        @JsonProperty("field") String field,

        /** The column comment, if the reviewer had one in front of them. */
        @JsonProperty("doc") String doc,

        /** The governance id the reviewer chose. Required. */
        @JsonProperty("chosenGovernanceId") String chosenGovernanceId,

        /** The governance id the matcher had suggested, when it differed. Optional. */
        @JsonProperty("suggestedGovernanceId") String suggestedGovernanceId,

        /** Whether the matcher's suggestion was right. */
        @JsonProperty("wasCorrect") boolean wasCorrect,

        /** Who decided. Required. */
        @JsonProperty("reviewer") String reviewer,

        /**
         * The client's own timestamp, stored verbatim.
         *
         * <p>NOT trusted for ordering: the server stamps its own {@code receivedAt} alongside it
         * and that is the field to sort by. A string rather than an {@link Instant} because the
         * server stores whatever it is given without parsing -- refusing a reviewer's verdict over
         * a clock format would lose the verdict, and the verdict is the thing worth keeping.
         */
        @JsonProperty("ts") String ts) {

    public Feedback {
        Objects.requireNonNull(field, "field");
        Objects.requireNonNull(chosenGovernanceId, "chosenGovernanceId");
        Objects.requireNonNull(reviewer, "reviewer");
        Objects.requireNonNull(ts, "ts");
    }

    /** A verdict stamped with an {@link Instant}, rendered ISO-8601. */
    public static Feedback of(
            String field,
            String chosenGovernanceId,
            boolean wasCorrect,
            String reviewer,
            Instant at) {
        return new Feedback(
                field, null, chosenGovernanceId, null, wasCorrect, reviewer,
                Objects.requireNonNull(at, "at").toString());
    }

    /** This verdict with the column comment attached. */
    public Feedback withDoc(String newDoc) {
        return new Feedback(
                field, newDoc, chosenGovernanceId, suggestedGovernanceId, wasCorrect, reviewer, ts);
    }

    /** This verdict recording what the matcher had suggested instead. */
    public Feedback withSuggestedGovernanceId(String suggested) {
        return new Feedback(
                field, doc, chosenGovernanceId, suggested, wasCorrect, reviewer, ts);
    }
}
