package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.time.Instant;
import java.util.Objects;
import java.util.Optional;

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
 *
 * <h2>{@link #verdict()} is optional, and the one worth filling in</h2>
 *
 * <p>{@link #wasCorrect()} is required and is not deprecated. {@link #verdict()} adds the third
 * state a boolean cannot hold -- see {@link ReviewDecision} -- and it is the member that makes
 * "retrieval never found it" countable. Omit it and the server records an explicit null, which is
 * exactly the shape every record written before the member existed already has.
 *
 * <p>The two must agree, and <strong>this record refuses the disagreement locally</strong> rather
 * than letting the caller discover it as a 422. That is a structural rule of the contract, not a
 * per-deployment setting: the server will not put a record in an audit trail that argues with
 * itself, so a {@link ReviewVerdict#APPROVED} beside {@code wasCorrect: false} can never be
 * recorded by any deployment and there is nothing to be gained by finding that out over the
 * network. Configurable caps are the opposite case and are deliberately NOT mirrored here.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
    "field", "doc", "chosenGovernanceId", "suggestedGovernanceId", "wasCorrect", "reviewer", "ts",
    "verdict"
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
        @JsonProperty("ts") String ts,

        /**
         * What the reviewer did, or null for the shape that predates this member.
         *
         * <p>Omitted from the request when null, never sent as an explicit null -- the server
         * records the null itself, and this is the last key in the emitted order for the same
         * reason it is the last key in the server's stored record: appending keeps every line a
         * previous build wrote a prefix-compatible reading of this one.
         *
         * <p>Must agree with {@link #wasCorrect()}. See the type javadoc.
         */
        @JsonProperty("verdict") ReviewVerdict verdict) {

    public Feedback {
        Objects.requireNonNull(field, "field");
        Objects.requireNonNull(chosenGovernanceId, "chosenGovernanceId");
        Objects.requireNonNull(reviewer, "reviewer");
        Objects.requireNonNull(ts, "ts");
        if (verdict != null) {
            if (!verdict.isKnown()) {
                throw new IllegalArgumentException(
                        "verdict " + verdict.wireValue() + " is not one this build can send. "
                                + "ReviewDecision.UNKNOWN is the sentinel for a value decoded from "
                                + "a newer server; the service publishes APPROVED, REJECTED and "
                                + "MANUAL_OVERRIDE and would answer 422.");
            }
            boolean required = verdict.decision().requiredWasCorrect().orElseThrow();
            if (required != wasCorrect) {
                throw new IllegalArgumentException(
                        "verdict " + verdict.wireValue() + " contradicts wasCorrect=" + wasCorrect
                                + ". APPROVED means the matcher's suggestion was accepted and "
                                + "requires wasCorrect=true; REJECTED and MANUAL_OVERRIDE both "
                                + "require false. The server refuses this rather than reconciling "
                                + "it, because a trail that argues with itself cannot be cited.");
            }
        }
    }

    /** A verdict stamped with an {@link Instant}, rendered ISO-8601, carrying no {@code verdict}. */
    public static Feedback of(
            String field,
            String chosenGovernanceId,
            boolean wasCorrect,
            String reviewer,
            Instant at) {
        return new Feedback(
                field, null, chosenGovernanceId, null, wasCorrect, reviewer,
                Objects.requireNonNull(at, "at").toString(), null);
    }

    /** This verdict with the column comment attached. */
    public Feedback withDoc(String newDoc) {
        return new Feedback(
                field, newDoc, chosenGovernanceId, suggestedGovernanceId, wasCorrect, reviewer, ts,
                verdict);
    }

    /** This verdict recording what the matcher had suggested instead. */
    public Feedback withSuggestedGovernanceId(String suggested) {
        return new Feedback(
                field, doc, chosenGovernanceId, suggested, wasCorrect, reviewer, ts, verdict);
    }

    /**
     * This record carrying what the reviewer did.
     *
     * <p>{@code wasCorrect} is set from the verdict rather than left to contradict it: the two
     * describe one decision and the server refuses a record where they disagree, so there is no
     * combination worth preserving here. Pass null to clear the verdict, which leaves
     * {@code wasCorrect} untouched.
     *
     * @throws IllegalArgumentException if the verdict is a value this build cannot send
     */
    public Feedback withVerdict(ReviewVerdict newVerdict) {
        if (newVerdict == null) {
            return new Feedback(
                    field, doc, chosenGovernanceId, suggestedGovernanceId, wasCorrect, reviewer, ts,
                    null);
        }
        if (!newVerdict.isKnown()) {
            // Rejected here as well as in the constructor, so the message names withVerdict rather
            // than an argument list the caller never wrote.
            throw new IllegalArgumentException(
                    "ReviewDecision.UNKNOWN is a decode sentinel, not a verdict this client can "
                            + "send. See ReviewVerdict.of(ReviewDecision).");
        }
        boolean agreeing = newVerdict.decision().requiredWasCorrect().orElseThrow();
        return new Feedback(
                field, doc, chosenGovernanceId, suggestedGovernanceId, agreeing, reviewer, ts,
                newVerdict);
    }

    /** {@link #verdict()} without the null. Empty is the shape that predates the member. */
    public Optional<ReviewVerdict> verdictValue() {
        return Optional.ofNullable(verdict);
    }
}
