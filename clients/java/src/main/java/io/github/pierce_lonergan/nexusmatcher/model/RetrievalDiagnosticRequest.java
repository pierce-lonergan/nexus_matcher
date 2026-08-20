package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.Objects;

/**
 * One field to diagnose, and optionally the entry the caller expected it to find.
 *
 * <p>Send the SAME {@link FieldSpec} that produced the disappointing match, {@code doc} included. A
 * column comment is real retrieval signal, so a diagnostic run without it diagnoses a different
 * query than the one that actually missed.
 *
 * <p>Naming {@link #expectedGovernanceId()} is what makes the answer actionable. Without it the
 * response says what came back; with it, the response also says where the entry you wanted landed
 * in each channel, or that it is not in the dictionary at all. Those are two different diagnoses --
 * "retrieved at rank 34" is a scoring problem, "not in the dictionary" is a glossary problem -- and
 * confusing them wastes a day.
 *
 * <p>Request keys are snake_case here and response keys are camelCase, exactly as on
 * {@link MatchRequest}. That asymmetry is the agreed wire contract and this record spells both
 * sides.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"field", "expected_governance_id", "top_k"})
public record RetrievalDiagnosticRequest(

        /** The field to diagnose, in the same shape {@code /api/v1/match} takes -- so the query
         *  this route reports is the query that route would have built. */
        @JsonProperty("field") FieldSpec field,

        /** The dictionary id the caller believes this field should have matched. Null asks only
         *  what came back. */
        @JsonProperty("expected_governance_id") String expectedGovernanceId,

        /**
         * Candidates shown per channel. Null takes the server's default of 10.
         *
         * <p>Ranks are computed over the FULL channel result rather than over this truncation, so an
         * expected entry at rank 34 is reported as 34 rather than as absent -- raising this number
         * shows you more rows, it does not change the diagnosis.
         */
        @JsonProperty("top_k") Integer topK) {

    public RetrievalDiagnosticRequest {
        Objects.requireNonNull(field, "field");
    }

    /** Diagnose a field, asking only what came back. */
    public static RetrievalDiagnosticRequest of(FieldSpec field) {
        return new RetrievalDiagnosticRequest(field, null, null);
    }

    /** Diagnose a field against the entry you expected it to find. */
    public static RetrievalDiagnosticRequest of(FieldSpec field, String expectedGovernanceId) {
        return new RetrievalDiagnosticRequest(field, expectedGovernanceId, null);
    }

    /** This request showing {@code topK} candidates per channel. */
    public RetrievalDiagnosticRequest withTopK(int newTopK) {
        return new RetrievalDiagnosticRequest(field, expectedGovernanceId, newTopK);
    }

    /** This request naming the entry the caller expected. */
    public RetrievalDiagnosticRequest withExpectedGovernanceId(String expected) {
        return new RetrievalDiagnosticRequest(field, expected, topK);
    }
}
