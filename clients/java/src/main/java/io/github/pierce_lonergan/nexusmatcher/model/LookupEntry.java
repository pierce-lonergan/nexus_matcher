package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Optional;

/**
 * One dictionary entry, resolved by id.
 *
 * <p>The same enrichment surface {@link MatchCandidate} carries, minus the four members that are
 * claims about a MATCH rather than about the entry: no rank, no confidence, no decision, no explain.
 * There is deliberately no score here. A lookup hit is exact, so a confidence would be either the
 * constant 1.0 -- a number that invites thresholding on something never measured -- or a fiction.
 *
 * <p>{@link #governance()} follows the candidate's rule exactly: present and null when the entry
 * carries no protection code, never absent, with {@link LookupResponse#vocabulary()} naming the tier
 * that null means. Use {@link #governanceValue()} rather than testing for null.
 *
 * <p>Only one of a candidate's two nulls can occur here, and that is worth knowing: an entry has no
 * verdict, so nothing can be withheld from it. A null class on a looked-up entry always means the
 * open tier.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record LookupEntry(

        /** The entry's own id, echoed from the entry rather than from the request -- so a
         *  dictionary whose id column disagrees with its key is visible rather than hidden. */
        @JsonProperty("governanceId") String governanceId,

        /** The entry's business name. */
        @JsonProperty("businessName") String businessName,

        /** The entry's definition. */
        @JsonProperty("definition") String definition,

        /** The entry's domain. */
        @JsonProperty("domain") String domain,

        /** The class this entry confers, or null for the open tier. */
        @JsonProperty("governance") Governance governance,

        /**
         * The deployment's own enrichment columns for this entry, carried through the pipeline and
         * never interpreted.
         *
         * <p>The server renders this and a match candidate's {@code sourceMetadata} with one
         * function over one object, so for a given id the two are identical rather than merely
         * similar -- which is what lets a pipeline resolve some columns by id and match the rest
         * without the two halves disagreeing about the same glossary row.
         */
        @JsonProperty("sourceMetadata") SourceMetadata sourceMetadata) {

    @JsonCreator
    public LookupEntry {
        sourceMetadata = sourceMetadata == null ? SourceMetadata.empty() : sourceMetadata;
    }

    /** The class this entry confers, empty when it sits at the vocabulary's open tier. */
    public Optional<Governance> governanceValue() {
        return Optional.ofNullable(governance);
    }

    /**
     * Which governance state this entry is in.
     *
     * <p>Only {@link GovernanceStatus#CONFERRED} and {@link GovernanceStatus#OPEN_TIER} are
     * reachable: {@link GovernanceStatus#WITHHELD_REJECTED_TOP_MATCH} is a statement about a
     * rejected match, and a lookup makes no match to reject.
     */
    public GovernanceStatus governanceStatus() {
        return governance == null ? GovernanceStatus.OPEN_TIER : GovernanceStatus.CONFERRED;
    }
}
