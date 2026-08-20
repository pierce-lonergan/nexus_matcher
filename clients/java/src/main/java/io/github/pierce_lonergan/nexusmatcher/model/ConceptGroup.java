package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * One group of columns the server's grouping rule believes are the same business concept, and the
 * answers those columns got.
 *
 * <p><strong>{@link #majorityGovernanceId()} is not an instruction.</strong> It is published so a
 * reviewer can see where the weight of evidence sits. Nothing in the service applies it and
 * {@link ConsistencyReport#promotionApplied()} is false for that reason -- promoting a majority can
 * move a correct answer to an incorrect one, which merely surfacing a disagreement cannot.
 *
 * <h2>A DISAGREE is not evidence that the model is wrong</h2>
 *
 * <p>It is evidence that these columns were GROUPED and then answered differently, and the group is
 * only as good as the rule that made it. At {@link ConsistencyReport#qualifierSegments()} of 0 --
 * the leaf name alone, which is NOT the server's default -- a leaf repeated under many different
 * parents collapses into one group, and columns that SHOULD have different answers are then
 * reported as contradicting each other. Read {@link ConsistencyReport} for the measured size of
 * that effect before acting on a finding; the short version is that on a schema built from
 * repeated leaves, every group that key produced was a collision and none was a concept.
 *
 * <p>The reader's own test for it is on this type: <strong>a group whose
 * {@link #distinctAnswers()} approaches its {@link #answeredCount()} is a collision, not a
 * disagreement.</strong> Six columns that are genuinely one concept and disagree give two or three
 * distinct answers; twenty-nine distinct concepts merged under one leaf give twenty-nine.
 *
 * <h2>{@link #agreement()} keeps the server's own string</h2>
 *
 * <p>The service publishes {@code agreement} as a closed schema component, and this record holds it
 * as a {@link String} with {@link Agreement} beside it as the thing you switch on -- for the reason
 * {@link Separation} sets out. One unrecognised word describing one group must not cost a caller
 * every verdict in the batch.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ConceptGroup(

        /**
         * The concept key as a printable label: the qualifier segments, the leaf's normalised
         * tokens, its class word and the data type, separated by {@code |}.
         *
         * <p>Stable for a given request and grouping policy, so it can be quoted in a ticket. It is
         * a grouping ARTIFACT rather than a name anyone chose -- do not key a downstream system on
         * it, and do not show it to a business reader as the concept's name.
         */
        @JsonProperty("concept") String concept,

        /** The group's members, in the order they were sent. */
        @JsonProperty("fields") List<String> fields,

        /**
         * Each member's rank-1 governance id, or null where the field had no answer to give -- no
         * candidates, or a {@link FieldDecision#NO_MATCH} verdict, which inherits nothing.
         *
         * <p>A null is SILENCE, not a dissenting answer. Counting it as one would report a
         * disagreement in a group where only one column was answered at all.
         */
        @JsonProperty("answers") Map<String, String> answers,

        /** How many different non-null answers the group got. See the type javadoc for what a
         *  value close to {@link #answeredCount()} means. */
        @JsonProperty("distinctAnswers") int distinctAnswers,

        /**
         * {@code AGREE} when two or more members answered and all agree, {@code DISAGREE} when two
         * or more answered and they do not, {@code UNDECIDED} when fewer than two answered at all.
         *
         * <p>{@code UNDECIDED} is deliberately not {@code AGREE}: one answer and five blanks is not
         * five columns confirming each other.
         */
        @JsonProperty("agreement") String agreement,

        /** The modal answer within the group, or null when no single answer holds a plurality.
         *  Evidence, never an instruction -- see the type javadoc. */
        @JsonProperty("majorityGovernanceId") String majorityGovernanceId,

        /** How many members gave the majority answer; 0 when there is none. */
        @JsonProperty("majorityCount") int majorityCount) {

    @JsonCreator
    public ConceptGroup {
        fields = fields == null ? List.of() : List.copyOf(fields);
        // LinkedHashMap: the members arrive in request order and null is a documented VALUE here,
        // which Map.copyOf refuses.
        answers = answers == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(answers));
    }

    /**
     * Whether two or more members answered and did not agree.
     *
     * <p>False for an {@link #agreement()} value this build does not know. Read the type javadoc
     * before treating true as a defect in the matcher: at the default grouping this is routinely a
     * collision of distinct concepts under one leaf name.
     */
    public boolean disagrees() {
        return agreementValue() == Agreement.DISAGREE;
    }

    /** Whether two or more members answered and all gave the same id. False on an unknown value. */
    public boolean agrees() {
        return agreementValue() == Agreement.AGREE;
    }

    /** Whether fewer than two members answered at all, so there was nothing to compare. */
    public boolean isUndecided() {
        return agreementValue() == Agreement.UNDECIDED;
    }

    /** {@link #agreement()} as a value you can switch on. {@link Agreement#UNKNOWN} if this build
     *  does not know the server's word for it. */
    public Agreement agreementValue() {
        return Agreement.fromWire(agreement);
    }

    /** How many members gave a non-null answer. The denominator {@link #distinctAnswers()} is
     *  read against. */
    public int answeredCount() {
        return (int) answers.values().stream().filter(each -> each != null).count();
    }

    /** {@link #majorityGovernanceId()} without the null. Empty when no answer holds a plurality. */
    public Optional<String> majorityAnswer() {
        return Optional.ofNullable(majorityGovernanceId);
    }

    /** One member's answer, empty where the column had none to give or is not in this group. */
    public Optional<String> answerFor(String path) {
        return Optional.ofNullable(answers.get(path));
    }
}
