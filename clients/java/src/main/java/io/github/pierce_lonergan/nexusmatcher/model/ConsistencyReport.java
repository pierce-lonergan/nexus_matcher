package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * The {@code consistency} block: columns the server grouped as one business concept, and whether
 * their rank-1 answers agree.
 *
 * <p>Present only when the request asked for it -- {@link MatchRequest#withConsistency(boolean)}.
 * {@link MatchResponse#consistency()} is null otherwise, and the block is strictly additive:
 * nothing in {@code results} or {@code fieldDecisions} changes whatever it finds, and
 * {@link #promotionApplied()} says so machine-readably rather than leaving a consumer to infer it.
 *
 * <p>The idea is sound and cheap: fields are matched one at a time and independently, so nothing
 * notices when two columns that are the same concept get different answers. Detecting that
 * disagreement needs no labelled data, which is what makes it deployable at all.
 *
 * <h2>READ THIS BEFORE TURNING IT ON. The feature is OFF BY DEFAULT because its grouping was
 * measured and the measurement came back negative.</h2>
 *
 * <p>Everything here depends on the GROUPING rule, and the rule is a heuristic over column names.
 * The service measured it against an oracle -- two columns are one concept when they share a leaf
 * name and the domain of their single correct answer -- on its own generated corpus. Those numbers
 * are about that corpus and are not a prediction about your schemas, and they are restated here
 * because a client that surfaces findings without them is lending the findings a confidence
 * nobody has:
 *
 * <ul>
 *   <li><strong>At the default {@link #qualifierSegments()} of 1</strong> -- a leaf groups only
 *       with a leaf under the same declared parent -- the rule emits <strong>no group at all on
 *       every generated profile</strong>. It reports nothing, and therefore claims nothing false.
 *       An empty {@link #groups()} is the expected answer, not a fault.
 *   <li><strong>At 0</strong>, the leaf alone: 0.86-1.00 pair-precision at 0.06-0.14 recall on a
 *       parent-diverse mixture, and <strong>0.0233 precision at recall 1.00</strong> on a
 *       repeated-leaf schema -- one leaf governed separately in ~30 domains, which is the shape
 *       this feature was proposed for. There it emitted four groups containing <strong>zero
 *       concepts and four collisions</strong>: 87 columns spanning 29 genuinely distinct answers
 *       merged into one "concept" and reported to a reviewer as a contradiction. Nothing in the
 *       names distinguishes the two cases.
 *   <li><strong>There is no operating point, and that was established by search rather than by
 *       observation.</strong> The service's own tests walk the entire published policy space --
 *       684 policies, every {@code qualifier_segments} x {@code includeDataType} x
 *       {@code orderSensitive} x {@code minGroupSize} combination, over two profiles, two scales
 *       and two repetition depths. The best precision reached by ANY policy that reports anything
 *       at all on the repeated-leaf schema is <strong>0.0235</strong>.
 * </ul>
 *
 * <p>So: a DISAGREE is a prompt to look, never a defect report about the matcher. Check
 * {@link ConceptGroup#distinctAnswers()} against {@link ConceptGroup#answeredCount()} first --
 * when the two are close the group is a collision of distinct concepts that happen to share a
 * column name, and the "inconsistency" is the grouping's, not the model's.
 * {@link #groupsDisagreeing()} is a count of groups, not a count of problems.
 *
 * <p>{@link #grouping()} publishes the rule that produced these groups, because a finding cannot be
 * judged without it: a group of six that disagree means one thing under a leaf-only key and quite
 * another under a key that also matched their parent.
 *
 * <p>One thing worth knowing when you choose your {@code path}s: the concept key is built from the
 * response key -- your own {@code path} -- and not from {@link FieldSpec#name()}. Segments are
 * boundaries you declared: dots, or the {@code __} array boundary. Single underscores are tokens
 * inside a segment, so {@code a_b__c_d_e} has two segments and {@code a.b.c} has three.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ConsistencyReport(

        /**
         * The policy these groups were built under: {@code qualifierSegments},
         * {@code includeDataType}, {@code orderSensitive} and {@code minGroupSize}.
         *
         * <p>Left as an open {@link Map} rather than restated as a record, because the server
         * publishes it as a free-form object and a policy that gains a knob would otherwise be
         * silently truncated here -- on the one member whose whole job is to let a reader judge the
         * finding. {@link #qualifierSegments()} and the three predicates below read it.
         */
        @JsonProperty("grouping") Map<String, Object> grouping,

        /** How many groups of two or more columns were found. Zero is the expected answer at the
         *  server's default grouping, not a fault -- see the type javadoc. */
        @JsonProperty("groupsFound") int groupsFound,

        /**
         * How many of this request's fields fell into a group of two or more.
         *
         * <p>A column that shares its concept with nothing else is not reported at all: it cannot
         * disagree with anyone.
         */
        @JsonProperty("fieldsGrouped") int fieldsGrouped,

        /** How many groups have an {@link ConceptGroup#agreement()} of {@code DISAGREE}. A count of
         *  groups, not of problems -- see the type javadoc. */
        @JsonProperty("groupsDisagreeing") int groupsDisagreeing,

        /**
         * Always false. This block changed nothing in {@code results} or {@code fieldDecisions}.
         *
         * <p>Published as a fact about the response rather than left implicit, so a consumer can
         * assert it rather than trust it. If it is ever true, this client is talking to a server
         * that overrides answers from a majority vote and the whole reading above changes.
         */
        @JsonProperty("promotionApplied") boolean promotionApplied,

        /** The groups, ordered by where their first member appeared in the request, so two
         *  identical requests produce the same list. */
        @JsonProperty("groups") List<ConceptGroup> groups) {

    @JsonCreator
    public ConsistencyReport {
        grouping = grouping == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(grouping));
        groups = groups == null ? List.of() : List.copyOf(groups);
    }

    /**
     * How many of a column's declared path segments joined its leaf in the concept key.
     *
     * <p>1 is the server's default: a leaf groups only with a leaf under the same declared parent,
     * which is the setting that reports nothing on the corpus rather than the one that reports
     * wrongly. 0 is the loose key the type javadoc's 0.0233 was measured at. Empty when the server
     * did not publish the knob.
     */
    public OptionalInt qualifierSegments() {
        Object value = grouping.get("qualifierSegments");
        return value instanceof Number number
                ? OptionalInt.of(number.intValue())
                : OptionalInt.empty();
    }

    /** Whether the data type joined the concept key. Empty when the server did not publish it. */
    public Optional<Boolean> includeDataType() {
        return flag("includeDataType");
    }

    /** Whether segment ORDER was part of the key. Empty when the server did not publish it. */
    public Optional<Boolean> orderSensitive() {
        return flag("orderSensitive");
    }

    /** The smallest group the server reports. Empty when the server did not publish it. */
    public OptionalInt minGroupSize() {
        Object value = grouping.get("minGroupSize");
        return value instanceof Number number
                ? OptionalInt.of(number.intValue())
                : OptionalInt.empty();
    }

    /** Only the groups whose members did not agree, in the order the server sent them. */
    public List<ConceptGroup> disagreeingGroups() {
        return groups.stream().filter(ConceptGroup::disagrees).toList();
    }

    /** The group containing one path, empty when that column was grouped with nothing. */
    public Optional<ConceptGroup> groupFor(String path) {
        return groups.stream().filter(group -> group.fields().contains(path)).findFirst();
    }

    private Optional<Boolean> flag(String key) {
        Object value = grouping.get(key);
        return value instanceof Boolean bool ? Optional.of(bool) : Optional.empty();
    }
}
