package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * The dictionary entry's PASS-THROUGH PLANE: the deployment's own enrichment columns, carried the
 * length of the pipeline and never interpreted.
 *
 * <p>A glossary carries columns this library has no opinion about -- a steward, a review date, an
 * upstream system's identifier, a lifecycle token. The loader is told which to carry; they ride on
 * the entry, through the index, and come back here, so a deployment does not have to join the
 * response back against its own spreadsheet to find out what it already knew.
 *
 * <h2>{@link #values()} is a Map and must never become a record</h2>
 *
 * <p>Both the keys and the values are the deployment's own vocabulary: the keys come from its
 * loader configuration and the values from its cells. Typing either side here -- a record with
 * named components, an enum of key names, a narrower value type -- would compile one organisation's
 * spreadsheet into this artifact and break every other one. It is the same rule that keeps
 * {@link Governance#code()} and {@link Governance#classification()} open strings, applied to a
 * surface where the caller owns even the key names. {@code Map<String, Object>} is the correct
 * binding and the widest one that is still honest.
 *
 * <p>Values are whatever the source held. A delimited-text glossary produces strings; a JSON or
 * Parquet one can produce numbers, booleans, nulls, lists and nested objects, and those arrive as
 * the Jackson equivalents. {@link #renderedKeys()} names the ones that could not.
 *
 * <h2>Nothing here was read</h2>
 *
 * <p>No score, no ranking, no threshold and no governance verdict in the response depends on
 * anything in this map -- that is the bargain that lets it be carried at all. It follows that
 * nothing here can be used to justify a classification either: if a deployment's own pipeline
 * branches on one of these values, that is the deployment's rule and not this service's answer.
 *
 * <h2>{@link #droppedKeyCount()} is the one number to check before trusting the map</h2>
 *
 * <p>The loader caps how many bytes of pass-through plane one entry may carry. Above the cap it
 * drops keys and records how many, and {@code droppedKeyCount} is that count reaching a consumer
 * for the first time. Non-zero means {@link #values()} is a BOUNDED SUBSEQUENCE of the source row
 * and this response is not the place to read that row from. It is a count and not a list of names
 * because the dropped names went with the dropped values.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record SourceMetadata(

        /** The deployment's own enrichment columns, in the order the loader carried them. Empty
         *  when the entry has none -- an empty map, never a missing member. */
        @JsonProperty("values") Map<String, Object> values,

        /** How many keys the loader dropped from this entry to fit its per-entry size cap. 0 means
         *  {@link #values()} is the whole plane the source row supplied. */
        @JsonProperty("droppedKeyCount") int droppedKeyCount,

        /**
         * The keys whose value is the source value RENDERED AS TEXT rather than the source value
         * itself, in the order they appear in {@link #values()}.
         *
         * <p>Empty for every source JSON can represent natively, which is every delimited-text
         * glossary. Non-empty when a spreadsheet or database column held something JSON has no form
         * for -- a date cell, a decimal, a blob, a non-finite number -- in which case that key's
         * value is that object's text form and the original type is NOT recoverable from this
         * response. Named rather than silently coerced, because a caller parsing one of these back
         * into a date needs to know it is parsing a rendering.
         */
        @JsonProperty("renderedKeys") List<String> renderedKeys) {

    @JsonCreator
    public SourceMetadata {
        // LinkedHashMap wrapped unmodifiable rather than Map.copyOf, for both of the usual
        // reasons at once: the key order is the loader's and is part of what "pass-through" means,
        // and a cell that was null in the source arrives as a null VALUE, which Map.copyOf refuses
        // outright. Refusing to decode a glossary row because one of its spare columns was blank
        // would be this client having an opinion about a plane it is not allowed to interpret.
        values = values == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(values));
        renderedKeys = renderedKeys == null ? List.of() : List.copyOf(renderedKeys);
    }

    /** The plane an entry with no enrichment columns carries. */
    public static SourceMetadata empty() {
        return new SourceMetadata(Map.of(), 0, List.of());
    }

    /** Whether this entry carries no pass-through columns at all. */
    public boolean isEmpty() {
        return values.isEmpty();
    }

    /**
     * Whether {@link #values()} is the whole plane the source row supplied.
     *
     * <p>False means the loader trimmed this entry to its size cap. Check it before treating this
     * map as a record of the glossary row; a key that is absent from a trimmed plane was not
     * necessarily absent from the source.
     */
    public boolean isComplete() {
        return droppedKeyCount == 0;
    }

    /** The column names present, in the order the loader carried them. */
    public Set<String> keys() {
        return values.keySet();
    }

    /**
     * One column's value, empty when the entry does not carry that column.
     *
     * <p>Empty is ALSO what a column present with a null value gives, and on a trimmed plane
     * ({@link #isComplete()} false) it is what a dropped column gives. None of those three is
     * "the deployment does not use this column", so do not read it as one.
     */
    public Optional<Object> value(String key) {
        return Optional.ofNullable(values.get(key));
    }

    /**
     * One column's value as text.
     *
     * <p>Convenience for the common case of a text glossary. A non-string value is rendered with
     * {@link String#valueOf(Object)} here, which is this CLIENT converting -- it is not the same as
     * the server's own rendering, and {@link #wasRendered(String)} is how to tell which you have.
     */
    public Optional<String> text(String key) {
        return value(key).map(String::valueOf);
    }

    /**
     * Whether this key's value is a text rendering of something JSON could not carry.
     *
     * <p>True means the original type is gone: a date cell is now the text a date prints as, and
     * parsing it back is a guess about the source system's format rather than a round trip.
     */
    public boolean wasRendered(String key) {
        return renderedKeys.contains(key);
    }
}
