package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * The {@code contrast} block: rank 1 against rank 2, for every field in the request.
 *
 * <p>Present only when the request asked for it -- {@link MatchRequest#withContrast(boolean)} --
 * and {@link MatchResponse#contrast()} is null otherwise. Off by default on the server, and adding
 * the block appends two keys to the response and changes nothing already in it.
 *
 * <p><strong>Every input path is present in {@link #fields()}, and a path with no runner-up maps to
 * null.</strong> "This field had one candidate" and "this pass skipped it" must not look alike, so
 * the server sends an explicit null rather than omitting the key -- the same call it makes for a
 * matchless field getting {@code []} in {@code results}. {@link #contrastFor(String)} folds the
 * null into an empty {@link Optional}; use {@link #paths()} when you need to tell a null entry from
 * a path that was never sent.
 *
 * <p>The runner-up is read from the FULL match list rather than from the {@code top_k} slice, so a
 * caller who asks for one candidate and a contrast is still told what the one they cannot see was.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ContrastReport(

        /**
         * The smallest difference the numbers in this response can express -- the precision every
         * published float is rounded to.
         *
         * <p>Nothing below it is reported as separating and nothing below it is named as a cause.
         * Derived by the server from its own serialiser's precision rather than typed as a
         * constant, so it moves if the precision does.
         */
        @JsonProperty("resolution") double resolution,

        /**
         * The scale contract for the contrast's own numbers, in the vocabulary
         * {@link ScoringContract#comparabilityScopesNarrowestFirst()} publishes.
         *
         * <p>A nested free-form object -- {@code confidenceGap} names the scope of the gap, and
         * {@code signals} is a map from signal name to the scope of that signal's {@code delta} and
         * {@code weightedDelta}. Left as a {@link Map} rather than restated as a record because the
         * server publishes it as an open object whose signal keys are whatever the live matcher
         * weights; a record here would silently drop a sixth signal's scope. Read it through
         * {@link #confidenceGapScope()} and {@link #signalScope(String)}.
         */
        @JsonProperty("comparability") Map<String, Object> comparability,

        /**
         * One entry per input field, keyed and ordered exactly like {@link MatchResponse#results()},
         * with a null value where the field has fewer than two candidates.
         */
        @JsonProperty("fields") Map<String, Contrast> fields) {

    @JsonCreator
    public ContrastReport {
        // LinkedHashMap, not Map.copyOf, twice over: the key order is the caller's own field order
        // and Map.copyOf does not promise to keep it -- and Map.copyOf refuses null VALUES, which
        // are a documented part of this shape, so copying through it would turn a legal body into
        // a crash on the one case the null was added to express.
        comparability = comparability == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(comparability));
        fields = fields == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(fields));
    }

    /** The paths this report covers, in the order they were sent. Includes the null entries. */
    public Set<String> paths() {
        return fields.keySet();
    }

    /**
     * The contrast for one field.
     *
     * <p>Empty means the field had fewer than two candidates, OR that the path was never sent.
     * {@link #paths()} tells those apart; the server guarantees every path you sent is a key here.
     */
    public Optional<Contrast> contrastFor(String path) {
        return Optional.ofNullable(fields.get(path));
    }

    /**
     * The declared comparability scope of {@link Contrast#confidenceGap()}, empty when this server
     * declares none -- in which case the gap must not be compared with anything.
     */
    public Optional<String> confidenceGapScope() {
        return stringAt("confidenceGap");
    }

    /**
     * The declared comparability scope of one signal's {@code delta} and {@code weightedDelta},
     * empty when this server declares none for that signal or does not carry it at all.
     */
    public Optional<String> signalScope(String signal) {
        Object signals = comparability.get("signals");
        if (!(signals instanceof Map<?, ?> map)) {
            return Optional.empty();
        }
        Object scope = map.get(signal);
        return scope == null ? Optional.empty() : Optional.of(String.valueOf(scope));
    }

    private Optional<String> stringAt(String key) {
        Object value = comparability.get(key);
        return value == null ? Optional.empty() : Optional.of(String.valueOf(value));
    }
}
