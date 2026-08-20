package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Optional;

/**
 * Everything an operator needs before starting a bulk run, in one byte-stable body.
 *
 * <p><strong>{@link #degraded()} is the question a health probe cannot answer.</strong>
 * {@link Readiness} answers "has this process finished starting" -- correctly shaped for
 * Kubernetes, and it can be perfectly green on a server that is answering every field out of a
 * fallback encoder nobody intended. This surface exists for that gap.
 *
 * <p>A NOT-READY server still answers this route 200. A diagnostic that fails when things are
 * broken is a diagnostic nobody can use, so {@link #ready()} is a field rather than a status code.
 *
 * <p>Gate a bulk run on {@code status.ready() && !status.degraded()}, and log
 * {@link #warnings()} when it is false.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ServiceStatus(

        /** Whether a dictionary is loaded and matching can be served. Mirrors
         *  {@link Readiness#isMatcherReady()}. */
        @JsonProperty("ready") boolean ready,

        /** {@code warnings != []}. True when a bulk run would produce results the operator did not
         *  intend. This is the field this surface exists for. */
        @JsonProperty("degraded") boolean degraded,

        /** Why, one entry per condition. Empty when {@link #degraded()} is false. */
        @JsonProperty("warnings") List<StatusWarning> warnings,

        /** What dictionary this server is answering out of. */
        @JsonProperty("dictionary") DictionaryStatus dictionary,

        /** Which encoder retrieval is running on. Null when no matcher is loaded. */
        @JsonProperty("encoder") EncoderStatus encoder,

        /** The thresholds in force. Null when no matcher is loaded -- never the shipped defaults,
         *  because reporting thresholds that are not in force is a wrong answer rather than a
         *  missing one. */
        @JsonProperty("thresholds") Thresholds thresholds,

        /** The caps to chunk against. */
        @JsonProperty("limits") ServiceLimits limits) {

    /** No dictionary is loaded; every match answers 503. */
    public static final String NO_DICTIONARY = "NO_DICTIONARY";

    /** A dictionary loaded and carries no entries; every field matches nothing. */
    public static final String EMPTY_DICTIONARY = "EMPTY_DICTIONARY";

    /** Retrieval is running on an encoder the selection ladder fell through to. */
    public static final String FALLBACK_ENCODER = "FALLBACK_ENCODER";

    @JsonCreator
    public ServiceStatus {
        warnings = warnings == null ? List.of() : List.copyOf(warnings);
    }

    /** Whether this server is fit to start a bulk run against: ready and not degraded. */
    public boolean fitForBulkRun() {
        return ready && !degraded;
    }

    /** Whether one warning code is present. The three the service documents are constants above. */
    public boolean hasWarning(String code) {
        return warnings.stream().anyMatch(warning -> code.equals(warning.code()));
    }

    /** The warning codes, in the order the server listed them. */
    public List<String> warningCodes() {
        return warnings.stream().map(StatusWarning::code).toList();
    }

    /** {@link #encoder()} as an {@link Optional}; empty when no matcher is loaded. */
    public Optional<EncoderStatus> encoderValue() {
        return Optional.ofNullable(encoder);
    }

    /** {@link #thresholds()} as an {@link Optional}; empty when no matcher is loaded. */
    public Optional<Thresholds> thresholdsValue() {
        return Optional.ofNullable(thresholds);
    }
}
