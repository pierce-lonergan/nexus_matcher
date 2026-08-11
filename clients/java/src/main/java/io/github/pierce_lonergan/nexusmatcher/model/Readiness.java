package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The answer from {@code GET /health/ready} -- including when the answer is no.
 *
 * <p>The server reports "not ready" as a 503 whose body is the error envelope, with the component
 * map under {@code details.components}. The client turns that into a {@code Readiness} with
 * {@link #ready()} false rather than into an exception: a readiness probe that throws when the
 * service is not ready is a probe every caller has to wrap in a try/catch before they can use it,
 * and "not ready" is the answer this endpoint exists to give.
 *
 * <p>Every other transport failure -- a connection refused, a 500, a body that will not parse --
 * still throws. Not-ready is a fact; unreachable is a failure, and they are not the same.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Readiness(

        /** Whether every component that gates readiness is healthy. */
        @JsonProperty("ready") boolean ready,

        /** The server's clock when it answered, or null on the 503 path. */
        @JsonProperty("timestamp") String timestamp,

        /** Component name to health. Carries {@code api}, {@code config} and {@code matcher}. */
        @JsonProperty("components") Map<String, Boolean> components) {

    @JsonCreator
    public Readiness {
        components = components == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(components));
    }

    /** Whether the matching routes will answer rather than 503. False when no dictionary loaded. */
    public boolean isMatcherReady() {
        return Boolean.TRUE.equals(components.get("matcher"));
    }

    /** The components reporting unhealthy, sorted. Empty when everything is green. */
    public List<String> unhealthyComponents() {
        return components.entrySet().stream()
                .filter(entry -> !Boolean.TRUE.equals(entry.getValue()))
                .map(Map.Entry::getKey)
                .sorted()
                .toList();
    }
}
