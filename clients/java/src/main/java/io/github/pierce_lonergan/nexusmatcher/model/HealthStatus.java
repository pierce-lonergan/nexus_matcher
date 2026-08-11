package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * The answer from {@code GET /health}.
 *
 * <p><strong>Do not point a rollout gate at this.</strong> It answers 200 with
 * {@code status: "degraded"} when a component is red, so a gate that only checks the status code
 * passes a server that can classify nothing. {@link io.github.pierce_lonergan.nexusmatcher.model.Readiness}
 * is the one to gate on.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record HealthStatus(

        /** {@code healthy} or {@code degraded}. */
        @JsonProperty("status") String status,

        /** The server's clock when it answered, ISO-8601. */
        @JsonProperty("timestamp") String timestamp,

        /** The running package version, resolved at startup. */
        @JsonProperty("version") String version,

        /** Free-form checks; carries {@code uptime_seconds} today. */
        @JsonProperty("checks") Map<String, Object> checks) {

    @JsonCreator
    public HealthStatus {
        checks = checks == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(checks));
    }

    /** Whether the server reported {@code healthy} rather than {@code degraded}. */
    public boolean isHealthy() {
        return "healthy".equals(status);
    }

    /** {@code checks.uptime_seconds}, when the server reported one. */
    public OptionalDouble uptimeSeconds() {
        Object value = checks.get("uptime_seconds");
        return value instanceof Number number
                ? OptionalDouble.of(number.doubleValue())
                : OptionalDouble.empty();
    }

    /** One entry from {@link #checks()}. */
    public Optional<Object> check(String name) {
        return Optional.ofNullable(checks.get(name));
    }
}
