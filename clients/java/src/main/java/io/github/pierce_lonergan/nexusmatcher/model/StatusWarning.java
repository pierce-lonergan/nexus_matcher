package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * One reason not to start a bulk run yet.
 *
 * <p>Branch on {@link #code()}, never on {@link #message()} -- the wording is not part of the
 * contract. The codes the service documents today are {@code NO_DICTIONARY},
 * {@code EMPTY_DICTIONARY} and {@code FALLBACK_ENCODER}.
 *
 * <p>{@code code} is an open {@link String} and not a Java enum, and that is a deliberate reversal
 * of the choice made for {@link MatchDecision}. The set is a diagnostic vocabulary the service is
 * expected to extend as it learns new ways to be degraded; a closed binding would turn "the server
 * grew a new warning" into "the operator's status check throws", on precisely the surface an
 * operator reaches for when something is already wrong. Constants for the three known codes are on
 * {@link ServiceStatus}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record StatusWarning(

        /** Stable machine-readable code. Branch on this. */
        @JsonProperty("code") String code,

        /** What is wrong and what to change. Human-readable; not part of the contract. */
        @JsonProperty("message") String message) {

    @JsonCreator
    public StatusWarning {
    }
}
