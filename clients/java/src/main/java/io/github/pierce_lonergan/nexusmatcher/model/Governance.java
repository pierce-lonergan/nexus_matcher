package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Optional;

/**
 * The protection class a matched glossary entry confers, in the caller's own controlled
 * vocabulary.
 *
 * <p><strong>Every string here is open.</strong> {@code code}, {@code name},
 * {@code classification} and {@code enhancement} come from a JSON file the caller owns; the
 * library defines none of them and ships no taxonomy at all. They are {@code String} and they
 * must never become a Java enum -- see {@link MatchDecision} for the one thing in this client that
 * legitimately is one.
 *
 * <p>To compare two classifications you need {@link Vocabulary#tiersMostOpenFirst()}, which rides
 * on the response that carried them. Do not sort tier names: alphabetical order puts
 * {@code CONFIDENTIAL} above {@code PUBLIC}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Governance(

        /** The vocabulary's code for this class. Open string. */
        @JsonProperty("code") String code,

        /** The vocabulary's human-readable name for this class. Open string. */
        @JsonProperty("name") String name,

        /** The tier this code derives, in the caller's own vocabulary. Open string. */
        @JsonProperty("classification") String classification,

        /** Whether this class marks personal information. */
        @JsonProperty("personalInformation") boolean personalInformation,

        /** Whether this class marks a direct identifier. */
        @JsonProperty("directIdentifier") boolean directIdentifier,

        /**
         * The caller's handling instruction for this class -- masking, tokenisation, a retention
         * rule -- passed through untouched and never interpreted, by the server or by this client.
         *
         * <p>Nullable, and null is a <em>declared</em> value rather than a defect: five of the nine
         * classes in the repository's example pack set it to null, which means the tier is the
         * whole instruction. Use {@link #enhancementValue()} if you would rather not hold a null.
         */
        @JsonProperty("enhancement") String enhancement) {

    @JsonCreator
    public Governance {
        // No validation of the strings. They are somebody else's vocabulary and this client is
        // not entitled to an opinion about what a code or a tier name may look like.
    }

    /** {@link #enhancement()} as an {@link Optional}, for callers who prefer not to hold a null. */
    public Optional<String> enhancementValue() {
        return Optional.ofNullable(enhancement);
    }
}
