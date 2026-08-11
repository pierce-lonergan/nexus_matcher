package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

/**
 * The verdict the server reached about one candidate.
 *
 * <p>This is the one closed set in this client, and it is closed because it is the
 * <em>library's</em> own vocabulary rather than a caller's: {@code /openapi.json} publishes it as
 * an enum of exactly these three values. Everything that comes out of a caller's controlled
 * vocabulary -- {@link Governance#code()}, {@link Governance#name()},
 * {@link Governance#classification()}, {@link Governance#enhancement()} -- is an open
 * {@code String} here and must stay one. Typing somebody's taxonomy as a Java enum bakes their
 * catalog into this artifact and breaks every other organisation.
 *
 * <p><strong>Per candidate, not per field.</strong> Every rank is compared against the server's
 * review threshold independently, so runner-ups are routinely {@link #REJECT} on a field whose
 * top match is fine. Only a {@code REJECT} at rank 1 means "no entry describes this field" -- see
 * {@link MatchCandidate#governanceStatus()}.
 */
public enum MatchDecision {

    /** The server would apply this candidate's governance without a human. */
    AUTO_APPROVE,

    /** A human must decide. Never read this as "probably fine". */
    REVIEW,

    /** The server would not apply this candidate. */
    REJECT;

    @JsonValue
    public String wireValue() {
        return name();
    }

    /**
     * Parse a wire value, refusing anything this client does not know.
     *
     * <p>Loudly, on purpose. A fourth decision arriving from a newer server is a contract change
     * that decides whether a classification gets applied without a human, and the worst way to
     * meet it is to map it onto {@link #REVIEW} and carry on. The client surfaces it as a protocol
     * failure naming the value; unknown keys elsewhere in the response are ignored, because those
     * are additive and this is not.
     */
    @JsonCreator
    public static MatchDecision fromWire(String value) {
        for (MatchDecision decision : values()) {
            if (decision.name().equals(value)) {
                return decision;
            }
        }
        throw new IllegalArgumentException(
                "unknown MatchDecision " + value + "; this client knows AUTO_APPROVE, REVIEW and "
                        + "REJECT. A newer server has added a decision and the meaning of this "
                        + "candidate cannot be guessed -- upgrade nexus-matcher-client.");
    }
}
