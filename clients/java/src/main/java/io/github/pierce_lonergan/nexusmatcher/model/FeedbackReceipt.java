package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * What the server stored, echoed back.
 *
 * <p>{@code submitFeedback} returns this rather than {@code void} on purpose. The stored record
 * carries the server's own {@code receivedAt}, which is the timestamp the audit trail is ordered
 * by and the only proof the caller has of what was actually written -- discarding it would throw
 * away the evidence the endpoint exists to produce.
 *
 * <p>{@link #record()} is left as an open map because the server's record shape is an audit format
 * that may gain keys -- and it has: {@code verdict} was appended to it, so a stored record now
 * carries nine keys where it carried eight, with an explicit null on every request that did not
 * send one. That is exactly the growth this member was left open for. The three that matter are
 * named by {@link #receivedAt()}, {@link #storedField()} and {@link #storedVerdict()}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record FeedbackReceipt(

        /** True when the verdict is on file. The server answers 201 only after the append and its
         *  fsync have returned. */
        @JsonProperty("recorded") boolean recorded,

        /** The stored line, verbatim. */
        @JsonProperty("record") Map<String, Object> record) {

    @JsonCreator
    public FeedbackReceipt {
        record = record == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(record));
    }

    /**
     * The server's UTC stamp for when the verdict arrived. Order the audit trail by this, never by
     * the client's own {@link Feedback#ts()}.
     */
    public Optional<String> receivedAt() {
        return stringValue("receivedAt");
    }

    /** The field path as stored. */
    public Optional<String> storedField() {
        return stringValue("field");
    }

    /**
     * The verdict as stored, decoded.
     *
     * <p>Empty means the server stored a null -- which is what it stores for every request that did
     * not send one, and what every record written before the member existed reads as. It does NOT
     * mean the key is missing: the server writes {@code "verdict": null} rather than omitting it,
     * so an absent key would mean a server predating the member entirely.
     *
     * <p>A value this build does not recognise decodes to {@link ReviewDecision#UNKNOWN} with the
     * server's own string on {@link ReviewVerdict#wireValue()}, rather than being refused. See
     * {@link ReviewDecision}.
     */
    public Optional<ReviewVerdict> storedVerdict() {
        return stringValue("verdict").map(ReviewVerdict::fromWire);
    }

    private Optional<String> stringValue(String key) {
        Object value = record.get(key);
        return value == null ? Optional.empty() : Optional.of(String.valueOf(value));
    }
}
