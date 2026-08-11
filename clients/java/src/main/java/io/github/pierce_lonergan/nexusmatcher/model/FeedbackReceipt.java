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
 * that may gain keys; the two that matter are named by {@link #receivedAt()} and
 * {@link #storedField()}.
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

    private Optional<String> stringValue(String key) {
        Object value = record.get(key);
        return value == null ? Optional.empty() : Optional.of(String.valueOf(value));
    }
}
