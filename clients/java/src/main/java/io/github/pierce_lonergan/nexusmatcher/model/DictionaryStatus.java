package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Optional;
import java.util.OptionalInt;

/**
 * What dictionary the server is answering out of.
 *
 * <p>Every member is nullable and null means one thing throughout: this server did not do the
 * loading, so it cannot name the answer. That happens when the matcher was handed over
 * already-indexed rather than built from a configured path.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record DictionaryStatus(

        /** Indexed entries, or null when no dictionary is loaded. */
        @JsonProperty("entryCount") Integer entryCount,

        /** The dictionary this server loaded, or null when it did not do the loading. */
        @JsonProperty("source") String source,

        /** UTC ISO-8601 instant this server finished INDEXING -- not when the file was written.
         *  Null when it did not do the indexing. */
        @JsonProperty("indexedAt") String indexedAt) {

    @JsonCreator
    public DictionaryStatus {
    }

    /** {@link #entryCount()} without the null. Empty means no dictionary is loaded. */
    public OptionalInt entries() {
        return entryCount == null ? OptionalInt.empty() : OptionalInt.of(entryCount);
    }

    /** {@link #source()} as an {@link Optional}. */
    public Optional<String> sourceValue() {
        return Optional.ofNullable(source);
    }
}
