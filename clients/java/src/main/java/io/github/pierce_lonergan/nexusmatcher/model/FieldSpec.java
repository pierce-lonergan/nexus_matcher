package io.github.pierce_lonergan.nexusmatcher.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One schema field to get governance for.
 *
 * <p><strong>These four keys and no others.</strong> The server sets {@code extra="forbid"} on
 * this object, so an unrecognised key is a 422 -- a misspelled {@code doc} silently ignored would
 * drop the column comment, and the column comment is real retrieval signal. In particular they are
 * NOT the {@code flattenedName} / {@code dataType} spellings used by the repository's example
 * {@code fields.json}: that file is the example pack's own input format, and pasting a row from it
 * into a request is the 422 two reviewers have already hit.
 *
 * <p>Nulls are omitted from the serialised body rather than sent, because the server's optional
 * members are typed as strings with defaults and would refuse an explicit {@code null}.
 *
 * <p><strong>Send a dotted {@code path}.</strong> The segment before the last dot becomes the
 * retrieval query's parent context, which is the single largest accuracy factor measured on this
 * task -- and it is also the key you look the answer up under.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({"name", "path", "doc", "type"})
public record FieldSpec(

        /** The column's own name. Required. */
        @JsonProperty("name") String name,

        /** The caller's identifier for this field, and the key the response comes back under.
         *  Null lets the server default it to {@code name}. */
        @JsonProperty("path") String path,

        /** Column comment or description, if any. */
        @JsonProperty("doc") String doc,

        /** Source type name, normalised server-side. Unknown types are accepted. */
        @JsonProperty("type") String type,

        /** Per-field query signals: the declared extension point for context this library
         *  has no opinion about. Known keys are {@code abbreviations} (a short-to-long map
         *  merged for THIS REQUEST ONLY -- the point being that a live catalog changes
         *  between calls, so a startup-time file cannot substitute), {@code entity} (the
         *  parent record name) and {@code domain} (a namespace or domain hint). A key the
         *  server does not know is CARRIED, not rejected.
         *
         *  Deliberately untyped: the values are the caller's vocabulary and closing over
         *  them here would make this artifact refuse signals a newer deployment understands.
         *  Note the asymmetry -- an unknown key beside {@code doc} is still a 422, because a
         *  typo and an extension are different events. Null sends nothing. */
        @JsonProperty("signals") Map<String, Object> signals) {

    public FieldSpec {
        // `name` only. The server's own length caps and field-count caps are deliberately NOT
        // mirrored here: they are configurable per deployment, so a copy in this artifact would
        // refuse requests a tuned server accepts, and would go stale silently. The one check kept
        // is the one that is structural rather than configured -- a field with no name cannot be
        // matched by any deployment.
        Objects.requireNonNull(name, "name");
        if (name.isBlank()) {
            throw new IllegalArgumentException("FieldSpec.name must not be blank");
        }
    }

    /** A field identified by name alone; the server keys the response by {@code name}. */
    public static FieldSpec of(String name) {
        return new FieldSpec(name, null, null, null, null);
    }

    /** A field with the dotted path the response will be keyed by. */
    public static FieldSpec of(String name, String path) {
        return new FieldSpec(name, path, null, null, null);
    }

    /** The shape worth sending: a dotted path, the column comment, and the source type. */
    public static FieldSpec of(String name, String path, String doc, String type) {
        return new FieldSpec(name, path, doc, type, null);
    }

    /** This field with a different {@code doc}. */
    public FieldSpec withDoc(String newDoc) {
        return new FieldSpec(name, path, newDoc, type, null);
    }

    /** This field with a different {@code type}. */
    public FieldSpec withType(String newType) {
        return new FieldSpec(name, path, doc, newType, null);
    }

    /**
     * The key this field's results will be returned under: the {@code path} when one was given,
     * otherwise the {@code name}, which is the server's own fallback.
     */
    public String responseKey() {
        return path == null || path.isEmpty() ? name : path;
    }

    /**
     * Split a list of fields into chunks of at most {@code size}.
     *
     * <p>Here because re-chunking is the documented response to a 413, and
     * {@code PayloadTooLargeException.suggestedChunkSize()} hands you the number to pass. Order is
     * preserved within and across chunks, which matters: the response is keyed by path in the order
     * sent.
     */
    public static List<List<FieldSpec>> chunk(List<FieldSpec> fields, int size) {
        Objects.requireNonNull(fields, "fields");
        if (size < 1) {
            throw new IllegalArgumentException("chunk size must be >= 1, got " + size);
        }
        List<List<FieldSpec>> chunks = new ArrayList<>();
        for (int start = 0; start < fields.size(); start += size) {
            chunks.add(List.copyOf(fields.subList(start, Math.min(start + size, fields.size()))));
        }
        return List.copyOf(chunks);
    }
}
