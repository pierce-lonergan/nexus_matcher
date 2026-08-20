package io.github.pierce_lonergan.nexusmatcher;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/**
 * Reads the captured response bodies under {@code src/test/resources/captured/}.
 *
 * <p><strong>Every file there is verbatim output from a running service</strong>, over the
 * repository's own {@code examples/governance/} pack, and none of them has been edited afterwards.
 * They are here so the decoding tests run without a service while still testing the real wire shape
 * -- they are captures, not hand-written expectations, which is the difference between a test of
 * the contract and a test of what the author believed it was.
 *
 * <p><strong>Regenerate them, never edit them.</strong> {@code clients/java/capture-fixtures.sh}
 * re-derives every byte of every file from the running fixtures, and it exists so that "the
 * fixture disagrees with the test" can only ever be resolved the honest way. A body somebody typed
 * to make a test pass tests the author's belief about the contract, which is the exact belief the
 * fixture was there to check. Run the script, then read the diff: it is the service having changed.
 *
 * <p>Two captures do not come from the default fixture, and both for the same reason: they hold a
 * behaviour that is REACHABLE but that no correctly-configured stock server produces, so on the
 * default fixture they could only ever have been bodies somebody typed.
 *
 * <ul>
 *   <li>{@code match-response-no-match.json} comes from the server started with
 *       {@code fixture-absolute-floor.json}, because a
 *       {@link io.github.pierce_lonergan.nexusmatcher.model.FieldDecision#NO_MATCH} verdict needs a
 *       configured absolute-score floor and the library ships none.
 *   <li>{@code match-response-approved-pair.json} comes from the server started by
 *       {@code clients/java/fixture_approved_pair_app.py}, because a
 *       {@link io.github.pierce_lonergan.nexusmatcher.model.MatchProvenance#APPROVED_PAIR}
 *       candidate needs an attached feedback consumer and the library attaches none.
 * </ul>
 *
 * <p>The behaviour tests against a live service live in the {@code *IT} classes and are not
 * replaced by these.
 */
final class Fixtures {

    private Fixtures() {
    }

    static String captured(String name) {
        String path = "/captured/" + name;
        try (InputStream stream = Fixtures.class.getResourceAsStream(path)) {
            if (stream == null) {
                throw new IllegalStateException("missing captured fixture " + path);
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException exc) {
            throw new UncheckedIOException(exc);
        }
    }
}
