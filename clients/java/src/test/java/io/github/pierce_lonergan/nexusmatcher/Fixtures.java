package io.github.pierce_lonergan.nexusmatcher;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/**
 * Reads the captured response bodies under {@code src/test/resources/captured/}.
 *
 * <p><strong>Every file there is verbatim output from a running service</strong>, taken from the
 * repository's own {@code examples/governance/} pack on 2026-08-11 and not edited afterwards. They
 * are here so the decoding tests run without a service while still testing the real wire shape --
 * they are captures, not hand-written expectations, which is the difference between a test of the
 * contract and a test of what the author believed it was.
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
