package io.github.pierce_lonergan.nexusmatcher;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Locates the running services the {@code *IT} classes talk to.
 *
 * <p><strong>These tests do not mock the service and they do not skip when it is missing.</strong>
 * A skipped integration test is a green build that proved nothing, and this contract gained two
 * response members while this client was being written -- exactly the drift a mock cannot see. So a
 * missing service is a FAILURE that names the script which starts one.
 *
 * <p>Five fixtures, because five of the behaviours worth pinning are properties of a server's
 * configuration rather than of a request:
 *
 * <ul>
 *   <li>{@code nexus.matcher.baseUrl} -- the example pack, loaded. Everything normal.
 *   <li>{@code nexus.matcher.unavailableBaseUrl} -- a server with NO dictionary, so every match is
 *       a real 503 and the retry loop can be watched against one.
 *   <li>{@code nexus.matcher.deadlineBaseUrl} -- the pack with a 1 ms deadline, so every match is a
 *       real 504.
 *   <li>{@code nexus.matcher.floorBaseUrl} -- the pack with an absolute-score floor configured, so
 *       a field the glossary does not describe earns a real
 *       {@link io.github.pierce_lonergan.nexusmatcher.model.FieldDecision#NO_MATCH}. The library
 *       ships no floor, so on any other fixture that verdict is unreachable and could only be
 *       tested against a body somebody typed.
 *   <li>{@code nexus.matcher.approvedPairBaseUrl} -- the pack with one reviewer verdict standing,
 *       so a candidate carries a real
 *       {@link io.github.pierce_lonergan.nexusmatcher.model.MatchProvenance#APPROVED_PAIR}. The
 *       library attaches no feedback consumer on any server it starts, so on every other fixture
 *       every candidate is RETRIEVAL and the other half of that vocabulary is unreachable.
 * </ul>
 *
 * <p>{@code clients/java/serve-fixtures.ps1} (and {@code .sh}) starts all five.
 */
final class LiveService {

    private static final HttpClient PROBE = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(2))
            .version(HttpClient.Version.HTTP_1_1)
            .build();

    private LiveService() {
    }

    /** The pack-loaded service. Fails the test if it is not up and ready to match. */
    static String matching() {
        String base = required("nexus.matcher.baseUrl", "NEXUS_MATCHER_BASE_URL");
        requireReachable(base);
        requireMatcherReady(base);
        return base;
    }

    /** The dictionary-less service, whose match routes answer 503. */
    static String unavailable() {
        String base = required(
                "nexus.matcher.unavailableBaseUrl", "NEXUS_MATCHER_UNAVAILABLE_BASE_URL");
        requireReachable(base);
        return base;
    }

    /**
     * The service with an absolute-score floor configured, on which NO_MATCH is reachable.
     *
     * <p>Same pack, same encoder, one extra config key -- so a NO_MATCH here differs from an
     * AUTO_APPROVE on {@link #matching()} by the server's configuration alone, which is what makes
     * the comparison worth making.
     */
    static String floor() {
        String base = required("nexus.matcher.floorBaseUrl", "NEXUS_MATCHER_FLOOR_BASE_URL");
        requireReachable(base);
        requireMatcherReady(base);
        return base;
    }

    /**
     * The service with one reviewer verdict standing, on which APPROVED_PAIR is reachable.
     *
     * <p>Same pack, same encoder, one attached consumer -- so a candidate that comes back here as
     * APPROVED_PAIR and comes back from {@link #matching()} as RETRIEVAL differs by the server's
     * configuration alone. That is the comparison worth making, and
     * {@code ApprovedPairIT} makes it by sending the SAME field spec to both.
     */
    static String approvedPair() {
        String base = required(
                "nexus.matcher.approvedPairBaseUrl", "NEXUS_MATCHER_APPROVED_PAIR_BASE_URL");
        requireReachable(base);
        requireMatcherReady(base);
        return base;
    }

    /** The service whose deadline is short enough that every match times out. */
    static String deadline() {
        String base = required("nexus.matcher.deadlineBaseUrl", "NEXUS_MATCHER_DEADLINE_BASE_URL");
        requireReachable(base);
        requireMatcherReady(base);
        return base;
    }

    private static String required(String property, String environmentVariable) {
        String value = System.getProperty(property);
        if (value == null || value.isBlank()) {
            value = System.getenv(environmentVariable);
        }
        if (value == null || value.isBlank()) {
            return fail(
                    "neither -D" + property + " nor " + environmentVariable + " is set. These "
                            + "tests run against a REAL service; start the five fixtures with "
                            + "clients/java/serve-fixtures.ps1 (or .sh) from the repository root.");
        }
        return value.strip();
    }

    private static void requireReachable(String base) {
        try {
            HttpResponse<String> response = PROBE.send(
                    HttpRequest.newBuilder(URI.create(base + "/health/live"))
                            .timeout(Duration.ofSeconds(3))
                            .GET()
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                fail(base + "/health/live answered " + response.statusCode()
                        + "; that is not a nexus_matcher service.");
            }
        } catch (IOException exc) {
            fail("no service at " + base + " (" + exc + "). Start the fixtures with "
                    + "clients/java/serve-fixtures.ps1 (or .sh) from the repository root.");
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
            fail("interrupted while probing " + base);
        }
    }

    private static void requireMatcherReady(String base) {
        io.github.pierce_lonergan.nexusmatcher.model.Readiness readiness =
                NexusMatcherClient.builder(base).build().readiness();
        if (!readiness.isMatcherReady()) {
            fail(base + " has no matcher loaded (unhealthy: " + readiness.unhealthyComponents()
                    + "). It needs NEXUS_API_DICTIONARY and NEXUS_API_GOVERNANCE; see "
                    + "clients/java/serve-fixtures.ps1.");
        }
    }
}
