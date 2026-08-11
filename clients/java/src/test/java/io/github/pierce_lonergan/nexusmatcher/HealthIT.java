package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.model.HealthStatus;
import io.github.pierce_lonergan.nexusmatcher.model.Readiness;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Health and readiness, against real servers in both states. */
class HealthIT {

    @Test
    @DisplayName("health reports the running version and an uptime")
    void healthReportsVersion() {
        HealthStatus health =
                NexusMatcherClient.builder(LiveService.matching()).build().health();

        assertTrue(health.isHealthy());
        assertEquals("healthy", health.status());
        assertFalse(health.version().isBlank());
        assertTrue(health.uptimeSeconds().orElseThrow() >= 0);
    }

    @Test
    @DisplayName("a ready service reports every component green")
    void readyServiceReportsGreen() {
        Readiness readiness =
                NexusMatcherClient.builder(LiveService.matching()).build().readiness();

        assertTrue(readiness.ready());
        assertTrue(readiness.isMatcherReady());
        assertTrue(readiness.unhealthyComponents().isEmpty());
    }

    @Test
    @DisplayName("a service with no dictionary answers 'not ready' as a value, not an exception")
    void notReadyIsAnAnswerRatherThanAFailure() {
        // The server answers this as a 503, and a readiness probe that throws on "not ready" is a
        // probe every caller has to wrap in a try/catch before they can use it. So the client
        // reads the 503's own body -- which carries the component map precisely so an operator can
        // see WHICH component is red -- and hands it back.
        Readiness readiness =
                NexusMatcherClient.builder(LiveService.unavailable()).build().readiness();

        assertFalse(readiness.ready());
        assertFalse(readiness.isMatcherReady());
        assertTrue(
                readiness.unhealthyComponents().contains("matcher"),
                "the map must name the red component, not just say something is: "
                        + readiness.components());
    }

    @Test
    @DisplayName("health is 200 even on a server that can classify nothing, so do not gate on it")
    void healthIsNotARolloutGate() {
        HealthStatus health =
                NexusMatcherClient.builder(LiveService.unavailable()).build().health();

        assertEquals(
                "degraded",
                health.status(),
                "this server cannot answer a single match, and /health still returns 200. A "
                        + "rollout gate pointed here passes it; that is what readiness() is for.");
        assertFalse(health.isHealthy());
    }
}
