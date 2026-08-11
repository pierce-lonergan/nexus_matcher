package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherException;

import java.time.Duration;
import java.util.Optional;

/**
 * Decides whether to send a failed request again, and how long to wait first.
 *
 * <p>Injectable so a caller can turn retrying off entirely ({@link #none()}), tighten it, or
 * replace it with their own -- a pipeline that already has a circuit breaker does not want a second
 * one inside its client.
 *
 * <p>Implementations must not retry a 4xx. The default one enforces that structurally by asking
 * {@link NexusMatcherException#isRetryable()}, which is false on the base class and overridden only
 * by 503 and 504.
 */
@FunctionalInterface
public interface RetryPolicy {

    /**
     * How long to wait before the next attempt, or empty to give up and throw.
     *
     * @param failure      what the last attempt failed with
     * @param attemptsMade how many attempts have already been made; 1 after the first failure
     */
    Optional<Duration> nextDelay(NexusMatcherException failure, int attemptsMade);

    /** Never retry: every failure is thrown to the caller as it arrives. */
    static RetryPolicy none() {
        return (failure, attemptsMade) -> Optional.empty();
    }

    /**
     * The shipped policy: 503 only, up to two retries, exponential backoff with jitter, capped.
     *
     * <p>504 is deliberately NOT retried here even though it is retryable once -- see
     * {@link io.github.pierce_lonergan.nexusmatcher.error.DeadlineExceededException} for why an
     * immediate retry adds load to a server that has just proved it could not finish the first
     * copy, and {@link ExponentialBackoffRetryPolicy.Builder#retryDeadlineExceeded(boolean)} to opt
     * in.
     */
    static RetryPolicy defaultPolicy() {
        return ExponentialBackoffRetryPolicy.builder().build();
    }
}
