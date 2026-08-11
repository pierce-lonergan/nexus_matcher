package io.github.pierce_lonergan.nexusmatcher;

import io.github.pierce_lonergan.nexusmatcher.error.DeadlineExceededException;
import io.github.pierce_lonergan.nexusmatcher.error.NexusMatcherException;

import java.time.Duration;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;

/**
 * Exponential backoff with jitter, capped, and refusing to retry anything it should not.
 *
 * <p>Three guards, in this order, and each one is the negation of a way a retry loop turns a
 * recoverable failure into an outage:
 *
 * <ol>
 *   <li><strong>Only what the failure says is retryable.</strong> Never a 4xx: the same malformed
 *       body produces the same 422, so retrying it converts one loud failure into three slow ones.
 *       Never a 500 either -- the server has already said no field was classified.
 *   <li><strong>Never more than the failure's own cap.</strong> A 504 allows at most one, because
 *       the timed-out match is still running server-side and a second copy makes the condition
 *       worse rather than better.
 *   <li><strong>Never more than {@code maxRetries}, and never longer than {@code maxDelay}.</strong>
 *       An uncapped backoff is a client that waits minutes for a server that is not coming back.
 * </ol>
 *
 * <p><strong>Jitter is not decoration.</strong> Without it, every client shed by one overload
 * retries in the same millisecond, and the second wave is as synchronised as the first. This uses
 * equal jitter -- half the computed delay, plus a random amount up to the other half -- rather than
 * full jitter, so a retry cannot arrive almost immediately at a server that has just said it is
 * full.
 */
public final class ExponentialBackoffRetryPolicy implements RetryPolicy {

    private static final int DEFAULT_MAX_RETRIES = 2;
    private static final Duration DEFAULT_BASE_DELAY = Duration.ofMillis(200);
    private static final Duration DEFAULT_MAX_DELAY = Duration.ofSeconds(5);

    private final int maxRetries;
    private final Duration baseDelay;
    private final Duration maxDelay;
    private final boolean retryDeadlineExceeded;
    private final Random random;

    private ExponentialBackoffRetryPolicy(Builder builder) {
        this.maxRetries = builder.maxRetries;
        this.baseDelay = builder.baseDelay;
        this.maxDelay = builder.maxDelay;
        this.retryDeadlineExceeded = builder.retryDeadlineExceeded;
        this.random = builder.random;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Optional<Duration> nextDelay(NexusMatcherException failure, int attemptsMade) {
        Objects.requireNonNull(failure, "failure");
        if (maxRetries < 1 || !failure.isRetryable()) {
            return Optional.empty();
        }
        if (failure instanceof DeadlineExceededException && !retryDeadlineExceeded) {
            return Optional.empty();
        }

        int retriesMade = attemptsMade - 1;
        if (retriesMade >= maxRetries || retriesMade >= failure.maxSafeRetries()) {
            return Optional.empty();
        }

        // Computed in millis as a long. Duration.multipliedBy with 2^n would overflow long before
        // maxDelay clamped it for a large maxRetries; capping the exponent first cannot.
        int exponent = Math.min(retriesMade, 30);
        long uncapped = baseDelay.toMillis() << exponent;
        long capped = Math.min(uncapped, maxDelay.toMillis());
        long half = capped / 2;
        long jittered = half + (long) (random.nextDouble() * (capped - half));
        return Optional.of(Duration.ofMillis(jittered));
    }

    /** How many retries this policy allows at most, before the failure's own cap is applied. */
    public int maxRetries() {
        return maxRetries;
    }

    /** Whether this policy will retry a 504 (at most once, whatever {@code maxRetries} says). */
    public boolean retriesDeadlineExceeded() {
        return retryDeadlineExceeded;
    }

    /** Builder for {@link ExponentialBackoffRetryPolicy}. */
    public static final class Builder {
        private int maxRetries = DEFAULT_MAX_RETRIES;
        private Duration baseDelay = DEFAULT_BASE_DELAY;
        private Duration maxDelay = DEFAULT_MAX_DELAY;
        private boolean retryDeadlineExceeded = false;
        private Random random = new Random();

        private Builder() {
        }

        /** Retries after the first attempt. 0 disables retrying; the default is 2. */
        public Builder maxRetries(int value) {
            if (value < 0) {
                throw new IllegalArgumentException("maxRetries must be >= 0, got " + value);
            }
            this.maxRetries = value;
            return this;
        }

        /** The delay before the first retry, doubled for each one after. Default 200 ms. */
        public Builder baseDelay(Duration value) {
            this.baseDelay = requirePositive(value, "baseDelay");
            return this;
        }

        /** The ceiling on any single delay. Default 5 s. */
        public Builder maxDelay(Duration value) {
            this.maxDelay = requirePositive(value, "maxDelay");
            return this;
        }

        /**
         * Whether to retry a 504 as well as a 503. Default false.
         *
         * <p>When enabled it is still capped at ONE retry regardless of {@code maxRetries}, because
         * the server does not stop the work the deadline fired on: the timed-out match is still
         * running and still holding its admission permit, so a second copy raises the in-flight
         * count that produced the timeout. Enable it only if your fields-per-request is small
         * enough that a 504 means a transient spike rather than a request that is simply too big.
         */
        public Builder retryDeadlineExceeded(boolean value) {
            this.retryDeadlineExceeded = value;
            return this;
        }

        /** The jitter source. Injectable so a test can make the backoff deterministic. */
        public Builder random(Random value) {
            this.random = Objects.requireNonNull(value, "random");
            return this;
        }

        public ExponentialBackoffRetryPolicy build() {
            if (maxDelay.compareTo(baseDelay) < 0) {
                throw new IllegalArgumentException(
                        "maxDelay (" + maxDelay + ") must be >= baseDelay (" + baseDelay + ")");
            }
            return new ExponentialBackoffRetryPolicy(this);
        }

        private static Duration requirePositive(Duration value, String name) {
            Objects.requireNonNull(value, name);
            if (value.isNegative() || value.isZero()) {
                throw new IllegalArgumentException(name + " must be > 0, got " + value);
            }
            return value;
        }
    }
}
