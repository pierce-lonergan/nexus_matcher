package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.Map;

/**
 * No response was received at all: connection refused, socket reset, local read timeout, or the
 * calling thread interrupted.
 *
 * <p>{@link #httpStatus()} is 0 and {@link #errorCode()} is empty, because neither exists -- there
 * was no envelope. The {@code X-Request-ID} is still populated with the id the client SENT, which
 * is the point of minting it client-side: a request that never got an answer can still be found in
 * the server's log if it got that far.
 *
 * <p>Not retried automatically. The client cannot tell a connection refused (safe to retry) from a
 * response lost after the server did the work (not obviously safe), and this endpoint's one
 * side-effecting route appends to an audit trail -- so retrying a lost feedback POST could file a
 * verdict twice. A caller who knows their own idempotency can retry.
 *
 * <p>One case worth naming, because it looks like a bug and is not: the server refuses an oversized
 * body <em>while it is still on the wire</em> and closes the connection without draining it. A
 * client still writing that body may see the reset before it reads the 413. If you get this
 * exception while sending a very large batch, you have hit the byte cap -- send fewer fields.
 */
public class NexusMatcherTransportException extends NexusMatcherException {

    @Serial
    private static final long serialVersionUID = 1L;

    public NexusMatcherTransportException(String message, String requestId, Throwable cause) {
        super(message, 0, null, Map.of(), requestId, cause);
    }
}
