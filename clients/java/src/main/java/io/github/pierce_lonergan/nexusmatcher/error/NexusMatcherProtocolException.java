package io.github.pierce_lonergan.nexusmatcher.error;

import java.io.Serial;
import java.util.Map;

/**
 * A response arrived and this client could not understand it.
 *
 * <p>Separate from {@link NexusMatcherServerException} because the diagnosis is different: the
 * server answered, so the fault is a contract mismatch rather than a failure. A decision value this
 * client does not know, a body that is not the published shape, or an error envelope that is not
 * the one envelope -- all of them mean the client and the service disagree about what they are
 * speaking, and the fix is a version, not a retry.
 *
 * <p>Loud rather than lenient, and only where being lenient would be dishonest. Unknown <em>keys</em>
 * are ignored everywhere in this client, because the service adds them additively and has twice in
 * this artifact's lifetime. Unknown <em>values</em> in a closed set are not: mapping a decision this
 * client has never seen onto the nearest one it has would silently change whether a classification
 * is applied without a human.
 */
public class NexusMatcherProtocolException extends NexusMatcherException {

    @Serial
    private static final long serialVersionUID = 1L;

    public NexusMatcherProtocolException(
            String message, int httpStatus, String requestId, Throwable cause) {
        super(message, httpStatus, null, Map.of(), requestId, cause);
    }
}
