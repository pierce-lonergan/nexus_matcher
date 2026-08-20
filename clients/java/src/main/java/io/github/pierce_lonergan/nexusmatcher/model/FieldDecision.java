package io.github.pierce_lonergan.nexusmatcher.model;

/**
 * The ONE verdict for one field. {@link MatchDecision} is per CANDIDATE, and they are not the same
 * question.
 *
 * <p>Read this, not {@code results[path].get(0).decision()}. Rolling a per-candidate verdict up
 * into a per-column one is a rule, and a rule every client reconstructs for itself is a rule nobody
 * wrote down; the server publishes the roll-up so there is one of them.
 *
 * <h2>This enum is OPEN and {@link MatchDecision} is CLOSED. That asymmetry is the point.</h2>
 *
 * <p>An unknown value on {@link MatchDecision} is a hard decode failure, deliberately. An unknown
 * value here becomes {@link #UNKNOWN} and the response still decodes. Three reasons, and the first
 * is the one that settles it:
 *
 * <p><strong>1. The server has committed to freezing one of these vocabularies and to growing the
 * other.</strong> {@code NO_MATCH} is not on {@code decision}, and the service's own model
 * documentation says why: widening an enum a client has already generated a closed Java
 * {@code enum} from turns an ordinary 200 into a deserialisation failure, so a NEW field carrying a
 * WIDER vocabulary was created instead. This is that field. It exists <em>because</em> vocabularies
 * grow, and it was born by growing. Binding it closed here would re-create, on the new field, the
 * exact break the new field was invented to avoid -- and would hand back the entire cost the server
 * paid to introduce it.
 *
 * <p><strong>2. The blast radius is different by a factor of the batch size.</strong> A
 * {@code decision} sits inside one candidate of one field. A field decision sits in a map with one
 * entry per field in the request, and this client's batch route sends up to 250 of them. Refusing
 * the whole body over one unrecognised verdict would discard 249 verdicts that decoded perfectly,
 * on a bulk run somebody is waiting on. An operator who can read 249 answers and must escalate one
 * is strictly better off than an operator who can read none.
 *
 * <p><strong>3. Degrading here is not a guess, because {@link #UNKNOWN} is not usable as an
 * answer.</strong> The failure worth fearing is a client that maps a value it does not understand
 * onto the nearest one it does -- which is how a new "APPROVE_WITH_CONDITIONS" silently becomes an
 * auto-approval. Nothing here does that. {@link #UNKNOWN} is not a verdict, it is the absence of
 * one: {@link #maySafelyInherit()} answers false for it, {@link FieldVerdict#wireValue()} hands back
 * the exact string the server sent so the value can be named in a ticket rather than merely
 * counted, and {@link FieldVerdict#isKnown()} is the single test for "did this client understand
 * the answer".
 *
 * <p>The narrow reading of "unknown values must fail loudly" is that the failure must be loud. It
 * is: the value is preserved, it is visibly not one of the four, and the one question a caller asks
 * of it answers no. What that reading actually forbids is a SILENT reinterpretation, and there is
 * none here.
 *
 * @see FieldVerdict the value type {@code fieldDecisions} decodes to, which carries the raw string
 */
public enum FieldDecision {

    /** The server would apply rank 1's governance to this field without a human. */
    AUTO_APPROVE,

    /** A human must decide this field. Never read it as "probably fine". */
    REVIEW,

    /** The server would not apply rank 1's governance to this field. */
    REJECT,

    /**
     * This response carries nothing this field may inherit from.
     *
     * <p><strong>The candidates are still there.</strong> A NO_MATCH field comes back with its
     * candidate list populated exactly as any other field does -- the server chose that over an
     * empty list on purpose, because the candidates are evidence for the reviewer who now has to
     * decide, and an empty list would have thrown that evidence away. So do not test
     * {@code candidatesFor(path).isEmpty()} to find no-match fields, and above all do not read
     * {@code results[path].get(0).governance()} on one: rank 1 may well carry a populated
     * protection class, and applying it is precisely the mistake this verdict exists to prevent.
     * {@link MatchResponse#inheritableGovernanceFor(String)} is the accessor that gets this right.
     *
     * <p>Earned two ways: the field came back with no candidates at all, or the server has an
     * absolute-score floor configured ({@code scoring.absoluteScoreFloor}) and rank 1 does not
     * clear it. The second requires the deployment to have chosen a floor -- the library ships
     * none -- so on a stock server this verdict means the first.
     */
    NO_MATCH,

    /**
     * A verdict a newer server sent that this client does not know. <strong>Never on the wire.</strong>
     *
     * <p>Not an answer. It means the server decided something and this build cannot say what, so the
     * field needs a human and this client needs an upgrade. {@link FieldVerdict#wireValue()} on the
     * verdict that produced it names the value the server actually sent.
     *
     * <p>It is deliberately not published by the service, and
     * {@code tests/packaging/test_java_client_contract.py} asserts that it is not: the moment a real
     * server starts sending the string {@code UNKNOWN}, this constant would stop meaning
     * "unrecognised" and start silently absorbing a real verdict.
     */
    UNKNOWN;

    /**
     * Whether this verdict permits inheriting rank 1's protection class.
     *
     * <p>True for {@link #AUTO_APPROVE} alone. A reading of the published contract, not a second
     * opinion about it: {@link #REVIEW} and {@link #REJECT} both mean a human has not agreed yet,
     * {@link #NO_MATCH} means there is nothing to inherit however confident the candidates look,
     * and {@link #UNKNOWN} means this client could not read the answer. Every one of those is a
     * "no", and the four are not interchangeable -- use the constant itself when the caller needs
     * to know WHICH no.
     */
    public boolean maySafelyInherit() {
        return this == AUTO_APPROVE;
    }

    /** Whether this client understood the server's verdict. False only for {@link #UNKNOWN}. */
    public boolean isKnown() {
        return this != UNKNOWN;
    }
}
