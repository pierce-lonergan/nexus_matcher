"""
tests.meta.test_no_confidential_terms | Layer: META-GATE
Confidential vocabulary must not reach the repository, the sdist or the wheel.

The evidence
------------
This library was written inside a bank. 2.0.0 and 2.0.1 both shipped to PyPI with

    # <BUSINESS-UNIT> (<Business Unit Expanded>) domain hierarchy

sitting above DEFAULT_HIERARCHY_DATA in domain/services/domain_hierarchy.py -- inside
the wheel, not just the repo -- naming the employer and the business unit whose data
model the default hierarchy was sketched from. Two more copies were in the module docs
and four in an integration test.

Nobody reviewed it in, and nothing could have caught it. The rule existed, but it
existed as a sentence in a task description and a grep somebody ran by hand once. It was
scoped to the vocabulary that was on that person's mind that day, so a two-year-old
comment fell outside it. A rule enforced by remembering to run a regex is not a rule; it
is a habit, and habits do not survive a deadline.

So the rule is a test, it runs on every commit, and it cannot be satisfied by intending
to be careful.

Why the blocklist is hashed
---------------------------
Writing the forbidden terms into the gate would publish them -- in git history, in the
sdist and on PyPI -- which is the exact outcome the gate exists to prevent. So the terms
live as keyed digests in confidential_term_hashes.txt. See that file for the algorithm
and for an honest statement of what hashing does and does not buy.

Why candidates are token runs, not substrings
---------------------------------------------
The first cut of this scanner squashed each line to alphanumerics and hashed every
substring. That flags the ordinary English word "pointless", which happens to contain a
blocked four-letter code spanning its middle syllables -- a boundary this sentence will
not spell out, for the obvious reason. Matching runs of whole tokens instead means a term
is found however it is spelled -- UPPER_SNAKE column name, hyphenated header, camelCase
identifier, dotted namespace, or quoted in prose -- and not when its letters happen to
fall inside an ordinary English word. test_gate_catches_every_spelling pins that, using
the canary.

Why this file is not exempt from itself
---------------------------------------
It was, for about thirty minutes, and that was long enough to leak. The first version of
this module carried a blanket SKIP_PATHS entry for itself, reasoning that a file
explaining the gate must be able to discuss the terms. It then used four real ones as
worked examples in these docstrings -- invisible to the gate, by the gate's own
exemption, in the file whose entire purpose is to stop exactly that. The sdist scan
caught it.

So there is no blanket exemption now. Two files may contain the CANARY and nothing else,
which is the narrowest rule that lets the gate be tested at all. Every real term is
scanned for here as everywhere else, and worked examples use invented vocabulary. An
exemption wide enough to be convenient is wide enough to hide what it was written to
catch.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HASHES = Path(__file__).with_name("confidential_term_hashes.txt")

SALT = b"nexus-matcher/confidential-terms/v1"
MAX_RUN = 4  # longest run of tokens joined into one candidate

# A public, meaningless string whose digest is in the blocklist, so the scanner can be
# shown to fire without this file naming anything real.
CANARY = "zzqx-canary-confidential-gate"

# Binary and model files. The int8 bge-small tokenizer ships the standard 30k-piece
# BERT-uncased wordpiece vocabulary, which is simply a list of common English -- several
# entries of which collide with blocked terms. These are upstream artefacts of a public
# model, byte-identical to what HuggingFace serves; they reference nothing.
SKIP_SUFFIXES = (
    ".onnx",
    ".bin",
    ".safetensors",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pkl",
    ".zip",
    ".whl",
    ".gz",
    ".parquet",
    ".xlsx",
)
SKIP_NAMES = ("tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")

# The only exemption, and it is not a path exemption -- it is a single-term one. These
# two files must name the CANARY: one defines it, the other plants it to prove the gate
# fires. Every other blocked term is scanned for in them exactly as in any other file, so
# neither can become a place where a real term hides. See the module docstring for what
# happened when this was a blanket path skip instead.
CANARY_ONLY_PATHS = {
    "tests/meta/test_no_confidential_terms.py",
    "tests/museum/NM-0029/replay.py",
}

# Digests only, enforced by test_blocklist_contains_no_plaintext. Scanning it would
# match nothing anyway -- but it is scanned, so that remains true by observation.


_ATOM = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")


def _digest(canon: str) -> str:
    return hashlib.blake2b(canon.encode("utf-8"), key=SALT, digest_size=8).hexdigest()


def _blocklist() -> set[str]:
    lines = HASHES.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def _atoms(line: str) -> list[str]:
    """Split a line into word atoms, breaking camelCase and every separator.

    "x-vendor-dataelem" -> [x, vendor, dataelem]
    "HTTPResponseCode"  -> [HTTP, Response, Code]
    "com.example.acme"  -> [com, example, acme]
    """
    return _ATOM.findall(line)


def _candidates(line: str) -> set[str]:
    """Every run of 1..MAX_RUN consecutive atoms, joined and canonicalised.

    Runs of two or more atoms require every atom to be at least two characters.
    Single letters carry no lexical signal, and juxtaposing them manufactures
    terms nobody wrote: the tuple ("c", "b", "a") in a test's ordering fixture
    yields the atoms c, b, a -- and the run c+c+b across two such tuples spelled a
    blocked three-letter code. A term that is genuinely present is written as a
    word or joined by separators inside one, never assembled from three separate
    single-character identifiers.
    """
    atoms = [a.lower() for a in _atoms(line)]
    out: set[str] = set()
    for i, atom in enumerate(atoms):
        out.add(atom)
        if len(atom) < 2:
            continue
        joined = atom
        for j in range(i + 1, min(i + MAX_RUN, len(atoms))):
            if len(atoms[j]) < 2:
                break
            joined += atoms[j]
            out.add(joined)
    return out


def scan_text(text: str, blocked: set[str]) -> list[tuple[int, str]]:
    """Return (line_number, digest) for each hit. The term itself is never returned."""
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for cand in _candidates(line):
            d = _digest(cand)
            if d in blocked:
                hits.append((n, d))
    return hits


def _git(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args, "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [r for r in out.split("\0") if r]


def _tracked_files() -> list[str]:
    """Tracked files PLUS untracked-but-not-ignored ones.

    The first version listed only tracked files, and that is a hole big enough to
    drive the original defect through. `git ls-files` cannot see a file that has
    not been added yet -- and a brand-new file is exactly where a term arrives.
    The gate would go green on a working tree containing a leak, stay green until
    the author committed, and only then have anything to find, by which point the
    term is in history where removing it means a rewrite.

    This was not hypothetical either. It hid a blocked code in this module's own
    docstring; only the sdist scan, which packages the working tree rather than
    the index, saw it.

    Ignored files are excluded on purpose: build outputs and virtualenvs are not
    authored content, and they are not distributed from here.
    """
    rels = [*_git("ls-files"), *_git("ls-files", "--others", "--exclude-standard")]
    keep = []
    for rel in dict.fromkeys(rels):  # de-dupe, preserve order
        p = Path(rel)
        if p.name in SKIP_NAMES or p.suffix.lower() in SKIP_SUFFIXES:
            continue
        if not (REPO / rel).is_file():  # staged deletion
            continue
        keep.append(rel)
    return keep


def _blocklist_for(rel: str) -> set[str]:
    """The blocklist to apply to one file: everything, minus the canary where allowed."""
    blocked = _blocklist()
    if rel in CANARY_ONLY_PATHS:
        return blocked - {_digest(_canon(CANARY))}
    return blocked


def _canon(term: str) -> str:
    return "".join(ch for ch in term.lower() if ch.isalnum())


# =============================================================================
# THE GATE
# =============================================================================


def test_no_confidential_terms_in_tracked_files():
    """No tracked file may contain a blocked term.

    The failure message gives file, line and digest -- never the term. Somebody
    debugging a hit can see exactly which line to look at, and the message itself
    stays safe to paste into an issue or a CI log.
    """
    assert _blocklist(), "blocklist is empty -- the gate would pass vacuously"

    findings: list[str] = []
    for rel in _tracked_files():
        try:
            text = (REPO / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, digest in scan_text(text, _blocklist_for(rel)):
            findings.append(f"  {rel}:{line_no}  (digest {digest})")

    assert not findings, (
        f"{len(findings)} confidential term(s) found in tracked files.\n"
        + "\n".join(findings)
        + "\n\nThe term is deliberately not printed. Open the line to see it.\n"
        "Do NOT silence this by exempting the path. Rewrite the line to use invented\n"
        "vocabulary -- a worked example does not need a real term to be a good example."
    )


# =============================================================================
# THE GATE ON THE GATE
# =============================================================================


def test_the_gate_actually_fires():
    """Prove the scanner detects a planted term, end to end.

    Without this, every part of the gate could be broken -- wrong salt, empty
    blocklist, tokenizer that never matches -- and it would still report a clean
    tree, forever, exactly like the hand-run grep it replaces.
    """
    blocked = _blocklist()
    planted = f"# see the {CANARY} for details\n"

    hits = scan_text(planted, blocked)

    assert hits == [(1, _digest("zzqxcanaryconfidentialgate"))], (
        "the canary was not detected -- the scanner is not working"
    )


@pytest.mark.parametrize(
    "written_as",
    [
        "zzqx-canary-confidential-gate",
        "zzqx_canary_confidential_gate",
        "ZZQX_CANARY_CONFIDENTIAL_GATE",
        "zzqxCanaryConfidentialGate",
        "ZzqxCanaryConfidentialGate",
        "com.example.zzqx.canary.confidential.gate",
        "zzqx canary confidential gate",
        '"zzqx-canary-confidential-gate"',
    ],
)
def test_gate_catches_every_spelling(written_as):
    """A term is a term however it is cased, joined or quoted.

    The leak that prompted this gate was in a comment; the same words could as
    easily arrive as an identifier, a namespace or a JSON key. Catching only the
    spelling that leaked last time is how a gate goes quietly toothless.
    """
    assert scan_text(written_as, _blocklist()), f"missed spelling: {written_as}"


def test_gate_does_not_fire_on_ordinary_prose():
    """Guard the false-positive direction.

    A gate that flags ordinary English gets muted, and a muted gate protects
    nothing. "pointless" contains a blocked code across a token boundary and is the
    reason this scanner matches token runs rather than raw substrings.
    """
    prose = (
        "This optimisation was pointless.\n"
        "Chasing the reference through the call graph.\n"
        "The interval is internal to the intlike accumulator.\n"
    )
    assert not scan_text(prose, _blocklist())


def test_gate_does_not_fire_on_juxtaposed_single_letters():
    """Single-character identifiers must not be joined into terms.

    A real hit, from tests/properties/test_metamorphic.py: two ordering tuples
    written as ("a", "b", "c") and ("c", "b", "a") put the atoms c, c, b next to
    each other, spelling a blocked three-letter code that nobody wrote. Left
    unfixed, the first thing anyone would do is delete the gate.
    """
    ordering_fixture = '        for order in (("a", "b", "c"), ("c", "b", "a")):\n'
    assert not scan_text(ordering_fixture, _blocklist())


@pytest.mark.parametrize("rel", sorted(CANARY_ONLY_PATHS))
def test_exempt_files_are_exempt_for_the_canary_only(rel):
    """An exempt file must still be scanned for every real term.

    This is the check that the original blanket exemption did not have, and its
    absence is what let four real terms sit in this module's own docstrings. The
    exemption must subtract exactly one digest -- the canary -- and leave every
    other one in force. Widening it to a whole file reopens the hole.
    """
    full = _blocklist()
    applied = _blocklist_for(rel)
    missing = full - applied

    assert missing == {_digest(_canon(CANARY))}, (
        f"{rel} is exempt from {len(missing)} term(s); it may be exempt from the canary "
        "and nothing else"
    )
    assert len(applied) == len(full) - 1


def test_scan_covers_untracked_files(tmp_path):
    """A file that is not yet `git add`ed must still be scanned.

    Pinned because the omission is invisible: the gate goes green, and it is green
    for the one reason you would never suspect -- it never looked. The window
    between writing a file and committing it is precisely when a pasted term is
    present and least examined.
    """
    scratch = REPO / "_gate_untracked_probe.py"
    scratch.write_text(f"# {CANARY}\n", encoding="utf-8")
    try:
        assert scratch.relative_to(REPO).as_posix() in _tracked_files(), (
            "an untracked file is invisible to the gate"
        )
    finally:
        scratch.unlink()


def test_no_path_is_exempt_from_everything():
    """There is no blanket path skip, and adding one should be deliberate.

    SKIP_NAMES and SKIP_SUFFIXES exclude binaries and vendored model vocabularies,
    which contain no prose anyone wrote. Nothing else is skipped wholesale -- a
    source file that is inconvenient to scan is precisely the kind of file a term
    ends up in.
    """
    assert "SKIP_PATHS" not in globals(), (
        "a blanket path exemption is back; use CANARY_ONLY_PATHS instead"
    )


def test_blocklist_contains_no_plaintext():
    """The blocklist must hold digests and comments -- never a term.

    Somebody adding a term by hand, in a hurry, will paste the word. This makes
    that mistake fail loudly instead of silently publishing what it was meant to
    hide.
    """
    for n, line in enumerate(HASHES.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        assert re.fullmatch(r"[0-9a-f]{16}", s), (
            f"{HASHES.name}:{n} is not a bare digest -- plaintext must never be stored here"
        )
