"""
examples.governance.validate | Layer: EXAMPLE (command 1 of 5)
Load the vocabulary, prove the glossary is clean, prove the broken glossary is refused.

Run this BEFORE indexing anything. A glossary row whose stated tier contradicts its own
protection code is a data defect, and indexing it lets a field inherit a tier its own
code disowns -- the failure this library exists to prevent.

    python examples/governance/validate.py

Exit code 0 means: the vocabulary loaded, all 30 glossary rows are clean, and every
deliberately-broken row in glossary_invalid.csv was refused for the reason the pack
expects. Anything else exits 1 and says which check failed.

## What each assertion here can catch

Written down because a check nobody has watched fail is a hypothesis, not a check.
Each of these has been observed going red against a mutation of the pack's own data:

  * a valid row reporting a problem            -> mutate glossary.csv's classification
  * a broken row reporting NO problem          -> delete a row from glossary_invalid.csv
  * a broken row reporting the WRONG problem   -> swap two expected_problem values
  * a legacy token stored raw                  -> point the alias at a different code
  * strict loading accepting a defective file  -> the final check below
"""

from __future__ import annotations

from _pack import (  # the pack is run as a script, so its own directory is on sys.path
    BANNER,
    GLOSSARY_FILE,
    INVALID_GLOSSARY_FILE,
    VOCABULARY_FILE,
    describe,
    load_vocabulary,
    rule,
    say,
)

# The row in glossary.csv that carries a LEGACY token rather than a current code, and the
# code the vocabulary maps it onto. Pinned by value: the point of declaring an alias is
# that the entry ends up carrying the CANONICAL code, never the token the source file
# happened to use, and a test that only checked "it loaded" would not notice the raw
# token being stored.
LEGACY_ROW_ID = "GBF-0001"
LEGACY_ROW_CANONICAL_CODE = "MANIFEST_NAME"

# The row whose code the vocabulary declares to be droppable noise. It must end up with
# NO code -- not with the noise token, and not with a guess.
DROPPED_ROW_ID = "GBF-0029"


def _show_vocabulary(vocabulary) -> None:
    """Section 1: what the caller's file declares."""
    say(f"file: {VOCABULARY_FILE.name}")
    say(f"codes declared: {len(vocabulary.codes)}")
    for code in sorted(vocabulary.codes):
        say(f"  {describe(vocabulary.get(code))}")
    say(f"a field with no code inherits: {vocabulary.classification_for(None)!r}")
    say("(the library ships no taxonomy of its own -- every code above came from that file)")


def _check_glossary(ingest, vocabulary) -> int:
    """Section 2: every row of the good glossary is clean, and the two special rows behave."""
    rows, header = ingest.read_source(GLOSSARY_FILE)
    say(f"file: {GLOSSARY_FILE.name}  rows: {len(rows)}  columns: {len(header)}")

    failures: list[str] = []
    for number, row in enumerate(rows, start=1):
        problems = vocabulary.problems_with(row)
        if problems:
            failures.append(f"row {number} of {GLOSSARY_FILE.name}: {'; '.join(problems)}")
    if failures:
        say("UNEXPECTED PROBLEMS in a glossary the pack claims is clean:")
        for line in failures:
            say(f"  {line}")
        return 1
    say(f"all {len(rows)} rows valid")

    entries = {e.id: e for e in ingest.load_entries(GLOSSARY_FILE, governance=vocabulary)}
    say(f"loaded {len(entries)} entries with governance attached")

    legacy = entries[LEGACY_ROW_ID]
    if legacy.governance_code != LEGACY_ROW_CANONICAL_CODE:
        say(
            f"FAIL {LEGACY_ROW_ID} carries governance_code {legacy.governance_code!r}; the "
            f"declared alias should have canonicalised it to {LEGACY_ROW_CANONICAL_CODE!r}. "
            f"A legacy token must be mapped or dropped, never stored raw."
        )
        return 1
    say(
        f"  {LEGACY_ROW_ID}: source says "
        f"{legacy.source_metadata.get('governance_code_raw')!r} -> stored as "
        f"{legacy.governance_code!r} (declared alias)"
    )

    dropped = entries[DROPPED_ROW_ID]
    if dropped.governance_code is not None:
        say(
            f"FAIL {DROPPED_ROW_ID} carries governance_code {dropped.governance_code!r}; the "
            f"vocabulary declares that token to be noise, so the entry must carry no code."
        )
        return 1
    say(
        f"  {DROPPED_ROW_ID}: source says "
        f"{dropped.source_metadata.get('governance_code_raw')!r} -> stored as None "
        f"(declared noise, dropped)"
    )
    return 0


def _check_broken_rows(ingest, vocabulary) -> int:
    """Section 3: every deliberately-broken row is refused, for the reason it declares."""
    bad_rows, _ = ingest.read_source(INVALID_GLOSSARY_FILE)
    say(f"file: {INVALID_GLOSSARY_FILE.name}  rows: {len(bad_rows)}")
    if not bad_rows:
        say("FAIL the broken glossary is empty, so this section asserts nothing.")
        return 1

    for number, row in enumerate(bad_rows, start=1):
        # `expected_problem` is a column of the PACK, not of a real glossary: it is the
        # substring the pack asserts appears in the loader's complaint. It couples this
        # example to the wording of `GovernanceVocabulary.problems_with`, deliberately --
        # a rejection whose reason changed silently is a rejection nobody can act on.
        expected = str(row.get("expected_problem", "")).strip()
        problems = vocabulary.problems_with(row)
        label = f"row {number} ({row.get('id')})"
        if not expected:
            say(f"FAIL {label} declares no expected_problem, so it asserts nothing.")
            return 1
        if not problems:
            say(f"FAIL {label} was ACCEPTED. It should have been refused: {expected}")
            return 1
        if not any(expected in problem for problem in problems):
            say(f"FAIL {label} was refused for the wrong reason.")
            say(f"     expected to contain: {expected}")
            for problem in problems:
                say(f"     got:                 {problem}")
            return 1
        say(f"  {label} refused: {problems[0]}")

    say(f"all {len(bad_rows)} broken rows refused, each for its stated reason")
    return 0


def _check_strict_load_refuses(ingest, vocabulary) -> int:
    """Section 4: a strict load of the broken file refuses the WHOLE file."""
    try:
        ingest.load_entries(INVALID_GLOSSARY_FILE, governance=vocabulary)
    except ValueError as exc:
        say(f"  refused, as it must: {str(exc).splitlines()[0]}")
        return 0
    # Returning the good rows would be the worse outcome: the rows that vanished are
    # exactly the rows whose governance was wrong, so the caller indexes a glossary that
    # looks healthy and inherits nothing where they should have inherited a class.
    say("FAIL the broken glossary loaded. A partial glossary is worse than no glossary.")
    return 1


def main() -> int:
    from nexus_matcher.application import ingest

    say(BANNER)
    vocabulary = load_vocabulary()

    rule("1. the vocabulary the CALLER supplies")
    _show_vocabulary(vocabulary)

    rule("2. the glossary, row by row, against that vocabulary")
    if _check_glossary(ingest, vocabulary):
        return 1

    rule("3. the broken glossary: every row refused, for the stated reason")
    if _check_broken_rows(ingest, vocabulary):
        return 1

    rule("4. a strict load of the broken glossary REFUSES the whole file")
    if _check_strict_load_refuses(ingest, vocabulary):
        return 1

    rule()
    say("VALIDATE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
