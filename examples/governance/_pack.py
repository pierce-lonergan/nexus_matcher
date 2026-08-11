"""
examples.governance._pack | Layer: EXAMPLE
Shared plumbing for the five commands. Not part of the library; copy what you need.

Everything user-visible in this pack goes through `say()`. That is not tidiness, it is
the encoding contract: this repository has already shipped a CLI that raised
`'charmap' codec can't encode character` on a console using a legacy Windows code page,
so the pack is pure ASCII, uses no box-drawing and no spinner, and behaves identically
when stdout is a pipe. Ordering is sorted or file order everywhere, never dict-iteration
order over a set, so two runs of the same command produce the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACK = Path(__file__).resolve().parent

VOCABULARY_FILE = PACK / "protection_classes.json"
GLOSSARY_FILE = PACK / "glossary.csv"
INVALID_GLOSSARY_FILE = PACK / "glossary_invalid.csv"
FIELDS_FILE = PACK / "fields.json"
LABELS_FILE = PACK / "labels.jsonl"
FEEDBACK_FILE = PACK / "feedback.jsonl"
RESULTS_FILE = PACK / "out" / "results.json"

BANNER = (
    "Gravel Bay Ferry Authority -- FICTIONAL example data. "
    "Neither the organisation nor this taxonomy exists."
)


def say(*parts: object) -> None:
    """Print one ASCII line. See the module docstring for why this exists."""
    print(" ".join(str(p) for p in parts))


def rule(title: str = "") -> None:
    say("-" * 78)
    if title:
        say(title)
        say("-" * 78)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, reporting the LINE NUMBER of a bad record.

    `json.loads` on a whole file says "line 1 column 4013"; on a 42-line labels file the
    only thing the reader needs is which record is malformed.
    """
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name} line {number}: {exc}") from exc
    return records


def read_fields() -> list[dict[str, Any]]:
    """The pack's schema fields, in file order."""
    document = json.loads(FIELDS_FILE.read_text(encoding="utf-8"))
    return list(document["fields"])


def load_vocabulary():
    """The caller-supplied controlled vocabulary. Nothing in the library defines one."""
    from nexus_matcher.domain.governance import GovernanceVocabulary

    return GovernanceVocabulary.from_json(VOCABULARY_FILE)


def build_matcher():
    """
    A matcher wired with the bundled offline encoder, the pack's vocabulary, and its
    glossary.

    `governance=` is the whole point: without it the matcher holds no vocabulary, every
    `governance_code` resolves to nothing, and every match comes back carrying no class.
    """
    from nexus_matcher.application.use_cases.match_schema import NexusMatcher

    matcher = NexusMatcher.from_config(governance=VOCABULARY_FILE)
    matcher.load_dictionary(GLOSSARY_FILE)
    return matcher


def governed_entries(vocabulary) -> dict[str, Any]:
    """
    The glossary read THROUGH the vocabulary: `{id: DictionaryEntry}`.

    `governance_strict` is left at its default, so a glossary with a defective row
    refuses to load here rather than producing a partial index. That is the whole point
    of validating: the rows that would vanish are exactly the rows whose governance is
    wrong.
    """
    from nexus_matcher.application import ingest

    entries = ingest.load_entries(GLOSSARY_FILE, governance=vocabulary)
    return {entry.id: entry for entry in entries}


def resolve_governance(match: Any, entries: dict[str, Any], vocabulary) -> tuple[str, Any, str]:
    """
    The governance a matched field inherits: `(governance_id, ProtectionClass|None, source)`.

    `MatchResult.governance_id` and `MatchResult.governance` are the contract, and they
    are what an adopter should read. This helper reads them, and then CHECKS them against
    the same glossary loaded through the same vocabulary by `ingest.load_entries`. The
    two must agree.

    They do not agree today, and the disagreement is silent, which is why this is not a
    one-liner. `matcher.load_dictionary()` goes through
    `BaseDictionaryLoader._convert_row`, which constructs its `DictionaryEntry` without
    reading the protection-code column at all -- so every indexed entry carries
    `governance_code=None`, every match comes back with `governance=None`, and a field
    the glossary marks SEALED_RESTRICTED and a direct identifier is auto-approved
    carrying no class. `None` there is indistinguishable from "this entry genuinely has
    no class", which is exactly why nothing notices. That is NM-0005's failure -- a field
    silently losing its classification -- in the documented loading path.

    So: prefer what the match carries, fall back to the caller-side join when the match
    carries nothing and the glossary says otherwise, and report WHICH in the returned
    `source` and in results.json. An audit artifact that does not say where its
    governance came from is not an audit artifact.

    A REJECTED RANK 1 is respected in both directions: `MatchResult.__post_init__` clears
    the class on a rejected top match, and so does the fallback, or the pack would attach
    a class to the very match the matcher refused.

    `rank == 1` is load-bearing in that check. The domain model used to clear the class on
    a reject at ANY rank and no longer does, because a field inherits from rank 1 only and
    blanking the runner-ups deleted the comparison a reviewer decides on. A fallback still
    clearing them would make results.json disagree with the service it exists to audit --
    the one thing an audit artifact must never do.
    """
    entry_id = match.dictionary_entry.id
    promoted_id = getattr(match, "governance_id", None) or entry_id
    carried = getattr(match, "governance", None)
    if carried is not None:
        return promoted_id, carried, "match_result"

    if match.rank == 1 and str(getattr(match.decision, "value", match.decision)) == "REJECT":
        return promoted_id, None, "match_result"

    code = getattr(entries.get(entry_id), "governance_code", None)
    from_glossary = vocabulary.get(code)
    if from_glossary is None:
        # Both say "no class", and the glossary agrees: this is an uncoded entry, and an
        # uncoded entry confers nothing. Nothing is wrong here.
        return promoted_id, None, "match_result"

    return promoted_id, from_glossary, "caller_side_join"


def describe(protection_class: Any) -> str:
    """One ASCII line for a ProtectionClass, or for the absence of one."""
    if protection_class is None:
        return "no class (open tier, inherits nothing)"
    flags = []
    if protection_class.personal_information:
        flags.append("personal-information")
    if protection_class.direct_identifier:
        flags.append("DIRECT-IDENTIFIER")
    if protection_class.enhancement:
        flags.append(f"enhancement={protection_class.enhancement}")
    suffix = ("  [" + ", ".join(flags) + "]") if flags else ""
    return f"{protection_class.code} -> {protection_class.classification}{suffix}"
