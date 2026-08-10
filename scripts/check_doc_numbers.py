"""
scripts.check_doc_numbers | Layer: GATE
Check every documented number against the JSON artifact the document names.

The premise
-----------
This repository has extensive machinery gating CODE and, until this file, none gating
DOCUMENTED NUMBERS. Two false statements shipped to PyPI as the wheel's long_description
before anyone noticed, and neither was found by a test:

  * "rank-bm25 ... core dependencies now" -- moved out of core when the inverted-index
    rewrite stopped importing it; the pyproject comment was updated and the README was not
  * "Accuracy at 100k entries is unmeasured" -- it IS measured, by us, in
    exp_scale_combined.json: P@1 0.589 and 0.591 at 100,000 entries

The second one is the interesting shape. The README did not overstate; it UNDERSTATED our
own evidence. A human reviewer looking for exaggeration would have read straight past it.
Only a machine comparing the sentence to the file catches that direction.

The contract this enforces
--------------------------
README.md says, in its own words: "Every number in this README comes from a JSON artifact
in `benchmarks/results/`, named next to the number." CHANGELOG.md says the same. That is a
stated contract, and a stated contract is enforceable today without anybody rewriting a
document: find the artifact named in the section, find the numbers in that section, and
require the file to say what the sentence says.

WHAT THIS CANNOT DO -- read this before trusting a green run
-------------------------------------------------------------
Stated plainly, because a checker that implies total coverage is worse than none: it makes
the next person believe the numbers are guarded when most of them are not.

  1. **Prose claims with no number.** "RRF is the worst fusion method measured here",
     "static embeddings are a fallback, not a shortcut", "auto-approve is deliberately
     conservative". Nothing here reads English.
  2. **Numbers naming no artifact.** README's Encoders table (P@1 ~0.536 / 0.560 / 0.494,
     ~1240 / ~973 / ~71000 q/s) names none, so it is invisible to this file even though
     `exp_encoders_combined.json` exists. Same for "0.5276 at batch 8", "0.5581 against
     0.5596", and every number under a heading that names no artifact. Attribution is by
     SECTION: an artifact named anywhere between two headings covers that section and
     nothing else.
  3. **Claims about behaviour rather than measurement.** "rank-bm25 is a core dependency",
     "the default threshold is 0.87", "there is no HTTP matching endpoint". Those are
     checked against pyproject, the config dataclass and the router -- by other tests, or
     by nobody. Not here. The first of the two escaped defects above is in this class and
     would NOT have been caught by this file.
  4. **Derived quantities.** "+19.3 points of P@1", "costs 2.1 points", "~7x throughput",
     "93.6x on average latency". A delta is a claim about two measurements and an
     arithmetic step; only the `a -> b` form is checked (both endpoints must exist in the
     artifact), and only when a metric name sits on the same line.
  5. **Mappings.** The calibration artifact's `recommended` block maps a precision target
     to a threshold. A document can transcribe both numbers correctly and pair them wrongly;
     every value is present, so this file sees nothing.
  6. **Which record a prose number came from.** Table rows resolve to a specific record via
     their row label; a sentence does not. "86.3% precise" is checked against every
     `precision` in the artifact, so a number that is right for SOME threshold passes even
     when it is wrong for the one the sentence names.
  7. **Units and semantics.** A number matching `p_at_1` somewhere in the named artifact
     passes, even if the sentence meant a different split.
  8. **The difference between a stale number and a quoted one.** README says "An earlier
     revision reported P@1 0.715. That number was wrong and is retracted"; RESEARCH_ALIGNMENT
     tabulates "Semantic-only MRR 1.0000" precisely to retract it. Those read identically to
     a stale claim, so they are reported and recorded like one. Deciding which is which
     needs English, and this file does not read English.

The rule that keeps it honest: a claim is only ever reported as MISMATCH when the metric
key it names EXISTS in a named artifact and no value there matches. A key that is absent is
UNVERIFIABLE, counted and printed, never silently treated as agreement.

The ledger
----------
This gate went in with 141 mismatches already present, which is the finding, not a
failure of it. `docs/BENCHMARK_REGISTRY.md`, `docs/ENHANCEMENT_JOURNEY.md`,
`docs/PROJECT_STATE.md`, `docs/RESEARCH_ALIGNMENT.md`, `CHANGELOG.md` and four module docs
still carry the numbers from BEFORE the OMOP leakage fix -- the exact figures README's own
"Benchmark construction, and a leak we found in it" section retracts. Those documents are
not this lane's to edit, so every mismatch is RECORDED here rather than fixed, and listed
in the lane's follow-ups.

`KNOWN_MISMATCHES` is an EXACT set, not a threshold, and each entry pins BOTH sides -- the
claimed text and the artifact's value. So:

  * a new mismatch fails the gate (it is not in the ledger)
  * fixing a document fails the gate (its entry no longer reproduces -- delete the line)
  * re-running a benchmark fails the gate (the artifact side of every entry moves)

A count-based ratchet would have allowed a fix and a fresh regression to cancel out. This
cannot: both directions are failures, and the only way to change the set is to paste
`--report` output into a diff somebody reviews.

Usage
-----
    python scripts/check_doc_numbers.py            # gate: exit 1 on any unrecorded mismatch
    python scripts/check_doc_numbers.py --report   # ledger lines for pasting, after a fix
    python scripts/check_doc_numbers.py --verbose  # also list what could not be checked
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"


# =============================================================================
# ASCII OUTPUT
# =============================================================================
# NM-0001 was a CLI that died with UnicodeEncodeError on a default Windows console. This
# script quotes documents full of U+2192, U+2014 and U+00D7, so the same failure is one
# print away -- and it would abort the gate rather than report a finding.

_TRANSLITERATE = {
    "→": "->", "←": "<-", "—": "--", "–": "-", "≤": "<=",
    "≥": ">=", "×": "x", "‘": "'", "’": "'", "“": '"',
    "”": '"', "…": "...", "µ": "u", "±": "+/-", "≈": "~",
    "✓": "ok", "•": "*",
}


def ascii_only(text: str) -> str:
    """
    Text every code page in the matrix (cp437/cp850/cp1252) can encode.

    `backslashreplace` for the residue rather than `replace`: what reaches this path is
    quoted document content, and a `?` would misreport which character was there.
    """
    for source, replacement in _TRANSLITERATE.items():
        text = text.replace(source, replacement)
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _soften_encoding_errors() -> None:
    """Belt and braces for a stream this process did not open."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            reconfigure(errors="backslashreplace")


# =============================================================================
# METRIC VOCABULARY
# =============================================================================
# Each entry maps a name as a DOCUMENT writes it to the key(s) an ARTIFACT writes it as.
# Deliberately small: a vocabulary that guesses produces confident nonsense, and the
# fallback for an unrecognised name is "unverifiable", which is printed rather than hidden.

# Matched in order, first hit wins, so the specific forms precede the general ones:
# "Recall@10" must not be read as "Recall@1", and "Precision@5" must not be read as the
# bare "precision" of the auto-approved slice. Boundaries rather than anchors, because a
# row label is a sentence -- "End-to-end P@1 (793-pair labelled benchmark)".
_METRIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"(?<![\w@])(?:p|precision)\s*@\s*1(?!\d)", ("p_at_1", "precision_at_1")),
    (r"(?<![\w@])(?:p|precision)\s*@\s*5(?!\d)", ("p_at_5", "precision_at_5")),
    (r"(?<![\w@])(?:p|precision)\s*@\s*10(?!\d)", ("p_at_10", "precision_at_10")),
    (r"(?<![\w])mrr(?:\s*@\s*10)?(?!\d)", ("mrr_at_10", "mrr")),
    (r"(?<![\w@])r(?:ecall)?\s*@\s*1(?!\d)", ("recall.1", "recall_at_1")),
    (r"(?<![\w@])r(?:ecall)?\s*@\s*5(?!\d)", ("recall.5", "recall_at_5")),
    (r"(?<![\w@])r(?:ecall)?\s*@\s*10(?!\d)", ("recall.10", "recall_at_10")),
    (r"(?<![\w@])r(?:ecall)?\s*@\s*50(?!\d)", ("recall.50", "recall_at_50")),
    (r"(?<![\w])index build|(?<![\w])index time", ("index_seconds",)),
    (r"(?<![\w])n[ _]auto(?![\w])", ("n_auto",)),
    (r"(?<![\w])coverage(?![\w])", ("coverage",)),
    (r"(?<![\w])precision(?![\w])", ("precision", "auto_approve_precision")),
    (r"(?<![\w])ceiling(?![\w])", ("ceiling",)),
    (r"(?<![\w])decisions(?![\w])", ("decisions",)),
    (r"(?<![\w])n[ _]entries(?![\w])|(?<![\w])entries(?![\w])", ("n_entries", "n")),
    (r"(?<![\w])n(?![\w])", ("n",)),
)

# Units written in the CELL override the column header, because "Throughput" alone does
# not say whether the artifact calls it fields_per_sec or queries_per_sec.
_UNIT_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"fields\s*/\s*sec|fields per sec", ("fields_per_sec",)),
    (r"quer(?:y|ies)\s*/\s*sec|queries per sec|q/s", ("queries_per_sec",)),
    (r"entries\s*/\s*sec|entries per sec", ("index_entries_per_sec",)),
    (r"texts\s*/\s*s", ("texts_per_sec",)),
)

_METRIC_TOKEN = r"P@1|P@5|P@10|MRR@10|MRR|Recall@\d+|R@\d+|Precision@1"


def metric_keys(label: str) -> tuple[str, ...]:
    """Artifact keys a document's metric name could mean, or () if unrecognised."""
    normalised = _normalise_label(label)
    for pattern, keys in _METRIC_ALIASES:
        if re.search(pattern, normalised):
            return keys
    return ()


def _normalise_label(label: str) -> str:
    """Strip markdown emphasis, backticks and surrounding punctuation; lowercase."""
    text = label.replace("**", "").replace("*", "").replace("`", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip(" .:")


def _tokens(label: str) -> frozenset[str]:
    """Word/identifier tokens of a table row label or an artifact key, for selector matching."""
    text = _normalise_label(label)
    text = re.sub(r"\(.*?\)", " ", text)  # "(depth 10)", "(default)" -- annotations, not identity
    return frozenset(t for t in re.split(r"[^a-z0-9=@.+_-]+", text) if t and t not in {"-", "+"})


# =============================================================================
# ARTIFACTS
# =============================================================================


@dataclass(frozen=True)
class Artifact:
    name: str
    leaves: tuple[tuple[str, float], ...]  # dotted path -> numeric value
    records: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]  # path -> its scalar fields


def _walk(obj: object, prefix: str, leaves: list, records: list) -> None:
    if isinstance(obj, dict):
        scalars = tuple(
            (key, float(value))
            for key, value in obj.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        records.append((prefix, scalars))
        for key, value in obj.items():
            _walk(value, f"{prefix}.{key}" if prefix else str(key), leaves, records)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _walk(value, f"{prefix}[{index}]", leaves, records)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        leaves.append((prefix, float(obj)))


def load_artifacts() -> dict[str, Artifact]:
    """Every parseable JSON file in benchmarks/results/, flattened once."""
    artifacts: dict[str, Artifact] = {}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # BENCHMARK_REGISTRY.md documents two artifacts as truncated/invalid JSON. An
            # unreadable artifact is not this gate's finding, but it must not abort it.
            continue
        leaves: list[tuple[str, float]] = []
        records: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        _walk(data, "", leaves, records)
        artifacts[path.name] = Artifact(path.name, tuple(leaves), tuple(records))
    return artifacts


def _path_matches(path: str, key: str) -> bool:
    """A leaf path names `key` when the path ends with it at a segment boundary."""
    return path == key or path.endswith("." + key)


def _under(path: str, scope: str) -> bool:
    """
    True when `path` is inside `scope`, at a segment boundary.

    A plain `startswith` puts `recommended.0.95` inside `recommended.0.9`, which quietly
    widened the scope of a table row and reported the wrong artifact value as the one the
    document contradicted.
    """
    return path == scope or path.startswith(scope + ".") or path.startswith(scope + "[")


def _select_records(artifact: Artifact, selector: str) -> tuple[str, ...]:
    """
    Record paths a table row label points at, widest-first; () when nothing resolves.

    Two independent routes, unioned rather than ranked, because a union can only ever
    ACCEPT more values -- a wrong narrowing would invent a mismatch, and this file must
    not do that.
    """
    if not selector:
        return ()
    label = _tokens(selector)
    if not label:
        return ()
    chosen: list[str] = []
    numbers = {
        float(match.group()) for match in re.finditer(r"\d+(?:\.\d+)?", _normalise_label(selector))
    }
    for path, scalars in artifact.records:
        segment = re.sub(r"\[\d+\]$", "", path).rsplit(".", 1)[-1]
        key_tokens = _tokens(segment.replace("_", " ").replace(":", " "))
        by_name = bool(key_tokens) and key_tokens <= label
        by_value = bool(numbers) and any(value in numbers for _, value in scalars)
        if by_name or by_value:
            chosen.append(path)
    return tuple(sorted(set(chosen)))


def candidate_values(
    artifact: Artifact, keys: tuple[str, ...], selector: str
) -> tuple[tuple[str, float], ...]:
    """Every (path, value) in `artifact` that the claim's metric could be referring to."""
    scopes = _select_records(artifact, selector)
    for restrict in (True, False):
        hits = [
            (path, value)
            for path, value in artifact.leaves
            if any(_path_matches(path, key) for key in keys)
            and (not restrict or not scopes or any(_under(path, scope) for scope in scopes))
        ]
        if hits:
            return tuple(hits)
    return ()


# =============================================================================
# CLAIMS
# =============================================================================


@dataclass(frozen=True)
class Claim:
    doc: str
    line: int
    metric: str
    keys: tuple[str, ...]
    value: float
    tolerance: float
    claimed_text: str
    scopes: tuple[tuple[str, ...], ...]  # artifact sets to try, narrowest first
    selector: str = ""


# The lookbehind keeps "SUITE-006" and "GAP-008" from reading as the number 6 -- a run id
# is not a measurement, and one of them sat in the same table cell as a real throughput
# figure, so the checker reported the wrong number as the claim.
_NUMBER = r"~?\s*(?<![\w.-])\d(?:[\d,]*\d)?(?:\.\d+)?\s*%?"


def parse_number(text: str) -> tuple[float, float, str] | None:
    """
    (value, tolerance, as-written) for the first number in `text`, or None.

    The tolerance is derived from how precisely the number was WRITTEN, not invented: a
    claim quoted to three decimals must be a correct rounding of the artifact, so half an
    ulp of its own last digit. "~" and "about" widen it to 5% relative, which is what a
    tilde means and no more. Nothing here is a fudge factor that can be loosened later to
    make a document pass -- loosening it requires editing this docstring's premise.
    """
    match = re.search(_NUMBER, text)
    if match is None:
        return None
    raw = match.group().strip()
    approximate = "~" in raw or re.search(r"\b(about|roughly|approx)\b", text, re.IGNORECASE)
    digits = raw.replace("~", "").replace(",", "").replace("%", "").strip()
    try:
        value = float(digits)
    except ValueError:
        return None
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    if "%" in raw:
        value /= 100.0
        decimals += 2
    tolerance = 0.5 * 10.0 ** (-decimals)
    if approximate:
        tolerance = max(tolerance, abs(value) * 0.05)
    return value, tolerance, ascii_only(raw)


def _unit_claim(cell: str) -> tuple[tuple[str, ...], str] | None:
    """
    (keys, the number written next to the unit) for a rate written in `cell`.

    The number has to be the one ADJACENT to the unit, not the first in the cell: one
    QUALITY_GATES cell reads "SUITE-006 has never run. End-to-end throughput is 652
    fields/sec", and taking the first number reported the claim as "006".
    """
    for pattern, keys in _UNIT_KEYS:
        match = re.search(rf"(?P<value>{_NUMBER})\s*(?:{pattern})", cell, re.IGNORECASE)
        if match:
            return keys, match.group("value")
    return None


def _unit_keys(cell: str) -> tuple[str, ...]:
    found = _unit_claim(cell)
    return found[0] if found else ()


# =============================================================================
# DOCUMENT SCAN
# =============================================================================

_ARTIFACT_REF = re.compile(r"(?:benchmarks/results/)?([A-Za-z0-9_.*-]+\.json)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _cells(line: str) -> list[str]:
    match = _TABLE_ROW.match(line)
    return [cell.strip() for cell in match.group(1).split("|")] if match else []


def _mentions(lines: list[str], known: dict[str, Artifact]) -> tuple[list[list[str]], list]:
    """Artifacts named on each line, plus references to files that do not exist."""
    named: list[list[str]] = []
    broken: list[tuple[int, str]] = []
    for offset, line in enumerate(lines):
        here: list[str] = []
        for raw in _ARTIFACT_REF.findall(line):
            if "*" in raw:
                here.extend(sorted(path.name for path in RESULTS.glob(raw)))
            elif raw in known:
                here.append(raw)
            elif "benchmarks/results/" in line and not (RESULTS / raw).exists():
                broken.append((offset + 1, raw))
        named.append(sorted(set(here)))
    return named, broken


def _spans(lines: list[str], is_boundary) -> list[int]:
    """Index of the span each line belongs to, spans delimited by `is_boundary`."""
    span: list[int] = []
    current = 0
    for line in lines:
        if is_boundary(line):
            current += 1
        span.append(current)
    return span


def scope_tiers(lines: list[str], known: dict[str, Artifact]) -> tuple[list[tuple], list]:
    """
    For every line, the artifact sets to try IN ORDER, narrowest first.

    Attribution is the whole difficulty. These documents name the artifact three different
    ways and a single rule gets one of them wrong:

      1. **same paragraph** -- CHANGELOG writes "...42.7% coverage. Artifact:
         `exp_calibration_combined.json`." with the numbers BEFORE the filename, so any
         forward-only rule checks them against the previous bullet's artifact
      2. **nearest naming line above, within the section** -- README and BENCHMARK_REGISTRY
         write "Artifact: X" and then a table; the table's own paragraph names nothing
      3. **every artifact named in the section** -- a bullet under "## Limitations" that
         names none of its own still sits in a section that does

    Narrowest-first matters in both directions. Tier 1 alone would leave tables unchecked.
    Tier 3 alone would let a per-split number pass because some OTHER split's artifact
    happens to carry that value -- the exact blindness H-004 is about. So a tier is used
    only when it actually carries the metric, and the first one that does wins; a wider
    tier is reached only when the narrower one cannot speak to the claim at all.
    """
    named, broken = _mentions(lines, known)
    section = _spans(lines, lambda line: bool(_HEADING.match(line)))
    paragraph = _spans(lines, lambda line: not line.strip())

    by_paragraph: dict[int, list[str]] = {}
    by_section: dict[int, list[str]] = {}
    for index, here in enumerate(named):
        by_paragraph.setdefault(paragraph[index], []).extend(here)
        by_section.setdefault(section[index], []).extend(here)

    tiers: list[tuple] = []
    for index in range(len(lines)):
        above: list[str] = []
        for previous in range(index, -1, -1):
            if named[previous] and section[previous] == section[index]:
                above = named[previous]
                break
        candidates = [
            tuple(sorted(set(by_paragraph.get(paragraph[index], [])))),
            tuple(sorted(set(above))),
            tuple(sorted(set(by_section.get(section[index], [])))),
        ]
        ordered: list[tuple] = []
        for tier in candidates:
            if tier and tier not in ordered:
                ordered.append(tier)
        tiers.append(tuple(ordered))
    return tiers, broken


def _column_artifact(base: str, header: str, known: dict[str, Artifact]) -> str:
    """
    `eval_pipeline_combined.json` + a column headed "bird" -> `eval_pipeline_bird.json`.

    Without this a per-split table is checked against the wrong split, which reports a
    mismatch for numbers that are correct -- and a gate that cries wolf gets an exemption
    written for it, which is how a ratchet dies.
    """
    stem = base[: -len(".json")]
    if "_" not in stem:
        return ""
    sibling = f"{stem.rsplit('_', 1)[0]}_{_normalise_label(header)}.json"
    return sibling if sibling in known and sibling != base else ""


def _table_claims(
    doc: str, lines: list[str], start: int, tiers: list[tuple], known: dict[str, Artifact]
) -> tuple[list[Claim], int]:
    """Claims from the markdown table beginning at `start`; also returns the line after it."""
    header = _cells(lines[start])
    end = start + 2
    while end < len(lines) and _TABLE_ROW.match(lines[end]):
        end += 1
    scopes = tiers[start]
    if not scopes:
        return [], end

    # `| | |` -- PROJECT_STATE's headline table has no header text at all, and its first
    # column is the metric name exactly as a "| Metric | Value |" table's is.
    metric_first = _normalise_label(header[0]) in {"metric", "measure", "measurement"} or not any(
        cell.strip() for cell in header
    )
    claims: list[Claim] = []
    for row_index in range(start + 2, end):
        row = _cells(lines[row_index])
        if len(row) < 2:
            continue
        for column in range(1, min(len(row), len(header))):
            label = row[0] if metric_first else header[column]
            selector = "" if metric_first else row[0]
            scoped = scopes
            if metric_first:
                sibling = _column_artifact(scopes[0][0], header[column], known)
                if sibling:
                    scoped = ((sibling,),)
            claims.extend(_cell_claims(doc, row_index + 1, label, row[column], scoped, selector))
    return claims, end


def _cell_claims(
    doc: str, line: int, label: str, cell: str, scopes: tuple[tuple[str, ...], ...], selector: str
) -> list[Claim]:
    """One table cell -> zero or more claims. Handles the two compound cells in these docs."""
    normalised = _normalise_label(label)

    if normalised == "decisions":
        # "AUTO_APPROVE 421, REVIEW 372, REJECT 0"
        return [
            Claim(doc, line, f"Decisions/{name}", (f"decisions.{name}",), float(count), 0.5,
                  count, scopes, selector)
            for name, count in re.findall(r"([A-Z_]{3,})\s+(\d+)", cell)
        ]

    slots = re.findall(r"@(\d+)", label)
    if len(slots) > 1 and "recall" in normalised:
        # "Recall@1 / @5 / @10" against "0.6999 / 0.8878 / 0.9193"
        parts = [part for part in cell.split("/") if re.search(r"\d", part)]
        claims = []
        # Lenient pairing on purpose: a cell with fewer numbers than the label has slots
        # still gets its leading values checked. Refusing the whole row would trade a
        # partial check for none, which is the wrong direction for a gate.
        for slot, part in zip(slots, parts, strict=False):
            parsed = parse_number(part)
            if parsed:
                value, tolerance, text = parsed
                claims.append(
                    Claim(doc, line, f"Recall@{slot}", (f"recall.{slot}", f"recall_at_{slot}"),
                          value, tolerance, text, scopes, selector)
                )
        return claims

    unit = _unit_claim(cell)
    # A unit written in the cell names the metric better than the column header does:
    # QUALITY_GATES puts a throughput figure in a column headed "Validation".
    keys, raw, metric = (
        (unit[0], unit[1], unit[0][0]) if unit else (metric_keys(label), cell, normalised)
    )
    if not keys:
        return []
    parsed = parse_number(raw)
    if parsed is None:
        return []
    value, tolerance, text = parsed
    return [Claim(doc, line, metric or "value", keys, value, tolerance, text, scopes, selector)]


_PROSE_METRIC = re.compile(
    rf"(?P<metric>{_METRIC_TOKEN})\s*\**\s*(?:=|:|is|of|at|->|→)?\s*\**\s*(?P<value>{_NUMBER})",
)
_PROSE_UNIT = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<unit>fields\s*/\s*sec|quer(?:y|ies)\s*/\s*sec|"
    rf"entries\s*/\s*sec)",
    re.IGNORECASE,
)
_PROSE_N = re.compile(r"\bn\s*=\s*(?P<value>\d[\d,]*)")
_PROSE_ENTRIES = re.compile(r"(?P<value>\d[\d,]*)\s+(?:\w+\s+){0,2}entries\b")
_PROSE_INDEX = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*s\s+index build")
_PROSE_PERCENT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?%)\s*(?P<what>precise|precision|coverage)", re.IGNORECASE
)
_PROSE_PAIR = re.compile(r"(?P<a>\d+\.\d+)\s*(?:->|→)\s*(?P<b>\d+\.\d+)")


def _prose_claims(doc: str, line_no: int, line: str, scopes: tuple[tuple[str, ...], ...]) -> list[Claim]:
    """Claims from a line of prose. Sparse by design -- see limitation 4 in the docstring."""
    claims: list[Claim] = []

    def add(metric: str, keys: tuple[str, ...], raw: str) -> None:
        parsed = parse_number(raw)
        if parsed and keys:
            value, tolerance, text = parsed
            claims.append(Claim(doc, line_no, metric, keys, value, tolerance, text, scopes))

    for match in _PROSE_METRIC.finditer(line):
        add(match.group("metric"), metric_keys(match.group("metric")), match.group("value"))
    for match in _PROSE_UNIT.finditer(line):
        add(match.group("unit"), _unit_keys(match.group("unit")), match.group("value"))
    for match in _PROSE_N.finditer(line):
        add("n", ("n",), match.group("value"))
    for match in _PROSE_ENTRIES.finditer(line):
        add("entries", ("n_entries", "n"), match.group("value"))
    for match in _PROSE_INDEX.finditer(line):
        add("index build", ("index_seconds",), match.group("value") + " s")
    for match in _PROSE_PERCENT.finditer(line):
        what = match.group("what").lower()
        keys = ("coverage",) if what == "coverage" else ("precision", "auto_approve_precision")
        add(what, keys, match.group("value"))
    return claims


def _pair_claims(doc: str, line_no: int, line: str, scopes: tuple[tuple[str, ...], ...]) -> list[Claim]:
    """
    "0.491 -> 0.691" asserts two measured values, and this is the single most-repeated
    stale pair in the repository. Only counted when a metric name sits on the same line,
    because "68.9 ms -> 3.2 ms" is an unarchived micro-benchmark, not an artifact claim.
    """
    if not re.search(_METRIC_TOKEN, line):
        return []
    claims: list[Claim] = []
    for match in _PROSE_PAIR.finditer(line):
        for side in ("a", "b"):
            parsed = parse_number(match.group(side))
            if parsed:
                value, tolerance, text = parsed
                claims.append(
                    Claim(doc, line_no, "value in a -> pair", ("p_at_1", "p_at_5", "mrr_at_10",
                          "recall.10", "recall_at_10", "precision", "coverage"),
                          value, tolerance, text, scopes)
                )
    return claims


def scan(doc: str, text: str, known: dict[str, Artifact]) -> tuple[list[Claim], list]:
    lines = text.splitlines()
    tiers, broken = scope_tiers(lines, known)
    claims: list[Claim] = []
    index = 0
    while index < len(lines):
        if (
            _TABLE_ROW.match(lines[index])
            and index + 1 < len(lines)
            and _TABLE_RULE.match(lines[index + 1])
        ):
            table, index = _table_claims(doc, lines, index, tiers, known)
            claims.extend(table)
            continue
        scopes = tiers[index]
        if scopes and not _TABLE_ROW.match(lines[index]):
            claims.extend(_prose_claims(doc, index + 1, lines[index], scopes))
            claims.extend(_pair_claims(doc, index + 1, lines[index], scopes))
        index += 1
    return claims, [(doc, line, name) for line, name in broken]


# =============================================================================
# VERDICTS
# =============================================================================


@dataclass(frozen=True)
class Finding:
    kind: str  # MISMATCH | BROKEN_REF
    doc: str
    line: int
    metric: str
    claimed: str
    actual: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        """
        The ledger identity of a finding. Deliberately EXCLUDES the line number.

        Line numbers were in this key for one afternoon. Another lane appended 44 lines to
        CHANGELOG.md and the gate reported eleven fixes and eleven regressions, none of
        which had happened. A ratchet that fires on line-shift churn is a ratchet somebody
        writes an exemption for, and then it guards nothing.

        The cost, stated rather than hidden: two identical wrong claims in the same
        document collapse to one entry, so fixing only one of them will not turn the gate
        red. The claimed value and the artifact value are both still pinned, which is what
        catches an actual change of substance.
        """
        return (self.doc, self.metric, self.claimed, self.actual)

    def render(self) -> str:
        return ascii_only(
            f"{self.doc}:{self.line}  {self.kind}  {self.metric} claimed {self.claimed}"
            f"  |  artifact says {self.actual}"
        )


def _format(value: float) -> str:
    return f"{value:.6g}"


def verify(claim: Claim, known: dict[str, Artifact]) -> tuple[str, Finding | None]:
    """
    (verdict, finding). UNVERIFIABLE when no named artifact carries the metric at all.

    Tiers are tried narrowest-first and the first one that CARRIES the metric decides. A
    wider tier is never consulted afterwards, so a value that is right for another split
    cannot rescue a claim the nearest artifact contradicts.
    """
    candidates: list[tuple[str, str, float]] = []
    for scope in claim.scopes:
        for name in scope:
            artifact = known.get(name)
            if artifact is None:
                continue
            for path, value in candidate_values(artifact, claim.keys, claim.selector):
                candidates.append((name, path, value))
        if candidates:
            break
    if not candidates:
        return "UNVERIFIABLE", None
    if any(abs(value - claim.value) <= claim.tolerance for _, _, value in candidates):
        return "OK", None
    name, path, value = min(
        candidates, key=lambda item: (abs(item[2] - claim.value), item[0], item[1])
    )
    return "MISMATCH", Finding(
        "MISMATCH", claim.doc, claim.line, claim.metric, claim.claimed_text,
        f"{name} {path}={_format(value)}",
    )


# =============================================================================
# THE LEDGER
# =============================================================================
# Every mismatch present on 2026-08-10, recorded because these documents belong to other
# lanes. Both sides are pinned: the claimed text AND the artifact's value. Re-running a
# benchmark therefore breaks every entry for that artifact, which is correct -- a stale
# doc measured against a new artifact is a different, unreviewed claim.
#
# (document, metric, claimed as written, artifact name + path = value)
# No line number, on purpose -- see Finding.key.

KNOWN_MISMATCHES: frozenset[tuple[str, str, str, str]] = frozenset(
    {
        ("CHANGELOG.md", "MRR@10", "0.781", "eval_pipeline_combined.json mrr_at_10=0.685252"),
        ("CHANGELOG.md", "P@5", "0.888", "eval_pipeline_combined.json p_at_5=0.825581"),
        ("CHANGELOG.md", "Recall@10", "0.919", "eval_pipeline_combined.json recall.10=0.877907"),
        ("CHANGELOG.md", "coverage", "42.7%", "exp_calibration_combined.json curve[30].coverage=0.412791"),
        ("CHANGELOG.md", "index build", "1.76", "eval_pipeline_combined.json index_seconds=2.06657"),
        ("CHANGELOG.md", "p@1", "0.490", "eval_pipeline_bird.json p_at_1=0.601108"),
        ("CHANGELOG.md", "p@1", "0.700", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("CHANGELOG.md", "p@1", "0.819", "eval_pipeline_omop.json p_at_1=0.574924"),
        ("CHANGELOG.md", "precise", "86.3%", "exp_calibration_combined.json curve[31].precision=0.860902"),
        ("CHANGELOG.md", "precise", "94.7%", "exp_calibration_combined.json curve[37].precision=0.952941"),
        ("CHANGELOG.md", "value in a -> pair", "0.491", "exp_query_repr_combined.json variants.abbrev.mrr_at_10=0.482244"),
        ("README.md", "P@1", "0.046", "eval_pipeline_omop.json p_at_1=0.574924"),
        ("README.md", "P@1", "0.598", "exp_scale_combined.json runs[1].p_at_1=0.591425"),
        ("README.md", "P@1", "0.715", "eval_pipeline_bird.json p_at_1=0.601108"),
        ("README.md", "index build (688 entries)", "~1.8", "eval_pipeline_combined.json index_seconds=2.06657"),
        ("README.md", "p@1", "0.361", "exp_query_repr_combined.json variants.raw.p_at_1=0.360465"),
        ("README.md", "p@1", "0.377", "exp_query_repr_combined.json variants.abbrev.p_at_1=0.376453"),
        ("docs/BENCHMARK_REGISTRY.md", "Decisions/AUTO_APPROVE", "421", "eval_pipeline_combined.json decisions.AUTO_APPROVE=85"),
        ("docs/BENCHMARK_REGISTRY.md", "Decisions/REVIEW", "372", "eval_pipeline_combined.json decisions.REVIEW=603"),
        ("docs/BENCHMARK_REGISTRY.md", "P@1", "0.490", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/BENCHMARK_REGISTRY.md", "P@1", "0.700", "suite_008_combined_20251209_165753.json baseline.precision_at_1=1"),
        ("docs/BENCHMARK_REGISTRY.md", "P@1", "0.819", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/BENCHMARK_REGISTRY.md", "Recall@1", "0.6999", "eval_pipeline_combined.json recall.1=0.581395"),
        ("docs/BENCHMARK_REGISTRY.md", "Recall@10", "0.9193", "eval_pipeline_combined.json recall.10=0.877907"),
        ("docs/BENCHMARK_REGISTRY.md", "Recall@5", "0.8878", "eval_pipeline_combined.json recall.5=0.825581"),
        ("docs/BENCHMARK_REGISTRY.md", "auto-approve precision at that operating point", "0.9264", "eval_pipeline_combined.json auto_approve_precision=0.952941"),
        ("docs/BENCHMARK_REGISTRY.md", "auto-approve precision", "0.8559", "exp_calibration_combined.json curve[25].precision=0.751445"),
        ("docs/BENCHMARK_REGISTRY.md", "auto-approve precision", "0.9007", "exp_calibration_combined.json curve[29].precision=0.793548"),
        ("docs/BENCHMARK_REGISTRY.md", "auto-approve precision", "0.9264", "exp_calibration_combined.json curve[30].precision=0.852113"),
        ("docs/BENCHMARK_REGISTRY.md", "auto-approve precision", "0.9469", "exp_calibration_combined.json curve[35].precision=0.916084"),
        ("docs/BENCHMARK_REGISTRY.md", "auto-approve precision", "0.9586", "exp_calibration_combined.json curve[36].precision=0.938776"),
        ("docs/BENCHMARK_REGISTRY.md", "coverage", "39.6%", "exp_calibration_combined.json curve[36].coverage=0.142442"),
        ("docs/BENCHMARK_REGISTRY.md", "coverage", "42.7%", "exp_calibration_combined.json curve[35].coverage=0.207849"),
        ("docs/BENCHMARK_REGISTRY.md", "coverage", "53.1%", "exp_calibration_combined.json curve[30].coverage=0.412791"),
        ("docs/BENCHMARK_REGISTRY.md", "coverage", "54.6%", "exp_calibration_combined.json curve[29].coverage=0.450581"),
        ("docs/BENCHMARK_REGISTRY.md", "coverage", "59.5%", "exp_calibration_combined.json curve[25].coverage=0.502907"),
        ("docs/BENCHMARK_REGISTRY.md", "fields_per_sec", "652.3", "eval_pipeline_combined.json fields_per_sec=364.183"),
        ("docs/BENCHMARK_REGISTRY.md", "index build (793 entries)", "1.756", "eval_pipeline_combined.json index_seconds=2.06657"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.5713", "exp_query_repr_combined.json variants.type.mrr_at_10=0.446409"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.5896", "exp_query_repr_combined.json variants.raw.mrr_at_10=0.461424"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.5963", "exp_query_repr_combined.json variants.underscores.mrr_at_10=0.465176"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.5995", "exp_query_repr_combined.json variants.split.mrr_at_10=0.472121"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.6022", "exp_query_repr_combined.json variants.abbrev.mrr_at_10=0.482244"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.7530", "exp_query_repr_combined.json variants.full.mrr_at_10=0.647513"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.7602", "exp_query_repr_combined.json variants.context.mrr_at_10=0.66743"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.7706", "exp_query_repr_combined.json variants.context.mrr_at_10=0.66743"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.7706", "exp_rerank_combined.json first_stage.mrr_at_10=0.66743"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.7814", "eval_pipeline_combined.json mrr_at_10=0.685252"),
        ("docs/BENCHMARK_REGISTRY.md", "mrr@10", "0.8089", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.mrr_at_10=0.662994"),
        ("docs/BENCHMARK_REGISTRY.md", "n auto", "314", "exp_calibration_combined.json curve[36].n_auto=98"),
        ("docs/BENCHMARK_REGISTRY.md", "n auto", "339", "exp_calibration_combined.json curve[35].n_auto=143"),
        ("docs/BENCHMARK_REGISTRY.md", "n auto", "421", "exp_calibration_combined.json curve[30].n_auto=284"),
        ("docs/BENCHMARK_REGISTRY.md", "n auto", "433", "exp_calibration_combined.json curve[29].n_auto=310"),
        ("docs/BENCHMARK_REGISTRY.md", "n auto", "472", "exp_calibration_combined.json curve[25].n_auto=346"),
        ("docs/BENCHMARK_REGISTRY.md", "n", "793", "exp_calibration_combined.json n=688"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.4691", "exp_query_repr_combined.json variants.type.p_at_1=0.353198"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.4880", "exp_query_repr_combined.json variants.raw.p_at_1=0.360465"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.4905", "exp_query_repr_combined.json variants.underscores.p_at_1=0.366279"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.4931", "exp_query_repr_combined.json variants.split.p_at_1=0.373547"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.4956", "exp_query_repr_combined.json variants.abbrev.p_at_1=0.376453"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.6671", "exp_query_repr_combined.json variants.full.p_at_1=0.530523"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.6709", "exp_query_repr_combined.json variants.context.p_at_1=0.559593"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.6910", "exp_query_repr_combined.json variants.context.p_at_1=0.559593"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.6910", "exp_rerank_combined.json first_stage.p_at_1=0.559593"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.6999", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/BENCHMARK_REGISTRY.md", "p@1", "0.7465", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.p_at_1=0.546512"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.7087", "exp_query_repr_combined.json variants.type.p_at_5=0.575581"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.7352", "exp_query_repr_combined.json variants.raw.p_at_5=0.604651"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.7402", "exp_query_repr_combined.json variants.underscores.p_at_5=0.606105"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.7427", "exp_query_repr_combined.json variants.split.p_at_5=0.610465"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.7503", "exp_query_repr_combined.json variants.abbrev.p_at_5=0.630814"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.8588", "exp_query_repr_combined.json variants.full.p_at_5=0.803779"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.8726", "exp_query_repr_combined.json variants.context.p_at_5=0.809593"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.8726", "exp_rerank_combined.json first_stage.p_at_5=0.809593"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.8764", "exp_query_repr_combined.json variants.full_no_type.p_at_5=0.827035"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.8878", "eval_pipeline_combined.json p_at_5=0.825581"),
        ("docs/BENCHMARK_REGISTRY.md", "p@5", "0.8916", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.p_at_5=0.824128"),
        ("docs/BENCHMARK_REGISTRY.md", "queries_per_sec", "18.1", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.queries_per_sec=87.327"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.8953", "exp_query_repr_combined.json variants.type.recall.50=0.80814"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.9016", "exp_query_repr_combined.json variants.raw.recall.50=0.837209"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.9029", "exp_query_repr_combined.json variants.abbrev.recall.50=0.859012"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.9029", "exp_query_repr_combined.json variants.underscores.recall.50=0.838663"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.9042", "exp_query_repr_combined.json variants.split.recall.50=0.837209"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.9533", "exp_query_repr_combined.json variants.full.recall.50=0.938953"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.9596", "exp_query_repr_combined.json variants.context.recall.50=0.949128"),
        ("docs/BENCHMARK_REGISTRY.md", "r@50", "0.9609", "exp_query_repr_combined.json variants.full_no_type.recall.50=0.952035"),
        ("docs/ENHANCEMENT_JOURNEY.md", "coverage", "39.6%", "exp_calibration_combined.json curve[36].coverage=0.142442"),
        ("docs/ENHANCEMENT_JOURNEY.md", "coverage", "42.7%", "exp_calibration_combined.json curve[35].coverage=0.207849"),
        ("docs/ENHANCEMENT_JOURNEY.md", "coverage", "53.1%", "exp_calibration_combined.json curve[30].coverage=0.412791"),
        ("docs/ENHANCEMENT_JOURNEY.md", "coverage", "59.0%", "exp_calibration_combined.json curve[25].coverage=0.502907"),
        ("docs/ENHANCEMENT_JOURNEY.md", "fields_per_sec", "652", "eval_pipeline_combined.json fields_per_sec=364.183"),
        ("docs/ENHANCEMENT_JOURNEY.md", "mrr@10", "0.771", "exp_rerank_combined.json first_stage.mrr_at_10=0.66743"),
        ("docs/ENHANCEMENT_JOURNEY.md", "mrr@10", "0.781", "eval_pipeline_combined.json mrr_at_10=0.685252"),
        ("docs/ENHANCEMENT_JOURNEY.md", "mrr@10", "0.809", "exp_rerank_combined.json first_stage.mrr_at_10=0.66743"),
        ("docs/ENHANCEMENT_JOURNEY.md", "n", "793", "exp_calibration_combined.json n=688"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.469", "exp_query_repr_combined.json variants.type.p_at_1=0.353198"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.488", "exp_query_repr_combined.json variants.full.p_at_1=0.530523"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.490", "eval_pipeline_bird.json p_at_1=0.601108"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.491", "exp_query_repr_combined.json variants.split.p_at_1=0.373547"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.493", "exp_query_repr_combined.json variants.split.p_at_1=0.373547"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.496", "exp_query_repr_combined.json variants.full.p_at_1=0.530523"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.691", "exp_query_repr_combined.json variants.context.p_at_1=0.559593"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.691", "exp_rerank_combined.json first_stage.p_at_1=0.559593"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.700", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.747", "exp_rerank_combined.json first_stage.p_at_1=0.559593"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@1", "0.819", "eval_pipeline_omop.json p_at_1=0.574924"),
        ("docs/ENHANCEMENT_JOURNEY.md", "p@5", "0.888", "eval_pipeline_combined.json p_at_5=0.825581"),
        ("docs/ENHANCEMENT_JOURNEY.md", "precision on auto-approved", "0.863", "exp_calibration_combined.json curve[25].precision=0.751445"),
        ("docs/ENHANCEMENT_JOURNEY.md", "precision on auto-approved", "0.926", "exp_calibration_combined.json curve[30].precision=0.852113"),
        ("docs/ENHANCEMENT_JOURNEY.md", "precision on auto-approved", "0.947", "exp_calibration_combined.json curve[35].precision=0.916084"),
        ("docs/ENHANCEMENT_JOURNEY.md", "precision on auto-approved", "0.959", "exp_calibration_combined.json curve[36].precision=0.938776"),
        ("docs/ENHANCEMENT_JOURNEY.md", "queries_per_sec", "18.1", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.queries_per_sec=87.327"),
        ("docs/ENHANCEMENT_JOURNEY.md", "recall@10", "0.919", "eval_pipeline_combined.json recall.10=0.877907"),
        ("docs/PROJECT_STATE.md", "auto-approve precision at default threshold", "0.947", "eval_pipeline_combined.json auto_approve_precision=0.952941"),
        ("docs/PROJECT_STATE.md", "end-to-end p@1 (793-pair labelled benchmark)", "0.700", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/PROJECT_STATE.md", "fields_per_sec", "652", "eval_pipeline_combined.json fields_per_sec=364.183"),
        ("docs/PROJECT_STATE.md", "p@1 on the abbreviation-heavy split (bird)", "0.490", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/PROJECT_STATE.md", "p@1 on the descriptive split (omop)", "0.819", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/QUALITY_GATES.md", "fields_per_sec", "652", "eval_pipeline_combined.json fields_per_sec=364.183"),
        ("docs/RESEARCH_ALIGNMENT.md", "MRR", "1.0000", "eval_pipeline_combined.json mrr_at_10=0.685252"),
        ("docs/RESEARCH_ALIGNMENT.md", "MRR@10", "0.781", "eval_pipeline_combined.json mrr_at_10=0.685252"),
        ("docs/RESEARCH_ALIGNMENT.md", "P@1", "0.700", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/RESEARCH_ALIGNMENT.md", "P@5", "0.888", "eval_pipeline_combined.json p_at_5=0.825581"),
        ("docs/RESEARCH_ALIGNMENT.md", "Recall@10", "0.919", "eval_pipeline_combined.json recall.10=0.877907"),
        ("docs/RESEARCH_ALIGNMENT.md", "precision", "100%", "eval_pipeline_combined.json auto_approve_precision=0.952941"),
        ("docs/RESEARCH_ALIGNMENT.md", "value in a -> pair", "0.491", "eval_pipeline_combined.json p_at_1=0.581395"),
        ("docs/RESEARCH_ALIGNMENT.md", "value in a -> pair", "0.691", "eval_pipeline_combined.json mrr_at_10=0.685252"),
        ("docs/modules/abbreviation_expansion.md", "P@1", "0.309", "exp_query_repr_combined.json variants.type.p_at_1=0.353198"),
        ("docs/modules/abbreviation_expansion.md", "value in a -> pair", "0.488", "exp_query_repr_combined.json variants.abbrev.mrr_at_10=0.482244"),
        ("docs/modules/abbreviation_expansion.md", "value in a -> pair", "0.496", "exp_query_repr_combined.json variants.abbrev.mrr_at_10=0.482244"),
        ("docs/modules/context_enricher.md", "p@1", "0.469", "exp_query_repr_combined.json variants.type.p_at_1=0.353198"),
        ("docs/modules/context_enricher.md", "p@1", "0.488", "exp_query_repr_combined.json variants.full.p_at_1=0.530523"),
        ("docs/modules/context_enricher.md", "p@1", "0.493", "exp_query_repr_combined.json variants.full.p_at_1=0.530523"),
        ("docs/modules/context_enricher.md", "p@1", "0.691", "exp_query_repr_combined.json variants.context.p_at_1=0.559593"),
        ("docs/modules/context_enricher.md", "value in a -> pair", "0.491", "exp_query_repr_combined.json variants.abbrev.mrr_at_10=0.482244"),
        ("docs/modules/context_enricher.md", "value in a -> pair", "0.691", "exp_query_repr_combined.json variants.split.recall.10=0.678779"),
        ("docs/modules/cross_encoder_reranker.md", "MRR@10", "0.771", "exp_rerank_combined.json first_stage.mrr_at_10=0.66743"),
        ("docs/modules/cross_encoder_reranker.md", "P@1", "0.691", "exp_rerank_combined.json first_stage.p_at_1=0.559593"),
        ("docs/modules/cross_encoder_reranker.md", "mrr@10", "0.809", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.mrr_at_10=0.662994"),
        ("docs/modules/cross_encoder_reranker.md", "p@1", "0.747", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.p_at_1=0.546512"),
        ("docs/modules/cross_encoder_reranker.md", "queries_per_sec", "18.1", "exp_rerank_combined.json rerankers.cross-encoder/ms-marco-MiniLM-L-6-v2.queries_per_sec=87.327"),
    }
)


def documents() -> list[Path]:
    """README.md, CHANGELOG.md and every doc under docs/, in a stable order."""
    found = [REPO / "README.md", REPO / "CHANGELOG.md"]
    found.extend(sorted((REPO / "docs").rglob("*.md")))
    return [path for path in found if path.exists()]


def run() -> tuple[list[Finding], list[Claim], dict[str, int]]:
    known = load_artifacts()
    findings: list[Finding] = []
    unverifiable: list[Claim] = []
    counts = {"OK": 0, "MISMATCH": 0, "UNVERIFIABLE": 0, "BROKEN_REF": 0, "documents": 0}
    for path in documents():
        doc = path.relative_to(REPO).as_posix()
        counts["documents"] += 1
        claims, broken = scan(doc, path.read_text(encoding="utf-8"), known)
        for _, line, name in broken:
            counts["BROKEN_REF"] += 1
            findings.append(
                Finding("BROKEN_REF", doc, line, "artifact reference", name,
                        "no such file in benchmarks/results/")
            )
        for claim in claims:
            verdict, finding = verify(claim, known)
            counts[verdict] += 1
            if finding is not None:
                findings.append(finding)
            elif verdict == "UNVERIFIABLE":
                unverifiable.append(claim)
    findings.sort(key=lambda f: (f.doc, f.line, f.metric, f.claimed))
    return findings, unverifiable, counts


def main() -> int:
    _soften_encoding_errors()
    parser = argparse.ArgumentParser(description="Check documented numbers against artifacts.")
    parser.add_argument("--report", action="store_true", help="print ledger lines and exit 0")
    parser.add_argument("-v", "--verbose", action="store_true", help="list unverifiable claims")
    args = parser.parse_args()

    findings, unverifiable, counts = run()

    if args.report:
        for finding in findings:
            print(
                f'        ("{finding.doc}", "{ascii_only(finding.metric)}", '
                f'"{finding.claimed}", "{ascii_only(finding.actual)}"),'
            )
        print(f"\n  {len(findings)} findings", file=sys.stderr)
        return 0

    print(f"\nDoc numbers: {counts['documents']} documents, {counts['OK']} claims verified")
    print("=" * 78)
    unrecorded = [f for f in findings if f.key not in KNOWN_MISMATCHES]
    reproduced = {f.key for f in findings}
    stale = sorted(KNOWN_MISMATCHES - reproduced)

    for finding in unrecorded:
        print("  NEW   " + finding.render())
    for entry in stale:
        print(f"  FIXED {entry[0]}  {entry[1]} claimed {entry[2]} -- now correct")

    print("=" * 78)
    print(
        f"  {counts['OK']} verified, {counts['MISMATCH']} mismatches "
        f"({len(KNOWN_MISMATCHES)} recorded), {counts['UNVERIFIABLE']} unverifiable, "
        f"{counts['BROKEN_REF']} broken references"
    )
    if args.verbose:
        for claim in unverifiable:
            print(
                ascii_only(
                    f"  ?     {claim.doc}:{claim.line}  {claim.metric} {claim.claimed_text}"
                    f"  -- no named artifact carries {'/'.join(claim.keys)}"
                )
            )

    if unrecorded or stale:
        print(
            "\nA documented number disagrees with the artifact it names.\n"
            "  NEW   -- fix the document, or record it with `--report` if it is not yours\n"
            "  FIXED -- the document is now right; delete that line from KNOWN_MISMATCHES"
        )
        return 1
    # Deliberately not "every documented number matches": 141 of them do not, and saying
    # otherwise here is the same overstatement this file exists to catch.
    print(
        f"\n  no unrecorded disagreement; {len(KNOWN_MISMATCHES)} known mismatches still "
        "stand, all reproduced exactly"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
