"""
benchmarks.synthetic.verify | Layer: BENCHMARK
The generator held to its own claims.

A generator that documents a distribution and does not produce it is worse than one that
documents nothing, because every number measured on its output inherits the error and
nothing in the output says so. So each property the pack advertises is checked against the
generated rows here, and `verify()` returns findings rather than printing reassurance.

Two checks earn their place beyond the obvious distribution assertions:

  LEAK  the token overlap between a query and the business name of its gold entry. This
        repository has already published an inflated benchmark whose gold labels were
        derived from the field names, making half the corpus a string-identity task; the
        giveaway was that nobody had measured the overlap. It is measured here, printed
        as a distribution, and flagged when the share of exact-identity pairs is high
        enough to make the corpus degenerate.

  DETERMINISM  the same spec generated twice, compared by checksum. Every claim the
        manifest makes about the corpus is worthless if the corpus is not the same
        corpus next time.

The English-collision count is reported and never fails the run. A manufactured word that
happens to spell an English one is not a leak -- the rule is about provenance, not about
vocabulary -- but "we assumed collisions were rare" is the kind of assumption that goes
stale silently, so it is a number in the report.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .glossary import GlossaryProfile
from .pack import PackSpec, SyntheticPack
from .truth import TruthClass

REPO_ROOT = Path(__file__).resolve().parents[2]
_VOCAB = (
    REPO_ROOT / "src" / "nexus_matcher" / "_models" / "bge-small-en-v1.5-onnx-int8" / "vocab.txt"
)


@dataclass
class Finding:
    check: str
    detail: str
    fatal: bool = True

    def render(self) -> str:
        return f"  [{'FAIL' if self.fatal else 'WARN'}] {self.check}: {self.detail}"


def _within(actual: float, target: float, tolerance: float) -> bool:
    return abs(actual - target) <= tolerance


def _tokens(text: str) -> set[str]:
    return {t for t in text.replace("_", " ").lower().split() if t}


def overlap_report(pack: SyntheticPack) -> dict[str, float]:
    """
    Query-to-gold token overlap on the READABLE, FLAT profiles only.

    Jaccard between the column name's tokens and the gold entry's business-name tokens. A
    corpus where this is 1.0 everywhere is measuring string identity; one where it is 0.0
    everywhere has no lexical arm to speak of. The number to watch is `identical_share`.

    Restricted to the uncontracted, unnested profiles on purpose. On a contracted schema
    the overlap is near zero by construction -- that is the abbreviation gap, not a
    property of the paraphrase -- and on a nested one the parent path adds tokens no term
    was ever going to carry. Measuring across all six would average three different
    effects into one number that answers no question.
    """
    scores: list[float] = []
    identical = 0
    for schema in pack.schemas:
        if schema.profile.contracted or schema.profile.depth or schema.profile.mixed:
            continue
        for row in schema.truth:
            if row.truth_class is not TruthClass.EXACT or not row.correct_ids:
                continue
            gold = pack.glossary.by_id.get(row.correct_ids[0])
            if gold is None:
                continue
            q = _tokens(row.flattened_name)
            g = _tokens(gold.name)
            if not q or not g:
                continue
            j = len(q & g) / len(q | g)
            scores.append(j)
            if j == 1.0:
                identical += 1
    if not scores:
        return {"n": 0, "mean": 0.0, "median": 0.0, "identical_share": 0.0}
    scores.sort()
    return {
        "n": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "median": round(scores[len(scores) // 2], 4),
        "identical_share": round(identical / len(scores), 4),
    }


def english_collisions(pack: SyntheticPack) -> dict[str, object]:
    """How many manufactured subject words spell a whole word in the encoder's vocabulary.

    Informational. The vocabulary file is the bundled BERT-uncased wordpiece list, so a
    hit means the stem is a common English word and will carry borrowed meaning into the
    embedding. Absent vocabulary file -> reported as unavailable, never as zero: a check
    that cannot run must not look like a check that passed.
    """
    if not _VOCAB.exists():
        return {"available": False}
    vocab = {
        line.strip()
        for line in _VOCAB.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.startswith("[") and not line.startswith("##")
    }
    subjects = [s.lower() for s in pack.pools.subjects]
    hits = sorted(s for s in subjects if s in vocab)
    return {
        "available": True,
        "subjects": len(subjects),
        "collide_with_english": len(hits),
        "share": round(len(hits) / len(subjects), 4) if subjects else 0.0,
        "examples": hits[:8],
    }


def _delimiter_trap(pack: SyntheticPack) -> tuple[int, int]:
    """(rows with >1 comma-separated sample value, rows with >1 semicolon-separated enum).

    Both have to be non-trivial for the trap to be a trap: a file where only one of the
    two columns is genuinely multi-valued cannot demonstrate that reading it with the
    other's separator is wrong.
    """
    multi_sample = sum(1 for r in pack.glossary.rows if len(r.sample_values) > 1)
    multi_enum = sum(1 for r in pack.glossary.rows if len(r.enum_values) > 1)
    return multi_sample, multi_enum


def _determinism(spec: PackSpec) -> str | None:
    """Generate the spec twice into temporary directories and compare every file."""
    digests: list[dict[str, str]] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pack"
            SyntheticPack.generate(spec).write(out)
            digests.append(
                {
                    p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name != "manifest.json"
                }
            )
    if digests[0] != digests[1]:
        differing = sorted(
            k for k in set(digests[0]) | set(digests[1]) if digests[0].get(k) != digests[1].get(k)
        )
        return f"{len(differing)} file(s) differ between two runs of the same spec: {differing[:5]}"
    return None


def verify(pack: SyntheticPack, check_determinism: bool = True) -> tuple[list[Finding], dict]:
    """Check every advertised property. Returns (findings, report)."""
    findings: list[Finding] = []
    manifest = pack.manifest()
    g = manifest["glossary"]
    profile = GlossaryProfile(rows=pack.spec.rows).scaled(pack.spec.difficulty)

    # -- glossary distribution ----------------------------------------------
    if not _within(g["non_approved_share"], profile.non_approved_share, 0.03):
        findings.append(
            Finding(
                "non-approved share",
                f"{g['non_approved_share']} against a target of {profile.non_approved_share}",
            )
        )
    if not _within(g["in_near_duplicate_cluster_share"], profile.near_duplicate_share, 0.06):
        findings.append(
            Finding(
                "near-duplicate share",
                f"{g['in_near_duplicate_cluster_share']} against a target of "
                f"{profile.near_duplicate_share}",
            )
        )
    if not _within(g["definition_echoes_name_share"], profile.definition_echo_share, 0.03):
        findings.append(
            Finding(
                "definition-echoes-name share",
                f"{g['definition_echoes_name_share']} against a target of "
                f"{profile.definition_echo_share}",
            )
        )
    if not 3 <= g["name_tokens_median"] <= 5:
        findings.append(
            Finding("name length", f"median {g['name_tokens_median']} tokens, wanted 4")
        )
    if g["name_tokens_max"] < 8:
        findings.append(
            Finding("name length tail", f"longest name is {g['name_tokens_max']} tokens, wanted 9")
        )
    if g["distinct_class_words"] < 15:
        findings.append(
            Finding("class-word spread", f"only {g['distinct_class_words']} distinct class words")
        )

    # -- the two traps -------------------------------------------------------
    multi_sample, multi_enum = _delimiter_trap(pack)
    if multi_sample < 10 or multi_enum < 10:
        findings.append(
            Finding(
                "delimiter trap",
                f"{multi_sample} rows with several comma-separated sample values and "
                f"{multi_enum} with several semicolon-separated enum values; both must be "
                f"substantial or reading one with the other's separator is not a mistake "
                f"the data can reveal",
            )
        )

    # -- abbreviation catalog ------------------------------------------------
    a = manifest["abbreviations"]
    uncovered = [t for t in pack.pools.all_tokens if t.lower() not in pack.catalog.contraction]
    if uncovered:
        findings.append(
            Finding("catalog coverage", f"{len(uncovered)} token(s) have no short form")
        )
    for name, value in (
        ("ambiguous shorts", a["ambiguous_shorts"]),
        ("multi-word rules", a["multi_word_rules"]),
        ("stopword collisions", a["stopword_collisions"]),
        ("never-expand acronyms", a["never_expand"]),
        ("delta entries", a["delta_entries"]),
    ):
        if not value:
            findings.append(Finding(f"catalog: {name}", "none present"))
    # Every row has to survive the consumer the artifact names. `AbbreviationMapping`
    # refuses an empty key, an empty expansion, and a short form equal to its own long
    # form -- and `from_dict` swallows the refusal, so a bad row vanishes on load with no
    # count anywhere. Checked here in plain Python rather than by importing the library,
    # because this package deliberately depends on nothing.
    unloadable = [
        s
        for s, long in pack.catalog.expansions.items()
        if not s.strip() or not long.strip() or s.strip().lower() == long.strip().lower()
    ]
    if unloadable:
        findings.append(
            Finding(
                "catalog rows that will not load",
                f"{len(unloadable)} row(s) would be dropped by AbbreviationMapping's "
                f"validation, silently, inside from_dict",
            )
        )
    unchanged = [
        s for s, long in pack.delta.changed.items() if pack.catalog.expansions.get(s) == long
    ]
    if unchanged:
        findings.append(Finding("delta", f"{len(unchanged)} delta row(s) do not change anything"))

    # -- truth ---------------------------------------------------------------
    truth = pack.truth_rows()
    counts = manifest["truth"]
    total = sum(counts.values())
    if total:
        for cls, target, tol in (
            ("NO_MATCH", 0.15, 0.06),
            ("TRAP", 0.05, 0.03),
        ):
            share = counts[cls] / total
            if not _within(share, target, tol):
                findings.append(
                    Finding(f"truth share {cls}", f"{share:.4f} against a target of {target}")
                )
    subjects = {s.lower() for s in pack.pools.subjects}
    orphans = {s.lower() for s in pack.pools.orphans}
    if not orphans.isdisjoint(subjects):
        findings.append(
            Finding(
                "orphan vocabulary",
                "the held-out pool overlaps the glossary's subjects, so a NO_MATCH row "
                "could describe a real term",
            )
        )
    ambiguous_single = [
        r for r in truth if r.truth_class is TruthClass.AMBIGUOUS and len(r.correct_ids) < 2
    ]
    if ambiguous_single:
        findings.append(
            Finding(
                "AMBIGUOUS rows",
                f"{len(ambiguous_single)} record fewer than two defensible terms",
            )
        )
    answered = [
        r
        for r in truth
        if r.truth_class in (TruthClass.NO_MATCH, TruthClass.TRAP) and r.correct_ids
    ]
    if answered:
        findings.append(
            Finding("no-match rows", f"{len(answered)} carry a correct id and must not")
        )

    # -- repeated leaf -------------------------------------------------------
    repeated = pack.schema("nested-repeated")
    by_leaf: dict[str, set[str]] = {}
    for row in repeated.truth:
        leaf = row.flattened_name.split("__")[-1]
        by_leaf.setdefault(leaf, set()).update(row.correct_ids)
    widest = max((len(v) for v in by_leaf.values()), default=0)
    if widest < 5:
        findings.append(
            Finding(
                "repeated leaf",
                f"the most contested leaf name has only {widest} distinct correct answers; "
                f"collapsing the cache would barely be visible",
            )
        )

    # -- feedback ------------------------------------------------------------
    f = manifest["feedback"]
    if not f["manual_override"]:
        findings.append(Finding("feedback", "no MANUAL_OVERRIDE records"))
    if f["chose_an_unproposed_term"] != f["manual_override"]:
        findings.append(
            Finding(
                "feedback",
                f"{f['manual_override']} overrides but {f['chose_an_unproposed_term']} chose "
                f"an unproposed term; an override that WAS proposed is a rerank, not an "
                f"override",
            )
        )

    # -- leak ----------------------------------------------------------------
    leak = overlap_report(pack)
    if leak["identical_share"] > 0.35:
        findings.append(
            Finding(
                "query/gold overlap",
                f"{leak['identical_share']:.2%} of EXACT queries are token-identical to "
                f"their gold business name; raise paraphrase_strength or the corpus is "
                f"measuring string identity",
            )
        )

    report = {
        "manifest": manifest,
        "overlap": leak,
        "english_collisions": english_collisions(pack),
        "delimiter_trap": {"multi_sample_values": multi_sample, "multi_enum_values": multi_enum},
        "widest_contested_leaf": widest,
    }

    if check_determinism:
        # A small spec: the check is about reproducibility, not about size, and a 100,000
        # row pack generated twice is a minute of nothing.
        small = PackSpec(rows=1_200, seed=pack.spec.seed, schema_scale=0.05, feedback_events=50)
        problem = _determinism(small)
        report["determinism"] = "identical" if problem is None else problem
        if problem is not None:
            findings.append(Finding("determinism", problem))

    return findings, report
