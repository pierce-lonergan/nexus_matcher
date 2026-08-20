"""
benchmarks.synthetic.pools | Layer: BENCHMARK
The vocabulary every other generator draws from, SYNTHESISED rather than sampled.

Why this module exists at all
-----------------------------
The point of a synthetic corpus is that it can be shared. A generator that reaches into
a real glossary for "realistic" nouns produces a corpus that is exactly as unshareable as
the glossary was, and the leak is invisible: nobody reviewing 100,000 generated rows will
notice that 600 of the subject words came from somewhere they may not go.

So the content words here are MANUFACTURED. `subjects` and `domain_words` come out of a
seeded syllable grammar -- onset/nucleus/coda triples assembled by rule -- and nothing in
this file is copied from any glossary, public or otherwise. Change the seed and you get a
different vocabulary of the same shape, which is the property that makes the corpus a
generator instead of a fixture.

What is NOT manufactured, and why that is fine
----------------------------------------------
Three pools are ordinary English and deliberately so:

  * `class_words` -- Identifier, Code, Name, Date and the rest. These are the grammatical
    tail of a data-modelling name, they are public knowledge in every naming standard
    ever written, and the whole reason the pool exists is that a matcher gets real signal
    from them. Manufacturing them would delete the signal the corpus is meant to carry.
  * `qualifiers` -- Primary, Adjusted, Reported. Generic modelling adjectives.
  * `definition scaffolding` (in glossary.py) -- ordinary connective prose.

The rule is about PROVENANCE, not about English: none of these were read off anyone's
term list. A manufactured subject word that happens to collide with an English word is
not a leak either, for the same reason -- but `verify.py` counts the collisions against
the bundled encoder's whole-word vocabulary and prints the number, because "we assumed it
was rare" is how this sort of claim goes stale.

Determinism
-----------
Everything is derived from one integer seed. Two runs at the same seed produce
byte-identical pools; there is no clock, no PID, no set iteration and no dict ordering
dependence anywhere in this package.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# The syllable grammar. Deliberately small and deliberately not English-shaped: the
# nuclei include digraphs English rarely puts in a stressed first syllable, so the
# generated stems read as plausible-but-invented rather than as near-misses of real
# words. They are still pronounceable, which matters more than it sounds -- an encoder's
# wordpiece tokenizer splits an unpronounceable string into single characters, and a
# corpus of single characters measures nothing.
_ONSETS = (
    "b",
    "br",
    "d",
    "dr",
    "f",
    "fl",
    "g",
    "gl",
    "gr",
    "h",
    "j",
    "k",
    "kl",
    "kr",
    "l",
    "m",
    "n",
    "p",
    "pl",
    "pr",
    "r",
    "s",
    "sk",
    "sl",
    "sn",
    "sp",
    "st",
    "sv",
    "t",
    "tr",
    "v",
    "vr",
    "z",
    "zh",
)
_NUCLEI = ("a", "e", "i", "o", "u", "ae", "ei", "oa", "ou", "ia", "eo", "au")
_CODAS = ("", "", "l", "ll", "m", "n", "nd", "ng", "nt", "r", "rd", "rk", "rn", "sk", "st", "th")

# Class words, with the short form a naming standard would give each and the relative
# weight of the head of the distribution.
#
# The shape is what matters, not the exact numbers: a handful of class words carry most
# of the mass and a long tail carries the rest. A flat distribution would make the class
# word a free discriminator and every name trivially separable, which no real glossary is.
_CLASS_WORDS: tuple[tuple[str, str, int], ...] = (
    ("Identifier", "ID", 180),
    ("Code", "CD", 150),
    ("Name", "NM", 130),
    ("Date", "DT", 110),
    ("Amount", "AMT", 60),
    ("Indicator", "IND", 55),
    ("Text", "TXT", 40),
    ("Number", "NBR", 38),
    ("Quantity", "QTY", 32),
    ("Timestamp", "TS", 30),
    ("Description", "DESC", 26),
    ("Percent", "PCT", 22),
    ("Rate", "RT", 20),
    ("Status", "STS", 18),
    ("Type", "TYP", 16),
    ("Value", "VAL", 14),
    ("Flag", "FLG", 12),
    ("Count", "CNT", 11),
    ("Address", "ADDR", 10),
    ("Duration", "DUR", 8),
    ("Ratio", "RATIO", 7),
    ("Score", "SCR", 6),
    ("Category", "CAT", 5),
)

# Where the data came from rather than what it means. A source system puts these in a
# column name; a glossary never puts them in a term.
_SOURCE_WORDS: tuple[str, ...] = (
    "Source",
    "Feed",
    "Raw",
    "Intake",
    "Upstream",
    "Landing",
    "Staged",
    "Inbound",
)

# Generic modelling adjectives. Public vocabulary, and the part of a name that a
# near-duplicate cluster varies while meaning the same thing.
_QUALIFIERS: tuple[str, ...] = (
    "Primary",
    "Secondary",
    "Original",
    "Adjusted",
    "Reported",
    "Effective",
    "Current",
    "Prior",
    "Net",
    "Gross",
    "Total",
    "Partial",
    "Internal",
    "External",
    "Preferred",
    "Alternate",
    "Legacy",
    "Derived",
    "Nominal",
    "Residual",
    "Scheduled",
    "Actual",
    "Estimated",
    "Confirmed",
    "Provisional",
    "Consolidated",
    "Standing",
    "Interim",
)


@dataclass(frozen=True)
class ClassWord:
    """One class word: the long form, the standard's short form, its share of the mass."""

    long: str
    short: str
    weight: int


@dataclass(frozen=True)
class Pools:
    """
    Every content word the pack is built from, plus the held-out orphan pool.

    `orphans` is the load-bearing one. NO_MATCH ground truth is only honest if no correct
    term can exist, and the cheapest way to guarantee that is to reserve a slice of the
    vocabulary that the glossary generator is never allowed to see. A NO_MATCH column
    built from orphan stems cannot accidentally describe a real row, however many rows
    there are, at any scale. Sampling "a term that happens not to be in the glossary"
    instead would be true at 1,000 rows and quietly false at 100,000.
    """

    subjects: tuple[str, ...]
    orphans: tuple[str, ...]
    domains: tuple[str, ...]
    qualifiers: tuple[str, ...]
    class_words: tuple[ClassWord, ...]
    # Words a SOURCE SYSTEM puts in a column name and a glossary never puts in a term:
    # the residue of where the data came from rather than what it means. The paraphrase
    # step swaps a qualifier for one of these, so a generated column carries lexical noise
    # that no term can match -- which is the ordinary condition and not an edge case.
    source_words: tuple[str, ...]
    # Uppercase acronyms that a naming standard treats as atoms: they are already short
    # and expanding them is the error, not the fix.
    never_expand: tuple[str, ...]

    @property
    def all_tokens(self) -> tuple[str, ...]:
        """Every word an abbreviation catalog has to cover, orphans included.

        Orphans are in here on purpose. A NO_MATCH column still gets contracted by the
        naming standard, so a catalog that did not cover them would leave those columns
        as the only ones on the query side carrying full words -- and "the matcher
        abstained on exactly the rows that looked different from every other row" is not
        a measurement of abstention.
        """
        words: list[str] = []
        words.extend(self.subjects)
        words.extend(self.orphans)
        words.extend(self.domains)
        words.extend(self.qualifiers)
        words.extend(self.source_words)
        words.extend(cw.long for cw in self.class_words)
        seen: set[str] = set()
        out: list[str] = []
        for word in words:
            if word not in seen:
                seen.add(word)
                out.append(word)
        return tuple(out)


def _syllable(rng: random.Random) -> str:
    return rng.choice(_ONSETS) + rng.choice(_NUCLEI) + rng.choice(_CODAS)


def _word(rng: random.Random, syllables: int) -> str:
    raw = "".join(_syllable(rng) for _ in range(syllables))
    return raw[:1].upper() + raw[1:]


def _distinct_words(
    rng: random.Random, count: int, min_len: int = 5, seen: set[str] | None = None
) -> tuple[str, ...]:
    """`count` distinct manufactured words, in generation order.

    Ordered, not a set: a set's iteration order is stable within a process and irrelevant
    across them, and this package's whole claim is that the same seed gives the same
    bytes on any machine.

    `seen` is SHARED across the pools by `build_pools`, and that sharing is load-bearing
    rather than tidy. With a fresh set per call the orphan pool could -- and on the first
    run at 10,000 rows did -- contain a word the glossary also uses as a subject, which
    silently converts a NO_MATCH row into one with a perfectly good correct answer. The
    verifier caught it; nothing else would have, because the truth file would still say
    NO_MATCH and the matcher would still be right to disagree.
    """
    out: list[str] = []
    seen = seen if seen is not None else set()
    # Bounded rather than `while True`: a degenerate grammar would otherwise spin
    # forever, and a generator that hangs is harder to diagnose than one that refuses.
    for _ in range(count * 200):
        if len(out) == count:
            break
        word = _word(rng, rng.choice((1, 2, 2, 3)))
        if len(word) < min_len or word in seen:
            continue
        seen.add(word)
        out.append(word)
    if len(out) < count:
        raise RuntimeError(
            f"the syllable grammar produced only {len(out)} distinct words of {count} "
            f"asked for; widen _ONSETS/_NUCLEI/_CODAS or lower the count"
        )
    return tuple(out)


def build_pools(
    seed: int,
    n_subjects: int = 900,
    n_orphans: int = 160,
    n_domains: int = 30,
    n_never_expand: int = 24,
) -> Pools:
    """
    Manufacture the vocabulary.

    `n_subjects` sets the ceiling on how many DISTINCT names the glossary can hold before
    it starts repeating combinations; 900 subjects times 23 class words times the
    qualifier combinations is far more than 100,000, so the glossary's near-duplicate
    share stays a deliberate parameter rather than an accident of exhaustion.
    """
    rng = random.Random(seed)
    # One `seen` set for all three word pools: subjects, orphans and domains must be
    # mutually disjoint, and disjointness is the property the NO_MATCH class rests on.
    taken: set[str] = set()
    subjects = _distinct_words(rng, n_subjects, seen=taken)
    orphans = _distinct_words(rng, n_orphans, seen=taken)
    domains = _distinct_words(rng, n_domains, min_len=6, seen=taken)
    never = tuple(
        "".join(rng.choice("BCDFGHJKLMNPRSTVZ") for _ in range(rng.choice((3, 3, 4, 5))))
        for _ in range(n_never_expand * 4)
    )
    # De-duplicate the acronyms in generation order and keep the first n.
    seen: set[str] = set()
    never_expand: list[str] = []
    for token in never:
        if token not in seen:
            seen.add(token)
            never_expand.append(token)
        if len(never_expand) == n_never_expand:
            break

    return Pools(
        subjects=subjects,
        orphans=orphans,
        domains=domains,
        qualifiers=_QUALIFIERS,
        source_words=_SOURCE_WORDS,
        class_words=tuple(ClassWord(long, short, weight) for long, short, weight in _CLASS_WORDS),
        never_expand=tuple(never_expand),
    )
