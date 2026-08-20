"""
benchmarks.synthetic.abbreviations | Layer: BENCHMARK
Artifact 2 of 5: the synthetic approved-abbreviation catalog, and its versioned delta.

Covering every token the glossary generator uses is the easy half. The half that makes
this a test is that expansion has to be HARD in the five ways it is hard in practice:

  ambiguous shorts       one short form, several defensible long forms. The standard
                         picks one; the other candidates are recorded, because the wrong
                         pick is a whole class of real defect and a catalog that never
                         had to choose cannot produce it.
  multi-word expansions  one token becomes two or three. A contraction that swallowed an
                         adjacent word cannot be undone by a per-token lookup that assumes
                         one-in-one-out.
  stopword collisions    short forms that are spelled exactly like English function words.
                         Any pipeline that drops stopwords before expanding loses them,
                         and loses them silently, because the query still looks fine.
  domain acronyms        already atoms. Expanding them is the error, so they are listed
                         and the contraction rule leaves them alone.
  a versioned delta      ~200 mappings that CHANGE. This is the one that cannot be
                         satisfied by a load-time configuration file: the feed a real
                         deployment reads is live, and a term abbreviated one way this
                         quarter is abbreviated differently the next.

Shape on disk
-------------
`expansions` is `{short: long}` and nothing else, because that is exactly what
`AbbreviationDictionary.from_dict` takes -- the artifact is directly consumable, with no
adapter to write and therefore no adapter to get wrong. Everything else in the file is
metadata ABOUT that map: what the single pick discarded, which shorts collide with
English, which tokens must never be touched.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .pools import Pools

_VOWELS = frozenset("aeiou")

# English function words that a naming standard nonetheless hands out as short forms.
# Short enough to be tempting, common enough that a stopword filter eats them.
_STOPWORD_SHORTS: tuple[str, ...] = (
    "as",
    "in",
    "no",
    "on",
    "at",
    "by",
    "or",
    "to",
    "is",
    "it",
    "of",
    "an",
)


@dataclass
class AbbreviationCatalog:
    """
    The catalog, plus the facts about it the experiments need.

    `expansions` is authoritative and one-to-one: a short resolves to exactly one long
    form, which is what a deployment actually applies. `ambiguous` is the record of what
    that choice threw away.
    """

    version: int
    expansions: dict[str, str] = field(default_factory=dict)
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    multi_word: dict[str, str] = field(default_factory=dict)
    stopword_collisions: tuple[str, ...] = ()
    never_expand: tuple[str, ...] = ()
    # Words the standard leaves alone because the contraction rule returns the word
    # itself. Reported rather than dropped silently: they are the difference between the
    # token count and the row count, and an unexplained gap between those two invites the
    # reader to assume the catalog lost something.
    identity: tuple[str, ...] = ()
    # long form (lowercased) -> the short the standard contracts it to. The direction the
    # SCHEMA generator needs; `expansions` is the direction a matcher needs.
    contraction: dict[str, str] = field(default_factory=dict)

    def with_delta(self, delta: AbbreviationDelta) -> AbbreviationCatalog:
        """
        This catalog as the delta leaves it. Neither input is modified.

        The delta changes what a short form EXPANDS to. Contraction inherits the change
        for the long forms named in it and is otherwise untouched -- the mechanical rule
        for every other word did not move just because the feed re-pointed 200 shorts.
        Rebuilding the whole contraction map from the new expansions instead would strip
        a short form from every long form that lost a collision, and the catalog would
        quietly stop covering its own vocabulary.
        """
        merged = dict(self.expansions)
        merged.update(delta.changed)
        contraction = dict(self.contraction)
        for short, long in delta.changed.items():
            contraction[long.lower()] = short
        return AbbreviationCatalog(
            version=delta.version,
            expansions=merged,
            ambiguous=dict(self.ambiguous),
            multi_word=dict(self.multi_word),
            stopword_collisions=self.stopword_collisions,
            never_expand=self.never_expand,
            identity=self.identity,
            contraction=contraction,
        )

    def contract_tokens(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        """
        Apply the naming standard to a sequence of words.

        Multi-word shorts are matched greedily and first, because that is the order a
        standard applies them: a rule that says "Primary Identifier contracts to PID" is
        not satisfied by contracting the two words separately, and a pipeline that expands
        PID back into two words has to know it was ever one token.
        """
        out: list[str] = []
        i = 0
        # Longest phrase first so a two-word rule cannot pre-empt a three-word one.
        phrases = sorted(self.multi_word.items(), key=lambda kv: -len(kv[1].split()))
        while i < len(tokens):
            matched = False
            for short, long in phrases:
                words = long.split()
                if tuple(t.lower() for t in tokens[i : i + len(words)]) == tuple(
                    w.lower() for w in words
                ):
                    out.append(short.upper())
                    i += len(words)
                    matched = True
                    break
            if matched:
                continue
            token = tokens[i]
            if token in self.never_expand:
                out.append(token)
            else:
                out.append(self.contraction.get(token.lower(), token.lower()).upper())
            i += 1
        return tuple(out)


@dataclass(frozen=True)
class AbbreviationDelta:
    """
    A quarter's worth of change to the catalog.

    `changed` is `{short: new long form}`, so it drops straight into
    `AbbreviationExpander.with_overlay` or `AbbreviationDictionary.merged_with`. Every
    entry in it REPLACES a mapping the base catalog already had -- an overlay that only
    added rows would be indistinguishable from a slightly larger catalog, and would prove
    nothing about whether the overlay took effect.
    """

    version: int
    base_version: int
    changed: dict[str, str]


def _short_form(word: str) -> str:
    """Vowel-drop, then truncate. A mechanical rule, so it encodes nobody's standard."""
    w = word.lower()
    devowelled = w[0] + "".join(c for c in w[1:] if c not in _VOWELS)
    candidate = devowelled if len(devowelled) >= 3 else w
    return candidate[:4]


def build_catalog(
    pools: Pools,
    seed: int,
    ambiguous_share: float = 0.08,
    multi_word_count: int = 240,
    version: int = 1,
) -> AbbreviationCatalog:
    """
    Build the base catalog over every token the glossary generator can emit.

    `ambiguous_share` is a FLOOR, not a target: the vowel-drop rule collides on its own
    at this vocabulary size, and forcing collisions on top guarantees the property holds
    even when the pools are small enough that natural collisions are rare.
    """
    rng = random.Random(seed ^ 0x5EED_0002)
    tokens = list(pools.all_tokens)

    # 1. The natural pass. Several long forms will land on the same short; the first one
    #    to claim it wins the EXPANSION slot and the rest are recorded as what the pick
    #    discarded. Every token still gets a contraction -- that is what an ambiguous
    #    short form IS. A catalog where the losers had no short form at all would leave
    #    those words uncontracted in every generated column, and the corpus would carry a
    #    subset of columns that are conspicuously readable for no modelled reason.
    claimed: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    contraction: dict[str, str] = {}
    # Words the standard leaves alone: short enough that the rule returns the word itself.
    # They belong in `contraction` (contracting them is a no-op) and NOT in `expansions`,
    # because a row whose short form equals its long form asserts nothing. It is also the
    # one row shape `AbbreviationMapping` rejects, so leaving them in meant the catalog
    # silently lost rows on load -- 7 of 999 on the first 10,000-row pack, discarded inside
    # `from_dict`'s except branch with no count anywhere. A generated artifact that does
    # not survive its own documented consumer intact is a broken artifact.
    identity: list[str] = []
    for token in tokens:
        short = _short_form(token)
        contraction[token.lower()] = short
        if short == token.lower():
            identity.append(token)
            continue
        if short in claimed and claimed[short] != token:
            ambiguous.setdefault(short, [claimed[short]]).append(token)
            continue
        claimed[short] = token

    # 2. Force the ambiguity floor. Tokens that got a unique short are re-pointed at a
    #    short that is already taken, so the catalog holds a documented number of
    #    genuinely contested short forms rather than however many the rule happened to
    #    produce.
    unique = [t for s, t in claimed.items() if s not in ambiguous]
    target = int(len(tokens) * ambiguous_share)
    forced = rng.sample(unique, min(max(0, target - len(ambiguous)), len(unique) // 2))
    taken_shorts = sorted(claimed)
    for token in forced:
        victim = rng.choice(taken_shorts)
        if claimed.get(victim) == token:
            continue
        ambiguous.setdefault(victim, [claimed[victim]]).append(token)
        contraction[token.lower()] = victim

    # 3. Stopword collisions. A handful of tokens get a short form spelled exactly like an
    #    English function word.
    stopword_targets = rng.sample(tokens, min(len(_STOPWORD_SHORTS), len(tokens)))
    stopword_collisions: list[str] = []
    for short, token in zip(_STOPWORD_SHORTS, stopword_targets, strict=False):
        if claimed.get(short) not in (None, token):
            ambiguous.setdefault(short, [claimed[short]]).append(token)
        claimed[short] = token
        contraction[token.lower()] = short
        stopword_collisions.append(short)

    # 4. Multi-word rules: an adjacent qualifier + class word that the standard collapses
    #    into a single token. One short, two or three long-form words.
    multi_word: dict[str, str] = {}
    class_longs = [cw.long for cw in pools.class_words]
    for _ in range(multi_word_count * 3):
        if len(multi_word) >= multi_word_count:
            break
        words = [rng.choice(pools.qualifiers), rng.choice(class_longs)]
        if rng.random() < 0.25:
            words.insert(1, rng.choice(pools.qualifiers))
        short = "".join(w[0] for w in words).upper() + str(len(multi_word) % 10)
        phrase = " ".join(words)
        if short.lower() in claimed or phrase in multi_word.values():
            continue
        multi_word[short.lower()] = phrase

    expansions = dict(claimed)
    expansions.update(multi_word)

    return AbbreviationCatalog(
        version=version,
        expansions=expansions,
        ambiguous={k: tuple(dict.fromkeys(v)) for k, v in sorted(ambiguous.items())},
        multi_word=multi_word,
        stopword_collisions=tuple(stopword_collisions),
        never_expand=pools.never_expand,
        identity=tuple(identity),
        contraction=contraction,
    )


def build_delta(
    catalog: AbbreviationCatalog,
    seed: int,
    size: int = 200,
) -> AbbreviationDelta:
    """
    The next version's changes: `size` short forms re-pointed at a different long form.

    Every changed row is a short that ALREADY resolved to something. Applying the delta
    therefore has a predictable direction: fields contracted under version 2 and expanded
    with the version 1 catalog get the wrong long form on exactly these rows, and the
    experiment measures the gap the overlay closes.
    """
    rng = random.Random(seed ^ 0x5EED_0003)
    # Single-word rows only. Re-pointing a multi-word rule would change how many tokens a
    # name contracts to, and then the two versions would not be comparable field by field.
    candidates = sorted(s for s in catalog.expansions if s not in catalog.multi_word)
    picked = rng.sample(candidates, min(size, len(candidates)))
    longs = sorted({v for k, v in catalog.expansions.items() if k not in catalog.multi_word})

    changed: dict[str, str] = {}
    for short in picked:
        current = catalog.expansions[short]
        replacement = rng.choice(longs)
        # A row that does not change is not a delta row; it would pad the count and
        # weaken every claim made about the size of the overlay's effect.
        for _ in range(8):
            if replacement != current:
                break
            replacement = rng.choice(longs)
        if replacement != current:
            changed[short] = replacement

    return AbbreviationDelta(
        version=catalog.version + 1, base_version=catalog.version, changed=changed
    )


def catalog_as_json(catalog: AbbreviationCatalog) -> dict[str, object]:
    """The on-disk document. Sorted throughout, so the file is byte-stable."""
    return {
        "notice": (
            "SYNTHETIC. Every long form here was manufactured by "
            "benchmarks/synthetic/pools.py from a seeded syllable grammar. This is not "
            "any organisation's approved-abbreviation list and must not be used as one."
        ),
        "version": catalog.version,
        "expansions": dict(sorted(catalog.expansions.items())),
        "ambiguous": {k: list(v) for k, v in sorted(catalog.ambiguous.items())},
        "multi_word": dict(sorted(catalog.multi_word.items())),
        "stopword_collisions": sorted(catalog.stopword_collisions),
        "never_expand": sorted(catalog.never_expand),
        "not_contracted": sorted(catalog.identity),
    }


def delta_as_json(delta: AbbreviationDelta) -> dict[str, object]:
    return {
        "notice": "SYNTHETIC. See abbreviations.json.",
        "version": delta.version,
        "base_version": delta.base_version,
        "changed": dict(sorted(delta.changed.items())),
    }
