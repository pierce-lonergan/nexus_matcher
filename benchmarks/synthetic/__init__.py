"""
benchmarks.synthetic | Layer: BENCHMARK
A generator for a schema-matching corpus that reproduces the statistical properties of a
large enterprise glossary without containing any of one.

Why a generator and not a fixture
---------------------------------
The accuracy features that matter most on a large corpus -- an approved-abbreviation
catalog, a caller-supplied domain prior, cache-key composition, row admission, abstention
-- are exactly the features the public benchmarks cannot exercise. BIRD and OMOP have
hundreds of readable English column names, high doc coverage, few near-duplicates and no
reviewer history. A matcher can score well on them and fail on the first schema that
presents a column with no English in it, no documentation, and forty plausible glossary
neighbours.

A fixture would fix the one property every question here is about: SCALE. So each of the
five artifacts is a generator with scale and difficulty as dials.

  pools          the manufactured vocabulary everything else draws from
  glossary       artifact 1: rows with the distribution that makes matching hard
  abbreviations  artifact 2: the naming standard, its ambiguity, and a versioned delta
  schemas        artifact 3: six profiles, raw .avsc and pre-flattened, with ground truth
  truth          artifact 4: EXACT / AMBIGUOUS / NO_MATCH / TRAP
  feedback       artifact 5: reviewer verdicts, including the one the wire cannot carry
  pack           the five as one reproducible object
  verify         the generator held to its own claims

Two traps are preserved deliberately, because getting either one wrong fails silently:

  1. Two multi-value columns in one file with DIFFERENT separators -- `sample_values` is
     comma-separated and `enum_values` semicolon-separated. Read either with the other's
     separator and you get one value per row containing every element, which indexes and
     matches and is wrong with no error anywhere.
  2. The vocabulary pools are SYNTHESISED, never sampled. A generator that borrows
     "realistic" nouns from a real glossary produces a corpus exactly as unshareable as
     the glossary was, and nobody reviewing 100,000 rows will spot which 600 words came
     from where.

What this pack does NOT prove
-----------------------------
It validates MECHANISMS. It does not measure real-world accuracy, and it cannot reproduce
genuine semantic subtlety -- two real governed terms are ambiguous in ways richer than any
near-duplicate rule. A pass here is necessary and not sufficient: it is the fast, cheap,
shareable gate that runs before the expensive one, not a substitute for a steward-labelled
golden set.

Usage
-----
    python benchmarks/gen_synthetic_pack.py --rows 10000 --out data/synthetic/dev
    python benchmarks/gen_synthetic_pack.py --rows 10000 --verify
"""

from __future__ import annotations

from .abbreviations import AbbreviationCatalog, AbbreviationDelta, build_catalog, build_delta
from .feedback import FeedbackEvent, build_feedback, wire_loss
from .glossary import (
    ADMIT_APPROVED,
    COLUMN_MAPPING,
    CSV_HEADER,
    VALUE_DELIMITERS,
    Glossary,
    GlossaryProfile,
    GlossaryRow,
    build_glossary,
    glossary_rows_as_dicts,
)
from .pack import DEFAULT_SEED, NOTICE, PackSpec, SyntheticPack
from .pools import Pools, build_pools
from .schemas import PROFILES, SyntheticSchema, build_schemas
from .truth import DEFAULT_SHARES, TruthClass, TruthRow, class_counts, read_truth_csv

__all__ = [
    "ADMIT_APPROVED",
    "COLUMN_MAPPING",
    "CSV_HEADER",
    "DEFAULT_SEED",
    "DEFAULT_SHARES",
    "NOTICE",
    "PROFILES",
    "VALUE_DELIMITERS",
    "AbbreviationCatalog",
    "AbbreviationDelta",
    "FeedbackEvent",
    "Glossary",
    "GlossaryProfile",
    "GlossaryRow",
    "PackSpec",
    "Pools",
    "SyntheticPack",
    "SyntheticSchema",
    "TruthClass",
    "TruthRow",
    "build_abbreviation_catalog",
    "build_catalog",
    "build_delta",
    "build_feedback",
    "build_glossary",
    "build_pools",
    "build_schemas",
    "class_counts",
    "glossary_rows_as_dicts",
    "read_truth_csv",
    "wire_loss",
]

# Alias kept because `build_catalog` on its own reads ambiguously at a call site three
# modules away, where "catalog" could as easily mean the glossary.
build_abbreviation_catalog = build_catalog
