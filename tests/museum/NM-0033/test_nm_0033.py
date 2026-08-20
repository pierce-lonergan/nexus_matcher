"""
NM-0033 -- `from_config(governance=...)` accepted a vocabulary and `load_dictionary()`
never applied it, so every match came back carrying no class.

The two lines the README, QUICKSTART and the governance example all show:

    matcher = NexusMatcher.from_config(governance="protection_classes.json")
    matcher.load_dictionary("glossary.csv")

Measured against `examples/governance/glossary.csv`, whose every row declares a protection
code: **0 of 30 indexed entries carried one**, and every `MatchResult.governance` was None.
Nothing errored. A glossary loaded without a vocabulary is a documented mode, and its
result is *identical* to this one -- which is why no layer below could notice and no
consumer above could tell the difference.

## Why nothing caught it

`DictionaryLoader` -- the port every loader implements -- returns finished
`DictionaryEntry` objects, and `ColumnMapping` has no field for a protection-code column.
So the loaders this package ships never read one, and `load_dictionary` indexed exactly
what they gave it. The vocabulary was consulted at MATCH time only, where
`self._governance.get(entry.governance_code)` on a code of None returns None, correctly and
uselessly.

Every governance test on the matcher called the private `_index_dictionary` with entries a
fixture had already coded by hand. That is a real gate for what it covers -- resolution,
inheritance, rank-1 rejection -- and it is blind to the entire loading path by
construction. The library's own HTTP app had already routed around the defect through
`ingest.load_entries`, and its docstring said so in full; the example pack printed a
`WIRING DEFECT` banner and rescued the class with a caller-side join. Two accurate
descriptions, no gate.

The defect is in the commit that cut 2.1.0: `git show
36ffc1d:src/nexus_matcher/application/use_cases/match_schema.py` has a `load_dictionary`
containing the word "governance" zero times. CHANGELOG.md records that 2.1.0 was never
published, so it did not reach a user; `release_preflight.py` declared the wheel built
from that tree fit to publish, so nothing between there and a publish was going to stop
it.

## What this asserts

The OBSERVABLE SYMPTOM at the boundary a caller uses: what a `MatchResult` carries after
the two documented lines, and what the load does with a glossary whose governance is wrong
or unreadable. Not the presence of any particular private method -- if the threading is
restructured, this file should still be right.

The last test is a CONTROL on what must NOT move. It pins the fields only the dictionary
loader produces (`sample_values`, `synonyms`, `parent_table`, and its `auto_N` ids),
because the tempting fix -- reading the glossary through `ingest.load_entries` instead of
the loader -- silently drops all four, and a governance fix that quietly changes what an
entry contains is a different defect wearing this one's name. Those four assertions hold
in the defective state too, which is the point of a control.

## The vocabulary is FICTIONAL

The Amberton Allotments Trust does not exist. This library ships no taxonomy: the
vocabulary is supplied by the caller, and this file supplies one exactly as a caller would.
What is under test is the STRUCTURE -- a closed set of codes, each deriving a tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_matcher.application import ingest
from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import CsvDictionaryLoader
from nexus_matcher.infrastructure.adapters.schema_parsers.avro import AvroSchemaParser
from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
from nexus_matcher.shared.types.base import Result

# PLOTHOLDER derives "Sealed", SITEPLAN derives "Open". Pinned as literals and asserted
# below, so the expectation does not come from the code under test (H-004).
VOCABULARY = {
    "open_classification": "Open",
    "classes": [
        {
            "code": "PLOTHOLDER",
            "name": "Allotment Tenant Identity",
            "classification": "Sealed",
            "personal_information": True,
            "direct_identifier": True,
        },
        {
            "code": "SITEPLAN",
            "name": "Published Site Plan Reference",
            "classification": "Open",
            "personal_information": False,
            "direct_identifier": False,
        },
    ],
}

# Written with the columns a spreadsheet export really has: an id, a code column and a
# separate free-text tier column that must AGREE with the code, plus three columns only
# the dictionary loader reads. The third row declares no code at all -- the open tier --
# so "every entry got a code" cannot pass by attaching one to everything.
GLOSSARY = (
    "id,business_name,definition,data_type,protection_class,classification,"
    "Sample Values,Synonyms,Parent Table\n"
    "AAT-1,Plot Holder Name,The full name of the tenant holding an allotment plot,"
    "string,PLOTHOLDER,Sealed,"
    '"A. Bell, C. Dray",'
    '"tenant name, holder name",'
    "tenancy\n"
    "AAT-2,Site Plan Reference,Identifier of a plot on the published site plan,"
    "string,SITEPLAN,Open,"
    '"SP-01, SP-02",'
    '"plan ref",'
    "sitemap\n"
    "AAT-3,Shed Colour Note,Free-text note about the colour a tenant painted their shed,"
    "string,,,"
    '"green",'
    '"shed note",'
    "tenancy\n"
)

# The NM-0028 shape: the code says PLOTHOLDER, which derives "Sealed"; the row claims the
# open tier. Honouring it publishes a tenant's name.
CONTRADICTING_GLOSSARY = GLOSSARY.replace("string,PLOTHOLDER,Sealed,", "string,PLOTHOLDER,Open,")

SCHEMA = {
    "type": "record",
    "name": "Tenancy",
    "fields": [
        {
            "name": "plot_holder_nm",
            "type": "string",
            "doc": "The full name of the tenant holding an allotment plot",
        }
    ],
}

# One reserved dimension per entry, so the ranking is fixed by construction rather than
# learned. Nothing in this file asserts a score; the vectors exist only so that the field
# below lands on a known entry.
_VECTORS = {
    "Plot Holder Name": (1.0, 0.0, 0.0, 0.0),
    "Site Plan Reference": (0.0, 1.0, 0.0, 0.0),
    "Shed Colour Note": (0.0, 0.0, 1.0, 0.0),
}
_QUERY = (0.90, 0.30, 0.20, 0.0)


class _StubProvider:
    """Model-free encoder with distinct vectors: no download, no network, no ties."""

    dimension = 4
    model_name = "stub"

    def _vector(self, text: str) -> tuple[float, ...]:
        for name, vector in _VECTORS.items():
            if name in text:
                return vector
        return _QUERY

    def embed(self, texts):
        rows = np.array([self._vector(t) for t in texts], dtype=np.float32)

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        return Result.success(np.array(self._vector(text), dtype=np.float32))


def _matcher(governance=VOCABULARY) -> NexusMatcher:
    """A matcher wired the way `from_config` wires one, minus the 33 MB encoder."""
    return NexusMatcher(
        embedding_provider=_StubProvider(),
        vector_store=InMemoryVectorStore(
            VectorStoreConfig(collection_name="dictionary", dimension=4)
        ),
        schema_parser_registry={"avro": AvroSchemaParser()},
        dictionary_loader_registry={"csv": CsvDictionaryLoader()},
        config=MatchingConfig(results_per_field=3),
        governance=governance,
    )


def _glossary(tmp_path, text=GLOSSARY, name="glossary.csv") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _codes(matcher: NexusMatcher) -> dict[str, str | None]:
    return {e.business_name: e.governance_code for e in matcher._dictionary_entries.values()}


def test_the_premise_this_glossary_really_does_declare_codes_the_vocabulary_defines(tmp_path):
    """
    Guards this file against passing vacuously.

    Every assertion below is about a code SURVIVING the documented loading path. If the
    fixture declared codes the vocabulary rejects, or the reader could not find the
    column, "the codes arrived" would be a claim about nothing.
    """
    entries = ingest.load_entries(_glossary(tmp_path), governance=VOCABULARY)

    assert {e.business_name: e.governance_code for e in entries} == {
        "Plot Holder Name": "PLOTHOLDER",
        "Site Plan Reference": "SITEPLAN",
        "Shed Colour Note": None,
    }


def test_load_dictionary_attaches_the_code_the_configured_vocabulary_defines(tmp_path):
    """THE DEFECT. Two documented lines, and 0 of 3 entries carried a code."""
    matcher = _matcher()

    matcher.load_dictionary(_glossary(tmp_path))

    assert _codes(matcher) == {
        "Plot Holder Name": "PLOTHOLDER",
        "Site Plan Reference": "SITEPLAN",
        # No code in the source: the open tier, and NOT the same state as the defect.
        "Shed Colour Note": None,
    }


def test_a_matched_field_inherits_the_protection_class_at_the_caller_boundary(tmp_path):
    """
    The symptom where a caller meets it. `governance_code` on a private dict is the
    mechanism; `MatchResult.governance` is the contract, and it was None on every
    candidate of every field.

    The tier is asserted against the literal the vocabulary declares, not read back off
    the class, so a vocabulary that derived the wrong tier could not satisfy this.
    """
    matcher = _matcher()
    matcher.load_dictionary(_glossary(tmp_path))

    results = matcher.match_schema(SCHEMA)
    top = next(iter(results.values()))[0]

    assert top.dictionary_entry.business_name == "Plot Holder Name"
    assert top.governance_id == "AAT-1"
    assert top.governance is not None
    assert top.governance.code == "PLOTHOLDER"
    assert top.governance.classification == "Sealed"
    assert top.governance.direct_identifier is True


def test_load_dictionary_refuses_a_self_contradicting_row_exactly_as_load_entries_does(tmp_path):
    """
    THE INVARIANT, and the reason this is not merely a lost-metadata bug. A path that
    attaches no codes also runs no derivation check, so the row NM-0028 refuses loaded
    here without a word and the matcher went on to confer a tier the row's own code
    disowns.

    Compared against the refusal `load_entries` gives for the same file rather than
    required to be "a ValueError": a load that refused for some unrelated reason would
    still have bypassed the derivation check, and an assertion that could not tell those
    apart would go green on the bypass.
    """
    path = _glossary(tmp_path, CONTRADICTING_GLOSSARY, "contradicting.csv")

    with pytest.raises(ValueError) as from_load:
        ingest.load_entries(path, governance=VOCABULARY)
    with pytest.raises(ValueError) as from_dictionary:
        _matcher().load_dictionary(path)

    assert "contradicts itself" in str(from_dictionary.value)
    assert str(from_dictionary.value) == str(from_load.value)


def test_no_entry_survives_carrying_the_contradicted_tier(tmp_path):
    """
    The consequence, stated as the thing that must not exist. A load that logged and
    carried on would satisfy the refusal above while leaving the matcher holding the row.
    """
    matcher = _matcher()

    with pytest.raises(ValueError):
        matcher.load_dictionary(_glossary(tmp_path, CONTRADICTING_GLOSSARY, "contradicting.csv"))

    assert matcher._dictionary_entries == {}


def test_a_code_column_with_no_vocabulary_configured_is_refused_here_too(tmp_path):
    """
    The other door to the same silence, and the one that needs no mis-wiring: never
    configure a vocabulary, point the matcher at a glossary whose header plainly says
    `protection_class`, and every entry comes back with no class. `load_entries` has
    refused exactly this since NM-0030; this path did not run the check at all.

    Byte-identical to `load_entries`' refusal, for the same reason as above.
    """
    path = _glossary(tmp_path)

    with pytest.raises(ValueError) as from_load:
        ingest.load_entries(path)
    with pytest.raises(ValueError) as from_dictionary:
        _matcher(governance=None).load_dictionary(path)

    assert "protection-code column" in str(from_dictionary.value)
    assert str(from_dictionary.value) == str(from_load.value)


def test_the_loaders_own_fields_survive_the_governance_pass(tmp_path):
    """
    THE CONTROL: the fix adds governance and changes NOTHING ELSE about an entry.

    The obvious fix is to read the glossary through `ingest.load_entries` and drop the
    dictionary loader. Measured, that swap also drops `sample_values`, `synonyms` and
    `parent_table` -- `load_entries` maps none of the three -- and replaces the loader's
    `auto_N` ids with content digests. Those are silent changes to what an entry
    CONTAINS, made under cover of a governance fix, and the first four assertions pin
    them so the next person to reach for that swap sees it fail here rather than in a
    caller's match quality. Those four hold in the defective state too, which is the
    point: they describe what must not move.

    The fifth is here because this glossary has NO id column, which is the case where the
    two readings can only be lined up by business name. Half the glossaries in the world
    are that shape, and a join that worked only where an id column happened to exist
    would leave them carrying no class -- this defect again, in a narrower doorway.
    """
    matcher = _matcher()
    # Same glossary, id column removed, so the loader's own id derivation is exercised.
    without_ids = "\n".join(line.split(",", 1)[1] for line in GLOSSARY.strip().splitlines())
    matcher.load_dictionary(_glossary(tmp_path, without_ids + "\n", "no_ids.csv"))

    entries = {e.business_name: e for e in matcher._dictionary_entries.values()}
    holder = entries["Plot Holder Name"]

    assert holder.id == "auto_1"
    assert holder.sample_values == ("A. Bell", "C. Dray")
    assert holder.synonyms == frozenset({"tenant name", "holder name"})
    assert holder.parent_table == "tenancy"
    # And the governance still arrived, on a source whose rows the two readers identify
    # by name alone.
    assert holder.governance_code == "PLOTHOLDER"
