"""
tests.unit.application.test_ingest_admission_and_metadata | Layer: TEST
Row admission, per-column value delimiters, and the bounds on the pass-through plane.

Three loader behaviours, each of which fails SILENTLY when it is wrong, which is why
each one is pinned by a test that asserts the loud outcome rather than the quiet one:

  * **Admission.** A glossary carries drafts and retired terms alongside approved ones.
    Indexing them pollutes retrieval in a way no threshold can repair -- the distractor
    scores like a real term because it IS a real term, just not one anyone approved. A
    filter that quietly drops nine rows in ten is indistinguishable from a broken file, so
    the counts are reported and a filter that admits nothing refuses instead of returning
    an empty glossary.

  * **Delimiters.** One file, two multi-value columns, two different separators. Reading
    a `;`-separated column as comma-separated does not raise -- it yields one giant value
    that still indexes, still matches, and is simply wrong. There is nothing downstream
    that can notice, so the loader has to.

  * **The pass-through plane.** It is carried, never interpreted, and therefore never
    hashed -- editing a value in it must not re-embed a single row. It is also bounded:
    an undeclared mapping that rakes a whole spreadsheet row into every entry more than
    quadruples what the index costs to hold (measured below).

Vocabulary here is invented -- Thornbury Water Authority does not exist, and TXN / AMT /
CUST are placeholders. What is under test is the STRUCTURE.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from nexus_matcher.application import ingest

# A glossary with the two shapes this file is about: a status column that decides which
# rows are real, and two multi-value columns that disagree about their separator.
#
# `Permitted Values` is `;`-separated. `Sample Values` is `,`-separated. In the same file,
# on the same row. That is not a contrived fixture -- it is the shape that motivates a
# per-column delimiter in the first place, and reading either one with the other's
# separator produces a plausible-looking value rather than an error.
TRAP_CSV = """Term,Business Definition,Status,Permitted Values,Sample Values,Steward
Transaction Amount,The gross amount of a transaction,Approved,DEBIT;CREDIT;REVERSAL,"12.00,145.50,0.00",ledger.team
Customer Identifier,The identifier assigned to a customer,Approved,ACTIVE;DORMANT;CLOSED,"C0001,C0002",cust.team
Transaction Channel,The channel a transaction arrived on,Draft,BRANCH;ONLINE;PHONE,"BR,ON,PH",ledger.team
Legacy Batch Marker,A marker used by the retired batch loader,Retired,ON;OFF,"Y,N",ledger.team
"""


@pytest.fixture
def trap(tmp_path):
    p = tmp_path / "glossary.csv"
    p.write_text(TRAP_CSV, encoding="utf-8")
    return p


class TestRowAdmission:
    def test_only_admitted_rows_become_entries(self, trap):
        entries = ingest.load_entries(trap, admit={"Status": {"Approved"}})
        assert sorted(e.business_name for e in entries) == [
            "Customer Identifier",
            "Transaction Amount",
        ], "a draft and a retired term reached the index"

    def test_the_report_counts_what_it_admitted_and_what_it_refused(self, trap):
        report = ingest.LoadReport()
        ingest.load_entries(trap, admit={"Status": {"Approved"}}, report=report)
        assert report.rows_read == 4
        assert report.admitted == 2
        assert report.refused == 2
        assert report.refused_by_column == {"Status": 2}
        # The count has to be visible in the one line a caller actually prints, or the
        # 90%-dropped case stays invisible in exactly the deployment that needs it.
        assert "2 refused" in str(report)

    def test_admission_ignores_case_and_surrounding_space(self, tmp_path):
        """
        A trailing space in an export is not a policy decision, and refusing every row
        over one is the same outcome as a broken file.
        """
        p = tmp_path / "spaced.csv"
        p.write_text(
            "Term,Business Definition,Status\n"
            "Transaction Amount,The gross amount, approved \n"
            "Customer Identifier,The customer identifier,APPROVED\n",
            encoding="utf-8",
        )
        entries = ingest.load_entries(p, admit={"Status": {"Approved"}})
        assert len(entries) == 2

    def test_several_columns_must_all_admit(self, trap):
        entries = ingest.load_entries(
            trap, admit={"Status": {"Approved"}, "Steward": {"cust.team"}}
        )
        assert [e.business_name for e in entries] == ["Customer Identifier"]

    def test_an_admission_column_the_file_does_not_have_is_refused(self, trap):
        """
        Otherwise every row is refused and the caller gets an empty glossary that looks
        exactly like a glossary of drafts.
        """
        with pytest.raises(ValueError) as excinfo:
            ingest.load_entries(trap, admit={"Lifecycle": {"Approved"}})
        message = str(excinfo.value)
        assert "Lifecycle" in message
        assert "Status" in message, "the refusal must show the columns the file DOES have"

    def test_an_accepted_set_with_nothing_in_it_is_refused(self, trap):
        with pytest.raises(ValueError, match="accepts nothing"):
            ingest.load_entries(trap, admit={"Status": set()})

    def test_a_filter_that_admits_nothing_refuses_and_says_what_it_saw(self, trap):
        """
        The single most useful thing a refusal can carry here is the values that are
        actually in the column, because the bug is nearly always that the deployment
        typed the neighbouring vocabulary.
        """
        with pytest.raises(ValueError) as excinfo:
            ingest.load_entries(trap, admit={"Status": {"Published"}})
        message = str(excinfo.value)
        assert "Published" in message
        assert "Approved" in message and "Draft" in message

    def test_no_filter_leaves_every_row_where_it_was(self, trap):
        entries = ingest.load_entries(trap)
        assert len(entries) == 4


class TestPerColumnDelimiters:
    def test_two_columns_two_delimiters_in_one_file(self, trap):
        entries = ingest.load_entries(
            trap,
            columns={"sample_values": "Sample Values", "enum_values": "Permitted Values"},
            value_delimiters={"sample_values": ",", "enum_values": ";"},
            admit={"Status": {"Approved"}},
        )
        by_name = {e.business_name: e for e in entries}
        amount = by_name["Transaction Amount"]
        assert amount.enum_values == ("DEBIT", "CREDIT", "REVERSAL")
        assert amount.sample_values == ("12.00", "145.50", "0.00")

    def test_the_wrong_delimiter_is_refused_rather_than_indexed(self, trap):
        """
        The failure this exists to stop: `;`-separated values read as comma-separated
        produce ONE value containing every element, which indexes and matches and is
        simply wrong. Nothing downstream can tell.
        """
        with pytest.raises(ValueError) as excinfo:
            ingest.load_entries(
                trap,
                columns={"enum_values": "Permitted Values"},
                value_delimiters={"enum_values": ","},
            )
        message = str(excinfo.value)
        assert "Permitted Values" in message
        assert "';'" in message, "the refusal must name the separator the data actually uses"

    def test_the_giant_value_is_what_you_get_if_you_insist(self, trap):
        """
        Pinning the harm, so the refusal above is not just a rule with no consequence
        behind it. This is the state the loader refuses to reach by accident.
        """
        entries = ingest.load_entries(
            trap,
            columns={"enum_values": "Permitted Values"},
            value_delimiters={"enum_values": ","},
            delimiter_strict=False,
        )
        assert entries[0].enum_values == ("DEBIT;CREDIT;REVERSAL",)

    def test_an_empty_separator_is_refused_at_configuration_time(self, trap):
        """
        The only way `str.split` shreds a value into single characters is an empty
        separator, and Python's own message for it names neither the column nor the field.
        """
        with pytest.raises(ValueError) as excinfo:
            ingest.load_entries(
                trap,
                columns={"enum_values": "Permitted Values"},
                value_delimiters={"enum_values": ""},
            )
        assert "enum_values" in str(excinfo.value)

    def test_a_delimiter_for_a_field_that_is_not_multi_valued_is_refused(self, trap):
        """A typo in configuration should not be a silently ignored declaration."""
        with pytest.raises(ValueError) as excinfo:
            ingest.load_entries(trap, value_delimiters={"definition": ";"})
        assert "definition" in str(excinfo.value)

    def test_synonyms_cannot_be_mapped_and_the_refusal_says_why(self, trap):
        """
        `DictionaryEntry.synonyms` is a frozenset and `to_searchable_text()` emits it
        unordered, so populating it would make the embedded text -- and every vector and
        content hash derived from it -- differ between processes. Measured: five
        interpreters, five different hashes for one entry with four synonyms. Refusing is
        the honest answer until the ordering is fixed where it lives.
        """
        with pytest.raises(ValueError) as excinfo:
            ingest.load_entries(trap, columns={"synonyms": "Permitted Values"})
        assert "synonyms" in str(excinfo.value)

    def test_a_column_too_small_to_judge_is_not_second_guessed(self, tmp_path):
        """
        Two rows is not evidence of a delimiter convention. Refusing on it would make the
        check fire on fixtures and small pilots, which is how a good check gets disabled.
        """
        p = tmp_path / "small.csv"
        p.write_text(
            "Term,Business Definition,Codes\n"
            "Transaction Amount,The gross amount,DEBIT;CREDIT\n"
            "Customer Identifier,The customer identifier,ACTIVE;CLOSED\n",
            encoding="utf-8",
        )
        entries = ingest.load_entries(
            p, columns={"enum_values": "Codes"}, value_delimiters={"enum_values": ","}
        )
        assert entries[0].enum_values == ("DEBIT;CREDIT",)


class TestTheMetadataPlaneIsDeclared:
    def test_declaring_columns_makes_them_the_whole_plane(self, trap):
        entries = ingest.load_entries(trap, metadata_columns=["Steward"])
        assert entries[0].source_metadata == {"Steward": "ledger.team"}

    def test_a_declared_column_the_file_does_not_have_is_refused(self, trap):
        with pytest.raises(ValueError) as excinfo:
            ingest.load_entries(trap, metadata_columns=["Steward", "Reviewer"])
        assert "Reviewer" in str(excinfo.value)

    def test_a_declared_column_wins_even_when_it_is_also_a_mapped_field(self, trap):
        """
        Declaring a column is a statement that the deployment wants it back. That it also
        feeds a domain field is not a reason to withhold it -- the enum is lossy, and the
        loader already keeps the raw classification string for exactly this reason.
        """
        entries = ingest.load_entries(trap, metadata_columns=["Term"])
        assert entries[0].source_metadata["Term"] == "Transaction Amount"

    def test_declaring_nothing_keeps_the_behaviour_every_caller_already_has(self, trap):
        entries = ingest.load_entries(trap)
        assert entries[0].source_metadata == {
            "Status": "Approved",
            "Permitted Values": "DEBIT;CREDIT;REVERSAL",
            "Sample Values": "12.00,145.50,0.00",
            "Steward": "ledger.team",
        }


class TestTheMetadataPlaneIsBounded:
    # 1,024 bytes of key + value. Measured against a realistic enterprise pass-through
    # payload -- twelve short enrichment columns, identifiers, two flags and five sample
    # values -- which costs 330 bytes. The cap is 3.1x that, and 10x the 103-byte median
    # of the example pack in examples/governance/.
    CAP: ClassVar[int] = 1024

    def _wide(self, tmp_path, filler: int):
        p = tmp_path / "wide.csv"
        p.write_text(
            "Term,Business Definition,Rationale,Note\n"
            f"Transaction Amount,The gross amount,{'x' * filler},{'y' * 40}\n",
            encoding="utf-8",
        )
        return p

    def test_the_cap_is_what_this_module_says_it_is(self):
        assert ingest.METADATA_MAX_BYTES == self.CAP

    def test_a_row_inside_the_cap_is_untouched(self, tmp_path):
        entries = ingest.load_entries(self._wide(tmp_path, 100))
        assert entries[0].source_metadata["Rationale"] == "x" * 100
        assert "metadata_truncated" not in entries[0].source_metadata

    def test_a_careless_mapping_is_bounded_and_says_so(self, tmp_path):
        report = ingest.LoadReport()
        entries = ingest.load_entries(self._wide(tmp_path, 4000), report=report)
        metadata = entries[0].source_metadata
        assert ingest.metadata_bytes(metadata) <= self.CAP
        assert "Rationale" not in metadata, "the largest key must be the one that goes"
        assert metadata["Note"] == "y" * 40, "a small key must survive a large neighbour"
        assert metadata["metadata_truncated"] == 1
        assert report.metadata_truncated == 1

    def test_truncation_drops_the_largest_first_and_deterministically(self, tmp_path):
        """
        Two keys have to go here, and the order they go in is the whole point: dropping
        the largest first keeps the most keys, and the tie-break on name makes the
        survivors identical run to run. A plane that reordered itself between builds
        would be the thing that broke byte-stable output.
        """
        p = tmp_path / "two_big.csv"
        p.write_text(
            "Term,Business Definition,Bigger,Big,Small\n"
            f"Transaction Amount,The gross amount,{'x' * 2000},{'y' * 1500},keep\n",
            encoding="utf-8",
        )
        first = ingest.load_entries(p)[0].source_metadata
        second = ingest.load_entries(p)[0].source_metadata
        assert list(first) == list(second), "the surviving keys must not depend on the run"
        assert set(first) == {"Small", "metadata_truncated"}
        assert first["metadata_truncated"] == 2
        assert ingest.metadata_bytes(first) <= self.CAP

    def test_truncation_stops_as_soon_as_the_map_fits(self, tmp_path):
        """The cap is a bound, not a budget to spend: no key goes that did not have to."""
        p = tmp_path / "one_big.csv"
        p.write_text(
            "Term,Business Definition,Bigger,Big,Small\n"
            f"Transaction Amount,The gross amount,{'x' * 900},{'y' * 800},keep\n",
            encoding="utf-8",
        )
        metadata = ingest.load_entries(p)[0].source_metadata
        assert set(metadata) == {"Big", "Small", "metadata_truncated"}
        assert metadata["metadata_truncated"] == 1

    def test_the_governance_evidence_keys_are_never_the_ones_dropped(self, tmp_path):
        """
        `governance_code_raw` is the ONLY place a rejected token survives, and
        `governance_problems` is the evidence for whoever fixes the source file. Dropping
        either to make room for a spreadsheet column would destroy the reason the row was
        refused.
        """
        p = tmp_path / "gov.csv"
        p.write_text(
            "Term,Business Definition,Protection Class,Rationale\n"
            f"Meter Serial,The serial stamped on a meter,METERID,{'x' * 4000}\n",
            encoding="utf-8",
        )
        entries = ingest.load_entries(
            p,
            governance={
                "open_classification": "Open",
                "classes": [
                    {
                        "code": "METERID",
                        "name": "Meter Serial Identifier",
                        "classification": "Sealed",
                        "personal_information": True,
                        "direct_identifier": True,
                    }
                ],
            },
        )
        assert entries[0].source_metadata["governance_code_raw"] == "METERID"
        assert "Rationale" not in entries[0].source_metadata

    def test_the_cap_can_be_lifted_by_a_caller_who_means_it(self, tmp_path):
        entries = ingest.load_entries(self._wide(tmp_path, 4000), metadata_max_bytes=None)
        assert entries[0].source_metadata["Rationale"] == "x" * 4000

    def test_the_reserved_keys_are_published_for_the_response_side(self):
        """
        The half of the plane that emits it needs to know which keys the LOADER wrote, so
        it can distinguish them from the caller's own columns. Naming them in two places
        is how they drift.
        """
        assert set(ingest.METADATA_RESERVED_KEYS) == {
            "governance_raw",
            "governance_code_raw",
            "governance_problems",
            "metadata_truncated",
        }


class _CountingProvider:
    """An encoder-shaped object that counts how many texts it was asked to encode."""

    dimension = 4
    model_name = "counting"

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def embed_documents(self, texts):
        import numpy as np

        texts = list(texts)
        self.encoded.extend(texts)
        return np.ones((len(texts), self.dimension), dtype="float32")


class _OrthogonalProvider:
    """
    Model-free encoder with one reserved dimension per entry, so the ranking is fixed by
    construction rather than learned. No download, no network, no ties.
    """

    dimension = 4
    model_name = "orthogonal"

    VECTORS: ClassVar[dict[str, tuple[float, ...]]] = {
        "Transaction Amount": (1.0, 0.0, 0.0, 0.0),
        "Customer Identifier": (0.0, 1.0, 0.0, 0.0),
        "Transaction Channel": (0.0, 0.0, 1.0, 0.0),
    }
    QUERY: ClassVar[tuple[float, ...]] = (0.80, 0.55, 0.30, 0.10)

    def _vector(self, text: str) -> tuple[float, ...]:
        for name, vector in self.VECTORS.items():
            if name in text:
                return vector
        return self.QUERY

    def embed(self, texts):
        import numpy as np

        from nexus_matcher.shared.types.base import Result

        rows = np.array([self._vector(t) for t in texts], dtype="float32")

        class _Batch:
            embeddings = rows

        return Result.success(_Batch())

    def embed_single(self, text):
        import numpy as np

        from nexus_matcher.shared.types.base import Result

        return Result.success(np.array(self._vector(text), dtype="float32"))


class TestTheCoreNeverReadsThePlane:
    """
    AR-1's first rule, and the one that makes the rest of the plane safe: no scoring,
    filtering, thresholding or governance decision may depend on a key in this map.

    The rule is not a style preference. A pass-through map that the core reads is an
    enterprise-specific back door through a hexagonal boundary: the value has no defined
    meaning, no validation and no name in the domain, and the first deployment that
    depends on it makes every other deployment's results depend on a column it has never
    heard of. If the core needs a value, it gets promoted to a first-class field with a
    stated meaning, the way `governance_code` was.

    Asserted BEHAVIOURALLY -- two indexes identical but for the plane must produce
    identical matches -- rather than by grepping the source for `source_metadata`. A grep
    catches the spelling; this catches the dependency however it is spelled.
    """

    ROWS: ClassVar[list[dict[str, str]]] = [
        {
            "Term": "Transaction Amount",
            "Business Definition": "The gross amount",
            "Data Type": "string",
        },
        {
            "Term": "Customer Identifier",
            "Business Definition": "The customer identifier",
            "Data Type": "string",
        },
        {
            "Term": "Transaction Channel",
            "Business Definition": "The arrival channel",
            "Data Type": "string",
        },
    ]

    def _matches(self, *, plane: bool, data_type: str = "string"):
        from nexus_matcher.application.use_cases.match_schema import MatchingConfig, NexusMatcher
        from nexus_matcher.domain.models.entities import SchemaField
        from nexus_matcher.domain.ports.vector_store import VectorStoreConfig
        from nexus_matcher.infrastructure.adapters.vector_stores.memory import InMemoryVectorStore
        from nexus_matcher.shared.types.base import DataType

        rows = [
            {
                **row,
                "Data Type": data_type,
                **(
                    {
                        "Steward": f"steward.{i}",
                        "Ledger Ref": f"LR-{i:05d}",
                        "Rationale": "carried, never interpreted",
                    }
                    if plane
                    else {}
                ),
            }
            for i, row in enumerate(self.ROWS)
        ]
        entries = ingest.load_entries(rows)
        assert bool(entries[0].source_metadata) is plane, "fixture did not vary the plane"

        matcher = NexusMatcher(
            embedding_provider=_OrthogonalProvider(),
            vector_store=InMemoryVectorStore(
                VectorStoreConfig(collection_name="dictionary", dimension=4)
            ),
            config=MatchingConfig(results_per_field=3),
        )
        matcher._index_dictionary(entries)
        results = matcher._match_field(
            SchemaField(
                name="txn_amt",
                data_type=DataType.STRING,
                full_path="ledger.txn_amt",
                parent_path="ledger",
                description="The gross amount of a transaction",
            )
        )
        return [
            (m.rank, m.dictionary_entry.id, m.decision, round(float(m.final_confidence), 12))
            for m in results
        ]

    def test_a_loaded_plane_changes_no_rank_no_decision_and_no_score(self):
        assert self._matches(plane=False) == self._matches(plane=True)

    def test_the_comparison_can_actually_see_a_difference(self):
        """
        The control, and it is not decoration: the first version of this test varied the
        DEFINITION, and the two runs compared equal -- because the stub encoder keys on
        the business name, so the definition never reached a score. A test that cannot
        fail proves nothing, and that one could not.

        `data_type` is a first-class field the scorer genuinely reads (`matches_type`,
        weight 0.05) and it does not feed the derived id, so the comparison moves on the
        score alone.
        """
        assert self._matches(plane=False) != self._matches(plane=False, data_type="integer")


class TestTheMetadataPlaneIsExemptFromTheContentHash:
    """
    AR-1's fourth rule, and the same reasoning that already exempts `governance_code`:
    changing a value nobody embeds must not re-embed anything.

    NM-0024 pins this on a hand-built entry. These pin it end to end, through the loader
    and through `sync`, because the hash is only exempt if the value never reaches
    `to_searchable_text()` -- and the loader is what decides where a column lands.
    """

    HEADER = "Term,Business Definition,Sample Values,Reviewer\n"

    def _write(self, path, reviewer="alice", samples="1,2", definition="The gross amount"):
        path.write_text(
            f'{self.HEADER}Transaction Amount,{definition},"{samples}",{reviewer}\n',
            encoding="utf-8",
        )
        return path

    def test_editing_a_pass_through_value_encodes_nothing(self, tmp_path):
        source = self._write(tmp_path / "g.csv")
        provider = _CountingProvider()
        index = ingest.build_index(source, provider=provider, metadata_columns=["Reviewer"])
        assert len(provider.encoded) == 1

        self._write(source, reviewer="bruno")
        report = ingest.sync(index, source)

        assert len(provider.encoded) == 1, "a pass-through edit re-embedded a row"
        assert report.embedded == 0
        assert index.entries["".join(index.order)].source_metadata["Reviewer"] == "bruno", (
            "the new value must still reach the entry -- exempt from the hash is not "
            "exempt from being refreshed"
        )

    def test_editing_sample_values_encodes_nothing(self, tmp_path):
        source = self._write(tmp_path / "g.csv")
        provider = _CountingProvider()
        index = ingest.build_index(
            source,
            provider=provider,
            columns={"sample_values": "Sample Values"},
            value_delimiters={"sample_values": ","},
        )
        assert len(provider.encoded) == 1

        self._write(source, samples="7,8,9")
        report = ingest.sync(index, source)

        assert report.embedded == 0
        assert index.entries["".join(index.order)].sample_values == ("7", "8", "9")

    def test_editing_the_definition_still_encodes(self, tmp_path):
        """The control. A hash so exempt that nothing moves it is not a hash."""
        source = self._write(tmp_path / "g.csv")
        provider = _CountingProvider()
        index = ingest.build_index(source, provider=provider)
        ingest.sync(index, source)  # no change
        assert len(provider.encoded) == 1

        self._write(source, definition="The net amount after adjustments")
        report = ingest.sync(index, source)
        assert report.embedded == 1


class TestTheIndexRemembersWhatTheLoadDid:
    def test_build_index_attaches_the_report(self, trap):
        index = ingest.build_index(
            trap, provider=_CountingProvider(), admit={"Status": {"Approved"}}
        )
        assert index.load_report is not None
        assert index.load_report.admitted == 2
        assert index.load_report.refused == 2

    def test_sync_replaces_it_rather_than_accumulating(self, trap):
        index = ingest.build_index(
            trap, provider=_CountingProvider(), admit={"Status": {"Approved"}}
        )
        ingest.sync(index, trap)
        assert index.load_report is not None
        assert index.load_report.admitted == 2, "counts doubled across a refresh"

    def test_the_report_is_not_remembered_as_a_load_option(self, trap):
        """
        `load_options` is replayed on every `sync`. A mutable report living in it would
        accumulate counts from every refresh the index ever did.
        """
        report = ingest.LoadReport()
        index = ingest.build_index(trap, provider=_CountingProvider(), report=report)
        assert "report" not in index.load_options
