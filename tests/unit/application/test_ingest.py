"""
tests.unit.application.test_ingest | Layer: TEST
Reading heterogeneous glossaries, and re-embedding only what changed.

Two behaviours here are worth more than the rest:

  * The content hash must cover the EMBEDDED TEXT only. If it covered the whole row, the
    first edit to an audit column would invalidate every vector and incremental sync
    would silently degrade into a full rebuild -- the exact cost it exists to avoid.

  * The governance/classification column must survive. An earlier version MAPPED it,
    which excluded it from source_metadata, and then never used it -- so the field a
    governance-inheritance workflow most needs was dropped without a word.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from nexus_matcher.application import ingest
from nexus_matcher.domain.models.entities import DictionaryEntry
from nexus_matcher.shared.types.base import DataType, ProtectionLevel

GLOSSARY_CSV = """Term,Business Definition,Subject Area,Technical Name,Data Type,Classification
Customer Identifier,Unique id assigned to a customer,Customer,cust_id,integer,Internal
Email Address,Primary contact email,Customer,email_addr,varchar(255),PII
Account Balance,Current monetary balance,Finance,acct_bal,decimal,Confidential
"""


@pytest.fixture
def glossary(tmp_path):
    p = tmp_path / "glossary.csv"
    p.write_text(GLOSSARY_CSV, encoding="utf-8")
    return p


class TestColumnMapping:
    def test_infers_real_world_headers(self):
        mapping = ingest.map_columns(
            ["Term", "Business Definition", "Subject Area", "Technical Name"]
        )
        assert mapping["business_name"] == "Term"
        assert mapping["definition"] == "Business Definition"
        assert mapping["domain"] == "Subject Area"
        assert mapping["logical_name"] == "Technical Name"

    @pytest.mark.parametrize(
        "header", ["business_name", "Business Name", "BUSINESS-NAME", "businessName"]
    )
    def test_matching_ignores_case_and_punctuation(self, header):
        assert ingest.map_columns([header]).get("business_name") == header

    def test_a_column_is_claimed_only_once(self):
        """'name' could match several fields; it must not be assigned twice."""
        mapping = ingest.map_columns(["name", "description"])
        assert len(set(mapping.values())) == len(mapping)

    def test_unmatched_headers_are_simply_absent(self):
        mapping = ingest.map_columns(["wibble", "flurb"])
        assert mapping == {}


class TestReading:
    def test_csv(self, glossary):
        rows, header = ingest.read_source(glossary)
        assert len(rows) == 3
        assert "Term" in header

    def test_csv_with_bom(self, tmp_path):
        """Excel writes a BOM that corrupts the first header unless handled."""
        p = tmp_path / "bom.csv"
        p.write_bytes(b"\xef\xbb\xbfTerm,Business Definition\nA,B\n")
        _, header = ingest.read_source(p)
        assert header[0] == "Term", f"BOM leaked into the header: {header[0]!r}"

    def test_tsv(self, tmp_path):
        p = tmp_path / "g.tsv"
        p.write_text("Term\tBusiness Definition\nA\tB\n", encoding="utf-8")
        rows, _ = ingest.read_source(p)
        assert rows[0]["Term"] == "A"

    def test_jsonl(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text('{"Term":"A","Business Definition":"B"}\n', encoding="utf-8")
        assert ingest.read_source(p)[0][0]["Term"] == "A"

    def test_json_envelope_is_unwrapped(self, tmp_path):
        p = tmp_path / "g.json"
        p.write_text(json.dumps({"entries": [{"Term": "A"}]}), encoding="utf-8")
        assert ingest.read_source(p)[0][0]["Term"] == "A"

    def test_iterable_of_dicts(self):
        rows, header = ingest.read_source([{"Term": "A", "Business Definition": "B"}])
        assert rows[0]["Term"] == "A"
        assert set(header) == {"Term", "Business Definition"}

    def test_unsupported_extension_names_what_is_supported(self, tmp_path):
        p = tmp_path / "g.docx"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.csv"):
            ingest.read_source(p)


class TestLoadEntries:
    def test_builds_entries(self, glossary):
        entries = ingest.load_entries(glossary)
        assert len(entries) == 3
        assert entries[0].business_name == "Customer Identifier"
        assert entries[0].logical_name == "cust_id"
        assert entries[0].domain == "Customer"

    def test_coerces_data_types(self, glossary):
        by_name = {e.business_name: e for e in ingest.load_entries(glossary)}
        assert by_name["Customer Identifier"].data_type == DataType.INTEGER
        assert by_name["Email Address"].data_type == DataType.STRING  # varchar(255)
        assert by_name["Account Balance"].data_type == DataType.DECIMAL

    def test_rows_with_no_name_and_no_definition_are_skipped(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text("Term,Business Definition\nReal,A definition\n,\n", encoding="utf-8")
        assert len(ingest.load_entries(p)) == 1

    def test_explicit_mapping_overrides_inference(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text("colA,colB\nName here,Definition here\n", encoding="utf-8")
        entries = ingest.load_entries(p, columns={"business_name": "colA", "definition": "colB"})
        assert entries[0].business_name == "Name here"

    def test_unmappable_source_says_what_to_do(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text("wibble,flurb\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="columns="):
            ingest.load_entries(p)

    def test_unmapped_columns_are_preserved(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text("Term,Business Definition,Owner\nA,B,alice\n", encoding="utf-8")
        assert ingest.load_entries(p)[0].source_metadata["Owner"] == "alice"

    def test_id_prefix(self, glossary):
        entries = ingest.load_entries(glossary, id_prefix="crm::")
        assert all(e.id.startswith("crm::") for e in entries)


class TestGovernanceClassification:
    """The field the whole governance-inheritance workflow depends on."""

    def test_classification_is_mapped_to_the_enum(self, glossary):
        by_name = {e.business_name: e for e in ingest.load_entries(glossary)}
        assert by_name["Email Address"].protection_level == ProtectionLevel.PII
        assert by_name["Account Balance"].protection_level == ProtectionLevel.CONFIDENTIAL
        assert by_name["Customer Identifier"].protection_level == ProtectionLevel.INTERNAL

    def test_raw_classification_text_survives(self, glossary):
        """The enum is lossy; an org's exact wording must not be thrown away."""
        by_name = {e.business_name: e for e in ingest.load_entries(glossary)}
        assert by_name["Email Address"].source_metadata["governance_raw"] == "PII"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("PII", ProtectionLevel.PII),
            ("PHI / Sensitive", ProtectionLevel.PII),
            ("Personal Data", ProtectionLevel.PII),
            ("Confidential", ProtectionLevel.CONFIDENTIAL),
            ("Highly Confidential", ProtectionLevel.RESTRICTED),
            ("Restricted - Legal Hold", ProtectionLevel.RESTRICTED),
            ("Internal Use Only", ProtectionLevel.INTERNAL),
            ("Public", ProtectionLevel.PUBLIC),
        ],
    )
    def test_free_text_classifications(self, text, expected):
        assert ingest._coerce_protection(text) == expected

    def test_ambiguous_labels_resolve_to_the_stricter_reading(self):
        """Under-protecting a field is the expensive mistake, so strictest wins."""
        assert ingest._coerce_protection("Highly Confidential") == ProtectionLevel.RESTRICTED

    def test_unknown_and_empty_are_not_treated_as_public(self):
        assert ingest._coerce_protection("Some Bespoke Label") == ProtectionLevel.INTERNAL
        assert ingest._coerce_protection("") == ProtectionLevel.INTERNAL


class TestIncrementalSync:
    def _entry(self, eid="e1", name="Customer Identifier", definition="An id", **kw):
        return DictionaryEntry(
            id=eid,
            business_name=name,
            logical_name=kw.pop("logical_name", "cust_id"),
            definition=definition,
            data_type=DataType.STRING,
            **kw,
        )

    def test_first_sync_embeds_everything(self):
        entries = [self._entry("a"), self._entry("b", name="Other")]
        to_embed, removed, report = ingest.diff_entries({}, entries)
        assert len(to_embed) == 2
        assert report.unchanged == 0
        assert removed == []

    def test_unchanged_entries_are_not_re_embedded(self):
        entries = [self._entry("a")]
        previous = {e.id: ingest.content_hash(e) for e in entries}
        to_embed, _, report = ingest.diff_entries(previous, entries)
        assert to_embed == []
        assert report.unchanged == 1

    def test_changed_definition_triggers_exactly_one_re_embed(self):
        entries = [self._entry("a"), self._entry("b", name="Other")]
        previous = {e.id: ingest.content_hash(e) for e in entries}
        entries[0] = self._entry("a", definition="A completely different definition")
        to_embed, _, report = ingest.diff_entries(previous, entries)
        assert [e.id for e in to_embed] == ["a"]
        assert report.unchanged == 1

    def test_metadata_only_change_does_NOT_re_embed(self):
        """
        The load-bearing property. If the hash covered the whole row, editing an owner or
        review-date column would invalidate every vector and incremental sync would
        quietly become a full rebuild.
        """
        before = self._entry("a", source_metadata={"reviewed_by": "alice"})
        previous = {before.id: ingest.content_hash(before)}
        after = self._entry("a", source_metadata={"reviewed_by": "bob", "ticket": "123"})
        to_embed, _, report = ingest.diff_entries(previous, [after])
        assert to_embed == []
        assert report.unchanged == 1

    def test_governance_change_does_not_re_embed(self):
        """Reclassifying a term changes governance, not meaning."""
        before = self._entry("a", protection_level=ProtectionLevel.INTERNAL)
        previous = {before.id: ingest.content_hash(before)}
        after = self._entry("a", protection_level=ProtectionLevel.RESTRICTED)
        to_embed, _, _ = ingest.diff_entries(previous, [after])
        assert to_embed == []

    def test_removed_entries_are_reported(self):
        previous = {"gone": "somehash", "kept": ingest.content_hash(self._entry("kept"))}
        _, removed, report = ingest.diff_entries(previous, [self._entry("kept")])
        assert removed == ["gone"]
        assert report.removed == ["gone"]

    def test_hash_is_stable_and_input_sensitive(self):
        a = self._entry("a")
        assert ingest.content_hash(a) == ingest.content_hash(self._entry("a"))
        assert ingest.content_hash(a) != ingest.content_hash(self._entry("a", name="Other"))

    def test_hash_is_not_confused_by_field_boundaries(self):
        """Concatenating without a separator would collide these two."""
        one = self._entry("a", name="ab", logical_name="c")
        two = self._entry("a", name="a", logical_name="bc")
        assert ingest.content_hash(one) != ingest.content_hash(two)


class TestHashManifest:
    def test_round_trip(self, tmp_path):
        entries = [
            DictionaryEntry(
                id="a",
                business_name="A",
                logical_name="a",
                definition="d",
                data_type=DataType.STRING,
            )
        ]
        path = tmp_path / "state" / "hashes.json"
        ingest.save_hashes(path, entries)
        assert ingest.load_hashes(path) == {"a": ingest.content_hash(entries[0])}

    def test_missing_manifest_means_full_rebuild(self, tmp_path):
        assert ingest.load_hashes(tmp_path / "nope.json") == {}

    def test_corrupt_manifest_degrades_to_full_rebuild(self, tmp_path):
        """
        Never crash, and never half-apply. A partially-updated index that silently
        disagrees with its source is worse than paying for a full re-embed.
        """
        path = tmp_path / "hashes.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert ingest.load_hashes(path) == {}


class TestSyncReport:
    def test_reports_what_happened(self):
        entries = [
            DictionaryEntry(
                id=str(i),
                business_name=f"Term {i}",
                logical_name="",
                definition="d",
                data_type=DataType.STRING,
            )
            for i in range(10)
        ]
        previous = {e.id: ingest.content_hash(e) for e in entries[:8]}
        _, _, report = ingest.diff_entries(previous, entries)
        assert len(report.added) == 2
        assert report.unchanged == 8
        assert report.embedded == 2
        assert "added" in str(report)


class TestThreeLineAPI:
    """build_index / sync -- the API the module docstring promises."""

    @pytest.fixture
    def source(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text(
            "Term,Business Definition,Classification\n"
            "Customer Identifier,Unique id for a customer,Internal\n"
            "Email Address,Primary contact email,PII\n",
            encoding="utf-8",
        )
        return p

    def test_build_index(self, source):
        index = ingest.build_index(source)
        assert len(index) == 2
        assert index.vectors.shape[0] == 2
        assert len(index.order) == 2

    def test_sync_with_no_change_embeds_nothing(self, source):
        index = ingest.build_index(source)
        report = ingest.sync(index, source)
        assert report.embedded == 0
        assert report.unchanged == 2

    def test_sync_embeds_only_the_changed_row(self, source):
        index = ingest.build_index(source)
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "Primary contact email", "Primary contact email address on file"
            ),
            encoding="utf-8",
        )
        report = ingest.sync(index, source)
        assert report.embedded == 1
        assert report.unchanged == 1

    def test_sync_appends_new_rows(self, source):
        index = ingest.build_index(source)
        source.write_text(
            source.read_text(encoding="utf-8") + "Account Balance,Current balance,Confidential\n",
            encoding="utf-8",
        )
        ingest.sync(index, source)
        assert len(index) == 3
        assert index.vectors.shape[0] == 3

    def test_sync_removes_deleted_rows(self, source):
        index = ingest.build_index(source)
        source.write_text(
            "Term,Business Definition,Classification\n"
            "Customer Identifier,Unique id for a customer,Internal\n",
            encoding="utf-8",
        )
        report = ingest.sync(index, source)
        assert len(report.removed) == 1
        assert len(index) == 1

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda t: t.replace("Primary contact email", "A different definition entirely"),
            lambda t: t + "Account Balance,Current balance,Confidential\n",
            lambda t: "\n".join(t.splitlines()[:2]) + "\n",
        ],
        ids=["edit", "append", "delete"],
    )
    def test_index_stays_aligned_after_any_change(self, source, mutate):
        """
        entries, order and vectors must agree after every operation. A misalignment here
        returns the WRONG glossary entry for a query while looking perfectly healthy --
        the worst kind of failure for a tool that assigns governance classifications.
        """
        index = ingest.build_index(source)
        source.write_text(mutate(source.read_text(encoding="utf-8")), encoding="utf-8")
        ingest.sync(index, source)
        assert len(index.entries) == len(index.order) == index.vectors.shape[0]
        assert set(index.order) == set(index.entries)

    def test_vectors_still_match_their_entry_after_a_delete(self, source):
        """Deleting a row must not shift other rows onto the wrong vector."""
        import numpy as np

        source.write_text(
            "Term,Business Definition\nAlpha,First term\nBeta,Second term\nGamma,Third term\n",
            encoding="utf-8",
        )
        index = ingest.build_index(source)
        gamma_id = next(i for i, e in index.entries.items() if e.business_name == "Gamma")
        gamma_vector = index.vectors[index.order.index(gamma_id)].copy()

        source.write_text(
            "Term,Business Definition\nAlpha,First term\nGamma,Third term\n", encoding="utf-8"
        )
        ingest.sync(index, source)

        assert gamma_id in index.entries
        assert np.allclose(index.vectors[index.order.index(gamma_id)], gamma_vector)


class _ConstantProvider:
    """
    An encoder-shaped object that returns one fixed vector.

    Everything in `TestSyncRemembersHowTheIndexWasBuilt` is about BOOKKEEPING -- which
    options a refresh re-reads the source with, and what the report says -- so the bundled
    model would add seconds per test and a dependency on what a vector means. Nothing below
    reads a vector back.
    """

    dimension = 4
    model_name = "constant"

    def embed_documents(self, texts):
        import numpy as np

        return np.ones((len(list(texts)), self.dimension), dtype="float32")


class TestSyncRemembersHowTheIndexWasBuilt:
    """
    `sync(index, source)` must re-read the source the way `build_index` read it.

    `GlossaryIndex` stored the provider and nothing else, so every other option went out
    of scope the moment `build_index` returned -- and `sync`'s own docstring example is
    `sync(index, "glossary.xlsx")` with no options at all. The governance vocabulary is the
    member that mattered: measured on a byte-identical file, a 30-entry index built with
    27 coded entries came back with 0 and reported `=30 unchanged`. Nothing errored,
    because there is nothing wrong with a glossary that has no vocabulary.

    Worse than the loss: `load_entries`' refusal gate went with it, so a row whose stated
    tier contradicts its own code raised from `build_index` and loaded silently through
    `sync`. That half is pinned in tests/museum/NM-0030.

    The vocabulary here is fictional -- Thornbury Water Authority does not exist. This
    library ships no taxonomy; what is under test is the STRUCTURE.
    """

    VOCABULARY: ClassVar[dict[str, Any]] = {
        "open_classification": "Open",
        "classes": [
            {
                "code": "METERID",
                "name": "Meter Serial Identifier",
                "classification": "Sealed",
                "personal_information": True,
                "direct_identifier": True,
            },
            {
                "code": "PUBMAP",
                "name": "Published Network Map Reference",
                "classification": "Open",
                "personal_information": False,
                "direct_identifier": False,
            },
        ],
    }

    HEADER = "Term,Business Definition,Protection Class\n"
    METER = "Meter Serial,The serial number stamped on a water meter,METERID\n"
    MAIN = "Trunk Main Reference,Identifier of a trunk main on the published map,PUBMAP\n"

    @pytest.fixture
    def source(self, tmp_path):
        p = tmp_path / "glossary.csv"
        p.write_text(self.HEADER + self.METER + self.MAIN, encoding="utf-8")
        return p

    def _built(self, source, **kwargs):
        return ingest.build_index(
            source, provider=_ConstantProvider(), governance=self.VOCABULARY, **kwargs
        )

    def _codes(self, index):
        return {e.business_name: e.governance_code for e in index.entries.values()}

    def test_the_vocabulary_survives_a_no_change_sync(self, source):
        """THE DEFECT, at the boundary a caller uses. The file did not change at all."""
        index = self._built(source)
        assert self._codes(index) == {"Meter Serial": "METERID", "Trunk Main Reference": "PUBMAP"}

        report = ingest.sync(index, source)

        assert self._codes(index) == {"Meter Serial": "METERID", "Trunk Main Reference": "PUBMAP"}
        assert report.unchanged == 2
        assert report.governance_changed == []

    def test_the_column_mapping_survives_too(self, tmp_path):
        """
        The same hole, in the member that failed loudly by accident. Dropping `columns`
        here raises "could not find a business-name or definition column" -- which is only
        luck: on a glossary whose headers happen to be inferable it would instead change
        every derived id and report the whole file added and removed.
        """
        p = tmp_path / "odd.csv"
        p.write_text("colA,colB\nMeter Serial,The serial number\n", encoding="utf-8")
        index = ingest.build_index(
            p,
            provider=_ConstantProvider(),
            columns={"business_name": "colA", "definition": "colB"},
        )

        report = ingest.sync(index, p)

        assert report.unchanged == 1
        assert report.added == [] and report.removed == []

    def test_a_code_that_moved_is_reported(self, source):
        index = self._built(source)
        meter_id = next(i for i, e in index.entries.items() if e.business_name == "Meter Serial")
        source.write_text(
            self.HEADER + self.METER.replace("METERID", "PUBMAP") + self.MAIN, encoding="utf-8"
        )

        report = ingest.sync(index, source)

        assert report.governance_changed == [meter_id]
        assert self._codes(index)["Meter Serial"] == "PUBMAP"

    def test_a_code_the_steward_blanked_is_reported_and_NOT_refused(self, source):
        """
        Deliberately not a refusal. Blanking a protection-code cell is a legitimate source
        edit -- a code retired, a row reclassified as uncoded -- and refusing it would
        leave `sync` unable to process a file a steward is entitled to write. It is
        REPORTED, which is what the wiring bug that motivated a refusal actually needed.
        """
        index = self._built(source)
        source.write_text(
            self.HEADER + self.METER.replace("METERID", "") + self.MAIN, encoding="utf-8"
        )

        report = ingest.sync(index, source)

        assert len(report.governance_changed) == 1
        assert self._codes(index)["Meter Serial"] is None

    def test_the_report_line_says_so(self, source):
        """
        The line every caller prints. A code moving does not move the content hash -- see
        `content_hash` -- so without this the whole event reads as `=2 unchanged`.
        """
        index = self._built(source)
        source.write_text(
            self.HEADER + self.METER.replace("METERID", "PUBMAP") + self.MAIN, encoding="utf-8"
        )

        assert "1 governance changed" in str(ingest.sync(index, source))

    def test_a_new_row_is_not_also_reported_as_a_governance_change(self, source):
        """An added entry is reported once, as added. Saying it twice is noise."""
        index = self._built(source)
        source.write_text(
            self.HEADER + self.METER + self.MAIN + "Meter Location,Where a meter sits,PUBMAP\n",
            encoding="utf-8",
        )

        report = ingest.sync(index, source)

        assert len(report.added) == 1
        assert report.governance_changed == []

    def test_an_explicit_argument_overrides_what_was_remembered(self, source):
        """
        Remembering is a default, not a lock. Turning governance off is still available;
        it is now a decision -- two keywords, both of them the caller's -- rather than the
        accident of omitting one.
        """
        index = self._built(source)

        report = ingest.sync(index, source, governance=None, governance_strict=False)

        assert self._codes(index) == {"Meter Serial": None, "Trunk Main Reference": None}
        assert len(report.governance_changed) == 2

    def test_turning_the_vocabulary_off_alone_is_refused(self, source):
        """
        The unread-code-column guard reaches this path too, and has to: `sync` re-reads a
        glossary that still carries codes, and dropping the vocabulary is exactly how every
        entry in the index lost its class in the first place.
        """
        index = self._built(source)

        with pytest.raises(ValueError, match="protection-code column"):
            ingest.sync(index, source, governance=None)


class TestStableIdentity:
    """
    Ids must survive edits elsewhere in the file.

    Generating ids from ROW POSITION looks harmless and silently destroys incremental
    sync: deleting one row of ten renumbers every row after it, so the diff reports 9
    updates plus 1 removal and re-embeds the entire glossary. Measured before the fix:
    9 of 9 rows re-embedded for a single deletion.
    """

    def _write(self, path, n):
        rows = "\n".join(f"Term {i},Definition number {i}" for i in range(1, n + 1))
        path.write_text(f"Term,Business Definition\n{rows}\n", encoding="utf-8")

    def test_deleting_a_row_does_not_renumber_the_rest(self, tmp_path):
        p = tmp_path / "g.csv"
        self._write(p, 10)
        before = ingest.load_entries(p)
        previous = {e.id: ingest.content_hash(e) for e in before}

        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join([lines[0], *lines[2:]]) + "\n", encoding="utf-8")

        to_embed, removed, report = ingest.diff_entries(previous, ingest.load_entries(p))
        assert to_embed == [], f"a single deletion re-embedded {len(to_embed)} rows"
        assert len(removed) == 1
        assert report.unchanged == 9

    def test_inserting_a_row_does_not_renumber_the_rest(self, tmp_path):
        p = tmp_path / "g.csv"
        self._write(p, 5)
        previous = {e.id: ingest.content_hash(e) for e in ingest.load_entries(p)}

        lines = p.read_text(encoding="utf-8").splitlines()
        lines.insert(1, "Brand New Term,A new definition")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        to_embed, _, report = ingest.diff_entries(previous, ingest.load_entries(p))
        assert len(to_embed) == 1
        assert report.unchanged == 5

    def test_reordering_rows_changes_nothing(self, tmp_path):
        p = tmp_path / "g.csv"
        self._write(p, 5)
        previous = {e.id: ingest.content_hash(e) for e in ingest.load_entries(p)}

        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join([lines[0], *list(reversed(lines[1:]))]) + "\n", encoding="utf-8")

        to_embed, removed, report = ingest.diff_entries(previous, ingest.load_entries(p))
        assert to_embed == []
        assert removed == []
        assert report.unchanged == 5

    def test_an_explicit_id_column_is_used_when_present(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_text("ID,Term,Business Definition\nTERM-1,A,First\n", encoding="utf-8")
        assert ingest.load_entries(p)[0].id == "TERM-1"

    def test_duplicate_rows_still_get_distinct_ids(self, tmp_path):
        """Two genuinely identical terms must not collapse into one entry."""
        p = tmp_path / "g.csv"
        p.write_text(
            "Term,Business Definition\nSame,First meaning\nSame,Second meaning\n",
            encoding="utf-8",
        )
        ids = [e.id for e in ingest.load_entries(p)]
        assert len(set(ids)) == 2, ids

    def test_ids_are_deterministic_across_runs(self, tmp_path):
        p = tmp_path / "g.csv"
        self._write(p, 5)
        assert [e.id for e in ingest.load_entries(p)] == [e.id for e in ingest.load_entries(p)]


class TestBugsFoundInReview:
    """
    Regressions for defects an adversarial review found in this module. Each one was
    silent: no exception, no wrong-looking output, just an incorrect governance answer.
    """

    def test_classification_only_edit_is_applied(self, tmp_path):
        """
        sync() used to refresh the entry OBJECT only for rows it re-embedded. Since
        content_hash covers just the embedded text, a row whose classification changed but
        whose name and definition did not counted as "unchanged" and kept its stale entry
        -- so the same edit was applied or dropped depending on whether an UNRELATED row
        also changed. The index looked healthy and returned a stale PII level.
        """
        p = tmp_path / "g.csv"
        p.write_text(
            "Term,Business Definition,Classification\n"
            "Alpha,First definition,Public\n"
            "Beta,Second definition,Internal\n",
            encoding="utf-8",
        )
        index = ingest.build_index(p)
        alpha = next(i for i, e in index.entries.items() if e.business_name == "Alpha")
        assert index.entries[alpha].protection_level == ProtectionLevel.PUBLIC

        # Reclassify Alpha AND rewrite Beta, so to_embed is non-empty for a different row.
        p.write_text(
            "Term,Business Definition,Classification\n"
            "Alpha,First definition,Restricted\n"
            "Beta,A totally rewritten definition,Internal\n",
            encoding="utf-8",
        )
        ingest.sync(index, p)
        assert index.entries[alpha].protection_level == ProtectionLevel.RESTRICTED

    def test_classification_edit_applies_when_nothing_else_changed(self, tmp_path):
        """The same edit, with to_embed empty. Both paths must agree."""
        p = tmp_path / "g.csv"
        p.write_text(
            "Term,Business Definition,Classification\nAlpha,A definition,Public\n", encoding="utf-8"
        )
        index = ingest.build_index(p)
        alpha = next(iter(index.entries))
        p.write_text(
            "Term,Business Definition,Classification\nAlpha,A definition,Restricted\n",
            encoding="utf-8",
        )
        ingest.sync(index, p)
        assert index.entries[alpha].protection_level == ProtectionLevel.RESTRICTED

    @pytest.mark.parametrize(
        "label,expected",
        [
            # Substring matching inverted all of these: "Non-Public" contains "public",
            # "Unrestricted" contains "restricted". GLBA nonpublic personal information
            # was resolving to the WEAKEST level in the enum.
            ("Non-Public", ProtectionLevel.CONFIDENTIAL),
            ("Nonpublic", ProtectionLevel.CONFIDENTIAL),
            ("NPI - Nonpublic", ProtectionLevel.CONFIDENTIAL),
            ("Not for public release", ProtectionLevel.CONFIDENTIAL),
            ("Unrestricted", ProtectionLevel.PUBLIC),
            ("Un-Restricted", ProtectionLevel.PUBLIC),
            ("No PII", ProtectionLevel.INTERNAL),
            ("Not PII", ProtectionLevel.INTERNAL),
        ],
    )
    def test_negated_classifications_are_not_inverted(self, label, expected):
        assert ingest._coerce_protection(label) == expected

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Public", ProtectionLevel.PUBLIC),
            ("Open Data", ProtectionLevel.PUBLIC),
            ("Restricted - Legal Hold", ProtectionLevel.RESTRICTED),
            ("Highly Confidential", ProtectionLevel.RESTRICTED),
            ("Confidential - Internal Use Only", ProtectionLevel.CONFIDENTIAL),
            ("PII / Sensitive", ProtectionLevel.PII),
            ("Internal Use Only", ProtectionLevel.INTERNAL),
        ],
    )
    def test_negation_handling_did_not_break_normal_labels(self, label, expected):
        assert ingest._coerce_protection(label) == expected

    def test_newlines_inside_quoted_fields_survive(self, tmp_path):
        """splitlines() destroyed them, mangling every multi-line definition."""
        p = tmp_path / "g.csv"
        p.write_text(
            'Term,Business Definition\nEmail,"The customer email.\nUsed for billing."\n',
            encoding="utf-8",
        )
        rows, _ = ingest.read_source(p)
        assert len(rows) == 1
        assert "\n" in rows[0]["Business Definition"]

    @pytest.mark.parametrize("char", ["\x0b", "\x0c", "\x85", "\u2028", "\u2029"])
    def test_unicode_line_breaks_do_not_fabricate_rows(self, tmp_path, char):
        """
        str.splitlines() breaks on these; CSV does not. A vertical tab or U+2028 pasted
        from Word split one record into two -- truncating the real definition and creating
        a PHANTOM entry that got embedded, indexed, and could be returned as a top-1 match
        carrying the default INTERNAL classification.
        """
        p = tmp_path / "g.csv"
        p.write_text(f'Term,Business Definition\nEmail,"contact{char}billing"\n', encoding="utf-8")
        rows, _ = ingest.read_source(p)
        assert len(rows) == 1, f"{char!r} fabricated {len(rows)} rows"
        assert rows[0]["Term"] == "Email"

    def test_ragged_rows_do_not_produce_a_none_key(self, tmp_path):
        """A None key becomes the literal string 'null' after a JSON round-trip."""
        p = tmp_path / "g.csv"
        p.write_text("Term,Business Definition\nA,B,EXTRA\n", encoding="utf-8")
        rows, _ = ingest.read_source(p)
        assert None not in rows[0]
        assert "_extra_columns" in rows[0]


class TestExcelMergedCells:
    """
    Excel stores a merged block's value once, in the top-left cell; every other cell in
    the block reads back as None. For a governance glossary that is a silent
    MISCLASSIFICATION rather than a cosmetic gap, so it gets its own class.
    """

    @staticmethod
    def _write(path, rows, merges=(), sheet_title=None):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        if sheet_title:
            ws.title = sheet_title
        for row in rows:
            ws.append(row)
        for ref in merges:
            ws.merge_cells(ref)
        wb.save(path)
        return path

    def test_merged_classification_is_not_downgraded(self, tmp_path):
        """
        The reported defect. "Restricted" merged down C2:C3 gave RESTRICTED then
        INTERNAL, while the author saw one label covering both rows. Downgrading a
        restricted field is precisely what this library exists to prevent.
        """
        path = self._write(
            tmp_path / "g.xlsx",
            [
                ["Term", "Definition", "Classification"],
                ["Patient SSN", "Social security number", "Restricted"],
                ["Patient MRN", "Medical record number", None],
                ["Visit date", "Date of visit", "Public"],
            ],
            merges=["C2:C3"],
        )
        levels = {e.business_name: e.protection_level for e in ingest.load_entries(path)}
        assert levels["Patient SSN"] is ProtectionLevel.RESTRICTED
        assert levels["Patient MRN"] is ProtectionLevel.RESTRICTED
        assert levels["Visit date"] is ProtectionLevel.PUBLIC

    def test_unmerged_blank_is_left_alone(self, tmp_path):
        """
        The guard on the fix. A blank cell that is NOT merged must stay blank -- filling
        every gap downward would invent classifications nobody wrote, which is the same
        defect in the opposite direction.
        """
        path = self._write(
            tmp_path / "g.xlsx",
            [
                ["Term", "Definition", "Classification"],
                ["A", "d1", "Restricted"],
                ["B", "d2", None],
            ],
        )
        levels = {e.business_name: e.protection_level for e in ingest.load_entries(path)}
        assert levels["A"] is ProtectionLevel.RESTRICTED
        assert levels["B"] is ProtectionLevel.INTERNAL  # the default, not inherited

    def test_span_covers_gap_rows_and_multiple_rows(self, tmp_path):
        path = self._write(
            tmp_path / "g.xlsx",
            [
                ["Term", "Class"],
                ["alpha", "Restricted"],
                [None, None],  # a blank row inside the span
                ["beta", None],
                ["gamma", None],
            ],
            merges=["B2:B5"],
        )
        rows, _ = ingest._read_tabular(path)
        assert [r["Class"] for r in rows] == ["Restricted"] * 4

    def test_merged_header_does_not_duplicate_a_column(self, tmp_path):
        """
        A merge in the HEADER is a layout banner spanning columns, not a value belonging
        to each. Copying it produced two "Classification" columns, and since rows are
        built as {header[i]: value} the second silently overwrote the first -- turning a
        cosmetic merge into data loss.
        """
        path = self._write(
            tmp_path / "g.xlsx",
            [["Term", "Classification", None], ["A", "Public", "x"]],
            merges=["B1:C1"],
        )
        rows, header = ingest._read_tabular(path)
        assert header.count("Classification") == 1
        assert rows[0]["Classification"] == "Public"

    def test_named_sheet_resolves_its_own_merges(self, tmp_path):
        """Merges are looked up by sheet name through the workbook rels, not by index."""
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "g.xlsx"
        wb = openpyxl.Workbook()
        first = wb.active
        first.title = "Cover"
        first.append(["ignore me"])
        second = wb.create_sheet("Glossary")
        for row in [["Term", "Class"], ["A", "Restricted"], ["B", None]]:
            second.append(row)
        second.merge_cells("B2:B3")
        wb.save(path)

        rows, _ = ingest._read_tabular(path, sheet="Glossary")
        assert [r["Class"] for r in rows] == ["Restricted", "Restricted"]

    @pytest.mark.parametrize("payload", [b"not a zip at all", b"PK\x03\x04truncated"])
    def test_unreadable_package_returns_no_merges(self, tmp_path, payload):
        """Merge handling is an enhancement; it must never stop a file from loading."""
        path = tmp_path / "bad.xlsx"
        path.write_bytes(payload)
        assert ingest._excel_merged_ranges(path, "Sheet") == []

    def test_unknown_sheet_name_returns_no_merges(self, tmp_path):
        path = self._write(tmp_path / "g.xlsx", [["Term"], ["A"]])
        assert ingest._excel_merged_ranges(path, "NoSuchSheet") == []


class TestDuplicateHeaders:
    """
    Rows are built as `{header[i]: value}`, so two columns sharing a name collapse and the
    last one wins. Suffixing surfaces the problem as "column not found" instead.
    """

    def test_duplicate_csv_columns_both_survive(self, tmp_path):
        path = tmp_path / "dup.csv"
        path.write_bytes(b"Term,Notes,Notes\nA,first,second\n")
        rows, header = ingest._read_tabular(path)
        assert header == ["Term", "Notes", "Notes_1"]
        assert rows[0]["Notes"] == "first"
        assert rows[0]["Notes_1"] == "second"

    def test_three_way_duplicate(self, tmp_path):
        path = tmp_path / "dup3.csv"
        path.write_bytes(b"A,A,A\n1,2,3\n")
        rows, header = ingest._read_tabular(path)
        assert header == ["A", "A_1", "A_2"]
        assert [rows[0][c] for c in header] == ["1", "2", "3"]

    def test_unique_headers_are_untouched(self, tmp_path):
        path = tmp_path / "ok.csv"
        path.write_bytes(b"Term,Definition\nA,d\n")
        _, header = ingest._read_tabular(path)
        assert header == ["Term", "Definition"]


class TestCsvReaderRegressions:
    """
    Re-asserted after the header pre-read was added, because splitting the header off with
    a separate csv.reader is exactly the kind of change that quietly breaks quoting.
    """

    def test_quoted_newline_survives_with_bom_and_crlf(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_bytes(
            b'\xef\xbb\xbfTerm,Definition\r\n"Email","The customer email.\r\nUsed for billing."\r\n'
        )
        rows, header = ingest._read_tabular(path)
        assert header == ["Term", "Definition"]  # BOM stripped
        assert len(rows) == 1
        assert rows[0]["Definition"] == "The customer email.\r\nUsed for billing."

    @pytest.mark.parametrize(
        "char", [b"\x0b", b"\x0c", b"\x1c", b"\x1d", b"\x1e", b"\xc2\x85", b"\xe2\x80\xa8"]
    )
    def test_unicode_line_breaks_do_not_split_a_row(self, tmp_path, char):
        """
        str.splitlines() breaks on all of these; none is a CSV record separator. Each one
        used to truncate a real definition AND fabricate a phantom entry that was
        embedded, indexed, and returnable as a top-1 match.
        """
        path = tmp_path / "u.csv"
        path.write_bytes(b'Term,Definition\nA,"before' + char + b'after"\n')
        rows, _ = ingest._read_tabular(path)
        assert len(rows) == 1

    def test_ragged_rows_avoid_a_none_key(self, tmp_path):
        path = tmp_path / "r.csv"
        path.write_bytes(b"Term,Definition\nA,d,EXTRA\nB\n")
        rows, _ = ingest._read_tabular(path)
        assert not any(None in r for r in rows)
        assert rows[0]["_extra_columns"] == ["EXTRA"]
        assert rows[1]["Definition"] == ""

    def test_tsv_delimiter(self, tmp_path):
        path = tmp_path / "t.tsv"
        path.write_bytes(b"Term\tDefinition\nA\td1\n")
        rows, header = ingest._read_tabular(path)
        assert header == ["Term", "Definition"]
        assert rows[0]["Definition"] == "d1"
