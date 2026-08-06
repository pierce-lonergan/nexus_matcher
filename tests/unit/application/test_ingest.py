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
