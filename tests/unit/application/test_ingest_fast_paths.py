"""
tests.unit.application.test_ingest_fast_paths | Layer: TEST
Guards for the fast paths added to the ingestion hot path.

Each optimisation here replaced a slow-but-obviously-correct implementation with a
faster one that takes a shortcut. Every shortcut has a specific way of being silently
wrong, and a silently wrong ingest is the worst outcome this library has: it produces an
index that looks healthy and hands back the wrong governance classification.

  * The protection patterns are now compiled once at import. The ORDER of that table is
    what makes "Non-Public" resolve to CONFIDENTIAL instead of PUBLIC. Build it wrong and
    the inversion the negation table exists to prevent comes straight back.

  * Merged-range discovery no longer parses the sheet's element tree. It decides from a
    byte scan whether a `mergeCells` element can exist at all. A false NEGATIVE there
    means merged governance labels stop propagating -- silently downgrading a RESTRICTED
    field to the default INTERNAL, which is precisely the failure `_apply_merged_values`
    was written to stop.

  * sync() no longer rehashes unchanged rows, relying on the invariant that a row judged
    unchanged already has the current hash stored. If that invariant ever breaks, the
    manifest drifts out of step with the entries and a later sync re-embeds the wrong set.
"""

from __future__ import annotations

import zipfile

import pytest

from nexus_matcher.application import ingest
from nexus_matcher.shared.types.base import ProtectionLevel

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {
    "m": MAIN_NS,
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


class TestCompiledProtectionPatterns:
    """The precompiled table must be the same table, in the same order."""

    def test_compiled_table_matches_the_source_tables_exactly(self):
        """
        Pins that precompiling did not drop, reorder or duplicate a token. Order is
        load-bearing: negations must precede positives, and positives are strictest
        first, so a reordered table under-protects a field rather than failing loudly.
        """
        expected = [level for _, level in (*ingest._NEGATED_PROTECTION, *ingest._PROTECTION_WORDS)]
        assert [level for _, level in ingest._PROTECTION_PATTERNS] == expected

        expected_tokens = [
            token for token, _ in (*ingest._NEGATED_PROTECTION, *ingest._PROTECTION_WORDS)
        ]
        for (pattern, _), token in zip(ingest._PROTECTION_PATTERNS, expected_tokens, strict=True):
            assert pattern.search(token), f"compiled pattern no longer matches {token!r}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Non-Public", ProtectionLevel.CONFIDENTIAL),
            ("nonpublic", ProtectionLevel.CONFIDENTIAL),
            ("NPI - Nonpublic", ProtectionLevel.CONFIDENTIAL),
            ("Unrestricted", ProtectionLevel.PUBLIC),
            ("un-restricted", ProtectionLevel.PUBLIC),
            ("un_restricted", ProtectionLevel.PUBLIC),
            ("no pii", ProtectionLevel.INTERNAL),
            ("Highly Confidential", ProtectionLevel.RESTRICTED),
            ("Confidential - Internal Use Only", ProtectionLevel.CONFIDENTIAL),
            ("Public", ProtectionLevel.PUBLIC),
            ("republic", ProtectionLevel.INTERNAL),
            ("restricteddata", ProtectionLevel.INTERNAL),
            ("", ProtectionLevel.INTERNAL),
        ],
    )
    def test_inversions_stay_fixed(self, text, expected):
        """
        The negation cases are the ones a substring scan gets backwards. "Non-Public"
        mapping to PUBLIC would publish GLBA nonpublic personal information; the word
        boundaries stop "republic" and "restricteddata" matching in the other direction.
        """
        assert ingest._coerce_protection(text) is expected

    def test_patterns_are_compiled_once_not_per_call(self):
        """
        Pins the actual optimisation. Rebuilding the patterns per call was ~20 pattern
        constructions per row and measured 5.9x slower over a 30k-row glossary; the table
        must stay a module-level constant of compiled patterns, not strings.
        """
        import re

        assert ingest._PROTECTION_PATTERNS, "protection table is empty"
        assert all(isinstance(pattern, re.Pattern) for pattern, _ in ingest._PROTECTION_PATTERNS)


def _sheet_xml(body: str, prefix: str = "") -> bytes:
    p = f"{prefix}:" if prefix else ""
    xmlns = f'xmlns:{prefix}="{MAIN_NS}"' if prefix else f'xmlns="{MAIN_NS}"'
    return f"<{p}worksheet {xmlns}>{body}</{p}worksheet>".encode()


class TestMergeRefFastPath:
    """
    The byte scan may only ever skip work it can PROVE is unnecessary. A false positive
    costs a little time; a false negative silently drops merged governance labels.
    """

    def test_absent_element_means_no_merges(self):
        """No `mergeCells` anywhere in the bytes: nothing to find, under any prefix."""
        raw = _sheet_xml("<sheetData/>")
        assert ingest._merge_refs_from_sheet_xml(raw, NS) == []

    def test_reads_the_ordinary_unprefixed_form(self):
        """The form Excel and openpyxl actually write -- the path that must be fast."""
        raw = _sheet_xml(
            '<sheetData/><mergeCells count="2">'
            '<mergeCell ref="C2:C3"/><mergeCell ref="D5:D9"/></mergeCells>'
        )
        assert ingest._merge_refs_from_sheet_xml(raw, NS) == ["C2:C3", "D5:D9"]

    def test_self_closing_empty_element(self):
        """`<mergeCells count="0"/>`: declared but empty, and must not raise."""
        raw = _sheet_xml('<sheetData/><mergeCells count="0"/>')
        assert ingest._merge_refs_from_sheet_xml(raw, NS) == []

    def test_cell_text_mentioning_mergecells_does_not_invent_merges(self):
        """
        A definition containing the word "mergeCells" trips the cheap substring probe.
        That is allowed to cost time, but it must not fabricate a merged range -- which
        would copy a governance label onto rows the author never labelled.
        """
        raw = _sheet_xml(
            "<sheetData><row><c t='inlineStr'><is><t>see mergeCells docs</t></is></c>"
            "</row></sheetData>"
        )
        assert ingest._merge_refs_from_sheet_xml(raw, NS) == []

    def test_escaped_angle_bracket_in_cell_text_is_not_a_start_tag(self):
        """
        `&lt;mergeCells count="9"&gt;` inside a cell is text, not markup. The scan keys on
        a RAW `<`, which XML forbids in content, so the real element is still the one found.
        """
        raw = _sheet_xml(
            "<sheetData><row><c t='inlineStr'><is><t>&lt;mergeCells count=\"9\"&gt;</t>"
            '</is></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:B1"/>'
            "</mergeCells>"
        )
        assert ingest._merge_refs_from_sheet_xml(raw, NS) == ["A1:B1"]

    def test_namespace_prefixed_form_falls_back_to_the_full_parse(self):
        """
        A writer that emits `<x:mergeCells>` must still be understood. The fast slice only
        recognises the unprefixed start tag, so this file has to take the slow path rather
        than silently report no merges.
        """
        raw = _sheet_xml(
            '<x:sheetData/><x:mergeCells count="1"><x:mergeCell ref="D5:D9"/></x:mergeCells>',
            prefix="x",
        )
        assert ingest._merge_refs_from_sheet_xml(raw, NS) == ["D5:D9"]

    def test_an_unusable_merge_ref_still_degrades_to_no_merges(self, tmp_path):
        """
        Merge handling is an ENHANCEMENT: `_excel_merged_ranges` promises that failing to
        work out the merges never stops a file loading. The fast path must keep that
        promise, so a `ref` the range parser rejects has to come back as [] rather than
        propagate out of merge discovery.
        """
        openpyxl = pytest.importorskip("openpyxl")
        good = tmp_path / "good.xlsx"
        wb = openpyxl.Workbook()
        wb.active.append(["Term", "Business Definition"])
        wb.save(good)

        with zipfile.ZipFile(good) as z:
            parts = {name: z.read(name) for name in z.namelist()}
        parts["xl/worksheets/sheet1.xml"] = parts["xl/worksheets/sheet1.xml"].replace(
            b"</sheetData>",
            b'</sheetData><mergeCells count="1"><mergeCell ref="NOT_A_RANGE"/></mergeCells>',
        )
        broken = tmp_path / "broken.xlsx"
        with zipfile.ZipFile(broken, "w") as z:
            for name, blob in parts.items():
                z.writestr(name, blob)

        assert ingest._excel_merged_ranges(broken, "Sheet") == []

    def test_a_missing_sheet_part_degrades_to_no_merges(self, tmp_path):
        """A file that is not an XLSX at all must not raise out of merge discovery."""
        not_a_workbook = tmp_path / "nope.xlsx"
        not_a_workbook.write_bytes(b"this is not a zip archive")
        assert ingest._excel_merged_ranges(not_a_workbook, "Sheet") == []


class TestMergedGovernanceStillPropagates:
    """End to end, through a real XLSX, because that is where the consequence lands."""

    @pytest.fixture
    def merged_workbook(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "g.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Term", "Business Definition", "Classification"])
        ws.append(["Patient SSN", "Social security number", "Restricted"])
        ws.append(["Patient MRN", "Medical record number", None])
        ws.append(["Patient Name", "Full legal name", "PII"])
        ws.merge_cells(start_row=2, start_column=3, end_row=3, end_column=3)
        wb.save(path)
        return path

    def test_a_merged_restricted_label_still_covers_the_rows_it_spans(self, merged_workbook):
        """
        The failure this whole feature exists to prevent: Excel stores a merged block's
        value only in its top-left cell, so "Patient MRN" reads back blank and takes the
        default INTERNAL -- downgrading a RESTRICTED field while the author sees one label
        covering both rows. The byte-scan fast path must not reintroduce it.
        """
        entries = {e.business_name: e for e in ingest.load_entries(merged_workbook)}
        assert entries["Patient SSN"].protection_level is ProtectionLevel.RESTRICTED
        assert entries["Patient MRN"].protection_level is ProtectionLevel.RESTRICTED
        assert entries["Patient Name"].protection_level is ProtectionLevel.PII

    def test_merged_ranges_are_found_through_the_public_reader(self, merged_workbook):
        with zipfile.ZipFile(merged_workbook) as z:
            assert b"<mergeCells" in z.read("xl/worksheets/sheet1.xml")
        assert ingest._excel_merged_ranges(merged_workbook, "Sheet") == [(2, 3, 3, 3)]

    def test_a_workbook_with_no_merges_reads_normally(self, tmp_path):
        """The fast exit must not break the overwhelmingly common no-merge file."""
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "plain.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Term", "Business Definition", "Classification"])
        ws.append(["Alpha", "First", "Public"])
        wb.save(path)

        assert ingest._excel_merged_ranges(path, "Sheet") == []
        entries = ingest.load_entries(path)
        assert [e.business_name for e in entries] == ["Alpha"]
        assert entries[0].protection_level is ProtectionLevel.PUBLIC


class TestSyncHashInvariant:
    """
    sync() now rewrites hashes only for rows it re-embedded, on the grounds that a row
    judged unchanged already carries the current hash. If that ever stops holding, the
    manifest silently disagrees with the entries and the NEXT sync re-embeds the wrong set.
    """

    @staticmethod
    def _write(path, rows):
        path.write_text(
            "Term,Business Definition,Classification\n" + "".join(rows), encoding="utf-8"
        )

    def _recomputed(self, index):
        return {e.id: ingest.content_hash(e) for e in index.entries.values()}

    def test_hashes_are_exact_after_a_no_change_sync(self, tmp_path):
        p = tmp_path / "g.csv"
        self._write(p, ["Alpha,First definition,Public\n", "Beta,Second definition,PII\n"])
        index = ingest.build_index(p)
        ingest.sync(index, p)
        assert index.hashes == self._recomputed(index)

    def test_hashes_are_exact_after_add_update_and_remove_together(self, tmp_path):
        """
        The mixed case. Skipping the rehash for unchanged rows must not skip it for a row
        that genuinely moved, or that row is never re-embedded again.
        """
        p = tmp_path / "g.csv"
        self._write(
            p,
            [
                "Alpha,First definition,Public\n",
                "Beta,Second definition,PII\n",
                "Gamma,Third definition,Internal\n",
            ],
        )
        index = ingest.build_index(p)
        self._write(
            p,
            [
                "Alpha,First definition,Public\n",
                "Beta,A completely rewritten definition,PII\n",
                "Delta,Fourth definition,Restricted\n",
            ],
        )
        report = ingest.sync(index, p)

        assert index.hashes == self._recomputed(index)
        assert len(report.added) == 1 and len(report.updated) == 1
        assert len(report.removed) == 1 and report.unchanged == 1
        assert set(index.hashes) == set(index.entries) == set(index.order)

    def test_a_governance_only_edit_leaves_the_hash_alone_but_updates_the_entry(self, tmp_path):
        """
        The hash covers embedded text only, so a reclassification must NOT move it -- but
        the entry object must still be refreshed. Rewriting hashes only for re-embedded
        rows must not quietly resurrect the stale-classification bug.
        """
        p = tmp_path / "g.csv"
        self._write(p, ["Alpha,First definition,Public\n"])
        index = ingest.build_index(p)
        alpha = next(iter(index.entries))
        before = index.hashes[alpha]

        self._write(p, ["Alpha,First definition,Restricted\n"])
        report = ingest.sync(index, p)

        assert index.hashes[alpha] == before, "embedded text did not change"
        assert report.embedded == 0, "a reclassification must not re-embed"
        assert index.entries[alpha].protection_level is ProtectionLevel.RESTRICTED
        assert index.hashes == self._recomputed(index)

    def test_repeated_syncs_stay_exact(self, tmp_path):
        """Drift accumulates. Five syncs must leave the manifest as exact as one."""
        p = tmp_path / "g.csv"
        self._write(p, ["Alpha,First definition,Public\n", "Beta,Second,PII\n"])
        index = ingest.build_index(p)
        for i in range(5):
            self._write(p, [f"Alpha,Definition revision {i},Public\n", "Beta,Second,PII\n"])
            ingest.sync(index, p)
            assert index.hashes == self._recomputed(index), f"drifted on sync {i}"


class TestSourceMetadataHoist:
    """`set(mapping.values())` moved out of the row loop; the contents must be unchanged."""

    def test_mapped_columns_are_excluded_and_extras_preserved(self, tmp_path):
        """
        source_metadata must keep exactly the UNMAPPED columns. Hoisting the exclusion set
        out of the loop would silently change which columns are dropped if it were built
        from the wrong thing -- and the unmapped columns are the ones a caller pays for.
        """
        p = tmp_path / "g.csv"
        p.write_text(
            "Term,Business Definition,Classification,Steward,Source System\n"
            "Alpha,First definition,Public,ann@example.org,EPIC\n",
            encoding="utf-8",
        )
        (entry,) = ingest.load_entries(p)
        assert entry.source_metadata["Steward"] == "ann@example.org"
        assert entry.source_metadata["Source System"] == "EPIC"
        assert "Term" not in entry.source_metadata
        assert "Business Definition" not in entry.source_metadata
        # The raw governance string is kept deliberately, even though the column is mapped.
        assert entry.source_metadata["governance_raw"] == "Public"

    def test_every_row_gets_the_same_exclusion_set(self, tmp_path):
        """Hoisting must not let the first row's mapping leak into later rows."""
        p = tmp_path / "g.csv"
        p.write_text(
            "Term,Business Definition,Steward\n"
            "Alpha,First definition,ann@example.org\n"
            "Beta,Second definition,bob@example.org\n"
            "Gamma,Third definition,cat@example.org\n",
            encoding="utf-8",
        )
        entries = ingest.load_entries(p)
        assert [sorted(e.source_metadata) for e in entries] == [["Steward"]] * 3
