"""
tests.unit.infrastructure.test_dictionary_loader_defaults | Layer: TEST
The out-of-the-box loading path, on a bare install and a realistic glossary.

Three defects met here, all silent:

  * CSV and Excel loading demanded pandas, so the documented quickstart raised
    ImportError on `pip install nexus-matcher`. Neither format needs it.
  * `ColumnMapping.default()` names exact literal columns ("Business Name",
    "Definition"). Real glossaries say "Term" and "Business Definition", so every row
    failed on "Missing business name".
  * The result of that was a SUCCESSFUL load reporting zero entries. The matcher then
    had nothing to match against and said nothing about why.
"""

from __future__ import annotations

import pytest

from nexus_matcher.domain.ports.dictionary_loader import (
    ColumnMapping,
    detect_column_mapping,
)
from nexus_matcher.infrastructure.adapters.dictionary_loaders.excel import (
    CsvDictionaryLoader,
    ExcelDictionaryLoader,
)
from nexus_matcher.shared.types.base import ProtectionLevel

REALISTIC = (
    b"Term,Business Definition,Subject Area,Classification\n"
    b"Customer Email Address,The email used to contact a customer,Customer,PII\n"
    b"Order Total Amount,Total value of an order including tax,Order,Internal\n"
)


class TestColumnDetection:
    def test_realistic_headers_are_understood(self):
        mapping = detect_column_mapping(
            ["Term", "Business Definition", "Subject Area", "Classification"]
        )
        assert mapping.business_name_column == "Term"
        assert mapping.definition_column == "Business Definition"
        assert mapping.domain_column == "Subject Area"
        assert mapping.protection_level_column == "Classification"

    def test_canonical_headers_still_work(self):
        mapping = detect_column_mapping(["Business Name", "Definition", "Data Type"])
        assert mapping.business_name_column == "Business Name"
        assert mapping.definition_column == "Definition"
        assert mapping.data_type_column == "Data Type"

    def test_absent_columns_keep_their_defaults(self):
        """Only the columns genuinely missing fall back -- not all of them."""
        mapping = detect_column_mapping(["Term"])
        assert mapping.business_name_column == "Term"
        assert mapping.definition_column == ColumnMapping().definition_column

    def test_classmethod_matches_the_function(self):
        cols = ["Term", "Business Definition"]
        assert ColumnMapping.detect(cols) == detect_column_mapping(cols)


class TestCsvLoaderWithoutPandas:
    def test_realistic_glossary_loads(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_bytes(REALISTIC)
        entries, stats = CsvDictionaryLoader().load(p).unwrap()
        assert stats.valid_entries == 2
        assert {e.business_name for e in entries} == {
            "Customer Email Address",
            "Order Total Amount",
        }

    def test_classification_column_is_applied(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_bytes(REALISTIC)
        entries, _ = CsvDictionaryLoader().load(p).unwrap()
        levels = {e.business_name: e.protection_level for e in entries}
        assert levels["Customer Email Address"] is ProtectionLevel.PII
        assert levels["Order Total Amount"] is ProtectionLevel.INTERNAL

    def test_explicit_mapping_still_wins(self, tmp_path):
        """Auto-detection must not override a caller who said what they wanted."""
        p = tmp_path / "g.csv"
        p.write_bytes(b"Term,Business Definition\nA,d1\n")
        result = CsvDictionaryLoader().load(
            p, column_mapping=ColumnMapping(business_name_column="Business Definition")
        )
        entries, _ = result.unwrap()
        assert entries[0].business_name == "d1"

    def test_header_row_skips_a_banner(self, tmp_path):
        p = tmp_path / "g.csv"
        p.write_bytes(b"Glossary export,,\n" + REALISTIC)
        entries, stats = CsvDictionaryLoader().load(p, header_row=1).unwrap()
        assert stats.valid_entries == 2
        assert entries[0].business_name == "Customer Email Address"

    def test_zero_entries_is_a_failure_not_a_quiet_success(self, tmp_path):
        """
        The defect that hid the other two. A file whose columns are unrecognisable used
        to load "successfully" with nothing in it.
        """
        p = tmp_path / "g.csv"
        p.write_bytes(b"colour,shape\nred,round\n")
        result = CsvDictionaryLoader().load(p)
        assert not result.is_success
        assert "no entries" in result.error

    def test_an_empty_file_is_not_reported_as_an_error(self, tmp_path):
        """Zero rows in means zero entries out; that is not a mapping failure."""
        p = tmp_path / "g.csv"
        p.write_bytes(b"Term,Business Definition\n")
        entries, stats = CsvDictionaryLoader().load(p).unwrap()
        assert entries == []
        assert stats.total_rows == 0


class TestExcelLoaderWithoutPandas:
    @staticmethod
    def _write(path, rows, merges=()):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        for row in rows:
            ws.append(row)
        for ref in merges:
            ws.merge_cells(ref)
        wb.save(path)
        return path

    def test_realistic_glossary_loads(self, tmp_path):
        p = self._write(
            tmp_path / "g.xlsx",
            [
                ["Term", "Business Definition", "Classification"],
                ["Customer Email Address", "The email used to contact a customer", "PII"],
                ["Order Total Amount", "Total value of an order including tax", "Internal"],
            ],
        )
        entries, stats = ExcelDictionaryLoader().load(p).unwrap()
        assert stats.valid_entries == 2
        assert entries[0].protection_level is ProtectionLevel.PII

    def test_merged_classification_reaches_the_entry(self, tmp_path):
        """
        The end-to-end version of the merged-cell fix: through the loader, not just the
        reader. pandas would have left this as NaN.
        """
        p = self._write(
            tmp_path / "g.xlsx",
            [
                ["Term", "Business Definition", "Classification"],
                ["Patient SSN", "Social security number", "Restricted"],
                ["Patient MRN", "Medical record number", None],
            ],
            merges=["C2:C3"],
        )
        entries, _ = ExcelDictionaryLoader().load(p).unwrap()
        levels = {e.business_name: e.protection_level for e in entries}
        assert levels["Patient SSN"] is ProtectionLevel.RESTRICTED
        assert levels["Patient MRN"] is ProtectionLevel.RESTRICTED

    def test_named_sheet(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        p = tmp_path / "g.xlsx"
        wb = openpyxl.Workbook()
        wb.active.title = "Cover"
        wb.active.append(["not the glossary"])
        sheet = wb.create_sheet("Glossary")
        sheet.append(["Term", "Business Definition"])
        sheet.append(["A", "d1"])
        wb.save(p)

        entries, _ = ExcelDictionaryLoader().load(p, sheet_name="Glossary").unwrap()
        assert [e.business_name for e in entries] == ["A"]

    def test_header_row_skips_a_banner(self, tmp_path):
        p = self._write(
            tmp_path / "g.xlsx",
            [
                ["Data Glossary Export", None],
                ["Term", "Business Definition"],
                ["A", "d1"],
            ],
        )
        entries, _ = ExcelDictionaryLoader().load(p, header_row=1).unwrap()
        assert [e.business_name for e in entries] == ["A"]
