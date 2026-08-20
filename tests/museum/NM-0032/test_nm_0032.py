"""
NM-0032 -- a misspelled option loaded a different glossary, and said nothing.

`load_entries` named a handful of options for itself and forwarded everything else to the
reader as `**kwargs`. The reader pops the options it recognises -- `sheet`, `delimiter`,
`encoding`, `header_row` -- and never looks at what is left. Nothing in that path has any
reason to complain, because nothing in that path knows an option was intended.

So `sheet_name="Approved"` -- the pandas spelling, and the obvious thing to type -- read
the workbook's FIRST sheet instead. Measured on a two-sheet fixture: asking for the
`Approved` sheet returned the single row of the `Retired` one, with no warning, no
degraded mode and no clue in the result. A deployment that did this would index a glossary
of retired terms, match every column against it, inherit classifications from it, and
report a completely healthy load.

That is the same shape as every other defect in this museum: the failure is not that
something broke, it is that a wrong answer is indistinguishable from a right one.

Why it escaped: every ingest test passes the option it means. A test that spells an option
correctly cannot observe what happens to one that does not, and `**kwargs` has no
signature to check against. The gate had to be an explicit statement of which options
exist -- there was nothing to derive it from.

The fix refuses the load and names both the option and the ones that exist. It is
deliberately a refusal rather than a warning: a warning on a startup path is read by
nobody, and the entire cost of this defect is that the caller believes the load did what
they asked.
"""

from __future__ import annotations

import pytest

from nexus_matcher.application import ingest


@pytest.fixture
def two_sheet_workbook(tmp_path):
    """
    A workbook whose first sheet is the one you must not load.

    Sheet order is the whole fixture: the reader falls back to `wb.active`, so a dropped
    `sheet` option is only observable when the sheet you wanted is not the first one.
    """
    openpyxl = pytest.importorskip("openpyxl")

    path = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()
    retired = workbook.active
    retired.title = "Retired"
    retired.append(["Term", "Business Definition"])
    retired.append(["Legacy Batch Marker", "A marker used by the retired batch loader"])
    approved = workbook.create_sheet("Approved")
    approved.append(["Term", "Business Definition"])
    approved.append(["Transaction Amount", "The gross amount of a transaction"])
    workbook.save(path)
    return path


def test_a_misspelled_reader_option_is_refused_not_dropped(two_sheet_workbook):
    """
    The symptom, stated as the thing that must not happen: a load that was asked for the
    Approved sheet must never come back holding the Retired one.
    """
    with pytest.raises(ValueError) as excinfo:
        ingest.load_entries(two_sheet_workbook, sheet_name="Approved")

    message = str(excinfo.value)
    assert "sheet_name" in message, "the refusal must name the option that was not understood"
    assert "sheet" in message, "and the ones that exist, or it cannot be acted on"


def test_the_option_spelled_correctly_still_works(two_sheet_workbook):
    """
    The control. A refusal that also refuses the correct spelling would be a different
    defect, and a worse one -- this one at least fails loudly.
    """
    entries = ingest.load_entries(two_sheet_workbook, sheet="Approved")
    assert [e.business_name for e in entries] == ["Transaction Amount"]


def test_the_reader_underneath_is_still_the_permissive_one(two_sheet_workbook):
    """
    Pins WHERE the check lives, and pins the harm it prevents.

    `read_source` is a lower-level entry point that has always taken whatever it was
    given; tightening it would break callers that legitimately pass reader-specific
    options this module does not enumerate. So the guarantee belongs to `load_entries`,
    which does enumerate them -- and this test is the standing evidence that the layer
    below still silently answers with the wrong sheet, which is exactly why the layer
    above has to refuse.
    """
    rows, _header = ingest.read_source(two_sheet_workbook, sheet_name="Approved")
    assert rows == [
        {
            "Term": "Legacy Batch Marker",
            "Business Definition": "A marker used by the retired batch loader",
        }
    ], "the reader no longer discards unknown options -- move this guarantee's test with it"
