"""
Measure the DX baseline against the PUBLISHED nexus-matcher 2.0.1.

Deliberately run from a temp directory against the installed wheel, never the repo tree,
so what is measured is what a stranger gets. Everything here is contention-immune
(counts, message text, error behaviour) except the cold-start block, which checks CPU
first and refuses to record above the threshold.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import time
import traceback

WORK = pathlib.Path(tempfile.mkdtemp(prefix="dx"))
RESULTS: dict = {}


def glossary(path: pathlib.Path, n: int = 3) -> pathlib.Path:
    rows = [b"Term,Business Definition,Classification\n"]
    names = [
        (b"Customer Email Address", b"The email address used to contact a customer", b"PII"),
        (b"Order Total Amount", b"Total monetary value of an order including tax", b"Internal"),
        (b"Shipping Street Line", b"Street portion of the delivery address", b"Restricted"),
    ]
    for i in range(n):
        t, d, c = names[i % 3]
        suffix = b"" if i < 3 else b" %d" % i
        rows.append(t + suffix + b"," + d + b"," + c + b"\n")
    path.write_bytes(b"".join(rows))
    return path


def schema(path: pathlib.Path) -> pathlib.Path:
    path.write_bytes(
        json.dumps(
            {
                "type": "record",
                "name": "Order",
                "fields": [
                    {"name": "email", "type": "string", "doc": "Contact email for the buyer"},
                    {"name": "total", "type": "double", "doc": "Order grand total with tax"},
                ],
            }
        ).encode()
    )
    return path


# ---------------------------------------------------------------------------
# Lines of user code + concepts, for the simplest useful task
# ---------------------------------------------------------------------------
def measure_minimal_program() -> None:
    """
    The shortest program that answers the library's actual question:
    "which of my fields map to which glossary entries, and which need review?"
    """
    g = glossary(WORK / "g.csv")
    s = schema(WORK / "s.avsc")

    program = textwrap.dedent(
        f"""
        from nexus_matcher import NexusMatcher

        matcher = NexusMatcher.from_config()
        matcher.load_dictionary(r"{g}")
        results = matcher.match_schema(r"{s}")
        for field, matches in results.items():
            top = matches[0]
            print(field, top.dictionary_entry.business_name,
                  top.dictionary_entry.protection_level.name, top.decision.name)
        """
    ).strip()

    code_lines = [ln for ln in program.splitlines() if ln.strip()]
    concepts = {
        "NexusMatcher": "the class",
        "from_config": "the constructor you must use instead of NexusMatcher()",
        "load_dictionary": "glossary ingestion",
        "match_schema": "the call",
        "results.items()": "dict-of-tuples result shape",
        "MatchResult.dictionary_entry": "nested entry object",
        "ProtectionLevel": "enum, reached via .protection_level.name",
        "MatchDecision": "enum, reached via .decision.name",
    }
    RESULTS["minimal_program"] = {
        "lines_of_user_code": len(code_lines),
        "concepts_required": len(concepts),
        "concept_list": list(concepts),
        "config_values_required": 0,
        "program": program,
    }


# ---------------------------------------------------------------------------
# Error actionability: what does the user actually see when they get it wrong?
# ---------------------------------------------------------------------------
def _capture(fn) -> tuple[str, str]:
    """Return (exception_type, message) or ('', output) if it did not raise."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn()
        return "", buf.getvalue().strip()
    except BaseException as exc:
        return type(exc).__name__, str(exc).strip() or traceback.format_exc().strip()[-400:]
    finally:
        sys.stdout = old


def _bad_inputs() -> dict[str, pathlib.Path]:
    """
    Write every malformed fixture once.

    Split out purely so the case list below reads as a list of user mistakes rather than
    as file plumbing -- the catalogue is the point of this module and should be scannable.
    """
    good_g = glossary(WORK / "ok.csv")
    good_s = schema(WORK / "ok.avsc")

    bad_cols = WORK / "badcols.csv"
    bad_cols.write_bytes(b"colour,shape,size\nred,round,big\n")

    empty = WORK / "empty.csv"
    empty.write_bytes(b"Term,Business Definition\n")

    notcsv = WORK / "notcsv.xyz"
    notcsv.write_bytes(b"whatever")

    malformed = WORK / "malformed.csv"
    malformed.write_bytes(
        b"Term,Business Definition,Classification\n"
        b"Good Term,A fine definition,PII\n"
        b'Broken "quote,unterminated,Internal\n'
    )

    latin1 = WORK / "latin1.csv"
    latin1.write_bytes(
        "Term,Business Definition,Classification\nCafé Name,Le café,PII\n".encode("latin-1")
    )

    empty_schema = WORK / "empty.avsc"
    empty_schema.write_bytes(b'{"type":"record","name":"E","fields":[]}')

    return {
        "good_g": good_g,
        "good_s": good_s,
        "bad_cols": bad_cols,
        "empty": empty,
        "notcsv": notcsv,
        "malformed": malformed,
        "latin1": latin1,
        "empty_schema": empty_schema,
    }


def measure_error_actionability() -> None:
    """
    Score each message 0-2:
      0 = does not say what happened, or names an internal detail the user cannot act on
      1 = says what happened, but not the exact next step
      2 = says what happened, why, and the exact next command or change
    Scoring is done by hand afterwards; this only CAPTURES the real text.
    """
    from nexus_matcher import NexusMatcher

    fx = _bad_inputs()
    good_g, good_s = fx["good_g"], fx["good_s"]
    bad_cols, empty, notcsv = fx["bad_cols"], fx["empty"], fx["notcsv"]
    malformed, latin1, empty_schema = fx["malformed"], fx["latin1"], fx["empty_schema"]

    cases: list[tuple[str, object]] = []

    def add(name, fn):
        cases.append((name, fn))

    add(
        "glossary file does not exist",
        lambda: NexusMatcher.from_config().load_dictionary(WORK / "nope.csv"),
    )
    add(
        "glossary has unrecognised columns",
        lambda: NexusMatcher.from_config().load_dictionary(bad_cols),
    )
    add(
        "glossary is empty (header only)", lambda: NexusMatcher.from_config().load_dictionary(empty)
    )
    add(
        "glossary has an unknown extension",
        lambda: NexusMatcher.from_config().load_dictionary(notcsv),
    )
    add(
        "glossary has a malformed row",
        lambda: NexusMatcher.from_config().load_dictionary(malformed),
    )
    add("glossary is latin-1 not utf-8", lambda: NexusMatcher.from_config().load_dictionary(latin1))

    def match_before_load():
        NexusMatcher.from_config().match_schema(str(good_s))

    add("match_schema before load_dictionary", match_before_load)

    def constructor_no_args():
        NexusMatcher()

    add("NexusMatcher() with no arguments", constructor_no_args)

    def unknown_schema_format():
        m = NexusMatcher.from_config()
        m.load_dictionary(good_g)
        m.match_schema(str(good_s), schema_format="protobuf")

    add("unknown schema_format", unknown_schema_format)

    def schema_missing():
        m = NexusMatcher.from_config()
        m.load_dictionary(good_g)
        m.match_schema(str(WORK / "nope.avsc"))

    add("schema file does not exist", schema_missing)

    def zero_field_schema():
        m = NexusMatcher.from_config()
        m.load_dictionary(good_g)
        return m.match_schema(str(empty_schema))

    add("schema parses to zero fields", zero_field_schema)

    def bad_config_key():
        NexusMatcher.from_config({"auto_approve_treshold": 0.9})

    add("mistyped config key (dict)", bad_config_key)

    def index_from_build_index():
        import nexus_matcher as nm

        idx = nm.build_index(str(good_g))
        NexusMatcher.from_config(idx)

    add("hand build_index() result to NexusMatcher", index_from_build_index)

    captured = []
    for name, fn in cases:
        exc, msg = _capture(fn)
        captured.append({"case": name, "exception": exc, "message": msg[:600]})
    RESULTS["error_actionability"] = captured


# ---------------------------------------------------------------------------
# Cold start: process start -> first result, including index build
# ---------------------------------------------------------------------------
def measure_cold_start() -> None:
    try:
        import psutil

        psutil.cpu_percent(None)
        time.sleep(1)
        busy = psutil.cpu_percent(1)
    except ImportError:
        busy = None

    if busy is not None and busy > 10:
        RESULTS["cold_start"] = {"refused": f"CPU busy {busy:.1f}% -- above the 10% precondition"}
        return

    RESULTS["cold_start"] = {"cpu_busy_percent": busy, "measurements": []}
    from nexus_matcher import NexusMatcher

    for n in (1000, 10000):
        g = glossary(WORK / f"g{n}.csv", n=n)
        s = schema(WORK / f"s{n}.avsc")
        t0 = time.perf_counter()
        m = NexusMatcher.from_config()
        m.load_dictionary(g)
        m.match_schema(str(s))
        RESULTS["cold_start"]["measurements"].append(
            {"entries": n, "seconds": round(time.perf_counter() - t0, 2)}
        )


def main() -> None:
    measure_minimal_program()
    measure_error_actionability()
    measure_cold_start()
    out = pathlib.Path(os.environ.get("DX_OUT", WORK / "dx.json"))
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"WROTE {out}")

    mp = RESULTS["minimal_program"]
    print(f"\nlines of user code : {mp['lines_of_user_code']}")
    print(f"concepts required  : {mp['concepts_required']}  {mp['concept_list']}")
    print(f"config values      : {mp['config_values_required']}")
    print("\nerror actionability -- raw messages:")
    for row in RESULTS["error_actionability"]:
        first = (row["message"].splitlines() or [""])[0]
        print(f"  [{row['exception'] or 'no raise':22}] {row['case']}")
        print(f"      {first[:150]}")
    print("\ncold start:", json.dumps(RESULTS["cold_start"])[:200])


if __name__ == "__main__":
    main()
