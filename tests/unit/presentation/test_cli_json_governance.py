"""
tests.unit.presentation.test_cli_json_governance | Layer: TEST
DX-002 -- the only documented machine-readable surface has to carry the governance
payload, has to be arithmetically checkable, and has to survive being piped.

## Relationships
# TESTS → presentation/cli/main :: _format_json, _scoring_weights, _status_console

`nexus-matcher match -f json` emitted `{id, business_name, logical_name, data_type}` and
three of the five score components. The single field the entire stated use case rests on
-- "so the object inherits that entry's classification" -- was absent, and the emitted
numbers could not reproduce the emitted confidence, so nobody could check the arithmetic
from the file. A fresh-eyes agent doing a real governance task abandoned the CLI and
rebuilt on the Python API. docs/research/fresh-eyes.md, DX-002.

Fixing it exposed a second defect on the same surface: the payload shared stdout with a
progress spinner and a summary line, so `-f json > results.json` produced a file no JSON
parser accepts. Covered here too, and in tests/museum/NM-0026.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

from nexus_matcher.application.use_cases.match_schema import (
    MatchingConfig,
    NexusMatcher,
    _signal_weights,
    _weighted_confidence,
)
from nexus_matcher.presentation.cli.main import (
    _JSON_PRECISION,
    _MATCHER_CONFIG_ATTR,
    _SCORE_COMPONENTS,
    _scoring_weights,
    app,
)
from nexus_matcher.shared.types.base import ProtectionLevel, ScoreBreakdown

from ._fakes import DEFAULT_SIGNALS, EXPECTED_CONFIDENCE, FakeStats, make_entry, make_match
from ._legacy_console import run_child

runner = CliRunner(env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"})

# Every emitted number is rounded to `_JSON_PRECISION` decimals, so each moves by at most
# 5e-(P+1). Recomputing the weighted sum accumulates one such error per score, one per
# weight, and one on the confidence itself -- eleven terms, all far below this bound.
# Derived from the constant rather than written as a literal, so tightening the precision
# tightens the check instead of leaving a stale tolerance behind.
ARITHMETIC_TOLERANCE = 10 ** -(_JSON_PRECISION - 1)


def _run_json(cli_inputs, extra=()):
    """Run `match -f json` in-process and return the parsed payload for the one field."""
    schema, dictionary = cli_inputs
    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "-f", "json", *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


# =============================================================================
# THE GOVERNANCE PAYLOAD
# =============================================================================


def test_json_carries_the_protection_level(cli_inputs, install_matcher):
    """
    The headline of DX-002. The stated use case of this library is that a matched field
    inherits the dictionary entry's classification; the classification was not in the one
    interface a script can consume, so the use case was impossible from the CLI.
    """
    install_matcher(
        {"customer.email": [make_match(entry=make_entry(protection_level=ProtectionLevel.PII))]}
    )
    entry = _run_json(cli_inputs)["customer.email"][0]["dictionary_entry"]

    assert entry["protection_level"] == "PII", (
        "the JSON output has no protection level, so a governance script cannot "
        "propagate the classification the whole use case is about"
    )


@pytest.mark.parametrize(
    "level",
    [
        ProtectionLevel.PUBLIC,
        ProtectionLevel.INTERNAL,
        ProtectionLevel.CONFIDENTIAL,
        ProtectionLevel.PII,
        ProtectionLevel.RESTRICTED,
    ],
)
def test_every_protection_level_survives_the_round_trip(cli_inputs, install_matcher, level):
    """
    Not just "a key is present". Emitting a constant, or the enum's repr, would satisfy a
    single-value test and still be useless -- and mislabelling RESTRICTED as INTERNAL is
    the specific failure that costs money.
    """
    install_matcher({"customer.email": [make_match(entry=make_entry(protection_level=level))]})
    entry = _run_json(cli_inputs)["customer.email"][0]["dictionary_entry"]

    assert entry["protection_level"] == level.value
    assert ProtectionLevel.from_string(entry["protection_level"]) is level


def test_json_carries_definition_and_domain(cli_inputs, install_matcher):
    """
    The other two the fresh-eyes agent needed and had none of. A reviewer cannot judge a
    match from the business name alone -- "Customer Email Address" in marketing and the
    same name in billing are different entries with different classifications.
    """
    install_matcher(
        {
            "customer.email": [
                make_match(
                    entry=make_entry(
                        definition="The email address used to contact a customer.",
                        domain="billing",
                    )
                )
            ]
        }
    )
    entry = _run_json(cli_inputs)["customer.email"][0]["dictionary_entry"]

    assert entry["definition"] == "The email address used to contact a customer."
    assert entry["domain"] == "billing"


def test_the_identity_fields_are_still_there(cli_inputs, fake_matcher):
    """The counterpart: adding governance fields must not drop what scripts already read."""
    entry = _run_json(cli_inputs)["customer.email"][0]["dictionary_entry"]

    assert entry["id"] == "dict_042"
    assert entry["business_name"] == "Customer Email Address"
    assert entry["logical_name"] == "cust_email"
    assert entry["data_type"] == "string"


# =============================================================================
# REPRODUCIBLE ARITHMETIC
# =============================================================================


def test_all_five_score_components_are_emitted(cli_inputs, fake_matcher):
    """
    Three of five were emitted -- `edit_distance` and `domain` were dropped -- so the
    numbers in the file could not add up to the confidence in the file no matter what the
    reader did with them.

    Values asserted individually rather than only through the sum, because three of the
    five weights are 0.05: a component swapped with another 0.05 component leaves the
    weighted total untouched and would pass a sum-only check while reporting the wrong
    number for both.
    """
    scores = _run_json(cli_inputs)["customer.email"][0]["scores"]
    fused, lexical, edit, type_, domain = DEFAULT_SIGNALS

    assert scores == {
        "fused_retrieval": fused,
        "lexical": lexical,
        "edit_distance": edit,
        "type": type_,
        "domain": domain,
    }


def test_the_weights_that_produced_the_total_are_emitted(cli_inputs, fake_matcher):
    """Without them the components are five numbers with no stated way to combine them."""
    weights = _run_json(cli_inputs)["customer.email"][0]["weights"]

    assert weights == {
        "fused_retrieval": 0.7,
        "lexical": 0.05,
        "edit_distance": 0.05,
        "type": 0.05,
        "domain": 0.15,
    }


def test_confidence_is_reproducible_from_the_file_alone(cli_inputs, fake_matcher):
    """
    The property the whole change exists for: an auditor holding only this file can
    recompute the number the decision was made on.

    This is the check a reader would actually perform -- multiply, add, compare -- and
    nothing but the file's own contents feeds it.
    """
    match = _run_json(cli_inputs)["customer.email"][0]
    recomputed = sum(match["scores"][key] * match["weights"][key] for key in match["weights"])

    assert match["scores"].keys() == match["weights"].keys(), (
        "a component with no weight, or a weight with no component, leaves the sum unverifiable"
    )
    assert abs(recomputed - match["confidence"]) < ARITHMETIC_TOLERANCE, (
        f"the emitted numbers do not reproduce the emitted confidence: "
        f"{recomputed} vs {match['confidence']}"
    )


def test_the_reproduced_confidence_is_the_hand_computed_one(cli_inputs, fake_matcher):
    """
    An absolute pin (H-004). The test above compares the file against itself, which would
    hold just as well if every number in it were wrong by the same factor. This one names
    the value: 0.7*0.9012 + 0.05*0.5 + 0.05*0.4211 + 0.05*0.8 + 0.15*0.25 = 0.754395.
    """
    match = _run_json(cli_inputs)["customer.email"][0]

    assert match["confidence"] == EXPECTED_CONFIDENCE


def test_the_fixture_confidence_is_the_matchers_own_arithmetic(cli_inputs):
    """
    Premise guard, not a gate on the CLI.

    The tests above are only meaningful if the fixture's confidence is the number the real
    matcher would have produced from those five signals. If the application's weighting
    ever diverges from the hand-computed constant, these tests would go on passing against
    an artificial world.
    """
    config = MatchingConfig()
    assert _weighted_confidence(DEFAULT_SIGNALS, _signal_weights(config)) == pytest.approx(
        EXPECTED_CONFIDENCE, abs=1e-12
    )


def test_the_emitted_weights_come_from_the_live_matcher(cli_inputs, install_matcher):
    """
    Not from `MatchingConfig()` defaults.

    Emitting the defaults would look right in every test that uses them and would hand a
    caller who tuned their weights a file whose arithmetic does not close -- the same
    defect this change closes, in a form that is harder to see.
    """
    tuned = MatchingConfig(
        semantic_weight=0.40,
        lexical_weight=0.20,
        edit_distance_weight=0.10,
        type_weight=0.10,
        domain_weight=0.20,
    )
    install_matcher({"customer.email": [make_match(config=tuned)]}, config=tuned)

    match = _run_json(cli_inputs)["customer.email"][0]

    assert match["weights"]["fused_retrieval"] == 0.4
    assert match["weights"]["domain"] == 0.2
    recomputed = sum(match["scores"][key] * match["weights"][key] for key in match["weights"])
    assert abs(recomputed - match["confidence"]) < ARITHMETIC_TOLERANCE


def _install_matcher_without_a_config(monkeypatch, results):
    """A matcher that cannot say what weights it used -- e.g. any older test double."""

    class _NoConfig:
        def load_dictionary(self, path):
            return FakeStats()

        def match_schema(self, path):
            return results

    assert not hasattr(_NoConfig(), _MATCHER_CONFIG_ATTR), "the premise of these two tests"
    monkeypatch.setattr("nexus_matcher.presentation.cli.main._get_matcher", _NoConfig)


def test_weights_that_do_not_explain_the_confidence_are_refused(cli_inputs, monkeypatch):
    """
    The failure mode has to be loud, and it has to be decided by ARITHMETIC rather than by
    whether a private attribute could be read.

    Here the confidences were produced with tuned weights and the matcher cannot report
    them. Emitting the shipped defaults would produce a file that looks complete, gets
    used as evidence, and does not add up. The command fails instead, says which numbers
    disagree, and names the formats that still work.
    """
    tuned = MatchingConfig(
        semantic_weight=0.40,
        lexical_weight=0.20,
        edit_distance_weight=0.10,
        type_weight=0.10,
        domain_weight=0.20,
    )
    _install_matcher_without_a_config(monkeypatch, {"customer.email": [make_match(config=tuned)]})
    schema, dictionary = cli_inputs

    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "-f", "json"])

    combined = result.output + result.stderr
    assert result.exit_code == 1, combined
    assert "would not be reproducible" in combined
    assert "--format table" in combined


def test_a_matcher_that_cannot_report_its_weights_still_works_when_they_check_out(
    cli_inputs, monkeypatch
):
    """
    The counterpart, and the reason the check is arithmetic and not a name lookup.

    Refusing on "no readable config" alone would take the JSON surface away from every
    caller whose matcher is not literally a NexusMatcher -- including this repository's own
    older test doubles -- punishing them for a coupling that is ours, not theirs. Weights
    that reproduce every emitted confidence ARE the weights that produced it.
    """
    _install_matcher_without_a_config(monkeypatch, {"customer.email": [make_match()]})

    match = _run_json(cli_inputs)["customer.email"][0]

    recomputed = sum(match["scores"][key] * match["weights"][key] for key in match["weights"])
    assert abs(recomputed - match["confidence"]) < ARITHMETIC_TOLERANCE
    assert match["confidence"] == EXPECTED_CONFIDENCE


# =============================================================================
# THE COUPLING TO THE LAYERS THIS READS FROM  (H-006)
# =============================================================================


def test_every_score_component_names_a_real_field(cli_inputs):
    """
    The CLI reads five score attributes and five weight attributes by name. A rename in
    the domain or application layer has to fail HERE, in the lane that owns the reader,
    rather than at a user's terminal.

    `semantic_score` was renamed to `fused_retrieval_score` while this change was being
    written, which is exactly the event this guards.
    """
    weight_fields = {f.name for f in dataclasses.fields(MatchingConfig)}

    for key, score_attr, weight_attr in _SCORE_COMPONENTS:
        assert hasattr(ScoreBreakdown(), score_attr), (
            f"the JSON writer emits {key!r} from ScoreBreakdown.{score_attr}, "
            f"which no longer exists"
        )
        assert weight_attr in weight_fields, (
            f"the JSON writer emits the {key!r} weight from MatchingConfig.{weight_attr}, "
            f"which no longer exists"
        )


def test_the_real_matcher_still_reports_its_weights_where_the_cli_looks():
    """
    Against the REAL `NexusMatcher`, not a double.

    The writer reads the config off a private attribute, and every other test in this file
    installs a stub that has one -- so all of them would keep passing on the day the real
    class renames it, and the arithmetic-verification fallback would silently start
    emitting the shipped defaults to real users who tuned their weights. That is H-006:
    two halves of a change landing in different lanes.

    No model is loaded: the ports are never touched by the constructor.
    """
    tuned = MatchingConfig(semantic_weight=0.42)
    matcher = NexusMatcher(
        embedding_provider=object(),  # type: ignore[arg-type]
        vector_store=object(),  # type: ignore[arg-type]
        config=tuned,
    )

    weights = _scoring_weights(matcher)

    assert weights["fused_retrieval"] == 0.42, (
        "the CLI could not read the real matcher's weights and fell back to the shipped "
        "defaults, so a tuned configuration would emit a file whose arithmetic is not its own"
    )
    assert tuple(weights[key] for key, _s, _w in _SCORE_COMPONENTS) == _signal_weights(tuned)


def test_the_component_table_covers_every_configured_weight():
    """
    The opposite drift: a sixth signal added to MatchingConfig, and to the confidence,
    that this writer knows nothing about. The file would keep looking self-consistent
    while silently omitting a term, and the recomputation would quietly stop matching.
    """
    declared = {f.name for f in dataclasses.fields(MatchingConfig) if f.name.endswith("_weight")}
    emitted = {weight_attr for _key, _score_attr, weight_attr in _SCORE_COMPONENTS}

    assert declared == emitted, (
        "MatchingConfig and the JSON writer disagree about how many weighted signals "
        "there are, so the emitted confidence would not be reproducible"
    )


def test_the_deprecated_alias_is_not_what_gets_read(cli_inputs, fake_matcher):
    """
    Reading through `semantic_score` would still work today -- it is a deprecated property
    -- and would emit a DeprecationWarning per match and break outright at 3.0.
    """
    attributes = {score_attr for _key, score_attr, _weight in _SCORE_COMPONENTS}

    assert "semantic_score" not in attributes
    assert "fused_retrieval_score" in attributes


# =============================================================================
# DETERMINISM -- two runs have to diff cleanly
# =============================================================================


def _key_order(text):
    """Every object's keys, in the order the document actually lists them."""
    orders = []

    def hook(pairs):
        orders.append([k for k, _ in pairs])
        return dict(pairs)

    json.loads(text, object_pairs_hook=hook)
    return orders


def test_every_object_in_the_document_has_sorted_keys(cli_inputs, install_matcher):
    """
    Field paths are fed in NON-alphabetical order on purpose. A sorted-output claim tested
    only against already-sorted input asserts nothing at all, and the dict literals in the
    writer are deliberately left in reading order so that dropping the sort shows up here.
    """
    install_matcher(
        {
            "zeta.updated_at": [make_match(field_path="zeta.updated_at")],
            "alpha.account_id": [make_match(field_path="alpha.account_id")],
            "mid.customer_email": [make_match(field_path="mid.customer_email")],
        }
    )
    schema, dictionary = cli_inputs
    result = runner.invoke(app, ["match", str(schema), "-d", str(dictionary), "-f", "json"])
    assert result.exit_code == 0, result.output

    for keys in _key_order(result.stdout):
        assert keys == sorted(keys), f"object keys are not sorted: {keys}"

    assert list(json.loads(result.stdout)) == [
        "alpha.account_id",
        "mid.customer_email",
        "zeta.updated_at",
    ]


def test_the_document_ends_with_a_newline(cli_inputs, fake_matcher, tmp_path):
    """A file without one shows `\\ No newline at end of file` in every diff, forever."""
    schema, dictionary = cli_inputs
    out = tmp_path / "results.json"

    result = runner.invoke(
        app, ["match", str(schema), "-d", str(dictionary), "-f", "json", "-o", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").endswith("}\n")


def test_two_runs_produce_byte_identical_output(tmp_path, cli_inputs):
    """
    The property that makes this file diffable: same input, same bytes.

    Two SEPARATE interpreters with different `PYTHONHASHSEED`, because that is the
    nondeterminism this repo has actually shipped -- ranking depended on the hash seed
    until the tie-break landed (H-005), and a dict or set iterated in hash order inside
    the writer would reintroduce it here. Comparing two calls inside one process cannot
    see it: the seed is fixed for the life of the interpreter.
    """
    _, dictionary = cli_inputs
    argv = ["match", "schema.json", "-d", str(dictionary), "-f", "json"]
    env = {"NEXUS_TEST_MULTI": "1"}
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    first = run_child(tmp_path, argv, encoding="utf-8", env_extra=env, hashseed=0)
    second = run_child(tmp_path, argv, encoding="utf-8", env_extra=env, hashseed=12345)

    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    assert second.returncode == 0, second.stderr.decode("utf-8", "replace")
    assert first.stdout == second.stdout, "two identical runs produced different bytes"
    assert json.loads(first.stdout)  # and the bytes are a document, not an empty stream


# =============================================================================
# THE NON-TTY PATH -- redirecting stdout has to yield JSON
# =============================================================================


def test_piped_json_output_is_parseable(tmp_path, cli_inputs):
    """
    `nexus-matcher match schema.avsc -d dict.csv -f json > results.json` used to write

        <spinner frame> Matching schema...
        { ...the document... }

        Summary: 0/3 fields auto-approved (0.0%)

    Progress and the summary both default to stdout, so the only documented
    machine-readable surface produced something no parser accepts, in the most obvious way
    anyone would script it. A pipe, not a terminal, is what a script actually gets.
    """
    _, dictionary = cli_inputs
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    proc = run_child(
        tmp_path,
        ["match", "schema.json", "-d", str(dictionary), "-f", "json"],
        encoding="utf-8",
        env_extra={"NEXUS_TEST_MULTI": "1"},
    )

    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    document = json.loads(proc.stdout.decode("utf-8"))  # the assertion
    assert len(document) == 3


def test_the_status_output_still_reaches_the_user(tmp_path, cli_inputs):
    """
    The counterpart, and the reason this is a redirect and not a deletion: moving the
    summary off stdout must not mean throwing it away. A user who pipes the payload still
    watches progress and still gets told what happened.
    """
    _, dictionary = cli_inputs
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    proc = run_child(
        tmp_path,
        ["match", "schema.json", "-d", str(dictionary), "-f", "json"],
        encoding="utf-8",
        env_extra={"NEXUS_TEST_MULTI": "1"},
    )

    assert "Summary" in proc.stderr.decode("utf-8", "replace")


def test_the_table_format_keeps_its_output_on_stdout(tmp_path, cli_inputs):
    """
    Nothing moves for the human path. stdout is nobody's data channel at `-f table`, and
    quietly relocating a user's results to stderr would break every existing pipeline.
    """
    _, dictionary = cli_inputs
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    proc = run_child(
        tmp_path,
        ["match", "schema.json", "-d", str(dictionary)],
        encoding="utf-8",
        columns="200",
    )

    stdout = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "Match Results" in stdout
    assert "Summary" in stdout


def test_writing_to_a_file_matches_what_a_pipe_receives(tmp_path, cli_inputs):
    """
    `-o results.json` and `> results.json` have to produce the same document. They take
    different branches in the CLI, and the branch that skipped the write is how the
    `--output` flag came to be a silent no-op in the first place.
    """
    _, dictionary = cli_inputs
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")
    written = tmp_path / "written.json"

    piped = run_child(
        tmp_path,
        ["match", "schema.json", "-d", str(dictionary), "-f", "json"],
        encoding="utf-8",
        env_extra={"NEXUS_TEST_MULTI": "1"},
    )
    to_file = run_child(
        tmp_path,
        ["match", "schema.json", "-d", str(dictionary), "-f", "json", "-o", str(written)],
        encoding="utf-8",
        env_extra={"NEXUS_TEST_MULTI": "1"},
    )

    assert to_file.returncode == 0, to_file.stderr.decode("utf-8", "replace")
    assert json.loads(written.read_text(encoding="utf-8")) == json.loads(
        piped.stdout.decode("utf-8")
    )


# =============================================================================
# THE ENCODING MATRIX  (NM-0001)
# =============================================================================

# A business name and a definition that no legacy Windows code page can encode: a Greek
# name, a Japanese word, an em dash and a non-breaking space.
_NON_ASCII_NAME = "Kundennummer — Πελάτης"
_NON_ASCII_DEFINITION = "顧客のメールアドレス (PII)"


@pytest.mark.parametrize("encoding", ["cp437", "cp850", "cp1252"])
def test_json_output_survives_a_legacy_codepage(tmp_path, cli_inputs, encoding):
    """
    The JSON surface inherits NM-0001. `match` died with a bare UnicodeEncodeError on any
    console whose code page cannot encode what it was handed, and a governance run is
    exactly where a non-ASCII business name turns up.
    """
    _, dictionary = cli_inputs
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    proc = run_child(
        tmp_path,
        ["match", "schema.json", "-d", str(dictionary), "-f", "json"],
        encoding=encoding,
        env_extra={
            "NEXUS_TEST_BUSINESS_NAME": _NON_ASCII_NAME,
            "NEXUS_TEST_DEFINITION": _NON_ASCII_DEFINITION,
        },
    )
    text = (proc.stdout + b"\n" + proc.stderr).decode(encoding, errors="replace")

    assert proc.returncode != 97, f"code page {encoding} never took effect: {text}"
    assert "UnicodeEncodeError" not in text
    assert "codec can't encode" not in text
    assert proc.returncode == 0, text


@pytest.mark.parametrize("encoding", ["cp437", "cp850", "cp1252"])
def test_a_non_ascii_entry_is_not_corrupted_on_the_way_out(tmp_path, cli_inputs, encoding):
    """
    Surviving is not enough. stdout carries `errors="backslashreplace"` so that an
    unencodable character degrades instead of aborting -- which for a DATA channel would
    mean handing a governance tool a mangled business name and no indication of it.

    The document stays pure ASCII (`ensure_ascii`), so the escape happens inside JSON,
    where `json.loads` gives the original string back exactly. That is the property, and
    it is asserted against the exact input rather than against "no exception".
    """
    _, dictionary = cli_inputs
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    proc = run_child(
        tmp_path,
        ["match", "schema.json", "-d", str(dictionary), "-f", "json"],
        encoding=encoding,
        env_extra={
            "NEXUS_TEST_BUSINESS_NAME": _NON_ASCII_NAME,
            "NEXUS_TEST_DEFINITION": _NON_ASCII_DEFINITION,
        },
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr).decode(encoding, "replace")

    entry = json.loads(proc.stdout.decode("ascii"))["customer.email"][0]["dictionary_entry"]

    assert entry["business_name"] == _NON_ASCII_NAME
    assert entry["definition"] == _NON_ASCII_DEFINITION


@pytest.mark.parametrize("encoding", ["cp437", "cp850", "cp1252"])
def test_the_codepages_really_reject_the_test_characters(encoding):
    """
    Guards the two tests above. If these code pages ever gained these characters, both
    would start passing for the wrong reason and would be pinning nothing.
    """
    import codecs

    codecs.lookup(encoding)
    with pytest.raises(UnicodeEncodeError):
        _NON_ASCII_NAME.encode(encoding)
    with pytest.raises(UnicodeEncodeError):
        _NON_ASCII_DEFINITION.encode(encoding)
