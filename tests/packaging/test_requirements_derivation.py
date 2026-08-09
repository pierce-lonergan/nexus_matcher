"""
tests.packaging.test_requirements_derivation | Layer: GATE
requirements.txt is DERIVED from pyproject.toml, and the tree, the wheel and the
changelog all describe the same release.

Two holes, closed here
----------------------
**Drift.** H-006 occurrence 2: `requirements.txt` still carried both packaging defects
already fixed in `pyproject.toml`, because the two files sat in different lanes and
nothing tied them together. It had drifted again by 2026-08-09 -- headed "v2.0.0",
declaring `pandas` (replaced by openpyxl) and `rank-bm25` (BM25 is built in and imports
nothing), and missing `onnxruntime`, `tokenizers` and `tomli`, the three dependencies that
make a `start.sh` install able to load the bundled encoder at all.

Noticing drift is not a fix. `scripts/gen_requirements.py` renders the file, and this
regenerates and diffs it on every test run, so drift is impossible rather than merely
detectable. The one-line remedy is in the failure message.

**Version coherence.** `tests/hazards/test_h006_reachability.py` already pins `__version__`
against the newest CHANGELOG heading. That covers two of the four places a version lives.
This adds the other two: what the WHEEL would declare, and whether the defects fixed since
the last release are written down anywhere a reader would find them.

Lane note
---------
The natural home for the version-coherence half is `tests/hazards/test_h006_reachability.py`,
which is another lane's file this session. It lives here instead, next to the other
derived-artifact checks, rather than crossing a lane boundary to sit in a nicer place.

Why the wheel metadata is cheap here
------------------------------------
Not `python -m build`: that packages a 32 MB ONNX encoder and takes tens of seconds. PEP
517's `prepare_metadata_for_build_wheel` asks the SAME backend for the SAME metadata and
writes only a `.dist-info`. Measured around 0.1 s of backend work in this environment --
a figure that needs re-measuring on an idle machine, and which is quoted only to justify
running it in the normal suite rather than as a performance claim.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MUSEUM = REPO / "tests" / "museum"


def _generator():
    """Import scripts/gen_requirements.py -- it is a script, not an installed module."""
    path = REPO / "scripts" / "gen_requirements.py"
    spec = importlib.util.spec_from_file_location("gen_requirements_under_test", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN = _generator()


# =============================================================================
# DERIVATION
# =============================================================================


def test_requirements_txt_is_exactly_what_pyproject_implies():
    """
    The gate. A dependency changed in pyproject.toml and not regenerated fails HERE,
    in the normal suite, instead of surfacing as a user whose install is missing the
    encoder runtime.
    """
    delta = GEN.diff("requirements.txt")
    assert not delta, (
        "requirements.txt has drifted from pyproject.toml.\n"
        "Fix: python scripts/gen_requirements.py\n\n" + delta
    )


def test_the_render_is_deterministic():
    """
    A generator whose output depends on dict ordering makes the diff above fail at random
    and teaches everyone to rerun it until it passes -- which is how a gate becomes noise.
    """
    assert GEN.render("requirements.txt") == GEN.render("requirements.txt")


def test_the_generated_file_is_not_trivially_empty():
    """Guards the vacuous pass: an empty render matches an empty file."""
    text = GEN.render("requirements.txt")
    pinned = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(pinned) >= 20, f"only {len(pinned)} requirements rendered"
    assert any(ln.startswith("numpy") for ln in pinned), "numpy vanished from the render"


def test_the_python_310_marker_survives_rendering():
    """
    `tomli>=2.0.0; python_version < '3.11'` is the whole reason
    `NexusMatcher.from_config("matching.toml")` works on 3.10. A renderer that drops
    environment markers would silently install tomli on every interpreter -- or, worse,
    drop the line and take 3.10 back to a ModuleNotFoundError on a documented call.
    """
    lines = GEN.render("requirements.txt").splitlines()
    marked = [ln for ln in lines if ln.startswith("tomli")]
    assert marked == ["tomli>=2.0.0; python_version < '3.11'"], marked


def test_every_extra_is_either_composed_in_or_excused_in_writing():
    """
    A new extra in pyproject.toml is a decision about requirements.txt whether or not
    anyone makes it. This forces the answer to be written down.
    """
    unexplained = GEN.missing_reasons()
    assert not unexplained, (
        f"extras that are neither in COMPOSITION nor EXCLUDED: {unexplained}. "
        "Decide whether requirements.txt needs them, in scripts/gen_requirements.py."
    )


def test_no_exclusion_reason_names_an_extra_that_is_gone():
    """The same rot `.ci-exceptions.yaml` is checked for: an exception excusing nothing."""
    stale = GEN.stale_reasons()
    assert not stale, f"EXCLUDED names extras that no longer exist: {stale}"


def test_every_group_named_in_the_composition_still_exists():
    """
    The composition is a list of extra NAMES, so deleting an extra from pyproject.toml
    breaks the render. Failing here says which name went, instead of surfacing as a
    DerivationError from inside the generator with no context about who asked for it.
    """
    declared = set(GEN.read_groups())
    for filename, groups in GEN.COMPOSITION.items():
        missing = [name for name in groups if name not in declared]
        assert not missing, (
            f"{filename} is composed from extras pyproject.toml no longer declares: "
            f"{missing}. Remove them from COMPOSITION in scripts/gen_requirements.py."
        )


def test_conflicting_declarations_are_refused_not_resolved_by_luck(tmp_path: Path):
    """
    Two spellings of the same dependency is exactly the drift shape, one file inward.
    pip would silently take whichever it saw last; the generator raises instead.
    """
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "x"\n'
        'dependencies = ["rapidfuzz>=3.0.0"]\n'
        "[project.optional-dependencies]\n"
        'parsers = ["rapidfuzz>=2.0.0"]\n'
        "loaders = []\n"
        "sparse = []\n"
        "api = []\n"
        "cli = []\n"
        "dev = []\n",
        encoding="utf-8",
    )
    with pytest.raises(GEN.DerivationError, match="declared twice"):
        GEN.render("requirements.txt", tmp_path / "pyproject.toml")


def test_the_same_pin_written_two_ways_is_not_a_conflict(tmp_path: Path):
    """
    The other half of the rule above. `a>=1,<2` and `a<2,>=1` are one requirement; if
    they read as a conflict, the generator cries wolf and someone loosens the check.
    """
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "x"\n'
        'dependencies = ["pydantic>=2.0.0,<3.0.0"]\n'
        "[project.optional-dependencies]\n"
        'parsers = ["pydantic<3.0.0,>=2.0.0"]\n'
        "loaders = []\n"
        "sparse = []\n"
        "api = []\n"
        "cli = []\n"
        "dev = []\n",
        encoding="utf-8",
    )
    rendered = GEN.render("requirements.txt", tmp_path / "pyproject.toml")
    assert rendered.count("pydantic") == 1


def test_self_referencing_extras_are_expanded_not_written_through(tmp_path: Path):
    """
    `full = ["nexus-matcher[api,cli]"]` written verbatim into requirements.txt makes
    `pip install -r` fetch this package from PyPI over the local checkout -- so the tests
    would run against a different version of the code than the one on disk, silently.
    """
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "x"\n'
        "dependencies = []\n"
        "[project.optional-dependencies]\n"
        'parsers = ["nexus-matcher[loaders,sparse]"]\n'
        'loaders = ["pyarrow>=14.0.0"]\n'
        'sparse = ["rank-bm25>=0.2.2"]\n'
        "api = []\n"
        "cli = []\n"
        "dev = []\n",
        encoding="utf-8",
    )
    rendered = GEN.render("requirements.txt", tmp_path / "pyproject.toml")
    assert "nexus-matcher[" not in rendered
    assert "pyarrow>=14.0.0" in rendered
    assert "rank-bm25>=0.2.2" in rendered


def test_the_diff_actually_reports_a_difference(tmp_path: Path):
    """
    `diff()` returning "" is the whole pass condition of the first test in this file. If
    it could only ever return "", that test would be a decoration.
    """
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "requirements.txt").write_text("numpy==0.0.1\n", encoding="utf-8")
    delta = GEN.diff("requirements.txt", tmp_path)
    assert "numpy==0.0.1" in delta and delta.startswith("---")


def _run_check(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "gen_requirements.py"), "--check"],
        cwd=repo,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


def test_the_script_run_as_ci_would_run_it_reports_this_tree_as_current():
    """The same fact as the first test, through the interface a CI step would use."""
    proc = _run_check(REPO)
    assert proc.returncode == 0, f"requirements.txt is stale:\n{proc.stderr}"


def test_the_scripts_exit_code_actually_goes_to_1_on_a_drifted_tree(tmp_path: Path):
    """
    The exit code is the entire value of `--check` as a CI step. A `--check` that returns
    0 whatever it finds is the `|| true` pattern with extra steps -- so it is proven here
    against a deliberately drifted copy of the repo, not asserted.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "gen_requirements.py", tmp_path / "scripts")
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "requirements.txt").write_text("numpy==0.0.1\n", encoding="utf-8")

    proc = _run_check(tmp_path)
    assert proc.returncode == 1, (
        f"--check exited {proc.returncode} on a tree whose requirements.txt is a single "
        f"wrong pin. stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "STALE" in proc.stderr


# =============================================================================
# VERSION COHERENCE -- the tree, the wheel, the changelog, the museum
# =============================================================================

_ASK_THE_BACKEND = r"""
import json, tempfile, pathlib
from email.parser import Parser
import hatchling.build as backend
directory = tempfile.mkdtemp()
dist_info = backend.prepare_metadata_for_build_wheel(directory)
text = (pathlib.Path(directory) / dist_info / "METADATA").read_text(encoding="utf-8")
message = Parser().parsestr(text)
print(json.dumps({
    "name": message["Name"],
    "version": message["Version"],
    "requires_python": message["Requires-Python"],
    "requires_dist": message.get_all("Requires-Dist") or [],
}))
"""


def _wheel_metadata() -> dict:
    """
    What the built wheel WOULD declare, without building it.

    Asked of the real backend named in [build-system], so this cannot agree with a
    hand-written model of hatchling's behaviour -- there is no second implementation for
    it to be wrong alongside.
    """
    pytest.importorskip(
        "hatchling",
        reason="the build backend from [build-system].requires; without it there is no "
        "wheel metadata to compare against",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _ASK_THE_BACKEND],
        cwd=REPO,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    assert proc.returncode == 0, f"the build backend could not produce metadata:\n{proc.stderr}"
    line = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{"))
    data = json.loads(line)
    assert data["requires_dist"], "the wheel would declare no dependencies at all"
    return data


def _declared_version() -> str:
    init = (REPO / "src" / "nexus_matcher" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    assert match, "__version__ not found in src/nexus_matcher/__init__.py"
    return match.group(1)


def _changelog() -> str:
    return (REPO / "CHANGELOG.md").read_text(encoding="utf-8")


def _newest_release_heading() -> str:
    headings = re.findall(r"^##\s*\[?([0-9]+\.[0-9]+\.[0-9]+)\]?", _changelog(), re.M)
    assert headings, "no version headings found in CHANGELOG.md"
    return headings[0]


def test_the_wheel_would_declare_the_version_the_tree_and_changelog_agree_on():
    """
    H-006's first occurrence was `__version__` and the changelog disagreeing. The wheel is
    the third place, and the only one a user ever sees: the tree built 2.0.0 while the
    changelog claimed 2.0.1, and nothing compared either of them against the artifact.
    """
    metadata = _wheel_metadata()
    version = _declared_version()
    assert metadata["version"] == version, (
        f"the wheel would be {metadata['version']} but __version__ is {version}"
    )
    assert version == _newest_release_heading(), (
        f"__version__ is {version} but the newest CHANGELOG heading is {_newest_release_heading()}"
    )
    assert metadata["name"].lower().replace("_", "-") == "nexus-matcher"


def test_the_wheels_own_dependencies_are_the_ones_requirements_txt_ships():
    """
    requirements.txt is generated from pyproject TEXT; the wheel is generated by the
    BACKEND. Comparing them catches anything the backend does that reading the TOML does
    not predict -- and it is the artifact, not the source, that a user installs.
    """
    metadata = _wheel_metadata()
    core = [
        GEN.normalize_requirement(req) for req in metadata["requires_dist"] if "extra ==" not in req
    ]
    rendered = GEN.render("requirements.txt").splitlines()
    generated = {
        GEN.normalize_requirement(line) for line in rendered if line.strip() and line[0] != "#"
    }
    missing = sorted(str(req) for req in core if req not in generated)
    assert not missing, (
        "the wheel declares core dependencies that requirements.txt does not ship:\n  "
        + "\n  ".join(missing)
        + "\nA `pip install -r requirements.txt` environment is missing something the "
        "published package needs."
    )


def _python_floor(requires_python: str) -> tuple[int, int]:
    match = re.search(r">=\s*(\d+)\.(\d+)", requires_python or "")
    assert match, f"unparseable Requires-Python: {requires_python!r}"
    return int(match.group(1)), int(match.group(2))


def _oldest_classified_python(pyproject_text: str) -> tuple[int, int]:
    found = sorted(
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r'"Programming Language :: Python :: (\d+)\.(\d+)"', pyproject_text)
    )
    assert found, "no per-minor Python classifiers declared"
    return found[0]


def test_the_floor_comparison_can_tell_a_mismatch_from_a_match():
    """
    Proven on hand-written inputs before it is trusted on the real ones. A comparison that
    cannot distinguish the two cases would pass on the real tree for the wrong reason --
    the shape that let nineteen tests miss a transposed multiply.
    """
    classifiers = (
        '"Programming Language :: Python :: 3.11"\n"Programming Language :: Python :: 3.12"'
    )
    assert _oldest_classified_python(classifiers) == (3, 11)
    assert _python_floor(">=3.11") == (3, 11)
    assert _python_floor(">=3.10") != _oldest_classified_python(classifiers)


def test_requires_python_agrees_with_the_oldest_classifier():
    """
    A floor raised in `requires-python` and not in the classifiers advertises support for
    an interpreter pip will refuse to install on -- and the tomli marker is written
    against that same floor.
    """
    metadata = _wheel_metadata()
    floor = _python_floor(metadata["requires_python"])
    oldest = _oldest_classified_python((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert floor == oldest, (
        f"Requires-Python floor is {floor} but the oldest Python classifier is {oldest}"
    )


# =============================================================================
# THE MUSEUM AND THE CHANGELOG
# =============================================================================


def _museum_entries() -> dict[str, str]:
    """{museum id -> the `fixed_in` it declares}."""
    entries: dict[str, str] = {}
    if not MUSEUM.is_dir():
        return entries
    for directory in sorted(p for p in MUSEUM.iterdir() if p.is_dir() and p.name.startswith("NM-")):
        meta = directory / "defect.yaml"
        if not meta.is_file():
            continue
        match = re.search(r"^fixed_in:\s*(.+?)\s*$", meta.read_text(encoding="utf-8"), re.M)
        entries[directory.name] = match.group(1).strip("\"'") if match else ""
    return entries


def _unreleased_section() -> str:
    text = _changelog()
    match = re.search(r"^##\s*\[?Unreleased\]?.*?$(.*?)(?=^##\s)", text, re.M | re.S)
    assert match, "CHANGELOG.md has no `## [Unreleased]` section"
    return match.group(1)


def test_there_are_museum_entries_to_check():
    """Guards the vacuous pass -- an empty museum satisfies the next test trivially."""
    entries = _museum_entries()
    assert entries, "no museum entries found; the check below would pass over nothing"
    assert all(fixed for fixed in entries.values()), (
        "these museum entries declare no `fixed_in`, so nothing can decide whether they "
        f"need a changelog line: {sorted(k for k, v in _museum_entries().items() if not v)}"
    )


def test_every_defect_fixed_since_the_last_release_is_in_the_changelog():
    """
    A museum entry is a defect that reached a user. If it was fixed after the last
    release and the changelog does not say so, the next release ships a fix nobody can
    find -- and the museum becomes the only record of a user-visible change, which is
    not where anyone looks.

    The check is on the ID, deliberately. Prose can describe a fix without being
    traceable back to the replay that proves it stays fixed; an id is checkable.
    """
    unreleased_ids = sorted(
        museum_id
        for museum_id, fixed_in in _museum_entries().items()
        if fixed_in.lower() == "unreleased"
    )
    section = _unreleased_section()
    undocumented = [museum_id for museum_id in unreleased_ids if museum_id not in section]
    assert not undocumented, (
        "these defects were fixed since the last release and appear nowhere in the "
        f"CHANGELOG's Unreleased section: {undocumented}\n"
        "Add a line naming each id under `## [Unreleased]`, in the form\n"
        f"  - **{undocumented[0]}** -- <the symptom, from its defect.yaml>."
    )


def test_no_museum_entry_claims_a_release_that_does_not_exist():
    """
    `fixed_in: 2.0.2` on an entry when the changelog's newest release is 2.0.1 means the
    museum is describing a release that was never cut -- the version-drift shape again,
    pointing the other way.
    """
    released = set(re.findall(r"^##\s*\[?([0-9]+\.[0-9]+\.[0-9]+)\]?", _changelog(), re.M))
    unknown = sorted(
        f"{museum_id} -> {fixed_in}"
        for museum_id, fixed_in in _museum_entries().items()
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", fixed_in) and fixed_in not in released
    )
    assert not unknown, (
        "museum entries naming a version with no CHANGELOG heading:\n  " + "\n  ".join(unknown)
    )
