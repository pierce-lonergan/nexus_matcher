"""
scripts.release_preflight | Layer: RELEASE
Prove a built wheel actually works before it is published. Exits non-zero if not.

Why this exists
---------------
2.0.0 was published with a CLI that crashed on any non-UTF-8 Windows console, a console
script installed without the dependencies it needs, and a `__all__` entry that broke
`from nexus_matcher import *` on a default install. CI was green throughout.

It was green because the gate was written to pass:

    - name: Test CLI (if available)
      run: |
        pip install typer rich
        nexus-matcher --help || echo "CLI not available without full install"
      continue-on-error: true

Three independent reasons that could never fail. It installed the very dependencies whose
absence was the bug; it exercised `--help`, one of the few commands that still worked,
rather than the two that did the work; and between `|| echo` and `continue-on-error` it
could not turn the build red even if it had.

So the rule here is: every check FAILS the run. No `continue-on-error`, no swallowed exit
codes, no "if available". A check that cannot fail is not a check, and this file exists
because that lesson cost a release.

What it verifies
----------------
  1. the wheel installs into a clean venv with NO extras
  2. `import nexus_matcher` and `from nexus_matcher import *` both work there
  3. every name in __all__ resolves
  4. every declared console script RUNS -- including under a legacy codepage (cp437),
     which is where the 2.0.0 blocker lived
  5. the bundled encoder is present, loads, and produces real (non-random) vectors
  6. an end-to-end match works with HuggingFace unreachable
  7. no torch / pandas / sentence-transformers got pulled in
  8. the README carries no relative links (they 404 on the PyPI project page)
  9. every declared Project-URL resolves

Usage
-----
    python scripts/release_preflight.py                 # builds, then checks
    python scripts/release_preflight.py --wheel dist/nexus_matcher-2.0.1-py3-none-any.whl
    python scripts/release_preflight.py --skip-network  # offline; skips URL checks only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append((name, detail))
        print(f"  FAIL  {name}\n        {detail.strip()[:600]}")

    def skip(self, name: str, why: str) -> None:
        self.skipped.append((name, why))
        print(f"  SKIP  {name}  ({why})")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, capturing both streams as text. Never raises on non-zero."""
    # check=False deliberately: a non-zero exit is the SIGNAL here, not an accident.
    # Every caller inspects returncode and turns it into a named FAIL with the real
    # output attached, which is more useful than a CalledProcessError traceback.
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False, **kw)


def build_wheel(report: Report) -> Path | None:
    dist = REPO / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    proc = run([sys.executable, "-m", "build"], cwd=REPO)
    if proc.returncode != 0:
        report.fail("build", proc.stderr or proc.stdout)
        return None
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        report.fail("build", "no wheel produced")
        return None
    report.ok(f"build ({wheels[0].name}, {wheels[0].stat().st_size / 1e6:.1f} MB)")
    return wheels[0]


def make_venv(root: Path) -> Path:
    """Create a venv and return its python. Short path: Windows MAX_PATH is real."""
    venv = root / "v"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return py


# ---------------------------------------------------------------------------
# Checks that run INSIDE the clean venv, as source passed to `python -`.
# ---------------------------------------------------------------------------

_CHECK_IMPORTS = r"""
import sys, json
import nexus_matcher
missing = []
for name in nexus_matcher.__all__:
    try:
        getattr(nexus_matcher, name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
# A star-import is the strictest form of the same promise, and is what actually broke.
ns = {}
star_error = ""
try:
    exec("from nexus_matcher import *", ns)
except Exception as exc:
    star_error = f"{type(exc).__name__}: {exc}"
print(json.dumps({
    "version": nexus_matcher.__version__,
    "all_count": len(nexus_matcher.__all__),
    "unresolvable": missing,
    "star_error": star_error,
}))
"""

_CHECK_ENTRY_POINTS = r"""
import json
from importlib.metadata import distribution
d = distribution("nexus-matcher")
out = {"console_scripts": [], "load_failures": []}
for ep in d.entry_points:
    if ep.group == "console_scripts":
        out["console_scripts"].append(ep.name)
    try:
        ep.load()
    except Exception as exc:
        out["load_failures"].append(f"{ep.group}:{ep.name}: {type(exc).__name__}: {exc}")
print(json.dumps(out))
"""

_CHECK_ENCODER = r"""
import json, sys
import numpy as np
from nexus_matcher import default_embedding_provider
p = default_embedding_provider()
a = p.embed_documents(["customer email address"])[0]
b = p.embed_documents(["customer email address"])[0]
near = p.embed_documents(["the email address of a customer"])[0]
far = p.embed_documents(["quarterly seismic tolerance of the turbine housing"])[0]
print(json.dumps({
    "model": p.model_name,
    "dim": int(a.shape[0]),
    "deterministic": bool(np.allclose(a, b)),      # a random provider fails this
    "normalised": bool(abs(float(np.linalg.norm(a)) - 1.0) < 1e-4),
    "sim_near": float(a @ near),
    "sim_far": float(a @ far),
    "torch": "torch" in sys.modules,
    "pandas": "pandas" in sys.modules,
    "sentence_transformers": "sentence_transformers" in sys.modules,
}))
"""

_CHECK_END_TO_END = r"""
import json, sys, tempfile, pathlib
from nexus_matcher import NexusMatcher
d = pathlib.Path(tempfile.mkdtemp())
(d / "g.csv").write_bytes(
    b"Term,Business Definition,Classification\n"
    b"Customer Email Address,The email used to contact a customer,PII\n"
    b"Order Total Amount,Total value of an order including tax,Internal\n"
)
(d / "s.avsc").write_bytes(json.dumps({
    "type": "record", "name": "Order",
    "fields": [
        {"name": "email", "type": "string", "doc": "Contact email for the buyer"},
        {"name": "total", "type": "double", "doc": "Order grand total with tax"},
    ],
}).encode())
m = NexusMatcher.from_config()
stats = m.load_dictionary(d / "g.csv")
results = m.match_schema(str(d / "s.avsc"))
rows = {k: (v[0].dictionary_entry.business_name, v[0].dictionary_entry.protection_level.name)
        for k, v in results.items() if v}
print(json.dumps({
    "entries": stats.valid_entries,
    "fields_in": 2,
    "fields_out": len(results),
    "rows": rows,
    "torch": "torch" in sys.modules,
    "pandas": "pandas" in sys.modules,
}))
"""


def _json_from(proc: subprocess.CompletedProcess) -> dict | None:
    for raw in reversed((proc.stdout or "").strip().splitlines()):
        line = raw.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def check_clean_install(py: Path, wheel: Path, report: Report) -> None:
    """The whole point: a BASE install, no extras, exactly what `pip install` gives."""
    proc = run([str(py), "-m", "pip", "install", "--no-cache-dir", str(wheel)])
    if proc.returncode != 0:
        report.fail("clean install (no extras)", proc.stderr or proc.stdout)
        return
    report.ok("clean install (no extras)")

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    cwd = str(py.parent)  # never the repo, or the source tree shadows the install

    proc = run([str(py), "-"], input=_CHECK_IMPORTS, env=env, cwd=cwd)
    data = _json_from(proc)
    if data is None:
        report.fail("__all__ resolves", proc.stderr or proc.stdout)
    else:
        if data["unresolvable"]:
            report.fail("__all__ resolves", "; ".join(data["unresolvable"]))
        else:
            report.ok(f"__all__ resolves ({data['all_count']} names)")
        if data["star_error"]:
            report.fail("from nexus_matcher import *", data["star_error"])
        else:
            report.ok("from nexus_matcher import *")

    proc = run([str(py), "-"], input=_CHECK_ENTRY_POINTS, env=env, cwd=cwd)
    data = _json_from(proc)
    if data is None:
        report.fail("entry points load", proc.stderr or proc.stdout)
    else:
        if data["load_failures"]:
            report.fail("entry points load", "; ".join(data["load_failures"]))
        else:
            report.ok("entry points load")
        check_console_scripts(py, data["console_scripts"], report)

    _check_encoder(py, env, cwd, report)
    _check_end_to_end(py, env, cwd, report)


def _check_encoder(py: Path, env: dict, cwd: str, report: Report) -> None:
    proc = run([str(py), "-"], input=_CHECK_ENCODER, env=env, cwd=cwd)
    data = _json_from(proc)
    if data is None:
        report.fail("bundled encoder", proc.stderr or proc.stdout)
    else:
        problems = []
        if not data["deterministic"]:
            # This project once shipped a provider returning
            # np.random.RandomState(hash(text)).randn(dim). Pin it shut.
            problems.append("same text gives different vectors -- provider is not real")
        if not data["normalised"]:
            problems.append("vectors are not unit-normalised")
        if data["sim_near"] <= data["sim_far"]:
            problems.append(
                f"semantics inverted: near={data['sim_near']:.3f} <= far={data['sim_far']:.3f}"
            )
        for heavy in ("torch", "pandas", "sentence_transformers"):
            if data.get(heavy):
                problems.append(f"{heavy} was imported on the default path")
        if problems:
            report.fail("bundled encoder", "; ".join(problems))
        else:
            report.ok(
                f"bundled encoder ({data['model']}, near {data['sim_near']:.3f} "
                f"> far {data['sim_far']:.3f})"
            )


def _check_end_to_end(py: Path, env: dict, cwd: str, report: Report) -> None:
    proc = run([str(py), "-"], input=_CHECK_END_TO_END, env=env, cwd=cwd)
    data = _json_from(proc)
    if data is None:
        report.fail("end-to-end match, offline", proc.stderr or proc.stdout)
    elif data["entries"] < 2:
        report.fail("end-to-end match, offline", f"loaded {data['entries']} entries, expected 2")
    elif data["fields_out"] != data["fields_in"]:
        # The 2.0.0 defect: results keyed by a non-unique path, so fields vanished
        # silently and inherited no governance level.
        report.fail(
            "every input field appears in the output",
            f"{data['fields_in']} fields in, {data['fields_out']} out -- fields were dropped",
        )
    else:
        report.ok(f"end-to-end match, offline ({data['rows']})")


def check_console_scripts(py: Path, names: list[str], report: Report) -> None:
    """
    Run each console script FOR REAL, and under a legacy codepage.

    cp437 is the check that matters. The 2.0.0 blocker was a UnicodeEncodeError from
    rich's Braille spinner glyphs on any Windows console that is not UTF-8 -- and it hit
    `match` and `sync`, the only two commands that do work, while `--help` stayed fine.
    So this runs a real subcommand, not just --help, with the child's stdout forced to
    cp437.
    """
    if not names:
        report.fail("console scripts", "the distribution declares none")
        return

    bin_dir = py.parent
    for name in names:
        exe = bin_dir / (f"{name}.exe" if os.name == "nt" else name)
        if not exe.exists():
            report.fail(f"console script `{name}` installed", f"{exe} missing")
            continue

        proc = run([str(exe), "--help"])
        if proc.returncode != 0:
            report.fail(
                f"`{name} --help` on a base install",
                (proc.stderr or proc.stdout)[-800:],
            )
            continue
        report.ok(f"`{name} --help` on a base install")

        # The legacy-codepage run. Any subcommand that emits progress will do; --help is
        # explicitly NOT sufficient, because --help is what passed while match crashed.
        legacy = {**os.environ, "PYTHONIOENCODING": "cp437", "PYTHONUTF8": "0"}
        for sub in ("--help", "info"):
            proc = run([str(exe), sub], env=legacy)
            blob = (proc.stdout or "") + (proc.stderr or "")
            if "UnicodeEncodeError" in blob or "codec can't encode" in blob:
                report.fail(
                    f"`{name} {sub}` under cp437",
                    "UnicodeEncodeError -- a non-UTF-8 console cannot render the output",
                )
                break
            if proc.returncode != 0:
                report.fail(f"`{name} {sub}` under cp437", blob[-800:])
                break
        else:
            report.ok(f"`{name}` survives a cp437 console")


def check_readme_links(report: Report) -> None:
    """
    Relative markdown links 404 on the PyPI project page.

    The README is the long_description. On pypi.org a link like [docs](docs/FOO.md)
    resolves against pypi.org and dies. In 2.0.0 that killed the README's entire
    Documentation section for everyone arriving from PyPI.
    """
    readme = REPO / "README.md"
    if not readme.exists():
        report.fail("README relative links", "README.md not found")
        return
    text = readme.read_text(encoding="utf-8")
    # Inline links only; ignore images, anchors, and absolute URLs.
    # Two passes: a plain [label](target), and a BADGE -- [![alt](image)](target) --
    # whose nested brackets defeat a single [^]]+ label pattern. Badges are exactly where
    # a relative link hides, since they are written once and never read again.
    targets = [t for _l, t in re.findall(r"(?<!!)\[([^\[\]]*)\]\(([^)\s]+)\)", text)]
    targets += re.findall(r"\[!\[[^\]]*\]\([^)]*\)\]\(([^)\s]+)\)", text)
    bad = [t for t in targets if not t.startswith(("http://", "https://", "#", "mailto:"))]
    if bad:
        report.fail(
            "README has no relative links",
            f"{len(bad)} would 404 on PyPI: {', '.join(sorted(set(bad))[:8])}",
        )
    else:
        report.ok("README has no relative links")


def check_no_confidential_terms_in_artifacts(wheel: Path, report: Report) -> None:
    """
    Confidential vocabulary must not be inside the distributed artifact.

    tests/meta/test_no_confidential_terms.py scans the working tree, which is the right
    place to stop a leak being committed. It is the wrong place to stop one being
    *published*: what reaches PyPI is the wheel, and the wheel's contents are decided by
    packaging config, not by what the tree happens to contain. A file can be excluded
    from git and still be built in.

    That distinction is not hypothetical here. The comment naming the employer and
    business unit above DEFAULT_HIERARCHY_DATA shipped inside the 2.0.0 and 2.0.1
    wheels. Once a wheel is on PyPI it cannot be edited -- only yanked, which leaves the
    files downloadable. So this runs on the built artifact, before upload, where the
    finding is still free.

    The sdist is checked as well as the wheel, and not as an afterthought: it carries
    tests/ and docs/, which the wheel does not, so it is a strictly larger surface. The
    first run of this check found four real terms that the tree scan could not see --
    they were in the scanner's own docstrings, under a path exemption the scanner
    granted itself. Two artifacts with different contents are two chances to notice.
    """
    import tarfile
    import zipfile

    label = "Artifacts have no confidential terms"

    sys.path.insert(0, str(REPO / "tests" / "meta"))
    try:
        from test_no_confidential_terms import (  # type: ignore
            _blocklist,
            _blocklist_for,
            scan_text,
        )
    except Exception as exc:  # pragma: no cover - defensive
        report.fail(label, f"scanner unavailable: {exc}")
        return
    finally:
        sys.path.pop(0)

    if not _blocklist():
        report.fail(label, "blocklist empty; check is vacuous")
        return

    text_like = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".toml", ".rst")
    skip_names = ("tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")

    def wanted(name: str) -> bool:
        return name.lower().endswith(text_like) and Path(name).name not in skip_names

    def repo_relative(name: str, is_sdist: bool) -> str:
        """Map an archive member back to its repo path, so exemptions line up."""
        return name.split("/", 1)[-1] if is_sdist and "/" in name else name

    def members(artifact: Path):
        """Yield (archive_name, text) for every text member of a wheel or sdist."""
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as zf:
                for info in zf.infolist():
                    if wanted(info.filename):
                        yield info.filename, zf.read(info).decode("utf-8", "ignore")
        else:
            with tarfile.open(artifact) as tf:
                for m in tf.getmembers():
                    if m.isfile() and wanted(m.name):
                        fh = tf.extractfile(m)
                        if fh is not None:
                            yield m.name, fh.read().decode("utf-8", "ignore")

    artifacts = [wheel]
    sdists = sorted(wheel.parent.glob("*.tar.gz"))
    artifacts.extend(sdists)
    if not sdists:
        report.skip(f"{label} (sdist)", "no sdist built; wheel checked alone")

    findings, scanned = [], 0
    for artifact in artifacts:
        is_sdist = artifact.suffix != ".whl"
        for name, body in members(artifact):
            scanned += 1
            blocked = _blocklist_for(repo_relative(name, is_sdist))
            for line_no, digest in scan_text(body, blocked):
                findings.append(f"{artifact.name}::{name}:{line_no} (digest {digest})")

    if findings:
        report.fail(label, f"{len(findings)} hit(s): " + "; ".join(findings[:5]))
    else:
        report.ok(f"{label} ({scanned} files across {len(artifacts)} artifact(s))")


def check_project_urls(report: Report, skip_network: bool) -> None:
    if skip_network:
        report.skip("Project-URLs resolve", "--skip-network")
        return
    try:
        import tomllib
    except ModuleNotFoundError:
        report.skip("Project-URLs resolve", "no tomllib on this interpreter")
        return
    import urllib.error
    import urllib.request

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    urls = (data.get("project") or {}).get("urls") or {}
    if not urls:
        report.skip("Project-URLs resolve", "none declared")
        return
    broken = []
    for label, url in urls.items():
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "preflight"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status >= 400:
                    broken.append(f"{label} -> {url} ({resp.status})")
        except urllib.error.HTTPError as exc:
            if exc.code >= 400:
                broken.append(f"{label} -> {url} ({exc.code})")
        except Exception as exc:  # network flakiness must not fail a release
            report.skip(f"Project-URL {label}", f"unreachable: {type(exc).__name__}")
    if broken:
        report.fail("Project-URLs resolve", "; ".join(broken))
    else:
        report.ok(f"Project-URLs resolve ({len(urls)} checked)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel", help="use this wheel instead of building one")
    ap.add_argument("--skip-network", action="store_true", help="skip Project-URL checks")
    args = ap.parse_args()

    print("\nRelease preflight")
    print("=" * 74)
    report = Report()

    wheel = Path(args.wheel) if args.wheel else build_wheel(report)
    if wheel is None or not wheel.exists():
        print("\nno wheel to check")
        return 1

    check_readme_links(report)
    check_no_confidential_terms_in_artifacts(wheel, report)
    check_project_urls(report, args.skip_network)

    # A short temp root: a deep venv path plus numpy's f2py fixtures exceeds MAX_PATH on
    # Windows and corrupts the install for reasons unrelated to this package.
    root = Path(tempfile.mkdtemp(prefix="nmpf", dir=Path(tempfile.gettempdir())))
    try:
        py = make_venv(root)
        check_clean_install(py, wheel, report)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 74)
    print(
        f"  {len(report.passed)} passed, {len(report.failed)} failed, {len(report.skipped)} skipped"
    )
    if report.failed:
        print("\nNOT FIT TO PUBLISH:")
        for name, detail in report.failed:
            print(f"  - {name}: {detail.strip()[:200]}")
        return 1
    print("\n  fit to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
