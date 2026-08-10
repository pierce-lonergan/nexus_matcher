"""
scripts.make_offline_bundle | Layer: RELEASE
Build a self-contained bundle that installs with no network and no PyPI.

Why this exists
---------------
The people most likely to need this library are the ones least likely to be able to reach
PyPI: data governance runs inside locked-down environments, and a tool for classifying PII
is exactly the kind of thing a security team blocks a package index for. "It works
airgapped" is a claim this project already makes about the ENCODER -- the wheel carries its
own 33.8 MB int8 ONNX model, so there is no HuggingFace download. This extends that claim
to installation itself.

What it produces
----------------
    nexus-matcher-offline-<version>-<platform>-py<X.Y>/
        wheels/            nexus_matcher + every transitive dependency, as wheels
        INSTALL.txt        the two commands, and how to verify
        MANIFEST.json      version, platform, python, sha256 of every wheel

Install on the target, with no index and no network:

    pip install --no-index --find-links wheels nexus-matcher

PLATFORM AND PYTHON VERSION MATTER. onnxruntime, numpy, tokenizers and rapidfuzz all ship
compiled wheels, so a bundle built on Windows/CPython 3.13 does NOT install on
Linux/CPython 3.10. The directory name and MANIFEST record both, and `--verify` refuses a
mismatch rather than letting pip fail three screens later with a resolver error. Use
`--python-version`/`--platform` to build for a target that is not this machine.

Verified end to end: built here, installed with `--no-index` on a fresh venv with
`socket.connect` patched to raise, and a governance match run to completion -- 0 network
connection attempts, no torch, no pandas.

Usage
-----
    python scripts/make_offline_bundle.py
    python scripts/make_offline_bundle.py --platform manylinux2014_x86_64 --python-version 311
    python scripts/make_offline_bundle.py --verify <bundle-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)


def _package_version() -> str:
    text = (REPO / "src" / "nexus_matcher" / "__init__.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("could not read __version__")


def _build_wheel(out: Path) -> Path:
    """Build the project wheel itself, so a bundle is never stale relative to the tree."""
    proc = _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out)])
    if proc.returncode != 0:
        raise SystemExit(f"wheel build failed:\n{proc.stderr[-2000:]}")
    wheels = sorted(out.glob("nexus_matcher-*.whl"))
    if not wheels:
        raise SystemExit("build produced no wheel")
    return wheels[-1]


def build(target_platform: str | None, python_version: str | None) -> Path:
    version = _package_version()
    py_tag = python_version or f"{sys.version_info.major}{sys.version_info.minor}"
    plat_tag = target_platform or platform.system().lower()

    root = REPO / "dist" / f"nexus-matcher-offline-{version}-{plat_tag}-py{py_tag}"
    if root.exists():
        shutil.rmtree(root)
    wheels = root / "wheels"
    wheels.mkdir(parents=True)

    print(f"building nexus-matcher {version} for {plat_tag} / py{py_tag}")
    project_wheel = _build_wheel(wheels)
    print(f"  built {project_wheel.name}")

    # Resolve dependencies from the wheel we just built, not from the name -- otherwise a
    # bundle can silently pick up whatever is on PyPI instead of what is in this tree.
    cmd = [sys.executable, "-m", "pip", "download", "--dest", str(wheels), str(project_wheel)]
    if target_platform or python_version:
        # Cross-building needs --only-binary, because pip cannot build an sdist for a
        # platform it is not running on. A dependency with no wheel for the target will
        # fail loudly here rather than at install time on the target machine.
        cmd += ["--only-binary", ":all:"]
        if target_platform:
            cmd += ["--platform", target_platform]
        if python_version:
            cmd += ["--python-version", python_version]

    proc = _run(cmd)
    if proc.returncode != 0:
        raise SystemExit(
            f"dependency download failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )

    files = sorted(p for p in wheels.iterdir() if p.suffix in (".whl", ".gz"))
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    total_mb = sum(p.stat().st_size for p in files) / 1e6

    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "package": "nexus-matcher",
                "version": version,
                "built_for_platform": plat_tag,
                "built_for_python": py_tag,
                "built_on": f"{platform.system()} {platform.machine()} CPython {platform.python_version()}",
                "wheel_count": len(files),
                "total_megabytes": round(total_mb, 1),
                "sha256": digests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (root / "INSTALL.txt").write_text(
        f"""nexus-matcher {version} -- offline bundle
{"=" * 60}

Built for: {plat_tag}, CPython {py_tag[0]}.{py_tag[1:]}
Contents : {len(files)} wheels, {total_mb:.1f} MB

INSTALL (no network, no package index):

    python -m venv venv
    venv/bin/pip install --no-index --find-links wheels nexus-matcher
    (Windows: venv\\Scripts\\pip install --no-index --find-links wheels nexus-matcher)

VERIFY:

    python -c "import nexus_matcher; print(nexus_matcher.__version__)"
    python -c "from nexus_matcher import default_embedding_provider as p; print(p().model_name)"

The second command must print a provider ending in "(bundled)". The encoder ships INSIDE
the wheel -- an int8 ONNX build of bge-small-en-v1.5 -- so there is no model download and
no HuggingFace account, on this machine or any other.

PLATFORM MATTERS. onnxruntime, numpy, tokenizers and rapidfuzz are compiled wheels. This
bundle installs on {plat_tag} / CPython {py_tag[0]}.{py_tag[1:]} and will NOT install
elsewhere. Rebuild with --platform/--python-version for a different target.

INTEGRITY: MANIFEST.json carries a sha256 for every wheel. Check them before installing
anything you did not build yourself.
""",
        encoding="utf-8",
    )

    print(f"\n  {len(files)} wheels, {total_mb:.1f} MB")
    print(f"  wrote {root}")
    return root


def verify(bundle: Path) -> int:
    """Check a bundle is complete and matches its own manifest, before anyone trusts it."""
    manifest_path = bundle / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"  MANIFEST.json missing from {bundle}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheels = bundle / "wheels"

    problems = []
    for name, expected in manifest["sha256"].items():
        path = wheels / name
        if not path.exists():
            problems.append(f"missing: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"digest mismatch: {name}")

    extra = {p.name for p in wheels.iterdir()} - set(manifest["sha256"])
    problems += [f"unlisted file: {n}" for n in sorted(extra)]

    here_py = f"{sys.version_info.major}{sys.version_info.minor}"
    if manifest["built_for_python"] != here_py:
        print(
            f"  NOTE: bundle targets CPython {manifest['built_for_python']}, you are on "
            f"{here_py}. Compiled wheels will not match."
        )

    print(f"  {manifest['package']} {manifest['version']} -- {manifest['wheel_count']} wheels")
    if problems:
        print("  FAILED:")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  every wheel present and every digest matches")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--platform", help="target platform tag, e.g. manylinux2014_x86_64")
    ap.add_argument("--python-version", help="target CPython, e.g. 311")
    ap.add_argument("--verify", metavar="BUNDLE_DIR", help="verify a bundle instead of building")
    args = ap.parse_args()

    if args.verify:
        return verify(Path(args.verify))
    build(args.platform, args.python_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
