# Installing NexusMatcher

Four routes. All of them end with the **same wheel** carrying its own encoder — an int8
ONNX build of `bge-small-en-v1.5`, 33.8 MB inside the package — so none of them downloads a
model, contacts HuggingFace, or needs an account.

Pick by what your environment can reach:

| Your machine can reach | Use |
|---|---|
| PyPI | [1. PyPI](#1-pypi) |
| GitHub, not PyPI | [2. Install from GitHub](#2-install-from-github) |
| Nothing (airgapped) | [3. Offline bundle](#3-offline-bundle) |
| A clone, and you want to modify it | [4. From source](#4-from-source) |

---

## 1. PyPI

```bash
pip install nexus-matcher
```

## 2. Install from GitHub

For environments where PyPI is blocked but GitHub is not. `pip` builds the wheel from the
repository, and the encoder comes with it — the model files are committed, not fetched.

```bash
pip install "git+https://github.com/pierce-lonergan/nexus_matcher.git@v2.0.1"
```

Pin the tag. `@main` installs whatever is on main right now, which is not a release.

Dependencies still come from PyPI on this route. If **both** are blocked, use the offline
bundle.

## 3. Offline bundle

For an airgapped target: no package index, no network, at install time or at run time.

**On a machine with network**, build the bundle:

```bash
git clone https://github.com/pierce-lonergan/nexus_matcher.git
cd nexus_matcher
python scripts/make_offline_bundle.py
```

That writes `dist/nexus-matcher-offline-<version>-<platform>-py<XY>/` containing every
wheel — the package and all 38 transitive dependencies, about 65 MB — plus an `INSTALL.txt`
and a `MANIFEST.json` with a sha256 for each.

**Move the directory to the target** by whatever means you have, then:

```bash
python -m venv venv
venv/bin/pip install --no-index --find-links wheels nexus-matcher
```

Verify before trusting it:

```bash
python scripts/make_offline_bundle.py --verify <bundle-dir>   # digests
python -c "from nexus_matcher import default_embedding_provider as p; print(p().model_name)"
```

The second must print a provider ending in `(bundled)`. If it prints anything else, the
encoder did not ship and matching will not work offline.

### Building for a different target

`onnxruntime`, `numpy`, `tokenizers` and `rapidfuzz` are compiled wheels, so a bundle is
specific to one platform **and** one CPython version. The directory name and the manifest
both record which. To build for a target that is not the machine you are on:

```bash
python scripts/make_offline_bundle.py --platform manylinux2014_x86_64 --python-version 311
```

A dependency with no wheel for that target fails **there**, while you can still do
something about it, rather than three screens into a resolver error on the airgapped box.

### Verified, not asserted

The bundle path was tested end to end: built here, installed with `--no-index` into a fresh
venv with `socket.connect` and `socket.getaddrinfo` patched to raise, then a real
governance match run to completion.

```
ssn        -> Patient SSN    RESTRICTED   REVIEW
visit_dt   -> Visit Date     PUBLIC       REVIEW

network connect() attempts: 0
```

Zero connection attempts, and no `torch`, no `pandas`, no `fastapi`.

## 4. From source

```bash
git clone https://github.com/pierce-lonergan/nexus_matcher.git
cd nexus_matcher
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`[dev]` brings the test and gate tooling. To check the tree is sound:

```bash
pytest tests/                       # the suite
python scripts/museum_replay.py     # every historical defect must turn its gate red
python scripts/release_preflight.py # the wheel must work on a bare install
```

---

## What the base install gives you

The complete pipeline, not a stub: the bundled encoder, BM25 lexical retrieval (built in —
a numpy inverted index, no `rank-bm25`), CSV/Excel glossary loading, and the CLI. **No
extra is needed for the quickstart.**

Extras add optional backends only — `api` for the REST server, `vector-stores` for
Qdrant/HNSW, `cache` for Redis, `embeddings` for the torch encoder, `loaders` for
database/Parquet glossaries. Anything the default path needs is a real dependency, because
an extra you must install before the quickstart runs is a bug, and this package once
shipped three of them.

## Known install issue on Windows

`pip install` can fail with `OSError [WinError 206] The filename or extension is too long`
while unpacking **numpy**, if your virtualenv sits at a deep path. It is numpy's f2py test
data exceeding `MAX_PATH`, not this package, but you will hit it here because numpy is a
dependency.

Either enable long paths:

```
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
```

or create the venv somewhere short, such as `C:\venvs\nexus`.
