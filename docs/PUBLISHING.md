# Publishing NexusMatcher to PyPI

This guide covers publishing NexusMatcher to the Python Package Index (PyPI).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [First-Time Setup](#first-time-setup)
3. [Publishing Process](#publishing-process)
4. [Automated Publishing](#automated-publishing)
5. [Version Management](#version-management)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

```bash
pip install build twine
```

### Accounts Required

1. **PyPI Account**: https://pypi.org/account/register/
2. **TestPyPI Account**: https://test.pypi.org/account/register/
3. (Optional) **GitHub Account** for automated publishing

---

## First-Time Setup

### 1. Configure PyPI Credentials

#### Option A: Using `.pypirc` file (Traditional)

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-<your-api-token>

[testpypi]
username = __token__
password = pypi-<your-test-api-token>
```

**Note**: Use API tokens, not passwords. Generate at:
- PyPI: https://pypi.org/manage/account/token/
- TestPyPI: https://test.pypi.org/manage/account/token/

#### Option B: Using Environment Variables

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-api-token>
```

#### Option C: Trusted Publishing (Recommended for GitHub Actions)

No credentials needed! Configure at:
https://pypi.org/manage/project/nexus-matcher/settings/publishing/

### 2. Reserve Package Name (First Publish)

First, publish to TestPyPI to verify everything works:

```bash
# Build
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*
```

Then reserve on PyPI with your first release.

---

## Publishing Process

### Quick Publish (Scripts)

```bash
# Linux/macOS
./scripts/publish.sh          # Publish to PyPI
./scripts/publish.sh test     # Publish to TestPyPI
./scripts/publish.sh build    # Build only

# Windows (PowerShell)
.\scripts\publish.ps1
.\scripts\publish.ps1 test
.\scripts\publish.ps1 build
```

### Manual Publish

#### Step 1: Clean Previous Builds

```bash
rm -rf dist/ build/ *.egg-info src/*.egg-info
```

#### Step 2: Run Tests

```bash
pytest tests/unit
```

#### Step 3: Update Version

Edit `src/nexus_matcher/__init__.py`:

```python
__version__ = "2.1.0"  # Update version
```

Update `CHANGELOG.md` with release notes.

#### Step 4: Build

```bash
python -m build
```

This creates:
- `dist/nexus_matcher-2.1.0-py3-none-any.whl` (wheel)
- `dist/nexus_matcher-2.1.0.tar.gz` (source)

#### Step 5: Verify Package

```bash
# Check metadata
twine check dist/*

# Test install locally
pip install dist/nexus_matcher-2.1.0-py3-none-any.whl
python -c "import nexus_matcher; print(nexus_matcher.__version__)"
pip uninstall nexus-matcher
```

#### Step 6: Upload to TestPyPI (Optional)

```bash
twine upload --repository testpypi dist/*
```

Test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    nexus-matcher==2.1.0
```

#### Step 7: Upload to PyPI

```bash
twine upload dist/*
```

#### Step 8: Verify Installation

```bash
pip install nexus-matcher==2.1.0
python -c "import nexus_matcher; print(nexus_matcher.__version__)"
```

#### Step 9: Create Git Tag

```bash
git tag -a v2.1.0 -m "Release v2.1.0"
git push origin v2.1.0
```

---

## Automated Publishing

### GitHub Actions (Recommended)

The repository includes `.github/workflows/publish.yml` for automated publishing.

#### Trusted Publishing Setup

1. Go to https://pypi.org/manage/project/nexus-matcher/settings/publishing/
2. Add a new publisher:
   - Owner: `plonergan` (your GitHub username)
   - Repository: `nexus-matcher`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

#### Publishing via GitHub Release

1. Create a new release on GitHub
2. Tag with version (e.g., `v2.1.0`)
3. GitHub Actions automatically:
   - Builds the package
   - Tests on multiple Python versions
   - Publishes to PyPI

#### Manual Trigger

1. Go to Actions → Publish to PyPI
2. Click "Run workflow"
3. Select target (testpypi or pypi)

---

## Version Management

### Semantic Versioning

Follow [SemVer](https://semver.org/):

- **MAJOR** (3.0.0): Breaking API changes
- **MINOR** (2.1.0): New features, backwards compatible
- **PATCH** (2.0.1): Bug fixes, backwards compatible

### Version Locations

Update version in:

1. `src/nexus_matcher/__init__.py` (source of truth)
2. `CHANGELOG.md` (release notes)

The `pyproject.toml` reads version dynamically from `__init__.py`.

### Pre-release Versions

```python
__version__ = "2.1.0a1"   # Alpha
__version__ = "2.1.0b1"   # Beta
__version__ = "2.1.0rc1"  # Release candidate
```

---

## Troubleshooting

### Common Issues

#### "File already exists"

```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists.
```

**Solution**: You cannot overwrite existing versions. Bump the version number.

#### "Invalid distribution file"

```
InvalidDistribution: Cannot find .dist-info directory
```

**Solution**: Rebuild with clean environment:
```bash
rm -rf dist/ build/ *.egg-info
python -m build
```

#### "Username/password incorrect"

**Solution**: Use API tokens instead of passwords:
```ini
[pypi]
username = __token__
password = pypi-AgEI...
```

#### Package Not Found After Upload

Wait 1-5 minutes for PyPI to index. Check status at:
https://pypi.org/project/nexus-matcher/

### Testing Locally

```bash
# Install from local wheel
pip install dist/nexus_matcher-*.whl

# Install from local source
pip install -e .

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    nexus-matcher
```

### Checking Package Contents

```bash
# List wheel contents
unzip -l dist/nexus_matcher-*.whl

# List sdist contents
tar -tzf dist/nexus_matcher-*.tar.gz

# View package metadata
pip show nexus-matcher
```

---

## Release Checklist

- [ ] All tests passing (`pytest`)
- [ ] Linting clean (`ruff check src tests`)
- [ ] Type checking clean (`mypy src`)
- [ ] Version updated in `__init__.py`
- [ ] `CHANGELOG.md` updated
- [ ] Documentation updated
- [ ] Clean build successful
- [ ] TestPyPI upload successful
- [ ] TestPyPI installation verified
- [ ] PyPI upload successful
- [ ] PyPI installation verified
- [ ] Git tag created and pushed
- [ ] GitHub release created

---

## Package URLs

- **PyPI**: https://pypi.org/project/nexus-matcher/
- **TestPyPI**: https://test.pypi.org/project/nexus-matcher/
- **Documentation**: https://nexus-matcher.readthedocs.io
- **Repository**: https://github.com/pierce-lonergan/nexus_matcher

---

*Happy Publishing! 🚀*
