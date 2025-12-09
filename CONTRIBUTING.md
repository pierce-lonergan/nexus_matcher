# Contributing to NexusMatcher

Thank you for your interest in contributing to NexusMatcher! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Style Guide](#style-guide)
8. [Architecture Guidelines](#architecture-guidelines)

---

## Code of Conduct

This project follows a standard Code of Conduct. Please be respectful and constructive in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Docker for running services

### Finding Issues

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Check `help wanted` for tasks where maintainers need assistance
- Feel free to propose new features via GitHub Issues

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/nexus-matcher.git
cd nexus-matcher
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install all dependencies including dev tools
pip install -e ".[full,dev]"

# Install pre-commit hooks
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Run linting
ruff check src tests
mypy src
```

---

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes
- `perf/` - Performance improvements

### 2. Make Your Changes

Follow the [Style Guide](#style-guide) and [Architecture Guidelines](#architecture-guidelines).

### 3. Write Tests

All new code should have tests. See [Testing](#testing) for details.

### 4. Update Documentation

- Update docstrings for new/changed functions
- Update README if adding new features
- Add module documentation in `docs/modules/` if creating new modules

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nexus_matcher --cov-report=html

# Run specific test file
pytest tests/unit/domain/test_models.py

# Run specific test
pytest tests/unit/domain/test_models.py::test_field_creation

# Run by marker
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m "not slow"  # Skip slow tests
```

### Test Structure

```
tests/
├── unit/                 # Fast, isolated tests
│   ├── domain/          # Domain layer tests
│   ├── application/     # Use case tests
│   ├── infrastructure/  # Adapter tests
│   └── presentation/    # API/CLI tests
├── integration/          # Tests with external dependencies
└── e2e/                  # End-to-end tests
```

### Writing Tests

```python
import pytest
from nexus_matcher.domain.models import Field

class TestField:
    """Tests for Field domain model."""
    
    def test_field_creation(self):
        """Should create field with valid attributes."""
        field = Field(
            path="customer.email",
            name="email",
            data_type="string",
        )
        
        assert field.name == "email"
        assert field.data_type == "string"
    
    def test_field_depth_calculation(self):
        """Should calculate depth from path."""
        field = Field(path="a.b.c", name="c", data_type="string")
        
        assert field.depth == 2
    
    @pytest.mark.parametrize("path,expected", [
        ("a", 0),
        ("a.b", 1),
        ("a.b.c.d", 3),
    ])
    def test_various_depths(self, path, expected):
        """Should handle various path depths."""
        field = Field(path=path, name="x", data_type="string")
        assert field.depth == expected
```

### Test Coverage

- Maintain minimum 80% coverage
- New code should have >90% coverage
- Cover edge cases and error conditions

---

## Submitting Changes

### 1. Ensure Quality

```bash
# Format code
black src tests

# Run linting
ruff check src tests --fix

# Type checking
mypy src

# Run tests
pytest
```

### 2. Commit Your Changes

```bash
git add .
git commit -m "feat: add semantic caching for embeddings"
```

#### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `perf`: Performance improvement
- `chore`: Maintenance

**Examples:**
```
feat(cache): add L1 LRU cache layer
fix(parser): handle nullable arrays in Avro
docs(readme): add deployment section
perf(embedding): enable INT8 quantization
```

### 3. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

### Pull Request Guidelines

- Fill out the PR template completely
- Link related issues
- Add screenshots for UI changes
- Ensure CI passes
- Request review from maintainers

---

## Style Guide

### Python Style

We follow PEP 8 with some modifications enforced by Ruff and Black.

```python
# Good
def match_field(
    self,
    field: Field,
    *,
    top_k: int = 5,
    min_confidence: float = 0.0,
) -> list[Match]:
    """Match a field to dictionary entries.
    
    Args:
        field: The field to match.
        top_k: Maximum number of matches to return.
        min_confidence: Minimum confidence threshold.
    
    Returns:
        List of matches sorted by confidence.
    
    Raises:
        ValueError: If top_k is less than 1.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    
    # Implementation...
```

### Type Hints

- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- Prefer `list[X]` over `List[X]` (Python 3.10+)

```python
from __future__ import annotations

def process_fields(fields: list[Field]) -> dict[str, list[Match]]:
    ...
```

### Documentation

- All public functions need docstrings (Google style)
- Include type information in docstrings
- Add examples for complex functions

### Import Order

```python
# Standard library
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party
import numpy as np
from pydantic import BaseModel

# Local
from nexus_matcher.domain.models import Field
from nexus_matcher.domain.ports import EmbeddingProvider

if TYPE_CHECKING:
    from nexus_matcher.domain.models import Match
```

---

## Architecture Guidelines

### Hexagonal Architecture

NexusMatcher follows hexagonal (ports & adapters) architecture:

```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                     │
│                  (API, CLI, Plugins)                     │
├─────────────────────────────────────────────────────────┤
│                   Application Layer                      │
│                    (Use Cases)                           │
├─────────────────────────────────────────────────────────┤
│                     Domain Layer                         │
│              (Models, Ports, Services)                   │
├─────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                    │
│                     (Adapters)                           │
└─────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Dependency Rule**: Inner layers don't depend on outer layers
2. **Ports**: Interfaces in domain layer define contracts
3. **Adapters**: Infrastructure implements ports
4. **Pure Domain**: Domain layer has no external dependencies

### Adding New Components

#### New Schema Parser

1. Create adapter in `infrastructure/adapters/schema_parsers/`
2. Implement `SchemaParser` protocol
3. Register in `pyproject.toml` entry points
4. Add tests

```python
# src/nexus_matcher/infrastructure/adapters/schema_parsers/xml.py
from nexus_matcher.domain.ports import SchemaParser
from nexus_matcher.domain.models import Schema, Field

class XmlSchemaParser(SchemaParser):
    """Parser for XML Schema Definition (XSD)."""
    
    def parse(self, content: str) -> Schema:
        # Implementation
        ...
    
    def parse_file(self, path: Path) -> Schema:
        return self.parse(path.read_text())
```

#### New Vector Store

1. Create adapter in `infrastructure/adapters/vector_stores/`
2. Implement `VectorStore` protocol
3. Register in entry points
4. Add integration tests

---

## Questions?

- Open a GitHub Issue for questions
- Check existing issues and documentation first
- Join discussions in GitHub Discussions

Thank you for contributing! 🎉
