"""Reintroduce NM-0005: key results by full_path instead of a unique key."""

from __future__ import annotations

import pathlib

TARGET = "src/nexus_matcher/application/use_cases/match_schema.py"
FIXED = "results[self._unique_result_key(field, results)] = tuple(field_results)"
BROKEN = "results[field.full_path] = tuple(field_results)"


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    if FIXED not in text:
        raise SystemExit(f"NM-0005 replay: anchor not found in {TARGET}")
    path.write_text(text.replace(FIXED, BROKEN, 1), encoding="utf-8")
