"""
Reintroduce NM-0027: restore the 0.6 default threshold on get_low_confidence_fields.

The whole method is replaced rather than just the default value, because the fix changed
what the method MEANS -- no-argument now asks "was this auto-approved?" instead of
comparing against a number -- and the historical body is what makes the empty list
inevitable. Putting only `0.6` back into the new body would reintroduce a different,
milder bug.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/domain/models/entities.py"

# From the signature line to whatever comes next at method indentation, or end of file.
# Anchoring on "the rest of the file" would rot silently the day somebody adds a method
# after this one; this way the replay keeps applying, and if the signature itself is
# renamed the LookupError below reports a hole instead of a false pass.
METHOD_RE = re.compile(
    r"^    def get_low_confidence_fields\(.*?\n(?:.*?\n)*?(?=^    (?:def |@)|\Z)",
    re.M,
)

BROKEN = '''    def get_low_confidence_fields(self, threshold: float = 0.6) -> list[str]:
        """Get fields with low confidence top matches."""
        return [
            path
            for path, matches in self.results.items()
            if matches and matches[0].final_confidence < threshold
        ]
'''


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = METHOD_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0027 replay: get_low_confidence_fields not found in {TARGET}")
    path.write_text(text[: match.start()] + BROKEN + text[match.end() :], encoding="utf-8")
