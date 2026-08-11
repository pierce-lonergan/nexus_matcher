"""
scripts.add_confidential_term | Layer: TOOLING
Add a term to the confidential blocklist without it touching the repository.

The term is read with getpass, so it is not echoed to the terminal and does not enter
shell history. Only its digest is written. Nothing prints the term back.

    python scripts/add_confidential_term.py

Pass --check to test whether a term is already blocked without adding it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from getpass import getpass
from pathlib import Path

HASHES = Path(__file__).resolve().parents[1] / "tests" / "meta" / "confidential_term_hashes.txt"
SALT = b"nexus-matcher/confidential-terms/v1"


def canon(term: str) -> str:
    return "".join(ch for ch in term.lower() if ch.isalnum())


def digest(term: str) -> str:
    return hashlib.blake2b(canon(term).encode("utf-8"), key=SALT, digest_size=8).hexdigest()


def existing() -> tuple[list[str], set[str]]:
    lines = HASHES.read_text(encoding="utf-8").splitlines()
    return lines, {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; do not add")
    args = ap.parse_args()

    term = getpass("Term (not echoed): ").strip()
    if not term:
        print("nothing entered", file=sys.stderr)
        return 2
    if len(canon(term)) < 3:
        # Two-character terms match far too much ordinary text; the gate would be
        # muted within a day.
        print("term is too short to block without flooding false positives", file=sys.stderr)
        return 2

    lines, digests = existing()
    d = digest(term)

    if d in digests:
        print(f"already blocked (digest {d})")
        return 0
    if args.check:
        print(f"NOT blocked (digest {d})")
        return 1

    body = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    header = lines[: len(lines) - len(body)] if body else lines
    merged = sorted({*(ln.strip() for ln in body), d})
    HASHES.write_text("\n".join([*[h.rstrip() for h in header], *merged]) + "\n", encoding="utf-8")
    print(f"added (digest {d}); {len(merged)} terms blocked")
    print("Now run: pytest tests/meta/test_no_confidential_terms.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
