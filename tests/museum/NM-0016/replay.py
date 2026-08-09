"""
Reintroduce NM-0016: make the encoder return seeded random noise of the right shape.

Anchored on the `_encode` signature WITHOUT its default argument. The first version of
this replay pinned `batch_size: int = 64`, and the encoder lane later tuned that to 512 --
which silently broke the replay and turned this entry into a hole. A replay that anchors
on a tunable constant rots the moment somebody tunes it.
"""

from __future__ import annotations

import pathlib
import re

TARGET = "src/nexus_matcher/infrastructure/adapters/embedding_providers/bundled_onnx.py"

ANCHOR_RE = re.compile(
    r"^    def _encode\(self, texts: Sequence\[str\][^)]*\) -> np\.ndarray:\n",
    re.M,
)

BODY = (
    "        # NM-0016 replay: the shape is right and every value is garbage.\n"
    "        if not len(texts):\n"
    "            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)\n"
    "        _r = np.stack(\n"
    "            [\n"
    "                np.random.RandomState(abs(hash(t)) % (2**32)).randn(EMBEDDING_DIM)\n"
    "                for t in texts\n"
    "            ]\n"
    "        ).astype(np.float32)\n"
    "        _n = np.linalg.norm(_r, axis=1, keepdims=True)\n"
    "        np.maximum(_n, 1e-12, out=_n)\n"
    "        return _r / _n\n"
)


def apply(repo_root: pathlib.Path) -> None:
    path = repo_root / TARGET
    text = path.read_text(encoding="utf-8")
    match = ANCHOR_RE.search(text)
    if match is None:
        raise LookupError(f"NM-0016 replay: _encode signature not found in {TARGET}")
    path.write_text(text[: match.end()] + BODY + text[match.end() :], encoding="utf-8")
