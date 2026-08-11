"""
nexus_matcher.infrastructure.adapters.embedding_providers.bundled_onnx | Layer: INFRASTRUCTURE
The zero-setup embedding provider: an int8 ONNX encoder shipped inside the wheel.

## Relationships
# IMPLEMENTS  → domain/ports/embedding_provider :: EmbeddingProvider protocol
# DEPENDS_ON  → onnxruntime, tokenizers :: inference only, NO torch
# USED_BY     → shared/container :: the default provider

## Attributes
# Security: Never touches the network. No download, no HuggingFace, no telemetry.
# Performance: ~1180 texts/sec encoding the real FHIR glossary on an idle 32-thread CPU
# Reliability: Works in an airgapped container with no configuration

## Why this is the default

The previous default (`SentenceTransformersProvider`) needs torch (~800 MB installed) and
downloads weights from HuggingFace on first use. That fails in an airgapped environment,
behind a proxy, in a locked-down CI image, or simply on a slow connection -- and it fails
at RUN time, not install time, which is the worst moment.

This provider carries its own weights. Measured against the torch fp32 path on the
labelled benchmark:

    build                 size            P@1      Recall@10   throughput   torch
    torch fp32       130 MB + ~800 MB   0.5596      0.8648       973 q/s     yes
    int8 ONNX (this)      33.8 MB       0.5378      0.8532      1238 q/s     no

int8 costs roughly 2 points of P@1 and buys a 27% throughput gain plus the removal of
torch.

"Roughly" is load-bearing. int8 inference is NOT batch-invariant: ONNX Runtime selects
different quantised GEMM kernels per input shape, so the SAME corpus scores
P@1 0.5276 at batch 8, 0.5378 at batch 64 and 0.5436 at batch 128 -- a 1.6-point spread
from batch size alone, wider than several effects that are worth acting on. Hold batch
size fixed when comparing anything, or you are measuring kernel selection rather than the
change you made. The fp32 torch path does not have this property. That is a good trade for eliminating setup, and the transformer path
stays available via `pip install nexus-matcher[embeddings]` when those points matter.

Both rows use the model's official CLS pooling. An earlier measurement of this same file
reported only -1.2 P@1, which was wrong: it compared torch-CLS against an ONNX export that
sentence-transformers had silently defaulted to MEAN pooling, because the exported
directory carries no 1_Pooling config. Measured directly, CLS 0.5378 vs mean 0.5407 is
within noise -- the pooling was never the story, the quantisation was. If you re-measure
this, pin the pooling on both sides or you will compare two variables at once.

## What is NOT used, and why

Static embeddings (model2vec / potion) are ~30 MB, need only numpy, and are ~15-70x
faster still. They were measured and rejected as a default: on a FHIR-derived corpus
built to mirror the flattened-Avro use case they cost **8.9 points of P@1** (0.407 ->
0.318). A separate finding that seven TRANSFORMER encoders from 22M to 335M parameters
all landed within 0.03 P@1 does NOT generalise to static models -- the gap is roughly
three times what their published MTEB ratio implies. For a tool whose output decides
whether a field inherits a PII classification, nine points is not purchasable with
latency. `StaticEmbeddingProvider` exists as a fallback for environments without
onnxruntime, not as a recommendation.

## Encoder throughput, and what did NOT move it

Indexing is dominated by this file, so it has been optimised against the real FHIR
glossary (4598 entries) rather than uniform synthetic text -- uniform text hides padding
waste entirely and would have scored every change below as a no-op. Idle 32-thread CPU,
best of interleaved repeats:

    encoder                                            time     texts/sec
    ORT default threads, char-sort, 64 rows/batch      8.84s        520
    + intra-op threads capped at 8                     6.90s        667
    + tokenise once, sort by token length, budget      3.91s       1177

That is **2.26x**, and it is all in `session.run` (98.8% of what remains). Retrieval
quality was measured on the same corpus and did NOT move: P@1 0.2757 -> 0.2815 with 45
queries gained and 36 lost, exact McNemar p=0.37. That is the int8 encoder's known
batch-composition churn, not an improvement -- do not quote it as one.

Measured and REJECTED, so they do not get proposed again:

  * **io_binding** to skip an output copy: 0.992x, i.e. very slightly slower. The premise
    was wrong -- writing the entire [batch, seq, hidden] output for the whole corpus is
    236 MB, which numpy fills in 0.020s against a 3.9s encode. It is 0.5% of the time.
  * **Truncating the graph output to CLS** so the full hidden state is never materialised:
    same 0.5% ceiling as above, and it would need graph surgery plus an `onnx` dependency
    on a provider whose entire selling point is not having heavy dependencies.
  * **ORT_SEQUENTIAL execution mode**: 1.003x. Noise. This graph has no parallel branches
    for inter-op scheduling to exploit.
  * **Disabling the thread pool's spin-wait**: 1.49x vs 1.52x for the plain thread cap,
    i.e. no help once threads are capped. It only ever looked good on a loaded machine,
    which is the condition it flatters.
  * **A persistent content-hash vector cache** in this provider: `application/ingest.py`
    already skips re-embedding unchanged rows via `content_hash`, so a second cache here
    would duplicate that logic one layer down and add its own invalidation bugs. The gap
    it would actually close is that `GlossaryIndex` cannot be saved to disk at all, which
    belongs in ingest, not here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nexus_matcher.shared.types.base import Result

# The bundled encoder. Query-side instruction is part of the model contract: BGE models
# are trained with it on the query side ONLY, and omitting it cost 5.3 points of P@1.
BUNDLED_MODEL_DIR = Path(__file__).resolve().parents[3] / "_models" / "bge-small-en-v1.5-onnx-int8"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384
MAX_TOKENS = 512

# Cap on rows x padded width in a single session.run, i.e. how much work one call does.
#
# This, not the row count, is what the encoder's cost actually tracks. Measured on the
# FHIR corpus (4598 real glossary definitions) at a fixed 8 intra-op threads, sweeping
# ROW count alone falls off a cliff -- 64 rows takes 7.7s, 128 rows takes 17.2s, because a
# 128-row batch of the long entries is 128x417 tokens and stops fitting in cache. Capping
# the token count instead lets short entries ride in batches of hundreds while long ones
# drop to a dozen, and every call stays the same size. Budgets from 1024 to 6144 measured
# indistinguishable (3.9-4.8s); 4096 is the one in that tied set with the least padding
# waste (1.04x) and the fewest calls.
#
# It also bounds peak memory: the [rows, width, 384] float32 output can no longer exceed
# ~6 MB per call, where a 64-row batch of 417-token entries was allocating 41 MB.
MAX_BATCH_TOKENS = 4096

# Hard cap on ROWS in one session.run, applied on top of MAX_BATCH_TOKENS. Re-derived
# 2026-08-11 on the FHIR corpus (4598 entries + 1556 queries, 252,168 real tokens) by
# benchmarks/exp_encoder_batch_size.py; artifact benchmarks/results/exp_encoder_batch_size.json.
#
# THE PREVIOUS DEFAULT OF 512 WAS NOT A CAP AT ALL. Against a 4096-token budget no batch
# on this corpus ever reaches 512 rows, so 512, 1024 and 4096 produce byte-identical batch
# plans (65 batches, widest 315 rows) and byte-identical embeddings. Whatever 512 was
# chosen to do, it was doing nothing; the token budget was the only thing shaping batches.
#
# Throughput, interleaved best-of-3, each row beside the band measured on IDENTICAL code
# in the same session -- H-007: the band is machine state, not a constant, so a speedup
# quoted without one is not a result:
#
#     1 intra-op thread     band 0.7%   16: 1.074x  32: 1.056x  64: 1.030x  512: 1.000x
#     8 (the shipped cap)   band 6.2%   16: 1.274x  32: 1.386x  64: 1.294x  512: 1.000x
#
# 32 and not 16 because the two regimes disagree and H-003 makes a batch-scheduling knob
# prove itself in both: 16 is 1.8 points better at one thread and 11.2 points WORSE at the
# thread count that actually ships. 32 is the only value at or near the top of both. Take
# the 1-thread margin as the durable one -- it reproduced at 4.8% and 5.6% in two quiet
# windows (bands 1.0% and 0.7%). A third run landed at 36% CPU busy, put it at 6.8%, and
# had a band of 15.3%; it certifies nothing and is recorded UNMEASURABLE rather than
# averaged in. The 8-thread margin swings 10.6-38.6% with machine state; do not quote it.
#
# ACCURACY DOES NOT CHOOSE THIS VALUE and no accuracy claim is made for it. int8 inference
# is not batch-invariant and the churn is large -- 886 of 1556 queries change rank between
# 32 and 512 -- but it is symmetric: 45 gained at rank 1, 48 lost, exact McNemar p = 0.84.
# Every size measured was inconclusive against 512; among the five that batch differently
# from it at all (16, 32, 64, 128, 256) the p-values run 0.33 to 0.84, and 32's 0.84 is the
# least distinguishable of them. 1024 and 4096 are p = 1 by construction. The 1.67-point P@1
# "regression" once blamed on batch_size came from a 300-query fixture and did not survive
# a paired test; it needs new evidence that does before anyone acts on it again.
#
# Padding is not the mechanism either, despite a retracted 2.230x figure for 512 -- that
# belongs to the pre-token-budget fixed-window encoder. On today's code the gap is 1.0151x
# at 32 against 1.0282x at 512, i.e. 1.3% of the tokens buying 4.8-38.6% of the time. What is
# actually bought is smaller session.run calls, not fewer padded tokens.
DEFAULT_BATCH_SIZE = 32

# Intra-op threads when the caller does not say. ONNX Runtime's own default is one thread
# per physical core, which on an idle 32-thread workstation measured 6.07s against 4.00s
# at 8 threads -- a 1.52x loss, with the two distributions not overlapping at all across 6
# interleaved repeats. bge-small's GEMMs at these batch sizes are too small to keep more
# than about 8 threads busy, so past that ONNX Runtime is paying fork/join on every op for
# cores with nothing to do. Measured against the PREVIOUS fixed-64 batching the same cap
# was worth 1.28x, so this is a property of the model, not of the batching above it.
#
# Capping rather than fixing at 8 leaves small machines, where ORT's default is already at
# or below the knee, exactly as they were.
MAX_DEFAULT_THREADS = 8


def bundled_model_available() -> bool:
    """True when the wheel actually carries the weights."""
    return (BUNDLED_MODEL_DIR / "model_quantized.onnx").is_file()


class BundledOnnxProvider:
    """
    Embedding provider backed by the int8 ONNX encoder shipped in this package.

    Requires `onnxruntime` and `tokenizers`, both pure-wheel installs. Does NOT require
    torch, transformers, or sentence-transformers, and never contacts the network.

    Args:
        model_dir: Override the model location. Defaults to the bundled copy.
        query_instruction: Prefix applied to QUERIES only. Pass "" to disable, but
            measure first -- dropping it cost 5.3 points of P@1 on our benchmark.
        num_threads: ONNX Runtime intra-op threads. None uses min(8, cpu_count); see
            MAX_DEFAULT_THREADS for why ORT's own default is not used. Set it explicitly
            when running many workers per host to stop them oversubscribing the same
            cores.

    Example:
        provider = BundledOnnxProvider()
        vectors = provider.embed_documents(["Customer Account Balance"])
        query = provider.embed_single("cust_acct_bal")
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        query_instruction: str = QUERY_INSTRUCTION,
        num_threads: int | None = None,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else BUNDLED_MODEL_DIR
        self._query_instruction = query_instruction
        self._num_threads = num_threads
        self._session: Any = None
        self._tokenizer: Any = None

    # -- lifecycle --------------------------------------------------------

    def _load(self) -> None:
        if self._session is not None:
            return

        onnx_path = self._model_dir / "model_quantized.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"No bundled encoder at {onnx_path}. This usually means the package was "
                f"built without its model data. Reinstall nexus-matcher, or pass "
                f"model_dir=..., or use SentenceTransformersProvider with "
                f"pip install nexus-matcher[embeddings]."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "onnxruntime is required for the bundled encoder. "
                "Install with: pip install onnxruntime"
            ) from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "tokenizers is required for the bundled encoder. "
                "Install with: pip install tokenizers"
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self._num_threads or self._default_threads()
        # Deterministic single-provider execution; no GPU probing, no surprises.
        self._session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )

        tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        # Padding is deliberately NOT enabled. `_encode` tokenises the whole input in one
        # call so it can sort by true token length, and a tokenizer with padding on would
        # pad that single call to the longest text in the ENTIRE input -- 417 tokens for
        # every 5-token entry on the FHIR corpus. Each batch is padded in numpy instead,
        # to its own width.
        self._tokenizer = tokenizer

    @staticmethod
    def _default_threads() -> int:
        return max(1, min(MAX_DEFAULT_THREADS, os.cpu_count() or 1))

    # -- properties -------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "bge-small-en-v1.5-onnx-int8 (bundled)"

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def max_tokens(self) -> int:
        return MAX_TOKENS

    @property
    def is_offline(self) -> bool:
        """True: this provider never reaches the network."""
        return True

    # -- encoding ---------------------------------------------------------

    @staticmethod
    def _plan_batches(order: np.ndarray, lengths: Sequence[int], max_rows: int) -> list[list[int]]:
        """
        Group already-length-sorted rows so no call exceeds MAX_BATCH_TOKENS.

        A batch's cost is rows x the longest member, because every row is padded to it.
        Rows are added while that product stays under budget, so short entries travel in
        large batches and long ones in small ones, and every session.run does about the
        same amount of work.
        """
        batches: list[list[int]] = []
        current: list[int] = []
        widest = 0
        # `.tolist()` rather than iterating the array: it yields Python ints, which index
        # the token list far faster than numpy scalars do.
        for i in order.tolist():
            candidate = max(widest, lengths[i])
            over_budget = candidate * (len(current) + 1) > MAX_BATCH_TOKENS
            if current and (over_budget or len(current) >= max_rows):
                batches.append(current)
                current, widest = [i], lengths[i]
            else:
                current.append(i)
                widest = candidate
        if current:
            batches.append(current)
        return batches

    def _encode(self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        expected = {i.name for i in self._session.get_inputs()}
        out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)

        # Tokenise EVERYTHING once, then encode in TRUE TOKEN-LENGTH order and scatter
        # back to the caller's order.
        #
        # Every row in a batch is padded to the batch's longest member, and attention is
        # quadratic in that padded length, so padding is real compute. Measured on the
        # FHIR corpus (4598 real glossary definitions, token lengths 5..417, p50 26), the
        # padded token count against the real one:
        #
        #     natural order, 64 rows/batch                    3.83x
        #     character-length order, 64 rows/batch           1.55x
        #     TOKEN-length order, 64 rows/batch               1.12x
        #     TOKEN-length order, 4096-token budget           1.04x
        #
        # The middle row is why this no longer sorts on character count. Doing so avoided
        # a second tokenisation pass, but the saving was illusory: tokenising up front
        # costs one pass either way (measured at 0.9% of the encoder's time, so it was
        # never worth optimising around), and the character proxy was leaving 40% more
        # padding on the table than the sort it was standing in for.
        #
        # On an idle machine, against the previous character-sorted fixed-64 encoder and
        # holding the thread count equal, this is 6.90s -> 3.91s on the FHIR corpus, or
        # 1.77x. Padded tokens only fall 1.49x, so roughly half the gain is packing and
        # the rest is the batch-size cliff described at MAX_BATCH_TOKENS.
        #
        # NOTE: this changes which texts share a batch. The int8 ONNX encoder is not
        # batch-invariant, so embeddings can differ in the last bits from a run before
        # this change. That was already true of any change to batch_size. Measured on the
        # FHIR corpus, P@1 went 0.2757 -> 0.2815 with 45 queries gained and 36 lost
        # (exact McNemar p=0.37). That is churn, not a gain -- do not quote it as one.
        encoded = self._tokenizer.encode_batch(list(texts))
        lengths = [len(e.ids) for e in encoded]
        order = np.argsort(lengths, kind="stable")

        for positions in self._plan_batches(order, lengths, batch_size):
            width = max(lengths[i] for i in positions)
            # Pad here rather than in the tokenizer: the tokenizer would have to pad the
            # single up-front call to the longest text in the whole input, which is the
            # 3.83x case above with extra steps.
            ids = np.zeros((len(positions), width), dtype=np.int64)
            mask = np.zeros((len(positions), width), dtype=np.int64)
            for row, i in enumerate(positions):
                token_ids = encoded[i].ids
                ids[row, : len(token_ids)] = token_ids
                mask[row, : len(token_ids)] = 1

            feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in expected:
                # Single-sequence input, so BERT's segment ids are all zero. Pinned by
                # test_token_type_ids_are_all_zero, because if a future tokenizer.json
                # ever emitted a second segment this shortcut would silently encode the
                # wrong thing rather than fail.
                feeds["token_type_ids"] = np.zeros_like(ids)
            feeds = {k: v for k, v in feeds.items() if k in expected}

            hidden = self._session.run(None, feeds)[0]

            # BGE uses CLS pooling, not mean pooling. Using the wrong pooling silently
            # degrades every score rather than raising, so it is pinned here explicitly.
            vectors = hidden[:, 0, :].astype(np.float32)

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)
            # Scatter back to the caller's ordering, not the sorted one.
            out[positions] = vectors / norms

        return out

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """Encode DICTIONARY entries. No instruction prefix -- BGE is asymmetric."""
        return self._encode(list(texts), batch_size)

    def embed_queries(
        self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """Encode QUERIES, applying the model's query instruction."""
        return self._encode([self._query_instruction + t for t in texts], batch_size)

    # -- EmbeddingProvider protocol ---------------------------------------

    def embed(self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE) -> Result:
        """Batch-encode as queries. This is the interface NexusMatcher calls per schema."""
        try:
            arr = self.embed_queries(list(texts), batch_size)
        except Exception as exc:
            return Result.failure(f"Embedding failed: {type(exc).__name__}: {exc}")

        class _Batch:
            embeddings = arr
            count = len(arr)

        return Result.success(_Batch())

    def embed_single(self, text: str) -> Result:
        try:
            return Result.success(self.embed_queries([text])[0])
        except Exception as exc:
            return Result.failure(f"Embedding failed: {type(exc).__name__}: {exc}")


def default_embedding_provider(prefer: str = "auto") -> Any:
    """
    Pick an embedding provider that will actually work here, in preference order.

    Order: bundled ONNX -> sentence-transformers -> static. The bundled encoder is first
    because it is the only option that cannot fail at run time for environmental reasons.

    Args:
        prefer: "auto" (default), "bundled", "transformer", or "static" to force one.

    Returns:
        A ready provider.

    Raises:
        RuntimeError: if nothing usable is installed, listing what to install.
    """
    tried: list[str] = []

    def try_bundled():
        if not bundled_model_available():
            tried.append("bundled ONNX: weights not present in this install")
            return None
        try:
            import onnxruntime  # noqa: F401
            import tokenizers  # noqa: F401
        except ImportError as exc:
            tried.append(f"bundled ONNX: {exc}")
            return None
        return BundledOnnxProvider()

    def try_transformer():
        # Respect an explicit offline request rather than letting HF hang on a socket.
        if os.environ.get("HF_HUB_OFFLINE") == "1" and not os.environ.get("NEXUS_ST_MODEL_PATH"):
            tried.append("sentence-transformers: HF_HUB_OFFLINE=1 and no local model path")
            return None
        try:
            from nexus_matcher.infrastructure.adapters.embedding_providers.sentence_transformers import (
                SentenceTransformersProvider,
            )

            return SentenceTransformersProvider(
                os.environ.get("NEXUS_ST_MODEL_PATH", "BAAI/bge-small-en-v1.5")
            )
        except Exception as exc:
            tried.append(f"sentence-transformers: {type(exc).__name__}: {exc}")
            return None

    def try_static():
        try:
            from model2vec import StaticModel  # noqa: F401
        except ImportError as exc:
            tried.append(f"static: {exc}")
            return None
        from nexus_matcher.infrastructure.adapters.embedding_providers.static_embedding import (
            StaticEmbeddingProvider,
        )

        return StaticEmbeddingProvider()

    order = {
        "auto": (try_bundled, try_transformer, try_static),
        "bundled": (try_bundled,),
        "transformer": (try_transformer,),
        "static": (try_static,),
    }.get(prefer)
    if order is None:
        raise ValueError(f"prefer must be auto/bundled/transformer/static, got {prefer!r}")

    for factory in order:
        provider = factory()
        if provider is not None:
            return provider

    raise RuntimeError(
        "No usable embedding provider. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\n\nFix with ONE of:\n"
        "  pip install nexus-matcher            # bundled encoder, no download (recommended)\n"
        "  pip install nexus-matcher[embeddings]  # torch + sentence-transformers\n"
        "  pip install model2vec                # static fallback, lower accuracy"
    )
