"""Two-phase retrieval: embedding recall + cross-encoder reranking.

Phase 1 — Approximate retrieval via cosine similarity (numpy dot product)
on pre-computed L2-normalized embeddings from the embed stage.

Phase 2 — Cross-encoder reranking using Qwen3-VL-Reranker via vLLM's
``LLM.score()`` API.  Each (query, document-image) pair is scored by the
reranker and results are sorted by relevance.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.linalg as la
import pandas as pd
from omegaconf import DictConfig

from dagspaces.common.vllm_inference import (
    _build_engine_kwargs,
    _shutdown_llm,
    get_pcie_nccl_env_vars,
    get_vllm_runtime_env_vars,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_embedding_parquet(path: str) -> pd.DataFrame:
    """Load embeddings from a single parquet file or a directory of parts."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        parts = sorted(Path(path).glob("part-*.parquet"))
        if not parts:
            parts = sorted(Path(path).glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet files found in {path}")
        print(f"[rerank] Loading {len(parts)} parquet parts from {path}", flush=True)
        dfs = [pd.read_parquet(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
    else:
        print(f"[rerank] Loading parquet from {path}", flush=True)
        df = pd.read_parquet(path)

    if "embedding" not in df.columns:
        raise ValueError(f"Parquet at {path} has no 'embedding' column")
    return df


def _get_query_embedding(
    cfg: DictConfig,
    tokenizer: Any = None,
) -> np.ndarray:
    """Obtain the query embedding vector.

    Priority:
      1. Pre-computed .npy file at ``cfg.reranking.query_embedding_path``
      2. Compute inline from ``cfg.reranking.query_text`` using the
         embedding model (lightweight single-vector forward pass).
    """
    npy_path = getattr(cfg.reranking, "query_embedding_path", None)
    if npy_path:
        npy_path = str(npy_path)
        print(f"[rerank] Loading query embedding from {npy_path}", flush=True)
        emb = np.load(npy_path).astype(np.float32)
        norm = la.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # Compute inline using the embedding model
    query_text = str(cfg.reranking.query_text)
    if not query_text:
        raise ValueError("reranking.query_text is empty and no query_embedding_path provided")

    emb_model = str(
        getattr(cfg.reranking, "embedding_model_source", None)
        or "/share/pierson/matt/zoo/models/Qwen3-VL-Embedding-8B"
    )
    print(f"[rerank] Computing query embedding inline with {emb_model}", flush=True)

    from transformers import AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(emb_model, trust_remote_code=True)

    instruction = str(
        getattr(cfg.reranking, "instruction", "")
        or "Given a search query, retrieve relevant images that match the query description."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [{"type": "text", "text": query_text}]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    from vllm import LLM

    embed_kwargs = {
        "model": emb_model,
        "runner": "pooling",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
    }
    llm = LLM(**embed_kwargs)
    try:
        outputs = llm.embed([{"prompt": prompt_text}])
        emb = np.array(outputs[0].outputs.embedding, dtype=np.float32)
        norm = la.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    finally:
        _shutdown_llm(llm, stage_name="rerank_query_embed")


def _format_doc_param(image_path: str) -> Dict[str, Any]:
    """Build a ScoreMultiModalParam for a single document image."""
    if not image_path.startswith(("http", "oss")):
        image_url = "file://" + os.path.abspath(image_path)
    else:
        image_url = image_path
    return {
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        ]
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_rerank_stage(cfg: DictConfig) -> str:
    """Run two-phase retrieval: embedding recall + cross-encoder reranking.

    Args:
        cfg: Hydra config with reranking, model, and runtime sections.

    Returns:
        Path to the output parquet file with ranked results.
    """
    # ── Load pre-computed embeddings ─────────────────────────────────────
    embeddings_path = str(cfg.reranking.embeddings_input_path)
    if not embeddings_path or embeddings_path == "None":
        raise ValueError("reranking.embeddings_input_path must be set")
    df = _load_embedding_parquet(embeddings_path)

    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if sample_n:
        n = int(sample_n)
        df = df.head(n)
        print(f"[rerank] Debug: limited to {n} rows", flush=True)

    total_rows = len(df)
    print(f"[rerank] {total_rows} rows loaded", flush=True)

    # Build embedding matrix efficiently — avoid np.stack on 1M+ Python objects.
    # Convert the Series of arrays into a contiguous 2-D float32 matrix.
    print(f"[rerank] Building embedding matrix...", flush=True)
    emb_list = df["embedding"].tolist()
    embeddings = np.array(emb_list, dtype=np.float32)
    del emb_list
    if embeddings.ndim == 1:
        # Edge case: single row or ragged — fall back to stack
        embeddings = np.stack(df["embedding"].values).astype(np.float32)
    print(f"[rerank] Embedding matrix: {embeddings.shape}", flush=True)

    # ── Phase 1: Approximate retrieval ───────────────────────────────────
    top_k = int(cfg.reranking.top_k)
    print(f"[rerank] Phase 1: dot-product retrieval (top_k={top_k})", flush=True)

    query_embedding = _get_query_embedding(cfg)
    scores = embeddings @ query_embedding
    top_k_actual = min(top_k, len(scores))
    top_indices = np.argsort(-scores)[:top_k_actual]

    candidates_df = df.iloc[top_indices].copy()
    candidates_df["retrieval_score"] = scores[top_indices].astype(np.float32)
    print(f"[rerank] Phase 1 complete: {len(candidates_df)} candidates "
          f"(score range: {scores[top_indices[-1]]:.4f} – {scores[top_indices[0]]:.4f})",
          flush=True)

    # Free the full embedding matrix
    del embeddings, df

    # ── Phase 2: Cross-encoder reranking ─────────────────────────────────
    print(f"[rerank] Phase 2: cross-encoder reranking with vLLM", flush=True)
    rerank_scores = _run_cross_encoder_reranking(cfg, candidates_df)
    candidates_df["rerank_score"] = np.array(rerank_scores, dtype=np.float32)

    # Sort by rerank score descending
    candidates_df = candidates_df.sort_values("rerank_score", ascending=False)
    candidates_df["rerank_rank"] = range(1, len(candidates_df) + 1)

    # Optional score threshold filter
    score_threshold = getattr(cfg.reranking, "score_threshold", None)
    if score_threshold is not None:
        before = len(candidates_df)
        candidates_df = candidates_df[candidates_df["rerank_score"] >= float(score_threshold)]
        print(f"[rerank] Score threshold {score_threshold}: {before} → {len(candidates_df)} rows",
              flush=True)

    # Optional top-N limit
    rerank_top_n = getattr(cfg.reranking, "rerank_top_n", None)
    if rerank_top_n is not None:
        candidates_df = candidates_df.head(int(rerank_top_n))

    # Drop the heavy embedding column from output
    if "embedding" in candidates_df.columns:
        candidates_df = candidates_df.drop(columns=["embedding"])

    # ── Write output ─────────────────────────────────────────────────────
    output_path = str(getattr(cfg.runtime, "output_path", None) or "")
    if not output_path:
        output_path = os.path.join("outputs", "rerank", "reranked.parquet")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    candidates_df.to_parquet(output_path, index=False)
    print(f"[rerank] Wrote {len(candidates_df)} ranked results → {output_path}", flush=True)
    return output_path


# ---------------------------------------------------------------------------
# Cross-encoder reranking via vLLM score()
# ---------------------------------------------------------------------------

def _run_cross_encoder_reranking(
    cfg: DictConfig,
    candidates_df: pd.DataFrame,
) -> List[float]:
    """Score (query, document) pairs using the reranker model."""
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    from vllm import LLM

    engine_kwargs = _build_engine_kwargs(cfg)
    engine_kwargs["runner"] = "pooling"
    engine_kwargs.pop("data_parallel_size", None)

    # Allow loading local image files — the cross-encoder must see the raw
    # images to score (query, image) pairs (Phase 2 is not embedding-based).
    engine_kwargs["allowed_local_media_path"] = "/"

    print(f"[rerank] Loading reranker: {engine_kwargs.get('model')}", flush=True)
    llm = LLM(**engine_kwargs)

    # Load the score-specific template (NOT the model's chat_template.jinja).
    # vLLM's score() passes messages with role="query"/"document"/"system"
    # to the template; the standard chat template expects "user"/"assistant"
    # and would produce malformed input.
    score_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "conf", "score_template", "qwen3_vl_reranker.jinja",
    )
    score_template: Optional[str] = None
    if os.path.exists(score_template_path):
        with open(score_template_path, "r") as f:
            score_template = f.read()
        print(f"[rerank] Loaded score template from {score_template_path}", flush=True)
    else:
        print(f"[rerank] Warning: no score template at {score_template_path}", flush=True)

    query_text = str(cfg.reranking.query_text)
    image_col = "image_path"
    rows = candidates_df.to_dict("records")

    # Build document params for all candidates
    doc_params = [
        _format_doc_param(str(row.get(image_col, "")))
        for row in rows
    ]

    # vLLM score() maps data_1 → role="query" and data_2 → role="document"
    # in the score template.  The instruction is baked into the template's
    # default (no way to inject a "system" message through the score API).
    # Pass query_text as a plain string — vLLM wraps it as role="query".
    total = len(doc_params)
    print(f"[rerank] Scoring {total} (query, image) pairs — "
          f"vLLM handles internal batching via max_num_seqs={engine_kwargs.get('max_num_seqs', '?')}",
          flush=True)

    try:
        # Pass all candidates at once — vLLM's score() accepts 1→N mode
        # and handles internal scheduling/batching via max_num_seqs.
        # This mirrors how llm.embed() is called in the embed stage.
        outputs = llm.score(
            query_text,
            doc_params,
            chat_template=score_template,
        )
        all_scores = [o.outputs.score for o in outputs]
        print(f"[rerank] Scored {total}/{total} candidates "
              f"(score range: {min(all_scores):.4f} – {max(all_scores):.4f})",
              flush=True)
    finally:
        _shutdown_llm(llm, stage_name="rerank")

    return all_scores
