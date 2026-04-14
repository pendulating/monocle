"""Batch image embedding using vLLM's LLM.embed() API.

Uses vLLM with ``runner="pooling"`` for dense embedding extraction.
Multi-GPU scaling uses vLLM's native ``data_parallel_size`` — vLLM handles
GPU slicing, process groups, and load balancing internally.

Streaming output: when runtime.streaming_io is True, results are written
to disk incrementally every batch — if the job dies at batch 500/1000,
you still have 500 batches of embeddings on disk.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.linalg as la
import pandas as pd
from omegaconf import DictConfig
from PIL import Image

from dagspaces.common.vllm_inference import (
    _build_engine_kwargs,
    _flatten_messages_for_template,
    _shutdown_llm,
    get_pcie_nccl_env_vars,
    get_vllm_runtime_env_vars,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocess / postprocess factories
# ---------------------------------------------------------------------------

def _make_preprocess(cfg: DictConfig) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a preprocess function that builds chat messages for embedding."""
    instruction = str(getattr(cfg.embedding, "instruction", "")).strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction = instruction + "."

    min_pixels = int(getattr(cfg.embedding, "min_pixels", 784))
    max_pixels = int(getattr(cfg.embedding, "max_pixels", 262144))
    image_col = "image_path"
    if hasattr(cfg.data, "columns"):
        image_col = str(getattr(cfg.data.columns, "image_path", "image_path"))

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        image_path = str(row.get(image_col, ""))
        if not image_path:
            raise ValueError(f"Missing image path in column '{image_col}'")

        image_ref = image_path if image_path.startswith(("http", "oss")) else "file://" + image_path

        row["messages"] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_ref,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                ],
            },
        ]
        return row

    return preprocess


def _make_postprocess(cfg: DictConfig) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a postprocess function that normalizes and truncates embeddings."""
    normalize = bool(getattr(cfg.embedding, "normalize", True))
    output_dim = getattr(cfg.embedding, "output_dim", None)
    if output_dim is not None:
        output_dim = int(output_dim)
    model_source = str(cfg.model.model_source)

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        emb = row.get("embedding")
        if emb is None:
            row["embedding"] = None
            row["embedding_dim"] = 0
            row["model_source"] = model_source
            return row

        if not isinstance(emb, np.ndarray):
            emb = np.array(emb, dtype=np.float32)

        if output_dim is not None:
            emb = emb[:output_dim]

        if normalize:
            norm = la.norm(emb)
            if norm > 0:
                emb = emb / norm

        row["embedding"] = emb
        row["embedding_dim"] = len(emb)
        row["model_source"] = model_source
        return row

    return postprocess


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_embed_stage(cfg: DictConfig) -> str:
    """Run the embedding stage using vLLM with streaming output.

    When streaming_io is True, results are flushed to parquet parts
    every batch — providing incremental checkpoint-like behavior.

    Args:
        cfg: Hydra config with data, model, embedding, and runtime sections.

    Returns:
        Path to the output parquet file/directory.
    """
    # Load input data
    parquet_path = str(cfg.data.parquet_path)
    print(f"[run_embed_stage] Reading parquet: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)

    # Debug sampling
    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if sample_n:
        n = int(sample_n)
        df = df.head(n)
        print(f"[run_embed_stage] Limited to {n} rows for debug", flush=True)

    streaming_io = bool(getattr(getattr(cfg, "runtime", {}), "streaming_io", False))
    total_rows = len(df)
    print(f"[run_embed_stage] {total_rows} rows to embed (streaming={streaming_io})",
          flush=True)

    # Determine output path
    output_path = str(getattr(cfg.runtime, "output_path", None) or "")
    if not output_path:
        output_path = os.path.join("outputs", "embed", "embeddings.parquet")
    output_path = os.path.abspath(output_path)

    if streaming_io:
        output_dir = output_path.replace(".parquet", "")
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    preprocess = _make_preprocess(cfg)
    postprocess = _make_postprocess(cfg)

    # ── Initialize vLLM engine ────────────────────────────────────────────
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    env_snapshot = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE", "<unset>"),
    }
    print(f"[run_embed_stage] Runtime env: {env_snapshot}", flush=True)

    from vllm import LLM

    engine_kwargs = _build_engine_kwargs(cfg)
    engine_kwargs["runner"] = "pooling"
    # Pop data_parallel_size — vLLM 0.19 workers get DP config from env vars
    dp_size = int(engine_kwargs.pop("data_parallel_size", 1) or 1)

    print(f"[run_embed_stage] Model: {engine_kwargs.get('model')}", flush=True)
    print(f"[run_embed_stage] Engine kwargs: "
          f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }", flush=True)
    if dp_size > 1:
        print(f"[run_embed_stage] Data parallelism: {dp_size} replicas "
              f"x TP={engine_kwargs.get('tensor_parallel_size', 1)}", flush=True)

    # For DP mode, load tokenizer standalone; for single-process, create LLM.
    if dp_size > 1:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            engine_kwargs["model"], trust_remote_code=True
        )
        llm = None
    else:
        llm = LLM(**engine_kwargs)
        tokenizer = llm.llm_engine.tokenizer

    chat_template_kwargs = dict(
        getattr(cfg.model, "chat_template_kwargs", {}) or {}
    )

    batch_size = int(getattr(cfg.model, "batch_size", 0) or 0)
    if batch_size <= 0:
        batch_size = 16

    # ── Preprocess all rows (lightweight — no image loading) ──────────────
    print(f"[run_embed_stage] Preprocessing {total_rows} rows...", flush=True)
    rows = df.to_dict("records")
    preprocessed: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            preprocessed.append(preprocess(row))
        except Exception as e:
            row["__preprocess_error__"] = str(e)
            preprocessed.append(row)

    # Extract prompt texts and image refs (no PIL loading yet)
    prompt_texts: List[str] = []
    image_refs: List[Optional[str]] = []
    valid_mask: List[bool] = []

    for row in preprocessed:
        if "__preprocess_error__" in row:
            prompt_texts.append("")
            image_refs.append(None)
            valid_mask.append(False)
            continue

        messages = row.get("messages", [])
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except Exception:
            try:
                flat = _flatten_messages_for_template(messages)
                prompt_text = tokenizer.apply_chat_template(
                    flat, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                parts = []
                for msg in messages:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    parts.append(content)
                prompt_text = "\n".join(parts)

        img_ref = None
        for msg in messages:
            for item in (msg.get("content", []) if isinstance(msg.get("content"), list) else []):
                if isinstance(item, dict) and item.get("type") == "image":
                    ref = item.get("image")
                    if isinstance(ref, str):
                        img_ref = ref.removeprefix("file://")

        prompt_texts.append(prompt_text)
        image_refs.append(img_ref)
        valid_mask.append(True)

    valid_count = sum(valid_mask)
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    print(f"[run_embed_stage] {valid_count} valid rows (batch_size={batch_size})",
          flush=True)

    # ── Embed: data-parallel or single-process ───────────────────────────
    checkpoint_interval = int(getattr(getattr(cfg, "embedding", {}), "checkpoint_interval", 50_000) or 50_000)
    dp_errors: List[str] = []

    try:
        if dp_size > 1:
            # DP path: dispatch to subprocess workers, collect embeddings.
            # The DP helper returns a tuple — on partial failure some
            # positions may be None; we still stream what we have to disk
            # and raise AFTER the final flush so the caller can recover.
            from dagspaces.common.vllm_inference import _run_data_parallel_embed

            valid_prompt_texts = [prompt_texts[i] for i in valid_indices]
            valid_image_refs = [image_refs[i] for i in valid_indices]

            print(f"[run_embed_stage] Running data-parallel embedding: "
                  f"{len(valid_prompt_texts)} inputs across {dp_size} replicas...",
                  flush=True)
            all_embeddings, dp_errors = _run_data_parallel_embed(
                engine_kwargs=engine_kwargs,
                dp_size=dp_size,
                prompt_texts=valid_prompt_texts,
                image_refs=valid_image_refs,
                stage_name="embed",
                batch_size=batch_size,
            )
            if dp_errors:
                missing = sum(1 for e in all_embeddings if e is None)
                print(f"[run_embed_stage] DP embed partial failure: "
                      f"{missing}/{len(all_embeddings)} positions missing; "
                      f"streaming recovered rows before re-raising.", flush=True)

            # Build results: merge embeddings with preprocessed rows
            pending_results: List[Dict[str, Any]] = []
            part_idx = 0
            rows_written = 0
            emb_iter = iter(all_embeddings)
            for i, row in enumerate(preprocessed):
                row = row.copy()
                row.pop("messages", None)
                if valid_mask[i]:
                    row["embedding"] = next(emb_iter)
                else:
                    row["embedding"] = None
                row = postprocess(row)
                pending_results.append(row)

                # Stream to disk incrementally (same as single-process path)
                if streaming_io and len(pending_results) >= checkpoint_interval:
                    chunk_df = pd.DataFrame(pending_results)
                    part_path = os.path.join(output_dir, f"part-{part_idx:05d}.parquet")
                    chunk_df.to_parquet(part_path, index=False)
                    rows_written += len(chunk_df)
                    print(f"[run_embed_stage] Checkpoint {part_idx}: "
                          f"{len(chunk_df)} rows → {part_path} "
                          f"({rows_written}/{total_rows} total)", flush=True)
                    part_idx += 1
                    del chunk_df
                    pending_results = []

        else:
            # Single-process path
            total_batches = (valid_count + batch_size - 1) // batch_size
            print(f"[run_embed_stage] {total_batches} batches", flush=True)

            pending_results: List[Dict[str, Any]] = []
            part_idx = 0
            rows_written = 0
            rows_embedded = 0

            for batch_start in range(0, len(valid_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_indices))
                batch_indices = valid_indices[batch_start:batch_end]
                batch_num = batch_start // batch_size + 1

                if batch_num % 50 == 1 or batch_num == total_batches:
                    print(f"[run_embed_stage] Batch {batch_num}/{total_batches}: "
                          f"{rows_embedded}/{valid_count} embedded, "
                          f"{rows_written} written to disk", flush=True)

                batch_inputs = []
                for i in batch_indices:
                    embed_input: Dict[str, Any] = {"prompt": prompt_texts[i]}
                    ref = image_refs[i]
                    if ref is not None:
                        embed_input["multi_modal_data"] = {
                            "image": Image.open(ref).convert("RGB")
                        }
                    batch_inputs.append(embed_input)

                batch_outputs = llm.embed(batch_inputs)
                del batch_inputs

                for j, output in enumerate(batch_outputs):
                    row = preprocessed[batch_indices[j]].copy()
                    row.pop("messages", None)

                    emb_data = output.outputs.embedding
                    if not isinstance(emb_data, np.ndarray):
                        emb_data = np.array(emb_data, dtype=np.float32)
                    row["embedding"] = emb_data
                    row = postprocess(row)
                    pending_results.append(row)

                del batch_outputs
                rows_embedded += len(batch_indices)

                if streaming_io and len(pending_results) >= checkpoint_interval:
                    chunk_df = pd.DataFrame(pending_results)
                    part_path = os.path.join(output_dir, f"part-{part_idx:05d}.parquet")
                    chunk_df.to_parquet(part_path, index=False)
                    rows_written += len(chunk_df)
                    print(f"[run_embed_stage] Checkpoint {part_idx}: "
                          f"{len(chunk_df)} rows → {part_path} "
                          f"({rows_written}/{valid_count} total)", flush=True)
                    part_idx += 1
                    del chunk_df
                    pending_results = []

            # Handle failed rows (single-process only — DP path handles inline)
            for i, row in enumerate(preprocessed):
                if not valid_mask[i]:
                    row = row.copy()
                    row.pop("messages", None)
                    row["embedding"] = None
                    row = postprocess(row)
                    pending_results.append(row)

        # ── Final flush ──────────────────────────────────────────────────
        if streaming_io:
            if pending_results:
                chunk_df = pd.DataFrame(pending_results)
                part_path = os.path.join(output_dir, f"part-{part_idx:05d}.parquet")
                chunk_df.to_parquet(part_path, index=False)
                rows_written += len(chunk_df)
                part_idx += 1

            print(f"[run_embed_stage] Done: {rows_written} rows "
                  f"in {part_idx} parts → {output_dir}", flush=True)
            result_location = output_dir
        else:
            result_df = pd.DataFrame(pending_results)
            result_df.to_parquet(output_path, index=False)
            print(f"[run_embed_stage] Wrote {len(result_df)} rows → {output_path}",
                  flush=True)
            result_location = output_path

        # Raise AFTER persistence so partial results are always on disk.
        if dp_errors:
            raise RuntimeError(
                f"[embed] Data-parallel embedding had errors "
                f"(partial results persisted to {result_location}):\n"
                + "\n".join(dp_errors)
            )
        return result_location

    finally:
        _shutdown_llm(llm, stage_name="embed")
