"""Train a linear projection from a browser-side text encoder to PCA-reduced Qwen space.

Generates diverse text descriptions, encodes them with both Qwen3-VL-Embedding
(on GPU via vLLM) and a small text encoder (bge-small-en-v1.5 on CPU), then trains
a linear map bridging the two embedding spaces after PCA reduction.

Artifacts produced:
  - W_proj.bin       — Projection matrix (pca_dim × bge_dim, float32)
  - projection_summary.json — Training metrics (MSE, Spearman correlation)
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text generation templates
# ---------------------------------------------------------------------------

_SCENE_TYPES = [
    "street", "intersection", "sidewalk", "crosswalk", "plaza",
    "park entrance", "parking lot", "alleyway", "bridge", "tunnel",
    "highway overpass", "bus stop", "subway entrance", "bike lane",
    "residential block", "commercial district", "construction site",
]

_OBJECTS = [
    "scaffolding", "fire hydrant", "traffic light", "stop sign",
    "street lamp", "bench", "mailbox", "trash can", "dumpster",
    "parked car", "bicycle", "scooter", "bus", "taxi", "truck",
    "tree", "flower bed", "pothole", "manhole cover", "bollard",
    "fence", "railing", "awning", "fire escape", "antenna",
    "security camera", "graffiti", "mural", "advertisement",
    "newspaper stand", "food cart", "parking meter",
]

_CONDITIONS = [
    "sunny", "overcast", "rainy", "snowy", "foggy",
    "at dawn", "at dusk", "at night", "in summer", "in winter",
    "crowded", "empty", "busy", "quiet",
]

_ACTIONS = [
    "pedestrians crossing", "people walking", "cars driving",
    "cyclists riding", "workers constructing", "children playing",
    "delivery truck unloading", "street vendor selling",
    "dog walker with dogs", "jogger running",
]

_DESCRIPTORS = [
    "narrow", "wide", "tree-lined", "well-lit", "poorly maintained",
    "newly paved", "historic", "modern", "dense", "open",
    "urban", "residential", "commercial", "industrial",
]

_QUERY_TEMPLATES = [
    "a photo of a {scene} with {object}",
    "a {descriptor} {scene} {condition}",
    "{scene} with {object} and {action}",
    "a {condition} {scene} showing {object}",
    "{action} on a {descriptor} {scene}",
    "an urban {scene} featuring {object}",
    "{object} near a {scene}",
    "a {descriptor} {scene} with {action}",
    "{object}",
    "{scene} {condition}",
    "image of {object} in {scene}",
    "{action} near {object}",
    "a {condition} day on a {descriptor} {scene}",
    "{descriptor} {scene} with multiple {object}",
    "close-up of {object}",
    "wide view of a {scene} with {object}",
]


def _generate_training_texts(n: int = 50_000, seed: int = 42) -> list[str]:
    """Generate diverse urban scene text descriptions."""
    print(f"[train_query_projection] Generating {n} training texts (seed={seed})...", flush=True)
    rng = random.Random(seed)
    texts: set[str] = set()
    max_attempts = n * 20  # guard against infinite loop if template space < n
    attempts = 0
    prev_len = 0
    while len(texts) < n and attempts < max_attempts:
        template = rng.choice(_QUERY_TEMPLATES)
        text = template.format(
            scene=rng.choice(_SCENE_TYPES),
            object=rng.choice(_OBJECTS),
            condition=rng.choice(_CONDITIONS),
            action=rng.choice(_ACTIONS),
            descriptor=rng.choice(_DESCRIPTORS),
        )
        texts.add(text)
        attempts += 1
        # Early exit: if no new texts in 10K attempts, we've exhausted the space
        if attempts % 10_000 == 0:
            if len(texts) == prev_len:
                print(
                    f"[train_query_projection] Template space exhausted at {len(texts)} unique texts "
                    f"(requested {n}), proceeding with available texts",
                    flush=True,
                )
                break
            prev_len = len(texts)
    result = list(texts)[:n]
    print(f"[train_query_projection] Generated {len(result)} unique texts", flush=True)
    return result


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode_with_qwen(
    texts: list[str],
    model_path: str,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Encode texts with Qwen3-VL-Embedding via vLLM and apply PCA.

    Returns (N, pca_dim) float32 matrix.
    """
    from vllm import LLM

    print(f"[train_query_projection] Loading Qwen model: {model_path}", flush=True)
    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=512,
    )

    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[train_query_projection] Encoding {len(texts)} texts with Qwen ({n_batches} batches of {batch_size})...", flush=True)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i : i + batch_size]
        outputs = llm.embed(batch)
        batch_embs = np.array(
            [o.outputs.embedding for o in outputs], dtype=np.float32
        )
        all_embeddings.append(batch_embs)
        print(
            f"[train_query_projection]   Qwen batch {batch_num}/{n_batches}: {min(i + batch_size, len(texts))}/{len(texts)} texts",
            flush=True,
        )

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[train_query_projection] Qwen embeddings: {embeddings.shape}", flush=True)

    # Apply PCA
    reduced = (embeddings - pca_mean) @ pca_components.T
    print(f"[train_query_projection] PCA-reduced: {reduced.shape}", flush=True)

    # Free GPU
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return reduced


def _encode_with_bge(
    texts: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts with bge-small on CPU. Returns (N, 384) float32."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[train_query_projection] Loading bge-small: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[train_query_projection] Encoding {len(texts)} texts with bge-small ({n_batches} batches of {batch_size})...", flush=True)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            outputs = model(**encoded)
            embs = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
            all_embeddings.append(embs)

            print(
                f"[train_query_projection]   bge batch {batch_num}/{n_batches}: {min(i + batch_size, len(texts))}/{len(texts)} texts",
                flush=True,
                )

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[train_query_projection] bge-small embeddings: {embeddings.shape}", flush=True)
    return embeddings


def _train_projection(
    bge_embs: np.ndarray,
    qwen_pca_embs: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Train linear projection. Returns (W_proj, metrics)."""
    print(
        f"[train_query_projection] Training projection: "
        f"({bge_embs.shape[0]}, {bge_embs.shape[1]}) -> "
        f"({qwen_pca_embs.shape[0]}, {qwen_pca_embs.shape[1]})",
        flush=True,
    )

    # 90/10 train/val split
    n = len(bge_embs)
    n_train = int(n * 0.9)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, X_val = bge_embs[train_idx], bge_embs[val_idx]
    Y_train, Y_val = qwen_pca_embs[train_idx], qwen_pca_embs[val_idx]

    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X_train, Y_train)

    # Validation MSE
    Y_pred = reg.predict(X_val)
    mse = float(np.mean((Y_val - Y_pred) ** 2))
    print(f"[train_query_projection] Validation MSE: {mse:.6f}", flush=True)

    # Rank correlation check
    from scipy.stats import spearmanr

    n_check = min(100, len(val_idx))
    correlations = []
    for i in range(n_check):
        true_scores = qwen_pca_embs @ qwen_pca_embs[val_idx[i]]
        proj_query = reg.predict(bge_embs[val_idx[i] : val_idx[i] + 1])[0]
        proj_scores = qwen_pca_embs @ proj_query
        corr, _ = spearmanr(true_scores, proj_scores)
        correlations.append(corr)

    mean_corr = float(np.mean(correlations))
    print(f"[train_query_projection] Mean Spearman rank correlation: {mean_corr:.4f}", flush=True)

    W_proj = reg.coef_.astype(np.float32)  # (pca_dim, bge_dim)
    metrics = {"validation_mse": mse, "spearman_correlation": mean_corr}
    return W_proj, metrics


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def run_train_query_projection_stage(cfg: DictConfig) -> str:
    """Train query projection from bge-small to PCA-reduced Qwen space.

    Args:
        cfg: Hydra config with ``query_projection`` and ``runtime`` sections.
             Requires PCA artifacts from a prior build_browser_index stage.

    Returns:
        Absolute path to the output directory containing W_proj.bin.
    """
    qp = cfg.query_projection

    # Resolve PCA artifacts path (from chained pipeline or config)
    pca_dir = str(getattr(qp, "pca_input_path", "") or "")
    if not pca_dir:
        raise ValueError(
            "query_projection.pca_input_path must be set — "
            "either via pipeline chaining or CLI override"
        )
    pca_dir = os.path.abspath(pca_dir)

    # Resolve output path
    output_path = str(getattr(cfg.runtime, "output_path", "") or "")
    if not output_path:
        output_path = "outputs/query_projection"
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Config
    n_texts = int(getattr(qp, "n_texts", 30_000))
    pca_dim = int(getattr(qp, "pca_dim", 256))
    model_path = str(getattr(qp, "qwen_model_path", "") or str(cfg.model.model_source))
    bge_model = str(getattr(qp, "bge_model", "BAAI/bge-small-en-v1.5"))
    alpha = float(getattr(qp, "alpha", 1.0))
    seed = int(getattr(qp, "seed", 42))

    print(
        f"[train_query_projection] pca_dir={pca_dir}, output={output_path}, "
        f"n_texts={n_texts}, pca_dim={pca_dim}, model={model_path}",
        flush=True,
    )

    # Load PCA artifacts
    pca_components_raw = np.frombuffer(
        Path(pca_dir, "pca_components.bin").read_bytes(), dtype=np.float32
    )
    pca_components = pca_components_raw.reshape(pca_dim, -1)
    pca_mean = np.frombuffer(
        Path(pca_dir, "pca_mean.bin").read_bytes(), dtype=np.float32
    )
    print(
        f"[train_query_projection] PCA components: {pca_components.shape}, "
        f"mean: {pca_mean.shape}",
        flush=True,
    )

    # Generate training texts
    texts = _generate_training_texts(n_texts, seed=seed)
    print(f"[train_query_projection] Generated {len(texts)} training texts", flush=True)

    # Encode with Qwen (GPU) + PCA reduce
    qwen_pca_embs = _encode_with_qwen(texts, model_path, pca_components, pca_mean)

    # Encode with bge-small (CPU)
    bge_embs = _encode_with_bge(texts, bge_model)

    # Train projection
    W_proj, metrics = _train_projection(bge_embs, qwen_pca_embs, alpha=alpha)

    # Export
    proj_path = os.path.join(output_path, "W_proj.bin")
    with open(proj_path, "wb") as f:
        f.write(W_proj.tobytes())
    print(
        f"[train_query_projection] Wrote {proj_path} "
        f"({os.path.getsize(proj_path):,} bytes, shape {W_proj.shape})",
        flush=True,
    )

    # Summary
    summary = {
        "w_proj_shape": list(W_proj.shape),
        "n_training_texts": n_texts,
        "pca_dim": pca_dim,
        "bge_model": bge_model,
        "qwen_model": model_path,
        **metrics,
    }
    summary_path = os.path.join(output_path, "projection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train_query_projection] Done. Metrics: {metrics}", flush=True)
    return output_path
