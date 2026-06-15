#!/usr/bin/env python3
"""Train a linear projection from bge-small-en-v1.5 to PCA-reduced Qwen space.

Generates diverse text descriptions, encodes them with both Qwen3-VL-Embedding-8B
(on GPU via vLLM) and bge-small-en-v1.5 (on CPU), then trains a linear map that
bridges the two embedding spaces after PCA reduction.

Requirements: numpy, torch, transformers, vllm, scikit-learn

Usage:
    python scripts/train_query_projection.py \
        --pca-dir viz/embedding_search/public/data \
        --output-dir viz/embedding_search/public/data \
        --model-path Qwen/Qwen3-VL-Embedding-8B \
        --n-texts 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text generation templates for training data diversity
# ---------------------------------------------------------------------------

SCENE_TYPES = [
    "street", "intersection", "sidewalk", "crosswalk", "plaza",
    "park entrance", "parking lot", "alleyway", "bridge", "tunnel",
    "highway overpass", "bus stop", "subway entrance", "bike lane",
    "residential block", "commercial district", "construction site",
]

OBJECTS = [
    "scaffolding", "fire hydrant", "traffic light", "stop sign",
    "street lamp", "bench", "mailbox", "trash can", "dumpster",
    "parked car", "bicycle", "scooter", "bus", "taxi", "truck",
    "tree", "flower bed", "pothole", "manhole cover", "bollard",
    "fence", "railing", "awning", "fire escape", "antenna",
    "security camera", "graffiti", "mural", "advertisement",
    "newspaper stand", "food cart", "parking meter", "hydrant",
]

CONDITIONS = [
    "sunny", "overcast", "rainy", "snowy", "foggy",
    "at dawn", "at dusk", "at night", "in summer", "in winter",
    "crowded", "empty", "busy", "quiet",
]

ACTIONS = [
    "pedestrians crossing", "people walking", "cars driving",
    "cyclists riding", "workers constructing", "children playing",
    "delivery truck unloading", "street vendor selling",
    "dog walker with dogs", "jogger running",
]

DESCRIPTORS = [
    "narrow", "wide", "tree-lined", "well-lit", "poorly maintained",
    "newly paved", "historic", "modern", "dense", "open",
    "urban", "residential", "commercial", "industrial",
]

QUERY_TEMPLATES = [
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


def generate_training_texts(n: int = 50_000, seed: int = 42) -> list[str]:
    """Generate diverse urban scene text descriptions."""
    rng = random.Random(seed)
    texts = set()

    while len(texts) < n:
        template = rng.choice(QUERY_TEMPLATES)
        text = template.format(
            scene=rng.choice(SCENE_TYPES),
            object=rng.choice(OBJECTS),
            condition=rng.choice(CONDITIONS),
            action=rng.choice(ACTIONS),
            descriptor=rng.choice(DESCRIPTORS),
        )
        texts.add(text)

    return list(texts)[:n]


def encode_with_qwen(
    texts: list[str],
    model_path: str,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Encode texts with Qwen3-VL-Embedding via vLLM and apply PCA reduction.

    Returns (N, pca_dim) float32 matrix.
    """
    from vllm import LLM

    logger.info("Loading Qwen model: %s", model_path)
    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=512,
    )

    logger.info("Encoding %d texts with Qwen...", len(texts))
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = llm.embed(batch)
        batch_embs = np.array(
            [o.outputs.embedding for o in outputs], dtype=np.float32
        )
        all_embeddings.append(batch_embs)
        if (i // batch_size) % 10 == 0:
            logger.info("  Encoded %d / %d", min(i + batch_size, len(texts)), len(texts))

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 4096)
    logger.info("Qwen embeddings shape: %s", embeddings.shape)

    # Apply PCA reduction
    reduced = (embeddings - pca_mean) @ pca_components.T  # (N, pca_dim)
    logger.info("PCA-reduced shape: %s", reduced.shape)

    # Clean up GPU memory
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return reduced


def encode_with_bge(
    texts: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts with bge-small-en-v1.5 on CPU. Returns (N, 384) float32."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading bge-small model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    logger.info("Encoding %d texts with bge-small...", len(texts))

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            # CLS pooling
            embs = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)
            # L2 normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
            all_embeddings.append(embs)

            if (i // batch_size) % 20 == 0:
                logger.info("  Encoded %d / %d", min(i + batch_size, len(texts)), len(texts))

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("bge-small embeddings shape: %s", embeddings.shape)
    return embeddings


def train_projection(
    bge_embs: np.ndarray,
    qwen_pca_embs: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Train linear projection W such that W @ bge ≈ qwen_pca.

    Returns W_proj of shape (pca_dim, bge_dim).
    """
    logger.info(
        "Training projection: (%d, %d) -> (%d, %d)",
        bge_embs.shape[0], bge_embs.shape[1],
        qwen_pca_embs.shape[0], qwen_pca_embs.shape[1],
    )

    # Split train/val (90/10)
    n = len(bge_embs)
    n_train = int(n * 0.9)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, X_val = bge_embs[train_idx], bge_embs[val_idx]
    Y_train, Y_val = qwen_pca_embs[train_idx], qwen_pca_embs[val_idx]

    # Ridge regression (per-output dimension)
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X_train, Y_train)

    # Validation MSE
    Y_pred = reg.predict(X_val)
    mse = np.mean((Y_val - Y_pred) ** 2)
    logger.info("Validation MSE: %.6f", mse)

    # Rank correlation check on a few random "queries"
    from scipy.stats import spearmanr

    n_check = min(100, len(val_idx))
    correlations = []
    for i in range(n_check):
        # True ranking by Qwen PCA dot product
        true_scores = qwen_pca_embs @ qwen_pca_embs[val_idx[i]]
        # Projected ranking
        proj_query = reg.predict(bge_embs[val_idx[i] : val_idx[i] + 1])[0]
        proj_scores = qwen_pca_embs @ proj_query
        corr, _ = spearmanr(true_scores, proj_scores)
        correlations.append(corr)

    mean_corr = np.mean(correlations)
    logger.info("Mean Spearman rank correlation: %.4f", mean_corr)

    W_proj = reg.coef_.astype(np.float32)  # (pca_dim, bge_dim)
    logger.info("W_proj shape: %s", W_proj.shape)
    return W_proj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train query projection from bge-small to PCA-reduced Qwen space"
    )
    parser.add_argument(
        "--pca-dir",
        required=True,
        help="Directory containing pca_components.bin and pca_mean.bin",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for W_proj.bin",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-VL-Embedding-8B",
        help="Qwen embedding model path or HF identifier",
    )
    parser.add_argument(
        "--bge-model",
        default="BAAI/bge-small-en-v1.5",
        help="BGE model for browser-side text encoding",
    )
    parser.add_argument(
        "--n-texts",
        type=int,
        default=50_000,
        help="Number of training texts to generate (default: 50000)",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=256,
        help="PCA output dimensionality (must match build_browser_index.py)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regression regularization strength",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for text generation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pca_dir = Path(args.pca_dir)

    # Load PCA artifacts
    logger.info("Loading PCA artifacts from %s", pca_dir)
    components_raw = np.frombuffer(
        (pca_dir / "pca_components.bin").read_bytes(), dtype=np.float32
    )
    pca_components = components_raw.reshape(args.pca_dim, -1)  # (256, 4096)
    pca_mean = np.frombuffer(
        (pca_dir / "pca_mean.bin").read_bytes(), dtype=np.float32
    )
    logger.info(
        "PCA components: %s, mean: %s", pca_components.shape, pca_mean.shape
    )

    # Generate training texts
    texts = generate_training_texts(args.n_texts, seed=args.seed)
    logger.info("Generated %d training texts", len(texts))

    # Encode with Qwen (GPU) + PCA reduce
    qwen_pca_embs = encode_with_qwen(
        texts, args.model_path, pca_components, pca_mean
    )

    # Encode with bge-small (CPU)
    bge_embs = encode_with_bge(texts, args.bge_model)

    # Train projection
    W_proj = train_projection(bge_embs, qwen_pca_embs, alpha=args.alpha)

    # Export
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    proj_path = out / "W_proj.bin"
    with open(proj_path, "wb") as f:
        f.write(W_proj.tobytes())
    logger.info("Wrote %s (%d bytes)", proj_path, proj_path.stat().st_size)

    logger.info("Done.")


if __name__ == "__main__":
    main()
