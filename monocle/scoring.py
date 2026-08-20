"""Score per-patch token distributions against the image-global distribution.

score(patch, t) = alpha * log p_patch(t) + [log p_patch(t) - log p_global(t)]

The bracketed term is the PMI-style "divide out the global distribution" from
Henderson's visualization: it surfaces what a patch predicts that the rest of
the image doesn't. The alpha * log p_patch term keeps rare-token noise from
dominating — a patch must actually predict a token, not merely predict it
*relatively* more than its neighbors. alpha=0 is pure PMI; alpha->inf is raw
probability.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

DEFAULT_K = 3
DEFAULT_ALPHA = 0.3


def pooled_dims(n_rows: int, n_cols: int, pool: int) -> tuple[int, int]:
    """Grid dims after pool x pool aggregation (ceil division)."""
    return -(-n_rows // pool), -(-n_cols // pool)


def pool_probs(
    p: torch.Tensor, n_rows: int, n_cols: int, pool: int,
) -> torch.Tensor:
    """Average [n_patches, vocab] probabilities over pool x pool blocks of the
    row-major patch grid -> [pooled_patches, vocab]. Edge blocks may be
    smaller; they average over their actual members."""
    if pool <= 1:
        return p
    p3 = p.view(n_rows, n_cols, -1)
    blocks = []
    for r0 in range(0, n_rows, pool):
        for c0 in range(0, n_cols, pool):
            blocks.append(p3[r0:r0 + pool, c0:c0 + pool].mean(dim=(0, 1)))
    return torch.stack(blocks)


def display_form(token: str) -> str:
    """Sentencepiece token -> human-readable form ('▁dog' -> 'dog')."""
    return token.replace("▁", " ").strip()


def build_token_mask(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Boolean [vocab] mask of tokens allowed in word clouds.

    Drops: special tokens, byte-fallback / control tokens (<...>), tokens
    whose display form has no alphabetic character or fewer than 2 chars.
    Built once per tokenizer (~seconds over the 262k vocab); cache the result.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(tokenizer.all_special_ids)
    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    for i, tok in enumerate(tokens):
        if i in special or tok is None:
            continue
        if tok.startswith("<") and tok.endswith(">"):
            continue
        disp = display_form(tok)
        if len(disp) < 2:
            continue
        if not any(unicodedata.category(c).startswith("L") for c in disp):
            continue
        mask[i] = True
    return mask


def score_patches(
    logits: torch.Tensor,
    tokenizer: Any,
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    token_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
    pool: int = 1,
    grid_shape: Optional[tuple[int, int]] = None,
) -> pd.DataFrame:
    """[n_patches, vocab] logits -> long DataFrame of per-patch top-k tokens.

    Columns: patch_idx (row-major), rank, token (display form), score,
    p_patch, p_global. Case/piece variants ('▁Dog' vs 'dog') are deduped
    within a patch before truncating to k.

    pool > 1 averages probabilities over pool x pool blocks of the model's
    patch grid before scoring (requires grid_shape=(n_rows, n_cols)); the
    global distribution and patch_idx then refer to the pooled grid — use
    pooled_dims() for attach_grid().
    """
    p = torch.softmax(logits, dim=-1)
    if pool > 1:
        if grid_shape is None:
            raise ValueError("pool > 1 requires grid_shape=(n_rows, n_cols)")
        p = pool_probs(p, grid_shape[0], grid_shape[1], pool)
    log_p = (p + eps).log()
    log_g = (p.mean(dim=0) + eps).log()
    score = alpha * log_p + (log_p - log_g)
    if token_mask is not None:
        score = score.masked_fill(~token_mask.to(score.device), float("-inf"))

    # over-fetch so post-dedupe we still have k survivors
    top = torch.topk(score, k=min(4 * k, score.shape[-1]), dim=-1)
    top_ids = top.indices.cpu()
    top_scores = top.values.cpu()
    p_cpu_rows = torch.gather(p, 1, top.indices).cpu()
    p_global = p.mean(dim=0).cpu()

    rows: list[dict] = []
    for patch_idx in range(p.shape[0]):
        seen: set[str] = set()
        rank = 0
        for j in range(top_ids.shape[1]):
            s = float(top_scores[patch_idx, j])
            if s == float("-inf"):
                break
            tid = int(top_ids[patch_idx, j])
            disp = display_form(tokenizer.convert_ids_to_tokens(tid))
            key = disp.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "patch_idx": patch_idx,
                "rank": rank,
                "token": disp,
                "token_id": tid,
                "score": s,
                "p_patch": float(p_cpu_rows[patch_idx, j]),
                "p_global": float(p_global[tid]),
            })
            rank += 1
            if rank >= k:
                break
    return pd.DataFrame(rows)


def attach_grid(df: pd.DataFrame, n_rows: int, n_cols: int) -> pd.DataFrame:
    """Add patch_row / patch_col assuming row-major patch order
    (validated by monocle.validate before anything trusts this)."""
    df = df.copy()
    df["patch_row"] = df["patch_idx"] // n_cols
    df["patch_col"] = df["patch_idx"] % n_cols
    return df


def save_outputs(
    df: pd.DataFrame,
    meta: dict,
    out_dir: str | Path,
    image_id: str,
) -> tuple[Path, Path]:
    """Write <image_id>.parquet + <image_id>.meta.json; returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq = out / f"{image_id}.parquet"
    mj = out / f"{image_id}.meta.json"
    df.assign(image_id=image_id).to_parquet(pq, index=False)
    mj.write_text(json.dumps({"image_id": image_id, **meta}, indent=2))
    return pq, mj
