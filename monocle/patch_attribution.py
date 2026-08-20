"""Which image patches move the decision.

Step 3 of the pairwise lens study. `answer_tokens.py` reads the answer
position; this module goes back to the image patches and asks which ones fed
the answer.

    c          = logit(pos_label) - logit(neg_label)      at the answer position
    a_l[patch] = < dc/dh_l[patch] , h_l[patch] >          grad x activation

reported separately for the image A and image B patch blocks.

Readout is not contribution
---------------------------
Rung B part (b) read, per patch, what the mm lens says that patch is disposed
to SAY. A patch over litter whose readout holds litter words is suggestive, but
it does not show that the patch moved the answer. This module measures the
movement.

Local, not averaged
-------------------
`dc/dh_l[patch]` is a per-image quantity: the true routing for THIS pair, not
the corpus-averaged `J`. It is one column of the local Jacobian, in the one
direction the decision lives along, so it costs ONE backward pass — not the
3,840 that the full 3840x3840 matrix needs (417 s/image, measured).

The gradient runs through the whole network, so no linearization enters. `c`
comes from `lens_model.unembed(h_final[answer])`, which is the model's own head
(final norm + lm_head + logit softcap) and reproduces `out.logits` exactly at
that position — `jlens_read.final_layer_consistency` measured 0.00e+00. Taking
`c` this way avoids materializing logits at all 703 positions.

The contrast
------------
`More - Less` is the default. It is NOT right for every case: at L47 schools
reads `Not` at 0.52 and restaurants at 0.31, so on those the live decision is
abstain-versus-judge. `--contrast abstain` selects `NotSure - max(judgment)`.
Run both on a high-abstention case.

Depth
-----
Every fitted layer is recorded, not only L42-L47. The answer is not sayable
before L42, but a patch can still feed the decision earlier — those are
different claims, and restricting the layers would assume the answer.

Outputs, per case
-----------------
| File | One row per |
|---|---|
| `patch_attrib.parquet` | (pair, cond, layer, slot, patch) — the attribution |
| `summary.json` | per (cond, layer, slot) aggregates |
| `maps/` | PNG overlays, when `--n-maps` > 0 |
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import answer_tokens as at  # noqa: E402
from monocle import canonical, extract  # noqa: E402
from monocle import jlens_read  # noqa: E402  (jlens imported lazily inside)
from monocle import safety_workspace as sw  # noqa: E402

MODEL_DIR = "/share/pierson/matt/zoo/models/gemma-4-12B-it"
OUT_DEFAULT = REPO / "outputs/_monocle/patch_attribution"

FITTED_LAYERS = [6, 12, 18, 24, 30, 36, 42, 46]
FINAL_LAYER = 47
SEED = 777

CONTRASTS = ("judgment", "abstain")


def log(m: str) -> None:
    print(f"[patch-attrib] {m}", flush=True)


# ---------------------------------------------------------------------------
# The contrast (pure)
# ---------------------------------------------------------------------------
def contrast_token_ids(
    kind: str, label_first: dict[str, int],
) -> tuple[int, int, str, str]:
    """(positive id, negative id, positive name, negative name).

    `judgment` is More minus Less — the ordinal direction.
    `abstain` is NotSure minus More — abstain against the commonest judgment.

    Raises KeyError when the run's label set lacks a needed label, rather than
    silently falling back to a different contrast.
    """
    if kind == "judgment":
        return label_first["More"], label_first["Less"], "More", "Less"
    if kind == "abstain":
        if canonical.NOT_SURE not in label_first:
            raise KeyError(
                "the abstain contrast needs NotSure in the label set, but this "
                "run did not enable abstention")
        return (label_first[canonical.NOT_SURE], label_first["More"],
                canonical.NOT_SURE, "More")
    raise ValueError(f"unknown contrast {kind!r}; expected one of {CONTRASTS}")


def grad_x_activation(
    grad: torch.Tensor, act: torch.Tensor,
) -> torch.Tensor:
    """Row-wise dot product of a gradient and its activation.

    Both are [n_positions, d_model]; the result is [n_positions]. The sign is
    meaningful: a positive value means the patch pushes the contrast toward the
    positive label.
    """
    if grad.shape != act.shape:
        raise ValueError(f"shape mismatch: {tuple(grad.shape)} vs {tuple(act.shape)}")
    return (grad.float() * act.float()).sum(dim=-1)


def center_over_patches(act: torch.Tensor) -> torch.Tensor:
    """Subtract the mean patch residual from every patch residual.

    grad x activation splits into

        <g_p, h_p> = <g_p, h_bar> + <g_p, delta_p>

    where the first term is the shared component and the second the patch
    content. Centering keeps only the second, so the map answers "what did
    THIS patch contribute" rather than "how sensitive is the output to this
    position at all".

    Measured, not assumed: at PATCH positions on gemma-4-12b the shared
    component is small — ``shared_component_ratio`` is 1.0-5.5 across layers
    (job 289403) — so centering changes the map only a little. That is a
    genuine finding, and it differs from the answer position, where the
    activation-probe work measured a ratio near 150 across pairs. The massive
    activation belongs to particular positions, not to patch positions in
    general. Do not carry the 150x number across.

    Centering stays the default because it is the correct decomposition and
    costs nothing; ``attrib_raw`` keeps the uncentered value so the difference
    stays measurable.

    ``act`` is [n_patches, d_model] for ONE image block; the mean is taken
    within that block, so the two images do not contaminate each other.
    """
    if act.ndim != 2:
        raise ValueError(f"expected [n_patches, d_model], got {tuple(act.shape)}")
    return act - act.mean(dim=0, keepdim=True)


def shared_component_ratio(act: torch.Tensor) -> float:
    """``||mean_p h_p|| / mean_p ||h_p - mean||`` for one image block.

    The diagnostic behind :func:`center_over_patches`. A large value means the
    shared component dominates and an uncentered attribution is unreadable.
    """
    if act.ndim != 2:
        raise ValueError(f"expected [n_patches, d_model], got {tuple(act.shape)}")
    a = act.float()
    mean = a.mean(dim=0)
    spread = float((a - mean).norm(dim=-1).mean())
    if spread == 0.0:
        return float("inf")
    return float(mean.norm()) / spread


def pool_map(
    values: np.ndarray, n_rows: int, n_cols: int, pool: int,
) -> tuple[np.ndarray, int, int]:
    """Mean-pool a [n_rows*n_cols] map over pool x pool blocks.

    Returns (pooled values row-major, pooled rows, pooled cols). 16 does not
    divide by 3, so a pool of 3 gives five 3-wide bands and one 1-wide edge
    band. The MEAN (not the sum) keeps the thin edge cell comparable with the
    others; a sum would make it look artificially quiet.

    Pooling is a RENDER-time choice. The parquet keeps all 256 patches, so a
    different pool needs no new GPU run.
    """
    if pool <= 1:
        return values, n_rows, n_cols
    grid = values.reshape(n_rows, n_cols)
    out_r = -(-n_rows // pool)
    out_c = -(-n_cols // pool)
    out = np.zeros((out_r, out_c), dtype=float)
    for r in range(out_r):
        for c in range(out_c):
            out[r, c] = grid[r * pool:(r + 1) * pool,
                             c * pool:(c + 1) * pool].mean()
    return out.reshape(-1), out_r, out_c


def pool_tokens(
    tokens: dict[int, list[str]], probs: dict[int, list[float]],
    n_rows: int, n_cols: int, pool: int, k: int = 3,
) -> dict[int, list[str]]:
    """Merge the per-patch readout tokens of each pool block.

    Within a block the same token can appear at several patches; its weight is
    the sum of its probabilities there, and the k heaviest are kept. Thus a
    token that many patches of a region agree on outranks one that a single
    patch shouts.
    """
    if pool <= 1:
        return {i: t[:k] for i, t in tokens.items()}
    out_c = -(-n_cols // pool)
    merged: dict[int, dict[str, float]] = {}
    for idx, toks in tokens.items():
        r, c = divmod(idx, n_cols)
        block = (r // pool) * out_c + (c // pool)
        bucket = merged.setdefault(block, {})
        for t, pr in zip(toks, probs.get(idx, [0.0] * len(toks))):
            bucket[t] = bucket.get(t, 0.0) + float(pr)
    return {b: [t for t, _ in sorted(v.items(), key=lambda kv: -kv[1])[:k]]
            for b, v in merged.items()}


def row_profile(values: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Share of the total |attribution| held by each patch row.

    The positional diagnostic. The patch block is raster-ordered and the model
    is causal, so a patch in the LAST row is the only kind that has attended to
    every other patch: it is the aggregation bottleneck of the image. Gradient
    from the answer position flows through it whatever the picture holds.

    Measured on subway_safety (job 289469), share held by row 15 of 16 against
    a uniform 0.0625:

    | layer | prod | neutral |
    |---|---|---|
    | L6 | 0.10 | 0.10 |
    | L24 | 0.08 | 0.07 |
    | L42 | 0.26 | 0.30 |
    | L46 | 0.68 | 0.56 |

    Row 0 (sky) is equally uniform and equally low-texture but stays at 0.02,
    so this is causal position, not a register or low-information effect.
    """
    mag = np.abs(values).reshape(n_rows, n_cols).sum(axis=1)
    total = mag.sum()
    return mag / total if total else mag


def prod_minus_neutral(
    prod: np.ndarray, neutral: np.ndarray,
) -> np.ndarray:
    """The question-specific part of an attribution map.

    Both arms share the causal geometry above, so the difference cancels it.
    This is the map to read. At L46 on libraries the raw prod and neutral maps
    correlate at r = 0.98 — the undifferenced map at depth is almost entirely
    question-independent.
    """
    if prod.shape != neutral.shape:
        raise ValueError(f"shape mismatch: {prod.shape} vs {neutral.shape}")
    return prod - neutral


def map_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two attribution maps over the same patch grid.

    High r between the prod and neutral arms means the map is driven by
    position rather than by the question.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def normalize_map(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Scale a signed attribution map to [-1, 1], clipping at a percentile.

    Per map, not per corpus: the question a map answers is which patches of
    THIS pair mattered most, and a shared scale would let one extreme pair
    flatten every other.

    The scale is the given percentile of |value|, NOT the maximum, and the
    result is clipped to [-1, 1]. Attribution concentrates sharply with depth:
    at L46 the top 5% of patches hold 95% of the total (job 289403). Dividing
    by the maximum then drives every other cell to about 0, and the map reads
    as empty when it is in fact extremely peaked. Clipping lets the few
    dominant cells saturate and keeps the rest legible.

    The default is the 95th percentile, chosen for the POOLED grid. A pool of
    3 leaves 36 cells, where the 99th percentile falls between the top and the
    second cell and the clip does nothing; the 95th clips the top 2. On the
    full 256-patch grid the 95th clips the top 13.

    Pass ``percentile=100`` for max-scaled behaviour.
    """
    mag = np.abs(values)
    scale = float(np.percentile(mag, percentile)) if mag.size else 0.0
    if scale == 0.0:
        scale = float(mag.max())
    if scale == 0.0:
        return np.zeros_like(values)
    return np.clip(values / scale, -1.0, 1.0)


# ---------------------------------------------------------------------------
# The attribution pass (GPU)
# ---------------------------------------------------------------------------
def attribute_pair(
    model: Any, lens_model: Any, inputs: dict, layers: list[int],
    pos_id: int, neg_id: int, answer_pos: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]:
    """One forward and ONE backward. Returns (grads, activations, contrast).

    Both dicts are keyed by layer and hold the FULL sequence, so the caller
    slices whichever positions it wants. The graph is rooted at the lowest
    requested layer, so nothing below it is retained.
    """
    from jlens.hooks import ActivationRecorder

    record_at = sorted(set(layers) | {FINAL_LAYER})
    root = min(record_at)
    with ActivationRecorder(
            lens_model.layers, at=record_at, start_graph_at=root) as rec:
        # No inference_mode here — it would forbid the backward pass.
        out = model(**inputs, use_cache=False)
        del out
        acts = dict(rec.activations)

    h_final = acts[FINAL_LAYER][0, answer_pos, :]
    logits = lens_model.unembed(h_final)
    c = logits[pos_id] - logits[neg_id]
    if not torch.isfinite(c):
        raise RuntimeError("non-finite contrast at the answer position")

    # The final block is recorded because `c` is built from it, but it is not
    # an attribution target: `c` depends on h_final at the ANSWER position
    # only, so dc/dh_final at any patch position is identically zero. Asking
    # for it would fill the frame with zeros that look like a finding.
    wanted = [l for l in layers if l != FINAL_LAYER]
    if not wanted:
        raise ValueError("no attribution layers below the final block")
    tensors = [acts[l] for l in wanted]
    grads = torch.autograd.grad(c, tensors, retain_graph=False)
    return ({l: g.detach() for l, g in zip(wanted, grads)},
            {l: acts[l].detach() for l in wanted},
            float(c.detach()))


def patch_rows(
    grads: dict[int, torch.Tensor], acts: dict[int, torch.Tensor],
    blocks: list[torch.Tensor], grids: dict[str, tuple[int, int]],
    base: dict,
) -> tuple[list[dict], list[dict]]:
    """Long rows for both image blocks, plus the per-layer diagnostic.

    Emits BOTH attributions:

    - ``attrib_centered`` — the one to read. The activation is centered over
      the patch positions of its own image block first.
    - ``attrib_raw`` — uncentered, kept so the massive-activation effect stays
      visible and measurable rather than silently corrected.
    """
    rows: list[dict] = []
    diag: list[dict] = []
    for layer in sorted(grads):
        for slot, positions in zip(("A", "B"), blocks):
            g = grads[layer][0, positions, :]
            h = acts[layer][0, positions, :]
            raw = grad_x_activation(g, h).cpu().numpy()
            cen = grad_x_activation(g, center_over_patches(h)).cpu().numpy()
            ratio = shared_component_ratio(h)
            n_rows, n_cols = grids[slot]
            diag.append({
                **base, "layer": int(layer), "slot": slot,
                "shared_ratio": ratio,
                "mean_abs_raw": float(np.abs(raw).mean()),
                "mean_abs_centered": float(np.abs(cen).mean()),
            })
            for i in range(len(raw)):
                rows.append({
                    **base, "layer": int(layer), "slot": slot,
                    "patch_idx": i, "patch_row": i // n_cols,
                    "patch_col": i % n_cols,
                    "attrib": float(cen[i]),
                    "attrib_raw": float(raw[i])})
    return rows, diag


def readout_rows(
    lens: Any, lens_model: Any, acts: dict[int, torch.Tensor],
    blocks: list[torch.Tensor], grids: dict[str, tuple[int, int]],
    tokenizer: Any, base: dict, k: int = 3,
) -> list[dict]:
    """Per-patch lens readout: what each patch is disposed to SAY.

    This is monocle's original question, computed here so a map can carry both
    channels at once — the colour says how much a patch moved the decision,
    the words say what that patch would utter. They are different quantities
    and they need not agree.
    """
    rows: list[dict] = []
    final = lens_model.n_layers - 1
    for layer in sorted(acts):
        if layer not in lens.jacobians or layer == final:
            continue
        for slot, positions in zip(("A", "B"), blocks):
            h = acts[layer][0, positions, :].float()
            logits = lens_model.unembed(lens.transport(h, layer)).float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k, dim=-1)
            n_rows, n_cols = grids[slot]
            ids = top.indices.cpu().tolist()
            vals = top.values.cpu().tolist()
            for i, (row_ids, row_vals) in enumerate(zip(ids, vals)):
                toks = tokenizer.convert_ids_to_tokens(row_ids)
                for rank, (tid, tok, pr) in enumerate(
                        zip(row_ids, toks, row_vals)):
                    rows.append({
                        **base, "layer": int(layer), "slot": slot,
                        "patch_idx": i, "patch_row": i // n_cols,
                        "patch_col": i % n_cols, "rank": rank,
                        "token_id": int(tid),
                        "token": scoring_display(tok),
                        "prob": float(pr)})
    return rows


def scoring_display(tok: Optional[str]) -> str:
    from monocle import scoring
    return scoring.display_form(tok or "")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _div_color(t: float) -> tuple[int, int, int]:
    """Diverging teal-white-coral ramp for t in [-1, 1] (the CVPR ramp)."""
    t = float(np.clip(t, -1.0, 1.0))
    if t >= 0:
        return (255, int(255 - 145 * t), int(255 - 165 * t))    # -> coral
    u = -t
    return (int(255 - 205 * u), int(255 - 90 * u), int(255 - 60 * u))  # -> teal


def render_attrib(
    image: Image.Image, values: np.ndarray, n_rows: int, n_cols: int,
    alpha: float = 0.55, pool: int = 1,
    tokens: Optional[dict[int, list[str]]] = None,
    upscale: int = 2, grid_lines: bool = True, percentile: float = 95.0,
) -> Image.Image:
    """Attribution colour, with the readout tokens written over it.

    Two channels in one picture:

    - the cell COLOUR is the signed attribution — coral pushes the answer
      toward the positive label, teal toward the negative;
    - the WORDS are the lens readout at that cell, largest first.

    They answer different questions. A patch can be coloured strongly and say
    nothing legible, or read a clear word and move the decision not at all.

    Colour is normalized within this map, so intensity is not comparable
    between panels. ``pool`` merges pool x pool patch blocks (16 / 3 -> a 6x6
    grid with one thin edge band).
    """
    from PIL import Image as PILImage, ImageDraw

    from monocle import render as mrender

    vals, R, C = pool_map(values, n_rows, n_cols, pool)
    norm = normalize_map(vals, percentile).reshape(R, C)

    W, H = image.size
    W, H = W * upscale, H * upscale
    base = image.convert("RGB").resize((W, H), PILImage.LANCZOS)

    cells = np.zeros((R, C, 3), dtype=np.uint8)
    for r in range(R):
        for c in range(C):
            cells[r, c] = _div_color(float(norm[r, c]))
    heat = PILImage.fromarray(cells).resize((W, H), PILImage.NEAREST)
    out = PILImage.blend(base, heat, alpha)

    cw, ch = W / C, H / R
    draw = ImageDraw.Draw(out)
    if grid_lines:
        for r in range(1, R):
            draw.line([(0, r * ch), (W, r * ch)], fill=(255, 255, 255), width=1)
        for c in range(1, C):
            draw.line([(c * cw, 0), (c * cw, H)], fill=(255, 255, 255), width=1)

    if tokens:
        min_font = max(9, int(min(cw, ch) * 0.12))
        for idx, toks in tokens.items():
            r, c = divmod(idx, C)
            if r >= R:
                continue
            weight = abs(float(norm[r, c]))
            placed = mrender._patch_layout(
                [t for t in toks if t], weight, cw * 0.94, ch, min_font)
            cx, cy = (c + 0.5) * cw, (r + 0.5) * ch
            for item in placed:
                font = mrender.get_font(int(item["size"]))
                draw.text(
                    (cx, cy + item["dy"]), item["token"], font=font,
                    fill=(15, 15, 15), anchor="mm",
                    stroke_width=item["stroke"], stroke_fill=(255, 255, 255))
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------
def run_case(
    case: str, kind: str, proc, model, tmpl, lens, lens_model, *,
    n_pairs: int, conds: Optional[list[str]], layers: list[int],
    contrast: str, out_dir: Path, device: str, shard: int, n_shards: int,
    seed: int, n_maps: int, pool: int, percentile: float, smoke: bool,
) -> dict:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{shard}of{n_shards}" if n_shards > 1 else ""
    pq = case_dir / f"patch_attrib{suffix}.parquet"
    tok_pq = case_dir / f"patch_tokens{suffix}.parquet"
    diag_pq = case_dir / f"patch_diag{suffix}.parquet"

    conditions = canonical.build_conditions(case, kind, conds)
    label_first, _, _ = canonical.label_classes(
        proc.tokenizer, conditions["prod"])
    pos_id, neg_id, pos_name, neg_name = contrast_token_ids(
        contrast, label_first)
    log(f"{case}: contrast {pos_name} - {neg_name} "
        f"(ids {pos_id} - {neg_id})")

    rows_in = at.shard_rows(
        sw.load_pairs(n_pairs, case, kind, seed=seed), shard, n_shards)
    done = at.completed_pairs(pq)
    if done:
        log(f"{case}: resume — {len(done)} pairs already written")
    n_part = len(list(pq.parent.glob(f"{pq.stem}.part*.parquet")))

    image_token_id = int(model.config.image_token_id)
    n_suffix = len(proc.tokenizer.encode(
        canonical.FORCE_PREFIX, add_special_tokens=False))
    maps_dir = case_dir / "maps"

    new: list[dict] = []
    new_tok: list[dict] = []
    new_diag: list[dict] = []
    map_cache: dict = {}
    t0, n_done = time.time(), 0
    for pi, row in enumerate(rows_in):
        pid = row["pair_id"]
        if pid in done:
            continue
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]

        for cond, cfg in conditions.items():
            inputs = canonical.build_pair_inputs(
                proc, tmpl, imgs, row, cfg, device=device)
            seq_len = int(inputs["input_ids"].shape[1])
            answer_pos = sw.read_positions(seq_len, n_suffix)["label"]
            blocks = sw.image_token_blocks(inputs, image_token_id)
            grids = {
                "A": sw.square_grid(int(blocks[0].numel()), inputs, imgs[0].size),
                "B": sw.square_grid(int(blocks[1].numel()), inputs, imgs[1].size),
            }
            grads, acts, c = attribute_pair(
                model, lens_model, inputs, layers, pos_id, neg_id, answer_pos)
            base = {"case": case, "cond": cond, "pair_id": pid,
                    "prod_label": row["presented_label"],
                    "contrast": contrast, "contrast_value": c}
            rows_a, diag = patch_rows(grads, acts, blocks, grids, base)
            new.extend(rows_a)
            new_diag.extend(diag)
            rows_t = readout_rows(
                lens, lens_model, acts, blocks, grids, proc.tokenizer, base)
            new_tok.extend(rows_t)

            if pi < n_maps:
                maps_dir.mkdir(parents=True, exist_ok=True)
                tok_by = {}
                for r in rows_t:
                    tok_by.setdefault(
                        (r["layer"], r["slot"]), {}).setdefault(
                        r["patch_idx"], []).append((r["rank"], r["token"],
                                                    r["prob"]))
                for layer in sorted(grads):
                    for slot, positions, im in zip(("A", "B"), blocks, imgs):
                        g = grads[layer][0, positions, :]
                        h = acts[layer][0, positions, :]
                        a = grad_x_activation(
                            g, center_over_patches(h)).cpu().numpy()
                        n_r, n_c = grids[slot]
                        raw = tok_by.get((layer, slot), {})
                        toks = {i: [t for _, t, _ in sorted(v)]
                                for i, v in raw.items()}
                        prbs = {i: [pr for _, _, pr in sorted(v)]
                                for i, v in raw.items()}
                        merged = pool_tokens(toks, prbs, n_r, n_c, pool)
                        render_attrib(
                            im, a, n_r, n_c, pool=pool, tokens=merged,
                            percentile=percentile).save(
                            maps_dir / f"{pid}_{cond}_L{layer:02d}_{slot}.png")
                        map_cache[(pid, cond, layer, slot)] = (
                            a, merged, n_r, n_c)
            del grads, acts

        # The question-specific map. Both arms share the causal geometry, so
        # the difference is the only part that answers "what did the QUESTION
        # make the model look at".
        if pi < n_maps and {"prod", "neutral"} <= set(conditions):
            for layer in sorted({k[2] for k in map_cache if k[0] == pid}):
                for slot, im in zip(("A", "B"), imgs):
                    pk = (pid, "prod", layer, slot)
                    nk = (pid, "neutral", layer, slot)
                    if pk not in map_cache or nk not in map_cache:
                        continue
                    ap_, tok_p, n_r, n_c = map_cache[pk]
                    an_, _, _, _ = map_cache[nk]
                    diff = prod_minus_neutral(ap_, an_)
                    r = map_correlation(ap_, an_)
                    new_diag.append({
                        "case": case, "cond": "prod-neutral", "pair_id": pid,
                        "layer": int(layer), "slot": slot,
                        "prod_neutral_r": r,
                        "last_row_share": float(
                            row_profile(ap_, n_r, n_c)[-1]),
                    })
                    render_attrib(
                        im, diff, n_r, n_c, pool=pool, tokens=tok_p,
                        percentile=percentile).save(
                        maps_dir / f"{pid}_DIFF_L{layer:02d}_{slot}.png")
            map_cache = {k: v for k, v in map_cache.items() if k[0] != pid}

        n_done += 1
        if smoke or n_done % 10 == 0:
            log(f"  {case}: {n_done} pairs | "
                f"{(time.time() - t0) / max(n_done, 1):.2f}s/pair")
        if n_done % 10 == 0:
            at.write_part(new, pq, n_part)
            at.write_part(new_tok, tok_pq, n_part)
            at.write_part(new_diag, diag_pq, n_part)
            new, new_tok, new_diag = [], [], []
            n_part += 1

    at.write_part(new, pq, n_part)
    at.write_part(new_tok, tok_pq, n_part)
    at.write_part(new_diag, diag_pq, n_part)
    df = at.merge_parts(pq)
    at.merge_parts(tok_pq)
    diag_df = at.merge_parts(diag_pq)
    log(f"{case}: {len(df)} rows -> {pq}")
    return summarize_case(case, df, contrast, pos_name, neg_name,
                          case_dir, suffix, diag_df)


def summarize_case(
    case: str, df: pd.DataFrame, contrast: str, pos_name: str, neg_name: str,
    case_dir: Path, suffix: str = "",
    diag_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Per (cond, layer, slot) aggregates of the signed attribution."""
    groups = []
    if len(df):
        for (cond, layer, slot), g in df.groupby(["cond", "layer", "slot"]):
            a = g["attrib"].to_numpy()   # centered
            # Per-pair concentration: what share of the total absolute
            # attribution the top 5% of patches hold. A high value means a few
            # patches carry the decision; a low one means it is spread out.
            conc = []
            for _, gp in g.groupby("pair_id"):
                v = np.abs(gp["attrib"].to_numpy())
                if v.sum() > 0:
                    k = max(1, int(round(0.05 * len(v))))
                    conc.append(float(np.sort(v)[-k:].sum() / v.sum()))
            groups.append({
                "cond": cond, "layer": int(layer), "slot": slot,
                "n_pairs": int(g["pair_id"].nunique()),
                "mean_attrib": float(a.mean()),
                "mean_abs_attrib": float(np.abs(a).mean()),
                "max_abs_attrib": float(np.abs(a).max()),
                "top5pct_share": float(np.mean(conc)) if conc else float("nan"),
            })
    # The massive-activation diagnostic, per layer. A large shared ratio means
    # the uncentered map is unreadable at that depth.
    diag = []
    positional = []
    if diag_df is not None and "prod_neutral_r" in getattr(
            diag_df, "columns", []):
        pn = diag_df[diag_df["cond"] == "prod-neutral"]
        for layer, g in pn.groupby("layer"):
            positional.append({
                "layer": int(layer),
                "prod_neutral_r": float(g["prod_neutral_r"].mean()),
                "last_row_share": float(g["last_row_share"].mean()),
            })
    if diag_df is not None and len(diag_df):
        diag_df = diag_df[diag_df["cond"] != "prod-neutral"]
    if diag_df is not None and len(diag_df):
        for layer, g in diag_df.groupby("layer"):
            diag.append({
                "layer": int(layer),
                "shared_ratio": float(g["shared_ratio"].mean()),
                "mean_abs_raw": float(g["mean_abs_raw"].mean()),
                "mean_abs_centered": float(g["mean_abs_centered"].mean()),
            })
    summary = {
        "case": case, "contrast": contrast,
        "positive": pos_name, "negative": neg_name,
        "n_pairs": int(df["pair_id"].nunique()) if len(df) else 0,
        "attribution": "grad x CENTERED activation (see center_over_patches)",
        "by_group": groups,
        "massive_activation": diag,
        # Read these BEFORE any map. A high prod_neutral_r means the layer's
        # map is question-independent; a high last_row_share means it is
        # dominated by the causal aggregation row (uniform = 1/16 = 0.0625).
        "positional": positional,
    }
    p = case_dir / f"summary{suffix}.json"
    p.write_text(json.dumps(summary, indent=2))
    log(f"{case}: wrote {p}")
    return summary


def print_case_table(s: dict) -> None:
    log("=" * 78)
    log(f"CASE {s['case']} — contrast {s['positive']} - {s['negative']}")
    for d in s.get("positional", []):
        log(f"  L{d['layer']:<3} prod-vs-neutral r = {d['prod_neutral_r']:+.3f}"
            f" | last-row share = {d['last_row_share']:.3f} (uniform 0.063)")
    for d in s.get("massive_activation", []):
        log(f"  L{d['layer']:<3} shared||mean||/spread = {d['shared_ratio']:8.1f}"
            f" | mean|a| raw {d['mean_abs_raw']:.4f}"
            f" centered {d['mean_abs_centered']:.4f}")
    rows = s["by_group"]
    if not rows:
        return
    layers = sorted({r["layer"] for r in rows})
    for cond in sorted({r["cond"] for r in rows}):
        for slot in ("A", "B"):
            by = {r["layer"]: r for r in rows
                  if r["cond"] == cond and r["slot"] == slot}
            if not by:
                continue
            log(f"  {cond:<8} img {slot}  mean|a| : " + " ".join(
                f"L{l}={by[l]['mean_abs_attrib']:.3f}" for l in layers))
            log(f"  {'':<8}       top5% : " + " ".join(
                f"L{l}={by[l]['top5pct_share']:.2f}" for l in layers))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=list(canonical.CASES),
                    choices=list(canonical.CASES))
    ap.add_argument("--kind", default="proxy", choices=list(canonical.KINDS))
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--contrast", default="judgment", choices=list(CONTRASTS),
                    help=("'judgment' is More-Less; 'abstain' is NotSure-More, "
                          "for the high-abstention cases."))
    ap.add_argument("--layers", nargs="+", type=int, default=FITTED_LAYERS)
    ap.add_argument("--pool", type=int, default=3,
                    help=("Merge pool x pool patch blocks when rendering. "
                          "16/3 -> a 6x6 grid. The parquet always keeps all "
                          "256 patches, so a new pool needs no new run."))
    ap.add_argument("--lens", default=at.LENS_DEFAULT,
                    help="Lens for the per-patch readout words.")
    ap.add_argument("--percentile", type=float, default=95.0,
                    help=("Colour scale clip. Attribution is extremely peaked "
                          "at depth (top 5%% of patches hold 95%% at L46), so "
                          "max-scaling renders those maps blank. Tuned for "
                          "the POOLED grid. 100 = max-scaled."))
    ap.add_argument("--n-maps", type=int, default=4,
                    help="Render overlays for the first N pairs of each case.")
    ap.add_argument("--shard", default=None, metavar="i/n")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    shard, n_shards = at.parse_shard(args.shard)
    n_pairs = 2 if args.smoke else args.n_pairs
    n_maps = 1 if args.smoke else args.n_maps
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc, model, tmpl = extract.load_model(args.model_dir, device=args.device)
    # The graph must root at the lowest recorded block, which needs every
    # parameter to sit outside the graph. This also drops the parameter
    # gradients we never use.
    model.requires_grad_(False)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens)
    layers = sorted(set(args.layers) | {FINAL_LAYER})
    log(f"layers {layers} | contrast {args.contrast} | pool {args.pool}")

    summaries = {}
    for case in args.cases:
        s = run_case(
            case, args.kind, proc, model, tmpl, lens, lens_model,
            n_pairs=n_pairs, conds=args.conditions, layers=layers,
            contrast=args.contrast, out_dir=out_dir, device=args.device,
            shard=shard, n_shards=n_shards, seed=args.seed,
            n_maps=n_maps, pool=args.pool, percentile=args.percentile,
            smoke=args.smoke)
        summaries[case] = s
        print_case_table(s)

    idx = out_dir / (f"index.shard{shard}of{n_shards}.json" if n_shards > 1
                     else "index.json")
    idx.write_text(json.dumps({
        "cases": args.cases, "kind": args.kind, "contrast": args.contrast,
        "layers": layers, "n_pairs": n_pairs, "seed": args.seed,
        "method": ("grad x CENTERED activation, one backward per "
                   "(pair, condition); attrib_raw keeps the uncentered value"),
        "pool": args.pool, "lens": args.lens,
        "shard": shard, "n_shards": n_shards,
        "n_pairs_read": {c: s["n_pairs"] for c, s in summaries.items()},
    }, indent=2))
    log(f"wrote {idx}")
    log("patch attribution complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
