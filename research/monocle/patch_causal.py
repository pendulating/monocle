"""Calibrate the patch attribution against a causal ablation.

`patch_attribution.py` scores a patch with grad x activation at a chosen
layer. That score is a first-order estimate, and a faithfulness test on the
200-pair run showed it tracks the decision at L30-L42 and INVERTS at L46. This
module measures the thing those scores try to predict.

    true delta_c(block) = c(image) - c(image with that block set to the
                          mean patch)

One forward pass for each ablated block. No linearization, no lens. The number
it returns is what the model does.

Two predictors are scored against it.

| Predictor | Where it reads | Cost |
|---|---|---|
| `pred_layer_<l>` | grad x centered activation at residual layer l | 1 backward |
| `pred_pixel` | dc/d(pixel_values) x (pixels - mean patch) | 1 backward |

`pred_pixel` exists because gemma-4-12b is encoder-free. The vision embedder
holds no attention and reads each patch alone, so `dc/d(pixel_values)[p]` has
no cross-patch path through the embedder. It is the input attribution the
architecture makes well posed. The residual-layer scores cannot be: by layer 1
bidirectional attention mixes every patch into every other patch.

Both predictors use the SAME baseline as the ablation -- the mean patch of the
image. Thus a perfect first-order predictor would return the true delta.

Geometry
--------
Scores are pooled to `pool` x `pool` patch blocks and SUMMED, not averaged.
Attribution is extensive: the prediction for one ablated block is the sum of
the scores of the patches inside it.

Outputs, per case
-----------------
| File | One row per |
|---|---|
| `patch_causal.parquet` | (pair, slot, block) -- true delta and each predictor |
| `summary.json` | correlation of each predictor with the true delta |
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import answer_tokens as at  # noqa: E402
from research.monocle import canonical, extract  # noqa: E402
from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import safety_workspace as sw  # noqa: E402

MODEL_DIR = pa.MODEL_DIR
OUT_DEFAULT = REPO / "outputs/_monocle/patch_causal"
SEED = pa.SEED


def log(m: str) -> None:
    print(f"[patch-causal] {m}", flush=True)


# ---------------------------------------------------------------------------
# Geometry (pure)
# ---------------------------------------------------------------------------
def grid_from_position_ids(pid: torch.Tensor) -> tuple[int, int, np.ndarray]:
    """(n_rows, n_cols, valid slot indices) for one image.

    `pid` is the processor's `image_position_ids` for one image, shape
    (n_slots, 2) holding (x, y) and -1 where the slot is padding. The
    processor pads every image to the full soft-token budget, so the live
    patches are a prefix and the tail is padding.
    """
    pid = torch.as_tensor(pid).long()
    if pid.ndim != 2 or pid.shape[-1] != 2:
        raise ValueError(f"expected (n_slots, 2), got {tuple(pid.shape)}")
    valid = (pid[:, 0] != -1).nonzero(as_tuple=True)[0].cpu().numpy()
    if valid.size == 0:
        raise ValueError("no valid patch slots")
    xs, ys = pid[valid, 0].cpu().numpy(), pid[valid, 1].cpu().numpy()
    n_cols, n_rows = int(xs.max()) + 1, int(ys.max()) + 1
    # The sequence order must be raster order, or every patch map is
    # transposed. Check it rather than trust it.
    expect = ys * n_cols + xs
    if not np.array_equal(expect, np.arange(valid.size)):
        raise RuntimeError(
            "patch slots are not in raster order; the row/col mapping in "
            "patch_attribution would be wrong")
    return n_rows, n_cols, valid


def pooled_blocks(
    n_rows: int, n_cols: int, pool: int,
) -> list[tuple[int, int, np.ndarray]]:
    """[(block_row, block_col, patch indices)] over pool x pool blocks.

    The last block of a row or column is short when `pool` does not divide the
    grid: 16 / 3 gives five blocks of 3 and one of 1.
    """
    if pool < 1:
        raise ValueError(f"pool must be >= 1, got {pool}")
    out = []
    for br in range(-(-n_rows // pool)):
        for bc in range(-(-n_cols // pool)):
            rows = range(br * pool, min((br + 1) * pool, n_rows))
            cols = range(bc * pool, min((bc + 1) * pool, n_cols))
            idx = np.array([r * n_cols + c for r in rows for c in cols],
                           dtype=np.int64)
            out.append((br, bc, idx))
    return out


def sum_over_blocks(
    values: np.ndarray, blocks: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Total score inside each block. Attribution is extensive, so SUM."""
    return np.array([float(values[idx].sum()) for _, _, idx in blocks])


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, or NaN when either side is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of blocks where the predictor and the truth agree in sign."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    live = (a != 0) & (b != 0)
    return float((np.sign(a[live]) == np.sign(b[live])).mean()) if live.any() \
        else float("nan")


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def fp32_contrast(
    lens_model: Any, h: torch.Tensor, pos_id: int, neg_id: int,
) -> torch.Tensor:
    """c = logit(pos) - logit(neg), computed in float32.

    `HFLensModel.unembed` runs the head in the model dtype, so its output is
    bfloat16 and the contrast lands on a grid about 0.125 wide. The ablation
    effect we measure is a DIFFERENCE of two contrasts, so that grid is the
    noise floor -- the smoke run returned a contrast of exactly 0.000 for one
    pair. Only 2 rows of the head are needed, so float32 is nearly free.

    `h` is the final block's output at the answer position, shape [B, d].
    """
    w = lens_model._lm_head.weight
    x = lens_model._final_norm(h.to(w.dtype)).float()
    rows = w[[pos_id, neg_id]].float()
    lg = x @ rows.T
    cap = lens_model._logit_softcap
    if cap is not None:
        lg = cap * torch.tanh(lg / cap)
    return lg[..., 0] - lg[..., 1]


IMAGE_KEYS = ("pixel_values", "image_position_ids")


def replicate(inputs: dict, n: int) -> dict:
    """The same prompt as a batch of `n` rows.

    Text tensors carry a batch dimension of 1 and image tensors carry one row
    for each image, so the two groups repeat separately. The images stay in
    (A, B, A, B, ...) order, which is the order the model consumes them in.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    out = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        reps = [n] + [1] * (v.ndim - 1)
        out[k] = v.repeat(*reps)
    return out


def final_residual(
    model: Any, lens_model: Any, inputs: dict, *, graph: bool,
) -> torch.Tensor:
    """Output of the last block at the answer position, shape [B, d].

    The answer position is the last token. Recording the block output rather
    than reading `out.logits` keeps the head in float32 and never materializes
    a 262,144-wide row for every position.
    """
    from jlens.hooks import ActivationRecorder

    ctx = torch.enable_grad() if graph else torch.no_grad()
    with ctx, ActivationRecorder(lens_model.layers, at=[pa.FINAL_LAYER]) as rec:
        out = model(**inputs, use_cache=False, logits_to_keep=1)
        del out
        return rec.activations[pa.FINAL_LAYER][:, -1, :]


def check_answer_pos(inputs: dict, answer_pos: int) -> None:
    n = int(inputs["input_ids"].shape[1])
    if answer_pos != n - 1:
        raise ValueError(
            f"answer_pos {answer_pos} is not the last token ({n - 1}); "
            "final_residual reads the last position")


def pixel_attribution(
    model: Any, lens_model: Any, inputs: dict, answer_pos: int,
    pos_id: int, neg_id: int, valid: list[np.ndarray],
) -> tuple[list[np.ndarray], float]:
    """dc/d(pixel_values) x (pixels - mean patch), per image, per patch.

    Returns ([attrib for image A, attrib for image B], c). The baseline is the
    mean over the live patches of the SAME image, which the ablation also uses.
    """
    check_answer_pos(inputs, answer_pos)
    px = inputs["pixel_values"].detach().clone().requires_grad_(True)
    fwd = dict(inputs)
    fwd["pixel_values"] = px
    h = final_residual(model, lens_model, fwd, graph=True)
    c = fp32_contrast(lens_model, h, pos_id, neg_id)[0]
    if not torch.isfinite(c):
        raise RuntimeError("non-finite contrast at the answer position")
    (grad,) = torch.autograd.grad(c, px)
    out = []
    for i, v in enumerate(valid):
        g = grad[i, v, :].float()
        q = px.detach()[i, v, :].float()
        out.append((g * (q - q.mean(0, keepdim=True))).sum(-1).cpu().numpy())
    return out, float(c.detach())


@torch.no_grad()
def ablate_batch(
    model: Any, lens_model: Any, inputs: dict, pos_id: int, neg_id: int,
    jobs: list[tuple[int, np.ndarray]], baselines: dict[int, torch.Tensor],
    batch: int,
) -> np.ndarray:
    """c for each (image index, patch slots) job, with those slots set to the
    mean patch of that image. Jobs run in batches of `batch` replicas."""
    n_img = int(inputs["pixel_values"].shape[0])
    out = np.empty(len(jobs), dtype=float)
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        fwd = replicate(inputs, len(chunk))
        px = fwd["pixel_values"]
        for r, (img, slots) in enumerate(chunk):
            idx = torch.as_tensor(slots, device=px.device)
            px[r * n_img + img, idx] = baselines[img].to(px.dtype)
        h = final_residual(model, lens_model, fwd, graph=False)
        c = fp32_contrast(lens_model, h, pos_id, neg_id)
        out[start:start + len(chunk)] = c.float().cpu().numpy()
    return out


@torch.no_grad()
def baseline_inputs(inputs: dict, valid: list[np.ndarray]) -> dict:
    """The same prompt with EVERY live patch of both images set to its image's
    mean patch. This is the origin of the integrated-gradients path, and
    `c` here is the endpoint the completeness check needs."""
    out = dict(inputs)
    px = inputs["pixel_values"].clone()
    for i, v in enumerate(valid):
        idx = torch.as_tensor(v, device=px.device)
        px[i, idx] = px[i, idx].float().mean(0).to(px.dtype)
    out["pixel_values"] = px
    return out


def integrated_gradients(
    model: Any, lens_model: Any, inputs: dict, pos_id: int, neg_id: int,
    valid: list[np.ndarray], steps: int, batch: int,
) -> tuple[list[np.ndarray], float, float]:
    """Integrated gradients from the mean-patch image to the real image.

    A single-point ablation to the mean patch is out of distribution -- it
    drops a flat square into a street photo, and the model answers the anomaly
    rather than the missing content. Measured: a single patch anywhere in the
    grid moves c by 0.22 to 0.46, with no dependence on position. Integrated
    gradients never evaluates that point. It walks the straight path from the
    mean-patch image to the real one and averages the gradient along it.

        IG[p] = (x[p] - xbar[p]) . mean over k of dc/dx[p] at
                xbar + (k + 0.5)/steps * (x - xbar)

    Returns (per-image per-patch IG, c at the real image, c at the baseline).

    IG satisfies completeness: the attributions SUM to c(x) - c(xbar). The
    caller checks that residual, which makes this attribution self-validating
    in a way grad x activation is not.

    Warning: `batch` here is NOT the ablation batch. Each step retains a graph
    over all 48 blocks, so the memory cost is about 8x an ablation forward. A
    batch of 8 runs the 48 GB card out of memory.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    px_real = inputs["pixel_values"]
    px_base = baseline_inputs(inputs, valid)["pixel_values"]
    delta = (px_real.float() - px_base.float())

    total = torch.zeros_like(delta)
    alphas = [(k + 0.5) / steps for k in range(steps)]
    for start in range(0, steps, batch):
        chunk = alphas[start:start + batch]
        fwd = replicate(inputs, len(chunk))
        n_img = int(px_real.shape[0])
        px = torch.cat([
            (px_base.float() + a * delta).to(px_real.dtype) for a in chunk])
        px = px.detach().requires_grad_(True)
        fwd["pixel_values"] = px
        h = final_residual(model, lens_model, fwd, graph=True)
        c = fp32_contrast(lens_model, h, pos_id, neg_id)
        (g,) = torch.autograd.grad(c.sum(), px)
        for r in range(len(chunk)):
            total += g[r * n_img:(r + 1) * n_img].float()
        del g, px, h, c, fwd
        torch.cuda.empty_cache()
    avg = total / steps

    out = []
    for i, v in enumerate(valid):
        out.append((avg[i, v, :] * delta[i, v, :]).sum(-1).cpu().numpy())

    with torch.no_grad():
        h_r = final_residual(model, lens_model, inputs, graph=False)
        c_real = float(fp32_contrast(lens_model, h_r, pos_id, neg_id)[0])
        h_b = final_residual(
            model, lens_model, baseline_inputs(inputs, valid), graph=False)
        c_base = float(fp32_contrast(lens_model, h_b, pos_id, neg_id)[0])
    return out, c_real, c_base


@torch.no_grad()
def path_probe(
    model: Any, lens_model: Any, inputs: dict, pos_id: int, neg_id: int,
    valid: list[np.ndarray], n: int, batch: int,
) -> list[tuple[float, float]]:
    """c at n points along the mean-patch -> real-image path.

    Integrated gradients assumes the path is one the model can be integrated
    over. A straight line from a flat grey image to a street photograph runs
    through pictures no camera makes. If c swings wildly along it, no
    baseline-path attribution converges, whatever the step count.
    """
    px_real = inputs["pixel_values"]
    px_base = baseline_inputs(inputs, valid)["pixel_values"]
    delta = px_real.float() - px_base.float()
    alphas = [k / (n - 1) for k in range(n)]
    out = []
    for start in range(0, n, batch):
        chunk = alphas[start:start + batch]
        fwd = replicate(inputs, len(chunk))
        fwd["pixel_values"] = torch.cat([
            (px_base.float() + a * delta).to(px_real.dtype) for a in chunk])
        h = final_residual(model, lens_model, fwd, graph=False)
        c = fp32_contrast(lens_model, h, pos_id, neg_id)
        out.extend(zip(chunk, c.float().cpu().tolist()))
    return out


def grad_check(
    model: Any, lens_model: Any, inputs: dict, pos_id: int, neg_id: int,
    targets: tuple[float, ...] = (0.05, 0.2, 1.0, 4.0),
) -> list[tuple[float, float, float]]:
    """Is the pixel gradient locally predictive at all?

    Step along the gradient direction by an amount the gradient itself says
    should move c by `target`, then measure what c actually does. Returns
    [(target, measured, ratio)].

    This separates two very different diagnoses of a failed integrated
    gradient. If the measured move tracks the prediction for small steps, the
    gradient is sound and the PATH is what breaks completeness. If even a tiny
    step misses, the gradient is numerical noise -- 48 bfloat16 blocks of
    accumulated error -- and no gradient method can work at pixel level.
    """
    px = inputs["pixel_values"].detach().clone().requires_grad_(True)
    fwd = dict(inputs)
    fwd["pixel_values"] = px
    h = final_residual(model, lens_model, fwd, graph=True)
    c0 = fp32_contrast(lens_model, h, pos_id, neg_id)[0]
    (g,) = torch.autograd.grad(c0, px)
    c0 = float(c0.detach())
    g = g.float()
    norm = float(g.norm())
    if norm == 0:
        raise RuntimeError("pixel gradient is identically zero")
    unit = (g / norm)

    out = []
    with torch.no_grad():
        for target in targets:
            # Predicted move for a step of t along the unit gradient is
            # t * ||g||, so t = target / ||g||.
            t = target / norm
            fwd2 = dict(inputs)
            fwd2["pixel_values"] = (
                inputs["pixel_values"].float() + t * unit
            ).to(inputs["pixel_values"].dtype)
            h2 = final_residual(model, lens_model, fwd2, graph=False)
            c1 = float(fp32_contrast(lens_model, h2, pos_id, neg_id)[0])
            out.append((target, c1 - c0, (c1 - c0) / target))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_case(
    case: str, kind: str, proc: Any, model: Any, tmpl: str, lens_model: Any,
    *, n_pairs: int, layers: list[int], contrast: str, out_dir: Path,
    device: str, shard: int, n_shards: int, seed: int, pool: int,
    batch: int, ig_steps: int, ig_batch: int,
    probe_path: bool = False, check_grad: bool = False,
) -> dict:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{shard}of{n_shards}" if n_shards > 1 else ""
    pq = case_dir / f"patch_causal{suffix}.parquet"

    cfg = canonical.build_conditions(case, kind, ["prod"])["prod"]
    label_first, _, _ = canonical.label_classes(proc.tokenizer, cfg)
    pos_id, neg_id, pos_name, neg_name = pa.contrast_token_ids(
        contrast, label_first)
    log(f"{case}: contrast {pos_name} - {neg_name}")

    rows_in = at.shard_rows(
        sw.load_pairs(n_pairs, case, kind, seed=seed), shard, n_shards)
    done = at.completed_pairs(pq)
    if done:
        log(f"{case}: resume — {len(done)} pairs already written")
    n_part = len(list(pq.parent.glob(f"{pq.stem}.part*.parquet")))

    image_token_id = int(model.config.image_token_id)
    n_suffix = len(proc.tokenizer.encode(
        canonical.FORCE_PREFIX, add_special_tokens=False))
    attr_layers = sorted(set(layers) | {pa.FINAL_LAYER})

    new: list[dict] = []
    t0, n_done = time.time(), 0
    for row in rows_in:
        pid_ = row["pair_id"]
        if pid_ in done:
            continue
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]
        inputs = canonical.build_pair_inputs(
            proc, tmpl, imgs, row, cfg, device=device)
        seq_len = int(inputs["input_ids"].shape[1])
        answer_pos = sw.read_positions(seq_len, n_suffix)["label"]
        seq_blocks = sw.image_token_blocks(inputs, image_token_id)

        geom = [grid_from_position_ids(inputs["image_position_ids"][i])
                for i in (0, 1)]
        for i, (n_r, n_c, v) in enumerate(geom):
            if v.size != int(seq_blocks[i].numel()):
                raise RuntimeError(
                    f"image {i}: {v.size} live patch slots but "
                    f"{int(seq_blocks[i].numel())} image tokens")

        # Predictor 1: grad x centered activation at each residual layer.
        _t = time.time()
        grads, acts, c_layer = pa.attribute_pair(
            model, lens_model, inputs, attr_layers, pos_id, neg_id, answer_pos)
        layer_pred = {}
        present = sorted(grads)
        for layer in present:
            for si, slot in enumerate(("A", "B")):
                g = grads[layer][0, seq_blocks[si], :]
                h = acts[layer][0, seq_blocks[si], :]
                layer_pred[(layer, slot)] = pa.grad_x_activation(
                    g, pa.center_over_patches(h)).float().cpu().numpy()
        del grads, acts
        t_layer = time.time() - _t
        _t = time.time()

        # Predictor 2: gradient into the raw pixels.
        px_attr, c = pixel_attribution(
            model, lens_model, inputs, answer_pos, pos_id, neg_id,
            [g[2] for g in geom])

        t_pixel = time.time() - _t
        _t = time.time()

        valids = [g[2] for g in geom]
        if check_grad:
            res = grad_check(model, lens_model, inputs, pos_id, neg_id)
            log(f"{pid_}: grad-check — " + "  ".join(
                f"pred{t:+.2f}/meas{m:+.2f}(x{r:.2f})" for t, m, r in res))
            n_done += 1
            continue
        if probe_path:
            pts = path_probe(model, lens_model, inputs, pos_id, neg_id,
                             valids, 17, batch)
            log(f"{pid_}: c along the path — " + "  ".join(
                f"{a:.2f}:{v:+.2f}" for a, v in pts))
            n_done += 1
            continue
        ig_attr, c_real, c_base = integrated_gradients(
            model, lens_model, inputs, pos_id, neg_id, valids, ig_steps,
            ig_batch)
        ig_total = float(sum(a.sum() for a in ig_attr))
        ig_target = c_real - c_base
        ig_resid = (abs(ig_total - ig_target) / abs(ig_target)
                    if ig_target else float("nan"))
        t_ig = time.time() - _t
        _t = time.time()

        # Truth: ablate each pooled block to the mean patch of its image.
        baselines = {si: inputs["pixel_values"][si, geom[si][2], :]
                     .float().mean(0) for si in (0, 1)}
        block_sets = {si: pooled_blocks(geom[si][0], geom[si][1], pool)
                      for si in (0, 1)}
        jobs = [(si, geom[si][2][idx])
                for si in (0, 1) for _, _, idx in block_sets[si]]
        c_ab = ablate_batch(model, lens_model, inputs, pos_id, neg_id,
                            jobs, baselines, batch)
        if n_done == 0:
            # Plumbing gate: an EMPTY slot set must reproduce c bit for bit.
            # Without it a batching bug would look like a scientific null.
            c_null = ablate_batch(
                model, lens_model, inputs, pos_id, neg_id,
                [(0, np.array([], dtype=np.int64))], baselines, batch)[0]
            if abs(c_null - c) > 1e-4:
                raise RuntimeError(
                    f"null ablation moved the contrast: {c_null} vs {c}")
            log(f"null-ablation gate passed (c={c:.4f})")
        if n_done == 0:
            log(f"timing: layers {t_layer:.1f}s  pixel {t_pixel:.1f}s  "
                f"ig({ig_steps}) {t_ig:.1f}s  {len(jobs)} ablations "
                f"{time.time() - _t:.1f}s (batch {batch})")
            log(f"IG completeness: sum {ig_total:+.3f} vs "
                f"c-c_base {ig_target:+.3f} (residual {ig_resid:.1%})")

        k = 0
        for si, slot in enumerate(("A", "B")):
            blocks = block_sets[si]
            pred_px = sum_over_blocks(px_attr[si], blocks)
            preds = {f"pred_layer_{l}":
                     sum_over_blocks(layer_pred[(l, slot)], blocks)
                     for l in present}
            for bi, (br, bc, idx) in enumerate(blocks):
                rec = {
                    "case": case, "pair_id": pid_, "slot": slot,
                    "prod_label": row["presented_label"],
                    "contrast": contrast, "c": c, "c_layer_path": c_layer,
                    "block": bi, "block_row": br, "block_col": bc,
                    "n_patches": int(idx.size),
                    "c_ablated": float(c_ab[k]),
                    "true_delta": c - float(c_ab[k]),
                    "pred_pixel": float(pred_px[bi]),
                    "pred_ig": float(sum_over_blocks(ig_attr[si], blocks)[bi]),
                    "c_baseline": c_base,
                    "ig_total": ig_total, "ig_residual": ig_resid,
                }
                for kk, vec in preds.items():
                    rec[kk] = float(vec[bi])
                new.append(rec)
                k += 1

        n_done += 1
        if n_done % 5 == 0:
            rate = (time.time() - t0) / n_done
            log(f"{case}: {n_done} pairs, {rate:.1f} s/pair")
            at.write_part(new, pq, n_part)
            n_part += 1
            new = []

    if new:
        at.write_part(new, pq, n_part)
    df = at.merge_parts(pq)
    summary = summarize(case, contrast, df)
    (case_dir / f"summary{suffix}.json").write_text(json.dumps(summary,
                                                              indent=2))
    return summary


def summarize(case: str, contrast: str, df: pd.DataFrame) -> dict:
    """Correlation of each predictor with the measured ablation effect."""
    preds = [c for c in df.columns if c.startswith("pred_")]
    rows = []
    for p in preds:
        rows.append({
            "predictor": p,
            "r_all": safe_corr(df[p], df["true_delta"]),
            "r_A": safe_corr(*(df[df.slot == "A"][[p, "true_delta"]].values.T)),
            "r_B": safe_corr(*(df[df.slot == "B"][[p, "true_delta"]].values.T)),
            "sign_agreement": sign_agreement(df[p], df["true_delta"]),
        })
    rows.sort(key=lambda r: -(r["r_all"] if np.isfinite(r["r_all"]) else -9))
    return {
        "case": case, "contrast": contrast,
        "n_pairs": int(df["pair_id"].nunique()),
        "n_blocks": int(len(df)),
        "mean_abs_true_delta": float(df["true_delta"].abs().mean()),
        "mean_abs_c": float(df["c"].abs().mean()),
        "predictors": rows,
    }


def print_summary(s: dict) -> None:
    log("=" * 70)
    log(f"CASE {s['case']} — {s['n_pairs']} pairs, {s['n_blocks']} blocks")
    log(f"  mean |true delta_c| = {s['mean_abs_true_delta']:.4f} "
        f"(mean |c| = {s['mean_abs_c']:.3f})")
    log(f"  {'predictor':<18}{'r all':>9}{'r A':>8}{'r B':>8}{'sign':>8}")
    for r in s["predictors"]:
        log(f"  {r['predictor']:<18}{r['r_all']:>9.3f}{r['r_A']:>8.3f}"
            f"{r['r_B']:>8.3f}{r['sign_agreement']:>8.2f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(canonical.CASES))
    ap.add_argument("--kind", default="proxy", choices=list(canonical.KINDS))
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--contrast", default="judgment",
                    choices=list(pa.CONTRASTS))
    ap.add_argument("--layers", nargs="+", type=int, default=pa.FITTED_LAYERS)
    ap.add_argument("--pool", type=int, default=3,
                    help="16 / 3 -> a 6x6 block grid, 36 ablations per image.")
    ap.add_argument("--grad-check", action="store_true",
                    help=("Step along the pixel gradient and report whether c "
                          "moves as the gradient predicts. Run this when the "
                          "completeness residual is large."))
    ap.add_argument("--probe-path", action="store_true",
                    help=("Print c at 17 points along the IG path and stop. "
                          "Use when the completeness residual is large."))
    ap.add_argument("--ig-batch", type=int, default=2,
                    help=("Path steps per backward. Each retains a graph over "
                          "all 48 blocks, so this must stay far below "
                          "--batch: 8 runs a 48 GB card out of memory."))
    ap.add_argument("--ig-steps", type=int, default=32,
                    help=("Integrated-gradients path steps. The completeness "
                          "residual reported per case says whether this is "
                          "enough."))
    ap.add_argument("--batch", type=int, default=8,
                    help=("Ablated replicas per forward pass. One prompt is "
                          "~703 tokens, so batch 1 leaves the GPU idle."))
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc, model, tmpl = extract.load_model(args.model_dir, device=args.device)
    model.requires_grad_(False)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    log(f"layers {args.layers} | pool {args.pool} | batch {args.batch} "
        f"| ig {args.ig_steps} steps @ {args.ig_batch} "
        f"| contrast {args.contrast}")

    for case in args.cases:
        s = run_case(
            case, args.kind, proc, model, tmpl, lens_model,
            n_pairs=n_pairs, layers=args.layers, contrast=args.contrast,
            out_dir=out_dir, device=args.device, shard=shard,
            n_shards=n_shards, seed=args.seed, pool=args.pool,
            batch=args.batch, ig_steps=args.ig_steps,
            ig_batch=args.ig_batch, probe_path=args.probe_path,
            check_grad=args.grad_check)
        print_summary(s)
    log("patch causal calibration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
