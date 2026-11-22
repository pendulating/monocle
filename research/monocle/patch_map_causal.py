"""A causal patch map, and whether the cheap map predicts it.

Why this exists
---------------
`patch_swap` found that the image is causally spent by L30, and that the
attribution window L40-L45 is inert. Thus a patch map belongs near L20, not
near L43.

But we cannot simply retarget the renderer, because the metric that chose L43
is now discredited. It scores a layer by how well the map AGGREGATE tracks
`c`, and that metric and the causal truth point opposite ways:

| Depth | map correlates with `c` | patches actually cause `c` |
|---|---|---|
| L18-L24 | about 0 | about 1.0 |
| L43 | 0.805 | 0.006 |

The correlation rewards a layer whose residual has become LINEARLY ALIGNED
with the decision, which happens after the decision is made. So it cannot be
used to validate a map at a low layer, and the open risk is that grad x
activation is simply a poor estimator down there, where the patch-to-judgment
map is still distributed.

The test
--------
Build the ground truth directly: swap EACH patch on its own and record how far
`c` moves. That is 256 interventions for one image, batched. Then ask whether
the 1-backward-pass map predicts it.

| Outcome | What to do |
|---|---|
| the maps agree | retarget the renderer to the low layer; keep the estimator |
| they do not | the red-blue map needs a different estimator, not a new layer |

The `all` arm rides along as a gate: at a layer inside the causal window it
must recover most of the judgment, or the layer is the wrong place to look.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_causal as pc  # noqa: E402
from research.monocle import patch_swap as psw  # noqa: E402
from research.monocle import safety_workspace as sw  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/patch_map_causal"
#: Layers inside the causal window the swap located (patches hold the
#: judgment to about L24, and are spent by L30).
DEFAULT_LAYERS = (12, 20, 24, 28)
#: The `all` arm must recover at least this much, or the layer is too high to
#: carry a causal patch map at all.
WINDOW_FLOOR = 0.3


def log(m: str) -> None:
    print(f"[patch-map] {m}", flush=True)


class BatchPatcher:
    """Row i of the batch gets its own position, or its own GROUP, replaced.

    2 input forms:

    | `positions` | `values` | Meaning |
    |---|---|---|
    | `[3, 7, 9]` | `[3, d]` | row i replaces 1 position |
    | `[[3, 4], [7, 8]]` | 2 tensors of `[2, d]` | row i replaces a group |

    The group form keeps a separate donor vector for each position in the
    group. A group is NOT 1 vector written 4 times.
    """

    def __init__(self, positions: Any, values: Any) -> None:
        flat = all(isinstance(p, (int, np.integer)) for p in positions)
        if flat:
            if len(positions) != values.shape[0]:
                raise ValueError(
                    f"{len(positions)} positions but {values.shape[0]} values")
            self.groups = [[int(p)] for p in positions]
            self.values = [values[i:i + 1] for i in range(len(positions))]
        else:
            self.groups = [[int(x) for x in g] for g in positions]
            self.values = list(values)
            if len(self.groups) != len(self.values):
                raise ValueError(
                    f"{len(self.groups)} positions but "
                    f"{len(self.values)} values")
            for g, v in zip(self.groups, self.values):
                if len(g) != v.shape[0]:
                    raise ValueError(
                        f"a group of {len(g)} positions got {v.shape[0]} "
                        "values. Each position needs its own donor vector.")

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        for i, (pos, vals) in enumerate(zip(self.groups, self.values)):
            idx = torch.as_tensor(pos, device=h.device)
            h[i, idx, :] = vals.to(h.dtype).to(h.device)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


def patch_groups(n_patches: int, k: int) -> list[list[int]]:
    """Square `k` x `k` blocks of the patch grid, in row-major order.

    Why group at all: 1 soft token covers 48 px of the 768 px image, which is
    a tyre or a piece of concrete, not a thing. A 2 x 2 group covers 96 px,
    the same area as a cell of the model's own level-70 grid, but the model
    still reads the picture at full resolution.
    """
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise ValueError(f"{n_patches} patches is not square")
    if k < 1 or side % k:
        raise ValueError(
            f"a {side}x{side} grid does not divide into {k}x{k} groups")
    out = []
    for r0 in range(0, side, k):
        for c0 in range(0, side, k):
            out.append([(r0 + dr) * side + (c0 + dc)
                        for dr in range(k) for dc in range(k)])
    return out



def pool_by_groups(values: Any, group: int) -> np.ndarray:
    """Sum a square patch map into `group` x `group` blocks, row-major.

    A map holds a contribution in logit units, so the blocks SUM. A mean would
    change the scale between grids and make 2 runs incomparable.

    Warning: `sum_over_blocks` does not accept `patch_groups` output. That
    function wants (row, col, index) tuples, and `patch_groups` gives plain
    index lists. Call this function instead.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if group == 1:
        return arr
    return np.array([float(arr[idx].sum())
                     for idx in patch_groups(arr.size, group)])


def causal_map(
    model: Any, lens_model: Any, inputs: dict, layer: int, block: list[int],
    donor_h: torch.Tensor, c0: float, pos_id: int, neg_id: int, *,
    batch: int = 8, group: int = 1,
) -> np.ndarray:
    """`c` change from a swap of 1 patch, or of a `group` x `group` block.

    Returns [n_patches] when `group` is 1, and [n_patches / group**2] when it
    is more. The order stays row-major, so `grid_of` reads it directly.
    """
    groups = (patch_groups(len(block), group) if group > 1
              else [[i] for i in range(len(block))])
    deltas = np.zeros(len(groups), dtype=np.float64)
    for start in range(0, len(groups), batch):
        chunk = groups[start:start + batch]
        n = len(chunk)
        rep = pc.replicate(inputs, n)
        # donor_h rows are in patch order, so a patch index reads it directly.
        vals = [donor_h[g, :] for g in chunk]
        handle = lens_model.layers[layer].register_forward_hook(
            BatchPatcher([[block[i] for i in g] for g in chunk], vals))
        try:
            with torch.no_grad():
                h = pc.final_residual(model, lens_model, rep, graph=False)
            c = pc.fp32_contrast(lens_model, h, pos_id, neg_id)
        finally:
            handle.remove()
        deltas[start:start + n] = (c.float().cpu().numpy() - c0)
    return deltas


def batched_baseline(
    model: Any, lens_model: Any, inputs: dict, pos_id: int, neg_id: int, *,
    batch: int,
) -> float:
    """`c` from an UNPERTURBED row, at the batch size the swaps run at.

    In bfloat16 a forward pass depends on the size of its batch. `causal_map`
    runs the swaps at `batch`, so a baseline taken at batch 1 offsets every
    delta of that image by a constant.

    Measured on the 2026-08-23 run, the constant ran from -0.50 to +1.23
    against a typical absolute delta of 0.31. A Pearson score does not see a
    constant, but a SIGN score sees nothing else, so the sign column of any
    earlier run is void.
    """
    rep = pc.replicate(inputs, batch)
    with torch.no_grad():
        h = pc.final_residual(model, lens_model, rep, graph=False)
    return float(pc.fp32_contrast(lens_model, h, pos_id, neg_id)[0])


def grad_and_act(
    model: Any, lens_model: Any, inputs: dict, layer: int, block: list[int],
    pos_id: int, neg_id: int, answer_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gradient and the activation at the patch positions, [n, d] each."""
    grads, acts, _ = pa.attribute_pair(
        model, lens_model, inputs, [layer], pos_id, neg_id, answer_pos)
    return grads[layer][0, block, :], acts[layer][0, block, :]


def cheap_map(g: torch.Tensor, h: torch.Tensor) -> np.ndarray:
    """The production map: grad x CENTERED activation."""
    return pa.grad_x_activation(g, pa.center_over_patches(h)).cpu().numpy()


def taylor_map(
    g: torch.Tensor, h: torch.Tensor, donor_h: torch.Tensor,
) -> np.ndarray:
    """First-order prediction of the swap: <g_p, h_donor,p - h_p>.

    This is the fair test of the GRADIENT. The production map dots the
    gradient with the patch's own centered content, while a swap moves the
    patch along `h_donor - h`. Those are different vectors, so the 2 maps can
    disagree even when the gradient is exact.

    Reading the pair of scores:

    | Taylor predicts the swap | production map predicts the swap | Meaning |
    |---|---|---|
    | yes | no | the gradient is fine, the SUMMARY is wrong |
    | no | no | the gradient does not survive a finite step here |
    """
    delta = donor_h.to(g.dtype) - h
    return (g.float() * delta.float()).sum(-1).cpu().numpy()


def lens_readout(
    lens: Any, lens_model: Any, h: torch.Tensor, layer: int,
    pos_id: int, neg_id: int,
) -> np.ndarray:
    """What each patch is disposed to make the model say, as pos minus neg.

    The transport is linear, but `unembed` is not: it applies the final norm
    and the tanh softcap. So a swap readout must transport BOTH residuals and
    subtract the 2 contrasts. It cannot transport the difference.
    """
    logits = lens_model.unembed(lens.transport(h.float(), layer)).float()
    return (logits[:, pos_id] - logits[:, neg_id]).cpu().numpy()


def lens_maps(
    lens: Any, lens_model: Any, h: torch.Tensor, donor_h: torch.Tensor,
    layer: int, pos_id: int, neg_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(the map we would render, the map that predicts a swap).

    `own` is the analogue of the production map: each patch scored on its
    own. `delta` is the analogue of `taylor_map`: what the readout does when
    the patch becomes the donor's. Only `delta` is comparable with a causal
    map, because a causal map measures a CHANGE.
    """
    own = lens_readout(lens, lens_model, h, layer, pos_id, neg_id)
    don = lens_readout(lens, lens_model, donor_h, layer, pos_id, neg_id)
    return own, don - own


def mean_grad_maps(
    mg: dict, layer: int, h: torch.Tensor, donor_h: torch.Tensor,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Swap prediction from a CONSTANT direction, `<gbar_p, dh_p>`.

    `taylor` uses the local gradient. These 2 arms use a gradient averaged
    over held-out images, so they are constants exactly as a fitted lens is.
    They isolate the question the lens arms cannot: does an input-independent
    direction predict a swap at all?

    | Arm | Direction | Assumes |
    |---|---|---|
    | `meang` | one `[d]` vector | every patch shares a direction |
    | `meangp` | one `[n, d]` block | each patch POSITION has a direction |
    """
    if mg is None or layer not in mg.get("global", {}):
        return None, None
    glob = mg["global"][layer].to(h.device)
    per_pos = mg["per_pos"][layer].to(h.device)
    if per_pos.shape[0] != h.shape[0]:
        # A different patch count means the 2 runs used different geometry.
        # Silently reusing it would compare unrelated positions.
        return taylor_map(glob, h, donor_h), None
    return taylor_map(glob, h, donor_h), taylor_map(per_pos, h, donor_h)


def agreement(causal: np.ndarray, cheap: np.ndarray, k: int = 20) -> dict:
    """Do the 2 maps rank the same patches?"""
    from scipy.stats import pearsonr, spearmanr

    ok = np.isfinite(causal) & np.isfinite(cheap)
    c, m = causal[ok], cheap[ok]
    if c.size < 3 or np.std(c) == 0 or np.std(m) == 0:
        return {"pearson": np.nan, "spearman": np.nan,
                "sign_agree": np.nan, "top_overlap": np.nan}
    top_c = set(np.argsort(-np.abs(c))[:k].tolist())
    top_m = set(np.argsort(-np.abs(m))[:k].tolist())
    return {
        "pearson": float(pearsonr(c, m)[0]),
        "spearman": float(spearmanr(c, m)[0]),
        "sign_agree": float(np.mean(np.sign(c) == np.sign(m))),
        "top_overlap": len(top_c & top_m) / k,
    }


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, layers: list[int], batch: int, out_dir: Path, device: str,
    lens: Any = None, mean_grad: Optional[dict] = None, seed: int = 777,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    store = []
    for r in sg.load_single_rows(n_images, case, seed=seed):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        seq_len = int(inputs["input_ids"].shape[1])
        block = [int(x) for x in
                 sw.image_token_blocks(inputs, image_token_id, expect=1)[0]]
        per = {}
        c0 = None
        for layer in layers:
            c0, h = psw.contrast_and_residual(
                model, lens_model, inputs, layer, pos_id, neg_id)
            per[layer] = h[block, :]
        store.append({**r, "inputs": inputs, "block": block,
                      "answer_pos": seq_len - 1, "c": c0, "h": per})
    log(f"{case}: {len(store)} images, c range "
        f"{min(s['c'] for s in store):+.2f} to {max(s['c'] for s in store):+.2f}")

    order = sorted(range(len(store)), key=lambda i: store[i]["c"])
    pairs = [(order[i], order[-1 - i]) for i in range(len(order) // 2)]
    rows: list[dict] = []
    maps: list[dict] = []
    for t_i, d_i in pairs:
        tgt, don = store[t_i], store[d_i]
        gap = don["c"] - tgt["c"]
        if abs(gap) < psw.MIN_GAP:
            continue
        for layer in layers:
            whole = psw.swapped_contrast(
                model, lens_model, tgt["inputs"], layer, tgt["block"],
                don["h"][layer], pos_id, neg_id)
            c0 = batched_baseline(model, lens_model, tgt["inputs"],
                                  pos_id, neg_id, batch=batch)
            cm = causal_map(model, lens_model, tgt["inputs"], layer,
                            tgt["block"], don["h"][layer], c0,
                            pos_id, neg_id, batch=batch)
            g, h = grad_and_act(model, lens_model, tgt["inputs"], layer,
                                tgt["block"], pos_id, neg_id,
                                tgt["answer_pos"])
            gm = cheap_map(g, h)
            tm = taylor_map(g, h, don["h"][layer])
            a = agreement(cm, gm)
            t = agreement(cm, tm)
            ct = agreement(gm, tm)
            meang, meangp = mean_grad_maps(
                mean_grad, layer, h, don["h"][layer])
            if meang is not None:
                extra_mg = {f"meang_{k}": v
                            for k, v in agreement(cm, meang).items()}
                if meangp is not None:
                    extra_mg.update({f"meangp_{k}": v for k, v
                                     in agreement(cm, meangp).items()})
            else:
                extra_mg = {}
            lens_own = lens_delta = None
            extra: dict = {}
            if lens is not None and layer in lens.jacobians:
                lens_own, lens_delta = lens_maps(
                    lens, lens_model, h, don["h"][layer], layer,
                    pos_id, neg_id)
                extra = {
                    **{f"lens_{k}": v
                       for k, v in agreement(cm, lens_own).items()},
                    **{f"lensd_{k}": v
                       for k, v in agreement(cm, lens_delta).items()}}
            rows.append({
                "case": case, "layer": layer, "target": tgt["image_id"],
                "donor": don["image_id"],
                "all_recovery": (whole - tgt["c"]) / gap,
                "c0_batched": c0,
                "c0_batch1": tgt["c"],
                "causal_sum": float(cm.sum()),
                "taylor_sum": float(tm.sum()),
                "whole_delta": float(whole - tgt["c"]),
                "causal_max_abs": float(np.max(np.abs(cm))),
                "causal_top5pct": float(
                    np.sum(np.sort(np.abs(cm))[-13:]) / np.sum(np.abs(cm))),
                **a,
                **{f"taylor_{k}": v for k, v in t.items()},
                **{f"cheap_vs_taylor_{k}": v for k, v in ct.items()},
                **extra_mg,
                **extra})
            maps.append({
                "layer": layer, "target": tgt["image_id"],
                "causal": cm.tolist(), "cheap": gm.tolist(),
                "taylor": tm.tolist(),
                "lens_own": None if lens_own is None else lens_own.tolist(),
                "lens_delta": (None if lens_delta is None
                               else lens_delta.tolist()),
                "meang": None if meang is None else meang.tolist(),
                "meangp": None if meangp is None else meangp.tolist()})
    df = pd.DataFrame(rows)
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "agreement.parquet", index=False)
    pd.DataFrame(maps).to_parquet(case_dir / "maps.parquet", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    extra = {}
    for name in ("lens_pearson", "lens_sign_agree",
                 "lensd_pearson", "lensd_sign_agree", "lensd_top_overlap"):
        if name in df.columns:
            extra[name] = (name, "mean")
    return df.groupby("layer").agg(
        **extra,
        n=("pearson", "size"),
        all_recovery=("all_recovery", "mean"),
        pearson=("pearson", "mean"),
        spearman=("spearman", "mean"),
        sign_agree=("sign_agree", "mean"),
        top_overlap=("top_overlap", "mean"),
        t_pearson=("taylor_pearson", "mean"),
        t_spearman=("taylor_spearman", "mean"),
        t_sign=("taylor_sign_agree", "mean"),
        t_top=("taylor_top_overlap", "mean"),
        cheap_vs_taylor=("cheap_vs_taylor_pearson", "mean"),
        causal_max_abs=("causal_max_abs", "mean"),
        causal_top5pct=("causal_top5pct", "mean"),
    ).reset_index()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--seed", type=int, default=777,
                    help="which images the run reads. 777 matches the "
                         "jlens ladder. 20260821 matches causal_maps "
                         "and deletion_curve.")
    ap.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lens", help=("A fitted lens checkpoint. Adds 2 arms: "
                                    "the readout of each patch, and how a "
                                    "swap moves it. Only fitted layers get "
                                    "them."))
    ap.add_argument("--mean-grad", help=(
        "A mean_grad.pt from research.monocle.mean_gradient. Adds 2 CONSTANT-direction "
        "arms that work at ANY layer, including the ones no lens is fitted "
        "at."))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar

    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = None
    if args.lens:
        lens = jlens_read.load_lens(args.lens)
        log(f"lens {args.lens} fitted at {lens.source_layers}")
        missing = [l for l in args.layers if l not in lens.jacobians]
        if missing:
            log(f"  WARNING: no lens arm at {missing}; those layers are not "
                f"fitted")
    mean_grad = None
    if args.mean_grad:
        mean_grad = torch.load(args.mean_grad, map_location="cpu",
                               weights_only=False)
        log(f"mean-grad {args.mean_grad} fitted at {mean_grad['layers']} "
            f"on {mean_grad['n_images']} images of {mean_grad['case']}")
        if mean_grad["case"] not in args.cases:
            log(f"  WARNING: it was fitted on {mean_grad['case']!r}, and this "
                f"run reads {args.cases}. A cross-case direction is a "
                f"different experiment; say so in the write-up.")
    for case in args.cases:
        df = run_case(case, proc, model, lens_model, tmpl,
                      n_images=args.n_images, layers=sorted(set(args.layers)),
                      batch=args.batch, out_dir=out_dir, device=args.device,
                      lens=lens, mean_grad=mean_grad, seed=args.seed)
        s = summarize(df)
        log("=" * 78)
        log(f"CASE {case} — does the 1-pass map predict the causal map?")
        log("  PRODUCTION MAP vs the causal map")
        log("  layer   n  all_rec  pearson  spearman  sign  top20")
        for _, r in s.iterrows():
            log(f"  {int(r.layer):>5} {int(r.n):>3}  {r.all_recovery:>+7.3f}  "
                f"{r.pearson:>+7.3f}  {r.spearman:>+8.3f}  "
                f"{r.sign_agree:>4.2f}  {r.top_overlap:>5.2f}")
        log("")
        log("  TAYLOR PREDICTION vs the causal map — is the GRADIENT sound?")
        log("  layer   n  pearson  spearman  sign  top20  cheap~taylor")
        for _, r in s.iterrows():
            log(f"  {int(r.layer):>5} {int(r.n):>3}  {r.t_pearson:>+7.3f}  "
                f"{r.t_spearman:>+8.3f}  {r.t_sign:>4.2f}  {r.t_top:>5.2f}  "
                f"{r.cheap_vs_taylor:>+12.3f}")
        best = s.loc[s["all_recovery"].idxmax()]
        if float(best["all_recovery"]) < WINDOW_FLOOR:
            raise RuntimeError(
                f"no layer recovered more than {float(best['all_recovery']):.3f}, "
                f"below {WINDOW_FLOOR}. None of these layers is inside the "
                "causal window, so an agreement score here means nothing.")
        (out_dir / case / "summary.json").write_text(
            json.dumps(s.to_dict(orient="records"), indent=2))
    log("causal patch map complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
