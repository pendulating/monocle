"""Gate B: does any cheap map predict an INPUT-level patch ablation?

Why a second gate
-----------------
`patch_map_causal` is Gate A. Its ground truth is a DONOR RESIDUAL SWAP at
layer L. The overnight run of 2026-08-23 retired that map as a statement about
patch CONTENT, for a measured reason: a layer-20 residual swap and an input
edit of the same cell agree at r = +0.09. See
`outputs/_monocle/overnight/RESEARCH_LOG.md`, 01:10.

The 3 questions are not one question:

| Map | Asks |
|---|---|
| donor residual swap | what if this residual were another photograph's? |
| **input ablation** | **what does the content of this patch contribute?** |
| `lens_own` | what is this patch disposed to make the model say? |

`lens_own` is the map a figure RENDERS. Gate A scored it against the donor
map, where it read about zero. That was the wrong ground truth for its
question. This module scores it against the input map instead.

The instruments
---------------
Ablation drives the patch content to ABSENT, so the first-order prediction of
its effect is the gradient dotted with the patch's own residual:

| Arm | Map | Direction |
|---|---|---|
| `gxa` | `<g_p, h_p>` | local |
| `cheap` | `<g_p, centre(h_p)>` | local, the production summary |
| `meang_own` | `<gbar, h_p>` | constant, one vector |
| `meangp_own` | `<gbar_p, h_p>` | constant, one vector per position |
| `lens_own` | `readout(J h_p)` | constant, the fitted lens |

Sign: the truth is the CONTRIBUTION, `-(c_ablated - c0)`. A patch that holds up
the positive label makes `c` fall when it goes, so its contribution is
positive. `<g_p, h_p>` carries the same sign, because ablation moves the
residual along `0 - h_p`.

Usage:
    python -m research.monocle.input_gate --case subway_safety --n-images 16 \
        --layers 0 4 8 12 16 20 24 --mean-grad <mean_grad.pt>
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_causal as pc  # noqa: E402
from research.monocle import single as sg  # noqa: E402

DEFAULT_LAYERS = (0, 4, 8, 12, 16, 20, 24)
DEFAULT_OUT = str(REPO / "outputs/_monocle/input_gate")


def log(m: str) -> None:
    print(f"[input-gate] {m}", flush=True)


def score(truth: np.ndarray, cand: Optional[np.ndarray], k: int = 50) -> dict:
    """Agreement on the top `k` cells by absolute truth, and overall."""
    from scipy.stats import pearsonr, spearmanr

    nan = {"pearson": np.nan, "spearman": np.nan, "sign": np.nan,
           "top_pearson": np.nan, "top_sign": np.nan}
    if cand is None:
        return nan
    a = np.asarray(truth, float).ravel()
    b = np.asarray(cand, float).ravel()
    if a.size != b.size:
        return nan
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return nan
    out = {"pearson": float(pearsonr(a, b)[0]),
           "spearman": float(spearmanr(a, b)[0]),
           "sign": float(np.mean(np.sign(a) == np.sign(b)))}
    if a.size >= k:
        idx = np.argsort(-np.abs(a))[:k]
        ta, tb = a[idx], b[idx]
        if ta.std() > 0 and tb.std() > 0:
            out["top_pearson"] = float(pearsonr(ta, tb)[0])
            out["top_sign"] = float(np.mean(np.sign(ta) == np.sign(tb)))
        else:
            out["top_pearson"] = out["top_sign"] = np.nan
    else:
        out["top_pearson"] = out["top_sign"] = np.nan
    return out


def instrument_maps(
    g: torch.Tensor, h: torch.Tensor, *, layer: int,
    mean_grad: Optional[dict], lens: Any, lens_model: Any,
    pos_id: int, neg_id: int,
) -> dict[str, Optional[np.ndarray]]:
    """Every candidate map for one image at one layer."""
    from research.monocle import patch_map_causal as pmc

    out: dict[str, Optional[np.ndarray]] = {
        "gxa": (g.float() * h.float()).sum(-1).cpu().numpy(),
        "cheap": pa.grad_x_activation(
            g, pa.center_over_patches(h)).cpu().numpy(),
        "meang_own": None, "meangp_own": None, "lens_own": None,
    }
    if mean_grad is not None and layer in mean_grad.get("global", {}):
        gb = mean_grad["global"][layer].to(h.device)
        out["meang_own"] = (gb.float() * h.float()).sum(-1).cpu().numpy()
        gp = mean_grad["per_pos"][layer].to(h.device)
        if gp.shape[0] == h.shape[0]:
            out["meangp_own"] = (gp.float() * h.float()).sum(-1).cpu().numpy()
    if lens is not None and layer in lens.jacobians:
        out["lens_own"] = pmc.lens_readout(
            lens, lens_model, h, layer, pos_id, neg_id)
    return out


def load_truth(path: str, case: str, group: int,
               stored_group: int = 1) -> dict[str, np.ndarray]:
    """The input ablation maps that a previous run measured.

    The 256-patch ablation is the costly part of this gate. When a run already
    measured it on the same images, we read it and we do not measure it again.

    Two layouts are accepted. `causal_maps` writes `image_id` and `values`.
    This module writes `target` and `causal`. Both hold the CONTRIBUTION, so
    no sign change is needed.

    Warning: a pool of the truth SUMS the cells, and that assumes the effects
    add. They do not. So pass `stored_group` when the file already holds the
    grain you want, and the pool does not run.
    """
    from research.monocle import patch_map_causal as pmc

    df = pd.read_parquet(path)
    id_col = "image_id" if "image_id" in df.columns else "target"
    val_col = "values" if "values" in df.columns else "causal"
    if id_col not in df.columns or val_col not in df.columns:
        raise ValueError(f"{path} carries no truth columns")
    out: dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        key = str(r[id_col])
        if key in out:
            continue
        arr = np.asarray(r[val_col], dtype=float).ravel()
        if arr.size < 2:
            continue
        factor = group // stored_group
        if factor > 1:
            arr = pmc.pool_by_groups(arr, factor)
        out[key] = arr
    return out


def run_case(
    case: str, *, proc: Any, model: Any, lens_model: Any, tmpl: str,
    n_images: int, layers: list[int], batch: int, group: int,
    device: str, lens: Any, mean_grad: Optional[dict], out_dir: Path,
    seed: int = 777, truth_from: Optional[str] = None,
    stored_group: int = 1, paraphrase: Optional[str] = None,
    label_pair: Optional[tuple] = None,
    save_acts: Optional[list[int]] = None,
    prompt_case: Optional[str] = None,
) -> pd.DataFrame:
    from PIL import Image
    from research.monocle import patch_map_causal as pmc

    tok = proc.tokenizer
    # The CROSS-QUESTION control. The images and the truth stay with `case`,
    # but the question and the 2 answer tokens come from `prompt_case`. An
    # instrument that ranks the same cells under the WRONG question does not
    # read the judgment; it reads the picture.
    ask = prompt_case or case
    labels = ({"positive": label_pair[0], "negative": label_pair[1]}
              if label_pair else {})
    pos_id, neg_id = sg.contrast_ids(tok, ask, labels or None)
    image_token_id = int(model.config.image_token_id)
    if prompt_case:
        log(f"the question comes from {ask}, the images from {case}")
    # The PARAPHRASE control. The judgment and the 2 answer tokens stay, and
    # only the wording changes. A reader of the judgment must KEEP its
    # separation here. If it does not, the cross-question drop says only that
    # any change of the prompt hurts.
    overrides = dict(labels)
    if paraphrase:
        overrides["question"] = paraphrase
    if label_pair:
        log(f"the answer words are {label_pair[0]!r} and "
            f"{label_pair[1]!r}")
    overrides = overrides or None
    if paraphrase:
        log(f"the question is reworded: {paraphrase!r}")
    rows: list[dict] = []
    maps: list[dict] = []
    t0 = time.time()
    acts_out: dict[str, dict] = {}
    stored = (load_truth(truth_from, case, group, stored_group)
              if truth_from else None)
    if stored is not None:
        log(f"the truth comes from {truth_from} ({len(stored)} images); "
            f"no ablation runs")

    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, ask,
                                        overrides=overrides,
                                        device=device)
        slots = pab.slots_of(inputs, image_token_id)
        seq_len = int(inputs["input_ids"].shape[1])
        c0 = pmc.batched_baseline(model, lens_model, inputs,
                                  pos_id, neg_id, batch=batch)
        if stored is not None:
            if r["image_id"] not in stored:
                log(f"  skip {r['image_id']}: the stored truth omits it")
                continue
            truth = stored[r["image_id"]]
        else:
            # The CONTRIBUTION: a patch that holds up the positive label makes
            # `c` fall when it goes, so we negate the delta.
            truth = -pab.ablation_map(
                model, lens_model, inputs, slots, c0, pos_id, neg_id,
                batch=batch, group=group, image_token_id=image_token_id)

        grads, acts, _ = pa.attribute_pair(
            model, lens_model, inputs, layers, pos_id, neg_id, seq_len - 1)
        for layer in layers:
            g = grads[layer][0, slots, :]
            h = acts[layer][0, slots, :]
            cand = instrument_maps(
                g, h, layer=layer, mean_grad=mean_grad, lens=lens,
                lens_model=lens_model, pos_id=pos_id, neg_id=neg_id)
            if group > 1:
                cand = {k: (None if v is None else
                            pmc.pool_by_groups(v, group))
                        for k, v in cand.items()}
            row = {"case": case, "ask": ask,
                   "reworded": bool(paraphrase), "layer": layer,
                   "target": r["image_id"],
                   "c0": c0, "n_cells": int(truth.size),
                   "truth_absmax": float(np.max(np.abs(truth))),
                   "truth_sum": float(truth.sum())}
            for name, m in cand.items():
                row.update({f"{name}_{k}": v
                            for k, v in score(truth, m).items()})
            rows.append(row)
            maps.append({"layer": layer, "target": r["image_id"],
                         "causal": truth.tolist(),
                         **{k: (None if v is None else np.asarray(v).tolist())
                            for k, v in cand.items()}})
        if save_acts:
            # The residual itself, so a probe can ask whether the causal
            # effect is linearly readable from it at all.
            #
            # The gradient goes with it, and it must. The image tokens come
            # BEFORE the question in the prompt, so a causal decoder gives
            # the SAME residual under any question. A probe on `h` alone is
            # thus question-blind, and its wrong-question control is 0 by
            # construction. Only `g` carries the question, because it flows
            # back from the answer position.
            acts_out[str(r["image_id"])] = {
                "truth": np.asarray(truth, dtype=np.float32),
                **{f"h{ly}": acts[ly][0, slots, :].float().cpu().numpy()
                   for ly in save_acts if ly in acts},
                **{f"g{ly}": grads[ly][0, slots, :].float().cpu().numpy()
                   for ly in save_acts if ly in grads},
            }
        log(f"  {i + 1}/{n_images} {r['image_id']} c0={c0:+.2f} "
            f"|truth|max={np.max(np.abs(truth)):.3f} {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "agreement.parquet", index=False)
    pd.DataFrame(maps).to_parquet(case_dir / "maps.parquet", index=False)
    if acts_out:
        torch.save({"acts": acts_out, "layers": list(save_acts), "case": case},
                   case_dir / "acts.pt")
        log(f"saved residuals for {len(acts_out)} images at {list(save_acts)}")
    return df


ARMS = ("gxa", "cheap", "meang_own", "meangp_own", "lens_own")


def report(df: pd.DataFrame) -> None:
    log("=" * 76)
    log("Gate B: agreement with the INPUT ablation map (mean over images)")
    log(f"{'layer':>6} {'n':>3} " + " ".join(f"{a:>13}" for a in ARMS))
    for layer, sub in df.groupby("layer"):
        cells = []
        for a in ARMS:
            col = f"{a}_pearson"
            v = sub[col].mean() if col in sub else np.nan
            cells.append("     --      " if not np.isfinite(v)
                         else f"{v:>+13.3f}")
        log(f"{int(layer):>6} {len(sub):>3} " + " ".join(cells))


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar
    from research.monocle import jlens_read

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="subway_safety",
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--save-acts", nargs="+", type=int, default=None,
                    metavar="LAYER",
                    help="keep the patch residuals at these layers. A probe then asks whether the causal effect is linearly readable.")
    ap.add_argument("--prompt-case", default=None,
                    help="ask THIS case's question about the other case's "
                         "images. It is the judgment-free control: a good "
                         "instrument must lose its ranking here.")
    ap.add_argument("--labels", nargs=2, default=None,
                    metavar=("POSITIVE", "NEGATIVE"),
                    help="answer words that replace the case pair")
    ap.add_argument("--paraphrase", default=None,
                    help="reword the question, keep the judgment and "
                         "the 2 answer tokens")
    ap.add_argument("--truth-group", type=int, default=1,
                    help="the grain the stored truth already holds. "
                         "1 means 1 patch for each cell.")
    ap.add_argument("--truth-from", default=None,
                    help="a parquet that already holds the input "
                         "ablation maps. It saves 256 forward "
                         "passes for each image.")
    ap.add_argument("--seed", type=int, default=777,
                    help="which images the run reads. 777 matches the "
                         "jlens ladder. 20260821 matches causal_maps "
                         "and deletion_curve.")
    ap.add_argument("--layers", nargs="+", type=int,
                    default=list(DEFAULT_LAYERS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--group", type=int, default=1,
                    help="1 gives 256 cells; 2 gives 64 and runs 4x faster.")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--mean-grad", default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    layers = sorted(set(args.layers))
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens) if args.lens else None
    if lens is not None:
        log(f"lens fitted at {lens.source_layers}")
    mean_grad = None
    if args.mean_grad:
        mean_grad = torch.load(args.mean_grad, map_location="cpu",
                               weights_only=False)
        log(f"mean-grad fitted at {mean_grad['layers']} on "
            f"{mean_grad['n_images']} images of {mean_grad['case']}")

    out_dir = Path(args.out_dir)
    df = run_case(args.case, proc=proc, model=model, lens_model=lens_model,
                  tmpl=tmpl, n_images=args.n_images, layers=layers,
                  batch=args.batch, group=args.group, device=args.device,
                  lens=lens, mean_grad=mean_grad, out_dir=out_dir,
                  seed=args.seed, truth_from=args.truth_from,
                  stored_group=args.truth_group,
                  paraphrase=args.paraphrase,
                  label_pair=(tuple(args.labels) if args.labels
                              else None),
                  save_acts=args.save_acts, prompt_case=args.prompt_case)
    if df.empty:
        raise SystemExit(
            "no image survived. The stored truth in --truth-from names "
            "another sample. Match --seed to the run that measured it.")
    report(df)
    (out_dir / args.case / "summary.json").write_text(json.dumps(
        df.groupby("layer").mean(numeric_only=True).reset_index()
        .to_dict(orient="records"), indent=1))
    log("gate B complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
