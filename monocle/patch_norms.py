"""Why one patch dominates a deep attribution map.

`a[patch] = <g, h>` with `g = dc/dh_l[patch]`. A cell can be bright because the
GRADIENT is large (the decision truly routes through it) or because the
ACTIVATION is large (a high-norm register position that any activation-weighted
score inflates). The two have opposite consequences and the map cannot tell
them apart.

This module records the decomposition:

    ||g||   how much c moves for a unit change at this position
    ||h||   how large the residual is there
    cos     their alignment
    a       the product, = ||g|| ||h|| cos

Read `cos` first. A register shows large ||h||, ordinary ||g||, and a cos near
zero -- it is bright by magnitude, not by relevance.

Also records the sequence geometry: where each image block sits, what
`mm_token_type_ids` marks as vision, and which positions bound the block. The
last patch of a block is the final position before the mask turns causal, so
it is the one to check.

Layer 47 is included even though `dc/dh_47[patch]` is identically zero -- `c`
reads only `h_47[answer]`. Its ||h|| still says whether the last block writes
something large at that position.
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

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import answer_tokens as at  # noqa: E402
from monocle import canonical, extract  # noqa: E402
from monocle import jlens_read  # noqa: E402
from monocle import patch_attribution as pa  # noqa: E402
from monocle import safety_workspace as sw  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/patch_norms"
LAYERS = [30, 36, 40, 42, 44, 45, 46, 47]


def log(m: str) -> None:
    print(f"[patch-norms] {m}", flush=True)


def decompose(
    g: torch.Tensor, h: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(||g||, ||h||, cos) per row. Rows are patch positions."""
    g, h = g.float(), h.float()
    gn = g.norm(dim=-1)
    hn = h.norm(dim=-1)
    denom = (gn * hn).clamp_min(1e-20)
    cos = (g * h).sum(-1) / denom
    return gn.cpu().numpy(), hn.cpu().numpy(), cos.cpu().numpy()


def block_geometry(inputs: dict, blocks: list[torch.Tensor]) -> dict:
    """Where the image blocks sit and what the mask treats as vision."""
    ids = inputs["input_ids"][0].tolist()
    out = {"seq_len": len(ids)}
    for i, b in enumerate(blocks):
        pos = b.tolist()
        out[f"block{i}_first"] = pos[0]
        out[f"block{i}_last"] = pos[-1]
        out[f"block{i}_len"] = len(pos)
        out[f"block{i}_token_before"] = ids[pos[0] - 1] if pos[0] else None
        out[f"block{i}_token_after"] = (
            ids[pos[-1] + 1] if pos[-1] + 1 < len(ids) else None)
    mm = inputs.get("mm_token_type_ids")
    if mm is not None:
        v = mm[0].tolist()
        out["mm_vision_count"] = int(sum(1 for x in v if x in (1, 2)))
        out["mm_values"] = sorted(set(v))
    return out


def run_case(
    case: str, kind: str, proc: Any, model: Any, tmpl: str, lens: Any,
    lens_model: Any, *, n_pairs: int, layers: list[int], out_dir: Path,
    device: str, seed: int,
) -> dict:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = canonical.build_conditions(case, kind, ["prod"])["prod"]
    label_first, _, _ = canonical.label_classes(proc.tokenizer, cfg)
    pos_id, neg_id, _, _ = pa.contrast_token_ids("judgment", label_first)

    image_token_id = int(model.config.image_token_id)
    n_suffix = len(proc.tokenizer.encode(
        canonical.FORCE_PREFIX, add_special_tokens=False))
    rows, geom_once = [], None
    t0 = time.time()
    for i, row in enumerate(sw.load_pairs(n_pairs, case, kind, seed=seed)):
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]
        inputs = canonical.build_pair_inputs(
            proc, tmpl, imgs, row, cfg, device=device)
        seq_len = int(inputs["input_ids"].shape[1])
        answer_pos = sw.read_positions(seq_len, n_suffix)["label"]
        blocks = sw.image_token_blocks(inputs, image_token_id)
        if geom_once is None:
            geom_once = block_geometry(inputs, blocks)
            log(f"{case}: geometry {json.dumps(geom_once)}")

        grads, acts, c = pa.attribute_pair(
            model, lens_model, inputs, sorted(set(layers) | {pa.FINAL_LAYER}),
            pos_id, neg_id, answer_pos)
        for layer in sorted(set(layers)):
            for si, slot in enumerate(("A", "B")):
                pos = blocks[si]
                h = acts[layer][0, pos, :] if layer in acts else None
                if h is None:
                    continue
                if layer in grads:
                    g = grads[layer][0, pos, :]
                else:  # the final block: dc/dh at a patch is exactly zero
                    g = torch.zeros_like(h)
                gn, hn, cos = decompose(g, h)
                a = pa.grad_x_activation(
                    g, pa.center_over_patches(h)).float().cpu().numpy()
                n = int(pos.numel())
                for p in range(n):
                    rows.append({
                        "case": case, "pair_id": row["pair_id"], "layer": layer,
                        "slot": slot, "patch_idx": p, "is_last": p == n - 1,
                        "is_first": p == 0, "grad_norm": float(gn[p]),
                        "act_norm": float(hn[p]), "cos": float(cos[p]),
                        "attrib": float(a[p]), "contrast_value": c,
                    })
        del grads, acts
        if (i + 1) % 5 == 0:
            log(f"{case}: {i + 1} pairs, {(time.time() - t0) / (i + 1):.1f} s/pair")

    df = pd.DataFrame(rows)
    df.to_parquet(case_dir / "patch_norms.parquet", index=False)
    summary = summarize(case, df, geom_once)
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize(case: str, df: pd.DataFrame, geom: Optional[dict]) -> dict:
    out = {"case": case, "n_pairs": int(df["pair_id"].nunique()),
           "geometry": geom, "by_layer": []}
    for layer, d in df.groupby("layer"):
        last = d[d.is_last]
        rest = d[~d.is_last]
        out["by_layer"].append({
            "layer": int(layer),
            "act_norm_last": float(last.act_norm.mean()),
            "act_norm_rest": float(rest.act_norm.mean()),
            "act_norm_ratio": float(last.act_norm.mean() / rest.act_norm.mean()),
            "grad_norm_last": float(last.grad_norm.mean()),
            "grad_norm_rest": float(rest.grad_norm.mean()),
            "grad_norm_ratio": float(
                last.grad_norm.mean() / rest.grad_norm.mean())
            if rest.grad_norm.mean() else float("nan"),
            "cos_last": float(last.cos.abs().mean()),
            "cos_rest": float(rest.cos.abs().mean()),
            "abs_attrib_last": float(last.attrib.abs().mean()),
            "abs_attrib_rest": float(rest.attrib.abs().mean()),
        })
    return out


def print_summary(s: dict) -> None:
    log("=" * 88)
    log(f"CASE {s['case']} — {s['n_pairs']} pairs.  'last' = the final patch "
        f"of each block")
    log(f"  {'layer':>5} {'||h|| last':>11} {'||h|| rest':>11} {'ratio':>7} "
        f"{'||g|| ratio':>12} {'|cos| last':>11} {'|cos| rest':>11} "
        f"{'|a| last':>9} {'|a| rest':>9}")
    for d in s["by_layer"]:
        log(f"  {d['layer']:>5} {d['act_norm_last']:>11.1f} "
            f"{d['act_norm_rest']:>11.1f} {d['act_norm_ratio']:>7.2f} "
            f"{d['grad_norm_ratio']:>12.2f} {d['cos_last']:>11.4f} "
            f"{d['cos_rest']:>11.4f} {d['abs_attrib_last']:>9.3f} "
            f"{d['abs_attrib_rest']:>9.3f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(canonical.CASES))
    ap.add_argument("--kind", default="proxy", choices=list(canonical.KINDS))
    ap.add_argument("--n-pairs", type=int, default=25)
    ap.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    ap.add_argument("--seed", type=int, default=pa.SEED)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lens", default=at.LENS_DEFAULT)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proc, model, tmpl = extract.load_model(args.model_dir, device=args.device)
    model.requires_grad_(False)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens)
    for case in args.cases:
        print_summary(run_case(
            case, args.kind, proc, model, tmpl, lens, lens_model,
            n_pairs=args.n_pairs, layers=args.layers, out_dir=out_dir,
            device=args.device, seed=args.seed))
    log("patch norms complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
