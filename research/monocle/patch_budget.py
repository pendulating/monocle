"""The removal map at each visual token budget (the object question).

At the production budget of 280 soft tokens a cell is a 48 by 48 pixel
square of a 768 by 768 picture, so one cell holds a texture or an edge,
not an object. Gemma-4 accepts smaller budgets, and each one gives the
model a coarser picture of the same photograph:

| ``max_soft_tokens`` | cells | grid | pixels per cell, of the 768 picture |
|---|---|---|---|
| 70 | 64 | 8 x 8 | 96 |
| 140 | 121 | 11 x 11 | 70 |
| 280 | 256 | 16 x 16 | 48 |

This module runs the same 60 eval images through each budget, records
the model's own contrast ``c`` at that budget, and draws the per-cell
removal map with `pab.ablation_map` (batch-padded, the zero measured at
the batch size). The maps of one image at 2 budgets are maps of 2
different forwards: the report compares them, and it compares each
smaller budget's judgment with the production one.

Outputs, under ``<out-dir>/<case>/``:

| File | Content |
|---|---|
| ``maps.parquet`` | per image and budget: ``zero`` (the contrast), the restricted argmax, the cell deltas |
| ``summary.json`` | per budget: mean contrast, positive share, agreement with 280, seconds |

Usage:
    sbatch --time=02:00:00 research/monocle/monocle.sub monocle.patch_budget \\
        --case restaurants --budgets 70,140,280
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

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import single_steer as ss  # noqa: E402
from research.monocle.jlens_steer import (  # noqa: E402
    load_model_and_lens,
    parse_int_list,
    split_calibration_eval,
)

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/patch_budget")
BUDGETS_DEFAULT = "70,140,280"
BATCH_DEFAULT = 12


def log(msg: str) -> None:
    print(f"[patch-budget] {msg}", flush=True)


def make_build(proc, tmpl, case: str, budget: int, device: str = "cuda:0",
               causal_image: bool = False):
    """The single-image input at one budget. The processor takes the budget
    as a per-call image kwarg."""
    from PIL import Image
    text = ss.render_single_chat(proc.tokenizer, tmpl, ss.user_text(case))

    def build(path: str) -> dict:
        img = Image.open(path).convert("RGB")
        out = dict(proc(text=[text], images=[[img]], max_soft_tokens=budget,
                        return_tensors="pt").to(device))
        return ss.strip_image_mask(out) if causal_image else out
    return build


def run(args: argparse.Namespace) -> int:
    budgets = parse_int_list(args.budgets)
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_path = out_dir / "maps.parquet"
    t0 = time.time()

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    paths = ss.load_images(args.case, args.kind)
    _, ev = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    eval_paths = [r["p"] for r in ev]
    mask_mode = ss.image_mask_mode(args)
    builds = {b: make_build(proc, tmpl, args.case, b,
                            causal_image=mask_mode != "bidirectional") for b in budgets}
    ss.install_image_mask(model, image_token_id, mask_mode)
    log(f"image mask: {mask_mode}")
    log(f"{args.case}: words={ss.CASE_WORDS[args.case]} budgets={budgets} "
        f"batch={args.batch} images={len(eval_paths)}")

    records: list[dict] = []
    done: set = set()
    if maps_path.exists() and not args.smoke:
        prev = pd.read_parquet(maps_path)
        records = prev.to_dict("records")
        done = {(r["image_path"], int(r["budget"])) for r in records}
        log(f"resuming: {len(done)} image-budget pairs done")

    n_done = 0
    for path in eval_paths:
        for b in budgets:
            if (path, b) in done:
                continue
            inputs = builds[b](path)
            slots = pab.slots_of(inputs, image_token_id)
            zero_rows = pab.ablation_batch(inputs, slots, [[]] * args.batch,
                                           image_token_id)
            c0 = pab.contrast(model, lens_model, zero_rows, tok_pos, tok_neg)
            zero = float(c0[0])
            if not np.allclose(c0, zero):
                raise RuntimeError("identical no-op rows disagree")
            delta = pab.ablation_map(
                model, lens_model, inputs, slots, zero, tok_pos, tok_neg,
                batch=args.batch, group=1, image_token_id=image_token_id)
            records.append({
                "image_path": path, "budget": int(b), "n_cells": len(slots),
                "seq_len": int(inputs["input_ids"].shape[1]),
                "zero": zero, "argmax_pos": bool(zero >= 0),
                "values": delta.tolist(),
            })
            n_done += 1
        if n_done and (n_done % (5 * len(budgets)) == 0 or args.smoke):
            pd.DataFrame(records).to_parquet(maps_path, index=False)
            el = time.time() - t0
            log(f"  {n_done} maps | {el:.0f}s | {el / n_done:.0f}s per map")
    df = pd.DataFrame(records)
    df.to_parquet(maps_path, index=False)

    summary = {}
    ref = df[df["budget"] == max(budgets)].set_index("image_path")
    for b, g in df.groupby("budget"):
        g = g.set_index("image_path")
        shared = g.index.intersection(ref.index)
        summary[str(int(b))] = {
            "n_images": int(len(g)),
            "n_cells": int(g["n_cells"].iloc[0]),
            "seq_len": int(g["seq_len"].iloc[0]),
            "mean_contrast": round(float(g["zero"].mean()), 3),
            "frac_pos": round(float(g["argmax_pos"].mean()), 3),
            "agree_with_max_budget": round(float(
                (g.loc[shared, "argmax_pos"] == ref.loc[shared, "argmax_pos"])
                .mean()), 3) if len(shared) else None,
            "r_contrast_with_max_budget": round(float(np.corrcoef(
                g.loc[shared, "zero"], ref.loc[shared, "zero"])[0, 1]), 3)
            if len(shared) > 2 else None,
        }
        log(f"budget {b}: {json.dumps(summary[str(int(b))])}")
    (out_dir / "summary.json").write_text(json.dumps({
        "case": args.case, "words": ss.CASE_WORDS[args.case],
        "budgets": budgets, "batch": args.batch,
        "per_budget": summary, "seconds": round(time.time() - t0)}, indent=2))
    log(f"wrote {out_dir} ({time.time() - t0:.0f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--budgets", default=BUDGETS_DEFAULT)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--causal-image", action="store_true",
                    help="strip mm_token_type_ids: a causal mask over the "
                         "image tokens (gemma-4-12b ablation)")
    ap.add_argument("--image-mask", choices=list(ss.IMAGE_MASK_MODES),
                    default=None,
                    help="how the image tokens attend to each other; "
                         "see single_steer.install_image_mask")
    ap.add_argument("--smoke", action="store_true", help="2 images")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.n_images = 2
        args.out_dir = str(Path(args.out_dir) / "smoke")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
