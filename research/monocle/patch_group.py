"""Grouped removal at the production budget (the fallback of the budget probe).

The budget probe (`patch_budget`) coarsens the picture the model sees. This
module keeps the production input (280 soft tokens, the 16 x 16 grid) and
removes ``k`` x ``k`` blocks of cells at once, so the map is coarse but the
judgment is the production judgment. A 2 x 2 block covers the same 96
pixels as one cell of the 70-token grid.

Per image and group size: `pab.ablation_map` with ``group=k``, batch-padded,
the zero measured at the batch size.

Outputs, under ``<out-dir>/<case>/``:

| File | Content |
|---|---|
| ``maps.parquet`` | per image and group size: the block deltas of ``c``, row-major |
| ``summary.json`` | per group size: mean block delta, mean absolute delta, seconds |

Usage:
    sbatch --time=01:30:00 research/monocle/monocle.sub monocle.patch_group \\
        --case restaurants --groups 2,4,8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/patch_group")
GROUPS_DEFAULT = "2,4,8"
BATCH_DEFAULT = 12


def log(msg: str) -> None:
    print(f"[patch-group] {msg}", flush=True)


def run(args: argparse.Namespace) -> int:
    groups = parse_int_list(args.groups)
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
    build = ss.make_build_inputs(proc, tmpl, args.case,
                                 causal_image=mask_mode != "bidirectional")
    ss.install_image_mask(model, image_token_id, mask_mode)
    log(f"image mask: {mask_mode}")
    log(f"{args.case}: words={ss.CASE_WORDS[args.case]} groups={groups} "
        f"batch={args.batch} images={len(eval_paths)}")

    records: list[dict] = []
    done: set = set()
    if maps_path.exists() and not args.smoke:
        prev = pd.read_parquet(maps_path)
        records = prev.to_dict("records")
        done = {(r["image_path"], int(r["group"])) for r in records}
        log(f"resuming: {len(done)} image-group pairs done")

    n_done = 0
    for path in eval_paths:
        inputs = build(path)
        slots = pab.slots_of(inputs, image_token_id)
        zero_rows = pab.ablation_batch(inputs, slots, [[]] * args.batch,
                                       image_token_id)
        zero = float(pab.contrast(model, lens_model, zero_rows,
                                  tok_pos, tok_neg)[0])
        for k in groups:
            if (path, k) in done:
                continue
            delta = pab.ablation_map(
                model, lens_model, inputs, slots, zero, tok_pos, tok_neg,
                batch=args.batch, group=k, image_token_id=image_token_id)
            records.append({
                "image_path": path, "group": int(k), "n_blocks": len(delta),
                "n_cells": len(slots), "zero": zero,
                "values": delta.tolist(),
            })
            n_done += 1
        if n_done and (n_done % (5 * len(groups)) == 0 or args.smoke):
            pd.DataFrame(records).to_parquet(maps_path, index=False)
            el = time.time() - t0
            log(f"  {n_done} maps | {el:.0f}s | {el / n_done:.0f}s per map")
    df = pd.DataFrame(records)
    df.to_parquet(maps_path, index=False)

    summary = {}
    for k, g in df.groupby("group"):
        vals = np.stack([np.asarray(v) for v in g["values"]])
        summary[str(int(k))] = {
            "n_images": int(len(g)), "n_blocks": int(vals.shape[1]),
            "mean_delta": round(float(vals.mean()), 4),
            "mean_abs_delta": round(float(np.abs(vals).mean()), 4),
            "mean_within_sd": round(float(vals.std(1).mean()), 4),
        }
        log(f"group {k}: {json.dumps(summary[str(int(k))])}")
    (out_dir / "summary.json").write_text(json.dumps({
        "case": args.case, "words": ss.CASE_WORDS[args.case],
        "groups": groups, "batch": args.batch,
        "per_group": summary, "seconds": round(time.time() - t0)}, indent=2))
    log(f"wrote {out_dir} ({time.time() - t0:.0f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--groups", default=GROUPS_DEFAULT)
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
