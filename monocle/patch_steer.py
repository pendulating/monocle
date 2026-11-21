"""Per-cell injection of the patch axis (step 5b of the steering work).

Step 5a (`single_steer --positions patches --arms axis`) fits the judgment
axis on the patch residuals and injects it at EVERY image token at once.
This module injects the same vector at ONE cell at a time. Each cell then
gets its own response: does the answer read the axis out of this cell?
The result is an injection map in the shape of the removal map of
`steer_coord`, and the two join on ``image_path`` and cell index.

Per image, block and condition (arm, dose): 256 forwards, in batches of
``batch`` identical no-op rows, the vector added at row b's own cell. The
zero is the same batch with no hook, so the bf16 batch-size offset of
`pab.ablation_map` cancels.

Outputs, under ``<out-dir>/<case>/``:

| File | Content |
|---|---|
| ``maps.parquet`` | per image, block, arm, dose: the 256 cell deltas of ``c`` and the contract flags |
| ``summary.json`` | per block, arm, dose: mean cell delta, mean absolute delta, contract share |

Usage:
    sbatch --time=04:30:00 monocle/monocle.sub monocle.patch_steer \\
        --case restaurants --layers 16,22
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

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import patch_ablate as pab  # noqa: E402
from monocle import patch_causal as pc  # noqa: E402
from monocle import single_steer as ss  # noqa: E402
from monocle.jlens_steer import (  # noqa: E402
    load_model_and_lens,
    parse_int_list,
    split_calibration_eval,
)

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/patch_steer")
VECTORS_DEFAULT = str(REPO / "outputs/_monocle/patch_axis")
CONDITIONS_DEFAULT = ("axis:-2", "axis:+2", "control_raw:+2")
BATCH_DEFAULT = 12


def log(msg: str) -> None:
    print(f"[patch-steer] {msg}", flush=True)


def parse_conditions(items: list[str]) -> list[tuple[str, float]]:
    out = []
    for it in items:
        arm, dose = it.split(":")
        out.append((arm, float(dose)))
    return out


def make_cell_hook(add_vec: torch.Tensor, positions: list[int]):
    """Add ``add_vec`` at row b's own position ``positions[b]``."""
    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        tensor = output[0] if is_tuple else output
        new = tensor.clone()
        add = add_vec.to(new.dtype).to(new.device)
        for b, p in enumerate(positions):
            new[b, p, :] = new[b, p, :] + add
        return (new, *tuple(output[1:])) if is_tuple else new
    return hook


def contrast_and_contract(
    model: Any, lens_model: Any, rows: dict, tok_pos: int, tok_neg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``c`` per row in float32, and whether the top token is a pole word."""
    with torch.no_grad():
        h = pc.final_residual(model, lens_model, rows, graph=False)
        c = pc.fp32_contrast(lens_model, h, tok_pos, tok_neg)
        logits = lens_model.unembed(h).float()
    top = logits.argmax(dim=-1)
    on = (top == tok_pos) | (top == tok_neg)
    return c.float().cpu().numpy(), on.cpu().numpy()


def cell_map(
    model: Any, lens_model: Any, rows: dict, slots: list[int], layer: int,
    add_vec: torch.Tensor, zero: float, tok_pos: int, tok_neg: int,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The delta of ``c`` and the contract flag when ``add_vec`` enters one
    cell at a time. ``rows`` is the batch of identical no-op rows."""
    n = len(slots)
    delta = np.zeros(n, dtype=np.float64)
    on = np.zeros(n, dtype=bool)
    for start in range(0, n, batch):
        cells = list(range(start, min(start + batch, n)))
        m = len(cells)
        while len(cells) < batch:
            cells.append(cells[-1])
        handle = lens_model.layers[layer].register_forward_hook(
            make_cell_hook(add_vec, [slots[c] for c in cells]))
        try:
            cs, flags = contrast_and_contract(
                model, lens_model, rows, tok_pos, tok_neg)
        finally:
            handle.remove()
        delta[start:start + m] = cs[:m] - zero
        on[start:start + m] = flags[:m]
    return delta, on


def run(args: argparse.Namespace) -> int:
    layers = parse_int_list(args.layers)
    conditions = parse_conditions(args.conditions)
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_path = out_dir / "maps.parquet"
    t0 = time.time()

    vecs = dict(np.load(Path(args.vectors) / args.case / "vectors.npz"))
    v_layers = [int(x) for x in vecs["layers"]]
    missing = [l for l in layers if l not in v_layers]
    if missing:
        raise ValueError(f"blocks {missing} are not in the vectors "
                         f"({v_layers})")
    scales = {l: float(vecs["scales"][v_layers.index(l)]) for l in layers}
    for arm, _ in conditions:
        for l in layers:
            if f"v_{arm}_L{l}" not in vecs:
                raise ValueError(f"no vector v_{arm}_L{l}")

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    paths = ss.load_images(args.case, args.kind)
    _, ev = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    eval_paths = [r["p"] for r in ev]
    build = ss.make_build_inputs(proc, tmpl, args.case)
    dev = {(arm, l): torch.tensor(vecs[f"v_{arm}_L{l}"]).float().to("cuda:0")
           for arm, _ in conditions for l in layers}
    log(f"{args.case}: words={ss.CASE_WORDS[args.case]} layers={layers} "
        f"conditions={conditions} batch={args.batch} "
        f"scales={ {l: round(s, 2) for l, s in scales.items()} }")

    records: list[dict] = []
    done: set = set()
    if maps_path.exists() and not args.smoke:
        prev = pd.read_parquet(maps_path)
        records = prev.to_dict("records")
        done = set(prev["image_path"])
        log(f"resuming: {len(done)} images done")

    processed = 0
    for path in eval_paths:
        if path in done:
            continue
        inputs = build(path)
        slots = pab.slots_of(inputs, image_token_id)
        rows = pab.ablation_batch(inputs, slots, [[]] * args.batch,
                                  image_token_id)
        c0, on0 = contrast_and_contract(model, lens_model, rows,
                                        tok_pos, tok_neg)
        zero = float(c0[0])
        if not np.allclose(c0, zero):
            raise RuntimeError("identical no-op rows disagree in one batch")
        for layer in layers:
            for arm, dose in conditions:
                add = dose * scales[layer] * dev[(arm, layer)]
                delta, on = cell_map(model, lens_model, rows, slots, layer,
                                     add, zero, tok_pos, tok_neg, args.batch)
                records.append({
                    "image_path": path, "layer": int(layer), "arm": arm,
                    "dose": float(dose), "scale_l": scales[layer],
                    "n_cells": len(slots), "zero": zero,
                    "zero_on_contract": bool(on0[0]),
                    "values": delta.tolist(), "on_contract": on.tolist(),
                    "frac_on_contract": float(on.mean()),
                })
        processed += 1
        if processed % 5 == 0 or args.smoke:
            pd.DataFrame(records).to_parquet(maps_path, index=False)
            el = time.time() - t0
            log(f"  {processed} images | {el:.0f}s | "
                f"{el / processed:.0f}s per image")
    df = pd.DataFrame(records)
    df.to_parquet(maps_path, index=False)

    summary = {}
    for (l, arm, dose), g in df.groupby(["layer", "arm", "dose"]):
        vals = np.stack([np.asarray(v) for v in g["values"]])
        summary[f"{arm}/L{l}/a{dose:+g}"] = {
            "n_images": int(len(g)),
            "mean_delta": round(float(vals.mean()), 4),
            "mean_abs_delta": round(float(np.abs(vals).mean()), 4),
            "mean_max_abs": round(float(np.abs(vals).max(1).mean()), 4),
            "frac_on_contract": round(float(g["frac_on_contract"].mean()), 4),
        }
        log(f"{arm}/L{l}/a{dose:+g}: " + json.dumps(summary[f'{arm}/L{l}/a{dose:+g}']))
    (out_dir / "summary.json").write_text(json.dumps({
        "case": args.case, "words": ss.CASE_WORDS[args.case],
        "layers": layers, "conditions": conditions, "batch": args.batch,
        "scales": {str(l): s for l, s in scales.items()},
        "vectors": str(Path(args.vectors) / args.case / "vectors.npz"),
        "n_images": int(df["image_path"].nunique()),
        "per_cell": summary, "seconds": round(time.time() - t0)}, indent=2))
    log(f"wrote {out_dir} ({time.time() - t0:.0f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--layers", default="16,22")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS_DEFAULT))
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--vectors", default=VECTORS_DEFAULT)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--smoke", action="store_true",
                    help="2 images, 1 block, 1 condition, batch 4")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.n_images = 2
        args.layers = args.layers.split(",")[0]
        args.conditions = args.conditions[:1]
        args.batch = 4
        args.out_dir = str(Path(args.out_dir) / "smoke")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
