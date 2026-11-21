"""The steering coordinate: its natural value, and its map over the patches.

`single_steer` injects ``v_l = pinv(J_l) d`` at the answer position and moves
the model's own judgment. This module asks 2 questions about that vector,
in the SAME prompt frame, on the SAME 60 eval images.

1. The natural coordinate. With no injection, how far does each image sit
   along ``v_l`` at the answer position? ``s_l = <h_l[answer], v_l>``. If
   the judgment of the model varies along ``v_l`` at the steering gain, the
   direction is one the model reads. If not, the direction is a handle the
   model does not use.

2. The map. Remove one image cell at the input (the `patch_ablate` edit that
   `causal_maps` uses), run the rest forward, and record BOTH ``c`` and
   ``h_l[answer]``. The change in ``c`` is the causal map of the judgment.
   The change in ``h_l[answer]`` gives, on the CPU, the change in the
   coordinate along ANY direction. So one GPU run serves every direction,
   now and later.

The forward is the `single_steer` prompt, not the `single.py` rating prompt.
The answer position is not question-blind, so the coordinate is only the
steering coordinate under the prompt the steering used.

Outputs, under ``<out-dir>/<case>/``:

| File | Content |
|---|---|
| ``vectors.npz`` | ``d``, ``d_control``, ``v_signal_L<l>``, ``v_control_L<l>``, scales |
| ``natural.parquet`` | one row per eval image: ``c``, ``logit_diff``, top token |
| ``natural_acts.npy`` | ``[n_images, n_layers, d_model]`` fp16, the answer residual |
| ``map_stats.parquet`` | one row per image: ``c``, the controls, the zero |
| ``maps.parquet`` | ``values`` = contribution toward the positive word |
| ``answer_acts.npy`` | ``[n_images, 1 + n_cells, n_layers, d_model]`` fp16; row 0 is the zero |
| ``summary.json`` | the run settings and the layer list |

The zero and every cell sit in a batch of ``--batch`` rows, as in
`causal_maps`. A change of batch size moves ``c`` by up to 0.67 logits, so a
difference is only readable inside one batch size.

Score it with ``scripts/steer_coord_report.py`` (CPU).

Usage:
    sbatch --time=00:30:00 monocle/monocle.sub monocle.steer_coord --smoke
    sbatch --time=01:30:00 monocle/monocle.sub monocle.steer_coord \\
        --case restaurants --n-images 60 --batch 12
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
    build_layer_vectors,
    load_lens,
    load_model_and_lens,
    parse_int_list,
    split_calibration_eval,
)

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/steer_coord")
DEFAULT_LAYERS = "41,42,44,45"
#: The sweep measured at 60 eval images after 16 calibration images.
N_CALIB_DEFAULT = 16
#: `causal_maps60` measured at this batch. Keep it, so a later comparison
#: of the 2 map sets sits on one batch size.
BATCH_DEFAULT = 12


def log(msg: str) -> None:
    print(f"[steer-coord] {msg}", flush=True)


# ---------------------------------------------------------------------------
# A) Pure helpers (CPU-testable)
# ---------------------------------------------------------------------------
def coordinate(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``<h, v>`` over the last axis, in float64.

    ``h`` is ``[..., d_model]``; ``v`` is ``[d_model]`` and unit-norm. The
    result has the shape of ``h`` without its last axis.
    """
    v = np.asarray(v, dtype=np.float64)
    if abs(float(np.linalg.norm(v)) - 1.0) > 1e-3:
        raise ValueError("the direction must be unit-norm")
    return np.asarray(h, dtype=np.float64) @ v


def delta_map(acts: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The change in the coordinate for each cell, against row 0.

    ``acts`` is ``[1 + n_cells, d_model]``; row 0 is the zero (no cell
    removed). Returns ``[n_cells]``, the coordinate of the cell row minus
    the coordinate of the zero row.
    """
    if acts.ndim != 2 or acts.shape[0] < 2:
        raise ValueError("acts must be [1 + n_cells, d_model]")
    s = coordinate(acts, v)
    return s[1:] - s[0]


def contribution_sign(delta: np.ndarray) -> np.ndarray:
    """Turn a removal delta into a contribution, as `causal_maps` does.

    A removal delta is the effect of taking the cell away. Its negative is
    what the cell contributes while it is there.
    """
    return -np.asarray(delta, dtype=np.float64)


# ---------------------------------------------------------------------------
# B) GPU measurement
# ---------------------------------------------------------------------------
def contrast_and_answer_acts(
    model: Any, lens_model: Any, rows: dict, layers: list[int],
    pos_id: int, neg_id: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """One forward. Returns ``c`` per row and the answer residual per layer.

    ``c`` comes from `pc.fp32_contrast` on the final block, so it is the
    same quantity `causal_maps` measures. The answer position is the last
    token of every row, which holds because each row drops the same number
    of image tokens.
    """
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder

    final = lens_model.n_layers - 1
    at = sorted(set(layers) | {final})
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=at) as rec:
        out = model(**rows, use_cache=False, logits_to_keep=1)
        del out
        h_final = rec.activations[final][:, -1, :]
        c = pc.fp32_contrast(lens_model, h_final, pos_id, neg_id)
        acts = {l: rec.activations[l][:, -1, :].detach().to(torch.float16)
                .cpu().numpy() for l in layers}
    return c.float().cpu().numpy(), acts


def rebuild_vectors(
    model: Any, lens_model: Any, lens: Any, build, calib_paths: list[str],
    layers: list[int], tok_pos: int, tok_neg: int,
) -> dict:
    """The sweep's vectors again: ``d``, the control, and ``v_l`` per layer.

    The construction is the one in `single_steer.run_experiment`: the same
    seed, the same calibration split, the same random control, and the pinv
    pull-back. So the vectors here ARE the vectors the sweep injected.
    """
    torch.manual_seed(ss.SEED)
    d, scales, scale_report = ss.compute_direction_and_scales(
        model, lens_model, build, calib_paths, layers, tok_pos, tok_neg,
        scale_mode="centered")
    gen = torch.Generator().manual_seed(ss.SEED)
    d_control = torch.randn(lens.d_model, generator=gen)
    d_control = d_control / d_control.norm()
    report: dict = {}
    vecs = build_layer_vectors(lens, d, d_control, layers, pullback="pinv",
                               report=report)
    out = {
        "d": d.numpy(), "d_control": d_control.numpy(),
        "layers": np.array(layers),
        "scales": np.array([scales[l] for l in layers]),
        "scale_raw": np.array([scale_report[l]["raw"] for l in layers]),
        "err_pinv": np.array([report["signal"][l]["err_pinv"] for l in layers]),
    }
    for l in layers:
        out[f"v_signal_L{l}"] = vecs["signal"][l].numpy()
        out[f"v_control_L{l}"] = vecs["control"][l].numpy()
    return out


def natural_read(
    model: Any, lens_model: Any, build, paths: list[str], layers: list[int],
    tok_pos: int, tok_neg: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Batch-1 forward per eval image: ``c``, the model's own logit_diff,
    and the answer residual at each layer."""
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder

    final = lens_model.n_layers - 1
    at = sorted(set(layers) | {final})
    rows, acts_all = [], []
    for path in paths:
        inputs = build(path)
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=at) as rec:
            out = model(**inputs, use_cache=False)
            logits = out.logits[0, -1].float()
            h_final = rec.activations[final][:, -1, :]
            c = float(pc.fp32_contrast(lens_model, h_final, tok_pos, tok_neg)[0])
            acts_all.append(np.stack([
                rec.activations[l][0, -1, :].detach().to(torch.float16)
                .cpu().numpy() for l in layers]))
        m = ss.score_polar_logits(logits, tok_pos, tok_neg)
        rows.append({"image_path": path, "c": c, **m})
    return pd.DataFrame(rows), np.stack(acts_all)


def map_one_image(
    model: Any, lens_model: Any, inputs: dict, slots: list[int],
    layers: list[int], tok_pos: int, tok_neg: int, image_token_id: int,
    batch: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """The zero, the 256 removal deltas of ``c``, and the answer residuals.

    Every chunk is padded to ``batch`` rows, and the zero is measured once in
    a batch of ``batch`` no-op rows, as `pab.ablation_map` does. Returns
    ``(zero, delta_c [n_cells], acts [1 + n_cells, n_layers, d_model])``.
    """
    n = len(slots)
    zero_rows = pab.ablation_batch(inputs, slots, [[]] * batch, image_token_id)
    c0, a0 = contrast_and_answer_acts(
        model, lens_model, zero_rows, layers, tok_pos, tok_neg)
    zero = float(c0[0])
    if not np.allclose(c0, zero):
        raise RuntimeError("identical no-op rows disagree inside one batch")
    acts = np.zeros((1 + n, len(layers), a0[layers[0]].shape[-1]),
                    dtype=np.float16)
    for j, l in enumerate(layers):
        acts[0, j] = a0[l][0]
    delta = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch):
        chunk = [[i] for i in range(start, min(start + batch, n))]
        m = len(chunk)
        while len(chunk) < batch:
            chunk.append(chunk[-1])
        rows = pab.ablation_batch(inputs, slots, chunk, image_token_id)
        cs, a = contrast_and_answer_acts(
            model, lens_model, rows, layers, tok_pos, tok_neg)
        delta[start:start + m] = cs[:m] - zero
        for j, l in enumerate(layers):
            acts[1 + start:1 + start + m, j] = a[l][:m]
    return zero, delta, acts


def run_case(args: argparse.Namespace) -> int:
    layers = parse_int_list(args.layers)
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lens = load_lens(args.lens)
    missing = [l for l in layers if l not in lens.jacobians]
    if missing:
        raise ValueError(f"layers {missing} not fitted ({lens.source_layers})")
    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    paths = ss.load_images(args.case, args.kind)
    calib, ev = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    calib_paths = [r["p"] for r in calib]
    eval_paths = [r["p"] for r in ev]
    build = ss.make_build_inputs(proc, tmpl, args.case)

    # The eval set must be the sweep's eval set, or the natural coordinate
    # cannot be joined to the sweep's baseline.
    sweep = Path(args.sweep_dir) / args.case / "steer_results.parquet"
    if sweep.exists():
        seen = set(pd.read_parquet(sweep, columns=["image_path"])["image_path"])
        if set(eval_paths) != seen:
            raise SystemExit(
                f"the eval images differ from the sweep's ({len(seen)} there, "
                f"{len(eval_paths)} here, {len(seen & set(eval_paths))} shared)")
        log(f"eval set matches the sweep at {sweep}")
    else:
        log(f"no sweep at {sweep}; the eval set is not checked")

    log(f"{args.case}: words={ss.CASE_WORDS[args.case]} layers={layers} "
        f"calib={len(calib_paths)} eval={len(eval_paths)} batch={args.batch}")

    vecs = rebuild_vectors(model, lens_model, lens, build, calib_paths,
                           layers, tok_pos, tok_neg)
    np.savez(out_dir / "vectors.npz", **vecs)
    log("vectors: " + " ".join(
        f"L{l}: scale={s:.2f} err_pinv={e:.4f}"
        for l, s, e in zip(layers, vecs["scales"], vecs["err_pinv"])))

    nat, nat_acts = natural_read(model, lens_model, build, eval_paths,
                                 layers, tok_pos, tok_neg)
    nat.to_parquet(out_dir / "natural.parquet", index=False)
    np.save(out_dir / "natural_acts.npy", nat_acts)
    log(f"natural read: logit_diff mean {nat['logit_diff'].mean():+.2f}, "
        f"c mean {nat['c'].mean():+.2f}, "
        f"{int(nat['argmax_pos'].sum())}/{len(nat)} positive "
        f"({time.time() - t0:.0f}s)")

    stats, maps = [], []
    acts_store: Optional[np.ndarray] = None
    for i, path in enumerate(eval_paths):
        inputs = build(path)
        slots = pab.slots_of(inputs, image_token_id)
        c_base = float(nat["c"].iloc[i])
        ctl = pab.controls(model, lens_model, inputs, slots, c_base,
                           tok_pos, tok_neg, image_token_id=image_token_id,
                           batch=args.batch)
        pab.gate_rebuild(ctl)
        zero, delta, acts = map_one_image(
            model, lens_model, inputs, slots, layers, tok_pos, tok_neg,
            image_token_id, args.batch)
        if abs(zero - ctl["zero"]) > 1e-3:
            raise RuntimeError(
                f"the zero moved between 2 reads of one batch: "
                f"{zero:.4f} vs {ctl['zero']:.4f}")
        if acts_store is None:
            acts_store = np.zeros((len(eval_paths),) + acts.shape,
                                  dtype=np.float16)
        acts_store[i] = acts
        values = contribution_sign(delta)
        stats.append({
            "case": args.case, "image_path": path, "c": c_base,
            "zero": zero, "c_all": ctl["c_all"],
            "image_worth": ctl["image_worth"], "null_error": ctl["null_error"],
            "batch_offset": ctl["batch_offset"], "worth_ok": bool(ctl["worth_ok"]),
            "measure_batch": args.batch, "n_cells": len(slots),
            "max_abs": float(np.max(np.abs(values))),
            "parts_over_whole": (float(values.sum() / ctl["image_worth"])
                                 if ctl["image_worth"] else float("nan")),
        })
        maps.append({"image_path": path, "values": values.tolist()})
        log(f"  {i + 1}/{len(eval_paths)} c={c_base:+.2f} "
            f"whole={ctl['image_worth']:+.2f} max|dc|={stats[-1]['max_abs']:.3f}"
            f"{'' if ctl['worth_ok'] else '  [weak image]'} "
            f"({time.time() - t0:.0f}s)")
        if (i + 1) % 10 == 0 or i + 1 == len(eval_paths):
            pd.DataFrame(stats).to_parquet(out_dir / "map_stats.parquet",
                                           index=False)
            pd.DataFrame(maps).to_parquet(out_dir / "maps.parquet", index=False)
            np.save(out_dir / "answer_acts.npy", acts_store[:i + 1])

    (out_dir / "summary.json").write_text(json.dumps({
        "case": args.case, "words": ss.CASE_WORDS[args.case],
        "token_ids": [tok_pos, tok_neg], "layers": layers,
        "n_calib": args.n_calib, "n_images": len(eval_paths),
        "measure_batch": args.batch, "lens": args.lens,
        "reference": "absent", "pullback": "pinv", "scale_mode": "centered",
        "data_sign": "values are the contribution toward the positive word",
        "acts_layout": "answer_acts[image, 0=zero | 1+cell, layer, dim] fp16",
        "seconds": round(time.time() - t0),
    }, indent=2))
    log(f"wrote {out_dir} ({time.time() - t0:.0f}s)")
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    """1 calib image, 1 eval image, the first 12 cells, with asserts."""
    layers = parse_int_list(args.layers)[:2]
    lens = load_lens(args.lens)
    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    paths = ss.load_images(args.case, args.kind)
    calib, ev = split_calibration_eval([{"p": p} for p in paths], 2, 1)
    build = ss.make_build_inputs(proc, tmpl, args.case)

    vecs = rebuild_vectors(model, lens_model, lens, build,
                           [r["p"] for r in calib], layers, tok_pos, tok_neg)
    for l in layers:
        v = vecs[f"v_signal_L{l}"]
        assert abs(np.linalg.norm(v) - 1.0) < 1e-3, "v is not unit"
    log("ASSERT vectors unit-norm OK")

    nat, nat_acts = natural_read(model, lens_model, build, [ev[0]["p"]],
                                 layers, tok_pos, tok_neg)
    assert nat_acts.shape == (1, len(layers), lens.d_model)
    assert abs(float(nat["c"].iloc[0]) - float(nat["logit_diff"].iloc[0])) < 0.5, (
        "fp32 c and the model's own logit_diff disagree by over 0.5")
    log(f"ASSERT natural read OK: c={nat['c'].iloc[0]:+.3f} "
        f"logit_diff={nat['logit_diff'].iloc[0]:+.3f}")

    inputs = build(ev[0]["p"])
    slots = pab.slots_of(inputs, image_token_id)
    ctl = pab.controls(model, lens_model, inputs, slots, float(nat["c"].iloc[0]),
                       tok_pos, tok_neg, image_token_id=image_token_id,
                       batch=4)
    pab.gate_rebuild(ctl)
    log(f"ASSERT rebuild null OK: null_error={ctl['null_error']:.4f} "
        f"whole={ctl['image_worth']:+.2f}")
    zero, delta, acts = map_one_image(
        model, lens_model, inputs, slots[:12], layers, tok_pos, tok_neg,
        image_token_id, 4)
    assert acts.shape == (13, len(layers), lens.d_model)
    assert np.isfinite(delta).all() and np.isfinite(acts.astype(np.float32)).all()
    s = delta_map(acts[:, 0].astype(np.float32), vecs[f"v_signal_L{layers[0]}"])
    log(f"ASSERT map OK: zero={zero:+.3f} |dc| max {np.abs(delta).max():.3f} "
        f"ds(L{layers[0]}) range {s.min():+.3f}..{s.max():+.3f}")
    log("SMOKE complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--n-calib", type=int, default=N_CALIB_DEFAULT)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--lens", default=ss.LENS_DEFAULT)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--sweep-dir",
                    default=str(REPO / "outputs/_monocle/single_steer_fine"),
                    help="the sweep whose eval set this run must match")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        return run_smoke(args)
    return run_case(args)


if __name__ == "__main__":
    raise SystemExit(main())
