"""Does a steered judgment survive free generation?

Every steering sweep reads a teacher-forced next token. This module lets
the model WRITE its answer under the same injections and counts the words
it emits. Two modes:

| Mode | The prompt ends with | The model writes |
|---|---|---|
| ``prefix`` | the forced ``{"answer": "`` | the word and the closing JSON |
| ``free`` | the bare generation prompt | the whole JSON object |

The vector is added at the last prompt position and at every generated
position after it. Generation is greedy and runs with no KV cache, so the
hook sees the whole sequence at every step and the injection is the same
at every position.

The vectors come from the 4-arm sweep (``vectors.npz``), so this run
injects exactly the directions the sweep measured.

Usage:
    sbatch --time=01:30:00 research/monocle/monocle.sub monocle.steer_generate \\
        --case restaurants --n-images 60
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import single_steer as ss  # noqa: E402
from research.monocle.jlens_steer import (  # noqa: E402
    load_model_and_lens,
    split_calibration_eval,
)

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/steer_generate")
VECTORS_DEFAULT = str(REPO / "outputs/_monocle/single_steer_axis")
#: (arm, dose) pairs. The axis doses sit at the grid edge; the pinv doses
#: inside every case's pinv window; the control doses at the last dose
#: where each control still answers in most cases.
CONDITIONS_DEFAULT = ("axis:-2", "axis:+2", "signal:-0.75", "signal:+0.75",
                      "control:-0.75", "control:+0.75",
                      "control_raw:-0.25", "control_raw:+0.25")
MAX_NEW = {"prefix": 4, "free": 14}


def log(msg: str) -> None:
    print(f"[steer-generate] {msg}", flush=True)


def parse_conditions(items: list[str]) -> list[tuple[str, float]]:
    out = [("baseline", 0.0)]
    for it in items:
        arm, dose = it.split(":")
        out.append((arm, float(dose)))
    return out


def classify(text: str, pos: str, neg: str) -> str:
    """Which pole word the text names first; ``other`` when neither."""
    t = text.lower()
    hits = {w: t.find(w) for w in (pos, neg)}
    hits = {w: i for w, i in hits.items() if i >= 0}
    if not hits:
        return "other"
    first = min(hits, key=hits.get)
    return "pos" if first == pos else "neg"


def make_tail_hook(add_vec: torch.Tensor, start: int):
    """Add ``add_vec`` at every position from ``start`` onward."""
    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        tensor = output[0] if is_tuple else output
        new = tensor.clone()
        new[0, start:, :] = new[0, start:, :] + add_vec.to(new.dtype)
        return (new, *tuple(output[1:])) if is_tuple else new
    return hook


def generate(model, lens_model, inputs: dict, layer: int,
             add_vec: Optional[torch.Tensor], max_new: int,
             eos_ids: set[int]) -> list[int]:
    """Greedy tokens, no cache, the hook live for every step."""
    ids = inputs["input_ids"]
    start = int(ids.shape[1]) - 1
    handle = None
    out_tokens: list[int] = []
    try:
        if add_vec is not None:
            handle = lens_model.layers[layer].register_forward_hook(
                make_tail_hook(add_vec, start))
        cur = {k: v for k, v in inputs.items()}
        for _ in range(max_new):
            with torch.inference_mode():
                out = model(**cur, use_cache=False)
            nxt = int(out.logits[0, -1].argmax())
            out_tokens.append(nxt)
            if nxt in eos_ids:
                break
            tok = torch.tensor([[nxt]], device=ids.device)
            # Every per-sequence tensor grows by 1: the id, a mask of 1,
            # and a 0 for any type id (the new token is text).
            seq_keys, _, _ = pab.split_keys(cur)
            for k in seq_keys:
                fill = tok if k == "input_ids" else (
                    torch.ones_like(tok) if k == "attention_mask"
                    else torch.zeros_like(tok).to(cur[k].dtype))
                cur[k] = torch.cat([cur[k], fill], dim=1)
    finally:
        if handle is not None:
            handle.remove()
    return out_tokens


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vec_file = Path(args.vectors) / args.case / "vectors.npz"
    vecs = dict(np.load(vec_file))
    layers = [int(x) for x in vecs["layers"]]
    layer = args.layer
    if layer not in layers:
        raise ValueError(f"layer {layer} not in the sweep's {layers}")
    scale = float(vecs["scales"][layers.index(layer)])
    conditions = parse_conditions(args.conditions)
    for arm, _ in conditions[1:]:
        if f"v_{arm}_L{layer}" not in vecs:
            raise ValueError(f"the sweep has no arm {arm!r}")

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    tok_pos, tok_neg = ss.polar_token_ids(tok, args.case)
    pos_w, neg_w = ss.CASE_WORDS[args.case]
    paths = ss.load_images(args.case, args.kind)
    _, ev = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    eos_ids = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()
    for t in ("<turn|>", "<end_of_turn>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            eos_ids.add(i)

    modes = args.modes.split(",")
    builders = {}
    from PIL import Image
    for mode in modes:
        text = ss.render_single_chat(
            tok, tmpl, ss.user_text(args.case),
            force_prefix=(ss.FORCE_PREFIX if mode == "prefix" else ""))

        def build(path, text=text):
            img = Image.open(path).convert("RGB")
            return dict(proc(text=[text], images=[[img]],
                             return_tensors="pt").to("cuda:0"))
        builders[mode] = build

    log(f"{args.case}: L{layer} scale={scale:.2f} modes={modes} "
        f"conditions={conditions}")
    rows = []
    for i, r in enumerate(ev):
        for mode in modes:
            inputs = builders[mode](r["p"])
            for arm, dose in conditions:
                add = None
                if arm != "baseline":
                    v = torch.tensor(vecs[f"v_{arm}_L{layer}"]).to("cuda:0")
                    add = dose * scale * v
                toks = generate(model, lens_model, inputs, layer, add,
                                MAX_NEW[mode], eos_ids)
                text = tok.decode(toks, skip_special_tokens=False)
                rows.append({
                    "image_path": r["p"], "mode": mode, "arm": arm,
                    "dose": dose, "layer": layer, "text": text,
                    "answer": classify(text, pos_w, neg_w),
                    "json_ok": bool(re.search(r'"answer"\s*:\s*"', text))
                    if mode == "free" else True,
                })
        if (i + 1) % 10 == 0:
            pd.DataFrame(rows).to_parquet(out_dir / "generations.parquet",
                                          index=False)
            log(f"  {i + 1}/{len(ev)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "generations.parquet", index=False)

    summary = {}
    for (mode, arm, dose), g in df.groupby(["mode", "arm", "dose"]):
        key = f"{mode}/{arm}/{dose:+g}"
        summary[key] = {
            "n": int(len(g)),
            "pos": round(float((g["answer"] == "pos").mean()), 3),
            "neg": round(float((g["answer"] == "neg").mean()), 3),
            "other": round(float((g["answer"] == "other").mean()), 3),
            "json_ok": round(float(g["json_ok"].mean()), 3),
            "example": str(g["text"].iloc[0])[:60],
        }
        log(f"{key:>28}: pos {summary[key]['pos']:.2f} neg "
            f"{summary[key]['neg']:.2f} other {summary[key]['other']:.2f} "
            f"| {summary[key]['example']!r}")
    (out_dir / "summary.json").write_text(json.dumps({
        "case": args.case, "layer": layer, "scale": scale, "modes": modes,
        "conditions": conditions, "words": [pos_w, neg_w],
        "n_images": len(ev), "cells": summary,
        "seconds": round(time.time() - t0)}, indent=2))
    log(f"wrote {out_dir} ({time.time() - t0:.0f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--layer", type=int, default=41)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--modes", default="prefix,free")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS_DEFAULT))
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--vectors", default=VECTORS_DEFAULT)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
