"""Does the final patch CARRY the judgment, or is it only the ADDRESS?

The question
------------
The answer position attends to the final patch of an image block about 100x
more than to any other patch. Two readings fit that fact:

| Reading | Claim |
|---|---|
| carrier | the block's content is routed INTO the last patch |
| address | any patch would do; the last one is where the answer looks |

The attention control already hurt the carrier reading. An ordinary patch
reads its own block MORE than the last patch does, at nearly every block
(ratio 0.33 to 0.98). Thus the last patch does not gather more.

This module settles it by intervention. Take a donor image and a target
image, and at layer L replace ONE patch of the target with the donor's
vector. Then measure how far `c` moves toward the donor's `c`:

    recovery = (c_swapped - c_target) / (c_donor - c_target)

| Result | Reading |
|---|---|
| recovery(last) >> recovery(mid) | carrier — the last patch holds it |
| recovery(last) about recovery(mid) | address — the position is arbitrary |

Why this test works when the earlier ones failed
------------------------------------------------
All 3 calibration attempts in `patch_causal` died at the same boundary: the
vision embedder casts pixels to bfloat16, so the model is a step function at
fine pixel scale. This intervention happens in the RESIDUAL STREAM, after
that cast, in the arithmetic the model really does. The bf16 pixel problem
cannot reach it.

The null swap is the gate. Overwriting a position with the value already
there must reproduce `c` exactly, or the plumbing is wrong and the run stops.

This runs on the SINGLE-IMAGE path, where every prompt has the same length
and the same patch positions, so a swap is position-aligned by construction.
The run asserts that alignment and refuses to guess.
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
from research.monocle import safety_workspace as sw  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/patch_swap"
#: L43 is the attribution peak and L46 is where 1 patch holds 43% of the map.
#: The early layers are not decoration. A swap late in the stack can read as
#: null because the answer position ALREADY absorbed the image lower down, and
#: only a depth sweep separates that from a swap that did nothing.
DEFAULT_LAYERS = (5, 20, 30, 38, 42, 43, 46)
#: A donor and a target must disagree by at least this much, or `recovery`
#: divides by a number close to zero.
MIN_GAP = 1.0


def log(m: str) -> None:
    print(f"[patch-swap] {m}", flush=True)


class Patcher:
    """Overwrite chosen sequence positions at the output of one block."""

    def __init__(self, positions: list[int], values: torch.Tensor) -> None:
        self.positions = positions
        self.values = values

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, self.positions, :] = self.values.to(h.dtype).to(h.device)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


def contrast_and_residual(
    model: Any, lens_model: Any, inputs: dict, layer: int,
    pos_id: int, neg_id: int,
) -> tuple[float, torch.Tensor]:
    """`c` for this input, and the whole layer-L residual [seq, d]."""
    from jlens.hooks import ActivationRecorder

    with torch.no_grad(), ActivationRecorder(
            lens_model.layers, at=[layer, pa.FINAL_LAYER]) as rec:
        out = model(**inputs, use_cache=False, logits_to_keep=1)
        del out
        h_final = rec.activations[pa.FINAL_LAYER][:, -1, :]
        h_layer = rec.activations[layer][0].detach().clone()
    c = float(pc.fp32_contrast(lens_model, h_final, pos_id, neg_id)[0])
    return c, h_layer


def swapped_contrast(
    model: Any, lens_model: Any, inputs: dict, layer: int,
    positions: list[int], values: torch.Tensor, pos_id: int, neg_id: int,
) -> float:
    """`c` after the chosen positions are overwritten at block `layer`."""
    handle = lens_model.layers[layer].register_forward_hook(
        Patcher(positions, values))
    try:
        h = pc.final_residual(model, lens_model, inputs, graph=False)
        return float(pc.fp32_contrast(lens_model, h, pos_id, neg_id)[0])
    finally:
        handle.remove()


def stream_groups(
    input_ids: torch.Tensor, block: list[int], answer_pos: int,
) -> dict[str, list[int]]:
    """The NON-patch position groups, so we can follow where the image goes.

    The single-image prompt is byte identical for every image. Thus two images
    give the same text TOKENS at these positions, and their residuals differ
    only by what each has absorbed from its own picture. Swapping a text
    position therefore measures how much image information has reached the
    text stream by layer L, with nothing else varying.

    That property is why this test wants the single-image path. On the
    pairwise path the 2 blocks sit at different offsets and the text between
    them differs by pair.
    """
    from research.monocle.attn_route import BOI_TOKEN_ID, EOI_TOKEN_ID

    ids = input_ids.tolist()
    lo, hi = block[0], block[-1]
    boi = [i for i, t in enumerate(ids) if t == BOI_TOKEN_ID]
    eoi = [i for i, t in enumerate(ids) if t == EOI_TOKEN_ID]
    delim = set(boi) | set(eoi)
    return {
        "text_before": [i for i in range(lo) if i not in delim],
        "text_after": [i for i in range(hi + 1, len(ids))
                       if i not in delim and i != answer_pos],
        "boi": boi,
        "eoi": eoi,
    }


def conditions(
    block: list[int], rng: np.random.Generator, answer_pos: int,
    stream: Optional[dict[str, list[int]]] = None,
) -> dict[str, list[int]]:
    """Which positions each arm overwrites. `block` is one image's patches.

    `answer` is the POSITIVE CONTROL and it is not optional. Overwriting the
    answer position itself must carry `c` nearly all the way to the donor,
    because `c` is read from that position. If it does not, the hook never
    took effect and no other arm here means anything.

    A null swap does NOT prove the hook works: writing back the value already
    present and writing nothing at all give the same answer. The first version
    of this module shipped with only that gate, and it passed while every arm
    read about zero.
    """
    n = len(block)
    interior = block[1:-1]
    arms = {
        "answer": [answer_pos],
        "last": [block[-1]],
        "mid": [block[n // 2]],
        "first": [block[0]],
        "random": [int(rng.choice(interior))],
        "random_8": [int(x) for x in rng.choice(interior, 8, replace=False)],
        "all": list(block),
    }
    for name, pos in (stream or {}).items():
        if pos:
            arms[name] = list(pos)
    return arms


#: The positive control must reach this recovery AT THE DEEPEST LAYER, or the
#: run is void. It is not required to reach it at shallow layers: an early
#: swap of the answer position is undone by the blocks above, which re-read
#: the target's own image. Measured, the L5 control recovers about 0.47.
CONTROL_FLOOR = 0.5


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, layers: list[int], out_dir: Path, device: str, seed: int,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rng = np.random.default_rng(seed)
    rows_meta = sg.load_single_rows(n_images, case)

    # --- pass 1: c and the layer residuals for every image ---
    store: list[dict] = []
    ref_geom: Optional[tuple] = None
    for r in rows_meta:
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        seq_len = int(inputs["input_ids"].shape[1])
        block = [int(x) for x in
                 sw.image_token_blocks(inputs, image_token_id, expect=1)[0]]
        geom = (seq_len, block[0], block[-1], len(block))
        if ref_geom is None:
            ref_geom = geom
        elif geom != ref_geom:
            raise RuntimeError(
                "the images do not share a geometry, so a swap is not "
                f"position aligned: {geom} vs {ref_geom}")
        per_layer = {}
        c = None
        for layer in layers:
            c, h = contrast_and_residual(
                model, lens_model, inputs, layer, pos_id, neg_id)
            per_layer[layer] = h
        store.append({**r, "inputs": inputs, "block": block,
                      "answer_pos": seq_len - 1,
                      "stream": stream_groups(
                          inputs["input_ids"][0], block, seq_len - 1),
                      "c": c, "h": per_layer})
    log(f"{case}: {len(store)} images, geometry {ref_geom}, "
        f"c range {min(s['c'] for s in store):+.2f} to "
        f"{max(s['c'] for s in store):+.2f}")

    # --- pass 2: swap, pairing each image with its opposite in c ---
    order = sorted(range(len(store)), key=lambda i: store[i]["c"])
    pairs = [(order[i], order[-1 - i]) for i in range(len(order) // 2)]
    rows: list[dict] = []
    gate_done = False
    for t_i, d_i in pairs:
        tgt, don = store[t_i], store[d_i]
        gap = don["c"] - tgt["c"]
        if abs(gap) < MIN_GAP:
            continue
        for layer in layers:
            arms = conditions(tgt["block"], rng, tgt["answer_pos"],
                              tgt["stream"])
            if not gate_done:
                # The gate: write back what is already there.
                c_null = swapped_contrast(
                    model, lens_model, tgt["inputs"], layer, tgt["block"],
                    tgt["h"][layer][tgt["block"], :], pos_id, neg_id)
                if abs(c_null - tgt["c"]) > 1e-3:
                    raise RuntimeError(
                        f"null swap moved c by {c_null - tgt['c']:.2e} — the "
                        "patch hook is wrong, so no arm below is meaningful")
                log(f"null-swap gate passed at L{layer} "
                    f"(|dc| = {abs(c_null - tgt['c']):.2e})")
                gate_done = True
            for arm, positions in arms.items():
                c_new = swapped_contrast(
                    model, lens_model, tgt["inputs"], layer, positions,
                    don["h"][layer][positions, :], pos_id, neg_id)
                rows.append({
                    "case": case, "layer": layer, "arm": arm,
                    "n_pos": len(positions),
                    "target": tgt["image_id"], "donor": don["image_id"],
                    "c_target": tgt["c"], "c_donor": don["c"],
                    "c_swapped": c_new,
                    "delta": c_new - tgt["c"],
                    "recovery": (c_new - tgt["c"]) / gap})
    df = pd.DataFrame(rows)
    # The control must be judged at the DEEPEST layer. Swapping the answer
    # position early leaves many blocks to re-derive the state from the
    # target's own image, so partial recovery there is the expected result and
    # not a fault. Only near the top does the swap have nothing left to undo.
    ctrl = df[df["arm"] == "answer"]
    if ctrl.empty:
        raise RuntimeError("the positive control did not run")
    by_layer = ctrl.groupby("layer")["recovery"].mean()
    log("positive control (answer-position swap) by depth:")
    for layer, rec in by_layer.items():
        log(f"    L{int(layer):<3} recovers {rec:+.3f}")
    deepest = float(by_layer.loc[by_layer.index.max()])
    if deepest < CONTROL_FLOOR:
        raise RuntimeError(
            f"at L{int(by_layer.index.max())} the answer-position swap "
            f"recovered only {deepest:.3f}, below {CONTROL_FLOOR}. Almost no "
            "computation remains above that layer, so the swap must carry c "
            "to the donor. The hook did not take effect, and no arm in this "
            "run is interpretable.")
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "patch_swap.parquet", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["layer", "arm"]).agg(
        n=("recovery", "size"),
        mean_recovery=("recovery", "mean"),
        mean_abs_delta=("delta", lambda s: float(np.mean(np.abs(s)))),
    ).reset_index()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    ap.add_argument("--seed", type=int, default=pa.SEED)
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
    for case in args.cases:
        df = run_case(case, proc, model, lens_model, tmpl,
                      n_images=args.n_images, layers=sorted(set(args.layers)),
                      out_dir=out_dir, device=args.device, seed=args.seed)
        s = summarize(df)
        log("=" * 74)
        log(f"CASE {case} — how far c moves toward the donor when 1 patch "
            "is swapped")
        log("  layer  arm         n   mean recovery   mean |dc|")
        for _, r in s.iterrows():
            log(f"  {int(r.layer):>5}  {r.arm:<10} {int(r.n):>3}   "
                f"{r.mean_recovery:>+13.3f}   {r.mean_abs_delta:>9.3f}")
        (out_dir / case / "summary.json").write_text(
            json.dumps(s.to_dict(orient="records"), indent=2))
    log("patch swap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
