"""Cut the answer's view of the image and see whether `c` notices.

The tension this resolves
-------------------------
`attn_route` says blocks 42 and 46 spend 38-40% of the answer's attention on
image patches. `patch_swap` says the image is causally spent below L30: swap
every one of the 256 patches at L42 and `c` moves 0.010 out of a range of 17.

Both cannot be doing work. But the swap does not settle it, because the swap
only tests the IMAGE-DEPENDENT part. If the answer's attention to patches
contributes a large term that is the SAME for every image, a swap sees
nothing and a knockout sees everything. So the 2 tests ask different
questions, and this one asks whether the attention carries anything at all.

The method
----------
Each decoder layer takes its `attention_mask` as a keyword argument. A
pre-hook can replace that mask for ONE block, which blocks a chosen query row
from reading chosen key positions.

**Warning:** `create_causal_mask` builds ONE mask tensor and the loop hands
the same object to every block of that layer type. Modifying it in place
would silently knock out every layer at once. This module always clones.

The gate
--------
A "change nothing" arm is NOT a gate. It passes when the hook works and when
the hook is dead, which is the mistake `patch_swap` shipped with. The gate
here is `answer_question`: block the answer from reading the QUESTION and `c`
must move a lot. If it does not, the mask never took effect.
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

OUT_DEFAULT = REPO / "outputs/_monocle/attn_knockout"
#: Block RANGES, not single blocks. The swap puts the handover between L22
#: and L30, so the ranges split there. "0-47" is the control's range.
DEFAULT_RANGES = ("0-47", "0-21", "22-29", "30-47", "42-47", "47")
#: The control cuts the answer off from the question at EVERY block. `c` must
#: then collapse. A 1-block cut only reaches 0.845, which is why the control
#: is a range and not a block.
CONTROL_FLOOR = 2.0


def log(m: str) -> None:
    print(f"[attn-knockout] {m}", flush=True)


class MaskPatcher:
    """Block chosen query rows from reading chosen key positions, at 1 block."""

    def __init__(self, queries: list[int], keys: list[int]) -> None:
        self.queries = queries
        self.keys = keys

    def __call__(self, module: Any, args: Any, kwargs: Any) -> Any:
        m = kwargs.get("attention_mask", None)
        if m is None:
            raise RuntimeError(
                "this block got no attention_mask, so there is nothing to "
                "knock out — the model is probably not using eager attention")
        if not isinstance(m, torch.Tensor):
            raise RuntimeError(f"attention_mask is {type(m)}, not a tensor")
        # The loop reuses ONE mask object across every block of this layer
        # type. Never write into it.
        m = m.clone()
        q = torch.as_tensor(self.queries, device=m.device)
        k = torch.as_tensor(self.keys, device=m.device)
        if m.dtype == torch.bool:
            m[..., q[:, None], k[None, :]] = False
        else:
            m[..., q[:, None], k[None, :]] = torch.finfo(m.dtype).min
        kwargs["attention_mask"] = m
        return args, kwargs


def knocked_contrast(
    model: Any, lens_model: Any, inputs: dict, blocks: list[int],
    queries: list[int], keys: list[int], pos_id: int, neg_id: int,
) -> float:
    """`c` when the cut is applied at EVERY block in `blocks`.

    A 1-block cut is weak, and that is a property of the model and not a
    fault. The answer reads the question at all 48 blocks, so closing 1 door
    leaves 47 open: measured, a 1-block cut of the answer-to-question route
    moves `c` by 0.845 out of a range of 17. Only a cut held across a range
    of blocks answers "does this route carry the judgment".
    """
    handles = [lens_model.layers[b].register_forward_pre_hook(
        MaskPatcher(queries, keys), with_kwargs=True) for b in blocks]
    try:
        h = pc.final_residual(model, lens_model, inputs, graph=False)
        return float(pc.fp32_contrast(lens_model, h, pos_id, neg_id)[0])
    finally:
        for h_ in handles:
            h_.remove()


def parse_range(spec: str) -> list[int]:
    """"30-47" -> [30..47]; "42" -> [42]."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def arms(
    block_pos: list[int], stream: dict[str, list[int]], answer_pos: int,
    seq_len: int,
) -> dict[str, tuple[list[int], list[int]]]:
    """arm -> (query rows to blind, key positions to hide)."""
    patch_set = set(block_pos)
    non_patch = [i for i in range(seq_len) if i not in patch_set]
    return {
        # The gate. Blinding the answer to the question must wreck `c`.
        "answer_question": ([answer_pos], list(stream["text_after"])),
        # The measurements.
        "answer_patches": ([answer_pos], list(block_pos)),
        "answer_last": ([answer_pos], [block_pos[-1]]),
        "stream_patches": (non_patch, list(block_pos)),
        # A "changes nothing" arm is a sanity line, NOT a gate.
        "none": ([answer_pos], []),
    }


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, ranges: list[str], out_dir: Path, device: str,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rows: list[dict] = []
    for r in sg.load_single_rows(n_images, case):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        seq_len = int(inputs["input_ids"].shape[1])
        block_pos = [int(x) for x in
                     sw.image_token_blocks(inputs, image_token_id, expect=1)[0]]
        answer_pos = seq_len - 1
        stream = psw.stream_groups(inputs["input_ids"][0], block_pos, answer_pos)
        with torch.no_grad():
            h = pc.final_residual(model, lens_model, inputs, graph=False)
        c0 = float(pc.fp32_contrast(lens_model, h, pos_id, neg_id)[0])
        table = arms(block_pos, stream, answer_pos, seq_len)
        for spec in ranges:
            blocks = parse_range(spec)
            for arm, (queries, keys) in table.items():
                if not keys:
                    c_new = c0
                else:
                    c_new = knocked_contrast(
                        model, lens_model, inputs, blocks, queries, keys,
                        pos_id, neg_id)
                rows.append({
                    "case": case, "image_id": r["image_id"], "blocks": spec,
                    "n_blocks": len(blocks), "arm": arm, "c0": c0,
                    "c_knocked": c_new, "delta": c_new - c0})
    df = pd.DataFrame(rows)
    ctrl = df[(df["arm"] == "answer_question") & (df["blocks"] == "0-47")]
    got = float(ctrl["delta"].abs().mean()) if not ctrl.empty else 0.0
    if got < CONTROL_FLOOR:
        raise RuntimeError(
            f"the positive control moved c by {got:.3f}, below "
            f"{CONTROL_FLOOR}. Cutting the answer off from the question at "
            "EVERY block must collapse c. The mask did not take effect, so no "
            "arm is meaningful.")
    log(f"positive control passed: cutting the answer off from the question "
        f"at all 48 blocks moves c by {got:.2f}")
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "attn_knockout.parquet", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["blocks", "arm"]).agg(
        n=("delta", "size"),
        mean_abs_delta=("delta", lambda s: float(np.mean(np.abs(s)))),
        mean_delta=("delta", "mean"),
    ).reset_index()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--ranges", nargs="+", default=list(DEFAULT_RANGES),
                    help='Block ranges to cut, e.g. 30-47 or 42.')
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
                      n_images=args.n_images, ranges=args.ranges,
                      out_dir=out_dir, device=args.device)
        s = summarize(df)
        log("=" * 70)
        log(f"CASE {case} — how far c moves when attention is cut")
        log("  blocks   arm                n   mean |dc|   mean dc")
        for _, r in s.iterrows():
            log(f"  {r.blocks:<8} {r.arm:<16} {int(r.n):>3}   "
                f"{r.mean_abs_delta:>9.3f}   {r.mean_delta:>+8.3f}")
        (out_dir / case / "summary.json").write_text(
            json.dumps(s.to_dict(orient="records"), indent=2))
    log("attention knockout complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
