"""Which WORD of the question carries the judgment, and when?

The gap
-------
The relay says the judgment sits in the question tokens over blocks 25 to 32,
and it peaks at L27 at a recovery of 0.912. That was measured by swapping ALL
the question tokens at once, so it does not say which of them holds anything.

The jacobian lens cannot answer this. Two lenses fitted on different corpora
correlate at only 0.46 to 0.82 on those positions, so a token readout there
describes the fitting corpus as much as the model.

The swap can answer it, and it needs no lens and no corpus.

The measurement
---------------
For 1 question-token position `p` and 1 layer `l`, replace `h_l[p]` with the
donor's value at the same position, and measure how far `c` moves toward the
donor. The prompt is byte identical for every photograph, so position `p`
holds the SAME word in both, and the 2 residuals differ only by what each has
absorbed from its own picture.

That property is why this needs the single-image path. On the pairwise path
the text between the 2 images differs by pair.

Reading the result
------------------
A (token, layer) grid of recovery. A word that carries the judgment shows a
band; a word that carries none stays at 0. The last question token sits next
to the answer, so it is expected to be the strongest — the question is whether
anything ELSE carries a share, and which words those are.

`text_before` rides along as a null control at every layer. Those positions
precede the image, so they cannot differ between 2 photographs, and a reading
other than 0.000 there means the measurement is broken.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_causal as pc  # noqa: E402
from research.monocle import patch_swap as psw  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/question_tokens"
#: The handover band the relay found, plus a layer on each side of it.
DEFAULT_LAYERS = tuple(range(22, 36))
#: The whole-question swap must reach this inside the band, or the run is void.
BAND_FLOOR = 0.4
#: A pair whose 2 judgments are closer than this is skipped. Recovery divides
#: by that gap, so a small gap turns bf16 noise into a huge number. This is
#: the same guard `patch_swap` uses, and the first version of this module
#: lacked it: the null control read 103.6 instead of 0.
MIN_GAP = 1.0
#: Every row is measured against a self row in its OWN batch, so the residual
#: numerical noise is tiny. The null control must clear this.
NULL_FLOOR = 1e-3


def log(m: str) -> None:
    print(f"[qtokens] {m}", flush=True)


def question_positions(
    input_ids: torch.Tensor, block: list[int], answer_pos: int,
) -> tuple[list[int], list[int]]:
    """(the question token positions, the null-control positions)."""
    groups = psw.stream_groups(input_ids, block, answer_pos)
    return list(groups["text_after"]), list(groups["text_before"])


def recovery(c_swapped: float, c_target: float, c_donor: float) -> float:
    """0 when nothing moved, 1 when the judgment reached the donor's."""
    gap = c_donor - c_target
    if abs(gap) < 1e-6:
        return float("nan")
    return float((c_swapped - c_target) / gap)


def swap_positions(
    model: Any, lens_model: Any, inputs: dict, layer: int,
    sets: list[list[int]], donor_h: torch.Tensor, own_h: torch.Tensor,
    pos_id: int, neg_id: int, batch: int,
) -> list[float]:
    """The CHANGE in `c` from replacing each set with the donor's residual.

    **Each set is measured against a SELF swap of the SAME positions, in the
    same batch.** Row `i` writes the donor's values at those positions; row
    `n+i` writes the TARGET'S OWN recorded values at exactly those positions.
    The change is the difference.

    Two earlier designs failed, and the second failure is the instructive one.

    | Attempt | Zero used | Null control read |
    |---|---|---|
    | a `c0` from its own forward | a separate batch of 1 | 103.6 (recovery) |
    | 1 self row, at other positions | same batch, wrong positions | 0.488 (dc) |
    | a self row at the SAME positions | same batch, same positions | see the gate |

    The second is subtle. The recorded activations come from a batch-1
    forward, and they are written into a batch-13 forward, where the natural
    values differ by bf16 accumulation. So writing a RECORDED value is never a
    true no-op, whatever the value is. Only a self swap at the same positions
    subtracts that away, and it leaves exactly the donor-against-target
    difference we want.
    """
    from research.monocle.patch_map_causal import BatchPatcher

    if not sets:
        return []
    out: list[float] = []
    for start in range(0, len(sets), batch):
        block = list(sets[start:start + batch])
        n = len(block)
        while len(block) < batch:
            block.append(block[-1])
        # n donor rows, then n self rows over the SAME position sets.
        chunk = [list(g) for g in block] * 2
        vals = ([donor_h[g, :] for g in block]
                + [own_h[g, :] for g in block])
        rep = pc.replicate(inputs, len(chunk))
        handle = lens_model.layers[layer].register_forward_hook(
            BatchPatcher(chunk, vals))
        try:
            with torch.no_grad():
                h = pc.final_residual(model, lens_model, rep, graph=False)
            c = pc.fp32_contrast(lens_model, h, pos_id, neg_id)
        finally:
            handle.remove()
        cs = c.float().cpu().numpy()
        m = len(block)
        out.extend(float(cs[i] - cs[m + i]) for i in range(n))
    return out


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, layers: tuple, seed: int, batch: int, out_dir: Path,
    device: str,
) -> pd.DataFrame:
    from jlens.hooks import ActivationRecorder

    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)

    store = []
    for r in sg.load_single_rows(n_images, case, seed=seed):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        block = pab.slots_of(inputs, image_token_id)
        answer_pos = int(inputs["input_ids"].shape[1]) - 1
        with ActivationRecorder(lens_model.layers,
                                at=sorted(set(layers))) as rec:
            with torch.no_grad():
                model(**inputs, use_cache=False)
            acts = {i: rec.activations[i][0].detach()
                    for i in rec.activations}
        c0 = float(pab.contrast(model, lens_model, inputs, pos_id, neg_id)[0])
        store.append({**r, "inputs": inputs, "block": block, "c": c0,
                      "acts": acts, "answer_pos": answer_pos})
        log(f"  read {len(store)}/{n_images} {r['image_id']} c={c0:+.2f}")

    ids0 = store[0]["inputs"]["input_ids"][0]
    q_pos, null_pos = question_positions(
        ids0, store[0]["block"], store[0]["answer_pos"])
    words = [tok.decode([int(ids0[p])]) for p in q_pos]
    log(f"{len(q_pos)} question tokens: "
        + " ".join(f"{i}:{w!r}" for i, w in enumerate(words)))
    log(f"{len(null_pos)} null-control positions")

    rows = []
    skipped = 0
    for i, tgt in enumerate(store):
        # The donor whose judgment is FARTHEST away. Recovery divides by that
        # gap, so the best-conditioned donor is the most distant one.
        others = [d for j, d in enumerate(store) if j != i]
        don = max(others, key=lambda d: abs(d["c"] - tgt["c"]))
        gap = don["c"] - tgt["c"]
        if abs(gap) < MIN_GAP:
            skipped += 1
            log(f"  {tgt['image_id']}: the widest gap is {gap:+.3f}, under "
                f"{MIN_GAP}; skipped")
            continue
        for layer in layers:
            donor_h = don["acts"][layer]
            own_h = tgt["acts"][layer]
            sets = [[p] for p in q_pos] + [list(q_pos), list(null_pos)]
            # `swap_positions` returns CHANGES, each against a self row that
            # sat in the same batch.
            cs = swap_positions(model, lens_model, tgt["inputs"], layer, sets,
                                donor_h, own_h, pos_id, neg_id, batch)
            for j, c in enumerate(cs):
                if j < len(q_pos):
                    name, kind = words[j], "token"
                elif j == len(q_pos):
                    name, kind = "<all question>", "whole"
                else:
                    name, kind = "<text before>", "null"
                rows.append({
                    "case": case, "image_id": tgt["image_id"],
                    "donor_id": don["image_id"], "layer": int(layer),
                    "slot": j if j < len(q_pos) else -1,
                    "token": name, "kind": kind,
                    "c_target": tgt["c"], "c_donor": don["c"],
                    "c": tgt["c"] + c, "dc": c,
                    "recovery": c / (don["c"] - tgt["c"])})
        log(f"  swapped {i + 1}/{len(store)} {tgt['image_id']} "
            f"(donor {don['image_id']}, gap {gap:+.2f})")
    if skipped:
        log(f"{skipped} images had no donor far enough away")

    df = pd.DataFrame(rows)
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "question_tokens.parquet", index=False)

    # Judge the null on |dc|, never on recovery. Recovery divides by the gap,
    # so it reports a ratio of 2 small numbers and says nothing about whether
    # the swap moved anything.
    worst = float(df[df.kind == "null"].dc.abs().max())
    log(f"null control ({int((df.kind == 'null').sum())} rows): "
        f"worst |dc| = {worst:.6f}, allowed {NULL_FLOOR:.6f}")
    if worst > NULL_FLOOR:
        raise SystemExit(
            f"the null control moves `c` by {worst:.5f}. Positions before the "
            "image cannot differ between 2 photographs, and every row is now "
            "measured against a self row in its own batch, so this is a real "
            "fault and not a batching offset.")

    whole = df[df.kind == "whole"].groupby("layer").recovery.median()
    peak = float(whole.max())
    log(f"the whole-question swap peaks at {peak:.3f} "
        f"(L{int(whole.idxmax())})")
    if peak < BAND_FLOOR:
        raise SystemExit(
            f"the whole-question swap only reaches {peak:.3f}, under "
            f"{BAND_FLOOR}. This band does not hold the judgment, so no "
            "per-token reading from it means anything.")

    tokens = df[df.kind == "token"]
    piv = tokens.groupby(["slot", "layer"]).recovery.median().unstack()
    log("recovery by question token and layer (median over images):")
    head = "  ".join(f"L{l}" for l in piv.columns)
    log(f"{'slot':>4} {'token':<14} {head}")
    for slot in piv.index:
        w = words[int(slot)].replace("\n", "\\n")
        vals = "  ".join(f"{piv.loc[slot, l]:>+.2f}" for l in piv.columns)
        log(f"{int(slot):>4} {w[:14]:<14} {vals}")
    (case_dir / "summary.json").write_text(json.dumps({
        "case": case, "n_images": len(store), "layers": list(layers),
        "words": words, "null_worst": worst, "whole_peak": peak,
        "whole_peak_layer": int(whole.idxmax()),
    }, indent=2))
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+",
                    default=list(DEFAULT_LAYERS))
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar

    args = build_parser().parse_args(argv)
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    for case in args.cases:
        run_case(case, proc, model, lens_model, tmpl,
                 n_images=args.n_images, layers=tuple(args.layers),
                 seed=args.seed, batch=args.batch,
                 out_dir=Path(args.out_dir), device=args.device)
    log("question tokens complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
