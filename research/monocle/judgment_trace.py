"""Where the judgment sits, at every position and every depth.

The measurement
---------------
For 1 layer `l` and 1 position `p`, transport the residual to the output space
and read the 2 label rows only:

    s_l(p) = logit_pos( J_l @ h_l[p] ) - logit_neg( J_l @ h_l[p] )

That is the contrast `c`, evaluated as if the model stopped reading at `p` and
`l`. A map of `s` over (position, layer) is a picture of the relay that the
swap and the knockout found by intervention:

| Blocks | What holds the judgment |
|---|---|
| 0-21 | the image patches |
| 22-29 | the question tokens |
| 30-47 | the answer position |

The swap and the knockout say WHERE a cut hurts. This says WHAT each place
carries. The 2 must agree, and where they do not, the lens is the suspect.

Why 2 lenses
------------
`answer_tokens` recorded that a layer at or below L30 is corpus-dominated: a
wikitext lens and an urban lens emit near-disjoint token sets there, at a
top-10 Jaccard of about 0. The question tokens live at L22-L29, right inside
that regime, so a single-lens reading of them cannot be trusted.

Both fitted lenses carry the SAME layers, `[6, 12, 18, 24, 30, 36, 42, 46]`.
So every number here is measured twice, once through each lens, and the report
carries their agreement. A layer where the 2 lenses disagree describes the
fitting corpus, not gemma-4.

The 0.8 floor gates the full-position content readout. It correlates the 2
lenses over every position of 1 prompt. A 2-token label contrast at the answer
position is a narrower read, and it needs its own number.
`research/scripts/lens_agreement_answer.py` measures the agreement for that narrow read
across images. Read that number, and not the wide floor, when the claim is
about the answer position.

The end point
-------------
Layer 47 needs no transport: the model's own final block feeds its own head.
So `s_47(answer)` must equal `c` exactly. `trace` checks it and refuses to
return a frame that fails, in the same way the group sweep checks its 1-cell
end point.

Scale
-----
Gemma carries massive activations, so a raw projection is dominated by a few
channels and by the norm of the position, not by its content. Every row thus
carries `s` and `s_centered`, where the centring runs over the positions of
the same layer. Read the centred column. The raw column is kept because a
change in the raw value across layers is still the quantity the head sees.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
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

OUT_DEFAULT = REPO / "outputs/_monocle/judgment_trace"
LENSES = {
    "wikitext": REPO / "outputs/_monocle/jlens/gemma4_12b_lens.pt",
    "mm": REPO / "outputs/_monocle/jlens/mm/gemma4_12b_lens.pt",
}
#: The last block feeds the head with no transport, so it is exact.
FINAL_LAYER = 47
#: `s_47(answer)` must reproduce `c` to this much, or the trace is wrong.
END_TOL = 1e-3
#: Two lenses that correlate below this describe their corpora, not the model.
AGREE_FLOOR = 0.8


def log(m: str) -> None:
    print(f"[trace] {m}", flush=True)


def position_groups(
    input_ids: torch.Tensor, block: list[int], answer_pos: int,
    image_token_id: int = 258880,
) -> dict[str, list[int]]:
    """Every position, in named groups.

    2 checks, and the second is the one that matters. Coverage alone is weak:
    a short `block` still covers the sequence, because the patches it drops
    fall into `text_after` and the relay picture then shows image content
    inside the question. So the image group is checked against the TOKEN IDS.
    """
    groups = dict(psw.stream_groups(input_ids, block, answer_pos))
    groups["image"] = list(block)
    groups["answer"] = [answer_pos]

    n = int(input_ids.shape[0])
    seen: list[int] = []
    for v in groups.values():
        seen.extend(v)
    if sorted(seen) != list(range(n)):
        missing = sorted(set(range(n)) - set(seen))
        extra = sorted([p for p in seen if seen.count(p) > 1])
        raise ValueError(
            f"the groups do not cover the {n} positions: {len(missing)} "
            f"missing {missing[:8]}, repeated {sorted(set(extra))[:8]}")

    truth = {i for i, t in enumerate(input_ids.tolist())
             if int(t) == image_token_id}
    if set(block) != truth:
        raise ValueError(
            f"the image group holds {len(block)} positions but the sequence "
            f"holds {len(truth)} image tokens. A patch inside `text_after` "
            "would read as question content.")
    return groups


def score_positions(
    lens: Any, lens_model: Any, h: torch.Tensor, layer: int,
    pos_id: int, neg_id: int,
) -> torch.Tensor:
    """`s` for every position of 1 layer. `h` is [seq, d]."""
    if layer == FINAL_LAYER:
        x = h.float()
    else:
        if layer not in lens.jacobians:
            raise ValueError(
                f"layer {layer} is not fitted; the lens holds "
                f"{lens.source_layers}")
        x = lens.transport(h.float(), layer)
    return pc.fp32_contrast(lens_model, x, pos_id, neg_id)


def trace(
    model: Any, lens_model: Any, lenses: dict, inputs: dict,
    pos_id: int, neg_id: int, layers: list[int], tokenizer: Any,
    image_token_id: int,
) -> tuple[pd.DataFrame, float]:
    """(1 row for each lens, layer and position; the model's own `c`)."""
    from jlens.hooks import ActivationRecorder

    ids = inputs["input_ids"][0]
    block = pab.slots_of(inputs, image_token_id)
    answer_pos = int(ids.shape[0]) - 1
    groups = position_groups(ids, block, answer_pos, image_token_id)
    where = {p: g for g, ps in groups.items() for p in ps}

    record = sorted(set(layers) | {FINAL_LAYER})
    with ActivationRecorder(lens_model.layers, at=record) as rec:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        acts = {i: rec.activations[i][0].detach() for i in rec.activations}
    c_model = float(pc.fp32_contrast(
        lens_model, acts[FINAL_LAYER][answer_pos:answer_pos + 1, :],
        pos_id, neg_id)[0])

    toks = [tokenizer.decode([int(t)]) for t in ids]
    rows = []
    for name, lens in lenses.items():
        for layer in record:
            s = score_positions(
                lens, lens_model, acts[layer], layer, pos_id, neg_id)
            s = s.float().cpu().numpy()
            centre = float(s.mean())
            for p in range(len(s)):
                rows.append({
                    "lens": name, "layer": layer, "pos": p,
                    "group": where[p], "token": toks[p],
                    "s": float(s[p]), "s_centered": float(s[p]) - centre,
                })
    df = pd.DataFrame(rows)

    end = df[(df.layer == FINAL_LAYER) & (df.group == "answer")]
    off = float((end.s - c_model).abs().max())
    if off > END_TOL:
        raise SystemExit(
            f"s_47(answer) is off `c` by {off:.5f}, over {END_TOL}. The "
            "readout does not reproduce the model, so read nothing from it.")
    return df, c_model


def lens_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per layer: do the 2 lenses tell the same story across positions?"""
    names = sorted(df.lens.unique())
    if len(names) < 2:
        return pd.DataFrame()
    a, b = names[:2]
    out = []
    for layer, sub in df.groupby("layer"):
        x = sub[sub.lens == a].sort_values("pos").s_centered.to_numpy()
        y = sub[sub.lens == b].sort_values("pos").s_centered.to_numpy()
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else np.nan
        out.append({
            "layer": int(layer), "lens_a": a, "lens_b": b, "pearson": r,
            "trustworthy": bool(r >= AGREE_FLOOR),
            "sign_agree": float(np.mean(np.sign(x) == np.sign(y))),
        })
    return pd.DataFrame(out)


def by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Per (lens, layer, group): the summary the relay is read from."""
    g = df.groupby(["lens", "layer", "group"]).agg(
        n=("pos", "size"),
        s_mean=("s", "mean"), s_max=("s", "max"), s_min=("s", "min"),
        centered_mean=("s_centered", "mean"),
        centered_absmax=("s_centered", lambda v: float(
            np.abs(np.asarray(v)).max())),
    ).reset_index()
    return g


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str,
    lenses: dict, *, n_images: int, seed: int, out_dir: Path, device: str,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    layers = sorted(set(next(iter(lenses.values())).source_layers))
    log(f"{case}: lens layers {layers} plus the exact layer {FINAL_LAYER}")

    frames, agrees = [], []
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        df, c_model = trace(model, lens_model, lenses, inputs, pos_id, neg_id,
                            layers, tok, image_token_id)
        df.insert(0, "image_id", r["image_id"])
        df.insert(1, "case", case)
        df["c_model"] = c_model
        frames.append(df)
        ag = lens_agreement(df)
        ag.insert(0, "image_id", r["image_id"])
        agrees.append(ag)
        log(f"  {i + 1}/{n_images} {r['image_id']} c={c_model:+.2f}")

    all_df = pd.concat(frames, ignore_index=True)
    ag_df = pd.concat(agrees, ignore_index=True)
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(case_dir / "trace.parquet", index=False)
    ag_df.to_parquet(case_dir / "agreement.parquet", index=False)
    grp = by_group(all_df)
    grp.to_parquet(case_dir / "by_group.parquet", index=False)

    log("lens agreement over positions (a layer under "
        f"{AGREE_FLOOR} reads its corpus, not the model):")
    log(f"{'layer':>6} {'pearson':>9} {'sign':>7} {'verdict':>12}")
    for layer, sub in ag_df.groupby("layer"):
        r = sub.pearson.median()
        log(f"{layer:>6} {r:>9.3f} {sub.sign_agree.median():>7.2f} "
            f"{'trustworthy' if r >= AGREE_FLOOR else 'CORPUS':>12}")

    log("the relay, centred `s` by group (mm lens):")
    mm = grp[grp.lens == "mm"]
    cols = ["image", "text_after", "answer"]
    log(f"{'layer':>6} " + " ".join(f"{c:>13}" for c in cols))
    for layer, sub in mm.groupby("layer"):
        vals = []
        for c in cols:
            row = sub[sub.group == c]
            vals.append(f"{row.centered_absmax.iloc[0]:>13.3f}"
                        if len(row) else f"{'-':>13}")
        log(f"{layer:>6} " + " ".join(vals))
    return all_df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260821)
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
    lenses = {}
    for name, path in LENSES.items():
        if not Path(path).exists():
            raise SystemExit(f"no lens at {path}")
        lenses[name] = jlens_read.load_lens(str(path))
        log(f"loaded the {name} lens: {lenses[name].source_layers}")
    a, b = (set(l.source_layers) for l in lenses.values())
    if a != b:
        raise SystemExit(
            f"the 2 lenses hold different layers ({sorted(a)} against "
            f"{sorted(b)}), so they cannot be compared layer by layer")
    for case in args.cases:
        run_case(case, proc, model, lens_model, tmpl, lenses,
                 n_images=args.n_images, seed=args.seed, out_dir=out_dir,
                 device=args.device)
    log("judgment trace complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
