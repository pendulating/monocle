"""Does the forced JSON prefix change what `c` means?

`build_single_inputs` accepted a `force_prefix` and then dropped it, so every
single-image run before 2026-08-23 02:40 read `c` at the end of the generation
prompt. The model's natural next token there is `{`, not a label.

| Form | The last position | What `c` is |
|---|---|---|
| no prefix | end of the generation prompt | a DISPOSITION toward the labels |
| the prefix | after `{\\n  "answer": "` | the ANSWER distribution |

The second is the production quantity. This module measures how far apart the
2 are, so the earlier results can be kept, restated, or thrown away on
evidence rather than on a guess.

It reports 3 things:

1. the correlation of `c` between the forms, over images;
2. the correlation of the 64-cell absence MAP between the forms, per image,
   which is what every colour claim rests on;
3. whether the map's top and bottom cells survive the change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/prefix_check"


def log(m: str) -> None:
    print(f"[prefix] {m}", flush=True)


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, group: int, seed: int, batch: int, out_dir: Path,
    device: str,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    forms = {"with_prefix": sg.FORCE_PREFIX, "no_prefix": ""}

    rows = []
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        got = {}
        for name, prefix in forms.items():
            inputs = sg.build_single_inputs(
                proc, tmpl, img, case, force_prefix=prefix, device=device)
            slots = pab.slots_of(inputs, image_token_id)
            c0 = float(pab.contrast(
                model, lens_model, inputs, pos_id, neg_id)[0])
            ctl = pab.controls(model, lens_model, inputs, slots, c0, pos_id,
                               neg_id, image_token_id=image_token_id)
            pab.gate_rebuild(ctl)
            got[name] = {
                "c": c0, "worth": ctl["image_worth"],
                "n_tokens": int(inputs["input_ids"].shape[1]),
                "map": -pab.ablation_map(
                    model, lens_model, inputs, slots, c0, pos_id, neg_id,
                    batch=batch, group=group,
                    image_token_id=image_token_id)}
        a, b = got["with_prefix"]["map"], got["no_prefix"]["map"]
        k = max(1, a.size // 8)
        rows.append({
            "case": case, "image_id": r["image_id"], "group": group,
            "c_with": got["with_prefix"]["c"], "c_without": got["no_prefix"]["c"],
            "worth_with": got["with_prefix"]["worth"],
            "worth_without": got["no_prefix"]["worth"],
            "tokens_with": got["with_prefix"]["n_tokens"],
            "tokens_without": got["no_prefix"]["n_tokens"],
            "map_pearson": float(np.corrcoef(a, b)[0, 1]),
            "map_spearman": float(np.corrcoef(
                np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]),
            "map_sign_agree": float(np.mean(np.sign(a) == np.sign(b))),
            "top_overlap": len(set(np.argsort(a)[-k:].tolist()) &
                               set(np.argsort(b)[-k:].tolist())) / k,
            "bottom_overlap": len(set(np.argsort(a)[:k].tolist()) &
                                  set(np.argsort(b)[:k].tolist())) / k,
        })
        log(f"  {i + 1}/{n_images} {r['image_id']} "
            f"c {rows[-1]['c_without']:+.2f} -> {rows[-1]['c_with']:+.2f}   "
            f"map r={rows[-1]['map_pearson']:+.3f}")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{case}_g{group}.parquet", index=False)

    r_c = float(np.corrcoef(df.c_with, df.c_without)[0, 1])
    r_w = float(np.corrcoef(df.worth_with, df.worth_without)[0, 1])
    log(f"{case}: {len(df)} images, group {group}")
    log(f"  the prefix adds {int((df.tokens_with - df.tokens_without).median())}"
        " tokens")
    log(f"  c:           r = {r_c:+.3f}   "
        f"median {df.c_without.median():+.2f} -> {df.c_with.median():+.2f}")
    log(f"  image worth: r = {r_w:+.3f}   "
        f"median {df.worth_without.median():+.2f} -> "
        f"{df.worth_with.median():+.2f}")
    log(f"  MAP pearson  median {df.map_pearson.median():+.3f} "
        f"(range {df.map_pearson.min():+.3f} .. {df.map_pearson.max():+.3f})")
    log(f"  MAP spearman median {df.map_spearman.median():+.3f}")
    log(f"  sign agree   median {df.map_sign_agree.median():.3f}")
    log(f"  top 1/8 kept {df.top_overlap.median():.3f}   "
        f"bottom 1/8 kept {df.bottom_overlap.median():.3f}  (chance 0.125)")
    (out_dir / f"{case}_g{group}_summary.json").write_text(json.dumps({
        "case": case, "group": group, "n_images": len(df),
        "c_pearson": r_c, "worth_pearson": r_w,
        "map_pearson_median": float(df.map_pearson.median()),
        "map_spearman_median": float(df.map_spearman.median()),
        "top_overlap_median": float(df.top_overlap.median()),
        "bottom_overlap_median": float(df.bottom_overlap.median()),
    }, indent=2))
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--group", type=int, default=2)
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
                 n_images=args.n_images, group=args.group, seed=args.seed,
                 batch=args.batch, out_dir=Path(args.out_dir),
                 device=args.device)
    log("prefix check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
