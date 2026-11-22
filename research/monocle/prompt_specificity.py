"""Does the map read SAFETY, or does it read the picture?

The problem
-----------
The deletion test showed that the cells of a map rank correctly: take away the
cells it calls Safe and `c` falls, take away the cells it calls Dangerous and
`c` rises. That result is real, but it does not yet earn the caption
"the model finds this square dangerous".

A map is built from `c = logit(positive) - logit(negative)`, so it is tied to
a pair of labels BY CONSTRUCTION. The danger is that the same squares would
come out on top for ANY question. Then the method reads which parts of a
photograph the model looks at, and the label pair only sets the sign.

The test
--------
Read the SAME photograph under several prompts and correlate the maps.

| Prompt | Labels | Relation to safety |
|---|---|---|
| `subway_safety` | Safe / Dangerous | the reference |
| `road_quality` | Good / Poor | a near question about the same street |
| `street_photography` | Good / Poor | a far question, about how it photographs |
| `daylight` | Day / Night | ORTHOGONAL, and purely photometric |

`daylight` is the control that matters. It asks nothing about quality, and
every photograph answers it strongly, so the map cannot be empty. A safety map
that matches the daylight map is not a safety map.

Reading the result
------------------
| r against the safety map | Reading |
|---|---|
| near 1 for every prompt | the method reads saliency; the caption is wrong |
| high for the near question, low for `daylight` | the map reads the question |
| near 0 for all | the maps carry little shared structure at all |

The near question SHOULD correlate somewhat. A cracked, dark, empty street
reads badly under several questions, and that is the world, not an artefact.
The control tells the 2 apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import jlens_read  # noqa: E402
from research.monocle import labels as lb  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/prompt_specificity"
#: The orthogonal control. It is not one of the 7 cases, because none of them
#: is free of a quality judgment. Both labels are 1 token and they do not
#: share a first token, which the label contract demands.
DAYLIGHT = {
    "subject": "New York City street scenes",
    "question": "Was this photograph taken in the day or at night?",
    "positive": "Day", "negative": "Night",
}
#: The prompts to compare, in order. The first is the reference.
DEFAULT_PROMPTS = ("subway_safety", "road_quality", "street_photography",
                   "daylight")
#: Images come from this pool whatever the prompt, so the comparison holds
#: the photograph fixed and moves only the question.
IMAGE_CASE = "subway_safety"


def log(m: str) -> None:
    print(f"[specificity] {m}", flush=True)


def prompt_spec(name: str) -> dict:
    """The rating dict of a named prompt, real case or control."""
    if name == "daylight":
        return dict(DAYLIGHT)
    return sg.rating(name)


def build_for(
    proc: Any, tmpl: str, image: Image.Image, name: str, device: str,
) -> dict:
    """Inputs for 1 image under 1 prompt. The image pool never changes."""
    spec = prompt_spec(name)
    if name == "daylight":
        # A control is not in CASE_RATING, so the base case supplies the
        # template and the override supplies every field of the question.
        return sg.build_single_inputs(
            proc, tmpl, image, IMAGE_CASE, overrides=spec, device=device)
    return sg.build_single_inputs(proc, tmpl, image, name, device=device)


def contrast_ids_for(tokenizer: Any, name: str) -> tuple[int, int]:
    spec = prompt_spec(name)
    ids = lb.check_label_tokens(tokenizer, [spec["positive"], spec["negative"]])
    return ids[spec["positive"]], ids[spec["negative"]]


def compare_maps(maps: dict[str, np.ndarray]) -> list[dict]:
    """Every pair of prompts: correlation, sign agreement, top-cell overlap."""
    out = []
    for a, b in combinations(sorted(maps), 2):
        x, y = maps[a], maps[b]
        k = max(1, x.size // 8)
        top_x = set(np.argsort(x)[-k:].tolist())
        top_y = set(np.argsort(y)[-k:].tolist())
        bot_x = set(np.argsort(x)[:k].tolist())
        bot_y = set(np.argsort(y)[:k].tolist())
        out.append({
            "prompt_a": a, "prompt_b": b,
            "pearson": float(np.corrcoef(x, y)[0, 1])
            if x.std() and y.std() else np.nan,
            "spearman": float(np.corrcoef(
                np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]),
            "sign_agree": float(np.mean(np.sign(x) == np.sign(y))),
            "top_overlap": len(top_x & top_y) / k,
            "bottom_overlap": len(bot_x & bot_y) / k,
        })
    return out


def run(
    proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    prompts: tuple, n_images: int, group: int, seed: int, batch: int,
    out_dir: Path, device: str,
) -> pd.DataFrame:
    tok = proc.tokenizer
    image_token_id = int(model.config.image_token_id)
    ids = {p: contrast_ids_for(tok, p) for p in prompts}
    for p, (a, b) in ids.items():
        log(f"{p}: {prompt_spec(p)['positive']}={a} "
            f"{prompt_spec(p)['negative']}={b}")

    pair_rows, worth_rows = [], []
    for i, r in enumerate(sg.load_single_rows(n_images, IMAGE_CASE,
                                              seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        maps: dict[str, np.ndarray] = {}
        for name in prompts:
            inputs = build_for(proc, tmpl, img, name, device)
            slots = pab.slots_of(inputs, image_token_id)
            pos_id, neg_id = ids[name]
            c0 = float(pab.contrast(
                model, lens_model, inputs, pos_id, neg_id)[0])
            ctl = pab.controls(model, lens_model, inputs, slots, c0,
                               pos_id, neg_id, image_token_id=image_token_id)
            pab.gate_rebuild(ctl)
            maps[name] = -pab.ablation_map(
                model, lens_model, inputs, slots, c0, pos_id, neg_id,
                batch=batch, group=group, image_token_id=image_token_id)
            worth_rows.append({
                "image_id": r["image_id"], "prompt": name, "c": c0,
                "image_worth": ctl["image_worth"],
                "worth_ok": bool(ctl["worth_ok"])})
        for row in compare_maps(maps):
            row.update(image_id=r["image_id"], path=r["path"], group=group)
            pair_rows.append(row)
        log(f"  {i + 1}/{n_images} {r['image_id']} "
            + "  ".join(f"{n}:c={w['c']:+.1f}" for n, w in
                        zip(prompts, worth_rows[-len(prompts):])))

    pairs = pd.DataFrame(pair_rows)
    worth = pd.DataFrame(worth_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    worth.to_parquet(out_dir / "worth.parquet", index=False)

    ref = prompts[0]
    log(f"the picture is worth (median over images), by prompt:")
    for name in prompts:
        w = worth[worth.prompt == name]
        log(f"  {name:<20} c={w.c.median():+7.2f}  worth={w.image_worth.median():+7.2f}"
            f"  strong={int(w.worth_ok.sum())}/{len(w)}")
    log(f"map against the {ref} map:")
    log(f"{'prompt':<20} {'pearson':>9} {'spearman':>9} {'sign':>7} "
        f"{'top 1/8':>9} {'bottom 1/8':>11}")
    for name in prompts[1:]:
        m = pairs[((pairs.prompt_a == ref) & (pairs.prompt_b == name)) |
                  ((pairs.prompt_b == ref) & (pairs.prompt_a == name))]
        log(f"{name:<20} {m.pearson.median():>9.3f} "
            f"{m.spearman.median():>9.3f} {m.sign_agree.median():>7.2f} "
            f"{m.top_overlap.median():>9.2f} {m.bottom_overlap.median():>11.2f}")
    (out_dir / "summary.json").write_text(json.dumps({
        "reference": ref, "prompts": list(prompts), "group": group,
        "n_images": int(worth.image_id.nunique()),
        "median_pearson_against_reference": {
            name: float(pairs[((pairs.prompt_a == ref) &
                               (pairs.prompt_b == name)) |
                              ((pairs.prompt_b == ref) &
                               (pairs.prompt_a == name))].pearson.median())
            for name in prompts[1:]},
    }, indent=2))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Is the map about the question?")
    ap.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    ap.add_argument("--n-images", type=int, default=12)
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
    for p in args.prompts:
        prompt_spec(p)          # raises early on a name that does not exist
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    run(proc, model, lens_model, tmpl, prompts=tuple(args.prompts),
        n_images=args.n_images, group=args.group, seed=args.seed,
        batch=args.batch, out_dir=Path(args.out_dir), device=args.device)
    log("prompt specificity complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
