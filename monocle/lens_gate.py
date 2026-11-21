"""The gate of step 1: do 2 lenses agree at the question positions?

Fit A (2 disjoint WikiText halves), fit B (1,000 WikiText prompts) and fit C
(text-after-image sources) exist to close one claim: "the lens cannot read
the question positions". This script measures every pair of lenses on
held-out images, under the single-image case prompts, at 3 position groups:

| Group | Positions |
|-------|-----------|
| `image` | the 256 patch tokens |
| `text_after` | the question and the generation prompt |
| `answer` | the forced-prefix position that emits the label |

Three agreements for each (pair, layer, group):

| Column | What |
|--------|------|
| `s_pearson` | the contrast `s = logit(pos) - logit(neg)` across the positions of the group, centred over the layer; the 0.8 floor of `judgment_trace` |
| `vocab_r` | the median over positions of the Pearson between the 2 full-vocabulary logit vectors |
| `top10_jaccard` | the mean over positions of the top-10 token overlap |

The held-out images are entries 128 onward of `mm/fit_images.json`: no
multimodal fit used them, and a text fit never saw an image. The case prompt
rotates over the 7 cases by image index.

    sbatch monocle/monocle.sub monocle.lens_gate --n-images 24
    sbatch monocle/monocle.sub monocle.lens_gate --lenses wikitext=... varA0=... --n-images 24

A lens whose file is absent is skipped with a note, thus the gate runs on
the lenses that exist. Two lenses at least must load.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import judgment_trace as jt  # noqa: E402
from monocle import patch_ablate as pab  # noqa: E402
from monocle import single_steer as ss  # noqa: E402
from monocle.jlens_read import load_lens  # noqa: E402
from monocle.jlens_steer import load_model_and_lens  # noqa: E402

JL = REPO / "outputs/_monocle/jlens"
DEFAULT_LENSES = {
    "wikitext": JL / "gemma4_12b_lens.pt",
    "mm": JL / "mm/gemma4_12b_lens.pt",
    "varA0": JL / "varA/half0/gemma4_12b_lens.pt",
    "varA1": JL / "varA/half1/gemma4_12b_lens.pt",
    "wikitext1k": JL / "wikitext1k/gemma4_12b_lens.pt",
    "mmC": JL / "mm_textafter/gemma4_12b_lens.pt",
}
HELDOUT_IMAGES = JL / "mm/fit_images.json"
HELDOUT_START = 128
OUT_DEFAULT = REPO / "outputs/_monocle/lens_gate"
FLOOR = jt.AGREE_FLOOR


def log(m: str) -> None:
    print(f"[gate] {m}", flush=True)


def parse_lenses(specs: list[str] | None) -> dict[str, Path]:
    if not specs:
        return dict(DEFAULT_LENSES)
    out = {}
    for s in specs:
        name, _, path = s.partition("=")
        if not path:
            raise ValueError(f"--lenses takes name=path, not {s!r}")
        out[name] = Path(path)
    return out


def vocab_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Pearson between 2 [n, vocab] tensors, on the device."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1)
    return num / den.clamp_min(1e-12)


def top_jaccard(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> torch.Tensor:
    ta = a.topk(k, dim=1).indices
    tb = b.topk(k, dim=1).indices
    out = torch.empty(a.shape[0])
    for i in range(a.shape[0]):
        sa, sb = set(ta[i].tolist()), set(tb[i].tolist())
        out[i] = len(sa & sb) / len(sa | sb)
    return out


def gate_image(model, lens_model, lenses: dict, inputs: dict, layers: list[int],
               pos_id: int, neg_id: int, image_token_id: int) -> list[dict]:
    from jlens.hooks import ActivationRecorder

    ids = inputs["input_ids"][0]
    block = pab.slots_of(inputs, image_token_id)
    answer_pos = int(ids.shape[0]) - 1
    groups = jt.position_groups(ids, block, answer_pos, image_token_id)
    with ActivationRecorder(lens_model.layers, at=layers) as rec:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        acts = {l: rec.activations[l][0].detach() for l in layers}

    rows = []
    names = list(lenses)
    for layer in layers:
        h = acts[layer].float()
        logits = {n: lens_model.unembed(lenses[n].transport(h, layer)).float()
                  for n in names}
        s = {n: (logits[n][:, pos_id] - logits[n][:, neg_id]) for n in names}
        s = {n: (v - v.mean()).cpu().numpy() for n, v in s.items()}
        for a, b in itertools.combinations(names, 2):
            vr = vocab_pearson(logits[a], logits[b]).cpu().numpy()
            jac = top_jaccard(logits[a], logits[b]).numpy()
            for g, ps in groups.items():
                if not ps:
                    continue
                idx = np.asarray(ps)
                xa, xb = s[a][idx], s[b][idx]
                if len(idx) >= 3 and xa.std() > 0 and xb.std() > 0:
                    sp = float(np.corrcoef(xa, xb)[0, 1])
                else:
                    sp = float("nan")
                rows.append({
                    "lens_a": a, "lens_b": b, "layer": int(layer), "group": g,
                    "n_pos": int(len(idx)), "s_pearson": sp,
                    "vocab_r": float(np.median(vr[idx])),
                    "top10_jaccard": float(np.mean(jac[idx])),
                })
        del logits
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lenses", nargs="*", help="name=path; default: the 6 of step 1")
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--images", default=str(HELDOUT_IMAGES))
    ap.add_argument("--start", type=int, default=HELDOUT_START,
                    help="first image index; 128 skips every image a mm fit saw")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--layers", default=None,
                    help="comma list; default: the layers every lens holds")
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)

    from PIL import Image

    specs = parse_lenses(args.lenses)
    lenses = {}
    for name, path in specs.items():
        if not path.exists():
            log(f"no lens at {path}; {name} skipped")
            continue
        lenses[name] = load_lens(str(path))
        log(f"loaded {name}: layers {lenses[name].source_layers}")
    if len(lenses) < 2:
        raise SystemExit("fewer than 2 lenses loaded; nothing to compare")
    common = set.intersection(*(set(l.source_layers) for l in lenses.values()))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted(common))
    missing = [l for l in layers if l not in common]
    if missing:
        raise SystemExit(f"layers {missing} are not in every lens ({sorted(common)})")

    cases = list(ss.CASE_WORDS) if args.cases == "all" else args.cases.split(",")
    entries = json.load(open(args.images))[args.start:args.start + args.n_images]
    if len(entries) < args.n_images:
        log(f"only {len(entries)} held-out images from index {args.start}")

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    image_token_id = int(model.config.image_token_id)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for i, e in enumerate(entries):
        case = cases[i % len(cases)]
        pos_id, neg_id = ss.polar_token_ids(proc.tokenizer, case)
        text = ss.render_single_chat(proc.tokenizer, tmpl, ss.user_text(case))
        img = Image.open(e["path"]).convert("RGB")
        inputs = dict(proc(text=[text], images=[[img]], return_tensors="pt").to("cuda:0"))
        r = gate_image(model, lens_model, lenses, inputs, layers, pos_id, neg_id,
                       image_token_id)
        for row in r:
            row["image_id"] = e.get("recording_id", Path(e["path"]).name)
            row["case"] = case
        rows += r
        log(f"  {i + 1}/{len(entries)} {case} ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "agreement.parquet", index=False)
    med = (df.groupby(["lens_a", "lens_b", "group", "layer"])
             [["s_pearson", "vocab_r", "top10_jaccard"]].median().reset_index())
    med.to_csv(out_dir / "agreement_median.csv", index=False)
    pd.set_option("display.width", 220)
    for g in ("text_after", "answer", "image"):
        sub = med[med.group == g]
        if sub.empty:
            continue
        log(f"median s_pearson at group {g!r} (floor {FLOOR}):")
        print(sub.pivot_table(index=["lens_a", "lens_b"], columns="layer",
                              values="s_pearson").round(2).to_string())
        log(f"median vocab_r at group {g!r}:")
        print(sub.pivot_table(index=["lens_a", "lens_b"], columns="layer",
                              values="vocab_r").round(2).to_string())
    log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
