"""E1b and E5c of the lens-refusal probe: the words at the positions.

For 1 probe and a draw of refused and answered images of the E1 census,
1 forward each. At every position group and layer the lens the gate
chose transports the residual to the final basis and the unembed gives a
next-token distribution:

| Group | Layers | Lens | Reason |
|-------|--------|------|--------|
| image (the patches) | 18, 24, 36, 42 | `mm` | patch sources; the matched estimand (E1b). Blocks 36 to 42 read scene words, 18 to 24 read the register (`monocle-instrument-traps`) |
| text_after (the question tokens) | 30, 34, 38, 42 | `mm_textafter` (C), with `wikitext1k` (B) beside it | text-after sources (E5c); B marks the tokens the estimand does not set |
| answer | 36, 38, 40, 42, 44 | C, with B beside it | the read |

Per (image, layer, position) it records the top-k tokens and the rank of
the 3 label tokens. The top-k goes through `scoring.build_token_mask`, the
word-cloud mask that drops special, control, and non-letter tokens; the
label ranks come from the full vocabulary. The patch residual is question-blind (the image tokens
precede the question), so E1b describes the scene the model sees, not
the inference.

Writes `outputs/_monocle/probe/scrape/<probe>/{tokens.parquet,label_ranks.parquet,summary.json}`.

    sbatch monocle/monocle.sub monocle.probe_scrape --probe race --n-images 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import judgment_trace as jt  # noqa: E402
from monocle import patch_ablate as pab  # noqa: E402
from monocle import probe_frame as pf  # noqa: E402
from monocle.jlens_read import load_lens  # noqa: E402
from monocle.scoring import build_token_mask  # noqa: E402
from monocle.jlens_steer import load_model_and_lens  # noqa: E402
from monocle.probe_vectors import load_scores  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"
JL = REPO / "outputs/_monocle/jlens"
LENS_PATHS = {"mm": JL / "mm/gemma4_12b_lens.pt",
              "C": JL / "mm_textafter/gemma4_12b_lens.pt",
              "B": JL / "wikitext1k/gemma4_12b_lens.pt"}
PLAN = {"image": {"layers": [18, 24, 36, 42], "lenses": ["mm"]},
        "text_after": {"layers": [30, 34, 38, 42], "lenses": ["C", "B"]},
        "answer": {"layers": [36, 38, 40, 42, 44], "lenses": ["C", "B"]}}


def log(m: str) -> None:
    print(f"[probe-scrape] {m}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, choices=list(pf.PROBES))
    ap.add_argument("--n-images", type=int, default=24, help="per split")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--scores-dir", default=str(ROOT / "scores"))
    ap.add_argument("--out-dir", default=str(ROOT / "scrape"))
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    args = ap.parse_args(argv)
    from jlens.hooks import ActivationRecorder
    from PIL import Image

    out = Path(args.out_dir) / args.probe
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, _ = load_scores(args.probe, Path(args.scores_dir))
    rng = np.random.default_rng(args.seed)
    draw = []
    for split, pool in (("refused", df[~df["answered"]]), ("answered", df[df["answered"]])):
        k = min(args.n_images, len(pool))
        if k:
            idx = rng.choice(len(pool), size=k, replace=False)
            draw += [(p, split) for p in pool["image_path"].to_numpy()[idx]]
    log(f"{args.probe}: {len(draw)} images "
        f"({Counter(s for _, s in draw)})")

    # The lens layers the plan needs, and only those (a 47-layer lens holds
    # every block; the 8-layer lenses hold 6, 12, ..., 46).
    lenses = {}
    for name, path in LENS_PATHS.items():
        lens = load_lens(str(path))
        have = set(int(l) for l in lens.source_layers)
        lenses[name] = (lens, have)
        log(f"lens {name}: {len(have)} layers")
    need = sorted({l for g in PLAN.values() for l in g["layers"]})
    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    ids3 = pf.token_ids(tok, args.probe)
    label_names = ("pos", "neg", "abstain")
    text = pf.render_chat(tok, tmpl, pf.user_text(args.probe))
    mask = None

    tok_rows, rank_rows = [], []
    for n, (q, split) in enumerate(draw):
        inputs = dict(proc(text=[text], images=[[Image.open(q).convert("RGB")]],
                           return_tensors="pt").to("cuda:0"))
        ids = inputs["input_ids"][0]
        block = pab.slots_of(inputs, image_token_id)
        answer_pos = int(ids.shape[0]) - 1
        groups = jt.position_groups(ids, block, answer_pos, image_token_id)
        with ActivationRecorder(lens_model.layers, at=need) as rec:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            acts = {l: rec.activations[l][0].detach().float() for l in need}
        for g, spec in PLAN.items():
            ps = groups.get(g, [])
            if not ps:
                continue
            idx = torch.tensor(ps, device="cuda:0")
            for l in spec["layers"]:
                for ln in spec["lenses"]:
                    lens, have = lenses[ln]
                    if l not in have:
                        continue
                    with torch.inference_mode():
                        logits = lens_model.unembed(lens.transport(acts[l][idx], l)).float()
                        if mask is None:
                            mask = build_token_mask(tok, logits.shape[1]).to(logits.device)
                        ranks = (logits.argsort(dim=1, descending=True)
                                 .argsort(dim=1)[:, list(ids3)])
                        topv, topi = logits.masked_fill(~mask, float("-inf")).topk(args.topk, dim=1)
                    topi, topv, ranks = topi.cpu().numpy(), topv.cpu().numpy(), ranks.cpu().numpy()
                    for j, p in enumerate(ps):
                        base = {"image_path": q, "split": split, "group": g,
                                "layer": l, "lens": ln, "pos": int(p),
                                "slot": j, "token_id_at": int(ids[p])}
                        for r in range(args.topk):
                            tok_rows.append({**base, "rank": r, "token": tok.decode([int(topi[j, r])]),
                                             "logit": float(topv[j, r])})
                        rank_rows.append({**base, **{f"rank_{nm}": int(ranks[j, i])
                                                     for i, nm in enumerate(label_names)}})
        if n % 8 == 0:
            log(f"  {n + 1}/{len(draw)} images, {time.time() - t0:.0f}s")
    toks = pd.DataFrame(tok_rows)
    rk = pd.DataFrame(rank_rows)
    toks.to_parquet(out / "tokens.parquet", index=False)
    rk.to_parquet(out / "label_ranks.parquet", index=False)

    # The summary: the scene words of each split at the patches (block 22),
    # the question words at block 38 under C and their overlap with B, and
    # the label ranks at the answer position by layer.
    summ = {"probe": args.probe, "images": len(draw), "seconds": int(time.time() - t0)}
    pat = toks[(toks.group == "image") & (toks.layer == 22) & (toks["rank"] < 3)]
    summ["patch_words_L22"] = {
        s: Counter(pat[pat.split == s]["token"].str.strip().str.lower()).most_common(25)
        for s in ("refused", "answered")}
    qa = toks[(toks.group == "text_after") & (toks.layer == 38) & (toks["rank"] < 5)]
    summ["question_words_L38"] = {}
    for s in ("refused", "answered"):
        c_set = set(qa[(qa.split == s) & (qa.lens == "C")]["token"])
        b_set = set(qa[(qa.split == s) & (qa.lens == "B")]["token"])
        summ["question_words_L38"][s] = {
            "top_C": Counter(qa[(qa.split == s) & (qa.lens == "C")]["token"].str.strip().str.lower()).most_common(25),
            "shared_with_B": sorted(c_set & b_set)[:40],
            "jaccard_C_B": (len(c_set & b_set) / len(c_set | b_set)) if (c_set | b_set) else None}
    ar = rk[(rk.group == "answer") & (rk.lens == "C")]
    summ["answer_label_rank_median"] = (
        ar.groupby(["split", "layer"])[[f"rank_{n}" for n in label_names]]
        .median().reset_index().to_dict("records"))
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    log(json.dumps(summ["answer_label_rank_median"], default=str))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
