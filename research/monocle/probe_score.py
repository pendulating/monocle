"""E0 and E1 of the lens-refusal probe: the batched prefill and the census.

For 1 probe and 1 image table, 1 forward for each batch of images under
the probe frame (`research/monocle/probe_frame.py`), teacher-forced to `Answer:`.
Every input has the same token count, so a batch needs no padding, and
the module checks it. Per image it records:

| Field | Meaning |
|-------|---------|
| `c` | logit(pos) minus logit(neg), the space forms |
| `logit_abstain`, `p_*` | the abstain logit, and the softmax mass of the 3 labels |
| `top1..top5` | the top tokens of the next-token distribution, id and text |
| `top_class` | pos, neg, abstain, or other (a refusal in free text) |
| `answered` | top_class is pos or neg |

The answer residual at the blocks of `--layers` goes to an `.npz` beside
the parquet (float16, 1 row per image, the same order), so E2 and E3 read
the lens without a second forward.

    sbatch research/monocle/monocle.sub monocle.probe_score --probe class --images <parquet> --limit 16   # E0
    sbatch research/monocle/monocle.sub monocle.probe_score --probe class --shard 0 --n-shards 2          # E1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import probe_frame as pf  # noqa: E402
from research.monocle.jlens_steer import load_model_and_lens  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"
IMAGES_DEFAULT = ROOT / "sample/images.parquet"
DEFAULT_LAYERS = "41,42,43,44,45"


def log(m: str) -> None:
    print(f"[probe-score] {m}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, choices=list(pf.PROBES))
    ap.add_argument("--images", default=str(IMAGES_DEFAULT),
                    help="a parquet with an image_path column")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="E0: the first N images")
    ap.add_argument("--out-dir", default=str(ROOT / "scores"))
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    from jlens.hooks import ActivationRecorder
    from PIL import Image

    paths = pd.read_parquet(args.images, columns=["image_path"])["image_path"].tolist()
    paths = paths[args.shard::args.n_shards]
    if args.limit:
        paths = paths[:args.limit]
    out_dir = Path(args.out_dir) / args.probe
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scores_shard{args.shard:02d}of{args.n_shards:02d}"

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    id_pos, id_neg, id_abs = pf.token_ids(tok, args.probe)
    sets = pf.label_sets(tok, args.probe)
    text = pf.render_chat(tok, tmpl, pf.user_text(args.probe))
    final = lens_model.n_layers - 1
    at = sorted(set(layers) | {final})
    p = pf.PROBES[args.probe]
    log(f"{args.probe}: poles {p.pos}/{p.neg}/{pf.ABSTAIN} ids "
        f"{(id_pos, id_neg, id_abs)}, shard {args.shard}/{args.n_shards}, "
        f"{len(paths)} images, batch {args.batch}, layers {layers}")
    log("prompt tail: " + repr(text[-160:]))

    rows, H = [], {l: [] for l in layers}
    t0 = time.time()
    for b0 in range(0, len(paths), args.batch):
        chunk = paths[b0:b0 + args.batch]
        imgs = [[Image.open(q).convert("RGB")] for q in chunk]
        inputs = dict(proc(text=[text] * len(chunk), images=imgs,
                           return_tensors="pt").to("cuda:0"))
        am = inputs.get("attention_mask")
        if am is not None and not bool(am.all()):
            raise RuntimeError("padding inside a batch; the inputs differ in length")
        inputs["use_cache"] = False
        with ActivationRecorder(lens_model.layers, at=at) as rec, torch.no_grad():
            model(**inputs)
            acts = {i: rec.activations[i][:, -1, :].float() for i in at}
            logits = lens_model.unembed(acts[final]).float()
        probs = torch.softmax(logits, dim=1)
        topv, topi = logits.topk(args.topk, dim=1)
        logits, probs = logits.cpu(), probs.cpu()
        topi = topi.cpu().numpy()
        for l in layers:
            H[l].append(acts[l].cpu().to(torch.float16).numpy())
        for i, q in enumerate(chunk):
            r = {"image_path": q, "probe": args.probe,
                 "c": float(logits[i, id_pos] - logits[i, id_neg]),
                 "logit_pos": float(logits[i, id_pos]),
                 "logit_neg": float(logits[i, id_neg]),
                 "logit_abstain": float(logits[i, id_abs]),
                 "p_pos": float(probs[i, id_pos]),
                 "p_neg": float(probs[i, id_neg]),
                 "p_abstain": float(probs[i, id_abs])}
            for k in range(args.topk):
                tid = int(topi[i, k])
                r[f"top{k + 1}_id"] = tid
                r[f"top{k + 1}"] = tok.decode([tid])
            r["top_class"] = pf.classify_top(int(topi[i, 0]), sets)
            r["answered"] = r["top_class"] in ("pos", "neg")
            rows.append(r)
        if (b0 // args.batch) % 20 == 0:
            done = b0 + len(chunk)
            log(f"  {done}/{len(paths)} images, {(time.time() - t0) / done:.2f} s/image")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / f"{stem}.parquet", index=False)
    np.savez(out_dir / f"{stem}_resid.npz",
             image_path=np.array(paths), layers=np.array(layers),
             **{f"h_L{l}": np.concatenate(H[l]) for l in layers})
    share = df["top_class"].value_counts(normalize=True).round(3).to_dict()
    forms = df.loc[df["top_class"] == "other", "top1"].value_counts().head(8).to_dict()
    summ = {"probe": args.probe, "images": int(len(df)), "top_class_share": share,
            "refusal_forms": forms, "c_mean": float(df["c"].mean()),
            "c_sd": float(df["c"].std()), "seconds": int(time.time() - t0)}
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ))
    log(f"wrote {out_dir / stem}.parquet and _resid.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
