"""The label-rank trajectory at the answer position over the whole census.

The scrape (E5c) read the ranks of the 3 label tokens at the answer
position on 24 refused and 24 answered images. This module reads them on
EVERY image of the census, at the blocks where the trajectory turns
(36 to 45), under lens C, in batches of 8. Per image and block it
records the rank of the pos, neg, and abstain tokens over the full
vocabulary, the transported contrast score `s = logit(pos) - logit(neg)`,
and whether the stigmatized pole leads the plain one.

    sbatch research/monocle/monocle.sub monocle.probe_ranks --probe class --shard 0 --n-shards 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import probe_frame as pf  # noqa: E402
from research.monocle.jlens_read import load_lens  # noqa: E402
from research.monocle.jlens_steer import load_model_and_lens  # noqa: E402
from research.monocle.probe_vectors import LENSES  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"
DEFAULT_LAYERS = "24,30,36,38,40,41,42,43,44,45"


def log(m: str) -> None:
    print(f"[probe-ranks] {m}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, choices=list(pf.PROBES))
    ap.add_argument("--images", default=str(ROOT / "sample/images.parquet"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "ranks"))
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--lens", choices=["C", "none"], default="C",
                    help="C transports each block through the fitted lens; "
                         "none applies the final norm and the LM head to the "
                         "block residual as it is (the plain logit lens)")
    ap.add_argument("--frame", choices=["chat", "raw"], default="chat",
                    help="chat = the model's chat template; raw = the completion "
                         "frame of a pretrained checkpoint")
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
    out = out_dir / f"ranks_shard{args.shard:02d}of{args.n_shards:02d}.parquet"

    lens = load_lens(str(LENSES["C"])) if args.lens == "C" else None
    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    ids3 = list(pf.token_ids(tok, args.probe))
    if args.frame == "chat":
        if not tmpl:
            raise RuntimeError(f"{args.model_dir} has no chat template; use --frame raw")
        text = pf.render_chat(tok, tmpl, pf.user_text(args.probe))
    else:
        text = pf.render_raw(pf.user_text(args.probe))
    log(f"lens {args.lens}, frame {args.frame}, model {args.model_dir}")
    classes = {ids3[0]: "pos", ids3[1]: "neg", ids3[2]: "abstain"}
    final = lens_model.n_layers - 1
    at = sorted(set(layers) | {final})
    log(f"{args.probe}: shard {args.shard}/{args.n_shards}, {len(paths)} images, blocks {layers}")

    rows, t0 = [], time.time()
    for b0 in range(0, len(paths), args.batch):
        chunk = paths[b0:b0 + args.batch]
        imgs = [[Image.open(q).convert("RGB")] for q in chunk]
        inputs = dict(proc(text=[text] * len(chunk), images=imgs,
                           return_tensors="pt").to("cuda:0"))
        am = inputs.get("attention_mask")
        if am is not None and not bool(am.all()):
            raise RuntimeError("padding inside a batch")
        inputs["use_cache"] = False
        with ActivationRecorder(lens_model.layers, at=at) as rec, torch.no_grad():
            model(**inputs)
            for l in at:
                h = rec.activations[l][:, -1, :].float()
                if l == final or lens is None:
                    logits = lens_model.unembed(h).float()
                else:
                    logits = lens_model.unembed(lens.transport(h, l)).float()
                order = logits.argsort(dim=1, descending=True)
                ranks = order.argsort(dim=1)[:, ids3].cpu()
                top5 = order[:, :5].cpu()
                s = (logits[:, ids3[0]] - logits[:, ids3[1]]).cpu()
                for i, q in enumerate(chunk):
                    t5 = [int(x) for x in top5[i]]
                    rows.append({"image_path": q, "probe": args.probe, "layer": l,
                                 "rank_pos": int(ranks[i, 0]), "rank_neg": int(ranks[i, 1]),
                                 "rank_abstain": int(ranks[i, 2]), "s": float(s[i]),
                                 "top1_id": t5[0], "top1": tok.decode([t5[0]]),
                                 "top5": " | ".join(tok.decode([x]) for x in t5),
                                 "top_class": classes.get(t5[0], "other")})
        if (b0 // args.batch) % 25 == 0:
            done = b0 + len(chunk)
            log(f"  {done}/{len(paths)} images, {(time.time() - t0) / done:.2f} s/image")
    df = pd.DataFrame(rows)
    df["neg_leads"] = df["rank_neg"] < df["rank_pos"]
    df["abstain_leads"] = (df["rank_abstain"] < df["rank_pos"]) & (df["rank_abstain"] < df["rank_neg"])
    df.to_parquet(out, index=False)
    log(f"wrote {out} ({len(df)} rows, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
