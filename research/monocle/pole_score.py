"""Score a drawn image set with the pole projections, batched.

Reads `outputs/_monocle/pole_score/<case>/sample.parquet` (from
`research/scripts/pole_score_sample.py`) and the pole vectors of
`outputs/_monocle/pole_sample/<case>/vectors.npz`. One forward per batch of
images under the case's own single-image question: every input has the
same token count, so a batch needs no padding (the module checks it).
Per image and block it writes the RAW projections of the answer residual
on the 'pos', 'neg' and 'contrast' vectors (center them over the scored
set on the CPU side), the logit gap `c`, and the top token.

    sbatch research/monocle/monocle.sub monocle.pole_score --case restaurants --shard 0 --n-shards 8

`--images` and `--out-dir` point it at another image table, for the case
reads on the probe images (E5 of the lens-refusal probe).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import single_steer as ss  # noqa: E402
from research.monocle.jlens_steer import load_model_and_lens  # noqa: E402

ROOT = REPO / "outputs/_monocle/pole_score"
VEC_ROOT = REPO / "outputs/_monocle/pole_sample"


def log(m: str) -> None:
    print(f"[pole-score] {m}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--layers", default="41,42,44,45")
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images")
    ap.add_argument("--images", default=None,
                    help="an image table with an image_path column; default "
                         "<root>/<case>/sample.parquet")
    ap.add_argument("--out-dir", default=None,
                    help="where the shard goes; default <root>/<case>")
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    from jlens.hooks import ActivationRecorder
    from PIL import Image

    images = Path(args.images) if args.images else ROOT / args.case / "sample.parquet"
    sample = pd.read_parquet(images)
    paths = sample["image_path"].tolist()[args.shard::args.n_shards]
    if args.limit:
        paths = paths[:args.limit]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"scores_shard{args.shard:02d}of{args.n_shards:02d}.parquet"
    vecs = dict(np.load(VEC_ROOT / args.case / "vectors.npz"))
    for l in layers:
        for k in ("pos", "neg", "contrast"):
            if f"v_{k}_L{l}" not in vecs:
                raise KeyError(f"vectors.npz lacks v_{k}_L{l}")

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    text = ss.render_single_chat(proc.tokenizer, tmpl, ss.user_text(args.case))
    final = lens_model.n_layers - 1
    at = sorted(set(layers) | {final})
    V = {l: {k: torch.tensor(vecs[f"v_{k}_L{l}"], dtype=torch.float32)
             for k in ("pos", "neg", "contrast")} for l in layers}
    log(f"{args.case}: shard {args.shard}/{args.n_shards}, {len(paths)} images, "
        f"batch {args.batch}, layers {layers}")

    rows, t0 = [], time.time()
    for b0 in range(0, len(paths), args.batch):
        chunk = paths[b0:b0 + args.batch]
        imgs = [[Image.open(p).convert("RGB")] for p in chunk]
        inputs = dict(proc(text=[text] * len(chunk), images=imgs,
                           return_tensors="pt").to("cuda:0"))
        am = inputs.get("attention_mask")
        if am is not None and not bool(am.all()):
            raise RuntimeError("padding inside a batch; the inputs differ in length")
        inputs["use_cache"] = False
        with ActivationRecorder(lens_model.layers, at=at) as rec, torch.no_grad():
            model(**inputs)
            acts = {i: rec.activations[i][:, -1, :].float().cpu() for i in at}
            logits = lens_model.unembed(acts[final].to("cuda:0")).float().cpu()
        c = (logits[:, tok_pos] - logits[:, tok_neg]).numpy()
        top = logits.argmax(dim=1).numpy()
        for l in layers:
            h = acts[l]
            pr = {k: (h @ V[l][k]).numpy() for k in V[l]}
            for i, p in enumerate(chunk):
                rows.append({"image_path": p, "layer": l, "c": float(c[i]),
                             "top_token": int(top[i]),
                             "raw_pos": float(pr["pos"][i]),
                             "raw_neg": float(pr["neg"][i]),
                             "raw_contrast": float(pr["contrast"][i])})
        if (b0 // args.batch) % 20 == 0:
            done = b0 + len(chunk)
            log(f"  {done}/{len(paths)} images, {(time.time() - t0) / done:.2f} s/image")
    pd.DataFrame(rows).to_parquet(out, index=False)
    log(f"wrote {out} ({len(rows)} rows, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
