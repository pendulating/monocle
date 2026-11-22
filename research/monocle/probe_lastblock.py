"""What the last block does to a word: the decomposition of `log p(word)`.

Round 3 found that the instruct model's block 47 removes the read of
`rich`, `smart`, `immigrant`, `prostitute` and `trustworthy` in one step
and keeps `ghetto` and `poor`. At the answer position with the plain
logit lens,

    log p(w) = logit(w) - lse,   logit(w) = rd(w) / rms(h),
    rd(w) = (W_U[w] * (1 + gamma)) . h

where `h` is the block residual, `rms(h)` the RMSNorm scale, `gamma`
the final-norm weight and `lse` the log-sum-exp over the vocabulary.
Across images, only 3 things can move the AUC of a word between block
46 and 47: the raw projection `rd`, the per-image scale `rms`, or the
per-image normalizer `lse`. This census records all 3 at blocks 46 and
47 and writes them as 3 pseudo-arms (`lp`, `lg`, `rd`) in the format of
`research/monocle/probe_words.py`, so `research/scripts/probe_word_test.py` reads each
with the same 3 bars. A scalar table holds `rms`, `lse`, `|h|`, the
top-1 logit and the block-46-to-47 geometry per image.

    sbatch research/monocle/monocle.sub monocle.probe_lastblock --template chat \\
        --shard 0 --n-shards 4 --out-dir outputs/_monocle/probe_lastblock/it
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
from research.monocle.jlens_steer import load_model_and_lens  # noqa: E402
from research.monocle.probe_words import POOL, WORDS, log, user_text, word_ids  # noqa: E402

BLOCKS = (46, 47)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", choices=["chat", "raw"], default="chat")
    ap.add_argument("--images",
                    default=str(REPO / "outputs/_monocle/probe_nta/sample/images.parquet"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    args = ap.parse_args(argv)
    from jlens.hooks import ActivationRecorder
    from PIL import Image

    paths = pd.read_parquet(args.images, columns=["image_path"])["image_path"].tolist()
    paths = paths[args.shard::args.n_shards]
    if args.limit:
        paths = paths[:args.limit]
    out_dir = Path(args.out_dir)
    for sub in ("lp", "lg", "rd"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    stem = f"words_shard{args.shard:02d}of{args.n_shards:02d}.parquet"

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    probe_ids, bad = word_ids(tok, WORDS)
    pool_ids, bad_pool = word_ids(tok, POOL)
    pool_ids = {w: i for w, i in pool_ids.items() if w not in probe_ids}
    if bad or bad_pool:
        log(f"dropped, not 1 token in the space form: probe {bad}, pool {bad_pool}")
    abstain_id = tok.encode(" " + pf.ABSTAIN, add_special_tokens=False)[0]
    words = {**probe_ids, **pool_ids, pf.ABSTAIN: abstain_id}
    names = list(words)
    ids = torch.tensor([words[w] for w in names], device="cuda:0")
    text = user_text("open")
    if args.template == "chat":
        if not tmpl:
            raise RuntimeError(f"{args.model_dir} has no chat template; use --template raw")
        rendered = pf.render_chat(tok, tmpl, text)
    else:
        rendered = pf.render_raw(text)
    norm = lens_model._final_norm
    head = lens_model._lm_head
    W = head.weight  # [vocab, d]
    final = lens_model.n_layers - 1
    assert final == BLOCKS[-1], f"final block is {final}, not {BLOCKS[-1]}"
    log(f"template {args.template}, model {args.model_dir}; shard {args.shard}/{args.n_shards}, "
        f"{len(paths)} images, blocks {BLOCKS}; {len(probe_ids)} probe + {len(pool_ids)} pool words")

    rows = {s: [] for s in ("lp", "lg", "rd")}
    scal, t0, checked = [], time.time(), False
    for b0 in range(0, len(paths), args.batch):
        chunk = paths[b0:b0 + args.batch]
        imgs = [[Image.open(q).convert("RGB")] for q in chunk]
        inputs = dict(proc(text=[rendered] * len(chunk), images=imgs,
                           return_tensors="pt").to("cuda:0"))
        am = inputs.get("attention_mask")
        if am is not None and not bool(am.all()):
            raise RuntimeError("padding inside a batch")
        inputs["use_cache"] = False
        with ActivationRecorder(lens_model.layers, at=list(BLOCKS)) as rec, torch.no_grad():
            model(**inputs)
            H = {l: rec.activations[l][:, -1, :].float() for l in BLOCKS}
            for l in BLOCKS:
                h = H[l]
                hn = norm(h.to(W.dtype)).float()                  # h * (1+gamma) / rms
                rms = torch.sqrt((h ** 2).mean(dim=1))            # [B]
                scaled = hn * rms[:, None]                        # h * (1+gamma)
                logits = (hn.to(W.dtype) @ W.T).float()           # [B, vocab]
                rd = (scaled.to(W.dtype) @ W.T).float()           # logit * rms
                if not checked:
                    err = (rd / rms[:, None] - logits).abs().max().item()
                    log(f"identity check rd/rms == logit: max abs err {err:.4f}")
                    checked = True
                lp = torch.log_softmax(logits, dim=1)
                lse = torch.logsumexp(logits, dim=1)
                stats = {"lp": lp, "lg": logits, "rd": rd}
                order = logits.argsort(dim=1, descending=True)
                top5 = order[:, :5].cpu()
                top1_logit = logits.gather(1, order[:, :1]).squeeze(1)
                for s, M in stats.items():
                    ranks = M.argsort(dim=1, descending=True).argsort(dim=1)[:, ids].cpu()
                    vals = M[:, ids].cpu()
                    for i, q in enumerate(chunk):
                        t5 = [int(x) for x in top5[i]]
                        row = {"image_path": q, "layer": l, "top1": tok.decode([t5[0]]),
                               "top5": " | ".join(tok.decode([x]) for x in t5)}
                        for j, w in enumerate(names):
                            row[f"lp_{w}"] = float(vals[i, j])
                            row[f"rk_{w}"] = int(ranks[i, j])
                        rows[s].append(row)
                hnorm = h.norm(dim=1)
                for i, q in enumerate(chunk):
                    scal.append({"image_path": q, "layer": l, "rms": float(rms[i]),
                                 "hnorm": float(hnorm[i]), "lse": float(lse[i]),
                                 "top1_logit": float(top1_logit[i]),
                                 "top1": tok.decode([int(top5[i][0])])})
            d = H[47] - H[46]
            cos = torch.nn.functional.cosine_similarity(H[47], H[46], dim=1)
            for i, q in enumerate(chunk):
                scal.append({"image_path": q, "layer": -1, "dnorm": float(d[i].norm()),
                             "cos47_46": float(cos[i])})
        if (b0 // args.batch) % 25 == 0:
            done = b0 + len(chunk)
            log(f"  {done}/{len(paths)} images, {(time.time() - t0) / done:.2f} s/image")
    for s in rows:
        pd.DataFrame(rows[s]).to_parquet(out_dir / s / stem, index=False)
    pd.DataFrame(scal).to_parquet(out_dir / f"scalars_shard{args.shard:02d}of{args.n_shards:02d}.parquet",
                                  index=False)
    log(f"wrote {out_dir}/{{lp,lg,rd}}/{stem} + scalars ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
