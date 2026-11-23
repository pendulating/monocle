#!/usr/bin/env python
"""F10.30: a lean capture of the F10 read at 6 positions, for many images and the 14 prompts.

The capture is the one of `hop_all.py` (F10.6): float32 HF model, the image first, the question after it,
the J++ lens at the saved layers. It keeps less: the residuals at the 6 named positions, the lens reads at
the noun, the adjective and the last token (float16, over the same word list as the F10 stores), and the
8-form margin. It does not keep the other tail tokens, the top-k lists, the patches or the text-only pass.
1 job loads the model once and runs the 14 prompts in turn. Shards of 50 images with claims, as F10.6, thus
many workers can fill 1 store, and a requeue resumes.

Layout of `data/f10/hop_more/lean/{store}/` (the store name is the F10.26 one):
  residuals/residuals-*.npy   [images, 6, layers, d] float16, the positions of `hop_depth.POSITIONS`
  reads/reads-*.npy           [images, 3, layers, vocab] float16, the positions `place`, `after_place`, `final`
  answers/answers-*.parquet   image_idx, margin (8 forms), yes_no, top, tokens; the last write of a shard
  vocab.parquet, manifest.json

Use:
    python research/scripts/f10/hop_lean.py --model-key gemma-4-31b --model-dir DIR --lens LENS --image-root DIR \
        --sample data/f10/hop_more/images_k10.parquet --out-root data/f10/hop_more/lean
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.scripts.f10.hop_all import ANSWERS, BATCH, claim, parts, pooled_margin  # noqa: E402
from research.scripts.f10.hop_depth import N_LAYERS, POSITIONS, ROOT, STEP, ask, atomic_save, locate  # noqa: E402
from research.scripts.f10.hop_prompts import ADJECTIVES, NOUNS, store_name  # noqa: E402

__version__ = "1.0.0"
READ_POSITIONS = ("place", "after_place", "final")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True, choices=list(N_LAYERS))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--sample", default="data/f10/hop_more/images_k10.parquet")
    ap.add_argument("--out-root", default="data/f10/hop_more/lean")
    ap.add_argument("--prompts", default=",".join(f"{a}:{n}" for n in NOUNS for a in ADJECTIVES), help="adjective:noun pairs")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from jlens.hooks import ActivationRecorder
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from research.monocle import jlens_read
    from research.scripts.f8.r3_readout import load_image

    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, device_map=device, attn_implementation="sdpa").eval()
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens)
    tok = proc.tokenizer
    layers = list(range(0, N_LAYERS[args.model_key], STEP))
    image_token = int(model.config.image_token_id)
    one = lambda t: tok.encode(t, add_special_tokens=False)[0]
    yes, no = one("Yes"), one("No")
    forms = torch.tensor([one(t) for t in ANSWERS], device=device)
    size = int(model.get_output_embeddings().weight.shape[0])
    vocab = pd.DataFrame([(i, tok.decode([i])) for i in range(min(size, len(tok)))], columns=["token_id", "token"])
    vocab = vocab[vocab["token"].map(lambda w: bool(re.fullmatch(r" [a-z]{3,}", w)))].reset_index(drop=True)
    vocab.insert(0, "index", vocab.index)
    keep = torch.tensor(vocab["token_id"].to_numpy(), device=device)
    images = pd.read_parquet(ROOT / args.sample)
    if args.limit:
        images = images.head(args.limit)
    n_shards = (len(images) + args.chunk - 1) // args.chunk
    me = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    read_idx = [POSITIONS.index(p) for p in READ_POSITIONS]

    def capture(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            h = torch.stack([rec.activations[l][0].float() for l in layers], dim=1)
        return h, logits

    def lens_read(h: torch.Tensor) -> torch.Tensor:
        """Give the word reads [tokens, layers, vocab] of [tokens, layers, d]."""
        words = []
        with torch.inference_mode():
            for b, l in enumerate(layers):
                x = lens_model.unembed(lens.transport(h[:, b], l)).float()
                if not torch.isfinite(x).all():
                    raise RuntimeError("non-finite read")
                words.append(x[..., keep])
        return torch.stack(words, dim=1)

    def render(question: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    for pair in args.prompts.split(","):
        adj, noun = pair.split(":")
        for word in (noun, adj):
            if len(tok.encode(f" {word}", add_special_tokens=False)) != 1:
                raise RuntimeError(f"` {word}` is not 1 token")
        question = ask(adj, noun)
        first, place, cool, mark = one("Is"), one(f" {noun}"), one(f" {adj}"), one("?")
        out = ROOT / args.out_root / store_name(args.model_key, adj, noun)
        for sub in ("reads", "residuals", "answers", "claims"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        done = lambda s: (out / "answers" / f"answers-{s:05d}.parquet").exists()
        if all(done(s) for s in range(n_shards)):
            print(f"[f10.30] {out.name}: all {n_shards} shards are done", flush=True)
            continue
        if not (out / "vocab.parquet").exists():
            vocab.to_parquet(out / f"vocab.parquet.{os.getpid()}.tmp", index=False)
            os.replace(out / f"vocab.parquet.{os.getpid()}.tmp", out / "vocab.parquet")
        print(f"[f10.30] {question} {len(images)} images, layers {layers}, {len(vocab)} words; {sum(map(done, range(n_shards)))} shards done", flush=True)
        t0, count, base = time.perf_counter(), 0, None
        for shard in range(n_shards):
            if done(shard) or not claim(out / "claims" / f"shard-{shard:05d}", me):
                continue
            part = images.iloc[shard * args.chunk:(shard + 1) * args.chunk]
            res, reads, answers = [], [], []
            for r in part.itertuples(index=False):
                inputs = dict(proc(text=[render(question)], images=[[load_image(args.image_root, r.sample_id)]], return_tensors="pt").to(device))
                seq = inputs["input_ids"][0].tolist()
                prefix, patches, tail = parts(seq, image_token)
                h, logits = capture(inputs)
                named = locate(seq, image_token, first, place, cool, mark)
                at = [named[p] for p in POSITIONS]
                if base is None:
                    base = {"tail": [seq[i] for i in tail], "patches": len(patches), "at": at}
                    manifest = {"model": args.model_key, "version": __version__, "question": question, "layout": "image first", "positions": list(POSITIONS),
                                "read_positions": list(READ_POSITIONS), "layers": layers, "sample": args.sample, "n_images": len(images), "chunk": args.chunk,
                                "d": int(h.shape[-1]), "vocab": len(vocab), "lens": args.lens, "dtype": "float16", "tail_tokens": [tok.decode([i]) for i in base["tail"]],
                                "tail_ids": base["tail"], "patches": len(patches), "answer_forms": list(ANSWERS)}
                    (out / f"manifest.json.{os.getpid()}.tmp").write_text(json.dumps(manifest, indent=1) + "\n")
                    os.replace(out / f"manifest.json.{os.getpid()}.tmp", out / "manifest.json")
                elif [seq[i] for i in tail] != base["tail"] or len(patches) != base["patches"] or at != base["at"]:
                    raise RuntimeError(f"image {r.image_idx}: the tokens differ")
                hp = h[at]
                res.append(hp.half().cpu().numpy())
                reads.append(lens_read(hp[read_idx]).half().cpu().numpy())
                lp = torch.log_softmax(logits, dim=-1)
                answers.append({"image_idx": int(r.image_idx), "yes_no": float(lp[yes] - lp[no]), "margin": float(pooled_margin(lp[forms])),
                                **{f"lp_{f}": float(lp[i]) for f, i in zip(ANSWERS, forms.tolist())}, "top": tok.decode([int(logits.argmax())]), "tokens": len(seq)})
                count += 1
            atomic_save(out / "residuals" / f"residuals-{shard:05d}.npy", np.stack(res))
            atomic_save(out / "reads" / f"reads-{shard:05d}.npy", np.stack(reads))
            pd.DataFrame(answers).to_parquet(out / "answers" / f"answers-{shard:05d}.parquet", index=False)  # the last write marks the shard done
            print(f"[f10.30] {out.name} shard {shard + 1}/{n_shards}, {count / (time.perf_counter() - t0):.2f} images/s", flush=True)
        print(f"[f10.30] {out.name}: worker finished, {sum(map(done, range(n_shards)))} of {n_shards} shards done", flush=True)


if __name__ == "__main__":
    main()
