#!/usr/bin/env python
"""F10.6: read the residual stream and the J++ lens at every token of the prompt.

The images, the prompt, the layers and the lens are those of F10.5 (`hop_depth.py`).
Only the positions change. A prompt has 3 parts:

| Part | Tokens | What the store holds |
|------|--------|----------------------|
| prefix | the template before the image | 1 read: these tokens cannot see the image |
| patches | the image tokens | residuals and the top words. The full reads are too large (1.3 TB) |
| tail | all tokens after the last patch | reads, residuals and the top words |

| File in `--out-dir` | Content |
|---------------------|---------|
| `manifest.json` | the axes, the tokens of each part, and the 6 names of F10.5 |
| `vocab.parquet` | `index`, `token_id`, `token` of the last axis of a read |
| `reads/reads-*.npy` | float32 [images, tail, layers, vocab]: lens logits, lower-case English word tokens |
| `residuals/residuals-*.npy` | float32 [images, tail, layers, d] |
| `topk/ids-*.npy`, `topk/logits-*.npy` | int32 and float32 [images, tail, layers, 30], of the whole vocabulary |
| `patches/residuals-*.npy` | float32 [images, patches, layers, d]: any read can be computed from these |
| `patches/ids-*.npy`, `patches/logits-*.npy` | int32 and float32 [images, patches, layers, 30] |
| `answers/answers-*.parquet` | `yes_no`, `yes_no_lc`, the pooled `margin`, `top`, and the token count of each image |
| `prefix_reads.npy`, `prefix_residuals.npy` | [prefix, layers, ...] |
| `text_reads.npy`, `text_residuals.npy` | the same text with no image; NaN at a token that the text does not have |

`hop_store.open_store(model, ALL_ROOT)` opens the tail. The 6 positions of F10.5 keep
their names, and the other tail tokens have the names `t00` and so on. Thus `after_place`
names the adjective of the prompt, also for another adjective.

Many workers can fill 1 store. A worker takes a missing shard with a claim file in
`claims/`, and the answers file of a shard is the last write. A preempted job takes its
own claim again after the requeue, and a claim older than `CLAIM_S` is free. The save of
a shard is atomic, thus a duplicate shard does no harm.

`--no-patches` skips the patch step. The image comes before the text and the attention
is causal, thus the patches of another adjective are the patches of the `cool` store.

`--noun` changes the noun of the question. The position `place` then names that noun.

Rules: vlm-narratives-docs/facct-runbook.md, F10.6.

Use:
    python scripts/f10/hop_all.py --model-key gemma-4-31b --model-dir DIR --lens FILE \
        --image-root DIR --out-dir data/f10/hop_all/gemma-4-31b
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
from hop_depth import IMAGES, N_LAYERS, POSITIONS, ROOT, STEP, TOP, ask, atomic_save, locate  # noqa: E402

__version__ = "1.2.0"

BATCH = 256
CLAIM_S = 30 * 60
# The yes forms come first, then the no forms, in the same order. Qwen3.6-27B writes the
# forms with a leading space, thus the no-space forms alone give a false margin.
ANSWERS = ("Yes", "yes", " Yes", " yes", "No", "no", " No", " no")


def pooled_margin(logits: torch.Tensor) -> torch.Tensor:
    """Give log p(any yes form) - log p(any no form). The first half of the last axis holds the yes forms."""
    half = logits.shape[-1] // 2
    return torch.logsumexp(logits[..., :half], -1) - torch.logsumexp(logits[..., half:], -1)


def claim(path: Path, me: str, stale: float = CLAIM_S) -> bool:
    """Claim a shard for the worker `me`. A worker can take its own claim again, or a stale claim."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if path.read_text() != me and time.time() - path.stat().st_mtime < stale:
            return False
        path.write_text(me)
        return True
    os.write(fd, me.encode())
    os.close(fd)
    return True


def parts(seq: list[int], image: int) -> tuple[list[int], list[int], list[int]]:
    """Give the indices of the prefix, the patches and the tail. The patches must be 1 block."""
    hits = [i for i, t in enumerate(seq) if t == image]
    if not hits or hits != list(range(hits[0], hits[-1] + 1)):
        raise RuntimeError("the patch tokens are not 1 block")
    return list(range(hits[0])), hits, list(range(hits[-1] + 1, len(seq)))


def tail_names(tail: list[int], named: dict[str, int]) -> list[str]:
    """Give a name to each tail token. The positions of F10.5 keep their names."""
    back = {i: p for p, i in named.items()}
    return [back.get(i, f"t{k:02d}") for k, i in enumerate(tail)]


def align_text(tail_ids: list[int], text_ids: list[int], first: int) -> list[int]:
    """Give the index in the tail of each token of the text from `first` on. The 2 must agree."""
    start, begin = tail_ids.index(first), text_ids.index(first)
    if tail_ids[start:] != text_ids[begin:]:
        raise RuntimeError("the text differs from the tail")
    return list(range(start, len(tail_ids)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True, choices=list(N_LAYERS))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adjective", default="cool", help="the word of the question")
    ap.add_argument("--noun", default="place", help="the noun of the question")
    ap.add_argument("--no-patches", action="store_true", help="do not read or save the patch tokens")
    args = ap.parse_args()

    out = Path(args.out_dir)
    for sub in ("reads", "residuals", "topk", "answers", "claims") + (() if args.no_patches else ("patches",)):
        (out / sub).mkdir(parents=True, exist_ok=True)
    images = pd.read_parquet(ROOT / IMAGES)
    if args.limit:
        images = images.head(args.limit)
    n_shards = (len(images) + args.chunk - 1) // args.chunk
    done = lambda s: (out / "answers" / f"answers-{s:05d}.parquet").exists()
    if all(done(s) for s in range(n_shards)):
        print(f"[f10.6] {out}: all {n_shards} shards are done", flush=True)
        return
    question = ask(args.adjective, args.noun)

    from jlens.hooks import ActivationRecorder
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from monocle import jlens_read
    from r3_readout import load_image

    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, device_map=device,
                                                        attn_implementation="sdpa").eval()
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens)
    tok = proc.tokenizer
    layers = list(range(0, N_LAYERS[args.model_key], STEP))
    image_token = int(model.config.image_token_id)
    one = lambda t: tok.encode(t, add_special_tokens=False)[0]
    for word in (args.noun, args.adjective):
        if len(tok.encode(f" {word}", add_special_tokens=False)) != 1:
            raise RuntimeError(f"` {word}` is not 1 token")
    first, place, cool, mark = one("Is"), one(f" {args.noun}"), one(f" {args.adjective}"), one("?")
    yes, no, low_yes, low_no = one("Yes"), one("No"), one("yes"), one("no")
    forms = torch.tensor([one(t) for t in ANSWERS], device=device)

    def capture(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Give the residuals [tokens, layers, d] and the logits of the last token."""
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            h = torch.stack([rec.activations[l][0].float() for l in layers], dim=1)
        return h, logits

    def lens_read(h: torch.Tensor, keep: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Give the word reads, and the top ids and logits, for [tokens, layers, d]."""
        words, ids, values = [], [], []
        with torch.inference_mode():
            for b, l in enumerate(layers):
                w, i, v = [], [], []
                for part in h[:, b].split(BATCH):
                    x = lens_model.unembed(lens.transport(part, l)).float()
                    if not torch.isfinite(x).all():
                        raise RuntimeError("non-finite read")
                    top = x.topk(TOP, dim=-1)
                    i.append(top.indices)
                    v.append(top.values)
                    if keep is not None:
                        w.append(x[..., keep])
                ids.append(torch.cat(i))
                values.append(torch.cat(v))
                if keep is not None:
                    words.append(torch.cat(w))
        return torch.stack(words, dim=1) if keep is not None else None, torch.stack(ids, dim=1), torch.stack(values, dim=1)

    def render(image: bool) -> str:
        text = {"type": "text", "text": question}
        msgs = [{"role": "user", "content": [{"type": "image"}, text] if image else [text]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    size = int(model.get_output_embeddings().weight.shape[0])
    vocab = pd.DataFrame([(i, tok.decode([i])) for i in range(min(size, len(tok)))], columns=["token_id", "token"])
    vocab = vocab[vocab["token"].map(lambda w: bool(re.fullmatch(r" [a-z]{3,}", w)))].reset_index(drop=True)
    vocab.insert(0, "index", vocab.index)
    vocab.to_parquet(out / f"vocab.parquet.{os.getpid()}.tmp", index=False)
    os.replace(out / f"vocab.parquet.{os.getpid()}.tmp", out / "vocab.parquet")
    keep = torch.tensor(vocab["token_id"].to_numpy(), device=device)

    print(f"[f10.6] {question} {len(images)} images, layers {layers}, {len(vocab)} words; {sum(map(done, range(n_shards)))} shards done", flush=True)

    t0, count, base = time.perf_counter(), 0, None
    me = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    for shard in range(n_shards):
        if done(shard) or not claim(out / "claims" / f"shard-{shard:05d}", me):
            continue
        part = images.iloc[shard * args.chunk:(shard + 1) * args.chunk]
        rows = {k: [] for k in ("reads", "residuals", "ids", "logits", "p_residuals", "p_ids", "p_logits")}
        answers = []
        for r in part.itertuples(index=False):
            inputs = dict(proc(text=[render(True)], images=[[load_image(args.image_root, r.sample_id)]], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0].tolist()
            prefix, patches, tail = parts(seq, image_token)
            h, logits = capture(inputs)
            if base is None:
                named = locate(seq, image_token, first, place, cool, mark)
                base = {"tail": [seq[i] for i in tail], "prefix": h[prefix].clone(), "patches": len(patches)}
                text_inputs = dict(proc(text=[render(False)], return_tensors="pt").to(device))
                text_seq = text_inputs["input_ids"][0].tolist()
                at = align_text(base["tail"], text_seq, first)
                th, _ = capture(text_inputs)
                th = th[text_seq.index(first):]
                tw, _, _ = lens_read(th, keep)
                blank = lambda a: np.concatenate([np.full((len(tail) - len(at), *a.shape[1:]), np.nan, np.float32), a])
                atomic_save(out / "text_residuals.npy", blank(th.cpu().numpy()))
                atomic_save(out / "text_reads.npy", blank(tw.cpu().numpy()))
                pw, _, _ = lens_read(base["prefix"], keep)
                atomic_save(out / "prefix_residuals.npy", base["prefix"].cpu().numpy())
                atomic_save(out / "prefix_reads.npy", pw.cpu().numpy())
                names = tail_names(tail, named)
                head = (json.dumps(
                    {"model": args.model_key, "version": __version__, "question": question, "layout": "image first", "positions": names, "layers": layers,
                     "n_images": len(images), "chunk": args.chunk, "d": int(h.shape[-1]), "vocab": len(vocab), "lens": args.lens, "top": TOP,
                     "tail_tokens": [tok.decode([i]) for i in base["tail"]], "tail_ids": base["tail"],
                     "prefix_tokens": [tok.decode([seq[i]]) for i in prefix], "prefix_ids": [seq[i] for i in prefix],
                     "patches": len(patches), "patches_saved": not args.no_patches, "f10_5_positions": list(POSITIONS)}, indent=1) + "\n")
                (out / f"manifest.json.{os.getpid()}.tmp").write_text(head)
                os.replace(out / f"manifest.json.{os.getpid()}.tmp", out / "manifest.json")
                print(f"[f10.6] check prefix {len(prefix)}, patches {len(patches)}, tail {len(tail)}: "
                      f"{[(n, tok.decode([i])) for n, i in zip(names, base['tail'])]}", flush=True)
            elif [seq[i] for i in tail] != base["tail"] or len(patches) != base["patches"]:
                raise RuntimeError(f"image {r.image_idx}: the tokens differ")
            elif float((h[prefix] - base["prefix"]).abs().max()) != 0.0 and count < 3:
                print(f"[f10.6] prefix gap {float((h[prefix] - base['prefix']).abs().max()):.2e}", flush=True)
            words, ids, values = lens_read(h[tail], keep)
            for key, x in (("reads", words), ("residuals", h[tail]), ("ids", ids), ("logits", values)):
                rows[key].append(x.cpu().numpy())
            if not args.no_patches:
                _, pid, pval = lens_read(h[patches], None)
                for key, x in (("p_residuals", h[patches]), ("p_ids", pid), ("p_logits", pval)):
                    rows[key].append(x.cpu().numpy())
            lp = torch.log_softmax(logits, dim=-1)
            answers.append({"image_idx": r.image_idx, "yes_no": float(lp[yes] - lp[no]), "yes_no_lc": float(lp[low_yes] - lp[low_no]),
                            "margin": float(pooled_margin(lp[forms])), "top": tok.decode([int(logits.argmax())]), "tokens": len(seq)})
            count += 1
        for key, path in (("reads", f"reads/reads-{shard:05d}.npy"), ("residuals", f"residuals/residuals-{shard:05d}.npy"),
                          ("ids", f"topk/ids-{shard:05d}.npy"), ("logits", f"topk/logits-{shard:05d}.npy"),
                          ("p_residuals", f"patches/residuals-{shard:05d}.npy"), ("p_ids", f"patches/ids-{shard:05d}.npy"),
                          ("p_logits", f"patches/logits-{shard:05d}.npy")):
            if rows[key]:
                atomic_save(out / path, np.stack(rows[key]))
        pd.DataFrame(answers).to_parquet(out / "answers" / f"answers-{shard:05d}.parquet", index=False)  # the last write marks the shard done
        print(f"[f10.6] shard {shard + 1}/{n_shards}, {count / (time.perf_counter() - t0):.2f} images/s", flush=True)
    print(f"[f10.6] worker finished, {sum(map(done, range(n_shards)))} of {n_shards} shards done", flush=True)


if __name__ == "__main__":
    main()
