#!/usr/bin/env python
"""F10.5: read the residual stream and the J++ lens at 6 text positions and each 4th layer.

The image comes first, then `Is this place cool? Answer with one word: yes or no.` The
images are `data/f10/hop_depth/images.parquet`. Row i of the arrays is row i of that file.

| Position | Token |
|----------|-------|
| `after_image` | the first token after the last patch: a delimiter of the template |
| `first_text` | `Is` |
| `place` | ` place` |
| `after_place` | ` cool` |
| `after_cool` | `?` |
| `final` | the last token of the generation prompt |

| File in `--out-dir` | Content |
|---------------------|---------|
| `manifest.json` | the axes, the prompt, the sources and the `spider` control |
| `vocab.parquet` | `index`, `token_id`, `token` of the 3rd axis of a read |
| `reads/reads-*.npy` | float32 [images, positions, layers, vocab]: lens logits, lower-case English word tokens |
| `residuals/residuals-*.npy` | float32 [images, positions, layers, d]: any token can be computed from these |
| `topk/topk-*.parquet` | the top 30 tokens of the whole vocabulary, 1 row for each image, position, layer and rank |
| `answers/answers-*.parquet` | `yes_no`, `yes_no_lc`, `top`, and the token count of each image |
| `text_reads.npy`, `text_residuals.npy` | the same text with no image; `after_image` is NaN |

`hop_store.py` opens these files. A restart resumes at the first missing shard. Rules:
vlm-narratives-docs/facct-runbook.md, F10.5.

Use:
    python scripts/f10/hop_depth.py --model-key gemma-4-31b --model-dir DIR --lens FILE \
        --image-root DIR --out-dir data/f10/hop_depth/gemma-4-31b
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sub/jacobian-lens"), str(ROOT / "scripts/f8")]

__version__ = "1.0.0"

N_LAYERS = {"gemma-4-31b": 59, "qwen3.5-9b": 31, "qwen3.6-27b": 63}
STEP = 4
POSITIONS = ("after_image", "first_text", "place", "after_place", "after_cool", "final")
QUESTION = "Is this place cool? Answer with one word: yes or no."
SPIDER = "Fact: The number of legs on the animal that spins webs is "
TOP = 30
IMAGES = "data/f10/hop_depth/images.parquet"


def ask(adjective: str = "cool", noun: str = "place") -> str:
    """Give the question with these 2 words."""
    return QUESTION.replace(" place cool?", f" {noun} {adjective}?")


def locate(seq: list[int], image: int | None, first: int, place: int, cool: int, mark: int) -> dict[str, int]:
    """Give the index of each position in seq. Each token must occur after the image, in order."""
    start = max(i for i, t in enumerate(seq) if t == image) + 1 if image in seq else 0
    tail = seq[start:]
    at = {"first_text": start + tail.index(first), "place": start + tail.index(place)}
    at |= {"after_place": at["place"] + 1, "after_cool": at["place"] + 2, "final": len(seq) - 1}
    if seq[at["after_place"]] != cool or seq[at["after_cool"]] != mark:
        raise RuntimeError("the tokens after ` place` are not ` cool` and `?`")
    return {"after_image": start, **at} if image in seq else at


def atomic_save(path: Path, x: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # many workers can write 1 store
    with open(tmp, "wb") as f:
        np.save(f, x)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True, choices=list(N_LAYERS))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from jlens.hooks import ActivationRecorder
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from monocle import jlens_read
    from r3_readout import load_image

    out = Path(args.out_dir)
    for sub in ("reads", "residuals", "topk", "answers"):
        (out / sub).mkdir(parents=True, exist_ok=True)
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
    first, place, cool, mark = one("Is"), one(" place"), one(" cool"), one("?")
    yes, no, low_yes, low_no = one("Yes"), one("No"), one("yes"), one("no")

    def read(inputs: dict, at: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Give residuals and lens logits [positions, layers, ...], and the logits of the last position."""
        pos = torch.tensor(at, device=device)
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            h = torch.stack([rec.activations[l][0, pos].float() for l in layers], dim=1)
            x = torch.stack([lens_model.unembed(lens.transport(h[:, i], l)).float() for i, l in enumerate(layers)], dim=1)
        if not (torch.isfinite(x).all() and torch.isfinite(h).all()):
            raise RuntimeError("non-finite read")
        return h, x, logits

    names = lambda x, k: [tok.decode([int(i)]) for i in x.topk(k).indices]
    spider_ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok.encode(SPIDER, add_special_tokens=False)
    ids = torch.tensor([spider_ids], device=device)
    _, x, logits = read({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, [ids.shape[1] - 1])
    control = {f"L{l}": names(x[0, i], 8) for i, l in enumerate(layers)}
    print(f"[f10.5] spider, model top {names(logits, 5)}", flush=True)
    for l, words in control.items():
        print(f"[f10.5] spider {l}: {words}", flush=True)

    size = x.shape[-1]
    vocab = pd.DataFrame([(i, tok.decode([i])) for i in range(min(size, len(tok)))], columns=["token_id", "token"])
    vocab = vocab[vocab["token"].map(lambda w: bool(re.fullmatch(r" [a-z]{3,}", w)))].reset_index(drop=True)
    vocab.insert(0, "index", vocab.index)
    vocab.to_parquet(out / "vocab.parquet", index=False)
    keep = torch.tensor(vocab["token_id"].to_numpy(), device=device)

    def render(image: bool) -> str:
        text = {"type": "text", "text": QUESTION}
        msgs = [{"role": "user", "content": [{"type": "image"}, text] if image else [text]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    inputs = dict(proc(text=[render(False)], return_tensors="pt").to(device))
    at = locate(inputs["input_ids"][0].tolist(), None, first, place, cool, mark)
    h, x, _ = read(inputs, [at[p] for p in POSITIONS[1:]])
    blank = lambda a: np.concatenate([np.full_like(a[:1], np.nan), a], axis=0)
    atomic_save(out / "text_residuals.npy", blank(h.cpu().numpy()))
    atomic_save(out / "text_reads.npy", blank(x[..., keep].cpu().numpy()))

    images = pd.read_parquet(IMAGES)
    if args.limit:
        images = images.head(args.limit)
    (out / "manifest.json").write_text(json.dumps(
        {"model": args.model_key, "version": __version__, "question": QUESTION, "layout": "image first", "positions": POSITIONS, "layers": layers,
         "n_images": len(images), "chunk": args.chunk, "d": int(h.shape[-1]), "vocab": len(vocab), "lens": args.lens,
         "spider": control}, indent=1) + "\n")
    done = len(list((out / "answers").glob("answers-*.parquet")))
    print(f"[f10.5] {len(images)} images, layers {layers}, {len(vocab)} words; {done} shards done", flush=True)

    t0, count = time.perf_counter(), 0
    n_shards = (len(images) + args.chunk - 1) // args.chunk
    for shard in range(done, n_shards):
        part = images.iloc[shard * args.chunk:(shard + 1) * args.chunk]
        res, words, tops, answers = [], [], [], []
        for r in part.itertuples(index=False):
            inputs = dict(proc(text=[render(True)], images=[[load_image(args.image_root, r.sample_id)]], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0].tolist()
            at = locate(seq, image_token, first, place, cool, mark)
            if count == 0:
                print(f"[f10.5] check tokens {len(seq)}, {[(p, tok.decode([seq[at[p]]])) for p in POSITIONS]}", flush=True)
            h, x, logits = read(inputs, [at[p] for p in POSITIONS])
            res.append(h.cpu().numpy())
            words.append(x[..., keep].cpu().numpy())
            v, i = x.topk(TOP, dim=-1)
            for a, p in enumerate(POSITIONS):
                for b, l in enumerate(layers):
                    tops.extend({"image_idx": r.image_idx, "position": p, "layer": l, "rank": k, "token_id": int(i[a, b, k]), "logit": float(v[a, b, k])} for k in range(TOP))
            lp = torch.log_softmax(logits, dim=-1)
            answers.append({"image_idx": r.image_idx, "yes_no": float(lp[yes] - lp[no]), "yes_no_lc": float(lp[low_yes] - lp[low_no]),
                            "top": tok.decode([int(logits.argmax())]), "tokens": len(seq)})
            count += 1
        atomic_save(out / "reads" / f"reads-{shard:05d}.npy", np.stack(words))
        atomic_save(out / "residuals" / f"residuals-{shard:05d}.npy", np.stack(res))
        tp = pd.DataFrame(tops)
        tp["token"] = tp["token_id"].map(lambda i: tok.decode([i]))
        tp.to_parquet(out / "topk" / f"topk-{shard:05d}.parquet", index=False)
        pd.DataFrame(answers).to_parquet(out / "answers" / f"answers-{shard:05d}.parquet", index=False)  # the last write marks the shard done
        print(f"[f10.5] shard {shard + 1}/{n_shards}, {count / (time.perf_counter() - t0):.2f} images/s", flush=True)
    print("[f10.5] finished", flush=True)


if __name__ == "__main__":
    main()
