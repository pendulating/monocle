#!/usr/bin/env python3
"""Replay the whatami answers through a model and read each lens at each layer.

The served run (research/scripts/whatami/query.py) gave 10 answers for each of 1,000
images. This script teacher-forces each answer: the served prompt, then the
answer tokens, then the end-of-turn token, in one forward pass. The model is
causal, thus the residual at a position is the residual that the model had
when it chose the next token. A hook reads every block.

At each position that predicts an answer token (the last prompt position, then
each answer token; the last one predicts the end of the turn), and at each
layer, each lens gives a distribution over the vocabulary:

| Lens | Readout at layer l |
|---|---|
| `logit` | unembed(h_l): the plain logit lens, the baseline |
| `text`, `mm`, `mm_textafter` | unembed(J_l h_l), with J_l from lens_fit.py |

The last block needs no lens: each lens reads it as unembed(h), the model's own
logits. A layer that a lens does not cover (the mllama cross-attention layers
in the text lens) gets -1 and NaN.

The output has one row for each (image_idx, repeat, pos), in one directory for
each lens, one parquet file for each chunk of images:

| Column | Type | Content |
|---|---|---|
| image_idx, repeat, pos | int | pos 0 is the last prompt position |
| target_id | int | the token at pos + 1 (an answer token, or the end of the turn) |
| topk_ids | fixed [n_layers][k] int32 | the top-k tokens at each layer |
| topk_logprob | fixed [n_layers][k] float32 | their log-probabilities |
| target_rank | fixed [n_layers] int32 | the rank of target_id (0 = top) |
| target_logprob | fixed [n_layers] float32 | its log-probability |

The answer text is tokenized again, because the served run kept the text and
not the token ids. Each chunk logs how often the token count agrees with the
served completion_tokens, and the final-layer rank of the answer tokens. A
final-layer rank far from 0 means that the replay does not match the served
prompt.

Usage:
    python research/scripts/whatami/lens_replay.py --model gemma-4-12b --limit-images 2   # smoke
    python research/scripts/whatami/lens_replay.py --model gemma-4-12b --shard 0/1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from research.scripts.whatami import lens_common as lc

SAMPLE = lc.REPO / "data/whatami/sample_1k.parquet"
RUNS = lc.REPO / "multirun/whatami/sample_1k"
OUT_ROOT = lc.REPO / "multirun/whatami/lens_replay"
#: Seconds spent in each part of a chunk, for the chunk log line.
TIMES = {"prompt": 0.0, "forward": 0.0, "lenses": 0.0, "write": 0.0}


def load_lenses(L: lc.Loaded, names: list[str], dtype) -> dict[str, dict[int, torch.Tensor]]:
    """{lens: {layer: J_l on the GPU}}; `logit` is the identity (no matrices)."""
    from jlens import JacobianLens

    out: dict[str, dict[int, torch.Tensor]] = {"logit": {}}
    for nm in names:
        path = lc.lens_dir(L.spec.key, nm) / "lens.pt"
        lens = JacobianLens.load(str(path))
        out[nm] = {int(l): J.to("cuda:0", dtype) for l, J in lens.jacobians.items()}
        lc.log(f"lens {nm}: {path} ({len(out[nm])} layers, {lens.n_prompts} items, meta={lens.meta.get('estimator')})")
    return out


@torch.no_grad()
def read_answer(L: lc.Loaded, lenses, inputs: dict, prompt_len: int, k: int):
    """One teacher-forced pass. Returns {lens: arrays} for the answer positions."""
    from jlens.hooks import ActivationRecorder

    n_layers = L.lens_model.n_layers
    ids = inputs["input_ids"][0]
    seq = int(ids.shape[0])
    pos = torch.arange(prompt_len - 1, seq - 1, device=ids.device)
    targets = ids[pos + 1]
    layers = list(range(n_layers))
    t0 = time.perf_counter()
    with ActivationRecorder(L.lens_model.layers, at=layers) as rec:
        L.model(**inputs, use_cache=False)
        hidden = {l: rec.activations[l][0, pos] for l in layers if l in rec.activations}
    torch.cuda.synchronize()
    TIMES["forward"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    n = int(pos.numel())
    vocab = int(L.lens_model.unembed(hidden[n_layers - 1][:1]).shape[-1])
    # Layers per unembed call: keep the fp32 log-probs under about 8 GB.
    per_call = max(1, int(8e9 // (n * vocab * 4)))
    res = {}
    for name, mats in lenses.items():
        top_ids = np.full((n, n_layers, k), -1, np.int32)
        top_lp = np.full((n, n_layers, k), np.nan, np.float32)
        t_rank = np.full((n, n_layers), -1, np.int32)
        t_lp = np.full((n, n_layers), np.nan, np.float32)
        todo = []
        for l in layers:
            if l not in hidden:
                continue
            h = hidden[l]
            if l == n_layers - 1 or name == "logit":
                todo.append((l, h))
            elif l in mats:
                J = mats[l]
                todo.append((l, (h.to(J.dtype) @ J.T).to(h.dtype)))
        for c in range(0, len(todo), per_call):
            part = todo[c:c + per_call]
            X = torch.stack([x for _, x in part])                     # [m, n, d]
            lp = torch.log_softmax(L.lens_model.unembed(X.flatten(0, 1)).float(), dim=-1)
            lp = lp.view(len(part), n, -1)
            v, i = lp.topk(k, dim=-1)                                 # [m, n, k]
            tl = lp.gather(2, targets[None, :, None].expand(len(part), n, 1)).squeeze(2)
            rank = (lp > tl[..., None]).sum(2)
            idx = [l for l, _ in part]
            top_ids[:, idx] = i.permute(1, 0, 2).cpu().numpy()
            top_lp[:, idx] = v.permute(1, 0, 2).cpu().numpy()
            t_rank[:, idx] = rank.T.cpu().numpy()
            t_lp[:, idx] = tl.T.cpu().numpy()
            del lp, X
        res[name] = (top_ids, top_lp, t_rank, t_lp)
    TIMES["lenses"] += time.perf_counter() - t0
    return pos.cpu().numpy() - (prompt_len - 1), targets.cpu().numpy(), res


def _fixed(arr: np.ndarray) -> pa.Array:
    """A [rows, a] or [rows, a, b] numpy array as nested fixed-size lists, no copy per row."""
    flat = pa.array(arr.reshape(-1))
    for size in reversed(arr.shape[1:]):
        flat = pa.FixedSizeListArray.from_arrays(flat, size)
    return flat


def to_table(rows_meta: list[dict], arrays: list[tuple]) -> pa.Table:
    cat = lambda j: np.concatenate([a[2][j] for a in arrays])  # noqa: E731
    n_of = [len(a[0]) for a in arrays]
    return pa.table({
        "image_idx": pa.array(np.repeat([m["image_idx"] for m in rows_meta], n_of).astype(np.int32)),
        "repeat": pa.array(np.repeat([m["repeat"] for m in rows_meta], n_of).astype(np.int16)),
        "pos": pa.array(np.concatenate([a[0] for a in arrays]).astype(np.int32)),
        "target_id": pa.array(np.concatenate([a[1] for a in arrays]).astype(np.int32)),
        "topk_ids": _fixed(cat(0)),
        "topk_logprob": _fixed(cat(1)),
        "target_rank": _fixed(cat(2)),
        "target_logprob": _fixed(cat(3)),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(lc.MODELS))
    ap.add_argument("--shard", default="0/1", help="over the chunks of images")
    ap.add_argument("--chunk-images", type=int, default=25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lens-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--limit-images", type=int, default=0, help="first N images only (smoke)")
    ap.add_argument("--image-root", default=None, help="staged copies as <sample_id>.jpg")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--lenses", nargs="+", default=None,
                    help="lens dirs to read (default: the J-lenses of the family); logit is always read")
    ap.add_argument("--check-only", action="store_true",
                    help="check the prompts and replay --limit-images with the logit lens; write nothing")
    args = ap.parse_args()

    L = lc.load(args.model, force_bos=False)
    names = [] if args.check_only else (args.lenses or lc.LENSES[L.spec.family])
    lenses = load_lenses(L, names, getattr(torch, args.lens_dtype))
    lens_names = ["logit", *names]
    if args.check_only:
        args.limit_images = args.limit_images or 3
        args.out_root = str(Path(lc.REPO) / "multirun/whatami/lens_replay_check")

    sample = pd.read_parquet(SAMPLE)
    answers = pd.read_parquet(RUNS / args.model / "responses.parquet",
                              columns=["image_idx", "repeat", "sample_id", "response", "completion_tokens", "error"])
    if answers["error"].notna().any():
        raise ValueError("responses.parquet holds failed requests; finish the served run first")
    images = sorted(sample["image_idx"].tolist())
    if args.limit_images:
        images = images[: args.limit_images]
    chunks = [images[i:i + args.chunk_images] for i in range(0, len(images), args.chunk_images)]
    si, sn = (int(x) for x in args.shard.split("/"))
    out_dir = Path(args.out_root) / args.model
    for nm in lens_names:
        (out_dir / nm).mkdir(parents=True, exist_ok=True)
    # Merge with an earlier config: a later replay (an R-lens) adds its lenses.
    cfg_path = out_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"lens_files": {}, "lenses": []}
    cfg.update({"model": args.model, "k": args.k, "lens_dtype": args.lens_dtype})
    cfg["lenses"] = sorted(set(cfg["lenses"]) | set(lens_names))
    cfg["lens_files"].update({nm: str(lc.lens_dir(args.model, nm) / "lens.pt") for nm in names})
    cfg_path.write_text(json.dumps(cfg, indent=2))
    path_of = dict(zip(sample["image_idx"], sample["image_path"]))
    sid_of = dict(zip(sample["image_idx"], sample["sample_id"]))
    tok = L.proc.tokenizer

    for ci, chunk in enumerate(chunks):
        if ci % sn != si:
            continue
        part = f"part-{ci:04d}.parquet"
        # Read only the lenses whose file is missing, thus a later lens (an
        # R-lens) adds its files beside the ones that exist.
        todo = [nm for nm in lens_names if not (out_dir / nm / part).exists()]
        if not todo:
            continue
        chunk_lenses = {nm: lenses[nm] for nm in todo}
        t0 = time.time()
        TIMES.update(dict.fromkeys(TIMES, 0.0))
        metas, per_lens = [], {nm: [] for nm in todo}
        n_match = n_ans = 0
        final_rank0 = []
        for image_idx in chunk:
            src = path_of[image_idx]
            if args.image_root:
                src = str(Path(args.image_root) / f"{sid_of[image_idx]}.jpg")
            tp = time.perf_counter()
            prompt = lc.build_prompt(L, src, lc.PROMPT)
            TIMES["prompt"] += time.perf_counter() - tp
            lc.check_prompt_length(L, prompt)
            plen = int(prompt["input_ids"].shape[1])
            rows = answers[answers.image_idx == image_idx].sort_values("repeat")
            for r in rows.itertuples():
                resp_ids = tok(r.response, add_special_tokens=False).input_ids
                n_ans += 1
                n_match += int(len(resp_ids) + 1 == r.completion_tokens or len(resp_ids) == r.completion_tokens)
                inputs = lc.append_tokens(prompt, [*resp_ids, L.end_of_turn_id])
                pos, tgt, res = read_answer(L, chunk_lenses, inputs, plen, args.k)
                metas.append({"image_idx": int(image_idx), "repeat": int(r.repeat)})
                for nm in todo:
                    per_lens[nm].append((pos, tgt, res[nm]))
                any_res = res[todo[0]]
                final_rank0.append(float((any_res[2][:, -1] == 0).mean()))
        tw = time.perf_counter()
        for nm in todo:
            pq.write_table(to_table(metas, per_lens[nm]), out_dir / nm / f"{part}.tmp", compression="zstd")
            (out_dir / nm / f"{part}.tmp").rename(out_dir / nm / part)
        TIMES["write"] += time.perf_counter() - tw
        lc.log(f"chunk {ci + 1}/{len(chunks)}: {len(metas)} answers in {time.time() - t0:.0f}s "
               f"(prompt {TIMES['prompt']:.0f}s, forward {TIMES['forward']:.0f}s, "
               f"lenses {TIMES['lenses']:.0f}s, write {TIMES['write']:.0f}s); "
               f"token count agrees with served for {n_match}/{n_ans}; "
               f"final-layer top-1 = answer token for {np.mean(final_rank0):.3f} of positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
