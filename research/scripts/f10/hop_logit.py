#!/usr/bin/env python
"""F10.7: logit metrics from the saved residuals of F10.6.

The script gives the metrics that need the whole vocabulary or the uncapped logits. It
reads the residuals of `data/f10/hop_all/{model}/`, thus the images and tokens are those
of F10.6. A read here is `W norm(J_l h)`. The softcap of Gemma comes after this step, and
the script applies it only where the name says `capped`.

The store of F10.6 lacks the last layer. Thus the script runs each image once more and
saves the last residuals, and a read has a true target: the output of the model at the
same token.

| File in `--out-dir` | Content |
|---------------------|---------|
| `manifest.json` | the axes, the answer tokens, the softcap and the checks |
| `last/last-*.npy` | float32 [images, tail, d]: the last block output, for each tail token |
| `metrics/metrics-*.npz` | 1 file for each shard. All arrays are [images, tail, layers, ...] |
| `uncapped/reads-*.npy` | float32 [images, tail, layers, vocab]: uncapped reads of the store words. Only with a softcap |
| `answers.parquet` | `image_idx`, and the model log probability of each answer token and the pooled margin |
| `text.npz` | the same answer logits for the text with no image; NaN at a token that the text does not have |
| `unembed_store.npy`, `unembed_mean.npy`, `unembed_cov.npy` | the rows of the store words, the mean row and the row covariance |
| `answer_rows.npy` | the rows of the answer tokens |

| Array in `metrics-*.npz` | Content |
|--------------------------|---------|
| `lens` | uncapped logits of the answer tokens, with the J++ lens |
| `plain` | the same with `J = I`, the plain logit lens |
| `rms` | `rms(J_l h)`. The final norm divides the read by this value |
| `mu`, `sd` | the mean and the SD of the uncapped read over the whole vocabulary |
| `js`, `js_capped` | Jensen-Shannon divergence to the model output at the same token, in nats |
| `kl_capped` | KL(model output || read), with the softcap on both |
| `agree` | 1 if the top token of the read is the top token of the model output |

Rules: research/docs/facct-runbook.md, F10.7.

Use:
    python research/scripts/f10/hop_logit.py --model-key gemma-4-31b --model-dir DIR --lens FILE \
        --image-root DIR --out-dir data/f10/hop_logit/gemma-4-31b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.scripts.f10.hop_all import ANSWERS, parts, pooled_margin  # noqa: E402
from research.scripts.f10.hop_depth import IMAGES, ROOT, ask, atomic_save  # noqa: E402

__version__ = "1.2.1"

BATCH = 256
EPS = 1e-30


def softcap(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    return x if cap is None else cap * torch.tanh(x / cap)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Give the Jensen-Shannon divergence in nats, on the last axis of 2 distributions."""
    m = 0.5 * (p + q)
    a = (p * (torch.log(p + EPS) - torch.log(m + EPS))).sum(-1)
    b = (q * (torch.log(q + EPS) - torch.log(m + EPS))).sum(-1)
    return 0.5 * (a + b)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Give KL(p || q) in nats, on the last axis."""
    return (p * (torch.log(p + EPS) - torch.log(q + EPS))).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="the first images of each shard, for a test")
    ap.add_argument("--shards", type=int, default=0, help="the first shards only, for a test")
    ap.add_argument("--adjective", default="cool", help="the word of the question. `--model-key` then names the store, for example gemma-4-31b-nice")
    ap.add_argument("--noun", default="place", help="the noun of the question. `--model-key` then names the store, for example gemma-4-31b-scene-nice")
    args = ap.parse_args()

    from jlens.hooks import ActivationRecorder
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from research.scripts.f10.hop_store import ALL_ROOT, open_store
    from research.monocle import jlens_read
    from research.scripts.f8.r3_readout import load_image

    out = Path(args.out_dir)
    for sub in ("last", "metrics", "uncapped"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    store = open_store(args.model_key, ALL_ROOT)
    head = store.manifest
    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, device_map=device,
                                                        attn_implementation="sdpa").eval()
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens)
    tok = proc.tokenizer
    cap = lens_model._logit_softcap
    layers, last_layer = head["layers"], lens_model.n_layers - 1
    image_token = int(model.config.image_token_id)
    ids = [tok.encode(t, add_special_tokens=False) for t in ANSWERS]
    assert all(len(i) == 1 for i in ids) and len({i[0] for i in ids}) == len(ANSWERS), f"an answer form is not 1 distinct token: {ids}"
    answer = torch.tensor([i[0] for i in ids], device=device)
    keep = torch.tensor(store.vocab["token_id"].to_numpy(), device=device)
    print(f"[f10.7] {args.model_key}: softcap {cap}, last layer {last_layer}, answers {answer.tolist()}", flush=True)

    def raw(h: torch.Tensor) -> torch.Tensor:
        """Give the logits before the softcap."""
        return lens_model._lm_head(lens_model._final_norm(h)).float()

    def render(image: bool) -> str:
        text = {"type": "text", "text": ask(args.adjective, args.noun)}
        msgs = [{"role": "user", "content": [{"type": "image"}, text] if image else [text]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def last_block(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        with ActivationRecorder(lens_model.layers, at=[last_layer]) as rec, torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0].float()
            h = rec.activations[last_layer][0].float()
        return h, logits

    with torch.inference_mode():
        weight = lens_model._lm_head.weight.float()
        mean_row, centred = weight.mean(0), weight - weight.mean(0)
        cov = centred.T @ centred / (weight.shape[0] - 1)
        atomic_save(out / "unembed_mean.npy", mean_row.cpu().numpy())
        atomic_save(out / "unembed_cov.npy", cov.cpu().numpy())
        atomic_save(out / "unembed_store.npy", weight[keep].cpu().numpy())
        atomic_save(out / "answer_rows.npy", weight[answer].cpu().numpy())
        del centred, cov

    text_inputs = dict(proc(text=[render(False)], return_tensors="pt").to(device))
    text_seq = text_inputs["input_ids"][0].tolist()
    start = text_seq.index(tok.encode("Is", add_special_tokens=False)[0])
    h_text, _ = last_block(text_inputs)
    text_h = torch.from_numpy(np.load(store.path / "text_residuals.npy")).to(device)
    gap = text_h.shape[0] - (len(text_seq) - start)
    text_lens, text_plain = torch.full((*text_h.shape[:2], len(ANSWERS)), torch.nan), torch.full((*text_h.shape[:2], len(ANSWERS)), torch.nan)
    with torch.inference_mode():
        for b, l in enumerate(layers):
            text_lens[gap:, b] = raw(lens.transport(text_h[gap:, b], l))[..., answer].cpu()
            text_plain[gap:, b] = raw(text_h[gap:, b])[..., answer].cpu()
        text_true = torch.full((text_h.shape[0], len(ANSWERS)), torch.nan)
        text_true[gap:] = softcap(raw(h_text[start:]), cap)[..., answer].cpu()
    np.savez(out / "text.npz", lens=text_lens.numpy(), plain=text_plain.numpy(), true=text_true.numpy())

    images = pd.read_parquet(ROOT / IMAGES)
    chunk, n_shards = head["chunk"], (len(images) + head["chunk"] - 1) // head["chunk"]
    if args.shards:
        n_shards = min(n_shards, args.shards)
    checks = {"max_output_gap": 0.0, "max_capped_gap": 0.0}
    answers = []
    t0, count = time.perf_counter(), 0
    for shard in range(n_shards):
        path = out / "metrics" / f"metrics-{shard:05d}.npz"
        part = images.iloc[shard * chunk:(shard + 1) * chunk]
        if args.limit:
            part = part.head(args.limit)
        n = len(part)
        last_path = out / "last" / f"last-{shard:05d}.npy"
        answer_path = out / "last" / f"answers-{shard:05d}.parquet"
        if not (last_path.exists() and answer_path.exists()):
            rows, ans = [], []
            for r in part.itertuples(index=False):
                inputs = dict(proc(text=[render(True)], images=[[load_image(args.image_root, r.sample_id)]], return_tensors="pt").to(device))
                _, _, tail = parts(inputs["input_ids"][0].tolist(), image_token)
                h, logits = last_block(inputs)
                with torch.inference_mode():
                    again = softcap(raw(h[tail]), cap)
                checks["max_output_gap"] = max(checks["max_output_gap"], float((again - logits[tail]).abs().max()))
                lp = torch.log_softmax(logits[-1], -1)[answer].cpu()
                ans.append({"image_idx": r.image_idx, **{f"lp_{t}": float(v) for t, v in zip(ANSWERS, lp)}, "margin": float(pooled_margin(lp))})
                rows.append(h[tail].cpu().numpy())
            atomic_save(last_path, np.stack(rows))
            pd.DataFrame(ans).to_parquet(answer_path, index=False)
        answers.append(pd.read_parquet(answer_path))
        if path.exists():
            count += n
            continue
        # Read each file once, in sequence. 15 strided passes through a memory map are slow on NFS.
        res = np.load(store.path / "residuals" / f"residuals-{shard:05d}.npy")[:n]
        saved = np.load(store.path / "reads" / f"reads-{shard:05d}.npy")[:n]
        last = torch.from_numpy(np.load(last_path)).to(device)
        n_tail, n_layer = res.shape[1], res.shape[2]
        keys = ("lens", "plain", "rms", "mu", "sd", "js", "js_capped", "kl_capped", "agree")
        got = {k: [] for k in keys}
        words = []
        with torch.inference_mode():
            target = torch.cat([raw(x) for x in last.reshape(-1, last.shape[-1]).split(BATCH)])
            for b, l in enumerate(layers):
                flat = torch.from_numpy(np.array(res[:, :, b])).to(device).reshape(n * n_tail, -1)
                row = {k: [] for k in keys}
                w = []
                for i, x in enumerate(flat.split(BATCH)):
                    moved = lens.transport(x, l)
                    z, plain = raw(moved), raw(x)
                    if not torch.isfinite(z).all():
                        raise RuntimeError("non-finite read")
                    true = target[i * BATCH:i * BATCH + len(x)]
                    p, q = torch.softmax(true, -1), torch.softmax(z, -1)
                    pc, qc = torch.softmax(softcap(true, cap), -1), torch.softmax(softcap(z, cap), -1)
                    row["lens"].append(z[:, answer])
                    row["plain"].append(plain[:, answer])
                    row["rms"].append(moved.pow(2).mean(-1).sqrt())
                    row["mu"].append(z.mean(-1))
                    row["sd"].append(z.std(-1))
                    row["js"].append(js_divergence(p, q))
                    row["js_capped"].append(js_divergence(pc, qc))
                    row["kl_capped"].append(kl_divergence(pc, qc))
                    row["agree"].append((z.argmax(-1) == true.argmax(-1)).float())
                    w.append(z[:, keep])
                w = torch.cat(w).reshape(n, n_tail, -1).cpu()
                check = softcap(w, cap).numpy() - np.array(saved[:, :, b])
                checks["max_capped_gap"] = max(checks["max_capped_gap"], float(np.abs(check).max()))
                words.append(w.numpy())
                for k in keys:
                    v = torch.cat(row[k]).cpu().numpy()
                    got[k].append(v.reshape(n, n_tail, *v.shape[1:]))
        if cap is not None:
            atomic_save(out / "uncapped" / f"reads-{shard:05d}.npy", np.stack(words, axis=2))
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{k: np.stack(v, axis=2) for k, v in got.items()})
        os.replace(tmp, path)  # the last write marks the shard done
        count += n
        print(f"[f10.7] shard {shard + 1}/{n_shards}, {count / (time.perf_counter() - t0):.2f} images/s, checks {checks}", flush=True)
    pd.concat(answers, ignore_index=True).to_parquet(out / "answers.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(
        {"model": args.model_key, "version": __version__, "positions": head["positions"], "layers": layers, "answers": list(ANSWERS),
         "softcap": cap, "last_layer": last_layer, "n_images": sum(len(a) for a in answers), "checks": checks}, indent=1) + "\n")
    print(f"[f10.7] finished, checks {checks}", flush=True)


if __name__ == "__main__":
    main()
