"""Direct logit attribution at layer 47: which component writes the new residual stream.

The gemma-4 decoder layer computes

    h47 = s * (h46 + A + M)

where `A` is the attention output after its post-norm, `M` the MLP
output after its post-norm (it reads `h46 + A`), and `s` the layer's
`layer_scalar`. The unnormalized logit of the final-layer decomposition,
`rd(w) = (W_U[w] * g) . h`, is linear in `h`, so at the last token
position

    rd47(w) = s * (rd_h46(w) + rd_A(w) + rd_M(w))

holds exactly (direct logit attribution). This analysis records the 3
parts and the total per image and word, in the format of
`monocle/probe_words.py` (4 pseudo-conditions: `h46`, `attn`, `mlp`,
`h47`; the `lp_` columns hold `rd`). It also saves the 4 hidden states
at the last token position (fp16), for the linear probe with held-out
neighborhoods (`scripts/probe_lastblock_parts_test.py`).

    sbatch monocle/monocle_gpu.sub monocle.probe_lastblock_parts --template chat \\
        --shard 0 --n-shards 4 --out-dir outputs/_monocle/probe_lastblock_parts/it
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import probe_frame as pf  # noqa: E402
from monocle.jlens_steer import load_model_and_lens  # noqa: E402
from monocle.probe_words import POOL, WORDS, log, user_text, word_ids  # noqa: E402

PARTS = ("h46", "attn", "mlp", "h47")


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
    for sub in PARTS:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard:02d}of{args.n_shards:02d}"

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    probe_ids, _ = word_ids(tok, WORDS)
    pool_ids, _ = word_ids(tok, POOL)
    pool_ids = {w: i for w, i in pool_ids.items() if w not in probe_ids}
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

    layers = lens_model.layers
    final = lens_model.n_layers - 1
    assert final == 47, f"final layer is {final}, not 47"
    block = layers[final]
    scalars = [float(l.layer_scalar.float().item()) for l in layers]
    s47 = scalars[final]
    log(f"layer_scalar of the 48 layers: {[round(x, 4) for x in scalars]}")
    norm = lens_model._final_norm
    W = lens_model._lm_head.weight  # [vocab, d]
    Wsel = W[ids].float()           # [n_words, d]
    # The final norm is h * g / rms(h). On a vector of ones rms = 1, so the
    # norm of ones gives g, the effective per-coordinate weight.
    g = norm(torch.ones(1, W.shape[1], device=W.device, dtype=W.dtype)).float()[0]

    got: dict[str, torch.Tensor] = {}
    hooks = [
        block.post_attention_layernorm.register_forward_hook(
            lambda m, i, o: got.__setitem__("attn", o[:, -1, :].float())),
        block.post_feedforward_layernorm.register_forward_hook(
            lambda m, i, o: got.__setitem__("mlp", o[:, -1, :].float())),
    ]
    log(f"template {args.template}, model {args.model_dir}; shard {args.shard}/{args.n_shards}, "
        f"{len(paths)} images; {len(probe_ids)} target + {len(pool_ids)} control words; s47 = {s47:.5f}")

    rows = {p: [] for p in PARTS}
    vecs = {p: [] for p in PARTS}
    geo, t0, checked = [], time.time(), False
    for b0 in range(0, len(paths), args.batch):
        chunk = paths[b0:b0 + args.batch]
        imgs = [[Image.open(q).convert("RGB")] for q in chunk]
        inputs = dict(proc(text=[rendered] * len(chunk), images=imgs,
                           return_tensors="pt").to("cuda:0"))
        am = inputs.get("attention_mask")
        if am is not None and not bool(am.all()):
            raise RuntimeError("padding inside a batch")
        inputs["use_cache"] = False
        with ActivationRecorder(layers, at=[46, 47]) as rec, torch.no_grad():
            model(**inputs)
            H = {"h46": rec.activations[46][:, -1, :].float(),
                 "h47": rec.activations[47][:, -1, :].float(),
                 "attn": got["attn"], "mlp": got["mlp"]}
        rebuilt = s47 * (H["h46"] + H["attn"] + H["mlp"])
        err = ((rebuilt - H["h47"]).norm(dim=1) / H["h47"].norm(dim=1)).max().item()
        if not checked:
            log(f"identity check s*(h46+A+M) == h47: max relative error {err:.5f}")
            hn = norm(H["h47"].to(W.dtype)).float()
            rms = torch.sqrt((H["h47"] ** 2).mean(dim=1))
            e2 = ((H["h47"] * g) - hn * rms[:, None]).abs().max().item()
            log(f"weight check h*g == norm(h)*rms: max abs error {e2:.5f}")
            checked = True
        if err > 0.02:
            raise RuntimeError(f"the layer-47 identity fails: relative error {err:.4f}")
        for p in PARTS:
            rd = (H[p] * g) @ Wsel.T                  # [B, n_words]
            vals = rd.cpu()
            vecs[p].append(H[p].half().cpu().numpy())
            for i, q in enumerate(chunk):
                row = {"image_path": q, "layer": 47}
                for j, w in enumerate(names):
                    row[f"lp_{w}"] = float(vals[i, j])
                rows[p].append(row)
        cos = torch.nn.functional.cosine_similarity
        for i, q in enumerate(chunk):
            geo.append({"image_path": q,
                        "n_h46": float(H["h46"][i].norm()), "n_attn": float(H["attn"][i].norm()),
                        "n_mlp": float(H["mlp"][i].norm()), "n_h47": float(H["h47"][i].norm()),
                        "cos_attn_h46": float(cos(H["attn"][i], H["h46"][i], dim=0)),
                        "cos_mlp_h46": float(cos(H["mlp"][i], H["h46"][i], dim=0)),
                        "cos_attn_mlp": float(cos(H["attn"][i], H["mlp"][i], dim=0)),
                        "cos_h47_h46": float(cos(H["h47"][i], H["h46"][i], dim=0))})
        if (b0 // args.batch) % 25 == 0:
            done = b0 + len(chunk)
            log(f"  {done}/{len(paths)} images, {(time.time() - t0) / done:.2f} s/image")
    for h in hooks:
        h.remove()
    for p in PARTS:
        pd.DataFrame(rows[p]).to_parquet(out_dir / p / f"words_{tag}.parquet", index=False)
    pd.DataFrame(geo).to_parquet(out_dir / f"geometry_{tag}.parquet", index=False)
    np.savez(out_dir / f"vectors_{tag}.npz", image_path=np.array(paths),
             layer_scalar=np.array(scalars, dtype=np.float32),
             **{p: np.concatenate(vecs[p]) for p in PARTS})
    log(f"wrote {out_dir}/{{{','.join(PARTS)}}}/words_{tag}.parquet + geometry + vectors "
        f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
