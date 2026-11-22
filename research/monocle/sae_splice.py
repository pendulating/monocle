"""G1 splice test: replace the residual with the dictionary reconstruction, read the verdict.

For each dictionary, a forward hook on `lens_model.layers[L]` replaces the
block output with the dictionary's reconstruction of it (threshold encoder,
in float32, cast back to the model dtype):

- A<L> dictionaries: at the answer position (the last token);
- P<L> dictionaries: at every image-token position.

The read is the canonical single-image verdict of each of the 7 cases,
c = logit(positive) - logit(negative) at the last block, on held-out images.
The gate: the verdict sign flips on less than 2 percent of the images.

    sbatch --time=01:30:00 research/monocle/monocle.sub monocle.sae_splice --arm it \
        --dicts P22_w32768_k48_s1,P40_w32768_k48_s1,A42_w8192_k48_s1,A46_w8192_k48_s1,P22_w16384_k48_s1

Outputs under `--out-root/<arm>/splice_test/`: `splice.csv` (one row per
image, case and dictionary), `splice_summary.csv` and `splice.md`.
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
sys.path.insert(0, str(REPO))

from research.monocle import canonical  # noqa: E402
from research.monocle import patch_causal as pc  # noqa: E402
from research.monocle import single  # noqa: E402
from research.monocle.jlens_steer import load_model_and_lens  # noqa: E402
from research.monocle.probe_words import log  # noqa: E402
from research.monocle.sae_extract import frame_texts  # noqa: E402

SAMPLE_DEFAULT = REPO / "outputs/_monocle/sae/sample/images.parquet"
MODEL_DIRS = {"it": "/share/ju/matt/zoo/models/gemma-4-12B-it",
              "base": "/share/ju/matt/zoo/models/gemma-4-12B"}


class Dictionary:
    """A trained dictionary, read from `dict.pt`, with the threshold encoder."""

    def __init__(self, path: Path, device) -> None:
        o = torch.load(path, map_location=device, weights_only=False)
        self.W_enc = o["W_enc"].float()
        self.W_dec = o["W_dec"].float()
        self.b_enc = o["b_enc"].float()
        self.b_dec = o["b_dec"].float()
        self.center = o.get("center", "pooled")
        self.means = o["means"].float() if "means" in o else o["mean"].float()[None]
        self.frames = o.get("frames")
        self.scale = o["scale"].float()
        self.thr = o["threshold"].float()
        self.block = int(o["config"]["dict"][1:])
        self.is_patch = o["config"]["dict"].startswith("P")

    def cond(self, case: str, n_pos: int, device) -> torch.Tensor:
        """The condition index of the spliced rows: the patch position, or the frame."""
        if self.center != "cond":
            return torch.zeros(n_pos, dtype=torch.long, device=device)
        if self.is_patch:
            return torch.arange(n_pos, device=device)
        return torch.full((n_pos,), self.frames.index(case), dtype=torch.long, device=device)

    @torch.no_grad()
    def recon(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mu = self.means[c]
        xn = (x.float() - mu) / self.scale
        pre = (xn - self.b_dec) @ self.W_enc + self.b_enc
        z = torch.relu(pre) * (pre > self.thr)
        return (z @ self.W_dec + self.b_dec) * self.scale + mu


def make_splice_hook(d: Dictionary, positions: list[int], case: str):
    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        t = output[0] if is_tuple else output
        new = t.clone()
        rows = new[:, positions, :]
        c = d.cond(case, len(positions), new.device).repeat(rows.shape[0])
        new[:, positions, :] = d.recon(rows.reshape(-1, rows.shape[-1]), c).reshape(rows.shape).to(new.dtype)
        return (new, *tuple(output[1:])) if is_tuple else new
    return hook


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["it", "base"])
    ap.add_argument("--dicts", required=True, help="comma list of dictionary dir names under <out-root>/<arm>")
    ap.add_argument("--n", type=int, default=400, help="held-out images")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--images", default=str(SAMPLE_DEFAULT))
    ap.add_argument("--out-root", default=str(REPO / "outputs/_monocle/sae"))
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args(argv)
    from PIL import Image

    template = "chat" if args.arm == "it" else "raw"
    model_dir = args.model_dir or MODEL_DIRS[args.arm]
    root = Path(args.out_root) / args.arm
    out = root / "splice_test"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.images, columns=["image_path", "split"])
    paths = df[df.split == "heldout"].sample(n=args.n, random_state=args.seed).image_path.tolist()

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(model_dir)
    tok = proc.tokenizer
    dev = next(model.parameters()).device
    dicts = {name: Dictionary(root / name / "dict.pt", dev) for name in args.dicts.split(",")}
    texts = frame_texts(tok, tmpl, template)
    log(f"{args.arm}: {len(paths)} held-out images, dictionaries {list(dicts)}, cases {list(canonical.CASES)}")
    t0 = time.time()
    rows = []
    for case in canonical.CASES:
        pos_id, neg_id = single.contrast_ids(tok, case)
        for b0 in range(0, len(paths), args.batch):
            chunk = paths[b0 : b0 + args.batch]
            imgs = [Image.open(q).convert("RGB") for q in chunk]
            inputs = dict(proc(text=[texts[case]] * len(imgs), images=[[im] for im in imgs],
                               return_tensors="pt").to(dev))
            am = inputs.get("attention_mask")
            if am is not None and not bool(am.all()):
                raise RuntimeError("padding inside a batch")
            ids = inputs["input_ids"]
            img_pos = (ids[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()
            if not bool((ids == image_token_id).sum(dim=1).eq(len(img_pos)).all()):
                raise RuntimeError("image-token count differs inside a batch")
            h = pc.final_residual(model, lens_model, inputs, graph=False)
            c_clean = pc.fp32_contrast(lens_model, h, pos_id, neg_id).cpu().numpy()
            for name, d in dicts.items():
                positions = img_pos if d.is_patch else [int(ids.shape[1]) - 1]
                handle = lens_model.layers[d.block].register_forward_hook(make_splice_hook(d, positions, case))
                try:
                    h = pc.final_residual(model, lens_model, inputs, graph=False)
                finally:
                    handle.remove()
                c_s = pc.fp32_contrast(lens_model, h, pos_id, neg_id).cpu().numpy()
                for p, a, b in zip(chunk, c_clean, c_s):
                    rows.append({"image_path": p, "case": case, "dict": name,
                                 "c_clean": float(a), "c_splice": float(b)})
        log(f"  {case}: {len(paths)} images, {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res.to_csv(out / "splice.csv", index=False)

    res["flip"] = np.sign(res.c_clean) != np.sign(res.c_splice)
    res["abs_dc"] = (res.c_splice - res.c_clean).abs()
    per = res.groupby(["dict", "case"]).agg(n=("flip", "size"), flip_rate=("flip", "mean"),
                                           mean_abs_dc=("abs_dc", "mean"),
                                           mean_c_clean=("c_clean", "mean"),
                                           mean_c_splice=("c_splice", "mean")).reset_index()
    pooled = res.groupby("dict").agg(n=("flip", "size"), flip_rate=("flip", "mean"),
                                     mean_abs_dc=("abs_dc", "mean")).reset_index()
    pooled["case"] = "all"
    summary = pd.concat([per, pooled], ignore_index=True)
    summary["gate"] = np.where(summary.flip_rate < 0.02, "pass", "FAIL")
    summary.to_csv(out / "splice_summary.csv", index=False)
    md = [f"# Splice test, {args.arm}: {len(paths)} held-out images, seed {args.seed}\n",
          "Flip rate: the fraction of images whose verdict sign changes when the",
          "dictionary reconstruction replaces the residual. Gate: below 0.02.\n",
          "## Pooled over the 7 cases\n",
          pooled.assign(gate=np.where(pooled.flip_rate < 0.02, "pass", "FAIL")).round(4).to_markdown(index=False),
          "\n## Per case\n", per.round(4).to_markdown(index=False), ""]
    (out / "splice.md").write_text("\n".join(md))
    log("\n" + "\n".join(md[:5]))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
