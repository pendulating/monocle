"""G2 diagnostic: seed-to-seed feature match, split by how often a feature fires.

For each (arm, dict, width) with seeds 1 and 2: the firing rate of every
latent on the validation vectors (threshold encoder), the best decoder cosine
of each seed-1 latent over seed 2, and the matched fraction (cosine above
0.7) over all latents, over the live latents, over the latents that fire on
more than 1 in 10,000 vectors, over the 1,000 most frequent, and weighted by
firing mass. Runs on klara (the validation vectors are node-local):

    sbatch --time=00:40:00 monocle/monocle.sub monocle.sae_seedmatch --arm it

Writes `<dict dir of seed 1>/seedmatch.json` and `<out-root>/seedmatch_<arm>.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from monocle.sae_train import D, SAMPLE_DEFAULT, load_answer, load_patch_part, patch_parts, train_image_set  # noqa: E402


def load_dict(path: Path, dev):
    o = torch.load(path, map_location=dev, weights_only=False)
    return {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}


def center(d: dict, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
    """The dictionary's input: centered by the pooled mean, or by the mean of
    each row's condition (frame or patch position) for a `--center cond` run."""
    if d.get("center") == "cond":
        if c is None:
            raise ValueError("a conditionally centered dictionary needs the condition of every row")
        return (x.float() - d["means"][c]) / d["scale"]
    return (x.float() - d["mean"]) / d["scale"]


@torch.no_grad()
def encode(d: dict, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
    pre = (center(d, x, c) - d["b_dec"]) @ d["W_enc"] + d["b_enc"]
    return torch.relu(pre) * (pre > d["threshold"])


@torch.no_grad()
def firing_rate(d: dict, x: torch.Tensor, c: torch.Tensor | None = None, batch: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """(rate per latent, mean activation mass per latent) on x, threshold encoder."""
    width = d["W_enc"].shape[1]
    fires = torch.zeros(width, device=x.device)
    mass = torch.zeros(width, device=x.device)
    for i in range(0, x.shape[0], batch):
        z = encode(d, x[i : i + batch], None if c is None else c[i : i + batch])
        fires += (z > 0).sum(0)
        mass += z.sum(0)
    return fires / x.shape[0], mass / x.shape[0]


@torch.no_grad()
def act_corr_match(d1: dict, d2: dict, x: torch.Tensor, c: torch.Tensor | None = None, batch: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Best Pearson correlation of each seed-1 latent's activation with any
    seed-2 latent on x, and the index of that partner. A latent that never
    fires gets 0. Two features can share a concept (correlation near 1) and
    still differ in decoder direction, so this match is the check on the
    cosine rule."""
    w1, w2 = d1["W_enc"].shape[1], d2["W_enc"].shape[1]
    n = x.shape[0]
    s1 = torch.zeros(w1, device=x.device, dtype=torch.float64)
    s2 = torch.zeros(w2, device=x.device, dtype=torch.float64)
    q1 = torch.zeros(w1, device=x.device, dtype=torch.float64)
    q2 = torch.zeros(w2, device=x.device, dtype=torch.float64)
    g = torch.zeros(w1, w2, device=x.device)
    for i in range(0, n, batch):
        cb = None if c is None else c[i : i + batch]
        z1 = encode(d1, x[i : i + batch], cb)
        z2 = encode(d2, x[i : i + batch], cb)
        s1 += z1.sum(0).double()
        s2 += z2.sum(0).double()
        q1 += (z1 * z1).sum(0).double()
        q2 += (z2 * z2).sum(0).double()
        g += z1.t() @ z2
    m1, m2 = s1 / n, s2 / n
    sd1 = torch.sqrt((q1 / n - m1 * m1).clamp_min(0))
    sd2 = torch.sqrt((q2 / n - m2 * m2).clamp_min(0))
    best, idx = [], []
    chunk = 2048  # the full [w1, w2] correlation in float64 does not fit a 48 GB card
    for i in range(0, w1, chunk):
        cov = g[i : i + chunk].double() / n - m1[i : i + chunk, None] * m2[None, :]
        den = sd1[i : i + chunk, None] * sd2[None, :]
        corr = torch.where(den > 0, cov / den.clamp_min(1e-12), torch.zeros_like(cov))
        b, a = corr.max(dim=1)
        best.append(b)
        idx.append(a)
        del cov, den, corr
    del g
    return torch.cat(best).float(), torch.cat(idx)


@torch.no_grad()
def best_cos(w1: torch.Tensor, w2: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    w1 = w1 / w1.norm(dim=1, keepdim=True)
    w2 = w2 / w2.norm(dim=1, keepdim=True)
    out = []
    for i in range(0, w1.shape[0], chunk):
        out.append((w1[i : i + chunk] @ w2.t()).max(dim=1).values)
    return torch.cat(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["it", "base"])
    ap.add_argument("--dicts", default="A42_w8192_k48,A46_w8192_k48,P22_w32768_k48,P40_w32768_k48")
    ap.add_argument("--cos", type=float, default=0.7)
    ap.add_argument("--data-root", default="/scratch/mwf62/sae")
    ap.add_argument("--out-root", default=str(REPO / "outputs/_monocle/sae"))
    ap.add_argument("--val-parts", type=int, default=2)
    ap.add_argument("--suffix", default="", help="suffix on the summary file name")
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    root = Path(args.out_root) / args.arm
    data = Path(args.data_root) / args.arm
    keep, val_keep = train_image_set(SAMPLE_DEFAULT)
    rows = []
    for entry in args.dicts.split(","):
        stem, _, tag = entry.partition("@")  # P22_w32768_k48@e6 -> dirs P22_w32768_k48_s<seed>_e6
        sfx = f"_{tag}" if tag else ""
        dname = stem.split("_")[0]
        d1 = load_dict(root / f"{stem}_s1{sfx}" / "dict.pt", dev)
        d2 = load_dict(root / f"{stem}_s2{sfx}" / "dict.pt", dev)
        if dname.startswith("P"):
            _, val_parts = patch_parts(data, dname, args.val_parts)
            x = torch.cat([load_patch_part(f) for f in val_parts]).to(dev)
            c = torch.arange(x.shape[0], device=dev) % 256
        else:
            x, c = load_answer(data, dname, val_keep)
            x, c = x.to(dev), c.to(dev)
        r1, m1 = firing_rate(d1, x, c)
        r2, _ = firing_rate(d2, x, c)
        bc = best_cos(d1["W_dec"], d2["W_dec"])
        hit = bc > args.cos
        corr, cidx = act_corr_match(d1, d2, x, c)
        chit = corr > args.cos
        w1n = d1["W_dec"] / d1["W_dec"].norm(dim=1, keepdim=True)
        w2n = d2["W_dec"] / d2["W_dec"].norm(dim=1, keepdim=True)
        cos_of_corr_partner = (w1n * w2n[cidx]).sum(1)
        live = r1 > 0
        common = r1 > 1e-4
        top = torch.zeros_like(live)
        top[torch.topk(r1, min(1000, r1.numel())).indices] = True
        q = torch.quantile(bc, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev)).tolist()
        rec = {
            "arm": args.arm, "dict": entry, "n_val": int(x.shape[0]), "width": int(r1.numel()),
            "live_frac_s1": live.float().mean().item(), "live_frac_s2": (r2 > 0).float().mean().item(),
            "common_frac_s1": common.float().mean().item(),
            "matched_all": hit.float().mean().item(),
            "matched_live": hit[live].float().mean().item() if live.any() else float("nan"),
            "matched_common": hit[common].float().mean().item() if common.any() else float("nan"),
            "matched_top1000": hit[top].float().mean().item(),
            "matched_mass_weighted": (hit.float() * m1).sum().item() / max(m1.sum().item(), 1e-9),
            "cos_q10": q[0], "cos_q25": q[1], "cos_q50": q[2], "cos_q75": q[3], "cos_q90": q[4],
            "cos_q50_common": torch.median(bc[common]).item() if common.any() else float("nan"),
            "corr_matched_all": chit.float().mean().item(),
            "corr_matched_live": chit[live].float().mean().item() if live.any() else float("nan"),
            "corr_matched_common": chit[common].float().mean().item() if common.any() else float("nan"),
            "corr_matched_top1000": chit[top].float().mean().item(),
            "corr_matched_mass_weighted": (chit.float() * m1).sum().item() / max(m1.sum().item(), 1e-9),
            "corr_q50_common": torch.median(corr[common]).item() if common.any() else float("nan"),
            "cos_of_corr_partner_q50_common": torch.median(cos_of_corr_partner[common]).item() if common.any() else float("nan"),
        }
        rows.append(rec)
        (root / f"{stem}_s1{sfx}" / "seedmatch.json").write_text(json.dumps(rec, indent=1))
        torch.save({"rate_s1": r1.cpu(), "mass_s1": m1.cpu(), "rate_s2": r2.cpu(), "best_cos": bc.cpu(),
                    "best_corr": corr.cpu(), "corr_partner": cidx.cpu(), "cos_of_corr_partner": cos_of_corr_partner.cpu()},
                   root / f"{stem}_s1{sfx}" / "seedmatch.pt")
        print(f"[seedmatch] {args.arm} {stem}: matched all {rec['matched_all']:.3f} live {rec['matched_live']:.3f} "
              f"common {rec['matched_common']:.3f} top1000 {rec['matched_top1000']:.3f} "
              f"mass {rec['matched_mass_weighted']:.3f} cos q50 {rec['cos_q50']:.2f} (common {rec['cos_q50_common']:.2f}); "
              f"corr: common {rec['corr_matched_common']:.3f} top1000 {rec['corr_matched_top1000']:.3f} "
              f"mass {rec['corr_matched_mass_weighted']:.3f} q50 common {rec['corr_q50_common']:.2f} "
              f"cos of partner {rec['cos_of_corr_partner_q50_common']:.2f}",
              flush=True)
        del x
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows).round(3)
    md = f"# Seed match, {args.arm} (cosine > {args.cos})\n\n" + df.to_markdown(index=False) + "\n"
    (Path(args.out_root) / f"seedmatch_{args.arm}{args.suffix}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
