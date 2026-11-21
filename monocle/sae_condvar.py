"""How much of the residual variance is the frame (answer) or the position (patches)?

On the validation vectors of each dictionary: the share of the total
variance (about the pooled mean) that the conditional means explain, where
the condition is the frame for A<L> and the patch position for P<L>. A
large share means the dictionary spends its latents on the condition, not
on the image, and the recipe should center per condition.

    sbatch --time=00:20:00 monocle/monocle.sub monocle.sae_condvar --arm it

Writes `<out-root>/condvar_<arm>.md`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from monocle.sae_train import D, SAMPLE_DEFAULT, load_patch_part, patch_parts, train_image_set  # noqa: E402


def share(x: torch.Tensor, cond: torch.Tensor) -> tuple[float, int]:
    """(variance share of the conditional means, number of conditions)."""
    x = x.float()
    mu = x.mean(0)
    total = ((x - mu) ** 2).sum()
    between = torch.zeros((), device=x.device)
    for c in cond.unique():
        m = cond == c
        between += m.sum() * ((x[m].mean(0) - mu) ** 2).sum()
    return (between / total).item(), int(cond.unique().numel())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["it", "base"])
    ap.add_argument("--data-root", default="/scratch/mwf62/sae")
    ap.add_argument("--out-root", default=str(REPO / "outputs/_monocle/sae"))
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    data = Path(args.data_root) / args.arm
    _, val_keep = train_image_set(SAMPLE_DEFAULT)
    rows = []
    for dname in ("A42", "A46"):
        xs, cs = [], []
        for i, frame in enumerate(sorted(p for p in (data / dname).iterdir() if p.is_dir())):
            for f in sorted(frame.glob("shard*.pt")):
                o = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                mask = torch.tensor([p in val_keep for p in o["image_path"]])
                if mask.any():
                    xs.append(o["acts"][mask])
                    cs.append(torch.full((int(mask.sum()),), i))
        x = torch.cat(xs).to(dev)
        c = torch.cat(cs).to(dev)
        s, k = share(x, c)
        rows.append({"dict": dname, "condition": "frame", "n_conditions": k, "n_val": int(x.shape[0]),
                     "variance_share": round(s, 3)})
        print(f"[condvar] {args.arm} {dname}: frame means explain {s:.3f} of the variance ({k} frames)", flush=True)
        del x
    for dname in ("P22", "P40"):
        _, val_parts = patch_parts(data, dname, 2)
        x = torch.cat([load_patch_part(f) for f in val_parts]).to(dev)
        c = torch.arange(256, device=dev).repeat(x.shape[0] // 256)
        s, k = share(x, c)
        last = torch.zeros_like(c)
        last[255::256] = 1
        s_last, _ = share(x, last)
        rows.append({"dict": dname, "condition": "position", "n_conditions": k, "n_val": int(x.shape[0]),
                     "variance_share": round(s, 3)})
        rows.append({"dict": dname, "condition": "last patch vs rest", "n_conditions": 2, "n_val": int(x.shape[0]),
                     "variance_share": round(s_last, 3)})
        print(f"[condvar] {args.arm} {dname}: position means explain {s:.3f}; the last patch alone {s_last:.3f}", flush=True)
        del x
    df = pd.DataFrame(rows)
    md = f"# Conditional variance share, {args.arm}\n\n" + df.to_markdown(index=False) + "\n"
    (Path(args.out_root) / f"condvar_{args.arm}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
