"""Stage-2 gates over the trained dictionaries.

G1, per dictionary: FVU (threshold encoder) below 0.10, L0 between 0.5k and
1.5k, dead latents below 5 percent over the last 500 training steps (2 million
vectors; the validation set is too small to measure a rare feature).
G2, per (arm, dict, width) seed pair: the fraction of seed-1 features whose
best seed-2 match has decoder cosine above 0.7 must be at least 0.60. The
matched pairs are written as `match_s1_s2.parquet` beside seed 1.

    .venv-vllm025cu129/bin/python research/scripts/sae_gates.py --root outputs/_monocle/sae
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

PAT = re.compile(r"^(?P<dict>[PA]\d+)_w(?P<width>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)(?:_(?P<tag>\w+))?$")


def g1(m: dict) -> tuple[bool, str]:
    k = m["k"]
    checks = {
        "fvu<0.10": m["fvu_thr"] < 0.10,
        "l0 in [0.5k,1.5k]": 0.5 * k <= m["l0_thr"] <= 1.5 * k,
        "dead<0.05 (train window)": m["dead_frac_train_window"] < 0.05,
    }
    return all(checks.values()), ", ".join(f"{n}:{'ok' if v else 'FAIL'}" for n, v in checks.items())


@torch.no_grad()
def seed_match(d1: Path, d2: Path, thr: float, chunk: int = 4096) -> tuple[pd.DataFrame, float]:
    w1 = torch.load(d1 / "dict.pt", map_location="cpu", weights_only=False)["W_dec"].float()
    w2 = torch.load(d2 / "dict.pt", map_location="cpu", weights_only=False)["W_dec"].float()
    w1 = w1 / w1.norm(dim=1, keepdim=True)
    w2 = w2 / w2.norm(dim=1, keepdim=True)
    best, arg = [], []
    for i in range(0, w1.shape[0], chunk):
        c = w1[i : i + chunk] @ w2.t()
        b, a = c.max(dim=1)
        best.append(b)
        arg.append(a)
    best = torch.cat(best)
    arg = torch.cat(arg)
    df = pd.DataFrame({"f1": range(w1.shape[0]), "f2": arg.numpy(), "cos": best.numpy()})
    return df[df.cos > thr].reset_index(drop=True), (best > thr).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/_monocle/sae")
    ap.add_argument("--cos", type=float, default=0.7)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for arm in ("it", "base"):
        for d in sorted((root / arm).glob("*_w*_k*_s*")):
            m = PAT.match(d.name)
            if not m or not (d / "metrics.json").exists():
                continue
            met = json.loads((d / "metrics.json").read_text())
            ok, detail = g1(met)
            gd = m.groupdict()
            gd["tag"] = gd.get("tag") or "-"
            rows.append({"arm": arm, **{k: int(v) if k not in ("dict", "tag") else v for k, v in gd.items()},
                         "fvu": round(met["fvu_thr"], 3), "l0": round(met["l0_thr"], 1),
                         "dead_train": round(met["dead_frac_train_window"], 3), "dead_val": round(met["dead_frac_thr"], 3), "min": round(met["train_minutes"], 1),
                         "G1": "pass" if ok else "FAIL", "detail": detail, "dir": d})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no trained dictionaries found")
    print("\n## G1\n")
    print(df.drop(columns=["dir"]).to_markdown(index=False))
    print("\n## G2\n")
    g2 = []
    for (arm, dname, width, tag), g in df.groupby(["arm", "dict", "width", "tag"]):
        seeds = dict(zip(g.seed, g.dir))
        if 1 not in seeds or 2 not in seeds:
            continue
        match, frac = seed_match(seeds[1], seeds[2], args.cos)
        match.to_parquet(seeds[1] / "match_s1_s2.parquet", index=False)
        g2.append({"arm": arm, "dict": dname, "width": width, "tag": tag, "matched_frac": round(frac, 3),
                   "n_matched": len(match), "G2": "pass" if frac >= 0.6 else "FAIL"})
    g2 = pd.DataFrame(g2)
    print(g2.to_markdown(index=False) if not g2.empty else "no seed pairs yet")
    (root / "gates.md").write_text("## G1\n\n" + df.drop(columns=["dir"]).to_markdown(index=False)
                                   + "\n\n## G2\n\n" + (g2.to_markdown(index=False) if not g2.empty else "none") + "\n")
    print(f"\nwrote {root / 'gates.md'}")


if __name__ == "__main__":
    main()
