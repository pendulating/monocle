"""Stage-1 check: count the shard files on klara, read the shapes, and sum the sizes.

Run on klara (the shards are node-local):
    sbatch research/monocle/monocle.sub monocle.sae_check --root /scratch/mwf62/sae --out outputs/_monocle/sae/extract_check.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def check_dir(d: Path) -> dict:
    files = sorted(d.glob("*.pt"))
    n_rows = 0
    n_bytes = 0
    shapes = set()
    dtypes = set()
    paths = []
    bad = []
    for f in files:
        n_bytes += f.stat().st_size
        try:
            o = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        except Exception as e:  # a truncated file
            bad.append(f"{f.name}: {e}")
            continue
        a = o["acts"]
        n_rows += a.shape[0]
        shapes.add(tuple(a.shape[1:]))
        dtypes.add(str(a.dtype))
        paths.extend(o["image_path"])
    return {
        "files": len(files),
        "rows": n_rows,
        "unique_images": len(set(paths)),
        "row_shape": sorted(str(s) for s in shapes),
        "dtype": sorted(dtypes),
        "gib": round(n_bytes / 2**30, 1),
        "bad": bad,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/mwf62/sae")
    ap.add_argument("--out", default="outputs/_monocle/sae/extract_check.json")
    args = ap.parse_args()
    root = Path(args.root)
    out = {}
    for arm in ("it", "base"):
        arm_out = {}
        for d in sorted((root / arm).glob("P*")):
            arm_out[d.name] = check_dir(d)
        for d in sorted((root / arm).glob("A*")):
            for frame in sorted(p for p in d.iterdir() if p.is_dir()):
                arm_out[f"{d.name}/{frame.name}"] = check_dir(frame)
        out[arm] = arm_out
        for k, v in arm_out.items():
            print(f"[sae-check] {arm:4s} {k:24s} files={v['files']:3d} rows={v['rows']:7d} "
                  f"images={v['unique_images']:6d} shape={v['row_shape']} {v['gib']} GiB bad={len(v['bad'])}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[sae-check] wrote {args.out}")


if __name__ == "__main__":
    main()
