"""An instrument that reads a RANGE of layers, not one layer.

The motive
----------
The attention knockout says the patches feed the token stream below block 30,
and most of it goes in blocks 22 to 29. A map that reads ONE layer sees only
what the patch still holds at that layer. A patch that already exported its
content reads as empty, and a patch that exports later reads as full.

So we sum the attribution over the export window. The sum asks a different
question: how much did this patch carry across the whole window?

Usage:
    python research/scripts/layer_sum_ranking.py --maps <gate maps.parquet> \
        --arm gxa --layers 22 24 26 28 --out <ranking.parquet>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "scripts"))

from make_ranking import pool  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--normalise", action="store_true",
                    help="scale each layer to unit spread before the sum, so "
                         "1 loud layer cannot own the total")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    df = pd.read_parquet(a.maps)
    if a.arm not in df.columns:
        raise SystemExit(f"{a.maps} holds no {a.arm!r}")
    want = sorted(set(a.layers))
    have = sorted(set(int(x) for x in df.layer.unique()))
    missing = [l for l in want if l not in have]
    if missing:
        raise SystemExit(f"layers {missing} are absent; have {have}")

    acc: dict[str, np.ndarray] = {}
    used: dict[str, int] = {}
    for layer in want:
        for _, r in df[df.layer == layer].iterrows():
            v = r[a.arm]
            if v is None or np.ndim(v) == 0:
                continue
            m = np.asarray(v, dtype=float).ravel()
            if a.normalise:
                sd = m.std()
                if sd == 0:
                    continue
                m = m / sd
            key = str(r.target)
            acc[key] = m if key not in acc else acc[key] + m
            used[key] = used.get(key, 0) + 1
    if not acc:
        raise SystemExit(f"{a.arm} is empty over {want}")
    n_layers = max(used.values())
    keep = [k for k, v in used.items() if v == n_layers]
    rows = [{"image_id": k, "values": pool(acc[k], a.group).tolist()}
            for k in sorted(keep)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out}: {len(rows)} images, {len(rows[0]['values'])} cells, "
          f"arm={a.arm} layers={want} normalise={a.normalise}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
