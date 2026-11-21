#!/usr/bin/env python
"""F10.29, item 6: how much do the phase rules of F10.26 task 2 depend on their thresholds? CPU only.

The script reads the per-layer tables of task 2 (`figures/f10/prompts/t2_*_by_layer.csv`, `t2_patch_depth.csv`)
and counts, for a grid of thresholds, the cells of 42 in which each rule holds. It also gives the first layer of
each band at each threshold, so that a reader sees how far a band edge moves.

Rules and the thresholds of F10.26 (in `RULES` of `hop_prompts.py`):
- scene code: mean-patch R2 at the best layer at least .50
- gain at `?`: at least .05 at the best layer, Holm p below .05
- verdict band: lens margin r at the last token at least .95 in some layer
- sense band: axis 1 share at least .40 and the cool-list score r at least .95 at the adjective, in some layer
- axis 1 does not follow the verdict: abs r below .20 at mid depth

Output: `figures/f10/robust/sweep.csv` (counts), `figures/f10/robust/band_edges.csv` (first layer of each band by cell and threshold).

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_robust.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "figures/f10/prompts"
OUT = ROOT / "figures/f10/robust"
__version__ = "1.0.0"
MID = {"gemma-4-31b": 28, "qwen3.5-9b": 16, "qwen3.6-27b": 32}


def first_layer(g: pd.DataFrame, ok: pd.Series) -> float:
    """Give the first layer of a cell that passes, or NaN."""
    hit = g.layer[ok.to_numpy()]
    return float(hit.min()) if len(hit) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key = ["model", "adjective", "noun"]
    patch = pd.read_csv(T / "t2_patch_depth.csv")
    gain = pd.read_csv(T / "t2_gain_by_layer.csv")
    verdict = pd.read_csv(T / "t2_verdict_by_layer.csv")
    axis = pd.read_csv(T / "t2_axis_by_layer.csv")
    rows, edges = [], []

    best_r2 = patch.groupby(key)["r2"].max()
    for th in (0.30, 0.40, 0.50, 0.60, 0.70):
        rows.append({"rule": "scene code: best mean-patch R2 at least", "threshold": th, "cells_held": int((best_r2 >= th).sum())})

    qm = gain[gain.position == "qm"]
    for th in (0.02, 0.03, 0.05, 0.08, 0.10):
        held = qm.groupby(key).apply(lambda g: bool(((g.gain >= th) & (g.p_holm < 0.05)).any()), include_groups=False)
        rows.append({"rule": "gain at ? at least (Holm p below .05)", "threshold": th, "cells_held": int(held.sum())})

    for th in (0.85, 0.90, 0.92, 0.95, 0.97, 0.99):
        held = verdict.groupby(key).apply(lambda g: bool((g.r_last_token >= th).any()), include_groups=False)
        rows.append({"rule": "verdict band: r of the lens margin at the last token at least", "threshold": th, "cells_held": int(held.sum())})
        for k, g in verdict.groupby(key):
            edges.append(dict(zip(key, k)) | {"band": "verdict", "threshold": th, "first_layer": first_layer(g, g.r_last_token >= th), "peak": g.r_last_token.max()})

    adj = axis[axis.position == "adjective"]
    for s in (0.30, 0.40, 0.50):
        for r in (0.80, 0.90, 0.95):
            held = adj.groupby(key).apply(lambda g: bool(((g.share1 >= s) & (g.r_axis1_coollist.abs() >= r)).any()), include_groups=False)
            rows.append({"rule": f"sense band: share at least {s} and abs r with the cool lists at least", "threshold": r, "cells_held": int(held.sum())})
            for k, g in adj.groupby(key):
                edges.append(dict(zip(key, k)) | {"band": f"sense (share {s})", "threshold": r, "first_layer": first_layer(g, (g.share1 >= s) & (g.r_axis1_coollist.abs() >= r)), "peak": g.share1.max()})
    held_cool = adj[adj.adjective == "cool"].groupby(key).apply(lambda g: bool(((g.share1 >= 0.40) & (g.r_axis1_coollist.abs() >= 0.95)).any()), include_groups=False)
    rows.append({"rule": "sense band at the F10.26 thresholds, cool cells only (of 6)", "threshold": 0.95, "cells_held": int(held_cool.sum())})

    mid = adj[adj.apply(lambda r: r.layer == MID[r.model], axis=1)]
    for th in (0.10, 0.20, 0.30, 0.40):
        rows.append({"rule": "axis 1 does not follow the verdict at mid depth: abs r below", "threshold": th, "cells_held": int((mid.r_axis1_margin.abs() < th).sum())})
    for th in (0.30, 0.40, 0.50):
        rows.append({"rule": "axis 1 share at mid depth at least", "threshold": th, "cells_held": int((mid.share1 >= th).sum())})
    # the same rule at the best layer of each cell instead of mid depth
    for th in (0.10, 0.20, 0.30):
        held = adj.groupby(key).apply(lambda g: bool((g.r_axis1_margin.abs() < th).all()), include_groups=False)
        rows.append({"rule": "axis 1 does not follow the verdict at any layer: abs r below", "threshold": th, "cells_held": int(held.sum())})

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "sweep.csv", index=False)
    e = pd.DataFrame(edges)
    e.to_csv(OUT / "band_edges.csv", index=False)
    pd.set_option("display.width", 250)
    print(sweep.to_string(index=False))
    v = e[e.band == "verdict"].pivot_table(index=["model", "noun", "adjective"], columns="threshold", values="first_layer")
    print("\nfirst layer of the verdict band by threshold\n", v.to_string())
    print("\nmedian move of the verdict first layer from .95 to .90 (saved layers are 4 apart):", (v[0.95] - v[0.90]).abs().median(), "cells with a band at .90 but not .95:", int(v[0.90].notna().sum() - v[0.95].notna().sum()))


if __name__ == "__main__":
    main()
