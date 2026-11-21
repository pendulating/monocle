"""Put the measured causal value back into a caption store.

`transfer_k1.load_maps` had a defect between 2026-08-25 20:45 and 22:00. It
renamed `pick` from `causal` to `values`, and a later test on the NEW name
then sent a measured map down the layer path. That path returns RANKS.

A job in that window chose the right cells, because a rank keeps the order,
so the captions are correct. But it wrote a rank of 1 to 256 into the `value`
column, not the measured change in the judgment logit.

This tool finds a caption store with rank values and writes the measured
values again from the map store. It does not touch the text.

Usage:

    python scripts/repair_caption_values.py --dry-run
    python scripts/repair_caption_values.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Each caption store names the map store it was cut from.
MAP_ROOTS = [
    REPO / "outputs/_monocle/causal_maps60",
    REPO / "outputs/_monocle/causal_maps60_new",
    REPO / "outputs/_monocle/causal_maps60_fresh",
    REPO / "outputs/_monocle/xq30",
    # The reader battery cuts from the fitted reader's PREDICTED map, which
    # is not a measured map at all. It must be a candidate, or the tool
    # scores that store against a map it never used.
    REPO / "outputs/_monocle/reader_cells",
    REPO / "outputs/_monocle/reader_cells_fresh",
]
CAPTION_ROOTS = [
    REPO / "outputs/_monocle/patch_caption",
    REPO / "outputs/_monocle/patch_caption_reader",
    REPO / "outputs/_monocle/patch_caption_pres",
    REPO / "outputs/_monocle/patch_caption_dense",
    REPO / "outputs/_monocle/patch_caption_fresh",
    REPO / "outputs/_monocle/patch_caption_reader_fresh",
]


def looks_like_ranks(v: np.ndarray, n_cells: int = 256) -> bool:
    """True when the values of 1 photograph are a rank of 1 to n_cells.

    Args:
        v: The `value` column of 1 photograph.
        n_cells: Cells in the map.
    """
    if v.size == 0:
        return False
    integral = np.allclose(v, np.round(v))
    in_range = float(v.min()) >= 1.0 and float(v.max()) <= float(n_cells)
    return bool(integral and in_range)


def map_candidates(case: str) -> list[Path]:
    """Every map store that can belong to a case, best first.

    A case has more than 1 map store, and they hold different photographs.
    So the caller must take the store that covers the images it needs, not
    the newest one.
    """
    hits: list[Path] = []
    for root in MAP_ROOTS:                      # `_absent` is the standard arm
        hits += list(root.glob(f"{case}_absent/maps.parquet"))
    for root in MAP_ROOTS:
        hits += [p for p in root.glob(f"{case}_*/maps.parquet")
                 if p not in hits]
        hits += [p for p in root.glob(f"{case}/maps.parquet") if p not in hits]
    return hits


def repair(store: Path, apply: bool) -> dict:
    df = pd.read_parquet(store)
    if "value" not in df.columns or "cell" not in df.columns:
        return {"store": str(store), "state": "no value/cell column"}
    case = str(df["case"].iloc[0])
    want = set(df.image_id.astype(str))
    mp, truth, best_rho = None, None, -2.0
    tried = []
    # A case owns more than 1 map store, and several cover the same images.
    # Covering is thus NOT enough to name the source. Score EVERY covering
    # candidate by the rank order it implies, and keep the best. A store cut
    # from the reader's predicted map would otherwise be scored against a
    # measured map that it never used.
    for cand in map_candidates(case):
        maps = pd.read_parquet(cand)
        col = "values" if "values" in maps.columns else "causal"
        got = {str(r["image_id"]): np.asarray(r[col], dtype=float).ravel()
               for _, r in maps.iterrows()}
        if want - set(got):
            tried.append(f"{cand.parent.name}({len(want - set(got))} missing)")
            continue
        rs = []
        for key, g in df.groupby("image_id"):
            a = g["value"].to_numpy(dtype=float)
            b = np.array([got[str(key)][int(c)] for c in g["cell"]])
            if a.size > 2 and np.std(a) > 0 and np.std(b) > 0:
                rs.append(float(np.corrcoef(np.argsort(np.argsort(a)),
                                            np.argsort(np.argsort(b)))[0, 1]))
        r = float(np.mean(rs)) if rs else -1.0
        tried.append(f"{cand.parent.name}(rho={r:.3f})")
        if r > best_rho:
            mp, truth, best_rho = cand, got, r
    if truth is None:
        return {"store": str(store), "state": "no map covers the images",
                "case": case, "tried": tried}

    # The decision is exact: does the stored value ALREADY equal the measured
    # value? A guess from the shape of the numbers is not enough. The old
    # guess asked every photograph to look rank-like and let a whole store
    # through when 1 did not.
    new_now = np.array([truth[str(i)][int(c)]
                        for i, c in zip(df.image_id, df.cell)], dtype=float)
    if np.allclose(new_now, df["value"].to_numpy(dtype=float)):
        return {"store": str(store), "state": "ok",
                "images": int(df.image_id.nunique()), "map": str(mp)}

    new = np.array([truth[str(i)][int(c)]
                    for i, c in zip(df.image_id, df.cell)], dtype=float)
    old = df["value"].to_numpy(dtype=float)
    # The order must agree, or the join is wrong.
    checks = []
    for key, g in df.groupby("image_id"):
        a = g["value"].to_numpy(dtype=float)
        b = np.array([truth[str(key)][int(c)] for c in g["cell"]])
        if a.size > 2:
            checks.append(float(np.corrcoef(np.argsort(np.argsort(a)),
                                            np.argsort(np.argsort(b)))[0, 1]))
    rho = float(np.mean(checks)) if checks else float("nan")
    out = {"store": str(store), "state": "RANKS -> repaired" if apply
           else "RANKS (dry run)", "case": case, "map": str(mp),
           "rows": int(len(df)), "images": int(df.image_id.nunique()),
           "rank_agreement": rho,
           "old_range": [float(old.min()), float(old.max())],
           "new_range": [float(new.min()), float(new.max())]}
    if rho < 0.999:
        # The order must match, or this is not the map the store was cut
        # from. Say so plainly: an unknown source is NOT a rank defect, and
        # calling it one sends a reader after a bug that is not there.
        out["state"] = "SKIP: no candidate map matches this store"
        return out
    if apply:
        df["value"] = new
        df.to_parquet(store, index=False)
        (store.parent / "value_repair.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply = args.apply and not args.dry_run

    stores = []
    for root in CAPTION_ROOTS:
        stores += sorted(root.glob("*/captions.parquet"))
    if not stores:
        print("no caption store found")
        return 1
    n_bad = 0
    for s in stores:
        r = repair(s, apply)
        tag = r["state"]
        if "RANKS" in tag:
            n_bad += 1
        print(f"{tag:34s} {s.parent.name}"
              + (f"  rho={r.get('rank_agreement'):.4f}"
                 if r.get("rank_agreement") is not None else ""))
    print(f"\n{len(stores)} stores, {n_bad} with rank values"
          + ("" if apply else "  (nothing written; pass --apply)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
