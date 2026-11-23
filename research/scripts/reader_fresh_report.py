"""Does the fitted reader's pole hold up on photographs it never saw?

The question
------------
On the first battery the reader's PREDICTED negative pole drew disorder
words about 3 times more often than the MEASURED negative pole. Two
readings fit that fact:

| Reading | What it says |
|---|---|
| the reader is more correct | the 256 ablations are noisy, and the fit cleans them |
| the reader follows content | the fit learns "disorder looks like this" and ignores the cause |

A fresh sample separates the 2 only in part, but it settles the first
worry: the reader's advantage could be an artifact of the photographs
that the first battery used. This tool scores the reader's pole on the
seed-20260827 photographs, which no earlier battery read.

What it reads
-------------
| Store | The map the crops came from |
|---|---|
| `patch_caption_reader_fresh` | the fitted reader, `shared` mode, fresh photographs |
| `patch_caption_fresh` | the measured ablation map, the SAME photographs |

The 2 stores share the photographs, the crop size, the lexicon and the
random arm, so a paired test compares the maps and nothing else.

The 3 numbers
-------------
1. the neg-against-random gap of each map, with a null that holds the
   photograph and the row band fixed
2. the paired difference of the 2 gaps, with a null that permutes only
   the MAP label inside 1 photograph and 1 arm
3. where the reader's pole cell sits in the measured map

Usage:

    python research/scripts/reader_fresh_report.py --out outputs/reports/reader_fresh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "scripts"))
from research.scripts.disorder_rubric import lexicon_v1          # noqa: E402

CASES = ["restaurants", "libraries", "subway_safety", "schools",
         "street_photography", "parks_plazas", "road_quality"]
STORES = {
    "reader": (REPO / "outputs/_monocle/patch_caption_reader_fresh",
               "{case}_readerfresh"),
    "measured": (REPO / "outputs/_monocle/patch_caption_fresh",
                 "{case}_fresh"),
}
MAPS = {
    "reader": REPO / "outputs/_monocle/reader_cells_fresh/{case}_shared",
    "measured": REPO / "outputs/_monocle/causal_maps60_fresh/{case}_absent",
}
N_PERM = 20000
SEED = 20260826


def band(row: int) -> str:
    """The row band of a cell. It holds the height in the frame fixed."""
    return "top" if row <= 5 else ("mid" if row <= 10 else "low")


def load() -> pd.DataFrame:
    """Every crop of both maps in 1 table."""
    frames = []
    for name, (root, pat) in STORES.items():
        for case in CASES:
            p = root / pat.format(case=case) / "captions.parquet"
            if not p.exists():
                print(f"missing {p}", flush=True)
                continue
            d = pd.read_parquet(p)
            d["map"] = name
            frames.append(d)
    if not frames:
        raise SystemExit("no caption store to read")
    df = pd.concat(frames, ignore_index=True)
    df["hit"] = df["text"].map(lexicon_v1).astype(float)
    df["band"] = df["row"].astype(int).map(band)
    return df


def gap_of(d: pd.DataFrame) -> float:
    """The neg rate less the random rate."""
    neg = d.loc[d["arm"] == "neg", "hit"]
    rnd = d.loc[d["arm"] == "random", "hit"]
    if neg.empty or rnd.empty:
        return float("nan")
    return float(neg.mean() - rnd.mean())


def stratified_test(d: pd.DataFrame, rng: np.random.Generator) -> dict:
    """The neg-against-random gap, with a null that holds photograph and band.

    A negative cell sits lower in the frame than a random cell, so the words
    could come from the height. The permutation moves the arm label only
    inside 1 photograph AND 1 row band, so the height cannot make the gap.
    """
    d = d[d["arm"].isin(["neg", "random"])].copy()
    if d.empty:
        return {"n": 0}
    is_neg = (d["arm"] == "neg").to_numpy()
    hit = d["hit"].to_numpy(dtype=float)
    obs = float(hit[is_neg].mean() - hit[~is_neg].mean())
    groups = d.groupby(["case", "image_id", "band"]).indices
    null = np.empty(N_PERM, dtype=float)
    for t in range(N_PERM):
        lab = is_neg.copy()
        for idx in groups.values():
            if idx.size > 1:
                lab[idx] = rng.permutation(lab[idx])
        null[t] = hit[lab].mean() - hit[~lab].mean()
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return {"n": int(len(d)), "observed": obs,
            "null_median": float(np.median(null)),
            "null_p95": float(np.quantile(null, 0.95)), "p": p}


def paired_map_test(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """The reader gap less the measured gap, on the same photographs.

    Warning: the key of the permutation decides whether the test works.
    A key on the CELL fails, because the 2 maps rarely pick the same cell
    and most groups then hold 1 row. The key here is the photograph and
    the arm, which both maps always fill.
    """
    d = df[df["arm"].isin(["neg", "random"])].copy()
    is_reader = (d["map"] == "reader").to_numpy()
    is_neg = (d["arm"] == "neg").to_numpy()
    hit = d["hit"].to_numpy(dtype=float)

    def diff(lab: np.ndarray) -> float:
        a = hit[lab & is_neg].mean() - hit[lab & ~is_neg].mean()
        b = hit[~lab & is_neg].mean() - hit[~lab & ~is_neg].mean()
        return float(a - b)

    obs = diff(is_reader)
    groups = d.groupby(["case", "image_id", "arm"]).indices
    null = np.empty(N_PERM, dtype=float)
    for t in range(N_PERM):
        lab = is_reader.copy()
        for idx in groups.values():
            if idx.size > 1:
                lab[idx] = rng.permutation(lab[idx])
        null[t] = diff(lab)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))
    return {"n": int(len(d)), "observed": obs,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_two_sided": p}


def pole_agreement() -> pd.DataFrame:
    """Where the reader's pole cell sits in the measured map.

    The rank runs from 0, the measured minimum, to 1, the measured
    maximum. A reader that reads the cause puts its pole near 0. A reader
    that reads the picture puts it anywhere.
    """
    rows = []
    for case in CASES:
        rp = MAPS["reader"].as_posix().format(case=case)
        mp = MAPS["measured"].as_posix().format(case=case)
        rf, mf = Path(rp) / "maps.parquet", Path(mp) / "maps.parquet"
        if not rf.exists() or not mf.exists():
            continue
        rd = pd.read_parquet(rf)
        md = pd.read_parquet(mf)
        mcol = "values" if "values" in md.columns else "causal"
        mm = {str(i): np.asarray(v, dtype=float)
              for i, v in zip(md.image_id, md[mcol])}
        for img, vals in zip(rd.image_id, rd["causal"]):
            img = str(img)
            if img not in mm:
                continue
            r = np.asarray(vals, dtype=float)
            m = mm[img]
            if r.size != m.size:
                continue
            r_neg, m_neg = int(np.argmin(r)), int(np.argmin(m))
            order = np.argsort(np.argsort(m))     # 0 is the measured minimum
            rows.append({
                "case": case, "image_id": img,
                "same_neg_cell": float(r_neg == m_neg),
                "reader_neg_rank": float(order[r_neg]) / (m.size - 1),
                "rho": float(spearmanr(r, m).statistic),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports/reader_fresh")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load()
    rng = np.random.default_rng(SEED)

    rows = []
    for name in STORES:
        sub = df[df["map"] == name]
        if sub.empty:
            continue
        for case in CASES + ["ALL"]:
            s = sub if case == "ALL" else sub[sub["case"] == case]
            if s.empty:
                continue
            rate = s.groupby("arm")["hit"].mean()
            rec = {"map": name, "case": case,
                   "images": int(s["image_id"].nunique()),
                   "crops": int(len(s)),
                   "neg": float(rate.get("neg", np.nan)),
                   "pos": float(rate.get("pos", np.nan)),
                   "random": float(rate.get("random", np.nan))}
            rec["gap"] = rec["neg"] - rec["random"]
            if case == "ALL":
                rec.update({f"perm_{k}": v
                            for k, v in stratified_test(s, rng).items()})
            rows.append(rec)
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "reader_fresh.csv", index=False)

    print(f"{'map':9s} {'case':19s} {'imgs':>4s} {'neg':>6s} {'pos':>6s} "
          f"{'rand':>6s} {'gap':>7s}")
    for _, r in rep.iterrows():
        star = "  <-" if r["case"] == "ALL" else ""
        print(f"{r['map']:9s} {r['case']:19s} {r['images']:4d} "
              f"{r['neg']:6.3f} {r['pos']:6.3f} {r['random']:6.3f} "
              f"{r['gap']:+7.3f}{star}")
    print()
    for _, r in rep[rep["case"] == "ALL"].iterrows():
        if pd.notna(r.get("perm_p")):
            print(f"{r['map']:9s} gap {r['perm_observed']:+.4f}  "
                  f"null median {r['perm_null_median']:+.4f}  "
                  f"null p95 {r['perm_null_p95']:+.4f}  p={r['perm_p']:.5f}")

    paired = {}
    if df["map"].nunique() == 2:
        paired = paired_map_test(df, rng)
        print(f"\npaired: reader less measured {paired['observed']:+.4f}  "
              f"null {paired['null_mean']:+.4f} +/- {paired['null_sd']:.4f}  "
              f"p={paired['p_two_sided']:.5f}")

    pol = pole_agreement()
    pole_summary = {}
    if not pol.empty:
        pol.to_csv(out / "pole_agreement.csv", index=False)
        print(f"\n{'case':19s} {'n':>4s} {'same cell':>9s} "
              f"{'reader neg rank':>15s} {'median rho':>10s}")
        for case in CASES + ["ALL"]:
            s = pol if case == "ALL" else pol[pol["case"] == case]
            if s.empty:
                continue
            print(f"{case:19s} {len(s):4d} {s['same_neg_cell'].mean():9.3f} "
                  f"{s['reader_neg_rank'].mean():15.3f} "
                  f"{s['rho'].median():10.3f}")
            pole_summary[case] = {
                "n": int(len(s)),
                "same_neg_cell": float(s["same_neg_cell"].mean()),
                "reader_neg_rank": float(s["reader_neg_rank"].mean()),
                "median_rho": float(s["rho"].median()),
            }

    (out / "reader_fresh.json").write_text(json.dumps({
        "n_perm": N_PERM, "seed": SEED, "rows": rows,
        "paired": paired, "pole": pole_summary}, indent=1))
    print(f"\nwrote {out}/reader_fresh.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
