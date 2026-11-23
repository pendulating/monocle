"""Does the way we present a crop make the disorder words?

The 1st caption battery cut each cell, then made it 5.3 times larger. A
larger picture can invent detail, so the words could come from the upscale
and not from the causal role.

This tool scores the same 4 cases under 4 presentations:

| Arm | The crop that the model reads |
|---|---|
| `x5.3` | the first battery: 1 cell, 5.3 times larger |
| `up448` | 1 cell, 2.7 times larger |
| `native` | 1 cell at its own size on a grey canvas |
| `pad2` | 2 cells of context, 5.3 times larger |

For each arm it reports the share of captions that hold a word of
`LEXICON_V1`, and the neg-against-random gap with a permutation test that
holds the photograph and the row band fixed.

Usage:

    python research/scripts/pres_control_report.py --out outputs/reports/pres_control
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
from research.scripts.disorder_rubric import lexicon_v1          # noqa: E402

PRES_DIR = REPO / "outputs/_monocle/patch_caption_pres"
BASE_DIR = REPO / "outputs/_monocle/patch_caption_reader"
CASES = ["restaurants", "libraries", "parks_plazas", "road_quality"]
ARMS = ["x5.3", "x5.3*", "up448", "native", "pad2"]
N_PERM = 20000
SEED = 20260825


def band(row: int) -> str:
    """The row band of a cell. It holds the height in the frame fixed."""
    return "top" if row <= 5 else ("mid" if row <= 10 else "low")


def load() -> pd.DataFrame:
    """Read every arm of every case into 1 table."""
    frames = []
    for case in CASES:
        # WARNING: the 1st battery (`patch_caption_reader`) was cut from the
        # `reader_cells/{case}_shared` maps. The control arms were cut from
        # `causal_maps60/{case}_absent`. The 2 stores agree at a median r of
        # only 0.164, so a comparison between them mixes the presentation
        # with the map. Take the matched `x53` arm when it exists.
        matched = PRES_DIR / f"{case}_x53" / "captions.parquet"
        base = BASE_DIR / f"{case}_causal" / "captions.parquet"
        if matched.exists():
            d = pd.read_parquet(matched)
            d["pres"] = "x5.3"
            frames.append(d)
        elif base.exists():
            print(f"WARNING {case}: no matched x53 arm; the reader store uses "
                  f"a different map, so this row is not a control.")
            d = pd.read_parquet(base)
            d["pres"] = "x5.3*"
            frames.append(d)
        for arm in ("up448", "native", "pad2"):
            p = PRES_DIR / f"{case}_{arm}" / "captions.parquet"
            if p.exists():
                d = pd.read_parquet(p)
                d["pres"] = arm
                frames.append(d)
    if not frames:
        raise SystemExit(f"no caption store under {PRES_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df["hit"] = df["text"].map(lexicon_v1).astype(float)
    df["band"] = df["row"].astype(int).map(band)
    return df


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

    groups = d.groupby(["image_id", "band"]).indices
    null = np.empty(N_PERM, dtype=float)
    for t in range(N_PERM):
        lab = is_neg.copy()
        for idx in groups.values():
            if idx.size > 1:
                lab[idx] = rng.permutation(lab[idx])
        null[t] = hit[lab].mean() - hit[~lab].mean()
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return {"n": int(len(d)), "observed": obs, "null_median": float(np.median(null)),
            "null_p95": float(np.quantile(null, 0.95)), "p": p}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports/pres_control")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load()
    rng = np.random.default_rng(SEED)

    rows = []
    for pres in ARMS:
        sub = df[df["pres"] == pres]
        if sub.empty:
            continue
        for case in CASES + ["ALL"]:
            s = sub if case == "ALL" else sub[sub["case"] == case]
            if s.empty:
                continue
            rate = s.groupby("arm")["hit"].mean()
            rec = {"pres": pres, "case": case,
                   "images": int(s["image_id"].nunique()),
                   "neg": float(rate.get("neg", np.nan)),
                   "pos": float(rate.get("pos", np.nan)),
                   "random": float(rate.get("random", np.nan))}
            rec["gap"] = rec["neg"] - rec["random"]
            if case == "ALL":
                rec.update({f"perm_{k}": v
                            for k, v in stratified_test(s, rng).items()})
            rows.append(rec)
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "pres_control.csv", index=False)

    print(f"{'arm':7s} {'case':17s} {'neg':>6s} {'pos':>6s} {'rand':>6s} "
          f"{'gap':>7s}")
    for _, r in rep.iterrows():
        star = "  <-" if r["case"] == "ALL" else ""
        print(f"{r['pres']:7s} {r['case']:17s} {r['neg']:6.3f} {r['pos']:6.3f} "
              f"{r['random']:6.3f} {r['gap']:+7.3f}{star}")
    print()
    for _, r in rep[rep["case"] == "ALL"].iterrows():
        if "perm_p" in r and pd.notna(r.get("perm_p")):
            print(f"{r['pres']:7s} gap {r['perm_observed']:+.4f}  "
                  f"null median {r['perm_null_median']:+.4f}  "
                  f"null p95 {r['perm_null_p95']:+.4f}  p={r['perm_p']:.5f}")

    (out / "pres_control.json").write_text(json.dumps(
        {"n_perm": N_PERM, "seed": SEED, "rows": rows}, indent=1))
    print(f"\nwrote {out}/pres_control.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
