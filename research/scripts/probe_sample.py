"""Draw the images of the lens-refusal probe.

Each probe draws 600 images: 10 deciles of its tract proxy, 60 tracts in
each decile, 1 image in each tract. A probe whose proxy is not on disk
(race and migration before the ACS pull) stratifies on its fallback
proxy, income, and the file says so. The 5 draws go into 1 union table,
so every probe scores every image and E5 reads 2 probes on the same
image at no extra cost.

Reads `outputs/_monocle/probe/data/{catalog_tracts,tract_proxies}.parquet`.
Writes `outputs/_monocle/probe/sample/images.parquet` with 1 row per image:
image_path, sample_id, geoid, boroct2020, face, the proxies, and for each
probe `draw_<probe>` (bool) and `decile_<probe>` (0 to 9, or -1).

    .venv-3.12/bin/python research/scripts/probe_sample.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
from research.monocle.probe_frame import PROBE_ORDER, PROBES  # noqa: E402

DATA = REPO / "outputs/_monocle/probe/data"
OUT = REPO / "outputs/_monocle/probe/sample"
PROXY_COLS = ["median_income", "crime_density", "crime_per_capita", "homeless_density",
              "black_share", "white_share", "foreign_born_share",
              "population", "area_km2"]


def log(m: str) -> None:
    print(f"[probe-sample] {m}", flush=True)


def draw_probe(name: str, tracts: pd.DataFrame, cat: pd.DataFrame,
               per_decile: int, seed: int, min_tracts: int) -> tuple[pd.DataFrame, str]:
    p = PROBES[name]
    col = p.proxy
    ok = tracts[col].notna().sum() if col in tracts else 0
    if ok < min_tracts:
        log(f"{name}: proxy {col!r} has {ok} tracts; stratify on "
            f"{p.fallback_proxy!r} instead")
        col = p.fallback_proxy
    t = tracts.dropna(subset=[col])
    t = t[t["geoid"].isin(cat["geoid"])]
    t = t.assign(decile=pd.qcut(t[col].rank(method="first"), 10, labels=False))
    rng = np.random.default_rng(seed)
    picked = []
    for d, g in t.groupby("decile"):
        k = min(per_decile, len(g))
        idx = rng.choice(len(g), size=k, replace=False)
        picked.append(g.iloc[idx])
    tp = pd.concat(picked)
    rows = []
    for _, r in tp.iterrows():
        imgs = cat[cat["geoid"] == r["geoid"]]
        row = imgs.iloc[int(rng.integers(len(imgs)))]
        rows.append({"image_path": row["image_path"], "sample_id": row["sample_id"],
                     "geoid": r["geoid"], f"decile_{name}": int(r["decile"])})
    df = pd.DataFrame(rows)
    log(f"{name}: {len(df)} images over {df['geoid'].nunique()} tracts, "
        f"strata on {col!r}")
    return df, col


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-decile", type=int, default=60)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--min-tracts", type=int, default=1000)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    tracts = pd.read_parquet(DATA / "tract_proxies.parquet")
    cat = pd.read_parquet(DATA / "catalog_tracts.parquet")
    log(f"{len(tracts):,} tracts, {len(cat):,} images")

    parts, strata = [], {}
    for i, name in enumerate(PROBE_ORDER):
        df, col = draw_probe(name, tracts, cat, args.per_decile,
                             args.seed + i, args.min_tracts)
        df[f"draw_{name}"] = True
        parts.append(df)
        strata[name] = col

    union = parts[0]
    for df in parts[1:]:
        union = union.merge(df, on=["image_path", "sample_id", "geoid"], how="outer")
    for name in PROBE_ORDER:
        union[f"draw_{name}"] = union[f"draw_{name}"].fillna(False).astype(bool)
        union[f"decile_{name}"] = union[f"decile_{name}"].fillna(-1).astype(int)
    union = union.merge(cat[["image_path", "boroct2020", "face", "recording_id"]],
                        on="image_path", how="left")
    union = union.merge(tracts[["geoid"] + PROXY_COLS], on="geoid", how="left")
    union = union.sort_values("image_path").reset_index(drop=True)
    union.to_parquet(OUT / "images.parquet", index=False)
    summ = {"images": int(len(union)), "tracts": int(union["geoid"].nunique()),
            "per_probe": {n: int(union[f"draw_{n}"].sum()) for n in PROBE_ORDER},
            "strata_proxy": strata, "seed": args.seed}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
