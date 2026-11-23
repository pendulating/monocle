"""Draw the images of round 2 of the lens-refusal probe: NTA extremes.

Round 1 stratified each probe on 10 deciles of a tract proxy. Round 2
takes the 10 neighborhoods (NTA 2020) with the HIGHEST value of the
probe's proxy and the 10 with the LOWEST, and draws `--per-nta` images in
each, spread over the NTA's tracts. Each probe then holds 2 groups of
about 600 images, and E3 becomes a two-group test
(`research/scripts/probe_group_test.py`).

An NTA enters the pool when it is residential (`NTAType == "0"`), holds
more than `--min-pop` people, and holds at least `--per-nta` catalog
images. The draws of all probes go into 1 union table, so every probe
scores every image.

Reads `outputs/_monocle/probe_nta/data/nta_proxies.parquet`,
`outputs/_monocle/probe/data/catalog_tracts.parquet`, and the 2020 tract
file for the tract-to-NTA map. Writes
`outputs/_monocle/probe_nta/sample/images.parquet` with 1 row per image:
image_path, sample_id, geoid, boroct2020, nta2020, ntaname, boroname,
face, recording_id, the NTA proxies, and for each probe `draw_<probe>`
(bool), `group_<probe>` ("top", "bottom", or ""), and `proxy_<probe>`
(the NTA value).

    .venv-vllm025cu129/bin/python research/scripts/probe_sample_nta.py

The density-matched redraw of round 3 (crime, gang, housing) pairs each
high NTA with a low NTA of near log population density:

    .venv-vllm025cu129/bin/python research/scripts/probe_sample_nta.py \
        --probes crime gang housing --match-density 0.3 \
        --out outputs/_monocle/probe_words_dm/sample
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
from research.monocle.probe_frame import PROBE_ORDER, PROBES  # noqa: E402

DATA_NTA = REPO / "outputs/_monocle/probe_nta/data"
DATA_R1 = REPO / "outputs/_monocle/probe/data"
TRACTS_GEO = REPO / "data/geo/2020_Census_Tracts_20260304.geojson"
OUT = REPO / "outputs/_monocle/probe_nta/sample"
NTA_COLS = ["nta2020", "ntaname", "boroname", "population", "area_km2", "pop_density",
            "homeless_per_capita",
            "median_income", "black_share", "white_share", "foreign_born_share",
            "bachelors_share", "median_home_value", "homeless_density", "crime_density",
            "crime_per_capita", "prostitution_density", "violent_per_capita"]


def log(m: str) -> None:
    print(f"[probe-sample-nta] {m}", flush=True)


def tract_to_nta() -> pd.DataFrame:
    import geopandas as gpd
    g = gpd.read_file(TRACTS_GEO, ignore_geometry=True)
    return g[["boroct2020", "nta2020"]].drop_duplicates("boroct2020")


def pick_extremes(nta: pd.DataFrame, col: str, k: int,
                  rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The k highest and the k lowest NTAs. A tie at the boundary (76 NTAs
    hold 0 prostitution complaints) breaks at random under the seed."""
    pool = nta.dropna(subset=[col]).assign(_tie=rng.random(len(nta.dropna(subset=[col]))))
    pool = pool.sort_values([col, "_tie"]).drop(columns="_tie")
    return pool.tail(k), pool.head(k)


def pick_matched(nta: pd.DataFrame, col: str, k: int, caliper: float, top_pool: int,
                 rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`k` pairs of NTAs, 1 from the `top_pool` highest on `col` and 1 from the
    `top_pool` lowest, matched on log population density within `caliper`.
    The picker walks the high end from the extreme inward and takes the
    nearest unused low NTA inside the caliper, so the proxy contrast stays
    as large as the match allows. Ties break at random under the seed."""
    pool = nta.dropna(subset=[col]).assign(_tie=rng.random(nta[col].notna().sum()))
    pool = pool.sort_values([col, "_tie"]).drop(columns="_tie")
    pool = pool[pool["pop_density"] > 0].assign(ld_=lambda d: np.log(d["pop_density"]))
    high = pool.tail(top_pool).iloc[::-1]
    low = pool.head(top_pool)
    used, pairs = set(), []
    for r in high.itertuples():
        cand = low[~low["nta2020"].isin(used)]
        gap = (cand["ld_"] - r.ld_).abs()
        if gap.empty or gap.min() > caliper:
            continue
        m = cand.loc[gap.idxmin()]
        used.add(m["nta2020"])
        pairs.append((r.Index, m.name))
        if len(pairs) == k:
            break
    if len(pairs) < k:
        raise ValueError(f"{col}: only {len(pairs)} matched pairs inside caliper {caliper}")
    top = pool.loc[[a for a, _ in pairs]].drop(columns="ld_")
    bot = pool.loc[[b for _, b in pairs]].drop(columns="ld_")
    return top, bot


def draw_nta(cat: pd.DataFrame, nta_code: str, n: int, rng) -> pd.DataFrame:
    """`n` images of 1 NTA, 1 tract at a time in a cycle, so no tract dominates."""
    imgs = cat[cat["nta2020"] == nta_code]
    tracts = list(imgs.groupby("boroct2020").indices.values())
    rng.shuffle(tracts)
    for t in tracts:
        rng.shuffle(t)
    picked, i = [], 0
    while len(picked) < n and any(len(t) > i for t in tracts):
        for t in tracts:
            if len(t) > i and len(picked) < n:
                picked.append(t[i])
        i += 1
    return imgs.iloc[picked]


def draw_probe(name: str, nta: pd.DataFrame, cat: pd.DataFrame, k: int, per_nta: int,
               seed: int, caliper: float = 0.0, top_pool: int = 30,
               proxy: str | None = None) -> tuple[pd.DataFrame, dict]:
    p = PROBES[name]
    col = proxy or p.proxy
    if col not in nta or nta[col].notna().sum() < 2 * k:
        raise ValueError(f"{name}: proxy {col!r} has too few NTAs in the pool")
    rng = np.random.default_rng(seed)
    if caliper > 0:
        top, bot = pick_matched(nta, col, k, caliper, top_pool, rng)
    else:
        top, bot = pick_extremes(nta, col, k, rng)
    parts = []
    for group, g in (("top", top), ("bottom", bot)):
        for r in g.itertuples():
            d = draw_nta(cat, r.nta2020, per_nta, rng)
            parts.append(d.assign(**{f"draw_{name}": True, f"group_{name}": group,
                                     f"proxy_{name}": float(getattr(r, col))}))
    df = pd.concat(parts, ignore_index=True)
    info = {"proxy": col, "sign": p.proxy_sign,
            "top": [(r.ntaname, r.boroname, round(float(getattr(r, col)), 4), int(r.n_img))
                    for r in top.itertuples()],
            "bottom": [(r.ntaname, r.boroname, round(float(getattr(r, col)), 4), int(r.n_img))
                       for r in bot.itertuples()],
            "images": int(len(df)),
            "log_density_gap": round(float(np.log(top["pop_density"]).mean()
                                           - np.log(bot["pop_density"]).mean()), 3)}
    log(f"{name}: {len(df)} images on {col!r}; top {[t[0][:18] for t in info['top']]}; "
        f"bottom {[t[0][:18] for t in info['bottom']]}")
    return df[["image_path", f"draw_{name}", f"group_{name}", f"proxy_{name}"]], info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10, help="NTAs at each end")
    ap.add_argument("--per-nta", type=int, default=60)
    ap.add_argument("--min-pop", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--probes", nargs="+", default=list(PROBE_ORDER))
    ap.add_argument("--match-density", type=float, default=0.0, metavar="CALIPER",
                    help="pair each high NTA with a low NTA within CALIPER of its log "
                         "population density (0 = the plain extremes)")
    ap.add_argument("--top-pool", type=int, default=30,
                    help="NTAs at each end of the proxy that the match may use")
    ap.add_argument("--proxy", nargs="*", default=[], metavar="PROBE=COLUMN",
                    help="a proxy column that replaces the probe's own, e.g. "
                         "housing=homeless_per_capita (a per-area rate cannot match on density)")
    ap.add_argument("--out", default=str(OUT), help="the sample folder")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nta = pd.read_parquet(DATA_NTA / "nta_proxies.parquet")
    cat = pd.read_parquet(DATA_R1 / "catalog_tracts.parquet").merge(
        tract_to_nta(), on="boroct2020", how="left")
    nta["homeless_per_capita"] = nta["homeless_density"] * nta["area_km2"] / nta["population"]
    proxies = dict(kv.split("=", 1) for kv in args.proxy)
    pool = nta[(nta["population"] > args.min_pop) & (nta["n_img"] >= args.per_nta)]
    log(f"{len(nta)} NTAs, {len(pool)} in the pool; {len(cat):,} images, "
        f"{cat['nta2020'].notna().sum():,} with an NTA")

    parts, infos = [], {}
    for i, name in enumerate(args.probes):
        df, info = draw_probe(name, pool, cat, args.k, args.per_nta, args.seed + i,
                              args.match_density, args.top_pool, proxies.get(name))
        parts.append(df)
        infos[name] = info
    union = parts[0]
    for df in parts[1:]:
        union = union.merge(df, on="image_path", how="outer")
    for name in args.probes:
        union[f"draw_{name}"] = union[f"draw_{name}"].fillna(False).astype(bool)
        union[f"group_{name}"] = union[f"group_{name}"].fillna("")
    union = union.merge(cat[["image_path", "sample_id", "geoid", "boroct2020", "nta2020",
                             "face", "recording_id"]], on="image_path", how="left")
    union = union.merge(nta[[c for c in NTA_COLS if c in nta]], on="nta2020", how="left")
    union = union.sort_values("image_path").reset_index(drop=True)
    union.to_parquet(out / "images.parquet", index=False)
    summ = {"images": int(len(union)), "ntas": int(union["nta2020"].nunique()),
            "tracts": int(union["geoid"].nunique()),
            "per_probe": {n: int(union[f"draw_{n}"].sum()) for n in args.probes},
            "k": args.k, "per_nta": args.per_nta, "min_pop": args.min_pop,
            "seed": args.seed, "match_density": args.match_density,
            "top_pool": args.top_pool, "proxy_overrides": proxies, "probes": infos}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps({k: v for k, v in summ.items() if k != "probes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
