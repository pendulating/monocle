"""The dose-response test of the single-word probe over all neighborhoods.

Round 3 read each word as a 2-group AUC over the 10 highest and 10
lowest NTAs of a proxy. On the all-neighborhood sample
(`research/scripts/probe_sample_all.py`, 197 NTAs) the read becomes a
correlation over neighborhoods:

| Column | What it is |
|--------|------------|
| `rho` | Spearman of the NTA-mean `log p(word)` with the proxy over the NTAs, signed so that + = the expected direction |
| `rho_partial` | the same after both sides are residualized on log population density (rank-based) |
| `lo`, `hi` | a 95% interval of `rho_partial` from a bootstrap over NTAs |
| `null_q95` | the 95th percentile of \\|rho_partial\\| over the pool words on the same proxy |
| `p_pool` | the share of pool words at or above the word's \\|rho_partial\\| |
| `auc_ext` | the round-3 statistic, recovered: the image-level AUC over the 10 highest and 10 lowest NTAs |
| `prior_shift` | closed-frame mean `log p` minus open-frame mean `log p` |

`reads` = the interval excludes 0 on the expected side AND
\\|rho_partial\\| clears `null_q95`. The housing proxy is the per-capita
rate (`homeless_per_capita`), as in the matched redraw.

    .venv-vllm025cu129/bin/python research/scripts/probe_word_dose.py --arm it_open --prior-arm it_closed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]
from research.monocle.probe_frame import PROBES  # noqa: E402
from research.monocle.probe_words import WORDS  # noqa: E402
from research.scripts.probe_group_test import auc_u  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe_words_all"
SAMPLE = ROOT / "sample/images.parquet"
PROXY_OVERRIDE = {"housing": "homeless_per_capita"}
K_EXT = 10


def log(m: str) -> None:
    print(f"[probe-word-dose] {m}", flush=True)


def proxy_of(probe: str) -> str:
    return PROXY_OVERRIDE.get(probe, PROBES[probe].proxy)


def load_arm(arm: str, root: Path) -> pd.DataFrame:
    files = sorted((root / arm).glob("words_shard*.parquet"))
    if not files:
        raise SystemExit(f"{arm}: no shard files under {root / arm}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    log(f"{arm}: {len(files)} shards, {df.image_path.nunique()} images, blocks {sorted(df.layer.unique())}")
    return df


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman of x and y with z partialled out, on ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    d = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / d) if d > 0 else float("nan")


def test_arm(arm: str, layers: list[int], n_boot: int, seed: int, prior_arm: str | None,
             root: Path, sample: Path) -> pd.DataFrame:
    df = load_arm(arm, root)
    images = pd.read_parquet(sample)
    if "homeless_per_capita" not in images and {"homeless_density", "area_km2", "population"} <= set(images):
        images["homeless_per_capita"] = images["homeless_density"] * images["area_km2"] / images["population"]
    words = [c[3:] for c in df.columns if c.startswith("lp_")]
    pool = [w for w in words if w not in WORDS and w != "unclear"]
    prior = load_arm(prior_arm, root) if prior_arm else None
    rng = np.random.default_rng(seed)
    nta_cols = ["nta2020", "pop_density"] + sorted({proxy_of(p) for p, _ in WORDS.values()})
    rows = []
    for L in layers:
        d = df[df.layer == L].merge(images[["image_path"] + nta_cols], on="image_path", how="left")
        if d.empty:
            log(f"block {L}: no rows")
            continue
        lp_cols = [f"lp_{w}" for w in words]
        nta_mean = d.groupby("nta2020")[lp_cols + ["pop_density"] + nta_cols[2:]].mean()
        mean_lp = {w: float(d[f"lp_{w}"].mean()) for w in words}
        ld = np.log(nta_mean["pop_density"].to_numpy(float))
        null_cache: dict[str, np.ndarray] = {}
        for w, (probe, expect) in WORDS.items():
            if w not in words:
                continue
            pcol = proxy_of(probe)
            ok = nta_mean[pcol].notna().to_numpy() & np.isfinite(ld)
            m = nta_mean[ok]
            x = m[pcol].to_numpy(float)
            sign = 1.0 if expect == "top" else -1.0
            y = m[f"lp_{w}"].to_numpy(float)
            rho = sign * float(spearmanr(x, y).correlation)
            rp = sign * partial_spearman(x, y, ld[ok])
            # the bootstrap over NTAs
            n = len(m)
            bs = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                bs[b] = sign * partial_spearman(x[idx], y[idx], ld[ok][idx])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            # the pool null on the same proxy
            if pcol not in null_cache:
                null_cache[pcol] = np.array([abs(partial_spearman(x, m[f"lp_{q}"].to_numpy(float), ld[ok]))
                                             for q in pool])
            devs = null_cache[pcol]
            null_q95 = float(np.percentile(devs, 95))
            p_pool = float((devs >= abs(rp)).mean())
            # the extremes AUC, as round 3
            order = m[pcol].sort_values()
            top_n, bot_n = order.index[-K_EXT:], order.index[:K_EXT]
            hi_grp, lo_grp = (top_n, bot_n) if expect == "top" else (bot_n, top_n)
            v_exp = d[d.nta2020.isin(hi_grp)][f"lp_{w}"].to_numpy(float)
            v_oth = d[d.nta2020.isin(lo_grp)][f"lp_{w}"].to_numpy(float)
            auc_ext = auc_u(v_exp, v_oth) if v_exp.size and v_oth.size else float("nan")
            prior_shift = float("nan")
            if prior is not None:
                pr = prior[prior.layer == L]
                if f"lp_{w}" in pr:
                    prior_shift = float(pr[f"lp_{w}"].mean() - mean_lp[w])
            rows.append({
                "arm": arm, "layer": L, "word": w, "probe": probe, "expect": expect, "proxy": pcol,
                "n_nta": int(n), "mean_lp": mean_lp[w],
                "median_rank": float(d[f"rk_{w}"].median()),
                "rho": rho, "rho_partial": rp, "lo": float(lo), "hi": float(hi),
                "n_pool": int(devs.size), "null_q95": null_q95, "p_pool": p_pool,
                "auc_ext": auc_ext, "prior_shift": prior_shift,
                "clears_pool": bool(abs(rp) > null_q95),
                "ci_excludes": bool(lo > 0), "reversed": bool(hi < 0),
            })
        log(f"block {L}: {sum(r['layer'] == L for r in rows)} words")
    return pd.DataFrame(rows)


def fmt(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_md(res: pd.DataFrame, layer: int) -> str:
    r = res[res.layer == layer].sort_values(["probe", "expect", "word"])
    md = [f"### Block {layer}", "",
          "| Word | Draw | Expected | proxy | n NTA | median rank | rho | rho partial [95% CI] "
          "| pool q95 | p_pool | AUC extremes | prior shift | Verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, g in r.iterrows():
        v = ("reversed" if g.reversed else "reads" if g.ci_excludes and g.clears_pool
             else "CI only" if g.ci_excludes else "null")
        md.append(f"| {g.word} | {g.probe} | {g.expect} | {g.proxy} | {g.n_nta} | {fmt(g.median_rank, 0)} | "
                  f"{fmt(g.rho)} | {fmt(g.rho_partial)} [{fmt(g.lo)}, {fmt(g.hi)}] | {fmt(g.null_q95)} | "
                  f"{fmt(g.p_pool)} | {fmt(g.auc_ext)} | {fmt(g.prior_shift, 1)} | {v} |")
    return "\n".join(md) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--prior-arm", default=None)
    ap.add_argument("--layers", default="24,30,36,40,42,45,46,47")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--report-layers", default="47,46,45,42")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--images", default=str(SAMPLE))
    args = ap.parse_args(argv)
    root = Path(args.root)
    layers = [int(x) for x in args.layers.split(",")]
    res = test_arm(args.arm, layers, args.n_boot, args.seed, args.prior_arm, root, Path(args.images))
    out = root / args.arm / "dose_test"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "dose.csv", index=False)
    md = [f"# Single-word dose-response test: {args.arm}", "",
          f"Spearman over NTAs of the NTA-mean log p(word) with the proxy, signed so + = the "
          f"expected direction; partial on log density; {args.n_boot} NTA resamples; seed "
          f"{args.seed}. `reads` = the interval excludes 0 AND |rho partial| clears the pool "
          f"words' 95th percentile on the same proxy.", ""]
    for L in [int(x) for x in args.report_layers.split(",")]:
        md.append(write_md(res, L))
    (out / "dose.md").write_text("\n".join(md))
    print("\n".join(md))
    summ = {"arm": args.arm, "n_words": int(res.word.nunique()),
            "n_nta": int(res.n_nta.max()) if len(res) else 0,
            "reads_at_47": sorted(res[(res.layer == 47) & res.ci_excludes & res.clears_pool].word),
            "reversed_at_47": sorted(res[(res.layer == 47) & res.reversed].word),
            "reads_at_46": sorted(res[(res.layer == 46) & res.ci_excludes & res.clears_pool].word)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
