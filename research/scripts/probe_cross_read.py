"""Test of independence from the wealth readout.

Is each wealth-linked association the wealth readout decoded again?
Experiment 2 found 4 contrast tasks with a significant association on
refused images: class (income), housing_type (home value), intelligence
(bachelor's share) and trust (income). All 4 ground-truth variables are
wealth variables. Experiment 3 found single words (`smart`,
`trustworthy`, `immigrant`, ...) that follow their own census variable.
Neither experiment tested if these are 1 direction decoded several times.

Part A, the contrast tasks of Experiment 2. The primary readout of each
contrast task exists on every image of the experiment. Per contrast task
X other than class: the image-level Spearman correlation of the X
readout with the wealth readout; on the task's own sample, the AUC of
the X readout, of the wealth readout, of the X readout with the wealth
readout regressed out, and of the wealth readout with the X readout
regressed out (95% cluster-bootstrap confidence interval, neighborhoods
resampled).

Part B, the single words on the all-NTA sample. Per word that is not a
class word: the partial Spearman correlation with the word's census
variable, controlling for 3 covariate sets: log population density;
density and the wealth readout (the NTA mean of `lp_rich - lp_poor` at
the same layer); density and log median income.

    .venv-vllm025cu129/bin/python research/scripts/probe_cross_read.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from research.monocle.probe_frame import PROBES  # noqa: E402
from research.monocle.probe_words import WORDS  # noqa: E402
from scripts.probe_black_control import partial  # noqa: E402
from scripts.probe_group_test import auc_u, cluster_draws  # noqa: E402
from scripts.probe_word_dose import load_arm, proxy_of  # noqa: E402

NTA_ROOT = REPO / "outputs/_monocle/probe_nta"
ALL_ROOT = REPO / "outputs/_monocle/probe_words_all"
OUT = REPO / "outputs/_monocle/probe_cross_read"
OTHERS = ["housing_type", "intelligence", "trust", "migration", "race"]


def log(m: str) -> None:
    print(f"[cross-read] {m}", flush=True)


def resid(y, x):
    X = np.column_stack([np.ones_like(x), x])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def auc_ci(v, is_pos, ntas, rng, n_boot):
    pos = [np.where(is_pos & (ntas == k))[0] for k in sorted(set(ntas[is_pos]))]
    neg = [np.where(~is_pos & (ntas == k))[0] for k in sorted(set(ntas[~is_pos]))]
    boot = (cluster_draws(pos, n_boot, rng), cluster_draws(neg, n_boot, rng))
    bs = np.array([auc_u(v[pi], v[ni]) for pi, ni in zip(*boot)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return auc_u(v[is_pos], v[~is_pos]), float(lo), float(hi)


def part_a(rng, n_boot) -> pd.DataFrame:
    summary = {r["probe"]: r for r in json.load(open(NTA_ROOT / "group_test/summary.json"))}
    images = pd.read_parquet(NTA_ROOT / "sample/images.parquet")
    reads = {}
    for p in ["class"] + OTHERS:
        d = pd.read_parquet(NTA_ROOT / "vectors" / p / "reads.parquet")
        col = summary[p]["primary_read"]
        reads[p] = d[["image_path", col, "answered", "calib"]].rename(
            columns={col: f"read_{p}", "answered": f"answered_{p}", "calib": f"calib_{p}"})
        log(f"{p}: primary readout {col}")
    df = images
    for p, d in reads.items():
        df = df.merge(d, on="image_path", how="left")
    rows = []
    for x in OTHERS:
        P = PROBES[x]
        ok = ~df[f"calib_{x}"].astype(bool) & ~df["calib_class"].astype(bool)
        r_all = float(spearmanr(df.loc[ok, f"read_{x}"], df.loc[ok, "read_class"]).correlation)
        keep = ok & df[f"draw_{x}"].fillna(False).astype(bool) & df[f"group_{x}"].isin(["top", "bottom"])
        for split in ("all", "refused"):
            k = keep & (~df[f"answered_{x}"].astype(bool) if split == "refused" else True)
            d = df[k]
            if len(d) < 60:
                continue
            is_pos = (d[f"group_{x}"] == ("top" if P.proxy_sign > 0 else "bottom")).to_numpy()
            ntas = d["nta2020"].to_numpy()
            rx, rc = d[f"read_{x}"].to_numpy(float), d["read_class"].to_numpy(float)
            # The wealth readout is signed toward the affluence terms; on the sample
            # of x it is signed toward x's positive group by the sign of its own AUC.
            a_c = auc_u(rc[is_pos], rc[~is_pos])
            sc = 1.0 if a_c >= 0.5 else -1.0
            rec = {"probe": x, "proxy": P.proxy, "split": split, "n": int(len(d)),
                   "rho_with_class_all_images": r_all,
                   "rho_with_class_on_draw": float(spearmanr(rx, rc).correlation)}
            for name, v in (("x", rx), ("class", sc * rc), ("x_minus_class", resid(rx, rc)),
                            ("class_minus_x", sc * resid(rc, rx))):
                a, lo, hi = auc_ci(v, is_pos, ntas, rng, n_boot)
                rec.update({f"auc_{name}": a, f"lo_{name}": lo, f"hi_{name}": hi})
            rec["class_read_sign_on_draw"] = sc
            rows.append(rec)
    return pd.DataFrame(rows)


def part_b(arms, layers, rng, n_boot) -> pd.DataFrame:
    images = pd.read_parquet(ALL_ROOT / "sample/images.parquet")
    if "homeless_per_capita" not in images:
        images["homeless_per_capita"] = images["homeless_density"] * images["area_km2"] / images["population"]
    rows = []
    for arm in arms:
        df = load_arm(arm, ALL_ROOT)
        have = {c[3:] for c in df.columns if c.startswith("lp_")}
        for L in layers:
            d = df[df.layer == L].merge(images, on="image_path", how="left")
            if d.empty:
                continue
            d["class_read"] = d["lp_rich"] - d["lp_poor"]
            pcols = sorted({proxy_of(p) for p, _ in WORDS.values()})
            m = d.groupby("nta2020")[[f"lp_{w}" for w in have] + ["class_read", "pop_density"] + pcols].mean()
            for w, (probe, expect) in WORDS.items():
                if w not in have or probe == "class":
                    continue
                pcol = proxy_of(probe)
                ok = m[pcol].notna() & m["median_income"].notna() & (m["pop_density"] > 0)
                t = m[ok]
                sign = 1.0 if expect == "top" else -1.0
                x, y = t[pcol].to_numpy(float), t[f"lp_{w}"].to_numpy(float)
                ld = np.log(t["pop_density"].to_numpy(float))
                Z = {"d": ld[:, None],
                     "d_class": np.column_stack([ld, t["class_read"].to_numpy(float)]),
                     "d_income": np.column_stack([ld, np.log(t["median_income"].to_numpy(float))])}
                rec = {"arm": arm, "layer": L, "word": w, "probe": probe, "proxy": pcol, "n_nta": len(t),
                       "rho_proxy_income": float(spearmanr(x, t["median_income"]).correlation)}
                n = len(t)
                idx = rng.integers(0, n, (n_boot, n))
                for k, z in Z.items():
                    if pcol == "median_income" and k == "d_income":
                        rec.update({f"rho_{k}": np.nan, f"lo_{k}": np.nan, f"hi_{k}": np.nan})
                        continue
                    r = sign * partial(x, y, z)
                    bs = np.array([sign * partial(x[i], y[i], z[i]) for i in idx])
                    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                    rec.update({f"rho_{k}": r, f"lo_{k}": float(lo), f"hi_{k}": float(hi)})
                rows.append(rec)
            log(f"{arm} layer {L}: done")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="it_open,base_open")
    ap.add_argument("--layers", default="42,47")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    A = part_a(rng, args.n_boot)
    A.to_csv(OUT / "cross_read_contrasts.csv", index=False)
    B = part_b(args.arms.split(","), [int(x) for x in args.layers.split(",")], rng, args.n_boot)
    B.to_csv(OUT / "cross_read_words.csv", index=False)
    f = lambda v: "n/a" if pd.isna(v) else f"{v:+.2f}"
    lines = ["# Test of independence from the wealth readout", "",
             "## Part A: the Experiment 2 contrast tasks against the wealth readout", "",
             "AUC on the task's own sample. `probe` = the contrast task; `proxy` = its ground-truth "
             "variable. `x` = the primary readout of the contrast task; `class` = the wealth readout "
             "(the class contrast) on the same images; `x - class` = the task's readout with the "
             "wealth readout regressed out; `class - x` = the reverse. Intervals are 95% "
             "cluster-bootstrap confidence intervals (neighborhoods resampled).", "",
             "| probe | proxy | split | n | rho(x, class) | AUC x | AUC class | AUC x - class | AUC class - x |",
             "|-------|-------|-------|--:|--------------:|-------|-----------|---------------|---------------|"]
    for r in A.itertuples():
        c = lambda n: f"{getattr(r, 'auc_' + n):.2f} [{getattr(r, 'lo_' + n):.2f}, {getattr(r, 'hi_' + n):.2f}]"
        lines.append(f"| {r.probe} | {r.proxy} | {r.split} | {r.n} | {r.rho_with_class_on_draw:+.2f} | "
                     f"{c('x')} | {c('class')} | {c('x_minus_class')} | {c('class_minus_x')} |")
    lines += ["", "## Part B: the single words, controlling for the wealth readout", "",
              "Partial Spearman correlation over the NTAs of the word's log-probability with its "
              "ground-truth variable (`proxy`). `d` = controlling for log population density; "
              "`d+class` = density and the wealth readout (the NTA-mean `lp_rich - lp_poor` at the "
              "same layer); `d+income` = density and log median income. `proxy~income` = the "
              "Spearman correlation of the ground-truth variable with median income.", ""]
    for (arm, L), g in B.groupby(["arm", "layer"], sort=False):
        lines += [f"### {arm}, layer {L}", "",
                  "| word | proxy | proxy~income | d | d+class | d+income |",
                  "|------|-------|-------------:|---|---------|----------|"]
        for r in g.itertuples():
            c = lambda k: (f"{f(getattr(r, 'rho_' + k))} [{f(getattr(r, 'lo_' + k))}, {f(getattr(r, 'hi_' + k))}]")
            lines.append(f"| `{r.word}` | {r.proxy} | {r.rho_proxy_income:+.2f} | {c('d')} | {c('d_class')} | {c('d_income')} |")
        lines.append("")
    (OUT / "cross_read.md").write_text("\n".join(lines))
    log(f"wrote {OUT}/cross_read.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
