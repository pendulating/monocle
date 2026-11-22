"""Colour control for the `Black` token: is the base model's association a colour association?

The single-word logit-lens analysis over all 197 neighborhoods found
that `log p(" Black")` in the base model follows the Black share of the
neighborhood. The token is also a colour word. 2 controls, on the same
197 NTAs and the same statistic as `research/scripts/probe_word_dose.py` (partial
Spearman correlation on ranks, bootstrap over NTAs, the control-word
threshold):

1. The luminance control. The partial Spearman correlation of the
   logit-lens readout with `black_share`, controlling for log population
   density AND the NTA-mean image luminance (mean luminance, dark-pixel
   share) as covariates.
2. The colour-word control. The same statistic for the colour words
   decoded on the same images: the lowercase ` black` and ` white`, the
   capitalized colour words, and the colour words of the control words.
   If `Black` is a colour association, the colour words follow the Black
   share too, and `Black` follows the luminance.

    .venv-vllm025cu129/bin/python research/scripts/probe_black_control.py \\
        --root outputs/_monocle/probe_black_control --arms it_answer,base_answer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from research.monocle.probe_words import WORDS  # noqa: E402
from scripts.probe_word_dose import load_arm  # noqa: E402

SAMPLE = REPO / "outputs/_monocle/probe_words_all/sample/images.parquet"
LUMA = REPO / "outputs/_monocle/probe_black_control/luma.parquet"
RACE = ["Black", "White"]
COLOUR_EXTRA = ["black", "white", "Brown", "Red", "Blue", "Green", "Gray", "Dark", "Bright"]
COLOUR_POOL = ["dark", "bright", "brown", "gray", "grey", "red", "blue", "green", "yellow",
               "orange", "pink", "purple"]


def log(m: str) -> None:
    print(f"[black-control] {m}", flush=True)


def partial(x, y, Z) -> float:
    """Partial Spearman correlation of x and y on ranks, controlling for the columns of Z."""
    rx, ry = rankdata(x), rankdata(y)
    R = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z.T])
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
    ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    d = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / d) if d > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "outputs/_monocle/probe_black_control"))
    ap.add_argument("--arms", default="it_answer,base_answer")
    ap.add_argument("--layers", default="30,36,42,47")
    ap.add_argument("--luma", default=str(LUMA))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    root = Path(args.root)
    images = pd.read_parquet(SAMPLE)[["image_path", "nta2020", "pop_density", "black_share"]]
    luma = pd.read_parquet(args.luma)[["image_path", "luma_mean", "dark_share"]]
    images = images.merge(luma, on="image_path", how="left")
    log(f"{int(images.luma_mean.isna().sum())} images have no luminance value")
    rng = np.random.default_rng(args.seed)
    rows = []
    for arm in args.arms.split(","):
        df = load_arm(arm, root)
        have = [c[3:] for c in df.columns if c.startswith("lp_")]
        pool = [w for w in have if w not in WORDS and w not in COLOUR_EXTRA and w != "unclear"]
        tested = [w for w in RACE + COLOUR_EXTRA + COLOUR_POOL if w in have]
        for L in [int(x) for x in args.layers.split(",")]:
            d = df[df.layer == L].merge(images, on="image_path", how="left")
            if d.empty:
                continue
            m = d.groupby("nta2020")[[f"lp_{w}" for w in have]
                                     + ["pop_density", "black_share", "luma_mean", "dark_share"]].mean()
            m = m[m.black_share.notna() & m.luma_mean.notna() & (m.pop_density > 0)]
            x = m.black_share.to_numpy(float)
            Zd = np.log(m.pop_density.to_numpy(float))[:, None]
            Zb = np.column_stack([Zd[:, 0], m.luma_mean.to_numpy(float), m.dark_share.to_numpy(float)])
            bars = {k: float(np.percentile([abs(partial(x, m[f"lp_{q}"].to_numpy(float), Z))
                                            for q in pool], 95))
                    for k, Z in (("density", Zd), ("density+brightness", Zb))}
            n = len(m)
            for w in tested:
                y = m[f"lp_{w}"].to_numpy(float)
                rec = {"arm": arm, "layer": L, "word": w, "n_nta": n,
                       "kind": "race" if w in RACE else "colour",
                       "rho_with_luma": partial(m.luma_mean.to_numpy(float), y, Zd)}
                for k, Z in (("density", Zd), ("density+brightness", Zb)):
                    r = partial(x, y, Z)
                    bs = np.empty(args.n_boot)
                    for b in range(args.n_boot):
                        i = rng.integers(0, n, n)
                        bs[b] = partial(x[i], y[i], Z[i])
                    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                    tag = "d" if k == "density" else "db"
                    rec.update({f"rho_{tag}": r, f"lo_{tag}": float(lo), f"hi_{tag}": float(hi),
                                f"bar_{tag}": bars[k],
                                f"reads_{tag}": bool(lo > 0 and abs(r) > bars[k])})
                rows.append(rec)
            log(f"{arm} layer {L}: {n} NTAs; rho(black_share, luma | density) = "
                f"{partial(x, m.luma_mean.to_numpy(float), Zd):+.2f}")
    res = pd.DataFrame(rows)
    out = root / "black_control"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "black_control.csv", index=False)
    lines = ["# Colour control for the `Black` token", "",
             "`rho_d` = the partial Spearman correlation of the NTA-mean logit-lens readout with "
             "`black_share`, controlling for log population density. `rho_db` = the luminance "
             "control: the same, with the NTA-mean image luminance and dark-pixel share as more "
             "covariates. `bar` = the control-word threshold (the 95th percentile of the control "
             "words under the same partial correlation). `*` = the confidence interval excludes 0 "
             "and rho exceeds the threshold. `luma` = the partial Spearman correlation of the "
             "readout with the mean image luminance.", ""]
    for arm, g in res.groupby("arm", sort=False):
        lines += [f"## {arm}", "",
                  "| layer | word | kind | rho_d | rho_db | bar_d | bar_db | luma |",
                  "|------:|------|------|------:|-------:|------:|-------:|-----:|"]
        for r in g.itertuples():
            lines.append(f"| {r.layer} | `{r.word}` | {r.kind} | {r.rho_d:+.2f}{'*' if r.reads_d else ''} "
                         f"[{r.lo_d:+.2f}, {r.hi_d:+.2f}] | {r.rho_db:+.2f}{'*' if r.reads_db else ''} "
                         f"[{r.lo_db:+.2f}, {r.hi_db:+.2f}] | {r.bar_d:.2f} | {r.bar_db:.2f} | "
                         f"{r.rho_with_luma:+.2f} |")
        lines.append("")
    (out / "black_control.md").write_text("\n".join(lines))
    log(f"wrote {out}/black_control.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
