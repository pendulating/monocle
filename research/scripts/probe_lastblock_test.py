"""The last-block decomposition: which term of `log p(word)` moves the AUC.

Reads the 3 pseudo-arms that `research/monocle/probe_lastblock.py` writes (`lp`
= log p, `lg` = logit, `rd` = the raw projection `logit * rms`) and the
scalar table, at blocks 46 and 47, and answers 3 questions per word:

1. Does the AUC of the word fall between 46 and 47 in `lp`, in `lg`,
   in `rd`? A fall in `lp` alone points at the normalizer `lse`; a fall
   in `lg` but not `rd` points at the scale `rms`; a fall in `rd` is a
   change of the residual's projection on the word.
2. Do `lse`, `rms`, `|h|` and the top-1 logit split the 2 groups of the
   word's draw at 46 and at 47 (AUC with the NTA cluster interval)?
3. Does the block-47 write erase the block-46 projection? The
   correlation across the draw's images of `rd47 - rd46` with `rd46`,
   and the standardized group gap of `rd` at 46 and 47.

    .venv-vllm025cu129/bin/python research/scripts/probe_lastblock_test.py --root outputs/_monocle/probe_lastblock/it
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.monocle.probe_words import WORDS  # noqa: E402
from scripts.probe_group_test import auc_u, cluster_draws  # noqa: E402
from scripts.probe_word_test import SAMPLE, test_arm  # noqa: E402

BLOCKS = [46, 47]
SCALARS = ["lse", "rms", "hnorm", "top1_logit"]


def log(m: str) -> None:
    print(f"[probe-lastblock-test] {m}", flush=True)


def mark(g) -> str:
    if g.reversed:
        return "-"
    if g.ci_excludes and g.clears_pool:
        return "*"
    return "+" if g.ci_excludes else " "


def group_auc(v, is_exp, ntas, rng, n_boot):
    a, b = v[is_exp], v[~is_exp]
    if a.size < 4 or b.size < 4:
        return float("nan"), float("nan"), float("nan")
    auc = auc_u(a, b)
    pos = [np.where(is_exp & (ntas == k))[0] for k in sorted(set(ntas[is_exp]))]
    neg = [np.where(~is_exp & (ntas == k))[0] for k in sorted(set(ntas[~is_exp]))]
    boot = (cluster_draws(pos, n_boot, rng), cluster_draws(neg, n_boot, rng))
    bs = np.array([auc_u(v[pi], v[ni]) for pi, ni in zip(*boot)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return auc, float(lo), float(hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="<out-dir> of probe_lastblock (holds lp/ lg/ rd/)")
    ap.add_argument("--images", default=str(SAMPLE))
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = root / "lastblock_test"
    out.mkdir(exist_ok=True)
    images = pd.read_parquet(args.images)
    rng = np.random.default_rng(args.seed)

    # 1. the word test on each statistic
    res = {}
    for s in ("lp", "lg", "rd"):
        r = test_arm(s, BLOCKS, args.n_boot, args.seed, 1.0, None, root, Path(args.images))
        r["stat"] = s
        res[s] = r
    allres = pd.concat(res.values(), ignore_index=True)
    allres.to_csv(out / "word_stats.csv", index=False)
    piv = {}
    for s, r in res.items():
        r = r.assign(cell=r.apply(lambda g: f"{g.auc:.2f}{mark(g)}", axis=1))
        piv[s] = r.pivot(index="word", columns="layer", values="cell")
    words = [w for w in WORDS if w in piv["lp"].index]
    md = [f"# The last-block decomposition: {root.name}", "",
          "AUC of the word between the 2 groups of its draw (expected group > other), "
          "at blocks 46 and 47, for the 3 terms. `*` reads under all 3 bars, `+` interval "
          "only, `-` reversed. `lp` = log p; `lg` = logit; `rd` = logit x rms(h).", "",
          "| Word | Draw, expected | lp 46 | lp 47 | lg 46 | lg 47 | rd 46 | rd 47 |",
          "|---|---|---|---|---|---|---|---|"]
    for w in words:
        p, e = WORDS[w]
        md.append(f"| {w} | {p}, {e} | " + " | ".join(
            piv[s].loc[w, L] for s in ("lp", "lg", "rd") for L in BLOCKS) + " |")

    # 2. the scalars by draw, and 3. the erase test by word
    sc = pd.concat([pd.read_parquet(f) for f in sorted(root.glob("scalars_shard*.parquet"))],
                   ignore_index=True)
    geo = sc[sc.layer == -1][["image_path", "dnorm", "cos47_46"]]
    sc = sc[sc.layer > 0].merge(images, on="image_path", how="left")
    rd = pd.concat([pd.read_parquet(f) for f in sorted((root / "rd").glob("words_shard*.parquet"))],
                   ignore_index=True).merge(images, on="image_path", how="left")
    md += ["", "## The scalars by draw", "",
           "AUC of the per-image scalar, top group > bottom group, with the cluster interval. "
           "`lse` = log-sum-exp of the logits; `rms` = the RMSNorm scale of h; `|h|` = the "
           "residual norm; `top1` = the top-1 logit.", "",
           "| Draw | block | lse | rms | \\|h\\| | top1 |", "|---|---|---|---|---|---|"]
    srows = []
    for probe in sorted({p for p, _ in WORDS.values()}):
        gcol, dcol = f"group_{probe}", f"draw_{probe}"
        for L in BLOCKS:
            d = sc[(sc.layer == L) & sc[dcol].fillna(False).astype(bool) & sc[gcol].isin(["top", "bottom"])]
            is_top = (d[gcol] == "top").to_numpy()
            ntas = d["nta2020"].to_numpy()
            cells = []
            for s in SCALARS:
                a, lo, hi = group_auc(d[s].to_numpy(float), is_top, ntas, rng, args.n_boot)
                cells.append(f"{a:.2f} [{lo:.2f}, {hi:.2f}]")
                srows.append({"probe": probe, "layer": L, "scalar": s, "auc_top": a, "lo": lo, "hi": hi})
            md.append(f"| {probe} | {L} | " + " | ".join(cells) + " |")
    pd.DataFrame(srows).to_csv(out / "scalars.csv", index=False)
    g = geo.merge(sc[sc.layer == 46][["image_path", "hnorm"]], on="image_path")
    md += ["", f"Block-47 write: |h47 - h46| / |h46| = {float((g.dnorm / g.hnorm).median()):.2f} (median), "
           f"cos(h47, h46) = {float(g.cos47_46.median()):.3f} (median), over {len(g)} images.", ""]

    md += ["## The erase test by word", "",
           "Across the images of the word's draw: `r(d, rd46)` = the correlation of the "
           "block-47 change of the raw projection with its block-46 value; `gap46`, `gap47` = "
           "the standardized gap (expected group minus other, over the pooled SD) of `rd`.", "",
           "| Word | Draw, expected | r(d, rd46) | gap46 | gap47 | mean d, expected | mean d, other |",
           "|---|---|---|---|---|---|---|"]
    erows = []
    for w in words:
        probe, expect = WORDS[w]
        gcol, dcol = f"group_{probe}", f"draw_{probe}"
        sel = rd[dcol].fillna(False).astype(bool) & rd[gcol].isin(["top", "bottom"])
        a = rd[sel & (rd.layer == 46)].set_index("image_path")
        b = rd[sel & (rd.layer == 47)].set_index("image_path").reindex(a.index)
        v46, v47 = a[f"lp_{w}"].to_numpy(float), b[f"lp_{w}"].to_numpy(float)
        dd = v47 - v46
        is_exp = (a[gcol] == expect).to_numpy()
        if is_exp.sum() < 2 or (~is_exp).sum() < 2:
            continue
        r = float(np.corrcoef(dd, v46)[0, 1])
        def gap(v):
            sd = np.sqrt((v[is_exp].var() + v[~is_exp].var()) / 2)
            return float((v[is_exp].mean() - v[~is_exp].mean()) / sd) if sd > 0 else float("nan")
        row = {"word": w, "probe": probe, "expect": expect, "r_d_rd46": r,
               "gap46": gap(v46), "gap47": gap(v47),
               "d_exp": float(dd[is_exp].mean()), "d_oth": float(dd[~is_exp].mean())}
        erows.append(row)
        md.append(f"| {w} | {probe}, {expect} | {r:.2f} | {row['gap46']:.2f} | {row['gap47']:.2f} | "
                  f"{row['d_exp']:.1f} | {row['d_oth']:.1f} |")
    pd.DataFrame(erows).to_csv(out / "erase.csv", index=False)
    text = "\n".join(md) + "\n"
    (out / "lastblock_test.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
