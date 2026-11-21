"""Round 3 of the lens-refusal probe: the single-word group test.

`monocle/probe_words.py` records `log p(word)` and the rank of each word
at the answer position, per image and block, under an open frame that
names no word. This script asks, for each probe word, whether the word
reads HIGHER in the neighborhood group where the model should hold the
association, against 3 bars:

| Bar | What it says |
|-----|--------------|
| 0.5 | the word does not follow the image |
| the cluster interval | a 95% interval from a bootstrap over the 20 NTAs |
| the pool band | the AUCs of the pool words whose mean `log p` sits within `--band` nats of the probe word's; any word splits 2 groups of different pictures a little |

AUC = P(log p on an image of the EXPECTED group > log p on an image of
the other group). The density-matched AUC follows round 2. The pool
statistic is 2-sided: `null_q95` is the 95th percentile of |AUC - 0.5|
over the band, and `p_pool` is the share of band words at or above the
probe word's |AUC - 0.5|.

    .venv-vllm025cu129/bin/python scripts/probe_word_test.py --arm it_open

The density-matched redraw lives under its own root:

    .venv-vllm025cu129/bin/python scripts/probe_word_test.py --arm it_open \
        --root outputs/_monocle/probe_words_dm \
        --images outputs/_monocle/probe_words_dm/sample/images.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from monocle.probe_words import WORDS  # noqa: E402
from scripts.probe_group_test import auc_u, cluster_draws, match_pairs  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe_words"
SAMPLE = REPO / "outputs/_monocle/probe_nta/sample/images.parquet"
MIN_GROUP = 4
MIN_PAIRS = 3
MIN_BAND = 20   # the fewest pool words in a band


def log(m: str) -> None:
    print(f"[probe-word-test] {m}", flush=True)


def load_arm(arm: str, root: Path = ROOT) -> pd.DataFrame:
    files = sorted((root / arm).glob("words_shard*.parquet"))
    if not files:
        raise SystemExit(f"{arm}: no shard files under {root / arm}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    log(f"{arm}: {len(files)} shards, {df.image_path.nunique()} images, "
        f"blocks {sorted(df.layer.unique())}")
    return df


def word_columns(df: pd.DataFrame) -> list[str]:
    return [c[3:] for c in df.columns if c.startswith("lp_")]


def test_arm(arm: str, layers: list[int], n_boot: int, seed: int, band: float,
             prior_arm: str | None, root: Path = ROOT, sample: Path = SAMPLE) -> pd.DataFrame:
    df = load_arm(arm, root)
    images = pd.read_parquet(sample)
    words = word_columns(df)
    pool = [w for w in words if w not in WORDS and w != "unclear"]
    prior = load_arm(prior_arm, root) if prior_arm else None
    rng = np.random.default_rng(seed)
    rows = []
    for L in layers:
        d = df[df.layer == L].merge(images, on="image_path", how="left")
        if d.empty:
            log(f"block {L}: no rows")
            continue
        mean_lp = {w: float(d[f"lp_{w}"].mean()) for w in words}
        pool_auc_cache: dict[tuple[str, str], float] = {}
        for w, (probe, expect) in WORDS.items():
            if w not in words:
                continue
            gcol, dcol = f"group_{probe}", f"draw_{probe}"
            if dcol not in d:
                continue
            sub = d[d[dcol].fillna(False).astype(bool) & d[gcol].isin(["top", "bottom"])]
            is_exp = (sub[gcol] == expect).to_numpy()
            ntas = sub["nta2020"].to_numpy()
            if sub.empty:
                continue
            v = sub[f"lp_{w}"].to_numpy(float)
            a, b = v[is_exp], v[~is_exp]
            auc = auc_u(a, b)
            # the cluster interval
            exp_ntas = sorted(set(ntas[is_exp]))
            oth_ntas = sorted(set(ntas[~is_exp]))
            pos_by = {k: np.where(is_exp & (ntas == k))[0] for k in exp_ntas}
            neg_by = {k: np.where(~is_exp & (ntas == k))[0] for k in oth_ntas}
            lo = hi = float("nan")
            if a.size >= MIN_GROUP and b.size >= MIN_GROUP:
                boot = (cluster_draws(list(pos_by.values()), n_boot, rng),
                        cluster_draws(list(neg_by.values()), n_boot, rng))
                bs = np.array([auc_u(v[pi], v[ni]) for pi, ni in zip(*boot)])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            # the density-matched AUC
            dens = sub.groupby("nta2020")["pop_density"].mean()
            ld = np.log(dens.where(dens > 0))
            pairs = match_pairs({k: float(ld[k]) for k in exp_ntas},
                                {k: float(ld[k]) for k in oth_ntas})
            per_pair = np.array([auc_u(v[pos_by[x]], v[neg_by[y]]) for x, y in pairs
                                 if pos_by[x].size and neg_by[y].size])
            am = float(per_pair.mean()) if per_pair.size >= MIN_PAIRS else float("nan")
            # the pool band: pool words of a near prior, on the same draw
            band_words = [q for q in pool if abs(mean_lp[q] - mean_lp[w]) <= band]
            if len(band_words) < MIN_BAND:
                # a word deep in the tail has few neighbours; take the nearest
                band_words = sorted(pool, key=lambda q: abs(mean_lp[q] - mean_lp[w]))[:MIN_BAND]
            devs = []
            for q in band_words:
                key = (probe, q)
                if key not in pool_auc_cache:
                    vq = sub[f"lp_{q}"].to_numpy(float)
                    top = (sub[gcol] == "top").to_numpy()
                    pool_auc_cache[key] = auc_u(vq[top], vq[~top])
                devs.append(abs(pool_auc_cache[key] - 0.5))
            devs = np.array(devs)
            dev = abs(auc - 0.5)
            null_q95 = float(np.percentile(devs, 95)) if devs.size else float("nan")
            p_pool = float((devs >= dev).mean()) if devs.size else float("nan")
            # the question-word prior: the mean log p under the closed frame
            # minus the open frame, on the same images
            prior_shift = float("nan")
            if prior is not None:
                pr = prior[prior.layer == L]
                if f"lp_{w}" in pr:
                    prior_shift = float(pr[f"lp_{w}"].mean() - mean_lp[w])
            rows.append({
                "arm": arm, "layer": L, "word": w, "probe": probe, "expect": expect,
                "n_exp": int(a.size), "n_oth": int(b.size),
                "mean_lp": mean_lp[w], "mean_rank": float(sub[f"rk_{w}"].mean()),
                "median_rank": float(sub[f"rk_{w}"].median()),
                "auc": auc, "lo": float(lo), "hi": float(hi), "auc_matched": am,
                "n_pairs": int(per_pair.size), "n_band": int(devs.size),
                "null_q95": null_q95, "p_pool": p_pool, "prior_shift": prior_shift,
                "clears_pool": bool(np.isfinite(null_q95) and dev > null_q95),
                "ci_excludes": bool(np.isfinite(lo) and lo > 0.5),
                "reversed": bool(np.isfinite(hi) and hi < 0.5),
            })
    return pd.DataFrame(rows)


def fmt(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_md(res: pd.DataFrame, out: Path, layer: int) -> str:
    r = res[res.layer == layer].sort_values(["probe", "expect", "word"])
    md = [f"### Block {layer}", "",
          "| Word | Draw | Expected group | mean log p | median rank | AUC [95% CI] "
          "| matched | pool band n | pool q95 of \\|AUC-0.5\\| | p_pool | prior shift | Verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, g in r.iterrows():
        if g.reversed:
            v = "reversed"
        elif g.ci_excludes and g.clears_pool:
            v = "reads"
        elif g.ci_excludes:
            v = "CI only"
        else:
            v = "null"
        md.append(f"| {g.word} | {g.probe} | {g.expect} | {fmt(g.mean_lp, 1)} | "
                  f"{fmt(g.median_rank, 0)} | {fmt(g.auc)} [{fmt(g.lo)}, {fmt(g.hi)}] | "
                  f"{fmt(g.auc_matched)} | {g.n_band} | {fmt(g.null_q95)} | "
                  f"{fmt(g.p_pool)} | {fmt(g.prior_shift, 1)} | {v} |")
    text = "\n".join(md) + "\n"
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="a folder under outputs/_monocle/probe_words")
    ap.add_argument("--prior-arm", default=None,
                    help="the closed-frame arm of the same checkpoint, for the prior shift")
    ap.add_argument("--layers", default="24,30,36,40,42,45,47")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--band", type=float, default=1.0,
                    help="half-width, in nats of mean log p, of the pool band")
    ap.add_argument("--report-layers", default="47,42")
    ap.add_argument("--root", default=str(ROOT), help="the arm folders' parent")
    ap.add_argument("--images", default=str(SAMPLE), help="the draw table")
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    root = Path(args.root)
    res = test_arm(args.arm, layers, args.n_boot, args.seed, args.band, args.prior_arm,
                   root, Path(args.images))
    out = root / args.arm / "word_test"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "word_test.csv", index=False)
    md = [f"# Single-word group test: {args.arm}", "",
          f"Pool band ±{args.band} nats of mean log p; {args.n_boot} cluster resamples; "
          f"seed {args.seed}. `reads` = the cluster interval excludes 0.5 AND |AUC-0.5| "
          f"clears the pool band's 95th percentile.", ""]
    for L in [int(x) for x in args.report_layers.split(",")]:
        md.append(write_md(res, out, L))
    (out / "word_test.md").write_text("\n".join(md))
    print("\n".join(md))
    summ = {"arm": args.arm, "n_words": int(res.word.nunique()),
            "reads_at_47": sorted(res[(res.layer == 47) & res.ci_excludes & res.clears_pool].word),
            "reversed_at_47": sorted(res[(res.layer == 47) & res.reversed].word)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
