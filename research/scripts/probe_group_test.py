"""E3 of the lens-refusal probe, round 2: the two-group separation test.

Round 1 (`research/scripts/probe_divergence.py`) put a Spearman of the lens read
against a continuous tract proxy. Round 2 draws the images from the TOP-10
and the BOTTOM-10 neighborhoods (NTA 2020) of each proxy, so the statistic
becomes a separation of 2 groups. The read and `c` are pos-minus-neg, so a
HIGH value means the POS pole. `Probe.proxy_sign` names the POS group: +1
puts it in the 'top' group of the proxy, -1 puts it in the 'bottom' group.

| Column | What it says |
|--------|--------------|
| auc | P(a POS-group read > a NEG-group read), ties count half |
| lo, hi | 95% interval, a cluster bootstrap over the 20 NTAs |
| auc_matched | the same AUC inside NTA pairs of equal density |
| d | the standardized mean difference |
| p | the 2-sided Mann-Whitney p-value over images |

H1 holds when the refused interval excludes 0.5 and reaches the answered
value. H0 puts the refused value at 0.5. A value between them means the
refusal weakens the judgment.

    .venv-vllm025cu129/bin/python research/scripts/probe_group_test.py --probes class
    .venv-vllm025cu129/bin/python research/scripts/probe_group_test.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from research.monocle.probe_frame import PROBE_ORDER, PROBES  # noqa: E402

DEFAULT_ROOT = REPO / "outputs/_monocle/probe_nta"
SELFTEST_DIR = Path("/tmp/claude-1643544/-share-pierson-matt-mllmsci/"
                    "2b389690-ac6d-4fa3-9298-ee79ced64a00/scratchpad/group_test_selftest")
PLANNED = "read_C_L42"
SPLITS = ("answered", "refused", "abstain", "all")
MIN_GROUP = 4     # images per group for a bootstrap interval
MIN_PAIRS = 3     # matched NTA pairs for a matched interval
MIN_ANSWERED = 8  # answered images for a verdict against the answered value


def log(m: str) -> None:
    print(f"[probe-group-test] {m}", flush=True)


def auc_u(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a POS read > a NEG read) from the Mann-Whitney U, ties count half."""
    n1, n2 = pos.size, neg.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def cohen_d(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n2 = pos.size, neg.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    s2 = ((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2)
    s = float(np.sqrt(s2))
    return float((pos.mean() - neg.mean()) / s) if s > 0 else float("nan")


def mw_p(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    try:
        return float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def cluster_draws(groups: list[np.ndarray], n_boot: int, rng) -> list[np.ndarray]:
    """Row positions of n_boot resamples of the clusters, with replacement."""
    if not groups:
        return []
    pick = rng.integers(0, len(groups), (n_boot, len(groups)))
    return [np.concatenate([groups[j] for j in row]) for row in pick]


def match_pairs(pos_ld: dict, neg_ld: dict) -> list[tuple[str, str]]:
    """Pair each POS NTA with a NEG NTA of near log density, greedy.

    The pair of the smallest gap goes first. Both NTAs then leave the grid,
    so no NTA enters 2 pairs.
    """
    a, b = sorted(pos_ld), sorted(neg_ld)
    if not a or not b:
        return []
    d = np.abs(np.array([pos_ld[k] for k in a])[:, None]
               - np.array([neg_ld[k] for k in b])[None, :])
    d = np.where(np.isfinite(d), d, np.inf)
    pairs = []
    for _ in range(min(len(a), len(b))):
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if not np.isfinite(d[i, j]):
            break
        pairs.append((a[i], b[j]))
        d[i, :] = np.inf
        d[:, j] = np.inf
    return pairs


def split_mask(df: pd.DataFrame, split: str) -> np.ndarray:
    ans = df["answered"].to_numpy(bool)
    if split == "answered":
        return ans
    if split == "refused":
        return ~ans
    if split == "abstain":
        return df["top_class"].to_numpy() == "abstain"
    return np.ones(len(df), bool)


def primary_read(reads: pd.DataFrame, read_cols: list[str]) -> tuple[str, str, dict]:
    """The read of record: the read that agrees best with c, round-1 rule.

    The choice sees the model's output alone, never the proxy, so the
    refused-image test stays held out.
    """
    ans = reads[reads["answered"] & ~reads["calib"]]
    agree_on = "answered"
    if len(ans) < 30:
        ans, agree_on = reads[~reads["calib"]], "all"
    agree = {c: float(spearmanr(ans[c], ans["c"]).statistic)
             for c in read_cols if c != "c"}
    agree = {k: v for k, v in agree.items() if np.isfinite(v)}
    return (max(agree, key=agree.get) if agree else PLANNED), agree_on, agree


def verdict_of(a_auc: float, r_auc: float, r_lo: float, r_hi: float, n_ans: int) -> str:
    # An interval above 0.5 reads WITH the proxy. An interval below 0.5 reads
    # AGAINST it, which is not H1: the residual holds a judgment, but the
    # opposite one. The verdict names that case "reversed".
    excl = bool(np.isfinite(r_lo) and (r_lo > 0.5))
    reversed_ = bool(np.isfinite(r_hi) and (r_hi < 0.5))
    if n_ans < MIN_ANSWERED or not np.isfinite(a_auc):
        if not np.isfinite(r_auc):
            return "n/a"
        if reversed_:
            return "reversed (no answered set)"
        return "H1 (no answered set)" if excl else "H0 (no answered set)"
    if reversed_:
        return "reversed"
    reaches = (r_hi >= a_auc) if a_auc >= 0.5 else (r_lo <= a_auc)
    return "H1" if (excl and reaches) else ("H0" if not excl else "between")


def run_probe(name: str, root: Path, images: pd.DataFrame, n_boot: int,
              seed: int) -> tuple[pd.DataFrame, dict]:
    p = PROBES[name]
    reads = pd.read_parquet(root / "vectors" / name / "reads.parquet")
    read_cols = [c for c in reads.columns if c.startswith("read_")] + ["c"]
    primary, agree_on, agree = primary_read(reads, read_cols)
    gcol = f"group_{name}"
    df = reads.merge(images, on="image_path", how="left")
    keep = df[f"draw_{name}"].fillna(False).astype(bool) & ~df["calib"].astype(bool)
    df = df[keep & df[gcol].isin(["top", "bottom"])].reset_index(drop=True)
    pos_label = "top" if p.proxy_sign > 0 else "bottom"
    is_pos = (df[gcol] == pos_label).to_numpy()
    ntas = df["nta2020"].to_numpy()
    dens = df.groupby("nta2020")["pop_density"].mean()
    ld = np.log(dens.where(dens > 0))
    side = df.groupby("nta2020")[gcol].first()
    gap = float(ld[side == "top"].mean() - ld[side == "bottom"].mean())
    if p.proxy in df and df[p.proxy].notna().any():
        hi_top = df.loc[df[gcol] == "top", p.proxy].mean()
        hi_bot = df.loc[df[gcol] == "bottom", p.proxy].mean()
        if not (hi_top > hi_bot):
            log(f"{name}: warning, the 'top' group is not high on {p.proxy}")
    pos_ntas = [k for k in ld.index if side[k] == pos_label]
    neg_ntas = [k for k in ld.index if side[k] != pos_label]
    pairs = match_pairs({k: float(ld[k]) for k in pos_ntas},
                        {k: float(ld[k]) for k in neg_ntas})
    rng = np.random.default_rng(seed)
    vals = {r: df[r].to_numpy(float) for r in read_cols if r in df}
    rows = []
    for split in SPLITS:
        m = split_mask(df, split)
        pos_by = {k: np.where(m & is_pos & (ntas == k))[0] for k in pos_ntas}
        neg_by = {k: np.where(m & ~is_pos & (ntas == k))[0] for k in neg_ntas}
        ipos, ineg = np.where(m & is_pos)[0], np.where(m & ~is_pos)[0]
        # The clusters resample inside each group, so no draw loses a group.
        boot = (cluster_draws(list(pos_by.values()), n_boot, rng),
                cluster_draws(list(neg_by.values()), n_boot, rng))
        ok = [(a, b) for a, b in pairs if pos_by[a].size and neg_by[b].size]
        pdraw = (rng.integers(0, len(ok), (n_boot, len(ok)))
                 if len(ok) >= MIN_PAIRS else None)
        for read in read_cols:
            v = vals.get(read)
            if v is None:
                continue
            a, b = v[ipos], v[ineg]
            lo = hi = lo_m = hi_m = float("nan")
            if a.size >= MIN_GROUP and b.size >= MIN_GROUP:
                bs = np.array([auc_u(v[pi], v[ni]) for pi, ni in zip(*boot)])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            per_pair = np.array([auc_u(v[pos_by[x]], v[neg_by[y]]) for x, y in ok])
            am = float(per_pair.mean()) if per_pair.size else float("nan")
            if pdraw is not None:
                lo_m, hi_m = np.percentile(per_pair[pdraw].mean(axis=1), [2.5, 97.5])
            rows.append({"probe": name, "proxy": p.proxy, "sign": p.proxy_sign,
                         "read": read, "split": split,
                         "n_pos": int(a.size), "n_neg": int(b.size),
                         "auc": auc_u(a, b), "lo": float(lo), "hi": float(hi),
                         "auc_matched": am, "lo_m": float(lo_m), "hi_m": float(hi_m),
                         "n_pairs": len(ok), "d": cohen_d(a, b), "p": mw_p(a, b)})
    res = pd.DataFrame(rows)
    res["primary"] = res["read"] == primary
    res["split"] = pd.Categorical(res["split"], SPLITS)
    res = res.sort_values(["read", "split"]).reset_index(drop=True)

    def cell(read, split, col="auc"):
        g = res[(res.read == read) & (res.split == split)]
        return round(float(g[col].iloc[0]), 3) if len(g) else None
    prim = res[res.read == primary].set_index("split")
    a, r = prim.loc["answered"], prim.loc["refused"]
    summ = {
        "probe": name, "proxy": p.proxy, "sign": int(p.proxy_sign),
        "pos_group": pos_label, "primary_read": primary, "agree_on": agree_on,
        "agree_with_c": round(agree.get(primary, float("nan")), 3),
        "n": {s: [int(prim.at[s, "n_pos"]), int(prim.at[s, "n_neg"])] for s in SPLITS},
        "n_pairs": int(r["n_pairs"]),
        "auc_answered": round(float(a["auc"]), 3),
        "ci_answered": [round(float(a["lo"]), 3), round(float(a["hi"]), 3)],
        "auc_refused": round(float(r["auc"]), 3),
        "ci_refused": [round(float(r["lo"]), 3), round(float(r["hi"]), 3)],
        "auc_refused_matched": round(float(r["auc_matched"]), 3),
        "ci_refused_matched": [round(float(r["lo_m"]), 3), round(float(r["hi_m"]), 3)],
        "log_density_gap": round(gap, 3),
        "d_refused": round(float(r["d"]), 3), "p_refused": float(r["p"]),
        "verdict": verdict_of(float(a["auc"]), float(r["auc"]), float(r["lo"]),
                              float(r["hi"]), int(a["n_pos"] + a["n_neg"])),
        "refusal_share": round(float((~df["answered"]).mean()), 3),
        "planned_read": {"read": PLANNED, "answered": cell(PLANNED, "answered"),
                         "refused": cell(PLANNED, "refused"),
                         "refused_matched": cell(PLANNED, "refused", "auc_matched")},
        "c": {"answered": cell("c", "answered"), "refused": cell("c", "refused")},
    }
    return res, summ


def fmt(x, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def run(root: Path, probes, n_boot: int, seed: int, tag: str) -> tuple[list, dict]:
    out = root / "group_test"
    out.mkdir(parents=True, exist_ok=True)
    images = pd.read_parquet(root / "sample/images.parquet")
    summs, tables = [], {}
    for name in probes:
        if not (root / "vectors" / name / "reads.parquet").exists():
            log(f"{name}: no reads.parquet; skipped")
            continue
        res, summ = run_probe(name, root, images, n_boot, seed)
        res.to_csv(out / f"{name}{tag}.csv", index=False)
        summs.append(summ)
        tables[name] = res
        log(json.dumps(summ))
    (out / f"summary{tag}.json").write_text(json.dumps(summs, indent=2))
    md = ["| Probe | Proxy | Read | n ans | n ref | AUC answered | AUC refused [95% CI] "
          "| AUC refused matched [95% CI] | log-density gap | Verdict | c ans / ref |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summs:
        md.append(
            f"| {s['probe']} | {s['proxy']} | {s['primary_read']} | "
            f"{sum(s['n']['answered'])} | {sum(s['n']['refused'])} | "
            f"{fmt(s['auc_answered'])} | {fmt(s['auc_refused'])} "
            f"[{fmt(s['ci_refused'][0])}, {fmt(s['ci_refused'][1])}] | "
            f"{fmt(s['auc_refused_matched'])} "
            f"[{fmt(s['ci_refused_matched'][0])}, {fmt(s['ci_refused_matched'][1])}] | "
            f"{fmt(s['log_density_gap'])} | {s['verdict']} | "
            f"{fmt(s['c']['answered'], 3)} / {fmt(s['c']['refused'], 3)} |")
    if summs:
        (out / f"group_test{tag}.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
    return summs, tables


def build_selftest(root: Path, probes=("class", "housing"), seed: int = 1) -> None:
    """A synthetic draw: 20 NTAs, 60 images each, 1 planted read, 1 null read.

    `read_C_L42` holds a planted group effect and a density confound.
    `read_C_L41` holds no effect at all. The seed is 1 because the null read
    of seed 0 sits 2 sigma from 0.5 by chance, and the test asserts on it.
    """
    rng = np.random.default_rng(seed)
    n_nta, per = 20, 60
    names = [f"NT{i:02d}" for i in range(n_nta)]
    top = np.arange(n_nta) < 10
    ld = 9.5 + 0.8 * top + rng.normal(0, 0.3, n_nta)  # the density confound
    idx = np.repeat(np.arange(n_nta), per)
    n = idx.size
    images = pd.DataFrame({
        "image_path": [f"/img/{names[i]}/{j}.jpg" for i in range(n_nta) for j in range(per)],
        "sample_id": [f"s{k}" for k in range(n)], "geoid": [f"3600{i:07d}" for i in idx],
        "boroct2020": [f"1{i:05d}" for i in idx], "nta2020": [names[i] for i in idx],
        "ntaname": [names[i] for i in idx], "boroname": "Bronx", "face": "F",
        "recording_id": [f"R{k}" for k in range(n)], "population": 40000.0,
        "area_km2": 2.0, "pop_density": np.exp(ld[idx])})
    images["median_income"] = np.where(top[idx], 90000.0, 30000.0)
    images["homeless_density"] = np.where(top[idx], 50.0, 5.0)
    for c in ("black_share", "white_share", "foreign_born_share", "bachelors_share",
              "median_home_value", "crime_density", "crime_per_capita",
              "prostitution_density", "violent_per_capita"):
        images[c] = np.where(top[idx], 0.6, 0.2)
    z = (ld[idx] - ld.mean()) / ld.std()
    for name in probes:
        images[f"draw_{name}"] = True
        images[f"group_{name}"] = np.where(top[idx], "top", "bottom")
        images[f"proxy_{name}"] = images[PROBES[name].proxy].to_numpy()
        pos = top[idx] if PROBES[name].proxy_sign > 0 else ~top[idx]
        planted = (1.2 * pos - 0.2 * z + rng.normal(0, 0.3, n_nta)[idx]
                   + rng.normal(0, 1.0, n))
        null = rng.normal(0, 1.0, n)  # no group effect and no NTA effect
        c = 0.8 * planted + rng.normal(0, 0.6, n)
        answered = rng.random(n) < 0.4
        top_class = np.where(answered, np.where(planted > 0, "pos", "neg"),
                             np.where(rng.random(n) < 0.15, "other", "abstain"))
        reads = pd.DataFrame({
            "image_path": images["image_path"], "probe": name, "c": c,
            "logit_pos": 0.0, "logit_neg": 0.0, "logit_abstain": 0.0,
            "p_pos": 0.0, "p_neg": 0.0, "p_abstain": 1.0, "top_class": top_class,
            "answered": answered, "calib": rng.random(n) < 0.05,
            "read_C_L41": null.astype("float32"), "read_C_L42": planted.astype("float32"),
            "read_C_L43": (0.5 * planted + rng.normal(0, 1.0, n)).astype("float32"),
            "read_47_L42": (rng.normal(0, 1.0, n)).astype("float32")})
        for k in range(1, 6):
            reads[f"top{k}_id"], reads[f"top{k}"] = 0, "x"
        d = root / "vectors" / name
        d.mkdir(parents=True, exist_ok=True)
        reads.to_parquet(d / "reads.parquet", index=False)
    (root / "sample").mkdir(parents=True, exist_ok=True)
    images.to_parquet(root / "sample/images.parquet", index=False)


def selftest(n_boot: int, seed: int) -> int:
    probes = ("class", "housing")
    root = SELFTEST_DIR / "root"
    root.mkdir(parents=True, exist_ok=True)
    build_selftest(root)
    t0 = time.time()
    summs, tables = run(root, probes, n_boot, seed, "")
    dt = time.time() - t0
    for name in probes:
        res = tables[name]
        for split in ("all", "refused", "answered"):
            g = res[(res.read == PLANNED) & (res.split == split)].iloc[0]
            assert g["auc"] > 0.6, f"{name}/{split}: planted auc {g['auc']:.3f} <= 0.6"
            assert g["lo"] > 0.5, f"{name}/{split}: planted lo {g['lo']:.3f} <= 0.5"
            assert np.isfinite(g["auc_matched"]), f"{name}/{split}: no matched auc"
            z = res[(res.read == "read_C_L41") & (res.split == split)].iloc[0]
            assert z["lo"] <= 0.5 <= z["hi"], (
                f"{name}/{split}: null ci [{z['lo']:.3f}, {z['hi']:.3f}] misses 0.5")
        assert tables[name]["primary"].any()
        s = [x for x in summs if x["probe"] == name][0]
        assert s["primary_read"] == PLANNED, f"{name}: primary {s['primary_read']}"
    log(f"selftest OK in {dt:.1f} s ({len(probes)} probes, {n_boot} bootstrap draws)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--probes", nargs="+", default=list(PROBE_ORDER))
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--tag", default="", help="a suffix for the output files")
    ap.add_argument("--selftest", action="store_true",
                    help="run on a synthetic frame in the scratch directory")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest(args.n_boot, args.seed)
    run(Path(args.root), args.probes, args.n_boot, args.seed, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
