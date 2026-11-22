"""E3 of the lens-refusal probe: the divergence test of H1.

H1 says the residual holds the judgment that the refusal withholds. The
statistic: Spearman of the lens read with the tract proxy, on the images
the model ANSWERS and on the images it REFUSES, with a bootstrap over
images (1 image per tract by construction). H1 holds when the refused
interval covers the answered value and excludes 0. H0 puts the refused
value at the floor. A value between them means the refusal weakens the
judgment, and the table reports the ratio.

The proxy is oriented so that a HIGH value goes with the POS pole
(`Probe.proxy_sign`). A probe whose proxy is not on disk is reported on
its fallback proxy, with the column named.

Reads `outputs/_monocle/probe/vectors/<probe>/reads.parquet` and
`outputs/_monocle/probe/sample/images.parquet`. The proxy columns come
from `outputs/_monocle/probe/data/tract_proxies.parquet` on `geoid` at
read time, because the sample table was built before the ACS pull
(2026-09-07) and its share columns are empty. Writes
`outputs/_monocle/probe/divergence/<probe>.csv`, `summary.json`, and
`divergence.md` (all probes).

    .venv-3.12/bin/python research/scripts/probe_divergence.py --probes class housing gang
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
from research.monocle.probe_frame import PROBE_ORDER, PROBES  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"
OUT = ROOT / "divergence"
PLANNED = "read_C_L42"


def log(m: str) -> None:
    print(f"[probe-divergence] {m}", flush=True)


def boot_spearman(x: np.ndarray, y: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    n = len(x)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    point = float(spearmanr(x, y).statistic)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bs[b] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi)


def refresh_proxies(images: pd.DataFrame) -> pd.DataFrame:
    """Replace the proxy columns of the sample with the tract table on disk."""
    tr = pd.read_parquet(ROOT / "data/tract_proxies.parquet")
    cols = sorted({p.proxy for p in PROBES.values()} | {p.fallback_proxy for p in PROBES.values()})
    cols = [c for c in cols if c in tr]
    out = images.drop(columns=[c for c in cols if c in images])
    out = out.merge(tr[["geoid"] + cols], on="geoid", how="left")
    for c in cols:
        log(f"proxy {c}: {int(out[c].notna().sum())} of {len(out)} images")
    return out


def run_probe(name: str, images: pd.DataFrame, n_boot: int, seed: int,
              subset: str, proxy: tuple[str, int] | None = None) -> tuple[pd.DataFrame, dict]:
    p = PROBES[name]
    reads = pd.read_parquet(ROOT / "vectors" / name / "reads.parquet")
    df = reads.merge(images, on="image_path", how="left")
    if subset == "own":
        df = df[df[f"draw_{name}"]]
    df = df[~df["calib"]]
    if proxy is not None:
        col, sign = proxy
    else:
        col = p.proxy if (p.proxy in df and df[p.proxy].notna().sum() >= 100) else p.fallback_proxy
        sign = p.proxy_sign if col == p.proxy else PROBES["class"].proxy_sign
    df = df.dropna(subset=[col]).copy()
    df["proxy_oriented"] = sign * df[col].astype(float)
    rng = np.random.default_rng(seed)
    # The read of record for this probe: the (lens, block) whose read agrees
    # best with c on the ANSWERED images of the union, outside the
    # calibration set. That choice sees the model's output alone, never the
    # proxy, so the refused-image test stays held out.
    read_cols = [c for c in reads.columns if c.startswith("read_")] + ["c"]
    ans_all = reads[reads["answered"] & ~reads["calib"]]
    agree_on = "answered"
    if len(ans_all) < 30:
        # A probe the model refuses almost everywhere: c is still defined on
        # every image, so the agreement runs over all of them.
        ans_all = reads[~reads["calib"]]
        agree_on = "all"
    agree = {c: float(spearmanr(ans_all[c], ans_all["c"]).statistic)
             for c in read_cols if c != "c"}
    primary = max(agree, key=agree.get) if agree else PLANNED
    rows = []
    for read in read_cols:
        if read not in df:
            continue
        for split, mask in (("answered", df["answered"]),
                            ("refused", ~df["answered"]),
                            ("abstain", df["top_class"] == "abstain"),
                            ("all", pd.Series(True, index=df.index))):
            g = df[mask]
            pt, lo, hi = boot_spearman(g[read].to_numpy(), g["proxy_oriented"].to_numpy(),
                                       n_boot, rng)
            rows.append({"probe": name, "proxy": col, "read": read, "split": split,
                         "n": int(len(g)), "rho": pt, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    res["primary"] = res["read"] == primary
    a = res[(res.read == primary) & (res.split == "answered")].iloc[0]
    r = res[(res.read == primary) & (res.split == "refused")].iloc[0]
    # H1: the refused read tracks the proxy (its interval excludes 0) and is
    # not below the answered read (the interval reaches it). H0: the refused
    # interval covers 0. Between: it excludes 0 but sits below the answered
    # value. With no answered value, the refused interval alone decides
    # H1 against H0.
    verdict = "n/a"
    excl0 = (r["lo"] > 0) or (r["hi"] < 0)
    if np.isfinite(r["rho"]) and np.isfinite(a["rho"]):
        reaches = (r["hi"] >= a["rho"]) if a["rho"] >= 0 else (r["lo"] <= a["rho"])
        verdict = "H1" if (excl0 and reaches) else ("H0" if not excl0 else "between")
    elif np.isfinite(r["rho"]):
        verdict = "H1 (no answered set)" if excl0 else "H0 (no answered set)"
    def cell(read, split):
        g = res[(res.read == read) & (res.split == split)]
        return round(float(g["rho"].iloc[0]), 3) if len(g) else None
    summ = {"probe": name, "proxy": col, "subset": subset, "primary_read": primary,
            "agree_on": agree_on,
            "agree_with_c_answered": round(agree.get(primary, float("nan")), 3),
            "planned_read": {"read": PLANNED, "answered": cell(PLANNED, "answered"),
                             "refused": cell(PLANNED, "refused")},
            "c": {"answered": cell("c", "answered"), "refused": cell("c", "refused")},
            "n_answered": int(a["n"]), "n_refused": int(r["n"]),
            "rho_answered": round(float(a["rho"]), 3),
            "rho_refused": round(float(r["rho"]), 3),
            "ci_refused": [round(float(r["lo"]), 3), round(float(r["hi"]), 3)],
            "ratio": round(float(r["rho"] / a["rho"]), 3) if a["rho"] else float("nan"),
            "verdict": verdict,
            "refusal_share": round(float((~df["answered"]).mean()), 3)}
    return res, summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", nargs="+", default=list(PROBE_ORDER))
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--subset", choices=["own", "union"], default="own",
                    help="own: the probe's stratified draw; union: every scored image")
    ap.add_argument("--proxy", nargs="*", default=[],
                    help="override, probe=column:sign, e.g. gang=crime_per_capita:-1")
    ap.add_argument("--tag", default="", help="a suffix for the output files")
    args = ap.parse_args(argv)
    overrides = {}
    for spec in args.proxy:
        pr, _, rest = spec.partition("=")
        col, _, sg = rest.partition(":")
        overrides[pr] = (col, int(sg or 1))
    OUT.mkdir(parents=True, exist_ok=True)
    images = refresh_proxies(pd.read_parquet(ROOT / "sample/images.parquet"))
    summs, tables = [], []
    for name in args.probes:
        if not (ROOT / "vectors" / name / "reads.parquet").exists():
            log(f"{name}: no reads.parquet; skipped")
            continue
        res, summ = run_probe(name, images, args.n_boot, args.seed, args.subset,
                              overrides.get(name))
        res.to_csv(OUT / f"{name}_{args.subset}{args.tag}.csv", index=False)
        summs.append(summ)
        tables.append(res)
        log(json.dumps(summ))
    (OUT / f"summary_{args.subset}{args.tag}.json").write_text(json.dumps(summs, indent=2))
    if summs:
        md = ["| Probe | Proxy | Read | agree(c) | n ans | n ref | rho answered | rho refused [95% CI] | ratio | Verdict | c ans / ref |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in summs:
            md.append(f"| {s['probe']} | {s['proxy']} | {s['primary_read']} | {s['agree_with_c_answered']:.2f} | "
                      f"{s['n_answered']} | {s['n_refused']} | "
                      f"{s['rho_answered']:.2f} | {s['rho_refused']:.2f} "
                      f"[{s['ci_refused'][0]:.2f}, {s['ci_refused'][1]:.2f}] | "
                      f"{s['ratio']:.2f} | {s['verdict']} | {s['c']['answered']} / {s['c']['refused']} |")
        (OUT / f"divergence_{args.subset}{args.tag}.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
