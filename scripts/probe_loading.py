"""E5 of the lens-refusal probe: what the probe vectors load on.

Two tables:

1. The cosine matrix at block 42 between the 5 probe contrast vectors and
   the 7 case contrast vectors. Both sets are pulled back under
   `mm_all47` (the case vectors were built under it), so the probe side
   takes its `47` vector.
2. The per-image Spearman of a case read with a probe read on the same
   images: the case reads come from `monocle.pole_score` on the probe
   images (`outputs/_monocle/probe/case_reads/<case>/`), the probe reads
   from `monocle.probe_vectors` (`read_47_L42` and `read_C_L42`).

There is no go rule. This is the finding. Writes
`outputs/_monocle/probe/loading/{cosines.csv,per_image.csv,loading.md}`.

    .venv-3.12/bin/python scripts/probe_loading.py
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
from monocle.probe_frame import PROBE_ORDER, PROBES  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"
CASE_VEC = REPO / "outputs/_monocle/pole_sample"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
CASE_WORDS = {"subway_safety": "safe/dangerous", "libraries": "maintained/worn",
              "schools": "good/bad", "road_quality": "smooth/damaged",
              "parks_plazas": "maintained/worn", "restaurants": "delicious/gross",
              "street_photography": "beautiful/ugly"}


def log(m: str) -> None:
    print(f"[probe-loading] {m}", flush=True)


def load_case_reads(case: str, layer: int) -> pd.DataFrame | None:
    parts = sorted((ROOT / "case_reads" / case).glob("scores_shard*of*.parquet"))
    if not parts:
        return None
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df[df["layer"] == layer].copy()
    df[f"case_read"] = df["raw_contrast"] - df["raw_contrast"].mean()
    return df[["image_path", "case_read", "c"]].rename(columns={"c": "case_c"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=42)
    args = ap.parse_args(argv)
    out = ROOT / "loading"
    out.mkdir(parents=True, exist_ok=True)
    L = args.layer

    probes = [p for p in PROBE_ORDER if (ROOT / "vectors" / p / "vectors.npz").exists()]
    pv = {p: np.load(ROOT / "vectors" / p / "vectors.npz")[f"v_contrast_47_L{L}"] for p in probes}
    cv = {c: np.load(CASE_VEC / c / "vectors.npz")[f"v_contrast_L{L}"] for c in CASES}
    rows = []
    for p in probes:
        for c in CASES:
            rows.append({"probe": p, "case": c,
                         "cos": float(pv[p] @ cv[c] / (np.linalg.norm(pv[p]) * np.linalg.norm(cv[c])))})
    for p in probes:
        for q in probes:
            rows.append({"probe": p, "case": f"probe:{q}",
                         "cos": float(pv[p] @ pv[q] / (np.linalg.norm(pv[p]) * np.linalg.norm(pv[q])))})
    cos = pd.DataFrame(rows)
    cos.to_csv(out / "cosines.csv", index=False)

    # The read of record of each probe, as E3 chose it (the best agreement
    # with c on the answered images); the planned C L42 and c go beside it.
    primary = {}
    sj = ROOT / "divergence" / "summary_union.json"
    if sj.exists():
        primary = {d["probe"]: d["primary_read"] for d in json.loads(sj.read_text())}
    rows = []
    for p in probes:
        reads = pd.read_parquet(ROOT / "vectors" / p / "reads.parquet")
        reads = reads[~reads["calib"]]
        prim = primary.get(p, f"read_C_L{L}")
        reads["read_primary"] = reads[prim]
        for c in CASES:
            cr = load_case_reads(c, L)
            if cr is None:
                continue
            m = reads.merge(cr, on="image_path", how="inner")
            for read in ("read_primary", f"read_47_L{L}", f"read_C_L{L}", "c"):
                for split, mask in (("all", pd.Series(True, index=m.index)),
                                    ("answered", m["answered"]),
                                    ("refused", ~m["answered"])):
                    g = m[mask]
                    if len(g) < 30:
                        continue
                    rows.append({"probe": p, "case": c, "probe_read": read, "split": split,
                                 "n": int(len(g)),
                                 "rho_case_read": float(spearmanr(g[read], g["case_read"]).statistic),
                                 "rho_case_c": float(spearmanr(g[read], g["case_c"]).statistic)})
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_image.csv", index=False)

    md = ["## Cosines at block %d (both sides under mm_all47)" % L, "",
          "| Probe | " + " | ".join(f"{c} ({CASE_WORDS[c]})" for c in CASES) + " |",
          "|---|" + "---|" * len(CASES)]
    for p in probes:
        vals = [cos[(cos.probe == p) & (cos.case == c)]["cos"].iloc[0] for c in CASES]
        md.append(f"| {p} ({PROBES[p].pos}/{PROBES[p].neg}) | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    md += ["", "| Probe | " + " | ".join(probes) + " |", "|---|" + "---|" * len(probes)]
    for p in probes:
        vals = [cos[(cos.probe == p) & (cos.case == f"probe:{q}")]["cos"].iloc[0] for q in probes]
        md.append(f"| {p} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    if len(per):
        md += ["", "## Per-image Spearman, the probe read of record against the case read, all images", "",
               "Read of record: " + ", ".join(f"{p} = {primary.get(p, f'read_C_L{L}')}" for p in probes), "",
               "| Probe | " + " | ".join(CASES) + " |", "|---|" + "---|" * len(CASES)]
        for p in probes:
            vals = []
            for c in CASES:
                g = per[(per.probe == p) & (per.case == c) & (per.probe_read == "read_primary") & (per.split == "all")]
                vals.append(f"{g['rho_case_read'].iloc[0]:+.2f}" if len(g) else "-")
            md.append(f"| {p} | " + " | ".join(vals) + " |")
        md += ["", "## Per-image Spearman, the probe read of record against the case read, REFUSED images", "",
               "| Probe | " + " | ".join(CASES) + " |", "|---|" + "---|" * len(CASES)]
        for p in probes:
            vals = []
            for c in CASES:
                g = per[(per.probe == p) & (per.case == c) & (per.probe_read == "read_primary") & (per.split == "refused")]
                vals.append(f"{g['rho_case_read'].iloc[0]:+.2f} (n={int(g['n'].iloc[0])})" if len(g) else "-")
            md.append(f"| {p} | " + " | ".join(vals) + " |")
    (out / "loading.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
