"""Layer 47 by component (direct logit attribution), and the linear probe on h46 and h47.

Reads what `monocle/probe_lastblock_parts.py` writes for 1 model (base
or IT).

Part A, direct logit attribution. At the last token position

    rd47(w) = s * (rd_h46(w) + rd_attn(w) + rd_mlp(w))

so the gap of a word between the 2 groups of its sample splits into 3
additive terms. Each term is the difference of the group means (expected
group minus the other), over the pooled SD of the sum. `gap_sum` is the
total; the 3 parts add to it. `auc_*` is the image-level AUC of the
cumulative sum: `h46`, then `h46 + attn`, then all 3.

Part B, the linear probe. A linear probe decodes the rich-poor split of
the class sample from the residual stream, with whole neighborhoods held
out. If the probe decodes layer 47 as well as layer 46 while the
logit-lens association falls, the information stays linearly decodable
and the output suppresses it. If the probe falls too, the information is
not linearly decodable at layer 47. A second probe predicts log median
income over all 98 neighborhoods of the sample (ridge, neighborhoods
held out).

    .venv-vllm025cu129/bin/python scripts/probe_lastblock_parts_test.py \\
        --root outputs/_monocle/probe_lastblock_parts/it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from monocle.probe_words import WORDS  # noqa: E402
from scripts.probe_group_test import auc_u, cluster_draws  # noqa: E402

SAMPLE = REPO / "outputs/_monocle/probe_nta/sample/images.parquet"
PARTS = ("h46", "attn", "mlp", "h47")
TABLE_WORDS = ["poor", "ghetto", "slum", "rich", "wealthy", "posh", "smart", "immigrant",
               "prostitute", "trustworthy", "luxury", "upscale", "educated", "dirty", "graffiti"]


def log(m: str) -> None:
    print(f"[lastblock-parts-test] {m}", flush=True)


def load_part(root: Path, part: str) -> pd.DataFrame:
    fs = sorted((root / part).glob("words_shard*.parquet"))
    if not fs:
        raise SystemExit(f"no shards under {root / part}")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def cluster_ci(v, is_exp, ntas, rng, n_boot):
    pos = [np.where(is_exp & (ntas == k))[0] for k in sorted(set(ntas[is_exp]))]
    neg = [np.where(~is_exp & (ntas == k))[0] for k in sorted(set(ntas[~is_exp]))]
    boot = (cluster_draws(pos, n_boot, rng), cluster_draws(neg, n_boot, rng))
    bs = np.array([auc_u(v[pi], v[ni]) for pi, ni in zip(*boot)])
    return [float(q) for q in np.nanpercentile(bs, [2.5, 97.5])]


def part_a(root: Path, images: pd.DataFrame, rng, n_boot: int) -> pd.DataFrame:
    P = {p: load_part(root, p).merge(images, on="image_path", how="left") for p in PARTS}
    base = P["h46"]
    rows = []
    for w, (probe, expect) in WORDS.items():
        col = f"lp_{w}"
        if col not in base:
            continue
        keep = (base[f"draw_{probe}"].fillna(False).astype(bool)
                & base[f"group_{probe}"].isin(["top", "bottom"])).to_numpy()
        if keep.sum() < 40:
            continue
        is_exp = (base.loc[keep, f"group_{probe}"] == expect).to_numpy()
        ntas = base.loc[keep, "nta2020"].to_numpy()
        v = {p: P[p].loc[keep, col].to_numpy(float) for p in PARTS}
        total = v["h46"] + v["attn"] + v["mlp"]
        sd = np.sqrt((total[is_exp].var(ddof=1) + total[~is_exp].var(ddof=1)) / 2)
        rec = {"word": w, "probe": probe, "expect": expect, "n": int(keep.sum())}
        for p in ("h46", "attn", "mlp"):
            rec[f"gap_{p}"] = float((v[p][is_exp].mean() - v[p][~is_exp].mean()) / sd)
        rec["gap_sum"] = float((total[is_exp].mean() - total[~is_exp].mean()) / sd)
        for name, x in (("h46", v["h46"]), ("h46_attn", v["h46"] + v["attn"]),
                        ("h46_mlp", v["h46"] + v["mlp"]), ("sum", total), ("h47", v["h47"])):
            rec[f"auc_{name}"] = auc_u(x[is_exp], x[~is_exp])
        rec["lo_h46"], rec["hi_h46"] = cluster_ci(v["h46"], is_exp, ntas, rng, n_boot)
        rec["lo_h47"], rec["hi_h47"] = cluster_ci(v["h47"], is_exp, ntas, rng, n_boot)
        rows.append(rec)
    return pd.DataFrame(rows)


def load_vectors(root: Path) -> tuple[np.ndarray, dict]:
    fs = sorted(root.glob("vectors_shard*.npz"))
    zs = [np.load(f, allow_pickle=False) for f in fs]
    paths = np.concatenate([z["image_path"] for z in zs])
    V = {p: np.concatenate([z[p] for z in zs]).astype(np.float32) for p in PARTS}
    return paths, V, zs[0]["layer_scalar"]


def part_b(root: Path, images: pd.DataFrame, rng, n_boot: int) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    paths, V, _ = load_vectors(root)
    meta = pd.DataFrame({"image_path": paths}).merge(images, on="image_path", how="left")
    feats = {"h46": V["h46"], "h46+attn": V["h46"] + V["attn"], "h47": V["h47"]}
    rows = []
    # 1. the class split, neighborhoods held out
    keep = (meta["draw_class"].fillna(False).astype(bool)
            & meta["group_class"].isin(["top", "bottom"])).to_numpy()
    y = (meta.loc[keep, "group_class"] == "top").to_numpy().astype(int)
    g = meta.loc[keep, "nta2020"].to_numpy()
    for name, X in feats.items():
        Xk = X[keep]
        score = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=10).split(Xk, y, g):
            if len(set(y[tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
            clf.fit(Xk[tr], y[tr])
            score[te] = clf.decision_function(Xk[te])
        ok = np.isfinite(score)
        is_exp = (y == 1) & ok
        auc = auc_u(score[is_exp], score[(y == 0) & ok])
        lo, hi = cluster_ci(np.where(ok, score, 0.0), y == 1, g, rng, n_boot)
        rows.append({"probe": "class split (AUC, NTAs held out)", "features": name,
                     "value": auc, "lo": lo, "hi": hi, "n": int(ok.sum())})
        log(f"class split, {name}: AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")
    # 2. log median income over every neighborhood of the sample
    ok = meta["median_income"].notna().to_numpy() & (meta["median_income"] > 0).to_numpy()
    yi = np.log(meta.loc[ok, "median_income"].to_numpy(float))
    gi = meta.loc[ok, "nta2020"].to_numpy()
    for name, X in feats.items():
        Xo = X[ok]
        pred = np.full(len(yi), np.nan)
        for tr, te in GroupKFold(n_splits=10).split(Xo, yi, gi):
            reg = make_pipeline(StandardScaler(), Ridge(alpha=3e4))
            reg.fit(Xo[tr], yi[tr])
            pred[te] = reg.predict(Xo[te])
        t = pd.DataFrame({"nta": gi, "pred": pred, "y": yi}).groupby("nta").mean()
        rho = float(spearmanr(t.pred, t.y).correlation)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(t), len(t))
            bs.append(spearmanr(t.pred.to_numpy()[i], t.y.to_numpy()[i]).correlation)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append({"probe": "log income (Spearman over held-out NTAs)", "features": name,
                     "value": rho, "lo": float(lo), "hi": float(hi), "n": int(len(t))})
        log(f"income, {name}: rho {rho:.3f} [{lo:.3f}, {hi:.3f}] over {len(t)} NTAs")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    images = pd.read_parquet(SAMPLE)
    rng = np.random.default_rng(args.seed)
    out = root / "parts_test"
    out.mkdir(parents=True, exist_ok=True)
    A = part_a(root, images, rng, args.n_boot)
    A.to_csv(out / "parts_gap.csv", index=False)
    geo = pd.concat([pd.read_parquet(f) for f in sorted(root.glob("geometry_shard*.parquet"))])
    _, _, scal = load_vectors(root)
    lines = [f"# Layer 47 by component, direct logit attribution: {root.name}", "",
             f"`layer_scalar` of layer 47: {float(scal[47]):.5f}. Median norms at the last token "
             f"position: h46 {geo.n_h46.median():.1f}, attn {geo.n_attn.median():.1f}, "
             f"mlp {geo.n_mlp.median():.1f}, h47 {geo.n_h47.median():.1f}. Median cosines with h46: "
             f"attn {geo.cos_attn_h46.median():+.2f}, mlp {geo.cos_mlp_h46.median():+.2f}, "
             f"h47 {geo.cos_h47_h46.median():+.2f}; attn with mlp {geo.cos_attn_mlp.median():+.2f}.", "",
             "## The gap of a word, split into the 3 direct-logit-attribution terms", "",
             "Units: the pooled SD of the sum. `gap_sum` = `gap_h46` + `gap_attn` + `gap_mlp`. "
             "`task` = the contrast task whose sample gives the 2 groups.", "",
             "| word | task | gap_h46 | gap_attn | gap_mlp | gap_sum | AUC h46 | AUC h46+attn | AUC h46+mlp | AUC sum |",
             "|------|------|--------:|---------:|--------:|--------:|--------:|-------------:|------------:|--------:|"]
    order = [w for w in TABLE_WORDS if w in set(A.word)] + [w for w in A.word if w not in TABLE_WORDS]
    for r in A.set_index("word").loc[order].reset_index().itertuples():
        lines.append(f"| `{r.word}` | {r.probe} | {r.gap_h46:+.2f} | {r.gap_attn:+.2f} | {r.gap_mlp:+.2f} | "
                     f"{r.gap_sum:+.2f} | {r.auc_h46:.2f} | {r.auc_h46_attn:.2f} | {r.auc_h46_mlp:.2f} | {r.auc_sum:.2f} |")
    if not args.skip_probe:
        B = part_b(root, images, rng, args.n_boot)
        B.to_csv(out / "probe.csv", index=False)
        lines += ["", "## Linear probe on h46 and h47 (held-out neighborhoods)", "",
                  "| probe | features | value | 95% interval | n |", "|------|----------|------:|--------------|--:|"]
        for r in B.itertuples():
            lines.append(f"| {r.probe} | {r.features} | {r.value:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | {r.n} |")
    (out / "parts_test.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {out}/parts_test.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
