"""Score a `monocle.steer_coord` run: the natural coordinate and the maps.

Two questions, per case and per layer.

1. Natural coordinate. Across the eval images, does ``s_l = <h_l[answer],
   v_l>`` track the model's own ``logit_diff``? The report gives the
   correlation, the regression slope (logits per unit norm along ``v_l``),
   and the steering gain from the sweep at the same layer. When the 2
   slopes agree, the images' own variation along ``v_l`` carries the
   judgment at the gain the injection showed.

2. Maps. For each image, the 256 cell removals give a change in ``c`` and a
   change in ``h_l[answer]``. The report projects the second onto ``v_l``
   and compares the 2 maps cell by cell: the per-image correlation, the
   sign agreement, and the top-cell overlap. The matched random control
   direction and a set of fresh random directions give the null.

Usage:
    python research/scripts/steer_coord_report.py --cases all
    python research/scripts/steer_coord_report.py --cases restaurants --figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle.steer_coord import coordinate, contribution_sign, delta_map  # noqa: E402

ROOT_DEFAULT = REPO / "outputs/_monocle/steer_coord"
SWEEP_DEFAULT = REPO / "outputs/_monocle/single_steer_fine"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
#: The steering gain is read from this dose pair. Both doses sit inside
#: every case's on-contract window.
GAIN_ALPHAS = (-0.5, 0.5)
N_RANDOM = 20
TOP_SHARE = 0.05


def log(m: str) -> None:
    print(f"[steer-coord-report] {m}", flush=True)


def steering_gain(sweep: pd.DataFrame, layer: int, scale: float) -> dict:
    """Logits per unit residual norm along ``v_l``, from the paired doses.

    For each image, ``(ld(+a) - ld(-a)) / (2 a scale)`` with the signal
    direction; the mean over images and its standard error. The control
    gets the same read.
    """
    out = {}
    lo, hi = GAIN_ALPHAS
    for direction in ("signal", "control"):
        g = sweep[(sweep["layer"] == layer) & (sweep["direction"] == direction)]
        a = g[g["alpha"] == lo].set_index("image_path")["logit_diff"]
        b = g[g["alpha"] == hi].set_index("image_path")["logit_diff"]
        idx = a.index.intersection(b.index)
        if len(idx) == 0:
            out[direction] = {"gain": float("nan"), "se": float("nan"), "n": 0}
            continue
        slope = (b.loc[idx] - a.loc[idx]) / ((hi - lo) * scale)
        out[direction] = {"gain": float(slope.mean()),
                          "se": float(slope.std(ddof=1) / np.sqrt(len(idx))),
                          "n": int(len(idx))}
    return out


def natural_report(
    case_dir: Path, sweep: Optional[pd.DataFrame], vecs: dict,
    layers: list[int], rng: np.random.Generator,
) -> list[dict]:
    nat = pd.read_parquet(case_dir / "natural.parquet")
    acts = np.load(case_dir / "natural_acts.npy").astype(np.float32)
    y = nat["logit_diff"].to_numpy(dtype=np.float64)
    rows = []
    for j, l in enumerate(layers):
        h = acts[:, j, :]
        hc = h - h.mean(0, keepdims=True)
        row = {"case": case_dir.name, "layer": l, "n": len(y),
               "sd_logit_diff": float(y.std(ddof=1))}
        for name in ("signal", "control"):
            s = coordinate(hc, vecs[f"v_{name}_L{l}"])
            r, p = sps.pearsonr(s, y)
            rho, _ = sps.spearmanr(s, y)
            slope = float(np.polyfit(s, y, 1)[0])
            row.update({f"r_{name}": float(r), f"p_{name}": float(p),
                        f"rho_{name}": float(rho),
                        f"slope_{name}": slope,
                        f"sd_s_{name}": float(s.std(ddof=1))})
        # A null over fresh unit directions in the same residual space.
        rs = []
        for _ in range(N_RANDOM):
            u = rng.standard_normal(h.shape[1])
            u /= np.linalg.norm(u)
            rs.append(abs(sps.pearsonr(coordinate(hc, u), y)[0]))
        row["abs_r_random_p95"] = float(np.percentile(rs, 95))
        if sweep is not None:
            gain = steering_gain(sweep, l, float(vecs["scales"][j]))
            row["gain_signal"] = gain["signal"]["gain"]
            row["gain_signal_se"] = gain["signal"]["se"]
            row["gain_control"] = gain["control"]["gain"]
            # The spread of the baseline that the natural coordinate would
            # explain at the steering gain, against the spread observed.
            row["predicted_sd"] = abs(gain["signal"]["gain"]) * row["sd_s_signal"]
            row["predicted_over_observed"] = (
                row["predicted_sd"] / row["sd_logit_diff"])
        rows.append(row)
    return rows


def map_report(
    case_dir: Path, vecs: dict, layers: list[int], rng: np.random.Generator,
) -> tuple[list[dict], pd.DataFrame]:
    maps = pd.read_parquet(case_dir / "maps.parquet")
    stats = pd.read_parquet(case_dir / "map_stats.parquet")
    acts = np.load(case_dir / "answer_acts.npy", mmap_mode="r")
    n = min(len(maps), acts.shape[0])
    c_maps = np.stack([np.asarray(v, dtype=np.float64)
                       for v in maps["values"].iloc[:n]])
    n_cells = c_maps.shape[1]
    k = max(1, round(TOP_SHARE * n_cells))
    rows, per_image = [], []
    randoms = [rng.standard_normal(acts.shape[-1]) for _ in range(N_RANDOM)]
    randoms = [u / np.linalg.norm(u) for u in randoms]
    for j, l in enumerate(layers):
        a = np.asarray(acts[:n, :, j, :], dtype=np.float32)
        dirs = {"signal": vecs[f"v_signal_L{l}"],
                "control": vecs[f"v_control_L{l}"]}
        per_dir: dict[str, dict[str, list]] = {}
        for name, v in dirs.items():
            rs, signs, tops, slopes = [], [], [], []
            for i in range(n):
                s_map = contribution_sign(delta_map(a[i], v))
                c_map = c_maps[i]
                r = sps.pearsonr(s_map, c_map)[0]
                rs.append(r)
                signs.append(float(np.mean(np.sign(s_map) == np.sign(c_map))))
                top_c = set(np.argsort(np.abs(c_map))[-k:])
                top_s = set(np.argsort(np.abs(s_map))[-k:])
                tops.append(len(top_c & top_s) / k)
                slopes.append(float(np.polyfit(s_map, c_map, 1)[0]))
                per_image.append({
                    "case": case_dir.name, "layer": l, "direction": name,
                    "image_path": maps["image_path"].iloc[i],
                    "c": float(stats["c"].iloc[i]),
                    "image_worth": float(stats["image_worth"].iloc[i]),
                    "r": float(r), "sign_agree": signs[-1],
                    "top_overlap": tops[-1], "slope": slopes[-1],
                    "sum_s": float(s_map.sum()), "sum_c": float(c_map.sum()),
                })
            per_dir[name] = {"r": rs, "sign": signs, "top": tops,
                             "slope": slopes}
        # The random null: the median r over images, for each direction.
        null_medians = []
        for u in randoms:
            rr = [sps.pearsonr(
                contribution_sign(delta_map(a[i], u)), c_maps[i])[0]
                for i in range(n)]
            null_medians.append(abs(float(np.median(rr))))
        # A pooled slope: every cell of every image at once.
        s_all = np.concatenate([
            contribution_sign(delta_map(a[i], dirs["signal"])) for i in range(n)])
        c_all = c_maps[:n].ravel()
        pooled = float(np.polyfit(s_all, c_all, 1)[0])
        pooled_r = float(sps.pearsonr(s_all, c_all)[0])
        rows.append({
            "case": case_dir.name, "layer": l, "n_images": n,
            "median_r_signal": float(np.median(per_dir["signal"]["r"])),
            "median_r_control": float(np.median(per_dir["control"]["r"])),
            "frac_r_pos_signal": float(np.mean(np.array(per_dir["signal"]["r"]) > 0)),
            "median_sign_agree_signal": float(np.median(per_dir["signal"]["sign"])),
            "median_top_overlap_signal": float(np.median(per_dir["signal"]["top"])),
            "median_top_overlap_control": float(np.median(per_dir["control"]["top"])),
            "chance_top_overlap": TOP_SHARE,
            "abs_median_r_random_p95": float(np.percentile(null_medians, 95)),
            "pooled_slope_signal": pooled, "pooled_r_signal": pooled_r,
        })
    return rows, pd.DataFrame(per_image)


def geometry_report(
    case_dir: Path, vecs: dict, layers: list[int],
) -> list[dict]:
    """How one-dimensional the answer residual is, and where ``v_l`` sits.

    Across the eval images, the centered answer residual has a first
    principal axis. The report gives its variance share, its correlation
    with the judgment, and the cosine of ``v_l`` with that axis and with
    the difference of the 2 pole means. It also gives the natural spread
    along ``v_l`` as a share of the centered norm the injection used.
    """
    nat = pd.read_parquet(case_dir / "natural.parquet")
    acts = np.load(case_dir / "natural_acts.npy").astype(np.float64)
    y = nat["logit_diff"].to_numpy(dtype=np.float64)
    pos = nat["argmax_pos"].to_numpy(dtype=bool)
    rows = []
    for j, l in enumerate(layers):
        h = acts[:, j, :]
        hc = h - h.mean(0, keepdims=True)
        _, sv, vt = np.linalg.svd(hc, full_matrices=False)
        share = sv ** 2 / np.sum(sv ** 2)
        pc1 = vt[0]
        score = hc @ pc1
        r_pc1 = float(sps.pearsonr(score, y)[0])
        v = vecs[f"v_signal_L{l}"]
        row = {"case": case_dir.name, "layer": l,
               "pc1_share": float(share[0]), "pc2_share": float(share[1]),
               "r_pc1": abs(r_pc1),
               "cos_v_pc1": abs(float(pc1 @ v)),
               "sd_along_v": float((hc @ v).std(ddof=1)),
               "sd_along_pc1": float(score.std(ddof=1)),
               "scale": float(vecs["scales"][j])}
        row["sd_v_over_scale"] = row["sd_along_v"] / row["scale"]
        if 2 <= pos.sum() <= len(pos) - 2:
            m = h[pos].mean(0) - h[~pos].mean(0)
            m /= np.linalg.norm(m)
            row["cos_v_meandiff"] = float(m @ v)
            row["cos_pc1_meandiff"] = abs(float(m @ pc1))
            row["sd_along_meandiff"] = float((hc @ m).std(ddof=1))
        rows.append(row)
    return rows


def natural_figure(case_dir: Path, vecs: dict, layers: list[int],
                   out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nat = pd.read_parquet(case_dir / "natural.parquet")
    acts = np.load(case_dir / "natural_acts.npy").astype(np.float32)
    y = nat["logit_diff"].to_numpy()
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.4),
                             squeeze=False)
    for j, l in enumerate(layers):
        ax = axes[0, j]
        h = acts[:, j, :]
        hc = h - h.mean(0, keepdims=True)
        for name, col in (("control", "#9aa5b1"), ("signal", "#d9534f")):
            s = coordinate(hc, vecs[f"v_{name}_L{l}"])
            r = sps.pearsonr(s, y)[0]
            ax.scatter(s, y, s=14, c=col, alpha=0.8, label=f"{name} r={r:+.2f}")
        ax.axhline(0, lw=0.5, c="k")
        ax.set_xlabel(f"centered coordinate along v, L{l}")
        if j == 0:
            ax.set_ylabel("logit(pos) - logit(neg)")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, transparent=True)
    plt.close(fig)


def map_figure(case_dir: Path, vecs: dict, layer: int, out: Path,
               n_show: int = 6) -> None:
    """Side by side: the photograph, the c-map, the s-map at one layer."""
    from PIL import Image

    from research.monocle import causal_maps as cm

    maps = pd.read_parquet(case_dir / "maps.parquet")
    stats = pd.read_parquet(case_dir / "map_stats.parquet")
    acts = np.load(case_dir / "answer_acts.npy", mmap_mode="r")
    j = list(vecs["layers"]).index(layer)
    v = vecs[f"v_signal_L{layer}"]
    n = min(n_show, len(maps), acts.shape[0])
    # Rank the shown images by |c|, so both poles appear.
    order = np.argsort(stats["c"].to_numpy()[:acts.shape[0]])
    pick = list(order[: n // 2]) + list(order[-(n - n // 2):])
    panels = []
    c_all, s_all = [], []
    for i in pick:
        c_map = np.asarray(maps["values"].iloc[i], dtype=np.float64)
        s_map = contribution_sign(delta_map(
            np.asarray(acts[i, :, j, :], dtype=np.float32), v))
        c_all.append(c_map)
        s_all.append(s_map)
    vmax_c = float(np.percentile(np.abs(np.concatenate(c_all)), 99))
    vmax_s = float(np.percentile(np.abs(np.concatenate(s_all)), 99))
    for i, c_map, s_map in zip(pick, c_all, s_all):
        img = Image.open(maps["image_path"].iloc[i]).convert("RGB")
        r = sps.pearsonr(s_map, c_map)[0]
        panels.append((f"c={stats['c'].iloc[i]:+.1f}", img))
        panels.append(("c-map", cm.overlay(img, c_map, vmax_c)))
        panels.append((f"s-map L{layer} r={r:+.2f}",
                       cm.overlay(img, s_map * vmax_c / vmax_s, vmax_c)))
    sheet = cm.contact_sheet(panels, vmax_c, cols=3,
                             teal_label="positive", coral_label="negative")
    sheet.save(out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    ap.add_argument("--sweep-dir", default=str(SWEEP_DEFAULT))
    ap.add_argument("--cases", default="all")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args(argv)
    root = Path(args.root)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rng = np.random.default_rng(args.seed)

    nat_rows, map_rows, per_image, geom_rows = [], [], [], []
    for case in cases:
        case_dir = root / case
        if not (case_dir / "vectors.npz").exists():
            log(f"{case}: no run yet")
            continue
        vecs = dict(np.load(case_dir / "vectors.npz"))
        layers = [int(l) for l in vecs["layers"]]
        sweep_path = Path(args.sweep_dir) / case / "steer_results.parquet"
        sweep = pd.read_parquet(sweep_path) if sweep_path.exists() else None
        if (case_dir / "natural.parquet").exists():
            nat_rows += natural_report(case_dir, sweep, vecs, layers, rng)
            geom_rows += geometry_report(case_dir, vecs, layers)
        if (case_dir / "answer_acts.npy").exists():
            rows, pi = map_report(case_dir, vecs, layers, rng)
            map_rows += rows
            per_image.append(pi)
        if args.figures:
            fig_dir = root / "research" / "figures"
            fig_dir.mkdir(exist_ok=True)
            natural_figure(case_dir, vecs, layers,
                           fig_dir / f"natural_{case}.png")
            if (case_dir / "answer_acts.npy").exists():
                for l in layers:
                    map_figure(case_dir, vecs, l,
                               fig_dir / f"maps_{case}_L{l}.png")
        log(f"{case}: scored")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    if nat_rows:
        nat = pd.DataFrame(nat_rows)
        cols = ["case", "layer", "r_signal", "rho_signal", "r_control",
                "abs_r_random_p95", "slope_signal", "gain_signal",
                "gain_control", "predicted_over_observed"]
        cols = [c for c in cols if c in nat.columns]
        print("\nNATURAL COORDINATE (per case, per layer)")
        print(nat[cols].round(3).to_string(index=False))
        nat.to_parquet(root / "natural_report.parquet", index=False)
    if geom_rows:
        ge = pd.DataFrame(geom_rows)
        print("\nGEOMETRY of the centered answer residual (per case, per layer)")
        print(ge.round(3).to_string(index=False))
        ge.to_parquet(root / "geometry_report.parquet", index=False)
    if map_rows:
        mp = pd.DataFrame(map_rows)
        cols = ["case", "layer", "n_images", "median_r_signal",
                "median_r_control", "abs_median_r_random_p95",
                "frac_r_pos_signal", "median_sign_agree_signal",
                "median_top_overlap_signal", "median_top_overlap_control",
                "pooled_r_signal", "pooled_slope_signal"]
        print("\nMAPS: c-map against s-map (per case, per layer)")
        print(mp[cols].round(3).to_string(index=False))
        mp.to_parquet(root / "map_report.parquet", index=False)
        pd.concat(per_image).to_parquet(root / "map_per_image.parquet",
                                        index=False)
    (root / "report.json").write_text(json.dumps(
        {"natural": nat_rows, "geometry": geom_rows, "maps": map_rows}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
