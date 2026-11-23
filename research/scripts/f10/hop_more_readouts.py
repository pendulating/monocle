#!/usr/bin/env python
"""F10.30: the readout analyses of F10.27 to F10.29 on the 10-image summaries. CPU only.

The summaries of `hop_more_summary.py` (`data/f10/prompts_k10/`) have the F10.26 layout, thus this script runs the
F10.27 fit, the F10.29 transfer and the F10.28 ACS test of the shared direction on them, with the roots of those
modules changed at run time. The F10.27 directions of the 591 images (`research/figures/f10/shared/`) stay as they are, and
the 10-image results go to `research/figures/f10/more/`.

| Stage | What it does | Output |
|-------|--------------|--------|
| `agree` | For each model and prompt, the r over NTAs of the projection of the NTA mean readout on the F10.27 direction (test layer), k = 3 against k = 10; and the r of the k = 10 projection with the k = 10 NTA margin, against the F10.27 held-out r | `agree.csv` |
| `transfer` | The F10.29 transfer (the direction of 2 models reads the third), with the k = 10 NTA means as the target | `transfer/` |
| `ses` | The F10.28 ACS test of the shared direction and the yes shares, on the k = 10 NTA means | `ses_readout.csv` |
| `fit` | The F10.27 fit of the shared direction on the k = 10 summaries (4 fit prompts, 10 held out) | `shared/{model}_place/` |
| `words` | The F10.28 word lists (the layer mean of the z-scored direction) from the k = 10 fit, against the k = 3 lists | `words.csv`, `words_agree.csv` |
| `dirs` | The F10.29 task-7 residual directions (`hop_steer.py dirs`) on the k = 10 residuals, against the 591-image directions | `steer/{model}/`, `dirs_agree.csv` |
| `project` | The projections of the 1,970 `cool place` residuals on the 2 directions of each cell: their r, and the r of each with the margin on the 1,379 new images and on the 591 old ones | `dirs_projection.csv` |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_readouts.py agree
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_readouts.py transfer
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_readouts.py ses
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_readouts.py fit --model gemma-4-31b
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_readouts.py words
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_readouts.py dirs --model gemma-4-31b
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_readouts.py project
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10")]
from research.scripts.f10.hop_prompts import MODELS, PROMPTS, store_name  # noqa: E402

__version__ = "1.1.0"
K3 = ROOT / "data/f10/prompts"
K10 = ROOT / "data/f10/prompts_k10"
SHARED = ROOT / "research/figures/f10/shared"
OUT = ROOT / "research/figures/f10/more"


def complete(root: Path, model: str) -> list[tuple[str, str]]:
    have = []
    for a, n in PROMPTS:
        meta = root / store_name(model, a, n) / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("complete"):
            have.append((a, n))
    return have


def nta_margin(root: Path, model: str, a: str, n: str) -> pd.Series:
    f = pd.read_parquet(root / store_name(model, a, n) / "frame.parquet")
    return f.groupby("codes")["margin"].mean()


def agree() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        s = json.loads((SHARED / f"{model}_place" / "summary.json").read_text())
        d = np.load(SHARED / f"{model}_place" / "direction.npz")
        layers = d["layers"].tolist()
        b = layers.index(s["test_layer"])
        u = d["direction"][b].astype(np.float64)
        held = pd.read_csv(SHARED / f"{model}_place" / "heldout.csv")
        held = held[(held.variant == "main") & (held.layer == s["test_layer"])].set_index("prompt")["r"]
        for a, n in complete(K10, model):
            name = store_name(model, a, n)
            x3 = np.load(K3 / name / "nta_place.npy")[:, b].astype(np.float64)
            x10 = np.load(K10 / name / "nta_place.npy")[:, b].astype(np.float64)
            p3, p10 = (x3 - x3.mean(0)) @ u, (x10 - x10.mean(0)) @ u
            m3, m10 = nta_margin(K3, model, a, n), nta_margin(K10, model, a, n)
            ok10 = np.isfinite(m10.reindex(range(len(p10))).to_numpy())
            ok3 = np.isfinite(m3.reindex(range(len(p3))).to_numpy())
            r_proj = stats.pearsonr(p3, p10)
            r10 = stats.pearsonr(p10[ok10], m10.reindex(range(len(p10))).to_numpy()[ok10])
            r3 = stats.pearsonr(p3[ok3], m3.reindex(range(len(p3))).to_numpy()[ok3])
            r_margin = stats.pearsonr(m3.reindex(range(len(p3))).to_numpy()[ok3 & ok10], m10.reindex(range(len(p10))).to_numpy()[ok3 & ok10])
            rows.append({"model": model, "noun": n, "adjective": a, "prompt": f"{n}-{a}", "test_layer": s["test_layer"],
                         "r_projection_k3_k10": float(r_proj[0]), "r_margin_k3_k10": float(r_margin[0]),
                         "r_direction_margin_k3": float(r3[0]), "r_direction_margin_k10": float(r10[0]), "p_k10": float(r10[1]),
                         "n_nta_k10": int(ok10.sum()), "heldout_r_f1027": float(held.get(f"{n}-{a}", np.nan))})
    df = pd.DataFrame(rows)
    df["p_k10_bonferroni"] = (df.p_k10 * df.groupby("model").p_k10.transform("size")).clip(upper=1.0)
    return df


def transfer() -> None:
    from research.scripts.f10 import hop_transfer as ht

    ht.DATA, ht.OUT = K10, OUT / "transfer"
    ht.main()


def ses() -> pd.DataFrame:
    from research.scripts.f10 import hop_hierarchy as hh

    hh.DATA, hh.TRACES = K10, ()
    return hh.ses()


def fit(model: str) -> None:
    from research.scripts.f10 import hop_shared as hs

    hs.DATA, hs.FIG = K10, OUT / "shared"
    hs.fit(model, "place")


def words() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The layer-mean word lists of the k = 10 direction, with the rank of each word in the k = 3 list."""
    from research.scripts.f10 import hop_hierarchy as hh

    n, rows, agree = hh.N_WORDS, [], []
    for model in MODELS:
        hh.SHARED = SHARED
        v3, w3 = hh.layer_mean(model)
        hh.SHARED = OUT / "shared"
        v10, w10 = hh.layer_mean(model)
        assert (v3 == v10).all()
        rank3 = pd.Series(np.empty(len(w3)), index=v3)
        rank3[:] = 0
        o3, o10 = np.argsort(-w3), np.argsort(-w10)
        rho = stats.spearmanr(w3, w10)[0]
        for end, i3, i10 in (("up", o3, o10), ("down", o3[::-1], o10[::-1])):
            pos3 = {v3[i]: k + 1 for k, i in enumerate(i3[:500])}
            for k, i in enumerate(i10[:n]):
                rows.append({"model": model, "end": end, "rank": k + 1, "word": v10[i], "z_k10": float(w10[i]), "z_k3": float(w3[i]),
                             "rank_k3": pos3.get(v10[i], np.nan), "in_k3_top20": v10[i] in set(v3[i3[:n]])})
            agree.append({"model": model, "end": end, "shared_of_20": len(set(v3[i3[:n]]) & set(v10[i10[:n]])),
                          "shared_of_100": len(set(v3[i3[:100]]) & set(v10[i10[:100]])), "rho_layer_mean_all_words": float(rho)})
    return pd.DataFrame(rows), pd.DataFrame(agree)


def dirs(model: str) -> pd.DataFrame:
    """The task-7 residual directions on the k = 10 residuals, and their cosine with the 591-image directions."""
    from research.scripts.f10 import hop_prompts as hp
    from research.scripts.f10 import hop_steer as hst

    hp.DATA, hst.OUT = K10, OUT / "steer"
    hst.dirs(model)
    d3, d10 = np.load(ROOT / "data/f10/steer" / model / "dirs.npz"), np.load(OUT / "steer" / model / "dirs.npz")
    j3, j10 = json.loads((ROOT / "data/f10/steer" / model / "dirs.json").read_text()), json.loads((OUT / "steer" / model / "dirs.json").read_text())
    rows = []
    for key in d10.files:
        if key == "control_u":
            continue
        pos = key.split("_")[0]
        layer = int(key[len("place_L"):-2]) if key.startswith("place_L") else j10[pos]["layer"]
        c3, c10 = (j3["place_layers"][str(layer)], j10["place_layers"][str(layer)]) if pos == "place" else (j3[pos], j10[pos])
        rows.append({"model": model, "position": pos, "layer": layer, "cosine_k3_k10": float(d3[key].astype(np.float64) @ d10[key].astype(np.float64)),
                     "sigma_k3": c3["sigma"], "sigma_k10": c10["sigma"], "share_sv1_k3": c3["share_sv1"], "share_sv1_k10": c10["share_sv1"],
                     "loading_nice_k3": c3["loadings"]["place nice"], "loading_nice_k10": c10["loadings"]["place nice"],
                     "loading_ugly_k3": c3["loadings"]["place ugly"], "loading_ugly_k10": c10["loadings"]["place ugly"]})
    return pd.DataFrame(rows).drop_duplicates(["position", "layer"]).sort_values(["position", "layer"])


def project() -> pd.DataFrame:
    """Compare the 591-image and the 1,970-image residual directions by what they read, not by their cosine."""
    rows = []
    for m in MODELS:
        d3, d10 = np.load(ROOT / "data/f10/steer" / m / "dirs.npz"), np.load(OUT / "steer" / m / "dirs.npz")
        j10 = json.loads((OUT / "steer" / m / "dirs.json").read_text())
        name = store_name(m, "cool", "place")
        fr10, fr3 = pd.read_parquet(K10 / name / "frame.parquet"), pd.read_parquet(K3 / name / "frame.parquet")
        new = ~fr10.image_idx.isin(set(fr3.image_idx)).to_numpy()
        layers = json.loads((K10 / name / "meta.json").read_text())["layers"]
        y = fr10.margin.to_numpy()
        ok = np.isfinite(y)
        for pos in ("place", "final"):
            R = np.load(K10 / name / f"residual_{pos}.npy")
            for key in [k for k in d10.files if k.startswith(pos)]:
                layer = int(key[len("place_L"):-2]) if key.startswith("place_L") else j10[pos]["layer"]
                X = R[:, layers.index(layer)].astype(np.float64)
                X -= X.mean(0)
                p3, p10 = X @ d3[key].astype(np.float64), X @ d10[key].astype(np.float64)
                r = lambda a, b, sel: float(stats.pearsonr(a[sel], b[sel])[0])  # noqa: E731
                rows.append({"model": m, "position": pos, "layer": layer, "cosine_k3_k10": float(d3[key] @ d10[key]),
                             "r_projection_1970": r(p3, p10, slice(None)), "r_projection_new_1379": r(p3, p10, new),
                             "r_u3_margin_new": r(p3, y, new & ok), "r_u10_margin_new": r(p10, y, new & ok),
                             "r_u3_margin_old_591": r(p3, y, ~new & ok), "r_u10_margin_old_591": r(p10, y, ~new & ok)})
            del R
    return pd.DataFrame(rows).drop_duplicates(["model", "position", "layer"]).sort_values(["model", "position", "layer"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["agree", "transfer", "ses", "fit", "words", "dirs", "project"])
    ap.add_argument("--model", choices=MODELS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    if args.stage == "agree":
        df = agree()
        df.to_csv(OUT / "agree.csv", index=False)
        print(df.round(2).to_string())
        print(df.groupby("model")[["r_projection_k3_k10", "r_margin_k3_k10", "r_direction_margin_k3", "r_direction_margin_k10", "heldout_r_f1027"]].median().round(2).to_string())
    elif args.stage == "transfer":
        transfer()
    elif args.stage == "ses":
        df = ses()
        df.to_csv(OUT / "ses_readout.csv", index=False)
        print(df.pivot_table(index=["model", "measure"], columns="feature", values="rho").round(2).to_string())
        print(df.pivot_table(index=["model", "measure"], columns="feature", values="p_bonferroni").round(3).to_string())
    elif args.stage == "words":
        w, a = words()
        w.to_csv(OUT / "words.csv", index=False)
        a.to_csv(OUT / "words_agree.csv", index=False)
        print(a.round(2).to_string())
        for (m, e), sub in w.groupby(["model", "end"], sort=False):
            print(m, e, ", ".join(f"{r.word}{'' if r.in_k3_top20 else '*'}" for r in sub.itertuples()))
    elif args.stage == "dirs":
        df = dirs(args.model)
        f = OUT / "dirs_agree.csv"
        if f.exists():
            old = pd.read_csv(f)
            df = pd.concat([old[old.model != args.model], df])
        df.to_csv(f, index=False)
        print(df.round(3).to_string())
    elif args.stage == "project":
        df = project()
        df.to_csv(OUT / "dirs_projection.csv", index=False)
        print(df.round(3).to_string())
    else:
        fit(args.model)


if __name__ == "__main__":
    main()
