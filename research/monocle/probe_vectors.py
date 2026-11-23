"""E2 of the lens-refusal probe: the contrast vectors and the lens reads.

For 1 probe:

1. Take the first `--n-calib` images of the E1 census that ANSWER (the top
   token is a pole). One forward each; the gradient of each pole logit
   with respect to the final residual, averaged, gives `d_pos`, `d_neg`,
   and the contrast `d = d_pos - d_neg`.
2. Pull `d` back to blocks 41 to 45 with `pinv(J_l)` under 2 lenses:
   `C` (`mm_textafter`, the question-position lens the gate chose) and
   `47` (`mm_all47`, the lens of the CVPR read). The vectors go to
   `vectors.npz`.
3. Project the saved E1 residual of EVERY image on each vector, center
   over the scored set, and write `reads.parquet`: 1 row per image with
   `c`, `top_class`, `answered`, `calib`, and `read_<lens>_L<l>`.

The go rule: on the answered images, Spearman(read_C_L42, c) >= 0.5.

    sbatch research/monocle/monocle.sub monocle.probe_vectors --probe class
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import probe_frame as pf  # noqa: E402
from research.monocle.jlens_steer import load_lens, load_model_and_lens, pull_back_pinv  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"
LENSES = {"C": REPO / "outputs/_monocle/jlens/mm_textafter/gemma4_12b_lens.pt",
          "47": REPO / "outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt"}
AGREE_FLOOR = 0.5


def log(m: str) -> None:
    print(f"[probe-vectors] {m}", flush=True)


def load_scores(probe: str, scores_dir: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Every shard of the census, and the residuals in the same row order."""
    d = scores_dir / probe
    parts = sorted(d.glob("scores_shard*of*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no census under {d}")
    dfs, H = [], {}
    for p in parts:
        df = pd.read_parquet(p)
        z = np.load(p.with_name(p.stem + "_resid.npz"))
        if not (z["image_path"] == df["image_path"].to_numpy()).all():
            raise ValueError(f"{p.name}: the residual order differs from the parquet")
        for l in z["layers"]:
            H.setdefault(int(l), []).append(z[f"h_L{int(l)}"].astype(np.float32))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df, {l: np.concatenate(v) for l, v in H.items()}


def pole_grad(lens_model, h_final: torch.Tensor, tok: int) -> torch.Tensor:
    h = h_final.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = lens_model.unembed(h).float()
        logits[tok].backward()
    g = h.grad.detach()
    return g / (g.norm() + 1e-12)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, choices=list(pf.PROBES))
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--layers", default="41,42,43,44,45")
    ap.add_argument("--scores-dir", default=str(ROOT / "scores"))
    ap.add_argument("--out-dir", default=str(ROOT / "vectors"))
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    from jlens.hooks import ActivationRecorder
    from PIL import Image

    out = Path(args.out_dir) / args.probe
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, H = load_scores(args.probe, Path(args.scores_dir))
    for l in layers:
        if l not in H:
            raise KeyError(f"the census holds no residual at block {l}")
    calib_paths = df.loc[df["answered"], "image_path"].tolist()[:args.n_calib]
    calib_kind = "answered"
    if len(calib_paths) < args.n_calib:
        # A probe the model refuses almost everywhere (race, migration in
        # E0) has too few answered images. The pole gradient is defined at
        # any final residual, so the fallback takes the images nearest to
        # an answer: the largest pole mass under the abstain label.
        calib_kind = "nearest_to_answer"
        near = (df["p_pos"] + df["p_neg"]).sort_values(ascending=False)
        calib_paths = df.loc[near.index[:args.n_calib], "image_path"].tolist()
        log(f"{args.probe}: only {int(df['answered'].sum())} answered images; "
            f"the calibration takes the {len(calib_paths)} nearest to an answer")
    df["calib"] = df["image_path"].isin(calib_paths)

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    id_pos, id_neg, _ = pf.token_ids(tok, args.probe)
    text = pf.render_chat(tok, tmpl, pf.user_text(args.probe))
    final = lens_model.n_layers - 1
    g_pos = g_neg = None
    for q in calib_paths:
        inputs = dict(proc(text=[text], images=[[Image.open(q).convert("RGB")]],
                           return_tensors="pt").to("cuda:0"))
        inputs["use_cache"] = False
        with ActivationRecorder(lens_model.layers, at=[final]) as rec, torch.no_grad():
            model(**inputs)
        h = rec.activations[final][0, -1].detach().float()
        a, b = pole_grad(lens_model, h, id_pos), pole_grad(lens_model, h, id_neg)
        g_pos = a if g_pos is None else g_pos + a
        g_neg = b if g_neg is None else g_neg + b
    d_pos = (g_pos / g_pos.norm()).cpu()
    d_neg = (g_neg / g_neg.norm()).cpu()
    d_con = d_pos - d_neg
    d_con = d_con / d_con.norm()
    log(f"{args.probe}: {len(calib_paths)} calibration images, "
        f"cos(d_pos, d_neg) = {float(d_pos @ d_neg):.3f}")
    del model
    torch.cuda.empty_cache()

    vecs = {"d_pos": d_pos.numpy(), "d_neg": d_neg.numpy(),
            "d_contrast": d_con.numpy(), "layers": np.array(layers),
            "calib_paths": np.array(calib_paths)}
    unit = lambda v: v / (v.norm() + 1e-12)  # noqa: E731
    V: dict[tuple[str, int], np.ndarray] = {}
    for name, path in LENSES.items():
        lens = load_lens(str(path))
        for l in layers:
            J = lens.jacobians[l].to("cuda:0")
            v = unit(pull_back_pinv(J, d_con.to("cuda:0"))).cpu().numpy()
            V[(name, l)] = v
            vecs[f"v_contrast_{name}_L{l}"] = v
        del lens
        log(f"lens {name}: pulled back at {layers}")
    np.savez(out / "vectors.npz", **vecs)

    for (name, l), v in V.items():
        raw = H[l] @ v
        df[f"read_{name}_L{l}"] = raw - raw.mean()
    df.to_parquet(out / "reads.parquet", index=False)

    ans = df[df["answered"] & ~df["calib"]]
    if len(ans) < 30:
        # Too few answers for a rank test: read the agreement over every
        # image instead, and say so.
        ans = df[~df["calib"]]
        summ_note = "spearman over ALL images (too few answered)"
    else:
        summ_note = "spearman over the answered images"
    summ = {"probe": args.probe, "images": int(len(df)),
            "answered": int(df["answered"].sum()), "n_calib": len(calib_paths),
            "calib_kind": calib_kind,
            "cos_d_pos_d_neg": float(d_pos @ d_neg), "spearman_read_c_answered": {}}
    for (name, l) in V:
        col = f"read_{name}_L{l}"
        rho = float(spearmanr(ans[col], ans["c"]).statistic) if len(ans) > 3 else float("nan")
        summ["spearman_read_c_answered"][col] = round(rho, 3)
    summ["note"] = summ_note
    go = summ["spearman_read_c_answered"].get("read_C_L42", float("nan"))
    summ["go"] = bool(go >= AGREE_FLOOR)
    summ["seconds"] = int(time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
