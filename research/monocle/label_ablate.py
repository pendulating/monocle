"""P1 of the Jacobian-lens review: label-subspace ABLATION (the causal leg).

The workspace claim says the judgment enters the lens channel at L42 under
the production prompt. That claim rests on a readout. This module tests it
by removal, not by injection: it deletes the label-direction subspace from
the residual at one block, at the answer position, and reads the change in
the model's own judgment.

If the judgment uses the channel from L42 up, removal at or above L42 must
move the judgment. Removal below the entry depth must not. The depth profile
is the finding.

The design
----------
1. Directions. Under the PROD prompt, on a small calibration set, at the
   answer position (teacher-forced ``{"answer": "`` prefix, seq -1): for
   each distinguishable first-token class (Much*, Less, Same, More,
   NotSure), take ``d_i = d logit_i / dh_final`` by autograd through
   ``lens_model.unembed`` (final norm + lm_head + softcap — never by hand).
   Average over the calibration pairs.
2. Pull-back. ``v_i = J_l^T d_i`` per layer (``jlens_steer.pull_back_raw``),
   with the orientation invariant checked. A QR step turns the 5 vectors
   into an orthonormal basis ``B_l`` of rank <= 5.
3. Ablation. A forward hook on block ``l`` edits the answer position only:
   ``h <- h - B_l (B_l^T h)``. One hook at a time, removed in a finally.
4. Arms per (pair, layer): ``none`` (baseline forward), ``label`` (the
   basis from step 2), ``ctrl_matched`` (the same pipeline on 5 random
   vocabulary rows — this matches the construction and the massive-channel
   concentration), ``ctrl_random`` (a random rank-5 orthonormal subspace in
   the residual space). Each hooked forward records the removed-norm
   fraction ``||B^T h|| / ||h||``.
5. Readout: the model's OWN final logits at the answer position. Metrics
   per row: restricted argmax class, p(prod-label first token), the
   contrast ``c = logit(More) - logit(Less)``, and a flip against the
   ``none`` arm of the same pair.
6. Statistics: per layer, the paired difference of ``|dc|`` between the
   label arm and the matched control (Wilcoxon signed-rank p, a
   20000-resample bootstrap 95% interval, seed 777), then a
   Benjamini-Hochberg step over the layers.

Warning: the registered subway run predates the 2026-08-20 layout decision.
The read stays internally consistent — every arm reads the same recorded
prompt — but a battery re-run will change the forward this module reads.

Usage:
    sbatch --time=00:30:00 research/monocle/monocle.sub monocle.label_ablate --smoke
    sbatch --time=04:00:00 research/monocle/monocle.sub monocle.label_ablate \\
        --n-pairs 150 --layers 24 30 36 42 46
    python -m research.monocle.label_ablate --summarize outputs/_monocle/label_ablate/ablate.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import canonical  # noqa: E402  (CPU-safe)
# Pure math helpers, shared with the steering rung. Stable API.
from research.monocle.jlens_steer import (  # noqa: E402  (CPU-safe module)
    orientation_residual, pull_back_raw, split_calibration_eval,
)

CASE_DEFAULT = "subway_safety"
KIND_DEFAULT = "proxy"
LENS_DEFAULT = str(REPO / "outputs/_monocle/jlens/mm/gemma4_12b_lens.pt")
OUT_DEFAULT = str(REPO / "outputs/_monocle/label_ablate")

FORCE_PREFIX = canonical.FORCE_PREFIX
SEED = 777
N_CALIB_DEFAULT = 16
DEFAULT_LAYERS = [24, 30, 36, 42, 46]
#: The hooked arms. ``none`` is the baseline and takes no hook.
HOOKED_ARMS = ("label", "ctrl_matched", "ctrl_random")
N_BOOT = 20000


def log(m: str) -> None:
    print(f"[label-ablate] {m}", flush=True)


# ---------------------------------------------------------------------------
# A) Pure, CPU-testable helpers
# ---------------------------------------------------------------------------
def orthonormal_basis(vectors: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis ``[d, r]`` for the row span of ``vectors`` ``[k, d]``.

    QR on the transpose, with the near-zero columns dropped, so a
    rank-deficient input gives ``r < k`` instead of a garbage direction.
    """
    if vectors.dim() != 2:
        raise ValueError(f"expected [k, d], got {tuple(vectors.shape)}")
    a = vectors.T.to(torch.float32)                     # [d, k]
    q, r = torch.linalg.qr(a, mode="reduced")           # q [d, k], r [k, k]
    keep = r.diagonal().abs() > tol * r.diagonal().abs().max().clamp(min=1e-30)
    return q[:, keep].contiguous()


def project_out(row: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``row - B (B^T row)`` in fp32. ``row`` is ``[d]``, ``basis`` ``[d, r]``."""
    row32 = row.to(torch.float32)
    coeff = basis.T @ row32
    return row32 - basis @ coeff


def removed_fraction(row: torch.Tensor, basis: torch.Tensor) -> float:
    """``||B^T row|| / ||row||`` in fp32."""
    row32 = row.to(torch.float32)
    denom = float(row32.norm()) + 1e-12
    return float((basis.T @ row32).norm()) / denom


def draw_matched_ids(
    vocab_size: int, exclude: set[int], k: int, seed: int = SEED,
) -> list[int]:
    """``k`` distinct vocabulary ids, none in ``exclude``. Deterministic."""
    gen = torch.Generator().manual_seed(seed)
    out: list[int] = []
    seen = set(exclude)
    while len(out) < k:
        tid = int(torch.randint(0, vocab_size, (1,), generator=gen))
        if tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def random_basis(d: int, k: int, seed: int = SEED) -> torch.Tensor:
    """A random orthonormal ``[d, k]`` basis, deterministic under ``seed``."""
    gen = torch.Generator().manual_seed(seed)
    return orthonormal_basis(torch.randn(k, d, generator=gen))


def answer_metrics(
    logits: torch.Tensor, class_ids: dict[str, int], prod_first: int,
    tok_more: int, tok_less: int,
) -> dict:
    """Metrics for one ``[vocab]`` answer-logits row."""
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    names = list(class_ids)
    cls_logits = torch.tensor([float(logits[class_ids[n]]) for n in names])
    return {
        "argmax_class": names[int(cls_logits.argmax())],
        "p_prod": float(probs[prod_first]),
        "c": float(logits[tok_more]) - float(logits[tok_less]),
    }


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank_from_end, idx in enumerate(order[::-1]):
        rank = m - rank_from_end
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    return q.tolist()


def paired_stats(
    a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Mean of ``a - b``, a bootstrap 95% interval, and a Wilcoxon p."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    try:
        from scipy.stats import wilcoxon
        nz = diff[diff != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
    except ImportError:
        p = float("nan")
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "p": p, "n": n}


def summarize_frame(df: pd.DataFrame) -> dict:
    """Aggregate the long frame; label-vs-control paired tests per layer."""
    base = df[df["arm"] == "none"].set_index("pair_id")
    rows = df[df["arm"] != "none"].copy()
    rows["dc"] = rows["c"] - base.loc[rows["pair_id"], "c"].to_numpy()
    rows["dp_prod"] = (rows["p_prod"]
                       - base.loc[rows["pair_id"], "p_prod"].to_numpy())
    rows["flip"] = (rows["argmax_class"].to_numpy()
                    != base.loc[rows["pair_id"], "argmax_class"].to_numpy())

    by_arm = []
    for (layer, arm), g in rows.groupby(["layer", "arm"], sort=True):
        by_arm.append({
            "layer": int(layer), "arm": arm, "n": int(len(g)),
            "flip_rate": float(g["flip"].mean()),
            "mean_dp_prod": float(g["dp_prod"].mean()),
            "mean_dc": float(g["dc"].mean()),
            "mean_abs_dc": float(g["dc"].abs().mean()),
            "mean_removed_frac": float(g["removed_frac"].mean()),
        })

    tests: dict[str, list[dict]] = {}
    for ctrl in ("ctrl_matched", "ctrl_random"):
        fam = []
        for layer in sorted(rows["layer"].unique()):
            lab = rows[(rows.layer == layer) & (rows.arm == "label")]
            con = rows[(rows.layer == layer) & (rows.arm == ctrl)]
            merged = lab.merge(con, on="pair_id", suffixes=("_l", "_c"))
            st = paired_stats(merged["dc_l"].abs().to_numpy(),
                              merged["dc_c"].abs().to_numpy())
            st.update({
                "layer": int(layer),
                "flip_rate_label": float(merged["flip_l"].mean()),
                "flip_rate_ctrl": float(merged["flip_c"].mean()),
            })
            fam.append(st)
        qs = bh_adjust([t["p"] for t in fam])
        for t, q in zip(fam, qs):
            t["q"] = q
            t["survives_bh"] = bool(q < 0.05)
        tests[f"label_vs_{ctrl}"] = fam
    return {"by_arm": by_arm, "paired_abs_dc": tests}


# ---------------------------------------------------------------------------
# B) GPU glue (deferred imports)
# ---------------------------------------------------------------------------
def _activation_recorder():
    from jlens.hooks import ActivationRecorder
    return ActivationRecorder


def load_model_and_lens(model_dir: str):
    """gemma-4 via the verified monocle recipe + jlens wrapper.

    force_bos=False is load-bearing: the chat template emits its own <bos>.
    Returns ``(proc, model, tmpl, lens_model)``.
    """
    import jlens
    from research.monocle import extract
    proc, model, tmpl = extract.load_model(model_dir)
    lens_model = jlens.from_hf(model, proc.tokenizer, force_bos=False)
    return proc, model, tmpl, lens_model


def load_lens(path: str):
    from jlens import JacobianLens
    return JacobianLens.load(path)


def load_pairs(case: str, kind: str) -> list[dict]:
    """Supervision rows through the registry; one presentation per pair.

    Abstained rows are KEPT: the primary statistic ``|dc|`` does not depend
    on the production label, and a drop would restrict the study to the
    pairs the model answered.
    """
    df = canonical.load_results(case, kind, columns=[
        "pair_id", "repeat_idx", "presented_label",
        "presented_left_path", "presented_right_path"])
    df = df[df["repeat_idx"] == 0]
    return df.reset_index(drop=True).to_dict("records")


def make_build_inputs(proc, tmpl, device: str):
    from PIL import Image

    def build(row: dict, cfg) -> dict:
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]
        inputs = canonical.build_pair_inputs(
            proc, tmpl, imgs, row, cfg, force_prefix=FORCE_PREFIX,
            device=device)
        out = dict(inputs)
        out["use_cache"] = False
        return out

    return build


def compute_class_directions(
    model, lens_model, build, calib_rows: list[dict], prod_cfg,
    token_ids: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Mean unit gradient ``d logit_tid / dh_final`` per class, fp32 CPU.

    One PROD forward per calibration pair; the gradients go through
    ``lens_model.unembed`` by autograd, never by hand.
    """
    ActivationRecorder = _activation_recorder()
    final = lens_model.n_layers - 1
    sums: dict[str, Optional[torch.Tensor]] = {k: None for k in token_ids}
    for i, row in enumerate(calib_rows):
        inputs = build(row, prod_cfg)
        with ActivationRecorder(lens_model.layers, at=[final]) as rec:
            with torch.no_grad():
                model(**inputs)
            h_final = rec.activations[final][0, -1].detach()
        h_leaf = h_final.float().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = lens_model.unembed(h_leaf).float()
            for name, tid in token_ids.items():
                g = torch.autograd.grad(
                    logits[tid], h_leaf, retain_graph=True)[0].detach()
                d_i = (g / (g.norm() + 1e-12)).cpu()
                sums[name] = d_i if sums[name] is None else sums[name] + d_i
        if (i + 1) % 4 == 0:
            log(f"  calibration {i + 1}/{len(calib_rows)}")
    out: dict[str, torch.Tensor] = {}
    for name, s in sums.items():
        assert s is not None
        out[name] = (s / (s.norm() + 1e-12)).float()
    return out


def build_bases(
    lens, directions: dict[str, torch.Tensor], layers: list[int],
    device: str, tol: float = 1e-3,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Per layer: pull every direction back, check orientation, QR.

    Returns ``({layer: B [d, r] fp32 on device}, {layer: worst residual})``.
    """
    bases: dict[int, torch.Tensor] = {}
    worst: dict[int, float] = {}
    for layer in layers:
        J = lens.jacobians[layer].float().to(device)
        vs = []
        res = 0.0
        for d in directions.values():
            d_dev = d.to(device)
            res = max(res, orientation_residual(lens, J, d_dev, layer))
            vs.append(pull_back_raw(J, d_dev))
        if res > tol:
            raise RuntimeError(
                f"orientation invariant failed at layer {layer}: {res:.2e}")
        worst[layer] = res
        bases[layer] = orthonormal_basis(torch.stack(vs).cpu()).to(device)
    return bases, worst


def make_projection_hook(basis: torch.Tensor, pos: int, holder: dict):
    """Forward hook: remove the basis span from the block output at ``pos``.

    Clones the output (never edits the original in place) and records the
    removed-norm fraction and a fire count into ``holder``.
    """
    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        tensor = output[0] if is_tuple else output
        new = tensor.clone()
        row = new[0, pos, :]
        holder["frac"] = removed_fraction(row, basis)
        new[0, pos, :] = project_out(row, basis).to(new.dtype)
        holder["fires"] = holder.get("fires", 0) + 1
        if is_tuple:
            return (new, *tuple(output[1:]))
        return new

    return hook


def ablated_answer_logits(
    model, lens_model, inputs: dict, layer: Optional[int],
    basis: Optional[torch.Tensor], holder: dict,
) -> torch.Tensor:
    """One forward, with (optionally) a projection hook on one block.

    Returns the model's own fp32 logits at the answer position (seq -1).
    The hook is removed in a finally, so exactly one is live per forward.
    """
    handle = None
    try:
        if basis is not None:
            pos = int(inputs["input_ids"].shape[1]) - 1
            hook = make_projection_hook(basis, pos, holder)
            handle = lens_model.layers[layer].register_forward_hook(hook)
        with torch.inference_mode():
            out = model(**inputs)
    finally:
        if handle is not None:
            handle.remove()
    logits = out.logits[0, -1].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite answer logits")
    return logits


# ---------------------------------------------------------------------------
# C) Checkpoint / resume (the safety_workspace pattern)
# ---------------------------------------------------------------------------
def completed_pair_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["pair_id"])["pair_id"].unique())


def write_long_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.tmp{os.getpid()}.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# D) Driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    layers = list(args.layers)

    proc, model, tmpl, lens_model = load_model_and_lens(args.model_dir)
    lens = load_lens(args.lens)
    missing = sorted(set(layers) - set(lens.source_layers))
    if missing:
        raise SystemExit(
            f"layers {missing} not fitted; the lens holds {lens.source_layers}")

    conds = canonical.build_conditions(args.case, args.kind, ["prod"])
    prod_cfg = conds["prod"]
    label_first, class_ids, _collapse = canonical.label_classes(
        proc.tokenizer, prod_cfg)
    tok_more, tok_less = label_first["More"], label_first["Less"]
    log(f"classes {class_ids}")

    rows = load_pairs(args.case, args.kind)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    calib_rows, eval_rows = split_calibration_eval(
        rows, args.n_calib, args.n_pairs)
    log(f"{len(calib_rows)} calibration pairs, {len(eval_rows)} eval pairs")

    build = make_build_inputs(proc, tmpl, device)

    # Directions + bases, cached so a resume does not recompute them.
    dir_path = out_dir / "directions.pt"
    cfg_key = {"case": args.case, "kind": args.kind, "lens": args.lens,
               "layers": layers, "n_calib": args.n_calib, "seed": SEED,
               "calib_pair_ids": [r["pair_id"] for r in calib_rows]}
    if dir_path.exists():
        blob = torch.load(dir_path, map_location="cpu", weights_only=True)
        if blob["cfg"] != cfg_key:
            raise SystemExit(
                f"{dir_path} was built under another config; move it aside")
        label_dirs = blob["label_dirs"]
        matched_dirs = blob["matched_dirs"]
        log("directions loaded from cache")
    else:
        log("calibrate the label directions")
        label_dirs = compute_class_directions(
            model, lens_model, build, calib_rows, prod_cfg, class_ids)
        vocab_size = int(lens_model._lm_head.weight.shape[0])
        matched_ids = draw_matched_ids(
            vocab_size, set(class_ids.values()), len(class_ids))
        log(f"matched control ids: {matched_ids}")
        matched_tokens = {f"ctrl_{t}": t for t in matched_ids}
        matched_dirs = compute_class_directions(
            model, lens_model, build, calib_rows, prod_cfg, matched_tokens)
        tmp = out_dir / f"directions.tmp{os.getpid()}.pt"
        torch.save({"cfg": cfg_key, "label_dirs": label_dirs,
                    "matched_dirs": matched_dirs,
                    "matched_ids": matched_ids}, tmp)
        os.replace(tmp, dir_path)

    label_bases, res_l = build_bases(lens, label_dirs, layers, device)
    matched_bases, res_m = build_bases(lens, matched_dirs, layers, device)
    random_bases = {
        l: random_basis(lens.d_model, len(class_ids), seed=SEED + l).to(device)
        for l in layers}
    for l in layers:
        log(f"  L{l}: ranks label={label_bases[l].shape[1]} "
            f"matched={matched_bases[l].shape[1]} "
            f"random={random_bases[l].shape[1]}, "
            f"orient resid {max(res_l[l], res_m[l]):.1e}")

    arm_bases = {"label": label_bases, "ctrl_matched": matched_bases,
                 "ctrl_random": random_bases}

    pq = out_dir / "ablate.parquet"
    done = completed_pair_ids(pq)
    prior = pd.read_parquet(pq) if pq.exists() else pd.DataFrame()
    if done:
        log(f"resume: {len(done)} pairs already written")

    new_rows: list[dict] = []
    since_flush = 0
    t0 = time.time()
    for pi, row in enumerate(eval_rows):
        pid = row["pair_id"]
        if pid in done:
            continue
        prod_label = row["presented_label"]
        prod_first = label_first[prod_label]
        inputs = build(row, prod_cfg)

        holder: dict = {}
        base_logits = ablated_answer_logits(
            model, lens_model, inputs, None, None, holder)
        m0 = answer_metrics(base_logits, class_ids, prod_first,
                            tok_more, tok_less)
        if args.smoke and pi == 0:
            _smoke_asserts(model, lens_model, inputs, base_logits,
                           label_bases, layers, proc.tokenizer)
        new_rows.append({"pair_id": pid, "prod_label": prod_label,
                         "layer": -1, "arm": "none",
                         "removed_frac": 0.0, **m0})

        for layer in layers:
            for arm in HOOKED_ARMS:
                holder = {}
                logits = ablated_answer_logits(
                    model, lens_model, inputs, layer,
                    arm_bases[arm][layer], holder)
                if holder.get("fires", 0) != 1:
                    raise RuntimeError(
                        f"hook fired {holder.get('fires', 0)} times")
                m = answer_metrics(logits, class_ids, prod_first,
                                   tok_more, tok_less)
                new_rows.append({
                    "pair_id": pid, "prod_label": prod_label,
                    "layer": layer, "arm": arm,
                    "removed_frac": holder["frac"], **m})
        since_flush += 1
        if (pi + 1) % 5 == 0:
            n_new = max(1, pi + 1 - len(done))
            log(f"  {pi + 1}/{len(eval_rows)} pairs "
                f"({(time.time() - t0) / n_new:.1f}s/pair)")
        if since_flush >= 10:
            frames = [f for f in (prior, pd.DataFrame(new_rows))
                      if not f.empty]
            write_long_frame(pd.concat(frames, ignore_index=True), pq)
            prior = pd.read_parquet(pq)
            new_rows = []
            since_flush = 0

    frames = [f for f in (prior, pd.DataFrame(new_rows)) if not f.empty]
    if frames:
        write_long_frame(pd.concat(frames, ignore_index=True), pq)
    log(f"long frame -> {pq}")
    return summarize_cmd(str(pq), out_dir, meta={
        **cfg_key, "n_pairs": args.n_pairs, "arms": list(HOOKED_ARMS),
        "force_prefix": FORCE_PREFIX,
        "note": ("the registered run predates the 2026-08-20 layout "
                 "decision; every arm reads the same recorded prompt")})


def summarize_cmd(pq: str, out_dir: Path, meta: Optional[dict] = None) -> int:
    df = pd.read_parquet(pq)
    summary = {"meta": meta or {}, **summarize_frame(df)}
    out = out_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"summary -> {out}")

    log(f"{'layer':>6} {'arm':>13} {'flip':>6} {'d|c|':>8} {'dp_prod':>8} "
        f"{'frac':>6}")
    for r in summary["by_arm"]:
        log(f"{r['layer']:>6} {r['arm']:>13} {r['flip_rate']:>6.3f} "
            f"{r['mean_abs_dc']:>8.3f} {r['mean_dp_prod']:>8.3f} "
            f"{r['mean_removed_frac']:>6.3f}")
    log("label vs matched control, paired |dc| (BH over layers):")
    for t in summary["paired_abs_dc"]["label_vs_ctrl_matched"]:
        star = " *" if t["survives_bh"] else ""
        log(f"  L{t['layer']:>2} diff {t['mean']:+.3f} "
            f"[{t['lo']:+.3f}, {t['hi']:+.3f}] p={t['p']:.4f} "
            f"q={t['q']:.4f}{star}")
    return 0


def _smoke_asserts(model, lens_model, inputs, base_logits, label_bases,
                   layers, tokenizer) -> None:
    ids = inputs["input_ids"][0]
    dec = tokenizer.decode([int(ids[-1])])
    assert '"' in dec, f"answer position decodes to {dec!r}, not the quote"
    # The baseline must repeat exactly (same kernels, same batch size).
    again = ablated_answer_logits(model, lens_model, inputs, None, None, {})
    assert torch.equal(base_logits, again), "baseline forward not repeatable"
    for l in layers:
        B = label_bases[l]
        gram = (B.T @ B - torch.eye(B.shape[1], device=B.device)).abs().max()
        assert float(gram) < 1e-4, f"basis at L{l} not orthonormal ({gram:.1e})"
    log("  [smoke] baseline repeatable, bases orthonormal, prefix position OK")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=CASE_DEFAULT,
                    choices=list(canonical.CASES))
    ap.add_argument("--kind", default=KIND_DEFAULT,
                    choices=list(canonical.KINDS))
    ap.add_argument("--lens", default=LENS_DEFAULT)
    ap.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--n-pairs", type=int, default=150)
    ap.add_argument("--n-calib", type=int, default=N_CALIB_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="2 pairs, 2 layers, asserts on.")
    ap.add_argument("--summarize", metavar="PARQUET", default=None,
                    help="Recompute the summary from a long frame; no GPU.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.summarize:
        return summarize_cmd(args.summarize,
                             Path(args.summarize).parent)
    if args.smoke:
        args.n_pairs = 2
        args.layers = [24, 42]
        args.out_dir = str(Path(args.out_dir) / "smoke")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
