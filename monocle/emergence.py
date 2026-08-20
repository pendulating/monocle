"""Per-patch emergence-depth (ignition-layer) maps over cyclomedia faces.

Phase-4 "rung A" of the Jacobian-lens monocle (see
vault/context/wiki/concept-jlens-monocle.md). For every image patch we ask: at
what depth does its Jacobian-lens readout "lock on" to what the model itself
will finally say at that position? The Jacobian lens gives per-patch vocab
logits at the fitted layers {6,12,18,24,30,36,42,46}; layer 47 (final) is the
model's own head output (out.logits verbatim). For each fitted layer we measure
agreement with the final layer two ways, per patch:

  * top-k Jaccard  — overlap of the two top-k token sets (1.0 = identical)
  * Jensen-Shannon — divergence of the softmaxed distributions (nats; 0 = same)

The *ignition layer* of a patch is the first fitted layer whose Jaccard with the
final readout crosses tau. A companion *sustained* ignition also requires every
later fitted layer to stay above tau, guarding against transient mid-stack
flickers. Rendering colours each patch cell by its ignition depth (cool=early
-> warm=late; hatched grey = never), giving an "emergence depth map" of the
street image. Known result the map should recover: colour content localizes
early (~L12) while reading rendered *text* ignites late and sharply (~L36).

Two layers of this file:
  * a pure-computation core (no model, CPU-testable) — the agreement/ignition
    maths, the corpus summary, the PIL renderer, and the eval-image sampler;
  * a GPU driver (main) that loads the model + wikitext lens, runs
    lens_image_layers over a fresh disjoint face sample, and writes per-image
    parquet + meta + png plus a corpus summary.

The mean-over-patch Jaccard/JS scalars used by monocle.jlens_compare are thin
wrappers over the vectorized per-patch primitives defined here (single source of
truth — jlens_compare imports them).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle.render import get_font  # noqa: E402

# Cyclomedia sampler constants (mirrors monocle.sample_fit_images). Imported
# lazily-safe: catalog is CPU-only (duckdb + constants), no model.
try:  # pragma: no cover - trivial import guard
    from dagspaces.common.cyclomedia.catalog import (
        DEFAULT_INDEX_PATH,
        RAW_ROOT,
    )
except Exception:  # noqa: BLE001
    DEFAULT_INDEX_PATH = str(
        REPO / "data/cyclomedia/browser/recordings_v1.parquet")
    RAW_ROOT = "/share/ju/cyclomedia/raw"

DEFAULT_JLENS = str(REPO / "outputs/_monocle/jlens/gemma4_12b_lens.pt")
DEFAULT_OUT_DIR = str(REPO / "outputs/_monocle/emergence")
DEFAULT_EXCLUDE = str(REPO / "outputs/_monocle/jlens/mm/fit_images.json")
FACES = ["F", "B", "L", "R"]  # cardinal only; never U (sky) / D (ground)
DEFAULT_K = 10
DEFAULT_TAU = 0.3
DEFAULT_SEED = 778  # disjoint from the mm-fit seed (777)
DEFAULT_SAMPLE = 24
LEGEND_H = 44  # px added below the image for the palette legend strip


def log(msg: str) -> None:
    print(f"[emergence] {msg}", flush=True)


# ===========================================================================
# Per-patch primitives (single source of truth; jlens_compare imports these)
# ===========================================================================
def topk_jaccard_per_patch(la: torch.Tensor, lb: torch.Tensor, k: int) -> torch.Tensor:
    """Per-patch Jaccard overlap of top-k token sets -> [n_patches] float.

    Vectorized: broadcast-compare the two [n, k] index tensors instead of
    looping patches. top-k indices are unique within a row, so
    |A ∪ B| = 2k - |A ∩ B| exactly, matching set-Jaccard.
    """
    k = min(int(k), la.shape[-1], lb.shape[-1])
    ka = la.topk(k, dim=-1).indices          # [n, k]
    kb = lb.topk(k, dim=-1).indices          # [n, k]
    match = ka.unsqueeze(2) == kb.unsqueeze(1)  # [n, k, k]
    inter = match.any(dim=2).sum(dim=1).to(torch.float64)  # a-elems present in b
    union = (2 * k) - inter
    return (inter / union).to(torch.float32)


def js_div_per_patch(la: torch.Tensor, lb: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-patch Jensen-Shannon divergence (natural log) -> [n_patches] float.

    Softmax over vocab, m = (pa + pb) / 2, JS = (KL(pa‖m) + KL(pb‖m)) / 2.
    max = ln 2 ≈ 0.693 (disjoint supports).
    """
    pa = torch.softmax(la.float(), dim=-1)
    pb = torch.softmax(lb.float(), dim=-1)
    m = 0.5 * (pa + pb)
    kl = lambda p, q: (p * ((p + eps).log() - (q + eps).log())).sum(-1)  # noqa: E731
    return (0.5 * kl(pa, m) + 0.5 * kl(pb, m)).to(torch.float32)


def topk_jaccard(la: torch.Tensor, lb: torch.Tensor, k: int) -> float:
    """Mean per-patch top-k Jaccard (the scalar jlens_compare reports)."""
    return float(topk_jaccard_per_patch(la, lb, k).to(torch.float64).mean())


def js_div(la: torch.Tensor, lb: torch.Tensor) -> float:
    """Mean per-patch Jensen-Shannon divergence (the scalar jlens_compare reports)."""
    return float(js_div_per_patch(la, lb).to(torch.float64).mean())


# ===========================================================================
# Agreement + ignition (pure, CPU-testable)
# ===========================================================================
def layer_agreement(
    per_layer: dict[int, torch.Tensor], final_layer: int, k: int = DEFAULT_K,
) -> pd.DataFrame:
    """Long agreement frame vs the final layer, one row per (patch, fitted layer).

    Columns: patch_idx (row-major), layer, jaccard (top-k overlap with the
    final-layer readout), js (Jensen-Shannon vs the final readout). Fitted
    layers are every key of ``per_layer`` except ``final_layer``, in ascending
    order. The final layer is not compared against itself (Jaccard would be a
    trivial 1.0); it is the reference.
    """
    if final_layer not in per_layer:
        raise ValueError(
            f"final_layer {final_layer} absent from per_layer {sorted(per_layer)}")
    final = per_layer[final_layer]
    n = final.shape[0]
    fitted = sorted(l for l in per_layer if l != final_layer)
    if not fitted:
        raise ValueError("no fitted layers to compare against the final layer")

    frames: list[pd.DataFrame] = []
    for layer in fitted:
        logits = per_layer[layer]
        if logits.shape != final.shape:
            raise ValueError(
                f"layer {layer} shape {tuple(logits.shape)} != "
                f"final {tuple(final.shape)}")
        jac = topk_jaccard_per_patch(logits, final, k).cpu().numpy()
        js = js_div_per_patch(logits, final).cpu().numpy()
        frames.append(pd.DataFrame({
            "patch_idx": np.arange(n, dtype=np.int64),
            "layer": np.full(n, int(layer), dtype=np.int64),
            "jaccard": jac.astype(np.float64),
            "js": js.astype(np.float64),
        }))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["patch_idx", "layer"]).reset_index(drop=True)


def ignition_layers(agreement_df: pd.DataFrame, tau: float = DEFAULT_TAU) -> pd.Series:
    """First fitted layer whose Jaccard >= tau, per patch (NaN if never).

    Returns a Series indexed by patch_idx, named 'ignition_layer'.
    """
    def _first(g: pd.DataFrame) -> float:
        g = g.sort_values("layer")
        hit = g.loc[g["jaccard"] >= tau, "layer"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    s = agreement_df.groupby("patch_idx", sort=True).apply(
        _first, include_groups=False)
    s.name = "ignition_layer"
    return s


def ignition_layers_sustained(
    agreement_df: pd.DataFrame, tau: float = DEFAULT_TAU,
) -> pd.Series:
    """Lowest layer of the all-above-tau *suffix* ending at the last fitted
    layer, per patch (NaN if the last fitted layer is itself below tau).

    Guards against a transient flicker above tau at, say, L18 that drops back
    below by L24: the sustained ignition only fires once the readout stays
    locked to the final layer for the rest of the stack. Returns a Series
    indexed by patch_idx, named 'ignition_layer_sustained'.
    """
    def _sustained(g: pd.DataFrame) -> float:
        g = g.sort_values("layer")
        ge = (g["jaccard"].to_numpy() >= tau)
        layers = g["layer"].to_numpy()
        if len(ge) == 0 or not ge[-1]:
            return float("nan")
        i = len(ge) - 1
        while i - 1 >= 0 and ge[i - 1]:
            i -= 1
        return float(layers[i])

    s = agreement_df.groupby("patch_idx", sort=True).apply(
        _sustained, include_groups=False)
    s.name = "ignition_layer_sustained"
    return s


def summarize(
    ignition: pd.Series | Iterable[float],
    fitted_layers: Optional[list[int]] = None,
) -> dict:
    """Corpus stats over a pooled ignition Series (all patches, all images).

    Returns histogram (count per layer), never-ignited fraction, mean, median.
    """
    s = pd.Series(list(ignition) if not isinstance(ignition, pd.Series) else ignition,
                  dtype="float64")
    n = int(s.shape[0])
    finite = s.dropna()
    n_never = int(s.isna().sum())
    n_ig = int(finite.shape[0])
    counts = finite.astype("int64").value_counts().to_dict()
    if fitted_layers is not None:
        hist = {int(l): int(counts.get(int(l), 0)) for l in sorted(fitted_layers)}
    else:
        hist = {int(l): int(c) for l, c in sorted(counts.items())}
    return {
        "n_patches": n,
        "n_ignited": n_ig,
        "n_never_ignited": n_never,
        "frac_never_ignited": (n_never / n) if n else float("nan"),
        "mean_ignition_layer": float(finite.mean()) if n_ig else float("nan"),
        "median_ignition_layer": float(finite.median()) if n_ig else float("nan"),
        "histogram": hist,
    }


# ===========================================================================
# Discrete emergence palette + PIL renderer (PIL-only, CPU-testable)
# ===========================================================================
# Cool -> warm ramp (RdYlBu reversed): early ignition blue, late ignition red.
_PALETTE_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, (49, 54, 149)),
    (0.25, (69, 117, 180)),
    (0.50, (224, 243, 248)),
    (0.75, (253, 174, 97)),
    (1.00, (165, 0, 38)),
)
_NEVER_RGB = (150, 150, 150)


def _ramp(t: float) -> tuple[int, int, int]:
    """Interpolate the cool->warm ramp at t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    for (t0, c0), (t1, c1) in zip(_PALETTE_STOPS, _PALETTE_STOPS[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(int(round(a + f * (b - a))) for a, b in zip(c0, c1))  # type: ignore[return-value]
    return _PALETTE_STOPS[-1][1]


def palette_for_layers(layers: list[int]) -> dict[int, tuple[int, int, int]]:
    """Map each fitted layer to an RGB colour by its rank in the sorted set."""
    ls = sorted(int(l) for l in layers)
    if not ls:
        return {}
    if len(ls) == 1:
        return {ls[0]: _ramp(0.5)}
    return {l: _ramp(i / (len(ls) - 1)) for i, l in enumerate(ls)}


def _ignition_array(ignition: Any, n_patches: int) -> np.ndarray:
    """Coerce ignition (Series / dict / array) to a [n_patches] float array."""
    arr = np.full(n_patches, np.nan, dtype=np.float64)
    if isinstance(ignition, pd.Series):
        for idx, val in ignition.items():
            if 0 <= int(idx) < n_patches:
                arr[int(idx)] = float(val)
    elif isinstance(ignition, dict):
        for idx, val in ignition.items():
            if 0 <= int(idx) < n_patches:
                arr[int(idx)] = float(val)
    else:
        vals = np.asarray(list(ignition), dtype=np.float64)
        arr[: len(vals)] = vals[:n_patches]
    return arr


def _pool_ignition(
    arr: np.ndarray, n_rows: int, n_cols: int, pool: int,
) -> tuple[np.ndarray, int, int]:
    """nan-median pool the per-patch ignition over pool x pool grid blocks."""
    if pool <= 1:
        return arr, n_rows, n_cols
    grid = arr.reshape(n_rows, n_cols)
    pr = -(-n_rows // pool)
    pc = -(-n_cols // pool)
    out = np.full(pr * pc, np.nan, dtype=np.float64)
    for r0 in range(0, n_rows, pool):
        for c0 in range(0, n_cols, pool):
            block = grid[r0:r0 + pool, c0:c0 + pool]
            vals = block[~np.isnan(block)]
            if vals.size:
                out[(r0 // pool) * pc + (c0 // pool)] = float(np.median(vals))
    return out, pr, pc


def render_emergence(
    image: Image.Image,
    ignition: Any,
    grid: Any,
    *,
    pool: int = 1,
    layers: Optional[list[int]] = None,
    dim: float = 0.45,
    alpha: int = 150,
    legend_h: int = LEGEND_H,
) -> Image.Image:
    """Semi-transparent emergence-depth heatmap over the street image.

    Each patch cell is tinted by its ignition layer via the discrete cool->warm
    palette; never-ignited patches are grey with a diagonal hatch. A legend
    strip (layer -> colour, plus 'never') is drawn with PIL below the image.
    ``grid`` is any object exposing ``n_rows`` / ``n_cols`` (an extract.PatchGrid).
    Returns an RGB image of size (W, H + legend_h). PIL-only.
    """
    n_rows, n_cols = int(grid.n_rows), int(grid.n_cols)
    n_patches = n_rows * n_cols
    arr = _ignition_array(ignition, n_patches)
    arr, n_rows, n_cols = _pool_ignition(arr, n_rows, n_cols, max(1, int(pool)))

    finite = arr[~np.isnan(arr)]
    fitted = (sorted(int(l) for l in layers) if layers is not None
              else sorted({int(v) for v in finite.tolist()}))
    pal = palette_for_layers(fitted) if fitted else {}

    base = image.convert("RGB")
    W, H = base.size
    darkened = Image.eval(base, lambda px: int(px * dim))
    canvas = darkened.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell_w = W / n_cols
    cell_h = H / n_rows
    for idx in range(n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        x0, y0 = round(c * cell_w), round(r * cell_h)
        x1, y1 = round((c + 1) * cell_w), round((r + 1) * cell_h)
        v = arr[idx]
        if np.isnan(v):
            draw.rectangle([x0, y0, x1, y1], fill=(*_NEVER_RGB, alpha))
            step = max(6, int(min(cell_w, cell_h) / 3))
            for off in range(-int(cell_h), int(cell_w) + 1, step):
                draw.line([(x0 + off, y1), (x0 + off + int(cell_h), y0)],
                          fill=(90, 90, 90, alpha), width=1)
        else:
            rgb = pal.get(int(v), _ramp(0.5))
            draw.rectangle([x0, y0, x1, y1], fill=(*rgb, alpha))

    composited = Image.alpha_composite(canvas, overlay).convert("RGB")

    out = Image.new("RGB", (W, H + legend_h), (20, 20, 20))
    out.paste(composited, (0, 0))
    _draw_legend(out, fitted, pal, y0=H, height=legend_h, width=W)
    return out


def _draw_legend(
    img: Image.Image,
    layers: list[int],
    pal: dict[int, tuple[int, int, int]],
    *,
    y0: int,
    height: int,
    width: int,
) -> None:
    """Palette legend strip: a swatch + label per fitted layer, then 'never'."""
    draw = ImageDraw.Draw(img)
    font = get_font(max(9, height // 3))
    entries: list[tuple[str, tuple[int, int, int], bool]] = [
        (f"L{l}", pal.get(l, _ramp(0.5)), False) for l in layers
    ]
    entries.append(("never", _NEVER_RGB, True))
    n = max(1, len(entries))
    slot = width / n
    sw = max(10, int(min(slot * 0.35, height * 0.45)))
    pad = 4
    cy = y0 + height // 2
    for i, (label, rgb, hatched) in enumerate(entries):
        sx = int(i * slot) + pad
        sy = cy - sw // 2
        draw.rectangle([sx, sy, sx + sw, sy + sw], fill=rgb, outline=(0, 0, 0))
        if hatched:
            for off in range(0, sw, 4):
                draw.line([(sx, sy + off), (sx + off, sy)], fill=(90, 90, 90))
        draw.text((sx + sw + pad, cy), label, font=font,
                  fill=(235, 235, 235), anchor="lm")


# ===========================================================================
# Eval-image sampler (pandas; disjoint from the mm-fit set)
# ===========================================================================
def _face_path(dataset: str, recording_id: str, face: str, raw_root: str) -> str:
    """Deterministic face image path (no globbing): bucket = recording_id[:5]."""
    return os.path.join(
        raw_root, dataset, recording_id[:5], recording_id, "faces", f"{face}.jpg")


def load_exclude_paths(exclude_json: str | Path) -> set[str]:
    """Set of image paths to exclude (the mm-fit images), from a fit_images.json."""
    p = Path(exclude_json)
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    return {str(d["path"]) for d in data if "path" in d}


def sample_emergence_faces(
    index_path: str | Path,
    n: int,
    seed: int,
    exclude_paths: set[str],
    *,
    faces: list[str] = FACES,
    raw_root: str = RAW_ROOT,
    oversample: int = 3,
    exists_fn=os.path.isfile,
) -> tuple[list[dict], dict]:
    """Stratified equal-per-dataset face sample, disjoint from ``exclude_paths``.

    Mirrors monocle.sample_fit_images (one random cardinal face per recording,
    balanced across dataset strata, existence-checked) but with pandas instead
    of DuckDB — trivially mockable and free of the SAMPLE-before-WHERE trap.
    Reads only [dataset, recording_id]. Returns (kept records, stats).
    """
    df = pd.read_parquet(index_path, columns=["dataset", "recording_id"])
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    target = oversample * n
    per_dataset = -(-target // max(1, len(datasets)))
    rng = np.random.RandomState(seed)

    per_ds_recs: dict[str, list[str]] = {}
    for ds in datasets:
        recs = df.loc[df["dataset"] == ds, "recording_id"].astype(str).to_numpy()
        if len(recs) == 0:
            per_ds_recs[ds] = []
            continue
        take = min(per_dataset, len(recs))
        idx = rng.choice(len(recs), size=take, replace=False)
        per_ds_recs[ds] = [recs[i] for i in idx]

    # Round-robin interleave by rank so the first-n-existing stays balanced.
    candidates: list[dict] = []
    max_rank = max((len(v) for v in per_ds_recs.values()), default=0)
    for rank in range(max_rank):
        for ds in datasets:
            recs = per_ds_recs[ds]
            if rank >= len(recs):
                continue
            rec = recs[rank]
            face = faces[int(rng.randint(len(faces)))]
            candidates.append({
                "recording_id": rec, "dataset": ds, "face": face,
                "path": _face_path(ds, rec, face, raw_root),
            })

    kept: list[dict] = []
    checked = missing = excluded = 0
    for cand in candidates:
        checked += 1
        if cand["path"] in exclude_paths:
            excluded += 1
            continue
        if exists_fn(cand["path"]):
            kept.append(cand)
            if len(kept) >= n:
                break
        else:
            missing += 1

    stats = {
        "n": len(kept), "seed": seed, "requested": n,
        "candidates_checked": checked, "missing": missing,
        "excluded_fit_images": excluded, "datasets": len(datasets),
        "per_dataset": {d: sum(1 for k in kept if k["dataset"] == d) for d in datasets},
        "provenance": (
            f"stratified equal-per-dataset pandas sample of {index_path} "
            f"(seed={seed}), one random cardinal face per recording, "
            f"existence-filtered, disjoint from {len(exclude_paths)} mm-fit images"
        ),
    }
    return kept, stats


def image_id_for(rec: Optional[dict], path: str) -> str:
    """Stable id: '<recording_id>_<face>' for sampled faces, else path-derived."""
    if rec is not None and rec.get("recording_id") and rec.get("face"):
        return f"{rec['recording_id']}_{rec['face']}"
    p = Path(path)
    return f"{p.parent.parent.name}_{p.stem}"


# ===========================================================================
# GPU driver
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="monocle.emergence",
        description="Per-patch emergence-depth maps over cyclomedia faces.")
    ap.add_argument("--images", nargs="+", default=None,
                    help="Explicit image paths (overrides sampling).")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help="Number of faces to sample when --images absent.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--jlens", default=DEFAULT_JLENS, help="Fitted lens .pt.")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--index", default=DEFAULT_INDEX_PATH)
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                    help="fit_images.json whose paths are excluded from the eval set.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--pool", type=int, default=1, help="pool x pool render pooling.")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--system", default=None, help="Optional system prompt.")
    ap.add_argument("--smoke", action="store_true",
                    help="2 images; print per-quartile ignition; nonzero on shape violation.")
    return ap


def _resolve_images(args) -> list[dict]:
    """Return [{path, recording_id?, dataset?, face?}] for the driver."""
    if args.images:
        return [{"path": p} for p in args.images]
    n = 2 if args.smoke else args.sample
    exclude = load_exclude_paths(args.exclude)
    kept, stats = sample_emergence_faces(args.index, n, args.seed, exclude)
    log(f"sampled {len(kept)}/{n} faces (excluded {stats['excluded_fit_images']} "
        f"fit images, {stats['missing']} missing); per_dataset={stats['per_dataset']}")
    return kept


def process_image(
    proc: Any, model: Any, tmpl: str, lens: Any, lens_model: Any,
    image: Image.Image, *, k: int, tau: float, system: Optional[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, Any, int, list[int]]:
    """One image -> (merged long frame, ignition, sustained, grid, final, fitted)."""
    from monocle import jlens_read

    final = lens_model.n_layers - 1
    per_layer, grid = jlens_read.lens_image_layers(
        proc, model, tmpl, lens, lens_model, image, system=system)
    fitted = sorted(l for l in per_layer if l != final)
    agree = layer_agreement(per_layer, final, k=k)
    ign = ignition_layers(agree, tau=tau)
    sus = ignition_layers_sustained(agree, tau=tau)
    merged = agree.merge(
        ign.rename("ignition_layer").reset_index(), on="patch_idx", how="left")
    merged = merged.merge(
        sus.rename("ignition_layer_sustained").reset_index(), on="patch_idx", how="left")
    return merged, ign, sus, grid, final, fitted


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from monocle import extract, jlens_read

    images = _resolve_images(args)
    if not images:
        log("FATAL: no images to process")
        return 1

    lens = jlens_read.load_lens(args.jlens)
    proc, model, tmpl = extract.load_model(args.model_dir)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)

    pooled_ign: list[float] = []
    per_image: list[dict] = []
    fitted_layers: list[int] = []
    shape_ok = True

    for rec in images:
        path = rec["path"]
        image_id = image_id_for(rec, path)
        t0 = time.time()
        image = extract.open_image(path)
        merged, ign, sus, grid, final, fitted = process_image(
            proc, model, tmpl, lens, lens_model, image,
            k=args.k, tau=args.tau, system=args.system)
        fitted_layers = fitted
        dt = time.time() - t0

        # Shape sanity: one ignition per patch; long frame = n_fitted * n_patches.
        n_patches = grid.n_patches
        if (len(ign) != n_patches
                or len(merged) != len(fitted) * n_patches):
            shape_ok = False
            log(f"  SHAPE VIOLATION {image_id}: patches={n_patches} "
                f"ign={len(ign)} rows={len(merged)} fitted={len(fitted)}")

        meta = {
            **grid.to_meta(), "k": args.k, "tau": args.tau,
            "final_layer": int(final), "fitted_layers": fitted,
            "jlens": args.jlens, "seed": args.seed,
            "source_image": path,
            "recording_id": rec.get("recording_id"),
            "dataset": rec.get("dataset"), "face": rec.get("face"),
        }
        pq = out_dir / f"{image_id}.emergence.parquet"
        mj = out_dir / f"{image_id}.meta.json"
        merged.assign(image_id=image_id).to_parquet(pq, index=False)
        mj.write_text(json.dumps({"image_id": image_id, **meta}, indent=2))

        if not args.no_render:
            png = out_dir / f"{image_id}.emergence.png"
            render_emergence(image, ign, grid, pool=args.pool,
                             layers=fitted).save(png)

        summ = summarize(ign, fitted_layers=fitted)
        pooled_ign.extend(ign.tolist())
        per_image.append({
            "image_id": image_id, "source_image": path,
            "n_patches": n_patches, "parquet": str(pq), **summ,
        })
        log(f"  {image_id}: n={n_patches} median_ignition="
            f"{summ['median_ignition_layer']} frac_never="
            f"{summ['frac_never_ignited']:.3f} ({dt:.1f}s)")

    corpus = summarize(pd.Series(pooled_ign, dtype="float64"),
                       fitted_layers=fitted_layers)
    summary = {
        "n_images": len(per_image), "k": args.k, "tau": args.tau,
        "seed": args.seed, "jlens": args.jlens, "fitted_layers": fitted_layers,
        "corpus": corpus, "per_image": per_image,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"corpus: median_ignition={corpus['median_ignition_layer']} "
        f"frac_never={corpus['frac_never_ignited']:.3f} "
        f"hist={corpus['histogram']}")
    log(f"summary -> {out_dir / 'summary.json'}")

    if args.smoke:
        s = pd.Series(pooled_ign, dtype="float64").dropna()
        if len(s):
            qs = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            log("smoke per-quartile ignition layers: "
                + " ".join(f"q{int(p*100)}={v:.0f}" for p, v in qs.items()))
        else:
            log("smoke: no patch ever ignited")
        if not shape_ok:
            log("SMOKE FAIL: shape violation")
            return 1
        log("SMOKE PASS")

    return 0 if shape_ok else 1


if __name__ == "__main__":
    sys.exit(main())
