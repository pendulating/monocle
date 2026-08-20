"""Phase-0 geometry validation. Run on klara BEFORE trusting any render.

Checks, in increasing order of what would kill monocle:
  1. SHAPES   — single-image forward works; image-token block is contiguous;
                grid inference yields rows*cols == n_patches.
  2. LOCALIZE — the load-bearing check. Synthetic image with the word DOG
                drawn in the top-left quadrant and a solid red fill in the
                bottom-right. If the patch->position mapping (row-major) is
                right, 'dog'-ish probability mass concentrates in the
                top-left patch quadrant and 'red'-ish mass bottom-right.
                If it's scrambled, every downstream overlay is confidently
                wrong.
  3. REAL     — extract + score one real cyclomedia face; save parquet +
                meta + a copy of the image so the renderer can be exercised
                on real data without a GPU.

Usage (via monocle/monocle.sub on klara, .venv-nightly + LD_PRELOAD):
    python -m monocle.validate [--image PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import extract, scoring  # noqa: E402

DEFAULT_IMAGE = "/share/ju/cyclomedia/raw/brooklyn_2025_1k/W0D0M/W0D0M3OU/faces/R.jpg"


def log(msg: str) -> None:
    print(f"[validate] {msg}", flush=True)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def synthetic_quadrant_image(side: int = 1024) -> Image.Image:
    """White canvas; 'DOG' text top-left quadrant; solid red bottom-right."""
    im = Image.new("RGB", (side, side), "white")
    d = ImageDraw.Draw(im)
    d.text((side // 8, side // 8), "DOG", fill="black", font=_font(side // 6))
    d.rectangle([side // 2, side // 2, side, side], fill=(220, 20, 20))
    return im


def quadrant_mass(
    p: torch.Tensor, grid: extract.PatchGrid, token_ids: list[int],
) -> dict[str, float]:
    """Fraction of total probability mass for `token_ids` per patch quadrant,
    assuming row-major patch order."""
    mass = p[:, token_ids].sum(dim=-1)  # [n_patches]
    q = {"TL": 0.0, "TR": 0.0, "BL": 0.0, "BR": 0.0}
    for idx in range(grid.n_patches):
        r, c = idx // grid.n_cols, idx % grid.n_cols
        key = ("T" if r < grid.n_rows / 2 else "B") + ("L" if c < grid.n_cols / 2 else "R")
        q[key] += float(mass[idx])
    total = sum(q.values()) or 1.0
    return {k: v / total for k, v in q.items()}


def mass_spread(
    p: torch.Tensor, grid: extract.PatchGrid, token_ids: list[int],
) -> tuple[float, float]:
    """Mass-weighted std of (row, col) indices for `token_ids` — measures
    whether the mass blob is tall (row_std larger) or wide (col_std larger)."""
    mass = p[:, token_ids].sum(dim=-1)
    mass = mass / (mass.sum() + 1e-12)
    idx = torch.arange(grid.n_patches, device=mass.device)
    rows = (idx // grid.n_cols).float()
    cols = (idx % grid.n_cols).float()
    r_mu = float((mass * rows).sum())
    c_mu = float((mass * cols).sum())
    r_std = float(((mass * (rows - r_mu) ** 2).sum()) ** 0.5)
    c_std = float(((mass * (cols - c_mu) ** 2).sum()) ** 0.5)
    return r_std, c_std


def vocab_ids_exact(tokenizer, word: str) -> list[int]:
    """Vocab ids whose display form is exactly `word` (case-insensitive).

    Substring matching is a trap: 858 gemma tokens contain "red" ("▁required",
    "▁considered", "ered", ...) and they dominate any mass measure — this is
    what produced the spurious first-run FAIL on the red quadrant.
    """
    w = word.lower()
    return [i for i, t in enumerate(
        tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))))
        if t and scoring.display_form(t).lower() == w]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/_monocle/validate"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"image": args.image, "model_dir": args.model_dir}

    proc, model, tmpl = extract.load_model(args.model_dir)
    tokenizer = proc.tokenizer

    # ==== 1. SHAPES on the real image =====================================
    log("=" * 60)
    log("CHECK 1: shapes + grid inference (real image)")
    im = extract.open_image(args.image)
    inputs = extract.build_inputs(proc, tmpl, im)
    log(f"  input keys : {sorted(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            log(f"  {k:<16}: {tuple(v.shape)} {v.dtype}")
    logits, pos = extract.forward_patch_logits(model, inputs)
    grid = extract.infer_grid(inputs, logits.shape[0], im.size)
    log(f"  n_patches  : {logits.shape[0]} at seq {int(pos[0])}..{int(pos[-1])}")
    log(f"  grid       : {grid.n_rows}x{grid.n_cols} via '{grid.strategy}'")
    if grid.n_patches != logits.shape[0]:
        log("  FATAL: grid does not multiply out to patch count")
        return 1
    report["check1"] = {"n_patches": logits.shape[0], **grid.to_meta()}

    # ==== 2. LOCALIZE (the load-bearing check) ============================
    log("=" * 60)
    log("CHECK 2: quadrant localization (DOG top-left, red fill bottom-right)")
    syn = synthetic_quadrant_image()
    syn.save(out_dir / "synthetic_quadrants.png")
    syn_logits, syn_grid = extract.lens_image(proc, model, tmpl, syn)
    syn_p = torch.softmax(syn_logits, dim=-1)

    dog_ids = vocab_ids_exact(tokenizer, "dog")
    red_ids = vocab_ids_exact(tokenizer, "red")
    q_dog = quadrant_mass(syn_p, syn_grid, dog_ids)
    q_red = quadrant_mass(syn_p, syn_grid, red_ids)

    # (a) position: dog mass concentrates where the text is drawn
    dog_ok = max(q_dog, key=lambda k: q_dog[k]) == "TL"
    # (b) orientation: 'DOG' is drawn ~2x wider than tall, so under row-major
    # mapping its mass spreads across columns; a transposed grid would spread
    # it across rows instead (TL is transpose-invariant, so (a) alone can't
    # catch this)
    row_std, col_std = mass_spread(syn_p, syn_grid, dog_ids)
    orient_ok = col_std > row_std
    # (c) column mapping: red fill is in the right half. Do NOT require
    # argmax==BR — next-token anticipation legitimately splits red mass
    # between BR (red is here) and TR (the raster-order predecessors of the
    # red block, which sit directly above it). Measured 2026-07-20:
    # TR=0.51 BR=0.48, left half 0.01.
    red_right = q_red["TR"] + q_red["BR"]
    red_ok = red_right >= 0.85

    log("  'dog' mass by quadrant: "
        + " ".join(f"{k}={v:.2f}" for k, v in q_dog.items())
        + f" -> argmax TL {'OK' if dog_ok else 'FAIL'}")
    log(f"  'dog' spread: col_std={col_std:.2f} vs row_std={row_std:.2f}"
        f" (wide text => col > row) {'OK' if orient_ok else 'FAIL (transposed?)'}")
    log("  'red' mass by quadrant: "
        + " ".join(f"{k}={v:.2f}" for k, v in q_red.items())
        + f" -> right-half {red_right:.2f} (>=0.85) {'OK' if red_ok else 'FAIL'}")

    ok = dog_ok and orient_ok and red_ok
    report["check2"] = {
        "grid": syn_grid.to_meta(),
        "dog": {"quadrants": q_dog, "argmax_tl": dog_ok,
                "col_std": col_std, "row_std": row_std, "orient_ok": orient_ok},
        "red": {"quadrants": q_red, "right_half": red_right, "ok": red_ok},
        "pass": ok,
    }

    # eyeball dump: top-1 token per patch of the synthetic image
    mask = scoring.build_token_mask(tokenizer, syn_logits.shape[-1])
    syn_df = scoring.score_patches(syn_logits, tokenizer, k=1, token_mask=mask)
    syn_df = scoring.attach_grid(syn_df, syn_grid.n_rows, syn_grid.n_cols)
    log("  top-1 token grid (synthetic):")
    for r in range(syn_grid.n_rows):
        row = syn_df[(syn_df["patch_row"] == r) & (syn_df["rank"] == 0)]
        row = row.sort_values("patch_col")["token"].tolist()
        log("    " + " | ".join(f"{t[:8]:<8}" for t in row))
    if not ok:
        log("  FATAL: localization failed — do NOT trust row-major mapping")

    # ==== 3. REAL extraction for the renderer =============================
    log("=" * 60)
    log("CHECK 3: real-image extract + score -> parquet for the renderer")
    t0 = time.time()
    df = scoring.score_patches(logits, tokenizer, k=3, token_mask=mask)
    df = scoring.attach_grid(df, grid.n_rows, grid.n_cols)
    image_id = Path(args.image).parent.parent.name + "_" + Path(args.image).stem
    meta = {**grid.to_meta(), "k": 3, "alpha": scoring.DEFAULT_ALPHA,
            "model_dir": args.model_dir, "source_image": args.image}
    pq, mj = scoring.save_outputs(df, meta, out_dir, image_id)
    shutil.copy(args.image, out_dir / f"{image_id}.jpg")
    log(f"  scored {grid.n_patches} patches in {time.time() - t0:.1f}s")
    log(f"  wrote {pq.name}, {mj.name}, {image_id}.jpg")
    report["check3"] = {"parquet": str(pq), "meta": str(mj), "n_rows": len(df)}

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"report -> {out_dir / 'report.json'}")
    log("PASS" if ok else "FAIL (localization)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
