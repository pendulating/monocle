"""Command-line driver: image path(s) or a cyclomedia recording -> per-patch
top-k word-cloud overlays + parquet.

Loads the model once, builds the token mask once (lazily, from the first
image's vocab dimension), then lenses each image, scores it against the
image-global distribution, and renders a PNG (and optional SVG) beside the
parquet. Mirrors the extraction recipe in `monocle.validate` so the CLI and the
phase-0 validator agree on geometry and scoring.

Usage (via monocle/monocle.sub on klara, .venv-nightly + LD_PRELOAD):
    python -m monocle.cli --image /path/to/face.jpg
    python -m monocle.cli --recording-id W0D0M3OU --dataset brooklyn_2025_1k
    python -m monocle.cli --image a.jpg b.jpg --k 5 --alpha 0.5 --svg \
        --system "You are a real-estate appraiser."
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import extract, scoring  # noqa: E402

DEFAULT_OUT_DIR = str(REPO / "outputs/_monocle/runs")
DEFAULT_FACES = ["F", "B", "L", "R"]  # skip U (sky) / D (ground) by default


def log(msg: str) -> None:
    print(f"[monocle] {msg}", flush=True)


@dataclasses.dataclass
class ImageJob:
    """One image to lens: a stable id and the file path to read."""

    image_id: str
    path: str


def build_parser() -> argparse.ArgumentParser:
    """Argparse setup, factored out so tests can parse without running."""
    ap = argparse.ArgumentParser(
        prog="monocle.cli",
        description="Per-patch logit-lens word-cloud overlays for gemma-4-12B.",
    )

    src = ap.add_argument_group("image sources (at least one required)")
    src.add_argument(
        "--image", action="append", nargs="+", metavar="PATH",
        help="Direct image file(s); repeatable.")
    src.add_argument(
        "--recording-id", metavar="ID",
        help="Cyclomedia recording id; resolves face image paths via the catalog.")
    src.add_argument(
        "--dataset", metavar="NAME",
        help="Cyclomedia dataset/partition for --recording-id (optional).")
    src.add_argument(
        "--faces", nargs="+", default=list(DEFAULT_FACES), metavar="F",
        help=f"Faces to lens for --recording-id (default: {' '.join(DEFAULT_FACES)}).")

    ap.add_argument(
        "--k", type=int, default=scoring.DEFAULT_K,
        help="Top-k tokens kept per patch.")
    ap.add_argument(
        "--alpha", type=float, default=scoring.DEFAULT_ALPHA,
        help="PMI/raw-probability trade-off (alpha=0 pure PMI).")
    ap.add_argument(
        "--pool", type=int, default=2,
        help="Average probabilities over pool x pool blocks of the model's "
             "patch grid before scoring — larger display cells, fewer clouds "
             "(16x16 -> 8x8 at 2). Use --pool 1 for full model resolution.")
    ap.add_argument(
        "--system", default=None, metavar="TEXT",
        help="System prompt placed BEFORE the image tokens; conditions the "
             "per-patch predictions. Empty/absent = minimal context.")
    ap.add_argument(
        "--model-dir", default=MODEL_DIR_DEFAULT,
        help="gemma4_unified model directory.")
    ap.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="Where parquet/meta/overlays are written.")
    ap.add_argument(
        "--svg", action="store_true",
        help="Also emit an SVG overlay next to the PNG.")
    ap.add_argument(
        "--no-render", action="store_true",
        help="Extraction + parquet only; skip PNG/SVG rendering.")

    jl = ap.add_argument_group("layer mode (Jacobian lens)")
    jl.add_argument(
        "--jlens", metavar="CKPT", default=None,
        help="Path to a fitted Jacobian lens (.pt). Enables LAYER-resolved "
             "readout: each image is lensed at every requested depth instead "
             "of only the final layer.")
    jl.add_argument(
        "--layers", metavar="L", default=None,
        help="Comma-separated source layers to read (e.g. '6,12,24'). "
             "Default (omitted): all fitted lens layers + the final layer.")
    jl.add_argument(
        "--gif", action="store_true",
        help="Emit a depth GIF scrubbing through layers (requires --jlens).")
    jl.add_argument(
        "--scrubber", action="store_true",
        help="Emit an interactive HTML layer scrubber (requires --jlens).")
    return ap


def parse_layers(spec: Optional[str]) -> Optional[list[int]]:
    """'6,12,24' -> [6, 12, 24]; None/'' -> None (pass-through: all layers)."""
    if spec is None:
        return None
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    return [int(p) for p in parts] or None


def validate_jlens_args(
    args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None,
) -> None:
    """--gif/--scrubber require --jlens. Raises ValueError (or parser.error
    when a parser is supplied) otherwise."""
    if (args.gif or args.scrubber) and not args.jlens:
        msg = "--gif/--scrubber require --jlens"
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)


def _image_id_from_path(path: str) -> str:
    """<parent-dir-name>_<stem>, matching monocle.validate."""
    p = Path(path)
    return f"{p.parent.parent.name}_{p.stem}"


CYCLOMEDIA_RAW_ROOT = "/share/ju/cyclomedia/raw"  # catalog.RAW_ROOT


def _resolve_faces_by_path(
    recording_id: str, dataset: Optional[str], faces: list[str],
) -> list[ImageJob]:
    """Catalog-free fallback: build face paths from the deterministic layout
    ``{RAW_ROOT}/{dataset}/{group}/{recording_id}/faces/{F}.jpg`` where group
    is the recording-id prefix bucket. Needed on GPU nodes whose venv lacks
    duckdb (.venv-nightly). No NFS tree walking — at most one single-level
    glob when the prefix guess misses."""
    import glob

    if not dataset:
        raise RuntimeError(
            "--dataset is required with --recording-id when the cyclomedia "
            "catalog is unavailable (duckdb not installed in this venv)")
    rec_dir = Path(CYCLOMEDIA_RAW_ROOT) / dataset / recording_id[:5] / recording_id
    if not rec_dir.is_dir():
        hits = glob.glob(
            f"{CYCLOMEDIA_RAW_ROOT}/{dataset}/*/{recording_id}")
        if not hits:
            raise RuntimeError(
                f"recording dir not found under {CYCLOMEDIA_RAW_ROOT}/{dataset} "
                f"for {recording_id!r}")
        rec_dir = Path(hits[0])
    jobs = []
    for f in faces:
        p = rec_dir / "faces" / f"{f.upper()}.jpg"
        if p.is_file():
            jobs.append(ImageJob(image_id=f"{recording_id}_{f.upper()}", path=str(p)))
        else:
            log(f"warning: missing face {f.upper()} for {recording_id} ({p})")
    if not jobs:
        raise RuntimeError(f"no face images found in {rec_dir}/faces")
    return jobs


def _resolve_recording_faces(
    recording_id: str, dataset: Optional[str], faces: list[str],
) -> list[ImageJob]:
    """Face image paths for one cyclomedia recording, via the DuckDB catalog,
    falling back to deterministic path construction when duckdb is missing.

    Any catalog/DuckDB failure is wrapped in a clear RuntimeError.
    """
    try:
        from dagspaces.common.cyclomedia import catalog

        con = catalog.connect()
        df = catalog.faces_for_recording(con, recording_id, dataset=dataset)
    except (ImportError, ModuleNotFoundError):
        log("cyclomedia catalog unavailable (no duckdb); using path fallback")
        return _resolve_faces_by_path(recording_id, dataset, faces)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"cyclomedia catalog lookup failed for recording "
            f"{recording_id!r} (dataset={dataset!r}): {exc}") from exc

    wanted = [f.upper() for f in faces]
    df = df[df["face"].isin(wanted)]
    if df.empty:
        raise RuntimeError(
            f"no catalog faces for recording {recording_id!r} matching "
            f"{wanted} (dataset={dataset!r})")
    jobs = [
        ImageJob(image_id=f"{recording_id}_{row['face']}", path=str(row["image_path"]))
        for _, row in df.iterrows()
    ]
    return jobs


def collect_jobs(args: argparse.Namespace) -> list[ImageJob]:
    """Flatten the requested image sources into a list of ImageJob."""
    jobs: list[ImageJob] = []
    if args.image:
        # action="append" + nargs="+" yields a list of lists.
        for group in args.image:
            for path in group:
                jobs.append(ImageJob(image_id=_image_id_from_path(path), path=path))
    if args.recording_id:
        jobs.extend(_resolve_recording_faces(
            args.recording_id, args.dataset, args.faces))
    return jobs


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_jlens_args(args, parser)

    jobs = collect_jobs(args)
    if not jobs:
        log("no image sources: pass --image PATH and/or --recording-id ID")
        return 2

    out_dir = Path(args.out_dir)
    system = args.system or None
    log(f"{len(jobs)} image(s) -> {out_dir}"
        + (f" | system={system!r}" if system else "")
        + (f" | jlens={args.jlens}" if args.jlens else ""))

    proc, model, tmpl = extract.load_model(args.model_dir)
    tokenizer = proc.tokenizer
    token_mask = None  # built lazily once the vocab dim is known

    render = None
    if not args.no_render:
        from monocle import render as render  # noqa: PLC0414 (lazy: written concurrently)

    # Layer mode: load the fitted lens + unembed wrapper ONCE (jlens lives in
    # .venv-nightly, so these imports stay lazy — only touched with --jlens).
    lens = lens_model = None
    layers = None
    jlens_read = None
    render_layers = None
    if args.jlens:
        from monocle import jlens_read as jlens_read  # noqa: PLC0414 (lazy: needs jlens)
        lens = jlens_read.load_lens(args.jlens)
        lens_model = jlens_read.wrap_for_unembed(model, tokenizer)
        layers = parse_layers(args.layers)
        if render is not None and (args.gif or args.scrubber):
            from monocle import render_layers as render_layers  # noqa: PLC0414 (lazy: concurrent)

    written: list[Path] = []
    n = len(jobs)
    for i, job in enumerate(jobs, start=1):
        t0 = time.time()
        image = extract.open_image(job.path)
        image_id = job.image_id if args.pool <= 1 else f"{job.image_id}_p{args.pool}"

        if args.jlens:
            token_mask_ref = [token_mask]
            written += _run_layer_job(
                args, job, image, image_id, out_dir, proc, model, tmpl,
                tokenizer, lens, lens_model, layers, system, render,
                render_layers, token_mask_ref, i, n, t0)
            token_mask = token_mask_ref[0]
            continue

        logits, grid = extract.lens_image(
            proc, model, tmpl, image, system=system)

        if token_mask is None:
            token_mask = scoring.build_token_mask(tokenizer, logits.shape[-1])

        df = scoring.score_patches(
            logits, tokenizer, k=args.k, alpha=args.alpha, token_mask=token_mask,
            pool=args.pool, grid_shape=(grid.n_rows, grid.n_cols))
        vis_rows, vis_cols = scoring.pooled_dims(
            grid.n_rows, grid.n_cols, args.pool)
        df = scoring.attach_grid(df, vis_rows, vis_cols)

        meta = {
            **grid.to_meta(),
            # renderer reads n_rows/n_cols — point them at the pooled grid,
            # keep the model's native grid alongside
            "n_rows": vis_rows,
            "n_cols": vis_cols,
            "model_n_rows": grid.n_rows,
            "model_n_cols": grid.n_cols,
            "pool": args.pool,
            "k": args.k,
            "alpha": args.alpha,
            "model_dir": args.model_dir,
            "source_image": job.path,
        }
        if system:
            meta["system"] = system

        pq, mj = scoring.save_outputs(df, meta, out_dir, image_id)
        written += [pq, mj]

        if render is not None:
            overlay = render.render_overlay(image, df, meta, k=args.k)
            png = out_dir / f"{image_id}.overlay.png"
            overlay.save(png)
            written.append(png)
            if args.svg:
                svg = render.render_svg(image, df, meta, k=args.k)
                svg_path = out_dir / f"{image_id}.overlay.svg"
                svg_path.write_text(svg)
                written.append(svg_path)

        dt = time.time() - t0
        log(f"[{i}/{n}] {image_id}: model grid {grid.n_rows}x{grid.n_cols} via "
            f"'{grid.strategy}' -> display {vis_rows}x{vis_cols} "
            f"(pool {args.pool}), {len(df)} rows in {dt:.1f}s")

    log(f"done: wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        log(f"  {p}")
    return 0


def _run_layer_job(
    args, job, image, image_id, out_dir, proc, model, tmpl, tokenizer,
    lens, lens_model, layers, system, render, render_layers,
    token_mask_ref, i, n, t0,
) -> list[Path]:
    """One image through the Jacobian lens: score every layer, write a single
    long parquet (leading `layer` column), then per-layer overlays and the
    optional depth GIF / HTML scrubber. Returns the paths written.

    ``token_mask_ref`` is a 1-element list so the lazily-built mask is shared
    back with the caller across images (built once from the first layer)."""
    from monocle import jlens_read  # lazy: jlens is imported lazily within it

    written: list[Path] = []
    per_layer, grid = jlens_read.lens_image_layers(
        proc, model, tmpl, lens, lens_model, image, system=system, layers=layers)

    if token_mask_ref[0] is None:
        first_logits = next(iter(per_layer.values()))
        token_mask_ref[0] = scoring.build_token_mask(
            tokenizer, first_logits.shape[-1])
    token_mask = token_mask_ref[0]

    vis_rows, vis_cols = scoring.pooled_dims(grid.n_rows, grid.n_cols, args.pool)
    used_layers = sorted(per_layer)
    frames: list[pd.DataFrame] = []
    for layer in used_layers:
        df_l = scoring.score_patches(
            per_layer[layer], tokenizer, k=args.k, alpha=args.alpha,
            token_mask=token_mask, pool=args.pool,
            grid_shape=(grid.n_rows, grid.n_cols))
        df_l = scoring.attach_grid(df_l, vis_rows, vis_cols)
        df_l.insert(0, "layer", layer)
        frames.append(df_l)
    df = pd.concat(frames, ignore_index=True)

    meta = {
        **grid.to_meta(),
        "n_rows": vis_rows,
        "n_cols": vis_cols,
        "model_n_rows": grid.n_rows,
        "model_n_cols": grid.n_cols,
        "pool": args.pool,
        "k": args.k,
        "alpha": args.alpha,
        "model_dir": args.model_dir,
        "source_image": job.path,
    }
    if system:
        meta["system"] = system
    meta["layers"] = used_layers
    meta["lens_path"] = args.jlens

    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"{image_id}.jlens.parquet"
    mj = out_dir / f"{image_id}.jlens.meta.json"
    df.assign(image_id=image_id).to_parquet(pq, index=False)
    mj.write_text(json.dumps({"image_id": image_id, **meta}, indent=2))
    written += [pq, mj]

    if render is not None:
        for layer in used_layers:
            df_l = df[df["layer"] == layer]
            overlay = render.render_overlay(image, df_l, meta, k=args.k)
            png = out_dir / f"{image_id}_L{layer:02d}.overlay.png"
            overlay.save(png)
            written.append(png)
        if args.gif:
            gif_frames = render_layers.render_layer_gif(image, df, meta, k=args.k)
            gif_path = out_dir / f"{image_id}.depth.gif"
            render_layers.save_layer_gif(gif_frames, gif_path)
            written.append(gif_path)
        if args.scrubber:
            html = render_layers.render_layer_scrubber(image, df, meta, k=args.k)
            html_path = out_dir / f"{image_id}.scrubber.html"
            html_path.write_text(html)
            written.append(html_path)

    dt = time.time() - t0
    log(f"[{i}/{n}] {image_id}: model grid {grid.n_rows}x{grid.n_cols} -> "
        f"display {vis_rows}x{vis_cols} (pool {args.pool}), "
        f"{len(used_layers)} layers, {len(df)} rows in {dt:.1f}s")
    return written


if __name__ == "__main__":
    sys.exit(main())
