"""What is IN the square that carries the judgment?

Why not the lens
----------------
The lens read of a causal patch gives token pieces like `gautam` and `ghaz`.
It separates a causal patch from a random one, so the patch is different, but
no reader can say WHAT is there. The lens says WHERE and SIGN, not WHAT.

The direct read
---------------
Cut the square out of the photograph and ask the model to describe it. The
model owns the vocabulary, so the answer is in words a person can read. The
crop keeps a ring of neighbour cells, because 1 cell of a 16x16 grid is 56
pixels and a bare 56-pixel square shows almost nothing.

| Arm | The square that is cut out |
|---|---|
| `pos` | the map's highest cell |
| `neg` | the map's lowest cell |
| `random` | arbitrary cells of the same photograph |

The random arm comes from the SAME photograph, so the scene, the camera and
the light are held fixed. Only the place in the frame changes.

The confound this cannot remove on its own
------------------------------------------
A pole cell may sit low in the frame while a random cell sits high. Then the
words differ by position, not by cause. The row and the column of every crop
go in the frame, so the analysis can hold them fixed.

The words go to `research/scripts/causal_word_report.py`, which runs the same Monroe
log-odds and the same within-photograph permutation test as the lens read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import caption_loop as cl  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import single as sg  # noqa: E402
from research.monocle import transfer_k1 as tk  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/patch_caption"
MAPS_DEFAULT = "outputs/_monocle/gateB60_g1L/{case}/maps.parquet"
CROP_QUESTION = ("Name the things you can see in this picture. "
                 "Give a short list.")


def log(m: str) -> None:
    print(f"[patch-caption] {m}", flush=True)


def crop_box(cell: int, side: int, size: int, pad: int) -> tuple:
    """The pixel box of a cell, with `pad` cells of context on each side.

    The box is clipped at the edge of the picture, so a corner cell gives a
    smaller crop than a middle one. The area goes in the frame, because a
    smaller crop holds fewer things and would name fewer words.
    """
    step = size / side
    r, c = divmod(int(cell), side)
    x0 = max(0, int(round((c - pad) * step)))
    y0 = max(0, int(round((r - pad) * step)))
    x1 = min(size, int(round((c + 1 + pad) * step)))
    y1 = min(size, int(round((r + 1 + pad) * step)))
    return x0, y0, x1, y1


def crop_of(img: Image.Image, cell: int, side: int, pad: int,
            out_size: int = 896, upscale: bool = True) -> tuple:
    """The crop, and the box it came from, on a square copy of the picture.

    With `upscale` the crop fills `out_size` pixels, which is about 5 times
    its own width. Without it the crop keeps its own pixels and sits in the
    middle of a grey canvas. The 2 modes test whether a word comes from the
    content of the crop or from the size we show it at.
    """
    square = img.resize((out_size, out_size), Image.BICUBIC)
    box = crop_box(cell, side, out_size, pad)
    piece = square.crop(box)
    if upscale:
        return piece.resize((out_size, out_size), Image.BICUBIC), box
    canvas = Image.new("RGB", (out_size, out_size), (128, 128, 128))
    canvas.paste(piece, ((out_size - piece.width) // 2,
                         (out_size - piece.height) // 2))
    return canvas, box


def pick_cells(values: np.ndarray, n_random: int,
               rng: np.random.Generator) -> tuple:
    order = np.argsort(values)
    pos_cell, neg_cell = int(order[-1]), int(order[0])
    rest = [c for c in range(values.size) if c not in (pos_cell, neg_cell)]
    rnd = rng.choice(rest, size=min(n_random, len(rest)), replace=False)
    cells = [pos_cell, neg_cell] + [int(c) for c in rnd]
    arms = ["pos", "neg"] + ["random"] * len(rnd)
    return cells, arms


def dense_cells(values: np.ndarray, side: int, step: int) -> tuple:
    """A grid subsample of the cells, plus the 2 poles, named as such.

    A pole-against-random test gives 1 contrast. A grid gives the whole
    range of the causal value, so the caption score can be regressed on it
    inside a photograph.
    """
    grid = [r * side + c
            for r in range(0, side, step) for c in range(0, side, step)]
    order = np.argsort(values)
    pos_cell, neg_cell = int(order[-1]), int(order[0])
    cells = [pos_cell, neg_cell] + [c for c in grid
                                    if c not in (pos_cell, neg_cell)]
    arms = ["pos", "neg"] + ["dense"] * (len(cells) - 2)
    return cells, arms


def run_case(
    case: str, proc: Any, model: Any, tmpl: str, *, n_images: int,
    n_random: int, seed: int, pad: int, max_new: int,
    maps: dict[str, np.ndarray], pick: str, out_dir: Path, device: str,
    out_size: int = 896, upscale: bool = True, dense: int = 0,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    done = 0
    for r in sg.load_single_rows(n_images, case, seed=seed):
        key = str(r["image_id"])
        if key not in maps:
            continue
        values = maps[key]
        side = int(round(float(values.size) ** 0.5))
        if side * side != values.size:
            raise SystemExit(f"{values.size} cells is not a square grid")
        img = Image.open(r["path"]).convert("RGB")
        if dense:
            cells, arms = dense_cells(values, side, dense)
        else:
            cells, arms = pick_cells(values, n_random, rng)
        for arm, cell in zip(arms, cells):
            piece, box = crop_of(img, cell, side, pad, out_size,
                                 upscale=upscale)
            inputs = cl.caption_inputs(proc, tmpl, piece, device)
            text = cl.generate(model, proc, inputs, max_new)
            rr, cc = divmod(int(cell), side)
            rows.append({
                "case": case, "pick": pick, "image_id": key, "arm": arm,
                "cell": int(cell), "row": rr, "col": cc,
                "value": float(values[cell]),
                "box_area": int((box[2] - box[0]) * (box[3] - box[1])),
                "pad": int(pad), "out_size": int(out_size),
                "upscale": bool(upscale),
                "text": text.strip()})
        done += 1
        log(f"  {done} {key}: pos={rows[-len(cells)]['text'][:60]!r}")
    df = pd.DataFrame(rows)
    case_dir = out_dir / (tag or f"{case}_{pick}")
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "captions.parquet", index=False)
    (case_dir / "verdict.json").write_text(json.dumps({
        "case": case, "pick": pick, "n_images": int(df.image_id.nunique()),
        "n_crops": int(len(df)), "pad": pad,
        "mean_words": float(df.text.str.split().str.len().mean()),
    }, indent=2))
    log(f"{case}: {df.image_id.nunique()} photographs, {len(df)} crops")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Describe the causal square.")
    ap.add_argument("--cases", nargs="+", default=["restaurants"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--maps", default=MAPS_DEFAULT)
    ap.add_argument("--pick", default="causal",
                    choices=("causal",) + tk.INSTRUMENTS)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--pad", type=int, default=1,
                    help="cells of context on each side of the cell")
    ap.add_argument("--out-size", type=int, default=896,
                    help="the pixel size the model reads")
    ap.add_argument("--no-upscale", action="store_true",
                    help="keep the crop's own pixels on a grey canvas")
    ap.add_argument("--dense", type=int, default=0,
                    help="caption every Nth row and column, not 6 cells")
    ap.add_argument("--tag", default=None,
                    help="the output folder name")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar

    args = build_parser().parse_args(argv)
    cl.CAPTION_QUESTION = CROP_QUESTION
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    tag = args.pick if args.layer is None else f"{args.pick}_L{args.layer}"
    for case in args.cases:
        path = REPO / args.maps.format(case=case)
        maps = tk.load_maps(path, args.pick, args.layer)
        log(f"{case}: {len(maps)} stored maps from {path}")
        run_case(case, proc, model, tmpl, n_images=args.n_images,
                 n_random=args.n_random, seed=args.seed, pad=args.pad,
                 max_new=args.max_new, maps=maps, pick=tag,
                 out_dir=out_dir, device=args.device,
                 out_size=args.out_size, upscale=not args.no_upscale,
                 dense=args.dense,
                 tag=(f"{case}_{args.tag}" if args.tag else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
