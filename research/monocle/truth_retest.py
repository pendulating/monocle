"""How much of the measured map comes back on a second measurement?

Why this matters
----------------
Every number in the 1-cell battery is scored against the measured map. The
map is the ground truth of the whole study, and no one has measured it
twice. If the map itself does not come back, then a low agreement from an
instrument says nothing about the instrument.

Two things could move it. The batch size moves `c` by up to 0.67 logits on
gemma-4-12b, and the cells that share a chunk change with the cell order.
The map subtracts a zero measured at the same batch size, so a constant
offset cancels. A chunk-dependent offset does not.

What this runs
--------------
For each photograph the map is measured again, with a DIFFERENT batch size
and a SHUFFLED cell order, and then put back in grid order. The new map is
compared with the stored one.

| Statistic | What it bounds |
|---|---|
| `pearson`, `spearman` | how well any instrument could rank the map |
| `top1_same` | how often the strongest cell is the same cell |
| `sep_z` | the ceiling of the 1-cell separation |

`sep_z` is the same statistic the battery reports, with the fresh map used
as the instrument and the stored map used as the truth. It is the honest
ceiling: no instrument can beat a second measurement of the thing itself.
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

from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/truth_retest"
MAPS_DEFAULT = "outputs/_monocle/gateB60_g1L/{case}/maps.parquet"


def log(m: str) -> None:
    print(f"[retest] {m}", flush=True)


def shuffled_map(
    model: Any, lens_model: Any, inputs: dict, slots: list[int],
    pos_id: int, neg_id: int, image_token_id: int, batch: int,
    rng: np.random.Generator, shuffle: bool = True,
) -> np.ndarray:
    """The 1-cell ablation map, measured in a random cell order.

    The order changes which cells share a chunk. A map that depends on the
    chunk shows up as a disagreement with the stored map.
    """
    n = len(slots)
    order = rng.permutation(n) if shuffle else np.arange(n)
    zero = float(pab.contrast(
        model, lens_model,
        pab.ablation_batch(inputs, slots, [[]] * batch, image_token_id),
        pos_id, neg_id)[0])
    deltas = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch):
        cells = [int(c) for c in order[start:start + batch]]
        chunk = [[c] for c in cells]
        while len(chunk) < batch:
            chunk.append(chunk[-1])
        rows = pab.ablation_batch(inputs, slots, chunk, image_token_id)
        cs = pab.contrast(model, lens_model, rows, pos_id, neg_id)
        for c, v in zip(cells, cs[:len(cells)]):
            deltas[c] = float(v) - zero
    # `causal_maps` stores a CONTRIBUTION, which is the removal delta
    # with the sign flipped. The retest must store the same thing, or
    # the 2 maps correlate at -1 and the run looks broken.
    return -deltas


def compare(fresh: np.ndarray, stored: np.ndarray) -> dict:
    """The agreement between 2 measurements of one map."""
    from scipy.stats import pearsonr, spearmanr
    a, b = np.asarray(fresh, float), np.asarray(stored, float)
    sd = float(np.std(b, ddof=1))
    sep = float("nan")
    if np.isfinite(sd) and sd > 0:
        sep = float(b[int(np.argmax(a))] - b[int(np.argmin(a))]) / sd
    return {
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "top1_same": bool(int(np.argmax(a)) == int(np.argmax(b))),
        "bottom1_same": bool(int(np.argmin(a)) == int(np.argmin(b))),
        "sep_z": sep,
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "sd_stored": sd,
        "sd_fresh": float(np.std(a, ddof=1)),
    }


def load_stored(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    col = "causal" if "causal" in df.columns else "values"
    idc = "target" if "target" in df.columns else "image_id"
    return {str(k): np.asarray(g.iloc[0][col], dtype=float).ravel()
            for k, g in df.groupby(idc)}


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, seed: int, batch: int, stored: dict[str, np.ndarray],
    out_dir: Path, device: str, shuffle: bool = True, tag: str = "",
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rng = np.random.default_rng(seed + 1)

    rows: list[dict] = []
    done = 0
    for r in sg.load_single_rows(60, case, seed=seed):
        key = str(r["image_id"])
        if key not in stored or done >= n_images:
            continue
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        slots = pab.slots_of(inputs, image_token_id)
        fresh = shuffled_map(model, lens_model, inputs, slots, pos_id,
                             neg_id, image_token_id, batch, rng,
                             shuffle=shuffle)
        st = compare(fresh, stored[key])
        rows.append({"case": case, "image_id": key, "batch": batch,
                     "fresh": fresh, "stored": stored[key], **st})
        done += 1
        log(f"  {done}/{n_images} {key} r={st['pearson']:.3f} "
            f"rho={st['spearman']:.3f} top1={st['top1_same']} "
            f"sep_z={st['sep_z']:+.2f}")
    df = pd.DataFrame(rows)
    case_dir = out_dir / (f"{case}_{tag}" if tag else case)
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "retest.parquet", index=False)
    v = {
        "case": case, "n": int(len(df)), "batch": batch,
        "mean_pearson": float(df.pearson.mean()),
        "mean_spearman": float(df.spearman.mean()),
        "top1_same": float(df.top1_same.mean()),
        "bottom1_same": float(df.bottom1_same.mean()),
        "mean_sep_z": float(df.sep_z.mean()),
        "median_max_abs_diff": float(df.max_abs_diff.median()),
        "mean_sd_stored": float(df.sd_stored.mean()),
    }
    (case_dir / "verdict.json").write_text(json.dumps(v, indent=2))
    log(f"{case}: r={v['mean_pearson']:.3f} rho={v['mean_spearman']:.3f} "
        f"top1={v['top1_same']:.2f} sep_z={v['mean_sep_z']:+.2f} "
        f"max|diff|={v['median_max_abs_diff']:.3f}")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Measure the map a second time.")
    ap.add_argument("--cases", nargs="+", default=["restaurants"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--maps", default=MAPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--no-shuffle", action="store_true",
                    help="keep the stored cell order")
    ap.add_argument("--tag", default="",
                    help="a suffix for the output directory")
    ap.add_argument("--batch", type=int, default=7,
                    help="a size the stored run did NOT use")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar

    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    for case in args.cases:
        path = REPO / args.maps.format(case=case)
        stored = load_stored(path)
        log(f"{case}: {len(stored)} stored maps from {path}")
        run_case(case, proc, model, lens_model, tmpl,
                 n_images=args.n_images, seed=args.seed, batch=args.batch,
                 stored=stored, out_dir=out_dir,
                 device=args.device,
                 shuffle=not args.no_shuffle, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
