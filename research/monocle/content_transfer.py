"""Does the content carry its meaning into another photograph?

Why a transfer test
-------------------
Every test so far removes something and watches `c` fall or rise. That shows
the square MATTERS to this photograph. It does not show the square holds
danger, because a square can matter by holding the thing the rest of the scene
is read against.

Transfer asks the other half of the question. Take the squares that image A's
map calls Dangerous, and put THEIR PIXELS into image B at the same
coordinates. If those squares carry danger, B's judgment must fall.

| Arm | What goes into B | Prediction |
|---|---|---|
| `transfer_danger` | A's most Dangerous cells | `c(B)` FALLS |
| `transfer_safe` | A's most Safe cells | `c(B)` RISES |
| `transfer_random` | k random cells of A | near 0 |

The 2 targeted arms are matched: both are extreme cells of the SAME map of the
SAME photograph, moved the same distance, covering the same area. So a gap
between them cannot come from how much the pixels changed. It can only come
from WHICH pixels they are.

This is also the only test here with no circularity at all. The map is built
on A and judged on B.

The statistic
-------------
`transfer_separation = dc_safe(B) - dc_danger(B)`, against the spread of the
random transfers. Note the sign is the opposite way round from the deletion
test: there we TAKE AWAY the safe cells and `c` falls, here we PUT IN the safe
cells and `c` rises.
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

from research.monocle import deletion_curve as dcv  # noqa: E402
from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_map_causal as pmc  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/content_transfer"
DEFAULT_KS = (4, 8, 16)
DEFAULT_RANDOM = 6
DEFAULT_GROUP = 2


def log(m: str) -> None:
    print(f"[transfer] {m}", flush=True)


def transfer_sets(
    values: np.ndarray, k: int, rng: np.random.Generator, n_random: int,
) -> dict[str, list[np.ndarray]]:
    """Cells of the SOURCE map, by what the source map calls them."""
    safe, danger = dcv.rank_cells(values)
    pool = np.arange(values.size)
    return {
        "transfer_safe": [safe[:k]],
        "transfer_danger": [danger[:k]],
        "transfer_random": [rng.choice(pool, size=k, replace=False)
                            for _ in range(n_random)],
    }


def dispersion(cells, side: int) -> float:
    """Mean distance between the cells of a set, in grid steps.

    A confound the random arm cannot control on its own: the top and the
    bottom of a map are often CLUSTERED (the sky, the road), while a random
    draw is scattered. A scattered paste disturbs a picture more than a
    contiguous one, whatever it holds. So the dispersion goes in the frame and
    the analysis can hold it fixed.
    """
    idx = [int(c) for c in cells]
    if len(idx) < 2:
        return 0.0
    rc = np.array([(i // side, i % side) for i in idx], dtype=float)
    d = np.linalg.norm(rc[:, None, :] - rc[None, :, :], axis=-1)
    iu = np.triu_indices(len(idx), 1)
    return float(d[iu].mean())


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (target, source, k): the arms and the separation."""
    out = []
    keys = ["target_id", "source_id", "k"]
    for (target, source, k), sub in df.groupby(keys):
        rnd = sub[sub.arm == "transfer_random"].dc
        safe = float(sub[sub.arm == "transfer_safe"].dc.iloc[0])
        danger = float(sub[sub.arm == "transfer_danger"].dc.iloc[0])
        sd = float(rnd.std(ddof=1)) if len(rnd) > 1 else np.nan
        out.append({
            "target_id": target, "source_id": source, "k": int(k),
            "c_target": float(sub.c_target.iloc[0]),
            "c_source": float(sub.c_source.iloc[0]),
            "target_worth_ok": bool(sub.target_worth_ok.iloc[0]),
            "dc_safe": safe, "dc_danger": danger,
            "dc_random": float(rnd.mean()), "random_sd": sd,
            "separation": safe - danger,
            "separation_z": (safe - danger) / sd if sd and sd > 0 else np.nan,
            "safe_sign_ok": safe > 0,
            "danger_sign_ok": danger < 0,
            "disp_safe": float(sub[sub.arm == "transfer_safe"]
                               .dispersion.iloc[0]),
            "disp_danger": float(sub[sub.arm == "transfer_danger"]
                                 .dispersion.iloc[0]),
            "disp_random": float(sub[sub.arm == "transfer_random"]
                                 .dispersion.mean()),
        })
    return pd.DataFrame(out)


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, group: int, ks: tuple, n_random: int, seed: int,
    batch: int, out_dir: Path, device: str, n_sources: int,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rng = np.random.default_rng(seed)
    blocks = None

    store = []
    for r in sg.load_single_rows(n_images, case, seed=seed):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        slots = pab.slots_of(inputs, image_token_id)
        c0 = float(pab.contrast(model, lens_model, inputs, pos_id, neg_id)[0])
        ctl = pab.controls(model, lens_model, inputs, slots, c0, pos_id,
                           neg_id, image_token_id=image_token_id)
        pab.gate_rebuild(ctl)
        values = -pab.ablation_map(
            model, lens_model, inputs, slots, c0, pos_id, neg_id,
            batch=batch, group=group, image_token_id=image_token_id)
        if blocks is None:
            blocks = (pmc.patch_groups(len(slots), group) if group > 1
                      else [[i] for i in range(len(slots))])
        store.append({**r, "inputs": inputs, "slots": slots, "c": c0,
                      "values": values, "worth": ctl["image_worth"],
                      "worth_ok": bool(ctl["worth_ok"])})
        log(f"  mapped {len(store)}/{n_images} {r['image_id']} "
            f"c={c0:+.2f} worth={ctl['image_worth']:+.2f}")
    pab.gate_battery([s["worth"] for s in store])

    rows = []
    for i, tgt in enumerate(store):
        pool = [j for j in range(len(store)) if j != i]
        picks = rng.choice(pool, size=min(n_sources, len(pool)),
                           replace=False)
        for j in picks:
            src = store[int(j)]
            donor_px = src["inputs"]["pixel_values"][0]
            for k in ks:
                if k > src["values"].size:
                    continue
                sets = transfer_sets(src["values"], k, rng, n_random)
                side = int(round(src["values"].size ** 0.5))
                flat, tags, disp = [], [], []
                for arm, draws in sets.items():
                    for d, cells in enumerate(draws):
                        flat.append(dcv.cells_to_patches(cells, blocks))
                        tags.append((arm, d))
                        disp.append(dispersion(cells, side))
                cs = dcv.measure_dc(
                    model, lens_model, tgt["inputs"], tgt["slots"], flat,
                    pos_id, neg_id, image_token_id, batch, "substitute",
                    donor_px)
                # `cs` already holds changes, against the no-op row that
                # rides in each batch. Do not subtract the target's `c` again.
                for (arm, d), delta, sp in zip(tags, cs, disp):
                    rows.append({
                        "case": case, "target_id": tgt["image_id"],
                        "source_id": src["image_id"], "k": k, "arm": arm,
                        "draw": d, "group": group, "dispersion": sp,
                        "c_target": tgt["c"], "c_source": src["c"],
                        "target_worth_ok": tgt["worth_ok"],
                        "c": tgt["c"] + float(delta),
                        "dc": float(delta)})
        log(f"  transferred into {i + 1}/{len(store)} {tgt['image_id']}")

    df = pd.DataFrame(rows)
    case_dir = out_dir / f"{case}_g{group}"
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "arms.parquet", index=False)
    s = summarize(df)
    s.to_parquet(case_dir / "summary.parquet", index=False)

    strong = s[s.target_worth_ok]
    log(f"{case}: {strong.target_id.nunique()} strong targets, "
        f"{len(strong)} (target, source, k) triples")
    log(f"{'k':>4} {'dc(safe in)':>12} {'dc(random in)':>14} "
        f"{'dc(danger in)':>14} {'separation':>11} {'z':>7} {'signs ok':>10}")
    for k, sub in strong.groupby("k"):
        signs = (f"{int(sub.safe_sign_ok.sum())}/"
                 f"{int(sub.danger_sign_ok.sum())} of {len(sub)}")
        log(f"{k:>4} {sub.dc_safe.median():>12.3f} "
            f"{sub.dc_random.median():>14.3f} {sub.dc_danger.median():>14.3f} "
            f"{sub.separation.median():>11.3f} "
            f"{sub.separation_z.median():>7.2f} {signs:>10}")
    (case_dir / "verdict.json").write_text(json.dumps({
        "case": case, "group": group,
        "n_targets": int(strong.target_id.nunique()),
        "median_separation": {int(k): float(v) for k, v in
                              strong.groupby("k").separation.median().items()},
        "median_separation_z": {int(k): float(v) for k, v in
                                strong.groupby("k").separation_z.median()
                                .items()},
    }, indent=2))
    return s


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Transfer content between images.")
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--n-sources", type=int, default=2,
                    help="source images for each target")
    ap.add_argument("--group", type=int, default=DEFAULT_GROUP)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--n-random", type=int, default=DEFAULT_RANDOM)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar

    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    for case in args.cases:
        run_case(case, proc, model, lens_model, tmpl,
                 n_images=args.n_images, group=args.group,
                 ks=tuple(args.ks), n_random=args.n_random, seed=args.seed,
                 batch=args.batch, out_dir=out_dir, device=args.device,
                 n_sources=args.n_sources)
    log("content transfer complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
