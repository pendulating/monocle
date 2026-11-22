"""Do the colours mean anything? A directed deletion test.

The question
------------
A causal map paints a cell coral or teal, and a reader reads "the model finds
this square dangerous". The measurement behind the colour does not say that.
It says `c` moved when the square went away, and `c` can move because of the
content, because of the hole, or because of noise.

The test
--------
Rank the cells of 1 map. Then take away several cells AT ONCE and watch `c`.

| Arm | The cells | Prediction if the colour is real |
|---|---|---|
| `top_safe` | the k that push hardest toward the positive label | `c` FALLS |
| `top_danger` | the k that push hardest toward the negative label | `c` RISES |
| `random` | k drawn at random, several draws | in between, near 0 |

The random arm is the whole point. It holds the hole size fixed, so it
measures what a hole of that size does on its own. A map that only reports
"a hole moved the judgment" gives 3 arms that lie on top of each other.

This is not circular. The ranking comes from cells taken away ONE at a time;
the test takes them away TOGETHER, and we already know the parts do not sum to
the whole. So the joint effect is new information about the ranking.

Reading the result
------------------
`separation = dc(top_danger) - dc(top_safe)`, in logit units, against the
spread of the random draws. A separation inside the random spread is a map
with no directional content, whatever its colours look like.
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
from research.monocle import patch_map_causal as pmc  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/deletion_curve"
#: Cells to take away. 32 of 64 is half the picture, which is the point where
#: the arms must run out of room to differ.
DEFAULT_KS = (1, 2, 4, 8, 16, 32)
#: Draws of the random arm. The spread of these is the yardstick.
DEFAULT_RANDOM = 8
#: The map is read at 2 x 2 cells: 1 patch is too small to hold a thing.
DEFAULT_GROUP = 2


def log(m: str) -> None:
    print(f"[deletion] {m}", flush=True)


def rank_cells(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cells that push toward the positive label, toward the negative).

    `values` is the contribution toward the POSITIVE label, so the highest
    entries hold the positive label up and the lowest push against it. Each
    order runs from the strongest cell down.
    """
    order = np.argsort(values)
    return order[::-1].copy(), order.copy()


def arm_sets(
    values: np.ndarray, k: int, rng: np.random.Generator, *,
    n_random: int = DEFAULT_RANDOM,
) -> dict[str, list[np.ndarray]]:
    """The cell sets of each arm, all of size `k`."""
    if k < 1 or k > values.size:
        raise ValueError(f"k must be in 1..{values.size}, got {k}")
    safe, danger = rank_cells(values)
    pool = np.arange(values.size)
    return {
        "top_safe": [safe[:k]],
        "top_danger": [danger[:k]],
        "random": [rng.choice(pool, size=k, replace=False)
                   for _ in range(n_random)],
    }


def cells_to_patches(cells, blocks: list[list[int]]) -> list[int]:
    """Every patch index under a set of group cells."""
    out: list[int] = []
    for c in cells:
        out.extend(blocks[int(c)])
    return sorted(set(out))


def run_image(
    model: Any, lens_model: Any, inputs: dict, slots: list[int], c0: float,
    values: np.ndarray, pos_id: int, neg_id: int, *,
    group: int, ks: tuple, rng: np.random.Generator, n_random: int,
    image_token_id: int, batch: int, mode: str = "ablate",
    donors: Optional[list] = None, keep_only: bool = False,
) -> list[dict]:
    """1 row for each (k, arm, draw). `values` is the map, in cell order.

    `mode` picks the test intervention:

    | mode | The edit | Rules out |
    |---|---|---|
    | `ablate` | the square goes away | nothing about holes |
    | `substitute` | the square holds a donor's pixels | the hole reading |

    Under `substitute` the effect is averaged over the donors, so the row
    reports what OTHER content does in that square, not what 1 donor does.
    """
    if mode not in ("ablate", "substitute"):
        raise ValueError(f"mode must be ablate or substitute, got {mode!r}")
    if mode == "substitute" and not donors:
        raise ValueError("the substitute mode needs donor pixels")
    blocks = (pmc.patch_groups(len(slots), group) if group > 1
              else [[i] for i in range(len(slots))])
    if len(blocks) != values.size:
        raise ValueError(
            f"the map holds {values.size} cells but a group of {group} makes "
            f"{len(blocks)} blocks")

    rows = []
    for k in ks:
        if k > values.size:
            continue
        arms = arm_sets(values, k, rng, n_random=n_random)
        # Every set of this k removes the same patch count, so 1 batch serves
        # them all. `ablation_batch` refuses a mixed size, which is the guard.
        flat, tags = [], []
        every = set(range(values.size))
        for arm, draws in arms.items():
            for j, cells in enumerate(draws):
                # KEEP-ONLY inverts the edit: everything BUT these cells goes.
                # The deletion arms ask whether the cells are NECESSARY. This
                # asks whether they are SUFFICIENT, which no removal can.
                chosen = (sorted(every - set(int(c) for c in cells))
                          if keep_only else cells)
                flat.append(cells_to_patches(chosen, blocks))
                tags.append((arm, j))
        if mode == "ablate":
            cs = np.array(measure_dc(model, lens_model, inputs, slots, flat,
                                   pos_id, neg_id, image_token_id, batch,
                                   "ablate", None))
        else:
            per = [np.array(measure_dc(model, lens_model, inputs, slots, flat,
                                     pos_id, neg_id, image_token_id, batch,
                                     "substitute", d)) for d in donors]
            cs = np.stack(per).mean(0)
        for (arm, j), patches, d in zip(tags, flat, cs):
            # `d` is already a change against the no-op of its own batch.
            rows.append({"k": k, "arm": arm, "draw": j,
                         "n_patches": len(patches), "mode": mode,
                         "keep_only": keep_only,
                         "c": c0 + float(d), "dc": float(d)})
    return rows


def measure_dc(
    model: Any, lens_model: Any, inputs: dict, slots: list[int],
    flat: list, pos_id: int, neg_id: int, image_token_id: int, batch: int,
    mode: str, donor_pixels,
) -> list[float]:
    """The CHANGE in `c` for each patch set, under 1 intervention.

    Named `_dc` because it already subtracts a baseline: the no-op row that
    rides in each chunk. A caller that subtracts its own `c0` again would
    double-count, and would reintroduce the batch-size offset it removes.
    """
    def measure(sets):
        if mode == "ablate":
            rowset = pab.ablation_batch(inputs, slots, sets, image_token_id)
        else:
            rowset = pab.substitution_batch(
                inputs, donor_pixels, sets, image_token_id)
        return pab.contrast(model, lens_model, rowset, pos_id, neg_id)

    # Batch size alone moves `c` by up to 0.67 logits, so every batch is the
    # same size and the zero is measured at that size. An ablation shortens
    # the sequence, so a no-op row cannot share a batch with an ablated one.
    zero = float(measure([[]] * batch)[0])
    out: list[float] = []
    for start in range(0, len(flat), batch):
        chunk = list(flat[start:start + batch])
        n = len(chunk)
        while len(chunk) < batch:
            chunk.append(chunk[-1])
        cs = measure(chunk)
        out.extend(float(x - zero) for x in cs[:n])
    return out


def summarize(df: pd.DataFrame, keep_only: bool = False) -> pd.DataFrame:
    """Per (image, k): the 3 arms, the random spread, and the separation.

    The sign predictions INVERT between the 2 edits, so the caller must say
    which one ran:

    | Edit | top_safe cells | top_danger cells | separation |
    |---|---|---|---|
    | delete | `c` falls | `c` rises | danger - safe |
    | keep only | `c` stays high | `c` stays low | safe - danger |

    Either way `separation` is positive when the map is right, so the 2 runs
    read on the same scale.
    """
    out = []
    for (image_id, k), sub in df.groupby(["image_id", "k"]):
        rnd = sub[sub.arm == "random"].dc
        safe = float(sub[sub.arm == "top_safe"].dc.iloc[0])
        danger = float(sub[sub.arm == "top_danger"].dc.iloc[0])
        sd = float(rnd.std(ddof=1)) if len(rnd) > 1 else np.nan
        sep = (safe - danger) if keep_only else (danger - safe)
        out.append({
            "image_id": image_id, "k": int(k), "keep_only": keep_only,
            "image_worth": float(sub.image_worth.iloc[0]),
            "worth_ok": bool(sub.worth_ok.iloc[0]),
            "dc_safe": safe, "dc_danger": danger,
            "dc_random": float(rnd.mean()), "random_sd": sd,
            "separation": sep,
            # How many random spreads apart the 2 targeted arms sit. Under 2
            # the ranking tells us nothing a random hole would not.
            "separation_z": sep / sd if sd and sd > 0 else np.nan,
            # The signs the colour promises, in the direction of this edit.
            "safe_sign_ok": (safe > float(rnd.mean())) if keep_only
            else (safe < 0),
            "danger_sign_ok": (danger < float(rnd.mean())) if keep_only
            else (danger > 0),
        })
    return pd.DataFrame(out)


def load_ranking(path: Path, group: int) -> dict[str, np.ndarray]:
    """A map from another run, to rank the cells with. `image_id` -> values.

    This is what makes the test non-circular. Rank with the DONOR map, which
    swaps a layer-20 residual, and test with the ABSENCE deletion, which edits
    the input. The 2 interventions share no machinery, so agreement between
    them cannot come from the method.
    """
    df = pd.read_parquet(path)
    for col in ("image_id", "values"):
        if col not in df.columns:
            raise SystemExit(f"{path} holds no {col} column")
    out = {}
    for i, image_id in enumerate(df["image_id"]):
        v = np.array(df["values"].iloc[i], dtype=np.float64)
        out[str(image_id)] = v
    sizes = {v.size for v in out.values()}
    if len(sizes) != 1:
        raise SystemExit(f"{path} mixes map sizes {sorted(sizes)}")
    want = 256 // (group * group)
    if sizes.pop() != want:
        raise SystemExit(
            f"{path} holds maps of a different grid; a group of {group} "
            f"needs {want} cells")
    return out


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, group: int, ks: tuple, n_random: int, seed: int,
    batch: int, out_dir: Path, device: str,
    rank_from: Optional[Path] = None, tag: str = "self",
    mode: str = "ablate", n_donors: int = 3, keep_only: bool = False,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rng = np.random.default_rng(seed)
    rows = []
    ranking = load_ranking(rank_from, group) if rank_from else None
    if ranking is not None:
        log(f"ranking from {rank_from} ({len(ranking)} images)")

    # The substitute mode needs other images' pixels, so every image is built
    # first and the test runs after.
    store = []
    for r in sg.load_single_rows(n_images, case, seed=seed):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        store.append({**r, "inputs": inputs})
    log(f"{case}: {len(store)} images built, mode={mode}")

    for i, r in enumerate(store):
        inputs = r["inputs"]
        slots = pab.slots_of(inputs, image_token_id)
        c0 = float(pab.contrast(model, lens_model, inputs, pos_id, neg_id)[0])
        # `measure_dc` reads every change against a no-op row in a batch of
        # `batch`, so the whole is measured there too. See `pab.controls`.
        ctl = pab.controls(model, lens_model, inputs, slots, c0, pos_id,
                           neg_id, image_token_id=image_token_id, batch=batch)
        pab.gate_rebuild(ctl)
        donors = None
        if mode == "substitute":
            pool = [j for j in range(len(store)) if j != i]
            pick = rng.choice(pool, size=min(n_donors, len(pool)),
                              replace=False)
            donors = [store[int(j)]["inputs"]["pixel_values"][0]
                      for j in pick]
        if ranking is not None:
            if r["image_id"] not in ranking:
                log(f"  {r['image_id']} is not in the ranking file; skipped")
                continue
            values = ranking[r["image_id"]]
        else:
            # The map this test judges, from cells taken 1 at a time.
            values = -pab.ablation_map(
                model, lens_model, inputs, slots, c0, pos_id, neg_id,
                batch=batch, group=group, image_token_id=image_token_id)
        got = run_image(
            model, lens_model, inputs, slots, c0, values, pos_id, neg_id,
            group=group, ks=ks, rng=rng, n_random=n_random,
            image_token_id=image_token_id, batch=batch, mode=mode,
            donors=donors, keep_only=keep_only)
        for g in got:
            g.update(case=case, image_id=r["image_id"], path=r["path"],
                     c0=c0, group=group, rank_source=tag,
                     n_donors=0 if donors is None else len(donors),
                     image_worth=ctl["image_worth"],
                     worth_ok=bool(ctl["worth_ok"]))
        rows.extend(got)
        log(f"  {i + 1}/{n_images} {r['image_id']} worth={ctl['image_worth']:+.2f}"
            f"{'' if ctl['worth_ok'] else '  [weak image]'}")

    df = pd.DataFrame(rows)
    case_dir = out_dir / (f"{case}_g{group}_{tag}_{mode}"
                          + ("_keep" if keep_only else ""))
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "arms.parquet", index=False)
    s = summarize(df, keep_only=keep_only)
    s.to_parquet(case_dir / "summary.parquet", index=False)

    strong = s[s.worth_ok]
    log(f"{case}: {strong.image_id.nunique()} strong images "
        f"({s.image_id.nunique() - strong.image_id.nunique()} weak, left out)")
    what = "keep only" if keep_only else "delete"
    log(f"edit: {what}. separation is positive when the map is right.")
    log(f"{'k':>4} {'dc(safe)':>10} {'dc(random)':>11} {'dc(danger)':>11} "
        f"{'separation':>11} {'z':>7} {'signs ok':>9}")
    for k, sub in strong.groupby("k"):
        signs = f"{int(sub.safe_sign_ok.sum())}/{int(sub.danger_sign_ok.sum())}"
        log(f"{k:>4} {sub.dc_safe.median():>10.3f} "
            f"{sub.dc_random.median():>11.3f} {sub.dc_danger.median():>11.3f} "
            f"{sub.separation.median():>11.3f} "
            f"{sub.separation_z.median():>7.2f} {signs:>9}")
    (case_dir / "verdict.json").write_text(json.dumps({
        "case": case, "group": group, "n_images": int(s.image_id.nunique()),
        "n_strong": int(strong.image_id.nunique()),
        "median_separation_z": {int(k): float(v) for k, v in
                                strong.groupby("k").separation_z.median()
                                .items()},
    }, indent=2))
    return s


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Directed deletion test.")
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--group", type=int, default=DEFAULT_GROUP)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--n-random", type=int, default=DEFAULT_RANDOM)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank-from", default=None,
                    help="a maps.parquet to rank the cells with. Use the "
                         "DONOR map here to break the circularity.")
    ap.add_argument("--tag", default="self",
                    help="names the output folder, e.g. `donor`")
    ap.add_argument("--mode", default="ablate",
                    choices=["ablate", "substitute"],
                    help="ablate: the cells go away. substitute: the cells "
                         "hold other images' pixels, which rules out the "
                         "reading that the map ranks holes.")
    ap.add_argument("--n-donors", type=int, default=3,
                    help="donors to average over in the substitute mode")
    ap.add_argument("--keep-only", action="store_true",
                    help="invert the edit: keep the k cells and take away "
                         "the rest. Deletion asks whether the cells are "
                         "necessary; this asks whether they are sufficient.")
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
                 rank_from=Path(args.rank_from) if args.rank_from else None,
                 tag=args.tag, mode=args.mode, n_donors=args.n_donors,
                 keep_only=args.keep_only)
    log("deletion curve complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
