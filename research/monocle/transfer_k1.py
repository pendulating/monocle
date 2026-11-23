"""Does 1 square carry the judgment into another photograph?

Why a transfer at k=1
---------------------
The deletion test asks what a photograph LOSES when a square goes. It reads
a hole. The transfer test asks what a DIFFERENT photograph gains when the
square arrives. It reads content, because the pixels move and nothing is
removed.

Every earlier transfer ran at k=4 or more. At those sizes the set-deletion
problem returns: the model saturates, and 2 sets of 4 cells are not
comparable. At k=1 the edit is 1 image token, and the arms differ in nothing
but WHICH token moves.

The arms
--------
All 3 arms take cells from the SAME source photograph and the SAME map.

| Arm | The cell that moves | Prediction for `c` of the target |
|---|---|---|
| `pos` | the map's highest cell | rises |
| `neg` | the map's lowest cell | falls |
| `random` | arbitrary cells | near 0 |

The statistic is `separation = dc_pos - dc_neg`, paired inside one (target,
source) pair. Both cells come from 1 map of 1 photograph, so the paste area
matches and the paste distance matches in expectation. Only the choice of
cell differs.

The map source
--------------
`--pick causal` names the cells by the MEASURED map. That arm asks whether a
single square carries content at all. `--pick gxa` names them by the
instrument. That arm asks whether the instrument finds such a square with no
measurement. The 2 arms answer different questions, and the second is the
one that makes a lens useful.

The gate
--------
A substitution of the target's own pixels must return `c` exactly. That row
rides in every batch as the zero, so the gate is never skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/transfer_k1"
MAPS_DEFAULT = "outputs/_monocle/gateB60_g1L/{case}/maps.parquet"
INSTRUMENTS = ("gxa", "cheap", "meang_own", "meangp_own", "lens_own")


def log(m: str) -> None:
    print(f"[transfer-k1] {m}", flush=True)


def load_maps(path: Path, pick: str,
              layer: Optional[int]) -> dict[str, np.ndarray]:
    """One 256-cell map for each photograph.

    `causal` is stored once for each layer row, and the copies agree, so the
    truth needs no layer. An instrument needs one: either a named layer or
    the mean RANK over every layer present.
    """
    df = pd.read_parquet(path)
    idc = "target" if "target" in df.columns else "image_id"
    # `causal_maps.py` names the measured map `values`; the gate store names
    # the same thing `causal`. Take either, so every map store reads here.
    # A measured map is 1 row for each image, so it needs no layer choice.
    # Keep that fact in `whole_map`: the alias renames `pick`, and a test on
    # the NEW name would send a measured map down the layer path. That path
    # returns RANKS, not the measured values, and the caller cannot see it.
    whole_map = pick == "causal"
    if whole_map and "causal" not in df.columns and "values" in df.columns:
        pick = "values"
    if pick not in df.columns:
        raise SystemExit(f"{path} holds no column {pick!r}. "
                         f"It holds {sorted(df.columns)}.")
    if whole_map or "layer" not in df.columns:
        return {str(k): np.asarray(g.iloc[0][pick], dtype=float).ravel()
                for k, g in df.groupby(idc)}
    if layer is not None:
        sub = df[df.layer == layer]
        if sub.empty:
            raise SystemExit(f"{path} holds no layer {layer}. "
                             f"It holds {sorted(df.layer.unique())}.")
        return {str(r[idc]): np.asarray(r[pick], dtype=float).ravel()
                for _, r in sub.iterrows()}
    from scipy.stats import rankdata
    out: dict[str, np.ndarray] = {}
    for k, g in df.groupby(idc):
        ranks = []
        for _, r in g.iterrows():
            v = r.get(pick)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            if v.size and np.isfinite(v).all():
                ranks.append(rankdata(v))
        if ranks:
            out[str(k)] = np.mean(ranks, axis=0)
    return out


def measure_zero(
    model: Any, lens_model: Any, inputs: dict, pos_id: int, neg_id: int,
    image_token_id: int, batch: int,
) -> float:
    """`c` under a batch of no-op substitutions, at the working batch size.

    Batch size alone moves `c` by up to 0.67 logits, so the zero is measured
    at the size the arms use.
    """
    own = inputs["pixel_values"][0]
    rows = pab.substitution_batch(inputs, own, [[]] * batch, image_token_id)
    return float(pab.contrast(model, lens_model, rows, pos_id, neg_id)[0])


def measure_arms(
    model: Any, lens_model: Any, inputs: dict, donor_px: torch.Tensor,
    cells: list[int], pos_id: int, neg_id: int, image_token_id: int,
    batch: int, zero: float,
) -> list[float]:
    """The change in `c` when each cell alone holds the donor's pixels."""
    out: list[float] = []
    for start in range(0, len(cells), batch):
        chunk = [[int(c)] for c in cells[start:start + batch]]
        n = len(chunk)
        while len(chunk) < batch:
            chunk.append(chunk[-1])
        rows = pab.substitution_batch(inputs, donor_px, chunk, image_token_id)
        cs = pab.contrast(model, lens_model, rows, pos_id, neg_id)
        out.extend(float(x - zero) for x in cs[:n])
    return out


def pixel_gap(donor_px: torch.Tensor, target_px: torch.Tensor,
              cell: int) -> float:
    """How far the pasted square is from the square it replaces.

    A cell can move `c` because it is UNLIKE its neighbourhood, not because
    of what it holds. The gap goes in the frame so the analysis can hold it
    fixed.
    """
    d = donor_px[int(cell)].float() - target_px[int(cell)].float()
    return float(torch.linalg.vector_norm(d).cpu())


def gate_self(
    model: Any, lens_model: Any, inputs: dict, cell: int, pos_id: int,
    neg_id: int, image_token_id: int, batch: int, zero: float,
) -> None:
    """A paste of the target's OWN square must change nothing."""
    own = inputs["pixel_values"][0]
    dc = measure_arms(model, lens_model, inputs, own, [cell], pos_id, neg_id,
                      image_token_id, batch, zero)[0]
    if abs(dc) > 1e-3:
        raise SystemExit(
            f"the self paste moved c by {dc:+.4f}. The plumbing writes the "
            "wrong square, and no transfer number can be trusted.")
    log(f"  self-paste gate: dc={dc:+.5f} (pass)")


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, n_sources: int, n_random: int, seed: int, batch: int,
    maps: dict[str, np.ndarray], pick: str, layer: Optional[int],
    out_dir: Path, device: str,
) -> pd.DataFrame:
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rng = np.random.default_rng(seed)

    store: list[dict] = []
    missing = 0
    for r in sg.load_single_rows(n_images, case, seed=seed):
        key = str(r["image_id"])
        if key not in maps:
            missing += 1
            continue
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        slots = pab.slots_of(inputs, image_token_id)
        values = maps[key]
        if values.size != len(slots):
            raise SystemExit(
                f"{key} holds {values.size} cells and the prompt holds "
                f"{len(slots)} patches. The stored map used another group.")
        c0 = float(pab.contrast(model, lens_model, inputs, pos_id, neg_id)[0])
        store.append({**r, "inputs": inputs, "slots": slots, "c": c0,
                      "values": values})
        log(f"  loaded {len(store)}/{n_images} {key} c={c0:+.2f}")
    if not store:
        raise SystemExit(
            "no photograph matched the stored maps. Match --seed to the run "
            "that measured them.")
    if missing:
        log(f"  {missing} photographs are absent from the stored maps")

    rows: list[dict] = []
    for i, tgt in enumerate(store):
        zero = measure_zero(model, lens_model, tgt["inputs"], pos_id, neg_id,
                            image_token_id, batch)
        if i == 0:
            gate_self(model, lens_model, tgt["inputs"], 0, pos_id, neg_id,
                      image_token_id, batch, zero)
        pool = [j for j in range(len(store)) if j != i]
        picks = rng.choice(pool, size=min(n_sources, len(pool)), replace=False)
        target_px = tgt["inputs"]["pixel_values"][0]
        for j in picks:
            src = store[int(j)]
            v = src["values"]
            order = np.argsort(v)
            pos_cell, neg_cell = int(order[-1]), int(order[0])
            rest = [c for c in range(v.size) if c not in (pos_cell, neg_cell)]
            rnd = rng.choice(rest, size=min(n_random, len(rest)),
                             replace=False)
            cells = [pos_cell, neg_cell] + [int(c) for c in rnd]
            arms = ["pos", "neg"] + ["random"] * len(rnd)
            donor_px = src["inputs"]["pixel_values"][0]
            dcs = measure_arms(model, lens_model, tgt["inputs"], donor_px,
                               cells, pos_id, neg_id, image_token_id, batch,
                               zero)
            for arm, cell, dc in zip(arms, cells, dcs):
                rows.append({
                    "case": case, "pick": pick,
                    "layer": -1 if layer is None else int(layer),
                    "target_id": str(tgt["image_id"]),
                    "source_id": str(src["image_id"]),
                    "arm": arm, "cell": int(cell), "dc": float(dc),
                    "c_target": tgt["c"], "c_source": src["c"],
                    "source_value": float(v[cell]),
                    "pixel_gap": pixel_gap(donor_px, target_px, cell)})
        log(f"  pasted into {i + 1}/{len(store)} {tgt['image_id']}")

    df = pd.DataFrame(rows)
    tag = pick if layer is None else f"{pick}_L{layer}"
    case_dir = out_dir / f"{case}_{tag}"
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "arms.parquet", index=False)
    s = summarize(df)
    s.to_parquet(case_dir / "summary.parquet", index=False)
    report(s, case, tag, case_dir)
    return s


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row for each (target, source) pair."""
    out = []
    for (t, s), sub in df.groupby(["target_id", "source_id"]):
        rnd = sub[sub.arm == "random"]
        p = sub[sub.arm == "pos"]
        n = sub[sub.arm == "neg"]
        if p.empty or n.empty:
            continue
        sd = float(rnd.dc.std(ddof=1)) if len(rnd) > 1 else np.nan
        dp, dn = float(p.dc.iloc[0]), float(n.dc.iloc[0])
        out.append({
            "target_id": t, "source_id": s,
            "dc_pos": dp, "dc_neg": dn,
            "dc_random": float(rnd.dc.mean()) if len(rnd) else np.nan,
            "random_sd": sd,
            "separation": dp - dn,
            "separation_z": (dp - dn) / sd if sd and sd > 0 else np.nan,
            "gap_pos": float(p.pixel_gap.iloc[0]),
            "gap_neg": float(n.pixel_gap.iloc[0]),
            "gap_random": float(rnd.pixel_gap.mean()) if len(rnd) else np.nan,
            "c_target": float(sub.c_target.iloc[0]),
            "c_source": float(sub.c_source.iloc[0])})
    return pd.DataFrame(out)


def report(s: pd.DataFrame, case: str, tag: str, case_dir: Path) -> None:
    from scipy.stats import wilcoxon
    sep = s.separation.to_numpy(dtype=float)
    sep = sep[np.isfinite(sep)]
    stat = {"case": case, "pick": tag, "n_pairs": int(len(sep)),
            "mean_separation": float(sep.mean()) if len(sep) else float("nan"),
            "median_separation": (float(np.median(sep)) if len(sep)
                                  else float("nan")),
            "share_positive": (float((sep > 0).mean()) if len(sep)
                               else float("nan")),
            "mean_dc_pos": float(s.dc_pos.mean()),
            "mean_dc_neg": float(s.dc_neg.mean()),
            "mean_dc_random": float(s.dc_random.mean()),
            "mean_gap_pos": float(s.gap_pos.mean()),
            "mean_gap_neg": float(s.gap_neg.mean()),
            "mean_gap_random": float(s.gap_random.mean())}
    if len(sep) > 5:
        stat["wilcoxon_p"] = float(wilcoxon(sep).pvalue)
    (case_dir / "verdict.json").write_text(json.dumps(stat, indent=2))
    log(f"{case} [{tag}]: {stat['n_pairs']} pairs")
    log(f"  dc(pos in)={stat['mean_dc_pos']:+.3f} "
        f"dc(random in)={stat['mean_dc_random']:+.3f} "
        f"dc(neg in)={stat['mean_dc_neg']:+.3f}")
    log(f"  separation mean={stat['mean_separation']:+.3f} "
        f"median={stat['median_separation']:+.3f} "
        f"positive={stat['share_positive']:.2f} "
        f"p={stat.get('wilcoxon_p', float('nan')):.4g}")
    log(f"  pixel gap pos={stat['mean_gap_pos']:.2f} "
        f"neg={stat['mean_gap_neg']:.2f} "
        f"random={stat['mean_gap_random']:.2f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Move 1 square between images.")
    ap.add_argument("--cases", nargs="+", default=["restaurants"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--n-sources", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--maps", default=MAPS_DEFAULT,
                    help="parquet path; {case} is filled in")
    ap.add_argument("--pick", default="causal",
                    choices=("causal",) + INSTRUMENTS)
    ap.add_argument("--layer", type=int, default=None,
                    help="instrument layer; leave out to pool the ranks")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--batch", type=int, default=12)
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
        maps = load_maps(path, args.pick, args.layer)
        log(f"{case}: {len(maps)} stored maps from {path}")
        run_case(case, proc, model, lens_model, tmpl,
                 n_images=args.n_images, n_sources=args.n_sources,
                 n_random=args.n_random, seed=args.seed, batch=args.batch,
                 maps=maps, pick=args.pick, layer=args.layer,
                 out_dir=out_dir, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
