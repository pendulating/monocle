"""Sample a diverse, reproducible set of Cyclomedia street-view face images for
the multimodal Jacobian-lens fit (monocle.jlens_fit_mm).

The multimodal fit needs image-patch residuals spread across the whole city,
not clustered in whichever borough dominates the pull. So we stratify across the
recording index's ``dataset`` column and draw an *equal* number of recordings
per dataset (not proportional — the smallest borough contributes as many as the
largest), random within each stratum under a fixed seed. One face per recording
is chosen at random from the four cardinal faces F/B/L/R (never U/D — sky and
ground carry no street content).

We never walk the NFS raw tree (directory listing on that mount is glacial). The
recording index (``recordings_v1.parquet``, ~5.24M rows) is sampled with DuckDB's
repeatable reservoir sampler; each face path is then *constructed* from the
deterministic layout

    {RAW_ROOT}/{dataset}/{recording_id[:5]}/{recording_id}/faces/{F}.jpg

and the ONLY filesystem contact is one ``os.path.isfile`` per candidate. We
oversample 2x, keep the candidates whose file exists (in a deterministic
round-robin order that preserves the per-dataset balance), and truncate to the
first --n.

Outputs (the sibling monocle.jlens_fit_mm codes against these):
    fit_images.json        list of n {"path","recording_id","dataset","face"}
    fit_images.stats.json  {"n","seed","per_dataset","per_face",
                            "candidates_checked","missing","provenance"}

Usage:
    python -m monocle.sample_fit_images --n 256 --seed 777
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO))

from dagspaces.common.cyclomedia.catalog import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    RAW_ROOT,
)

DEFAULT_OUT_DIR = str(REPO / "outputs/_monocle/jlens/mm")
FACES = ["F", "B", "L", "R"]  # cardinal only; never U (sky) / D (ground)
OVERSAMPLE = 2  # draw 2x candidates so existence-check attrition still fills n


def log(msg: str) -> None:
    print(f"[sample-fit] {msg}", flush=True)


def face_path(dataset: str, recording_id: str, face: str) -> str:
    """Deterministic face image path (no globbing): the group bucket is the
    recording id's 5-char prefix (verified group == recording_id[:5])."""
    return os.path.join(
        RAW_ROOT, dataset, recording_id[:5], recording_id, "faces", f"{face}.jpg"
    )


def sample_candidates(
    index_path: str, seed: int, per_dataset: int,
) -> list[dict]:
    """Draw ``per_dataset`` recordings from each dataset stratum with DuckDB's
    repeatable reservoir sampler, assign each a random cardinal face, and return
    candidates ordered round-robin across datasets (rank, dataset) so that
    truncating to the first n preserves the per-dataset balance."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    datasets = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT dataset FROM read_parquet('{index_path}') "
            "ORDER BY dataset"
        ).fetchall()
    ]
    log(f"{len(datasets)} dataset strata: {datasets}")

    rng = np.random.RandomState(seed)
    per_ds_recs: dict[str, list[str]] = {}
    for ds in datasets:
        # REPEATABLE makes the reservoir draw deterministic for a given
        # (data, seed). The sample MUST wrap the *filtered* subquery: DuckDB
        # applies USING SAMPLE right after FROM, before WHERE, so sampling in
        # the same SELECT would draw from all 5.24M rows and keep only the
        # handful that happen to fall in this dataset (the catalog's bbox trap).
        df = con.execute(
            f"""
            SELECT recording_id FROM (
                SELECT recording_id
                FROM read_parquet('{index_path}')
                WHERE dataset = ?
            ) USING SAMPLE reservoir({int(per_dataset)} ROWS)
                    REPEATABLE({int(seed)})
            """,
            [ds],
        ).fetchdf()
        recs = list(df["recording_id"])
        if len(recs) < per_dataset:
            log(f"  {ds}: only {len(recs)} recordings available (< {per_dataset})")
        per_ds_recs[ds] = recs

    # Round-robin interleave by rank so the first n existing stays balanced.
    candidates: list[dict] = []
    max_rank = max((len(v) for v in per_ds_recs.values()), default=0)
    for rank in range(max_rank):
        for ds in datasets:
            recs = per_ds_recs[ds]
            if rank >= len(recs):
                continue
            rec = recs[rank]
            face = FACES[int(rng.randint(len(FACES)))]
            candidates.append(
                {
                    "recording_id": rec,
                    "dataset": ds,
                    "face": face,
                    "path": face_path(ds, rec, face),
                }
            )
    return candidates


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="monocle.sample_fit_images",
        description="Sample cyclomedia face images for the multimodal jlens fit.",
    )
    ap.add_argument("--n", type=int, default=256,
                    help="Number of final images kept (default 256).")
    ap.add_argument("--seed", type=int, default=777,
                    help="RNG seed for stratified sampling + face choice.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Where fit_images.json / .stats.json are written.")
    ap.add_argument("--index", default=DEFAULT_INDEX_PATH,
                    help="Recording-level index parquet.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con_datasets = duckdb.connect()
    n_datasets = con_datasets.execute(
        f"SELECT count(DISTINCT dataset) FROM read_parquet('{args.index}')"
    ).fetchone()[0]
    con_datasets.close()

    target_candidates = OVERSAMPLE * args.n
    per_dataset = -(-target_candidates // n_datasets)  # ceil, equal per stratum
    log(f"n={args.n} seed={args.seed} index={args.index}")
    log(f"drawing {per_dataset} recordings x {n_datasets} datasets "
        f"= {per_dataset * n_datasets} candidates (target {target_candidates})")

    candidates = sample_candidates(args.index, args.seed, per_dataset)
    log(f"{len(candidates)} candidates constructed; checking existence...")

    kept: list[dict] = []
    checked = 0
    missing = 0
    for cand in candidates:
        checked += 1
        if os.path.isfile(cand["path"]):
            kept.append(cand)
            if len(kept) >= args.n:
                break
        else:
            missing += 1

    if len(kept) < args.n:
        log(f"WARNING: only {len(kept)} existing files found for n={args.n}; "
            "candidate pool exhausted (increase OVERSAMPLE or --n lower).")

    per_dataset_counts: dict[str, int] = {}
    per_face_counts: dict[str, int] = {f: 0 for f in FACES}
    for c in kept:
        per_dataset_counts[c["dataset"]] = per_dataset_counts.get(c["dataset"], 0) + 1
        per_face_counts[c["face"]] += 1

    fit_images = [
        {
            "path": c["path"],
            "recording_id": c["recording_id"],
            "dataset": c["dataset"],
            "face": c["face"],
        }
        for c in kept
    ]
    stats = {
        "n": len(kept),
        "seed": args.seed,
        "per_dataset": per_dataset_counts,
        "per_face": per_face_counts,
        "candidates_checked": checked,
        "missing": missing,
        "provenance": (
            f"stratified equal-per-dataset reservoir sample of {args.index} "
            f"(seed={args.seed}), one random cardinal face per recording, "
            f"path-existence filtered, {OVERSAMPLE}x oversampled"
        ),
    }

    (out_dir / "fit_images.json").write_text(json.dumps(fit_images, indent=2))
    (out_dir / "fit_images.stats.json").write_text(json.dumps(stats, indent=2))

    log(f"wrote {len(kept)} images -> {out_dir}/fit_images.json")
    log(f"per_dataset: {per_dataset_counts}")
    log(f"per_face:    {per_face_counts}")
    log(f"candidates checked={checked}  missing={missing}  "
        f"rate={missing / checked:.1%}" if checked else "no candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
