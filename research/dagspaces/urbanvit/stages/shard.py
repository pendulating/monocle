"""Shard stage: parquet → per-split WebDataset tar shards on /scratch.

Writes `{split}-{shard_idx:05d}.tar` files under `output_dir`, plus a
`manifest.json` describing splits, shard files, and sample counts.

Each tar entry is two files keyed on `sample_id`:
  - `<sample_id>.jpg`  — JPEG, resized to model.image_size × image_size, quality per config
  - `<sample_id>.json` — dict with the row's label columns and metadata

The split decision is **group-aware on `recording_id`** and stratified by
(borough × year) — see docs/plans/urbanvit-dagspace.md. All 4 faces (L/B/R/F)
of a single recording always land in the same split.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_shard_stage(cfg: DictConfig) -> Dict[str, Any]:
    backend = str(cfg.shard.get("backend", "webdataset"))
    if backend == "ffcv":
        raise NotImplementedError(
            "FFCV shard backend is not yet implemented. Install FFCV and "
            "add an FFCV writer, or use shard=webdataset (default)."
        )
    if backend != "webdataset":
        raise ValueError(f"Unknown shard backend: {backend}")

    output_dir = _resolve_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.json")
    if bool(cfg.shard.get("skip_if_exists", True)) and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        print(f"[shard] Using existing shards at {output_dir} "
              f"({manifest.get('total_samples', '?')} samples)", flush=True)
        return {
            "shards_dir": output_dir,
            "manifest": manifest_path,
            "metrics": {"samples_total": manifest.get("total_samples", 0)},
            "metadata": {"reused_existing": True},
        }

    df = _load_dataframe(cfg)
    print(f"[shard] Loaded parquet: {len(df)} rows", flush=True)

    splits = _compute_group_splits(df, cfg)
    for split_name, split_df in splits.items():
        print(f"[shard] split={split_name}: {len(split_df)} samples, "
              f"{split_df['recording_id'].nunique()} recordings", flush=True)

    label_columns = _resolve_label_columns(cfg)
    label_columns_present = [lc for lc in label_columns if lc["column"] in df.columns]
    if label_columns_present:
        print(f"[shard] label columns present in parquet: "
              f"{[lc['column'] for lc in label_columns_present]}", flush=True)
    else:
        print(f"[shard] WARN: none of heads[*].column are in the parquet. "
              f"Inference will still work; training will abort until labels "
              f"are populated.", flush=True)
    metadata_columns = _resolve_metadata_columns(cfg, label_columns)

    image_size = int(cfg.shard.get("resize") or cfg.model.image_size)
    jpeg_quality = int(cfg.shard.get("jpeg_quality", 90))
    samples_per_shard = int(cfg.shard.samples_per_shard)
    image_path_col = str(cfg.data.columns.image_path)
    id_col = str(cfg.data.columns.id)
    num_workers = int(cfg.shard.get("num_workers", 0) or max(1, (os.cpu_count() or 4) // 2))

    t0 = time.time()
    shard_records: List[Dict[str, Any]] = []
    total_samples = 0
    total_written = 0

    for split_name, split_df in splits.items():
        split_shards = _plan_shards(split_df, split_name, samples_per_shard)
        if not split_shards:
            continue

        tasks = []
        for shard_idx, (rows_slice,) in enumerate(split_shards):
            tar_path = os.path.join(
                output_dir, f"{split_name}-{shard_idx:05d}.tar"
            )
            tasks.append({
                "tar_path": tar_path,
                "rows": rows_slice.to_dict("records"),
                "image_path_col": image_path_col,
                "id_col": id_col,
                "label_columns": label_columns,
                "metadata_columns": metadata_columns,
                "image_size": image_size,
                "jpeg_quality": jpeg_quality,
                "split": split_name,
            })

        # Parallel tar write
        if num_workers <= 1:
            results = [_write_one_tar(task) for task in tasks]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_write_one_tar, task) for task in tasks]
                for fut in as_completed(futures):
                    results.append(fut.result())

        # Sort by shard index extracted from tar filename for deterministic manifest
        results.sort(key=lambda r: r["tar_path"])
        for r in results:
            shard_records.append({
                "split": r["split"],
                "tar_path": r["tar_path"],
                "samples": r["samples_written"],
                "skipped": r["samples_skipped"],
            })
            total_written += r["samples_written"]
        total_samples += len(split_df)

    duration = time.time() - t0

    manifest = {
        "output_dir": output_dir,
        "image_size": image_size,
        "jpeg_quality": jpeg_quality,
        "label_columns": [c["column"] for c in label_columns_present],
        "splits": {name: int(len(df_)) for name, df_ in splits.items()},
        "shard_files": shard_records,
        "total_samples": int(total_samples),
        "total_written": int(total_written),
        "duration_s": duration,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[shard] Wrote {total_written}/{total_samples} samples across "
          f"{len(shard_records)} tar shards in {duration:.1f}s → {output_dir}",
          flush=True)

    return {
        "shards_dir": output_dir,
        "manifest": manifest_path,
        "metrics": {
            "samples_total": int(total_samples),
            "samples_written": int(total_written),
            "shards_total": len(shard_records),
            "duration_s": duration,
        },
        "metadata": {"reused_existing": False},
    }


# ---------------------------------------------------------------------------
# Dataframe loading + split computation
# ---------------------------------------------------------------------------

def _load_dataframe(cfg: DictConfig) -> pd.DataFrame:
    parquet_path = str(cfg.data.parquet_path)
    filters = cfg.data.get("pandas_filters", None)
    if filters is not None:
        filters = OmegaConf.to_container(filters, resolve=True)
    df = pd.read_parquet(parquet_path, filters=filters)

    limit = cfg.data.get("limit", None)
    if limit is not None:
        df = df.head(int(limit))

    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if sample_n:
        df = df.head(int(sample_n))

    # Required columns
    for col_key in ("id", "image_path", "recording_id"):
        col = str(cfg.data.columns[col_key])
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' (data.columns.{col_key}) "
                             f"not found in {parquet_path}")

    # Rename to canonical names for downstream simplicity
    rename = {
        str(cfg.data.columns.id): "sample_id",
        str(cfg.data.columns.image_path): "image_path",
        str(cfg.data.columns.recording_id): "recording_id",
    }
    # Only rename if source != target
    rename = {k: v for k, v in rename.items() if k != v}
    if rename:
        df = df.rename(columns=rename)

    return df


def _compute_group_splits(
    df: pd.DataFrame, cfg: DictConfig
) -> Dict[str, pd.DataFrame]:
    """Assign each recording_id to train/val/test with stratification.

    Algorithm: within each stratum (borough × year), shuffle recordings
    deterministically by (seed, recording_id) hash, then slice by ratio.
    This keeps splits balanced across strata and reproducible across runs.
    """
    split_cfg = cfg.shard.split
    group_col = str(split_cfg.group_column)
    stratify_cols = [str(c) for c in split_cfg.get("stratify_columns", [])]
    ratios = dict(split_cfg.ratios)
    seed = int(split_cfg.seed)

    total = float(ratios.get("train", 0.8) + ratios.get("val", 0.1) + ratios.get("test", 0.1))
    r_train = float(ratios["train"]) / total
    r_val = float(ratios["val"]) / total

    if group_col not in df.columns:
        raise ValueError(f"split.group_column '{group_col}' not in dataframe")

    # Deterministic per-recording ranking within each stratum.
    # Use stable hash of (seed, recording_id) → uniform rank in [0,1).
    def _rank(rid: str) -> float:
        h = hashlib.blake2b(f"{seed}:{rid}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") / (1 << 64)

    # If stratify columns absent, use a single stratum.
    present_strats = [c for c in stratify_cols if c in df.columns]
    if present_strats:
        stratum = df[present_strats].astype(str).agg("|".join, axis=1)
    else:
        stratum = pd.Series(["_all"] * len(df), index=df.index)

    # One row per (recording_id, stratum) — pick first stratum assignment
    rec_to_stratum = (
        pd.DataFrame({"recording_id": df[group_col].values,
                      "stratum": stratum.values})
        .drop_duplicates("recording_id")
        .set_index("recording_id")
    )

    # Per-stratum slicing
    split_assignment: Dict[str, str] = {}
    for strat_val, strat_group in rec_to_stratum.groupby("stratum"):
        ids = strat_group.index.tolist()
        ids_sorted = sorted(ids, key=_rank)
        n = len(ids_sorted)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        for i, rid in enumerate(ids_sorted):
            if i < n_train:
                split_assignment[rid] = "train"
            elif i < n_train + n_val:
                split_assignment[rid] = "val"
            else:
                split_assignment[rid] = "test"

    df = df.copy()
    df["_split"] = df[group_col].map(split_assignment)
    return {
        "train": df[df["_split"] == "train"].drop(columns=["_split"]),
        "val":   df[df["_split"] == "val"].drop(columns=["_split"]),
        "test":  df[df["_split"] == "test"].drop(columns=["_split"]),
    }


def _resolve_label_columns(cfg: DictConfig) -> List[Dict[str, Any]]:
    heads = cfg.get("heads", {}).get("heads", [])
    out = []
    for h in heads:
        out.append({
            "name": str(h.name),
            "column": str(h.column),
            "task": str(h.get("task", "binary")),
            "num_classes": int(h.get("num_classes", 2)),
            "conditional_on": h.get("conditional_on", None),
        })
    return out


def _resolve_metadata_columns(
    cfg: DictConfig, label_columns: List[Dict[str, Any]]
) -> List[str]:
    """Columns to preserve into the tar's json sidecar (beyond the label cols)."""
    keep = {"sample_id", "recording_id", "image_path"}
    # Always keep stratification inputs for debugging
    keep.add(str(cfg.data.columns.get("borough", "borough")))
    keep.add(str(cfg.data.columns.get("year", "year")))
    for lc in label_columns:
        keep.add(lc["column"])
    return sorted(keep)


# ---------------------------------------------------------------------------
# Tar writing
# ---------------------------------------------------------------------------

def _plan_shards(
    df: pd.DataFrame, split_name: str, samples_per_shard: int
) -> List[Tuple[pd.DataFrame]]:
    if len(df) == 0:
        return []
    # Shuffle within a split deterministically (seed + split)
    rng = random.Random(f"{split_name}-shuffle")
    idx = list(range(len(df)))
    rng.shuffle(idx)
    df = df.iloc[idx].reset_index(drop=True)
    shards = []
    for start in range(0, len(df), samples_per_shard):
        shards.append((df.iloc[start:start + samples_per_shard],))
    return shards


def _write_one_tar(task: Dict[str, Any]) -> Dict[str, Any]:
    """Write a single tar shard from a list of rows. Runs in a worker process."""
    tar_path: str = task["tar_path"]
    rows: List[Dict[str, Any]] = task["rows"]
    image_size: int = task["image_size"]
    jpeg_quality: int = task["jpeg_quality"]
    id_col: str = task["id_col"]  # noqa: F841  (kept for API symmetry)
    image_path_col: str = task["image_path_col"]  # noqa: F841
    label_columns: List[Dict[str, Any]] = task["label_columns"]
    metadata_columns: List[str] = task["metadata_columns"]
    split_name: str = task["split"]

    samples_written = 0
    samples_skipped = 0
    tmp_path = tar_path + ".tmp"

    with tarfile.open(tmp_path, "w") as tar:
        for row in rows:
            sid = str(row.get("sample_id"))
            img_path = str(row.get("image_path"))
            try:
                img_bytes = _load_and_resize_jpeg(img_path, image_size, jpeg_quality)
            except Exception as e:
                samples_skipped += 1
                if samples_skipped < 5:
                    print(f"[shard:{split_name}] skip {sid}: {e}", flush=True)
                continue

            json_payload = {
                k: _json_safe(row.get(k)) for k in metadata_columns if k in row
            }
            for lc in label_columns:
                if lc["column"] in row:
                    json_payload[lc["column"]] = _json_safe(row[lc["column"]])
            json_bytes = json.dumps(json_payload, default=str).encode("utf-8")

            _add_bytes(tar, f"{sid}.jpg", img_bytes)
            _add_bytes(tar, f"{sid}.json", json_bytes)
            samples_written += 1

    os.replace(tmp_path, tar_path)
    return {
        "tar_path": tar_path,
        "split": split_name,
        "samples_written": samples_written,
        "samples_skipped": samples_skipped,
    }


def _load_and_resize_jpeg(path: str, size: int, quality: int) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        # Resize so the shorter side == `size`, then center-crop to square.
        w, h = im.size
        scale = size / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        im = im.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "item"):  # numpy scalar
        try:
            return v.item()
        except Exception:
            return str(v)
    return str(v)


def _resolve_output_dir(cfg: DictConfig) -> str:
    """Resolve the shard output directory.

    Precedence: `shard.output_dir` (set by orchestrator pipeline outputs) >
    `shard.scratch_root`/<experiment-name>.
    """
    explicit = cfg.shard.get("output_dir", None)
    if explicit:
        return os.path.abspath(str(explicit))
    base = str(cfg.shard.scratch_root)
    experiment_name = str(cfg.experiment.name)
    return os.path.abspath(os.path.join(base, experiment_name))
