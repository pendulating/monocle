#!/usr/bin/env python3
"""
Create stratified train/test splits for the BayFlood dataset.

The script copies the selected images into separate output directories and writes
metadata CSV files for each split. Splitting is performed on the `gt` column in
`md.csv` to preserve the flooded/non-flooded ratio.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/bayflood"),
        help="Root directory containing BayFlood images and metadata.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to the metadata CSV. Defaults to <dataset-root>/md.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bayflood_splits"),
        help="Directory where the train/test splits will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples to place in the training split (rest go to test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser.parse_args()


def validate_paths(dataset_root: Path, metadata_path: Path, output_dir: Path, overwrite: bool) -> None:
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory {output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_image_index(dataset_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for img_path in dataset_root.rglob("*.jpg"):
        basename = img_path.name
        if basename in index:
            raise ValueError(f"Duplicate image basename detected: {basename}")
        index[basename] = img_path
    if not index:
        raise RuntimeError(f"No JPG images found under {dataset_root}")
    return index


def stratified_split(df: pd.DataFrame, label_col: str, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = pd.Series(range(len(df)), index=df.index)
    train_indices: Iterable[int] = []
    test_indices: Iterable[int] = []
    for label, group in df.groupby(label_col):
        group_indices = group.sample(frac=1.0, random_state=seed).index.tolist()
        split_point = int(len(group_indices) * train_ratio)
        split_point = max(1, min(split_point, len(group_indices) - 1)) if len(group_indices) > 1 else len(group_indices)
        train_indices = list(train_indices) + group_indices[:split_point]
        test_indices = list(test_indices) + group_indices[split_point:]
    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def copy_split(images: pd.DataFrame, image_index: Dict[str, Path], destination: Path) -> None:
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_csv_path = destination / "metadata.csv"
    meta_parquet_path = destination / "metadata.parquet"

    missing: list[str] = []
    copied = 0
    enriched = images.copy()

    # Normalize and enrich metadata for downstream VQA/GEPA consumers
    prompt_text = "Does this image show more than a foot of standing water?"
    enriched["prompt"] = prompt_text
    enriched["expected_answer"] = enriched["gt"].apply(lambda x: "Yes" if int(x) == 1 else "No")
    enriched["sample_id"] = enriched["image"].apply(lambda name: Path(name).stem)

    image_paths: list[str] = []
    for _, row in enriched.iterrows():
        basename = row["image"]
        src = image_index.get(basename)
        if src is None:
            missing.append(basename)
            image_paths.append(str(images_dir / basename))
            continue
        dst = images_dir / basename
        shutil.copy2(src, dst)
        copied += 1
        image_paths.append(str(dst.resolve()))

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images referenced in metadata: {missing[:5]}...")

    enriched["image_path"] = image_paths

    # Ensure we don't carry accidental index columns forward
    if "Unnamed: 0" in enriched.columns:
        enriched = enriched.drop(columns=["Unnamed: 0"])

    enriched.to_csv(meta_csv_path, index=False)
    enriched.to_parquet(meta_parquet_path, index=False)

    print(f"  Copied {copied} images -> {images_dir}")
    print(f"  Wrote metadata -> {meta_csv_path}")
    print(f"  Wrote metadata -> {meta_parquet_path}")


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root.resolve()
    metadata_path: Path = (args.metadata or dataset_root / "md.csv").resolve()
    output_dir: Path = args.output_dir.resolve()

    validate_paths(dataset_root, metadata_path, output_dir, args.overwrite)

    print("Building image index...")
    image_index = build_image_index(dataset_root)

    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    required_columns = {"image", "gt"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Metadata CSV missing required columns: {missing_cols}")

    print(f"Creating stratified split with train_ratio={args.train_ratio:.2f}, seed={args.seed}")
    train_df, test_df = stratified_split(df, label_col="gt", train_ratio=args.train_ratio, seed=args.seed)

    splits = {
        "train": train_df,
        "test": test_df,
    }

    for split_name, split_df in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {split_name} split ({len(split_df)} samples)")
        copy_split(split_df, image_index, split_dir)

    summary = {
        "train": len(train_df),
        "test": len(test_df),
    }
    summary_path = output_dir / "split_summary.json"
    pd.DataFrame([summary]).to_json(summary_path, orient="records", indent=2)
    print(f"Wrote summary -> {summary_path}")
    print("Split creation completed successfully.")


if __name__ == "__main__":
    main()

