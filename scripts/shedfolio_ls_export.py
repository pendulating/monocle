#!/usr/bin/env python3
"""
Export high-confidence scaffolding embedding images to Label Studio format
for image classification annotation.

Adapted from bayflood/scripts/df_2_ls.py for the Shedfolio project.

Usage:
    python scripts/shedfolio_ls_export.py \
        --rerank-dir outputs/rerank \
        --output-dir outputs/annotation/scaffolding_v1 \
        --min-rerank-score 0.15 \
        --max-samples 1000 \
        --shuffle

Output:
    <output-dir>/tasks.json          -- Label Studio import file
    <output-dir>/labeling_config.xml -- Label Studio project config (paste into UI)
    <output-dir>/source_data.parquet -- Filtered source data for reference
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("shedfolio_ls_export")

# ── Label Studio project config (image classification) ──────────────────
LABELING_CONFIG_XML = """\
<View>
  <Style>
    .lsf-main-content { max-width: 900px; margin: 0 auto; }
    .htx-text { font-size: 13px; color: #666; margin-bottom: 8px; }
  </Style>

  <View style="display:flex; flex-direction:column; align-items:center;">
    <Image name="image" value="$image" maxWidth="800" zoom="true" rotateControl="true"/>
  </View>

  <View style="margin-top:8px; padding:8px 12px; background:#f8f8f8; border-radius:6px; font-size:12px; color:#555;">
    <Text name="meta" value="Score: $rerank_score  |  Recording: $recording_id ($face)  |  Run: $source_run"/>
  </View>

  <Choices name="scaffold_type" toName="image" choice="single" showInline="true">
    <Choice value="regular_green_scaffolding" alias="Green Scaffolding"/>
    <Choice value="fancy_white_scaffolding"   alias="White / Arched Scaffolding"/>
    <Choice value="outdoor_dining"            alias="Outdoor Dining Shed"/>
    <Choice value="bridge_highway"            alias="Bridge / Highway"/>
    <Choice value="other_false_positive"      alias="Other False Positive"/>
    <Choice value="no_scaffolding_features"   alias="No Scaffolding-Like Features"/>
  </Choices>
</View>
"""

# ── Columns to carry into Label Studio task metadata ─────────────────────
METADATA_COLS = [
    "sample_id",
    "recording_id",
    "face",
    "lat",
    "lon",
    "retrieval_score",
    "rerank_score",
    "rerank_rank",
    "source_run",
    "recordedAt",
]


def load_rerank_data(rerank_dir: Path) -> pd.DataFrame:
    """Load and combine rerank parquet file(s). Accepts a directory or a single file."""
    if rerank_dir.is_file() and rerank_dir.suffix == ".parquet":
        parquets = [rerank_dir]
    else:
        parquets = sorted(rerank_dir.glob("*.parquet"))
    if not parquets:
        log.error(f"No parquet files found in {rerank_dir}")
        sys.exit(1)

    dfs = []
    for f in parquets:
        df = pd.read_parquet(f)
        df["source_run"] = f.stem
        dfs.append(df)
        log.info(
            f"  {f.name}: {len(df):,} rows, "
            f"rerank_score=[{df.rerank_score.min():.3f}, {df.rerank_score.max():.3f}]"
        )

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total rows: {len(combined):,}")
    return combined


def resolve_image_path(row: pd.Series, image_col: str) -> str | None:
    """Pick the best available image path for a row."""
    # Prefer the original (stable shared storage)
    for col in ["image_path_original", image_col, "image_path"]:
        val = row.get(col)
        if pd.notna(val) and isinstance(val, str):
            p = Path(val)
            if p.exists():
                return str(p)
    return None


def build_ls_tasks(
    df: pd.DataFrame,
    image_col: str = "image_path_original",
    local_files_prefix: str = "",
) -> list[dict]:
    """Convert a DataFrame of detections into Label Studio task JSON."""
    tasks = []
    skipped = 0

    for idx, row in df.iterrows():
        img_path = resolve_image_path(row, image_col)
        if img_path is None:
            skipped += 1
            continue

        # Label Studio local-files protocol
        if local_files_prefix:
            image_url = f"/data/local-files/?d={local_files_prefix}/{Path(img_path).name}"
        else:
            image_url = f"/data/local-files/?d={img_path}"

        # Build task data dict
        data = {"image": image_url}

        # Add metadata columns
        for col in METADATA_COLS:
            if col in row.index:
                val = row[col]
                # Serialize non-JSON-native types
                if pd.isna(val):
                    data[col] = None
                elif hasattr(val, "isoformat"):
                    data[col] = val.isoformat()
                elif isinstance(val, (float,)):
                    data[col] = round(float(val), 4)
                else:
                    data[col] = val

        tasks.append({"data": data, "id": int(idx)})

    if skipped:
        log.warning(f"Skipped {skipped} rows with missing image files")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Export rerank detections to Label Studio for scaffolding classification"
    )
    parser.add_argument(
        "--rerank-dir",
        type=Path,
        default=Path("outputs/rerank"),
        help="Directory containing rerank parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/annotation/scaffolding_v1"),
        help="Output directory for LS export files",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=None,
        help="Minimum rerank score threshold (e.g. 0.15)",
    )
    parser.add_argument(
        "--min-retrieval-score",
        type=float,
        default=None,
        help="Minimum retrieval score threshold",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to export",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples (important for unbiased annotation)",
    )
    parser.add_argument(
        "--deduplicate-recording",
        action="store_true",
        default=False,
        help="Keep only the best-scoring face per recording_id (off by default "
             "— multiple angles are useful training examples)",
    )
    parser.add_argument(
        "--local-files-prefix",
        type=str,
        default="",
        help="Prefix for Label Studio local-files path (e.g. 'cyclomedia')",
    )
    parser.add_argument(
        "--image-col",
        type=str,
        default="image_path_original",
        help="Column containing image file paths",
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=None,
        help="Use a specific parquet instead of --rerank-dir (for custom data)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    if args.source_parquet:
        log.info(f"Loading source parquet: {args.source_parquet}")
        df = pd.read_parquet(args.source_parquet)
        df["source_run"] = args.source_parquet.stem
    else:
        log.info(f"Loading rerank data from: {args.rerank_dir}")
        df = load_rerank_data(args.rerank_dir)

    # ── Deduplicate ──────────────────────────────────────────────────────
    # Always drop exact duplicate sample_ids (same image from overlapping runs)
    if "sample_id" in df.columns:
        before = len(df)
        df = (
            df.sort_values("rerank_score", ascending=False)
            .drop_duplicates(subset="sample_id", keep="first")
            .reset_index(drop=True)
        )
        if before != len(df):
            log.info(f"Dropped duplicate sample_ids: {before:,} -> {len(df):,}")

    # Optionally collapse to one face per recording (off by default)
    if args.deduplicate_recording and "recording_id" in df.columns:
        before = len(df)
        df = (
            df.sort_values("rerank_score", ascending=False)
            .drop_duplicates(subset="recording_id", keep="first")
            .reset_index(drop=True)
        )
        log.info(f"Deduplicated by recording_id: {before:,} -> {len(df):,}")

    # ── Filter ───────────────────────────────────────────────────────────
    if args.min_rerank_score is not None:
        before = len(df)
        df = df[df.rerank_score >= args.min_rerank_score]
        log.info(
            f"Filtered rerank_score >= {args.min_rerank_score}: "
            f"{before:,} -> {len(df):,}"
        )

    if args.min_retrieval_score is not None:
        before = len(df)
        df = df[df.retrieval_score >= args.min_retrieval_score]
        log.info(
            f"Filtered retrieval_score >= {args.min_retrieval_score}: "
            f"{before:,} -> {len(df):,}"
        )

    # ── Shuffle ──────────────────────────────────────────────────────────
    if args.shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        log.info("Shuffled samples")

    # ── Sample ───────────────────────────────────────────────────────────
    if args.max_samples is not None and len(df) > args.max_samples:
        df = df.head(args.max_samples)
        log.info(f"Capped at {args.max_samples} samples")

    log.info(f"Final dataset: {len(df):,} images")

    if len(df) == 0:
        log.error("No images remaining after filtering. Exiting.")
        sys.exit(1)

    # ── Score summary ────────────────────────────────────────────────────
    log.info(
        f"  rerank_score:    [{df.rerank_score.min():.3f}, {df.rerank_score.max():.3f}] "
        f"(mean={df.rerank_score.mean():.3f})"
    )
    log.info(
        f"  retrieval_score: [{df.retrieval_score.min():.3f}, {df.retrieval_score.max():.3f}] "
        f"(mean={df.retrieval_score.mean():.3f})"
    )

    # ── Build Label Studio tasks ─────────────────────────────────────────
    tasks = build_ls_tasks(
        df,
        image_col=args.image_col,
        local_files_prefix=args.local_files_prefix,
    )
    log.info(f"Built {len(tasks):,} Label Studio tasks")

    # ── Write outputs ────────────────────────────────────────────────────
    tasks_path = args.output_dir / "tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=2)
    log.info(f"Saved tasks:    {tasks_path}")

    config_path = args.output_dir / "labeling_config.xml"
    with open(config_path, "w") as f:
        f.write(LABELING_CONFIG_XML)
    log.info(f"Saved config:   {config_path}")

    source_path = args.output_dir / "source_data.parquet"
    df.to_parquet(source_path, index=False)
    log.info(f"Saved source:   {source_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"LABEL STUDIO EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Tasks:     {len(tasks):,} images")
    print(f"  Output:    {args.output_dir}")
    print(f"  Import:    tasks.json (paste into Label Studio)")
    print(f"  Config:    labeling_config.xml (paste into project settings)")
    print(f"\nLabel Studio setup:")
    print(f"  1. Create new project in Label Studio")
    print(f"  2. Settings > Labeling Interface > Code > paste labeling_config.xml")
    print(f"  3. Settings > Cloud Storage > Add Local Storage")
    print(f"     - Set local path to image root directory")
    print(f"  4. Import tasks.json via the project import button")
    print(f"  5. Start annotating!")
    print(f"\nCategories:")
    print(f"  - regular_green_scaffolding  (standard green mesh/netting)")
    print(f"  - fancy_white_scaffolding    (Urban Umbrella / arched / white)")
    print(f"  - outdoor_dining             (restaurant outdoor dining sheds)")
    print(f"  - bridge_highway             (bridge/overpass false positive)")
    print(f"  - other_false_positive       (awnings, canopies, etc.)")
    print(f"  - no_scaffolding_features    (no scaffolding-like structure)")


if __name__ == "__main__":
    main()
