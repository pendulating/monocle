#!/usr/bin/env python3
"""Create a parquet dataset from Nexar dashcam images for VQA.

This script scans a directory of images and creates a parquet file with:
- prompt: A default prompt (can be customized per image)
- sample_id: Generated from image filename
- image_path: Relative path to the image file (or absolute if --absolute_paths)

Usage:
    python scripts/create_nexar_dataset.py \
        --image_dir /share/ju/nexar_data/2023/2023-08-20/604222321527357439/frames \
        --output_path /share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet \
        --prompt "What urban planning features are visible in this dashcam image?"
    
    Or generate directly from config:
    python scripts/create_nexar_dataset.py \
        --config dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml \
        --output_path /share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet
    
    Note: Use a path accessible from GPU nodes (not /tmp)
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml


def create_nexar_dataset(
    image_dir: str,
    output_path: str,
    prompt: Optional[str] = None,
    max_images: Optional[int] = None,
    sample_id_prefix: str = "nexar",
    absolute_paths: bool = True
) -> pd.DataFrame:
    """Create a parquet dataset from images in a directory.
    
    Args:
        image_dir: Directory containing image files
        output_path: Path to output parquet file
        prompt: Default prompt to use for all images (can be customized)
        max_images: Maximum number of images to include (None for all)
        sample_id_prefix: Prefix for sample IDs
        absolute_paths: If True, use absolute paths; if False, use relative paths
        
    Returns:
        DataFrame with columns: prompt, sample_id, image_path
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    # Default prompt for dashcam images
    default_prompt = prompt or "What urban planning features are visible in this dashcam image?"
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    # Sort by filename for reproducibility
    image_files = sorted(image_files)
    
    # Limit if max_images specified
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} image files")
    
    # Create DataFrame
    data = []
    for idx, img_path in enumerate(image_files):
        # Generate sample_id from filename
        sample_id = f"{sample_id_prefix}_{img_path.stem}"
        
        # Use absolute or relative path
        if absolute_paths:
            img_path_str = str(img_path.resolve())
        else:
            img_path_str = str(img_path)
        
        data.append({
            "prompt": default_prompt,
            "sample_id": sample_id,
            "image_path": img_path_str,
        })
    
    df = pd.DataFrame(data)
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved dataset to {output_path}")
    print(f"Dataset contains {len(df)} rows")
    
    return df


def create_from_config(config_path: str, output_path: str, **overrides) -> pd.DataFrame:
    """Create dataset from Hydra config file.
    
    Args:
        config_path: Path to Hydra config YAML file
        output_path: Path to output parquet file
        **overrides: Override config values
        
    Returns:
        DataFrame with columns: prompt, sample_id, image_path
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get image directory
    image_dir = overrides.get('image_dir') or config.get('image_path')
    if not image_dir:
        raise ValueError("image_path must be specified in config or via --image_dir")
    
    # Get prompt
    prompt = overrides.get('prompt') or config.get('default_prompt')
    
    # Get other options
    max_images = overrides.get('max_images')
    sample_id_prefix = overrides.get('sample_id_prefix', 'nexar')
    absolute_paths = overrides.get('absolute_paths', True)
    
    return create_nexar_dataset(
        image_dir=image_dir,
        output_path=output_path,
        prompt=prompt,
        max_images=max_images,
        sample_id_prefix=sample_id_prefix,
        absolute_paths=absolute_paths
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a parquet dataset from Nexar dashcam images"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing image files (or use --config)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Hydra config file (alternative to --image_dir)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Default prompt to use for all images (overrides config)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to include (default: all)"
    )
    parser.add_argument(
        "--sample_id_prefix",
        type=str,
        default="nexar",
        help="Prefix for sample IDs (default: nexar)"
    )
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        default=True,
        help="Use absolute paths for images (default: True)"
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Use relative paths for images"
    )
    
    args = parser.parse_args()
    
    # Handle relative paths flag
    absolute_paths = not args.relative_paths if args.relative_paths else args.absolute_paths
    
    if args.config:
        df = create_from_config(
            config_path=args.config,
            output_path=args.output_path,
            image_dir=args.image_dir,
            prompt=args.prompt,
            max_images=args.max_images,
            sample_id_prefix=args.sample_id_prefix,
            absolute_paths=absolute_paths
        )
    elif args.image_dir:
        df = create_nexar_dataset(
            image_dir=args.image_dir,
            output_path=args.output_path,
            prompt=args.prompt,
            max_images=args.max_images,
            sample_id_prefix=args.sample_id_prefix,
            absolute_paths=absolute_paths
        )
    else:
        parser.error("Either --image_dir or --config must be specified")
    
    print("\nDataset preview:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
