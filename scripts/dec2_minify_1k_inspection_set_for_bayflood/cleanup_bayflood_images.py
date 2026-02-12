#!/usr/bin/env python3
"""Script to remove images from bayflood_1k/0 and bayflood_1k/1 that are not in the metadata CSV."""

import pandas as pd
import os
from pathlib import Path


def get_basename(image_path):
    """Extract basename from image path, handling query parameter format."""
    if pd.isna(image_path):
        return ""
    path_str = str(image_path)
    # Extract the actual file path after ?d= if present
    if "?d=" in path_str:
        path_part = path_str.split("?d=")[1]
    else:
        path_part = path_str
    return os.path.basename(path_part)


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "bayflood_1k"
    md_path = data_dir / "md.csv"
    dir_0 = data_dir / "0"
    dir_1 = data_dir / "1"
    
    # Read metadata CSV
    print(f"Reading metadata from {md_path}...")
    df = pd.read_csv(md_path)
    
    # Extract image basenames from metadata
    image_basenames = set(df["image"].apply(get_basename))
    # Remove empty strings
    image_basenames.discard("")
    
    print(f"Found {len(image_basenames)} unique images in metadata")
    
    # Get all files in directories
    files_in_0 = set(f.name for f in dir_0.iterdir() if f.is_file())
    files_in_1 = set(f.name for f in dir_1.iterdir() if f.is_file())
    
    print(f"Found {len(files_in_0)} files in 0/")
    print(f"Found {len(files_in_1)} files in 1/")
    
    # Find files to remove (in directories but not in metadata)
    to_remove_0 = files_in_0 - image_basenames
    to_remove_1 = files_in_1 - image_basenames
    
    print(f"\nFiles to remove from 0/: {len(to_remove_0)}")
    print(f"Files to remove from 1/: {len(to_remove_1)}")
    
    if len(to_remove_0) == 0 and len(to_remove_1) == 0:
        print("No files to remove!")
        return
    
    # Ask for confirmation
    if len(to_remove_0) > 0:
        print(f"\nSample files to remove from 0/: {list(to_remove_0)[:5]}")
    if len(to_remove_1) > 0:
        print(f"\nSample files to remove from 1/: {list(to_remove_1)[:5]}")
    
    # Remove files
    removed_count_0 = 0
    removed_count_1 = 0
    
    for filename in to_remove_0:
        file_path = dir_0 / filename
        try:
            file_path.unlink()
            removed_count_0 += 1
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    for filename in to_remove_1:
        file_path = dir_1 / filename
        try:
            file_path.unlink()
            removed_count_1 += 1
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    print(f"\nRemoved {removed_count_0} files from 0/")
    print(f"Removed {removed_count_1} files from 1/")
    print("Done!")


if __name__ == "__main__":
    main()

