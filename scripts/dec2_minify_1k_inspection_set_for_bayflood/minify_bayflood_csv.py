#!/usr/bin/env python3
"""Script to create a minified version of bayflood CSV with image, gt, and pred columns."""

import pandas as pd
import os
from pathlib import Path


def main():
    # Read the input CSV
    input_path = Path(__file__).parent.parent / "data" / "bayflood_1k" / "md.csv"
    output_path = input_path.parent / "md_minified.csv"
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Extract basename from image path
    # Handle paths like: /data/local-files/?d=/share/ju/nexar_data/.../filename.jpg
    def get_basename(image_path):
        if pd.isna(image_path):
            return ""
        path_str = str(image_path)
        # Extract the actual file path after ?d= if present
        if "?d=" in path_str:
            # Get the path after ?d=
            path_part = path_str.split("?d=")[1]
        else:
            path_part = path_str
        return os.path.basename(path_part)
    
    # Create minified dataframe
    result = pd.DataFrame()
    result["image"] = df["image"].apply(get_basename)
    
    # Factorize gt column: 1 for "Flooded road", 0 otherwise
    result["gt"] = (df["gt"] == "Flooded road").astype(int)
    
    # Create pred: 1 if response_1 contains "yes" (case-insensitive), 0 otherwise
    result["pred"] = df["response_1"].str.contains("yes", case=False, na=False).astype(int)
    
    # Save to output
    print(f"Writing minified CSV to {output_path}...")
    result.to_csv(output_path, index=False)
    print(f"Done! Created {len(result)} rows with columns: {list(result.columns)}")
    
    # Print a sample
    print("\nFirst 5 rows:")
    print(result.head())


if __name__ == "__main__":
    main()

