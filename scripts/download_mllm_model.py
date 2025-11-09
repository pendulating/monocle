#!/usr/bin/env python3
"""Download a multimodal LLM model to the zoo directory.

Usage:
    python scripts/download_mllm_model.py \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --zoo_path /share/pierson/matt/zoo/models
    
    Or use a smaller model:
    python scripts/download_mllm_model.py \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --zoo_path /share/pierson/matt/zoo/models
"""

import argparse
import os
from pathlib import Path


def download_model(model_name: str, zoo_path: str, use_symlinks: bool = False):
    """Download a model from HuggingFace Hub to the zoo directory.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        zoo_path: Path to zoo models directory
        use_symlinks: If True, use symlinks instead of copying (saves disk space)
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    zoo_path = Path(zoo_path)
    zoo_path.mkdir(parents=True, exist_ok=True)
    
    # Extract model directory name from model identifier
    model_dir_name = model_name.split("/")[-1]
    target_path = zoo_path / model_dir_name
    
    if target_path.exists():
        print(f"Model directory already exists: {target_path}")
        print("Skipping download. Use --force to re-download.")
        return str(target_path)
    
    print(f"Downloading {model_name} to {target_path}...")
    print("This may take a while depending on model size and network speed.")
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target_path),
            local_dir_use_symlinks=use_symlinks,
            token=hf_token,
            resume_download=True,
        )
        print(f"✓ Successfully downloaded {model_name} to {target_path}")
        return str(target_path)
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download a multimodal LLM model to the zoo directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--zoo_path",
        type=str,
        default="/share/pierson/matt/zoo/models",
        help="Path to zoo models directory (default: /share/pierson/matt/zoo/models)",
    )
    parser.add_argument(
        "--use_symlinks",
        action="store_true",
        help="Use symlinks instead of copying files (saves disk space but requires original location)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model directory exists",
    )
    
    args = parser.parse_args()
    
    target_path = Path(args.zoo_path) / args.model_name.split("/")[-1]
    
    if target_path.exists() and not args.force:
        print(f"Model already exists at {target_path}")
        print("Use --force to re-download.")
        return
    
    if args.force and target_path.exists():
        import shutil
        print(f"Removing existing model directory: {target_path}")
        shutil.rmtree(target_path)
    
    download_model(args.model_name, args.zoo_path, args.use_symlinks)
    
    print(f"\nModel downloaded to: {target_path}")
    print(f"\nTo use this model, update your pipeline config:")
    print(f"  model.model_source: {target_path}")
    print(f"\nOr use the model config:")
    print(f"  - override /model: vllm_multimodal_zoo")
    print(f"  And set model.model_source: {target_path}")


if __name__ == "__main__":
    main()

