"""Embed event keyframes with Qwen3-VL-Embedding-8B through vLLM.

Run this script with the repository venv on a GPU node. The launcher
scripts/embed_events.sub does that. The input is the frames manifest
from ``trashbot-annotate embed-frames``. The output parquet holds one
L2-normalized embedding per event keyframe.

This mirrors the message and pooling pattern of the urbanembed embed
stage (dagspaces/urbanembed/stages/embed.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# One instruction for all frames. The embedding then favors the people
# and the action over the raw scene layout.
INSTRUCTION = (
    "Represent the people in this park scene and their appearance, "
    "clothing, and action toward the trash robot."
)

# Fixed 1024x1024, the same pin as the urbanembed 8B config. A fixed
# shape keeps the batch memory stable.
MIN_PIXELS = 1049600
MAX_PIXELS = 1049600
BATCH_SIZE = 16


def build_messages(image_path: str) -> list[dict[str, object]]:
    """Build the chat messages for one keyframe."""
    return [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }
            ],
        },
    ]


def main() -> int:
    """Embed every frame in the manifest. Write the embeddings parquet."""
    parser = argparse.ArgumentParser(description="Embed TrashBot event keyframes.")
    parser.add_argument("--frames-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from PIL import Image
    from vllm import LLM

    manifest = pd.read_parquet(args.frames_manifest)
    if args.limit is not None:
        manifest = manifest.head(max(args.limit, 0))
    print(f"[embed_events] {len(manifest)} frames to embed", flush=True)

    llm = LLM(
        model=args.model_path,
        runner="pooling",
        max_model_len=8192,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()

    rows: list[dict[str, object]] = []
    records = manifest.to_dict("records")
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        inputs = []
        kept = []
        for row in batch:
            image_path = str(row["image_path"])
            if not Path(image_path).is_file():
                print(f"[embed_events] SKIP missing frame: {image_path}", flush=True)
                continue
            prompt_text = tokenizer.apply_chat_template(
                build_messages(image_path), tokenize=False, add_generation_prompt=True
            )
            try:
                image = Image.open(image_path).convert("RGB")
            except OSError as exc:
                print(f"[embed_events] SKIP unreadable frame: {image_path}: {exc}", flush=True)
                continue
            inputs.append({"prompt": prompt_text, "multi_modal_data": {"image": image}})
            kept.append(row)
        if not inputs:
            continue
        outputs = llm.embed(inputs)
        for row, output in zip(kept, outputs):
            embedding = np.asarray(output.outputs.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
            rows.append(
                {
                    "event_id": row["event_id"],
                    "embedding": embedding.tolist(),
                    "embedding_dim": int(embedding.shape[0]),
                    "model_source": args.model_path,
                }
            )
        print(f"[embed_events] {len(rows)} embedded", flush=True)

    if len(manifest) and not rows:
        print("[embed_events] ERROR: no frame produced an embedding", flush=True)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[embed_events] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
