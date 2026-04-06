"""Batch image embedding using Qwen3-VL-Embedding models via Ray actors.

Each EmbeddingActor loads the model on a single GPU and processes batches
of images. The run_embed_stage() function distributes work across available
GPUs using Ray Data map_batches with ActorPoolStrategy.
"""

from __future__ import annotations

import logging
import os
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

logger = logging.getLogger(__name__)

MAX_LENGTH = 8192


# ---------------------------------------------------------------------------
# Vendored model class from Qwen3-VL-Embedding model scripts
# ---------------------------------------------------------------------------

@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pooling_last(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Extract the last non-padding token's hidden state per row."""
    flipped = attention_mask.flip(dims=[1])
    last_one_pos = flipped.argmax(dim=1)
    col = attention_mask.shape[1] - last_one_pos - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


def _format_image_conversation(
    image_path: str,
    instruction: str,
    min_pixels: int,
    max_pixels: int,
) -> List[Dict]:
    """Build a conversation list for a single image embedding request."""
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction = instruction + "."

    image_ref = image_path if image_path.startswith(("http", "oss")) else "file://" + image_path

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_ref,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            ],
        },
    ]


def _load_images_from_conversations(conversations: List[List[Dict]]) -> List[Image.Image]:
    """Load PIL images from conversation dicts (one image per conversation)."""
    images = []
    for conv in conversations:
        for msg in conv:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    ref = item["image"]
                    if isinstance(ref, Image.Image):
                        images.append(ref)
                    elif isinstance(ref, str):
                        path = ref.removeprefix("file://")
                        images.append(Image.open(path).convert("RGB"))
    return images


def _preprocess_batch(
    processor: Qwen3VLProcessor,
    conversations: List[List[Dict]],
) -> Dict[str, torch.Tensor]:
    """Tokenize + load images for a batch of conversations."""
    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    images = _load_images_from_conversations(conversations)
    inputs = processor(
        text=text,
        images=images if images else None,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    return inputs


# ---------------------------------------------------------------------------
# Ray actor for distributed embedding
# ---------------------------------------------------------------------------

class EmbeddingActor:
    """Stateful Ray actor that loads the model on a single GPU."""

    def __init__(self, cfg: Dict[str, Any]):
        import torch

        self.instruction: str = cfg["instruction"]
        self.normalize: bool = cfg["normalize"]
        self.output_dim: Optional[int] = cfg.get("output_dim")
        self.min_pixels: int = cfg["min_pixels"]
        self.max_pixels: int = cfg["max_pixels"]
        self.image_col: str = cfg.get("image_col", "image_path")
        self.model_source: str = cfg["model_source"]

        attn_impl = "flash_attention_2"
        try:
            self.model = Qwen3VLForEmbedding.from_pretrained(
                self.model_source,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            ).cuda().eval()
        except (ImportError, ValueError):
            attn_impl = "sdpa"
            self.model = Qwen3VLForEmbedding.from_pretrained(
                self.model_source,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            ).cuda().eval()

        self.processor = Qwen3VLProcessor.from_pretrained(
            self.model_source, padding_side="right"
        )
        print(f"[EmbeddingActor] Model loaded from {self.model_source} (attn={attn_impl})", flush=True)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image_paths = batch[self.image_col]
        batch_size = len(image_paths)

        conversations = [
            _format_image_conversation(
                str(path), self.instruction, self.min_pixels, self.max_pixels
            )
            for path in image_paths
        ]

        inputs = _preprocess_batch(self.processor, conversations)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embs = _pooling_last(outputs.last_hidden_state, inputs["attention_mask"])
        if self.output_dim:
            embs = embs[:, : int(self.output_dim)]
        if self.normalize:
            embs = F.normalize(embs, p=2, dim=-1)

        embs_np = embs.cpu().float().numpy()

        result = {}
        for col_name, col_data in batch.items():
            result[col_name] = col_data
        result["embedding"] = list(embs_np)
        result["embedding_dim"] = np.full(batch_size, embs_np.shape[1], dtype=np.int32)
        result["model_source"] = np.array(
            [self.model_source] * batch_size, dtype=object
        )

        return result


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def _detect_gpu_count() -> int:
    """Detect available GPUs from CUDA_VISIBLE_DEVICES or torch."""
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_vis:
        return len([d for d in cuda_vis.split(",") if d.strip()])
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


def run_embed_stage(cfg: DictConfig) -> str:
    """Run the embedding stage using Ray Data with actor pool.

    Args:
        cfg: Hydra config with data, model, embedding, and runtime sections.

    Returns:
        Path to the output parquet file/directory.
    """
    import ray
    import ray.data
    from ray.data import ActorPoolStrategy

    from ..multiprocessing_utils import ensure_ray_init

    ensure_ray_init(cfg, caller="run_embed_stage")

    # Read input data
    parquet_path = cfg.data.parquet_path
    print(f"[run_embed_stage] Reading parquet: {parquet_path}", flush=True)
    ds = ray.data.read_parquet(parquet_path)

    # Debug sampling
    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if sample_n:
        n = int(sample_n)
        ds = ds.limit(n)
        print(f"[run_embed_stage] Limited to {n} rows for debug", flush=True)

    # Detect GPUs
    num_gpus = _detect_gpu_count()
    print(f"[run_embed_stage] Using {num_gpus} GPUs", flush=True)

    # Build actor config dict (must be serializable)
    image_col = str(getattr(getattr(cfg, "data", {}).get("columns", {}), "image_path", "image_path"))
    if hasattr(cfg.data, "columns"):
        image_col = str(getattr(cfg.data.columns, "image_path", "image_path"))

    actor_cfg = {
        "model_source": str(cfg.model.model_source),
        "instruction": str(cfg.embedding.instruction),
        "normalize": bool(cfg.embedding.normalize),
        "output_dim": int(cfg.embedding.output_dim) if cfg.embedding.get("output_dim") else None,
        "min_pixels": int(cfg.embedding.min_pixels),
        "max_pixels": int(cfg.embedding.max_pixels),
        "image_col": image_col,
    }

    batch_size = int(cfg.embedding.batch_size)

    ds = ds.map_batches(
        EmbeddingActor,
        fn_constructor_kwargs={"cfg": actor_cfg},
        compute=ActorPoolStrategy(size=num_gpus),
        num_gpus=1,
        batch_size=batch_size,
    )

    # Determine output path
    output_path = str(getattr(cfg.runtime, "output_path", None) or "")
    if not output_path:
        output_path = os.path.join("outputs", "embed", "embeddings.parquet")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Check if streaming_io — write directory of parquet parts
    streaming_io = bool(getattr(getattr(cfg, "runtime", {}), "streaming_io", False))
    if streaming_io:
        # Write partitioned parquet directory
        output_dir = output_path.replace(".parquet", "")
        os.makedirs(output_dir, exist_ok=True)
        ds.write_parquet(output_dir)
        print(f"[run_embed_stage] Wrote streaming parquet to: {output_dir}", flush=True)
        return output_dir
    else:
        # Materialize and write single file
        result_df = ds.to_pandas()
        result_df.to_parquet(output_path, index=False)
        print(f"[run_embed_stage] Wrote {len(result_df)} rows to: {output_path}", flush=True)
        return output_path
