"""Shared parts of the whatami lens study: the model table, the HF loaders,
and the input builders.

The lens study reads the residual stream of the four whatami models in HF
transformers (the default venv, transformers 5.13). The prompts must agree
with the prompts that the served run used (research/scripts/whatami/query.py): one user
turn, the image and then the text, no system prompt, thinking off. The served
prompt lengths below are the check (from responses.parquet).

Where the image sits in the sequence:

| Family | Image in the decoder sequence | Image lens sources |
|---|---|---|
| gemma-4, qwen3.5 | a contiguous block of image tokens | the patches, or the text after them |
| mllama (Llama 3.2 Vision) | 1 `<|image|>` token; the pixels enter through cross-attention layers | the text after the image token only |
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

REPO = Path("/share/ju/matt/mllmsci")
for _p in (REPO, REPO / "sub/jacobian-lens"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ZOO = Path("/share/ju/matt/zoo/models")
LENS_ROOT = REPO / "outputs/_monocle/jlens/whatami"
TEXT_CORPUS = REPO / "outputs/_monocle/jlens/corpus_wikitext_1k.json"
IMAGE_CORPUS = REPO / "outputs/_monocle/jlens/mm/fit_images.json"
PROMPT = "What am I looking at?"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    zoo_dir: str
    family: str  # gemma4 | qwen3_5 | mllama
    #: prompt_tokens in the served run (responses.parquet).
    served_prompt_tokens: int
    #: For mllama the server counts the image tiles; HF does not.
    served_image_tokens: int = 0
    chat_template_kwargs: dict = field(default_factory=dict)
    processor_kwargs: dict = field(default_factory=dict)

    @property
    def path(self) -> str:
        return str(ZOO / self.zoo_dir)

    @property
    def hf_prompt_tokens(self) -> int:
        return self.served_prompt_tokens - self.served_image_tokens


MODELS: dict[str, ModelSpec] = {
    "gemma-4-12b": ModelSpec(
        "gemma-4-12b", "gemma-4-12B-it", "gemma4", 277,
        chat_template_kwargs={"enable_thinking": False}),
    "qwen3.5-9b": ModelSpec(
        "qwen3.5-9b", "Qwen3.5-9B", "qwen3_5", 1044,
        chat_template_kwargs={"enable_thinking": False},
        # The same pixel cap as the served run (serve_and_query.sbatch).
        processor_kwargs={"images_kwargs": {"min_pixels": 784, "max_pixels": 1048576}}),
    "llama-3.2-11b": ModelSpec(
        "llama-3.2-11b", "Llama-3.2-11B-Vision-Instruct", "mllama", 6421,
        served_image_tokens=4 * 1601),
    "llama-3.2-90b": ModelSpec(
        "llama-3.2-90b", "Llama-3.2-90B-Vision-Instruct", "mllama", 6421,
        served_image_tokens=4 * 1601),
}

#: The R-lenses (LRP rules in the backward pass, research/monocle/rlens_rules.py) take
#: the same names with the prefix "r_".
RLENS_PREFIX = "r_"

#: The lenses of each family. `mm` takes the image patches as its sources; mllama has no patches.
LENSES = {
    "gemma4": ["text", "mm", "mm_textafter"],
    "qwen3_5": ["text", "mm", "mm_textafter"],
    "mllama": ["text", "mm_textafter"],
}


def log(msg: str) -> None:
    print(f"[whatami-lens] {msg}", flush=True)


def lens_dir(model_key: str, lens: str) -> Path:
    return LENS_ROOT / model_key / lens


# ── loading ────────────────────────────────────────────────────────────
@dataclass
class Loaded:
    spec: ModelSpec
    proc: object
    model: torch.nn.Module
    lens_model: object
    chat_template: str | None
    image_token_id: int
    end_of_turn_id: int


def _model_class(family: str):
    import transformers as tf
    return {
        "gemma4": tf.Gemma4UnifiedForConditionalGeneration,
        "qwen3_5": tf.AutoModelForImageTextToText,
        "mllama": tf.MllamaForConditionalGeneration,
    }[family]


def load(model_key: str, *, model_dir: str | None = None, force_bos: bool = False,
         device: str = "cuda:0", eager: bool = False) -> Loaded:
    """Load the processor, the model (bf16, eval) and the jlens wrapper.

    `force_bos` goes to jlens.from_hf. Use True for the raw-text lens only:
    it sets add_bos_token on the SHARED tokenizer, and a chat template that
    writes its own BOS then gets two.

    `eager` loads eager attention. An R-lens fit needs it: SDPA computes the
    softmax inside a fused kernel, where the LRP softmax rule cannot reach it.
    """
    import jlens
    from transformers import AutoProcessor

    spec = MODELS[model_key]
    # A job sets WHATAMI_MODEL_DIR to the /scratch copy (stage_inputs.sh).
    src = model_dir or os.environ.get("WHATAMI_MODEL_DIR") or spec.path
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(src)
    extra = {"attn_implementation": "eager"} if eager else {}
    model = _model_class(spec.family).from_pretrained(
        src, dtype=torch.bfloat16, device_map=device, **extra)
    model.eval()
    model.requires_grad_(False)
    lens_model = jlens.from_hf(model, proc.tokenizer, force_bos=force_bos)
    tmpl = None
    jinja = Path(src) / "chat_template.jinja"
    if spec.family == "gemma4" and jinja.exists():
        # The tokenizer's default template omits the image token for
        # gemma-4 unified (see _gemma4_unified_chat_template).
        tmpl = jinja.read_text()
    cfg = model.config
    image_token_id = int(getattr(cfg, "image_token_id", None)
                         or getattr(cfg, "image_token_index"))
    tok = proc.tokenizer
    eot = {"gemma4": "<turn|>", "qwen3_5": "<|im_end|>", "mllama": "<|eot_id|>"}[spec.family]
    eot_id = tok.convert_tokens_to_ids(eot)
    if eot_id is None or eot_id == tok.unk_token_id:
        raise ValueError(f"{model_key}: no end-of-turn token {eot!r}")
    log(f"loaded {model_key} in {time.time() - t0:.0f}s: {type(model).__name__}, "
        f"n_layers={lens_model.n_layers}, d_model={lens_model.d_model}, "
        f"image_token_id={image_token_id}, eot={eot}:{eot_id}")
    return Loaded(spec, proc, model, lens_model, tmpl, image_token_id, int(eot_id))


# ── inputs ─────────────────────────────────────────────────────────────
def open_image(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def build_prompt(L: Loaded, image_path: str, user_text: str, device: str = "cuda:0") -> dict:
    """The served prompt: one user turn, the image then the text, no system prompt."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    kw = dict(L.spec.chat_template_kwargs)
    if L.chat_template is not None:
        kw["chat_template"] = L.chat_template
    text = L.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    image = open_image(image_path)
    pk = dict(L.spec.processor_kwargs)
    # The served engines add no BOS beyond the one the template writes.
    last: Exception | None = None
    for images in ([[image]], [image]):
        try:
            return dict(L.proc(text=[text], images=images, return_tensors="pt",
                               add_special_tokens=False, **pk).to(device))
        except Exception as exc:  # noqa: BLE001 - processors differ on nesting
            last = exc
    raise RuntimeError(f"processor rejected both image nestings: {last}")


def image_positions(L: Loaded, inputs: dict) -> torch.Tensor:
    """Positions of the image token(s); a contiguous block for gemma and qwen."""
    ids = inputs["input_ids"][0]
    pos = (ids == L.image_token_id).nonzero(as_tuple=True)[0].cpu()
    if pos.numel() == 0:
        raise ValueError("no image token in the prompt")
    if L.spec.family == "mllama":
        if pos.numel() != 1:
            raise ValueError(f"mllama: expected 1 <|image|> token, found {pos.numel()}")
    elif int(pos[-1] - pos[0]) + 1 != pos.numel():
        raise ValueError("the image block is not contiguous")
    return pos


def append_tokens(inputs: dict, new_ids: list[int]) -> dict:
    """Append token ids to a 1-row processor output.

    Every [1, seq] tensor grows: input_ids with the new ids, attention_mask
    with ones, and a type-id tensor with zeros (text). The mllama
    cross_attention_mask [1, seq, images, tiles] repeats its last row, thus the
    new text attends to the image as the prompt text does.
    """
    seq = int(inputs["input_ids"].shape[1])
    n = len(new_ids)
    if n == 0:
        return dict(inputs)
    dev = inputs["input_ids"].device
    out = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v) or v.dim() < 2 or v.shape[0] != 1 or v.shape[1] != seq:
            out[k] = v
        elif k == "input_ids":
            out[k] = torch.cat([v, torch.tensor([new_ids], dtype=v.dtype, device=dev)], 1)
        elif k == "attention_mask":
            out[k] = torch.cat([v, torch.ones(1, n, dtype=v.dtype, device=dev)], 1)
        elif k == "cross_attention_mask":
            out[k] = torch.cat([v, v[:, -1:].expand(1, n, *v.shape[2:])], 1)
        elif v.dim() == 2:
            out[k] = torch.cat([v, torch.zeros(1, n, dtype=v.dtype, device=dev)], 1)
        else:
            raise ValueError(f"do not know how to extend {k} {tuple(v.shape)}")
    return out


def check_prompt_length(L: Loaded, inputs: dict) -> None:
    n = int(inputs["input_ids"].shape[1])
    if n != L.spec.hf_prompt_tokens:
        raise ValueError(
            f"{L.spec.key}: the HF prompt has {n} tokens, the served prompt "
            f"{L.spec.served_prompt_tokens} (expected {L.spec.hf_prompt_tokens} in HF). "
            "The prompts do not agree.")


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)
