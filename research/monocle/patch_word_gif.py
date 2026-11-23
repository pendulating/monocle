"""A GIF of the top-3 words of each image patch, 1 frame for each fitted layer.

The lens reads every patch of 1 photograph at every layer that the lens holds.
Each frame draws the top-3 words over the patch that says them. The frames
then run from the first fitted layer to the last, so the animation shows how
the words change with depth.

The font size tracks the score across the WHOLE stack, and not inside 1 frame.
Thus a quiet early layer draws small and a confident late layer draws large.
`monocle.render_layers` holds that normalization.

Warning: the R-lens gives junk below block 30, and the J-lens gives junk below
block 36. Read `research/docs/r-lens.md` before you trust an early frame.

Usage:
    python -m research.monocle.patch_word_gif --case restaurants --n-images 3
    python -m research.monocle.patch_word_gif --image /path/to.jpg --case libraries \\
        --lens outputs/_monocle/jlens/mm/lrp/gemma4_12b_lens.pt --tag rlens
    python -m research.monocle.patch_word_gif --sample data/rq1/patch_words_sample.parquet \
        --pool 2 --out data/rq1/patch_words

`--sample` reads a parquet with the columns `photo_id`, `image_path` and
`lens_case` (the question of each image), and loads the model 1 time for all
the images. The RQ1 figure uses it (`research/scripts/rq1/patch_words_sample.py`).

`--layout text_first` puts the question before the image. A patch reads only
the tokens before it, so only this layout lets the question change the patch
words (`research/docs/modality-order.md`). Give it its own `--tag`.

`text_first` also moves the patches to later positions. The pair
`question_first` (question, image, filler) and `neutral_first` (filler,
image, question) holds the positions fixed: the filler repeats "This is a
photograph." to the token count of the question
(`monocle.modality_order.matched_filler`). Thus `question_first` minus
`neutral_first` is the effect of the question alone. The meta file gives
`patch_start`, so a reader can check that the 2 arms match.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

from research.monocle import attn_route as ar
from research.monocle import jlens_read
from research.monocle import patch_attribution as pa
from research.monocle import render_layers
from research.monocle import scoring
from research.monocle import single as sg

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LENS = str(REPO / "outputs/_monocle/jlens/mm/lrp/gemma4_12b_lens.pt")
DEFAULT_OUT = str(REPO / "outputs/_monocle/patch_word_gif")


def log(msg: str) -> None:
    print(f"[word-gif] {msg}", flush=True)


def load_model(model_dir: str, device: str) -> tuple[Any, Any, str]:
    """Give (processor, model, chat template) for a gemma4_unified or a released J++ model.

    `device` is a device or `auto`. `auto` spreads a model that does not fit
    on 1 GPU; the inputs then start on `cuda:0`.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from dagspaces.common.model_registry import resolve_model_source
    from dagspaces.common.vllm_inference import gemma4_unified_chat_template

    if gemma4_unified_chat_template(model_dir)[0]:
        return ar.load_eager(model_dir, device)
    src = resolve_model_source(model_dir, stage_name="word-gif")
    proc = AutoProcessor.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map=device,
                                                        attn_implementation="sdpa")
    return proc, model.eval(), proc.chat_template


def layer_frame(logits: torch.Tensor, tok: Any, layer: int, *, k: int,
                grid: tuple[int, int], mask: torch.Tensor,
                pool: int = 1) -> pd.DataFrame:
    """The top-k words of every patch at 1 layer, as long rows.

    `pool` averages the word probabilities over pool x pool blocks first. A
    16x16 grid holds 768 labels at k=3, which no reader can see. pool=2 gives
    an 8x8 grid, and pool=4 gives 4x4.
    """
    df = scoring.score_patches(logits, tok, k=k, token_mask=mask, pool=pool,
                               grid_shape=grid)
    rows, cols = scoring.pooled_dims(grid[0], grid[1], pool)
    df = scoring.attach_grid(df, rows, cols)
    return df.assign(layer=int(layer))


def one_image(image: Any, case: str, *, proc: Any, model: Any, lens_model: Any,
              lens: Any, tmpl: str, k: int, device: str, pool: int = 1,
              layers: Optional[list[int]] = None,
              layout: str = "image_first") -> tuple[pd.DataFrame, dict]:
    """Read every patch at every layer, and return the long frame and the meta."""
    filler = None
    if layout in sg.FILLER_LAYOUTS:
        from research.monocle.modality_order import matched_filler
        filler = matched_filler(proc.tokenizer, sg.prompt_text(case))
    inputs = sg.build_single_inputs(proc, tmpl, image, case, device=device,
                                    layout=layout, filler=filler)
    want = list(layers) if layers else list(lens.source_layers)
    acts, pos, final = jlens_read.record_patch_activations(
        model, lens_model, inputs, want)
    per_layer = jlens_read.lens_patch_logits(
        lens, lens_model, acts, pos, final, layers=want)

    n_patch = int(pos.numel())
    side = int(round(n_patch ** 0.5))
    if side * side != n_patch:
        raise ValueError(f"{n_patch} patches is not a square grid")
    mask = scoring.build_token_mask(proc.tokenizer, int(final.shape[-1]))

    parts = [layer_frame(lg, proc.tokenizer, layer, k=k, grid=(side, side),
                         mask=mask, pool=pool)
             for layer, lg in sorted(per_layer.items())]
    del acts, per_layer, final
    rows, cols = scoring.pooled_dims(side, side, pool)
    meta = {"n_rows": rows, "n_cols": cols, "patch_start": int(pos[0]),
            "seq_len": int(inputs["input_ids"].shape[-1])}
    return pd.concat(parts, ignore_index=True), meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="restaurants",
                    help="the question that the model reads")
    ap.add_argument("--image", default=None,
                    help="1 image path; without it the case supplies them")
    ap.add_argument("--sample", default=None,
                    help="a parquet with photo_id, image_path and lens_case")
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="a subset of the fitted layers")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=1,
                    help="average over pool x pool blocks; 2 gives an 8x8 grid")
    ap.add_argument("--ms", type=int, default=900, help="ms for each frame")
    ap.add_argument("--dim", type=float, default=0.55,
                    help="how much the photograph fades behind the words")
    ap.add_argument("--lens", default=DEFAULT_LENS)
    ap.add_argument("--tag", default="rlens")
    ap.add_argument("--layout", default="image_first",
                    choices=sg.LAYOUTS,
                    help="the block order of the prompt (monocle.single.build_messages)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keep-frames", action="store_true",
                    help="write each frame as a PNG beside the GIF")
    a = ap.parse_args(argv)

    from PIL import Image

    proc, model, tmpl = load_model(a.model_dir, a.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(a.lens)
    log(f"lens {a.lens}")
    a.device = "cuda:0" if a.device == "auto" else a.device
    log(f"fitted at {lens.source_layers}")

    if a.sample:
        s = pd.read_parquet(a.sample)
        jobs = list(zip(s["photo_id"].astype(str), s["image_path"], s["lens_case"]))
    elif a.image:
        jobs = [(sg.photograph_id(a.image), a.image, a.case)]
    else:
        jobs = [(str(r["image_id"]), r["path"], a.case)
                for r in sg.load_single_rows(a.n_images, a.case, seed=a.seed)]

    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (image_id, path, case) in enumerate(jobs):
        img = Image.open(path).convert("RGB")
        df, meta = one_image(img, case, proc=proc, model=model,
                             lens_model=lens_model, lens=lens, tmpl=tmpl,
                             k=a.top_k, device=a.device, pool=a.pool,
                             layers=a.layers, layout=a.layout)
        meta["image_id"] = image_id
        meta["layout"] = a.layout
        frames = render_layers.render_layer_gif(
            img, df, meta, k=a.top_k, dim=a.dim)
        stem = f"{image_id}_{case}"
        if a.pool > 1:
            stem += f"_p{a.pool}"
        gif = out_dir / f"{stem}.gif"
        render_layers.save_layer_gif(frames, gif, ms_per_frame=a.ms)
        df.to_parquet(out_dir / f"{stem}.jlens.parquet", index=False)
        (out_dir / f"{stem}.jlens.meta.json").write_text(
            json.dumps(meta, indent=2))
        if a.keep_frames:
            for layer, frame in zip(sorted(df.layer.unique()), frames):
                frame.save(out_dir / f"{stem}_L{int(layer):02d}.png")
        log(f"  {i + 1}/{len(jobs)} {gif.name} "
            f"({len(frames)} frames, {time.time() - t0:.0f}s)")
    log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
