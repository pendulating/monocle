"""Stage 1 of the dictionary plan: the residual extraction.

Writes 2 kinds of activation to a node-local root:

| Kind | Positions | Blocks | Images | Frame |
|------|-----------|--------|--------|-------|
| P<block> | the 280 image patches | `--patch-blocks` (22, 40) | the `train` split | the open one-word frame; the patch residual is question-blind, so the frame does not matter here |
| A<block>/<frame> | the answer position | `--answer-blocks` (42, 46) | every image | `open` and the 7 canonical single-image cases |

Layout under `--out-root`: `P22/shard03of24_part0.pt` holds
`{"acts": bf16 [N, 280, 3840], "image_path": [...], "block": 22}`;
`A42/open/shard03of24.pt` holds `{"acts": bf16 [N, 3840], "image_path": [...]}`.
Raw residuals, no centering: the training step centers.

The pretrained checkpoint has no chat template. Under `--template raw`
every frame is the completion form `<|image|>\\n` + text + the forced
prefix, and the case texts come from the same vqa config.

    sbatch --time=02:00:00 research/monocle/monocle.sub monocle.sae_extract --template chat \\
        --shard 0 --n-shards 24 --out-root /scratch/mwf62/sae/it
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import canonical  # noqa: E402
from research.monocle import probe_frame as pf  # noqa: E402
from research.monocle import single  # noqa: E402
from research.monocle.jlens_steer import load_model_and_lens  # noqa: E402
from research.monocle.probe_words import log, user_text  # noqa: E402

SAMPLE_DEFAULT = REPO / "outputs/_monocle/sae/sample/images.parquet"
PART = 512   # images per patch part file (about 1.1 GB per block in bf16)


def frame_texts(tok, tmpl, template: str) -> dict[str, str]:
    """{frame: rendered prompt string}. `open` is the one-word frame; the
    7 cases are the canonical single-image questions with the forced JSON
    prefix."""
    out = {}
    if template == "chat":
        out["open"] = pf.render_chat(tok, tmpl, user_text("open"))
        for case in canonical.CASES:
            out[case] = single.render_single_text(tok, tmpl, case)
    else:
        out["open"] = pf.render_raw(user_text("open"))
        for case in canonical.CASES:
            out[case] = "<|image|>\n" + single.prompt_text(case) + canonical.FORCE_PREFIX
    return out


def run_batch(model, proc, rendered, imgs, blocks, recorder_cls, layers):
    inputs = dict(proc(text=[rendered] * len(imgs), images=[[im] for im in imgs],
                       return_tensors="pt").to("cuda:0"))
    am = inputs.get("attention_mask")
    if am is not None and not bool(am.all()):
        raise RuntimeError("padding inside a batch")
    inputs["use_cache"] = False
    with recorder_cls(layers, at=list(blocks)) as rec, torch.no_grad():
        model(**inputs)
        acts = {l: rec.activations[l] for l in blocks}
    return inputs, acts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", choices=["chat", "raw"], default="chat")
    ap.add_argument("--images", default=str(SAMPLE_DEFAULT))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patch-blocks", default="22,40")
    ap.add_argument("--answer-blocks", default="42,46")
    ap.add_argument("--frames", default="all", help="all | open | a comma list of frames")
    ap.add_argument("--no-patches", action="store_true")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    args = ap.parse_args(argv)
    from jlens.hooks import ActivationRecorder
    from PIL import Image

    pblocks = [int(x) for x in args.patch_blocks.split(",")] if not args.no_patches else []
    ablocks = [int(x) for x in args.answer_blocks.split(",")]
    df = pd.read_parquet(args.images, columns=["image_path", "split"])
    df = df.iloc[args.shard::args.n_shards]
    if args.limit:
        df = df.head(args.limit)
    paths_all = df.image_path.tolist()
    paths_train = df[df.split == "train"].image_path.tolist()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(root)
    log(f"out root {root}: {du.free / 1e9:.0f} GB free of {du.total / 1e9:.0f}")
    tag = f"shard{args.shard:02d}of{args.n_shards:02d}"

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    if args.template == "chat" and not tmpl:
        raise RuntimeError(f"{args.model_dir} has no chat template; use --template raw")
    texts = frame_texts(tok, tmpl, args.template)
    frames = list(texts) if args.frames == "all" else args.frames.split(",")
    log(f"template {args.template}, model {args.model_dir}, {tag}: {len(paths_all)} images, "
        f"{len(paths_train)} train; patch blocks {pblocks}; answer blocks {ablocks}; frames {frames}")
    t0 = time.time()

    # 1. the patch residuals, train images, open frame
    if pblocks:
        for l in pblocks:
            (root / f"P{l}").mkdir(exist_ok=True)
        n_pos = None
        for p0 in range(0, len(paths_train), PART):
            part_paths = paths_train[p0:p0 + PART]
            buf = {l: [] for l in pblocks}
            for b0 in range(0, len(part_paths), args.batch):
                chunk = part_paths[b0:b0 + args.batch]
                imgs = [Image.open(q).convert("RGB") for q in chunk]
                inputs, acts = run_batch(model, proc, texts["open"], imgs, pblocks,
                                         ActivationRecorder, lens_model.layers)
                ids = inputs["input_ids"]
                pos = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
                if n_pos is None:
                    n_pos = int(pos.numel())
                    log(f"{n_pos} image tokens per image, span {int(pos[0])}..{int(pos[-1])}")
                if int(pos.numel()) != n_pos or int(pos[-1] - pos[0]) != n_pos - 1:
                    raise RuntimeError("image-token block changed size or is not contiguous")
                if not bool((ids == image_token_id).sum(dim=1).eq(n_pos).all()):
                    raise RuntimeError("image-token count differs inside the batch")
                for l in pblocks:
                    buf[l].append(acts[l][:, pos, :].to(torch.bfloat16).cpu())
            k = p0 // PART
            for l in pblocks:
                torch.save({"acts": torch.cat(buf[l]), "image_path": part_paths, "block": l,
                            "n_pos": n_pos}, root / f"P{l}" / f"{tag}_part{k}.pt")
            done = p0 + len(part_paths)
            log(f"  patches {done}/{len(paths_train)}, {(time.time() - t0) / done:.2f} s/image")

    # 2. the answer residuals, every image, every frame
    for fr in frames:
        for l in ablocks:
            (root / f"A{l}" / fr).mkdir(parents=True, exist_ok=True)
        buf = {l: [] for l in ablocks}
        t1 = time.time()
        for b0 in range(0, len(paths_all), args.batch):
            chunk = paths_all[b0:b0 + args.batch]
            imgs = [Image.open(q).convert("RGB") for q in chunk]
            _, acts = run_batch(model, proc, texts[fr], imgs, ablocks,
                                ActivationRecorder, lens_model.layers)
            for l in ablocks:
                buf[l].append(acts[l][:, -1, :].to(torch.bfloat16).cpu())
        for l in ablocks:
            torch.save({"acts": torch.cat(buf[l]), "image_path": paths_all, "block": l,
                        "frame": fr, "template": args.template},
                       root / f"A{l}" / fr / f"{tag}.pt")
        log(f"  answer frame {fr}: {len(paths_all)} images, "
            f"{(time.time() - t1) / max(len(paths_all), 1):.3f} s/image")
    log(f"done {tag} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
