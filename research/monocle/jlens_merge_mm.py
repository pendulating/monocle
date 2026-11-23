"""Merge multimodal J-lens shards into 1 lens, with a small memory cost.

`jlens_fit.py --merge` loads every shard at the same time. For the 47-layer
fit each shard holds 47 x 3840 x 3840 values, which is 1.4 GB in fp16 and
2.8 GB in fp32, so 16 shards need more than 22 GB. This merger reads 1 shard
at a time and adds it into a float32 accumulator, so the peak is about 4 GB.

A shard now goes to disk in fp32, and so does the merged lens. The fp16 error
does not cancel in the weighted mean.

It reads 2 kinds of file:

| File | Meaning |
|---|---|
| `shard{i}of{n}.pt` | a shard that ran to the end; holds the MEAN |
| `fit_ckpt_shard{i}of{n}.pt` | a shard that still runs; holds the SUM |

Both carry an image count, so 1 weighted mean covers a mix of the 2. A
checkpoint lets you merge before every shard is complete.

Usage:

    python -m research.monocle.jlens_merge_mm \\
        --in-dir outputs/_monocle/jlens/mm_all47 \\
        --out outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt

Add `--final-only` to refuse a partial shard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

def log(msg: str) -> None:
    print(f"[jlens-merge] {msg}", flush=True)


def _load_lens_class():
    try:
        from jlens import JacobianLens
    except ModuleNotFoundError:
        from jlens import JacobianLens
    return JacobianLens


def read_part(
    path: Path,
) -> tuple[dict[int, torch.Tensor], int, list[int], dict]:
    """Read 1 shard file. Return the SUM of the Jacobians, the image count,
    the source layers, and the shard meta.

    A checkpoint carries no lens meta, so its meta is an empty dict. A file
    from an older fitter also has no meta.

    Args:
        path: A `shard*.pt` file or a `fit_ckpt_shard*.pt` file.

    Raises:
        ValueError: If the file holds neither layout.
    """
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if "jacobian_sum" in blob:                      # a live checkpoint
        n = int(blob["n_done"])
        layers = [int(x) for x in blob["source_layers"]]
        js = {int(k): v for k, v in blob["jacobian_sum"].items()}
        return js, n, layers, {}
    if "J" in blob:                                 # a complete shard
        n = int(blob["n_prompts"])
        js = {int(k): v.to(torch.float32) * n for k, v in blob["J"].items()}
        return js, n, sorted(js), dict(blob.get("meta", {}))
    raise ValueError(f"{path} holds neither 'jacobian_sum' nor 'J'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final-only", action="store_true",
                    help="Use only shards that ran to the end.")
    ap.add_argument("--min-images", type=int, default=1,
                    help="Skip a shard that fitted fewer images than this.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    finals = sorted(in_dir.glob("shard*of*.pt"))
    done_ids = {p.name for p in finals}
    parts = list(finals)
    if not args.final_only:
        # A checkpoint counts only when the same shard has no final file.
        for ck in sorted(in_dir.glob("fit_ckpt_shard*of*.pt")):
            if ck.name.replace("fit_ckpt_", "") not in done_ids:
                parts.append(ck)
    if not parts:
        log(f"no shard files under {in_dir}")
        return 1

    JacobianLens = _load_lens_class()
    total: dict[int, torch.Tensor] = {}
    n_total = 0
    layers_ref: list[int] | None = None
    used: list[dict] = []

    for path in parts:
        js, n, layers, part_meta = read_part(path)
        kind = "ckpt" if path.name.startswith("fit_ckpt_") else "final"
        if n < args.min_images:
            log(f"  skip {path.name} ({kind}): only {n} images")
            continue
        if layers_ref is None:
            layers_ref = layers
        elif layers != layers_ref:
            raise ValueError(
                f"{path.name} holds layers {layers[:4]}...({len(layers)}), "
                f"not the {len(layers_ref)} of the first shard")
        for layer, mat in js.items():
            if layer in total:
                total[layer] += mat
            else:
                total[layer] = mat.clone()
        n_total += n
        used.append({"file": path.name, "kind": kind, "images": n,
                     "meta": part_meta})
        log(f"  + {path.name} ({kind}): {n} images  -> {n_total} total")
        del js

    if n_total == 0:
        log("no images in any shard")
        return 1

    assert layers_ref is not None
    mean = {layer: total[layer] / n_total for layer in layers_ref}
    d_model = mean[layers_ref[0]].shape[0]
    meta = {"merged": True, "parts": used}
    lens = JacobianLens(jacobians=mean, n_prompts=n_total, d_model=d_model,
                        meta=meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # fp32, not the fp16 default: the merge accumulates in float32, and fp16
    # would give that precision away at the last step. The R0 check measured a
    # 4e-4 residual that came from fp16 saves alone.
    lens.save(str(out), dtype=torch.float32)
    log(f"saved {out}: {len(layers_ref)} layers, {n_total} images, "
        f"d_model {d_model}")

    side = out.with_suffix(".merge.json")
    side.write_text(json.dumps(
        {"out": str(out), "n_images": n_total, "n_layers": len(layers_ref),
         "source_layers": layers_ref, "d_model": d_model, "parts": used,
         "meta": meta},
        indent=1))
    log(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
