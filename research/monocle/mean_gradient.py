"""The mean-gradient direction: the cheapest test of a CAUSAL constant lens.

Why this module exists
----------------------
Read the 2 maps the causal gate compares, and they turn out to be the same
inner product with a different direction:

| Map | What it computes | Direction |
|---|---|---|
| `taylor` | `<g_p, dh_p>` | LOCAL, `g_p = dc/dh_l` at this image |
| `lens_delta` | `readout(J h_don) - readout(J h)` | AVERAGED, `J^T w` |

`w` is the contrast row of the unembed, so `J^T w = E[dc/dh_l]`. Thus the
whole question "can a fitted lens be causal" reduces to ONE question:

    How well does an input-independent direction approximate the local one?

A jlens answers that with 480 backward passes per image. This module answers
it with ONE, because it never builds the matrix -- it averages the contrast
gradient directly. That buys 2 things:

  * a 480x cheaper estimate of the same direction, and
  * ANY layer. The jlens is fitted at [6,12,18,24,30,36,42,46], so it has
    never been fitted at L20, which is where the image acts hardest.

The 2 constants
---------------
| Name | Shape | What it assumes |
|---|---|---|
| `global` | `[d_model]` | one direction for every patch |
| `per_pos` | `[n_patches, d_model]` | one direction per patch POSITION |

`per_pos` is still input-independent, so it is still a constant lens. It is a
strictly richer constant, and it can carry positional structure such as "the
lower-centre patches matter more".

The hold-out rule
-----------------
`single.load_single_rows` returns a deterministic prefix. The causal gate
reads the first `ANALYSIS_RESERVE` rows. We fit on the rows AFTER them.

Usage:
    python -m research.monocle.mean_gradient --case subway_safety --n-images 48
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import single as sg  # noqa: E402

#: Rows the causal gate may read. We never fit on these. Matches
#: monocle.jlens_fit_ladder.ANALYSIS_RESERVE.
ANALYSIS_RESERVE = 32

#: A dense grid around L20, where the image acts hardest, plus the layers the
#: jlens is fitted at so the 2 instruments can be compared directly.
DEFAULT_LAYERS = (12, 16, 18, 20, 22, 24, 26, 28)

DEFAULT_OUT = str(REPO / "outputs/_monocle/mean_grad")


def log(m: str) -> None:
    print(f"[mean-grad] {m}", flush=True)


def heldout_rows(case: str, n: int, *, reserve: int = ANALYSIS_RESERVE
                 ) -> list[dict]:
    """Images the causal gate never reads."""
    rows = sg.load_single_rows(reserve + n, case)
    if len(rows) <= reserve:
        raise RuntimeError(
            f"{case}: pool holds {len(rows)} rows, at or below the reserve "
            f"of {reserve}")
    return rows[reserve:]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine in float64.

    Warning: a float32 reduction over a gemma-4 residual loses enough
    precision to report a cosine above 1.0. The massive activations widen the
    range and make it worse.
    """
    x, y = a.double().flatten(), b.double().flatten()
    denom = x.norm() * y.norm()
    return float(x.dot(y) / denom) if denom > 0 else float("nan")


def collect(
    case: str, rows: list[dict], layers: list[int], *,
    proc: Any, model: Any, lens_model: Any, tmpl: str, device: str,
    n_samples: int = 0, sample_seed: int = 777,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[dict],
           dict[int, torch.Tensor]]:
    """Accumulate the contrast gradient over images.

    Returns ``(per_pos_mean, global_mean, per_image_diagnostics, samples)``.

    When `n_samples` is above 0, the run also keeps that many patch gradients
    for each image. Those vectors answer a question that the mean cannot: does
    ONE direction serve every patch, or does a small basis serve them?
    """
    from PIL import Image
    from research.monocle import safety_workspace as sw

    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)

    total: dict[int, torch.Tensor] = {}
    samples: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    srng = np.random.default_rng(sample_seed)
    n_done = 0
    diags: list[dict] = []
    t0 = time.time()

    for i, r in enumerate(rows):
        try:
            img = Image.open(r["path"]).convert("RGB")
            inputs = sg.build_single_inputs(
                proc, tmpl, img, case, device=device)
            seq_len = int(inputs["input_ids"].shape[1])
            block = [int(x) for x in
                     sw.image_token_blocks(inputs, image_token_id,
                                           expect=1)[0]]
            grads, _acts, c = pa.attribute_pair(
                model, lens_model, inputs, layers, pos_id, neg_id,
                seq_len - 1)
        except Exception as exc:  # noqa: BLE001 -- unreadable image
            log(f"  skip {i} ({r.get('image_id', r['path'])}): {exc}")
            continue

        d: dict[str, Any] = {"image_id": r.get("image_id"), "c": float(c)}
        take = (srng.choice(len(block), size=min(n_samples, len(block)),
                            replace=False) if n_samples > 0 else None)
        for layer in layers:
            g = grads[layer][0, block, :].float().cpu()
            if take is not None:
                samples[layer].append(g[take].clone())
            if layer not in total:
                total[layer] = torch.zeros_like(g)
            # How far is this image's direction from the running mean? A high
            # cosine means a constant direction can work; a low one means no
            # constant can.
            if n_done > 0:
                d[f"cos_L{layer}"] = cosine(g, total[layer] / n_done)
            total[layer] += g
        diags.append(d)
        n_done += 1
        if n_done % 8 == 0:
            log(f"  {n_done}/{len(rows)} images, {time.time() - t0:.0f}s")

    if n_done == 0:
        raise RuntimeError("no image contributed a gradient")
    per_pos = {l: total[l] / n_done for l in layers}
    glob = {l: per_pos[l].mean(dim=0) for l in layers}
    log(f"fitted on {n_done} images")
    stacked = {l: (torch.cat(samples[l], dim=0) if samples[l]
                   else torch.empty(0)) for l in layers}
    return per_pos, glob, diags, stacked


def summarize(diags: list[dict], layers: list[int]) -> dict[int, dict]:
    """Per-layer spread of the gradient direction across images."""
    out = {}
    for layer in layers:
        vals = [d[f"cos_L{layer}"] for d in diags
                if f"cos_L{layer}" in d and np.isfinite(d[f"cos_L{layer}"])]
        out[layer] = {
            "n": len(vals),
            "mean_cos_to_running_mean": float(np.mean(vals)) if vals else
            float("nan"),
            "min": float(np.min(vals)) if vals else float("nan"),
            "max": float(np.max(vals)) if vals else float("nan"),
        }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar
    from research.monocle import jlens_read

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="subway_safety",
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=48)
    ap.add_argument("--layers", nargs="+", type=int,
                    default=list(DEFAULT_LAYERS))
    ap.add_argument("--save-grads", type=int, default=0,
                    metavar="N",
                    help="keep N patch gradients for each image. "
                         "They give the rank of the gradient set, "
                         "which the mean cannot give.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    layers = sorted(set(args.layers))
    rows = heldout_rows(args.case, args.n_images)
    log(f"case={args.case} layers={layers} images={len(rows)} "
        f"(after a reserve of {ANALYSIS_RESERVE})")

    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)

    per_pos, glob, diags, samples = collect(
        args.case, rows, layers, proc=proc, model=model,
        lens_model=lens_model, tmpl=tmpl, device=args.device,
        n_samples=args.save_grads)

    stats = summarize(diags, layers)
    log("how much does the direction VARY across images?")
    log(f"{'layer':>6} {'n':>4} {'mean cos':>10} {'min':>8} {'max':>8}")
    for layer in layers:
        s = stats[layer]
        log(f"{layer:>6} {s['n']:>4} {s['mean_cos_to_running_mean']:>10.3f} "
            f"{s['min']:>8.3f} {s['max']:>8.3f}")

    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"per_pos": per_pos, "global": glob, "layers": layers,
                "case": args.case, "n_images": len(diags),
                "reserve": ANALYSIS_RESERVE}, out_dir / "mean_grad.pt")
    (out_dir / "spread.json").write_text(json.dumps(
        {"case": args.case, "n_images": len(diags),
         "per_layer": {str(k): v for k, v in stats.items()}}, indent=1))
    if args.save_grads > 0:
        torch.save({"samples": samples, "layers": layers,
                    "case": args.case, "per_image": args.save_grads,
                    "n_images": len(diags)},
                   out_dir / "grad_samples.pt")
        n = int(samples[layers[0]].shape[0])
        log(f"saved {n} patch gradients for each layer")
    log(f"saved {out_dir}/mean_grad.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
