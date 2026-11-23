"""Does the model NAME the thing the reader points at?

The gap
-------
The reader names a cell, and a crop of that cell names an object. But we
never asked the model itself. A judgment the model can explain and a
judgment the reader can find may be 2 different things.

The test
--------
For each photograph:

1. Ask the case question and read the answer the model gives.
2. Ask a follow-up, with the same picture: name the ONE thing that
   most made you answer that way.
3. Compare the words of that answer with the caption of the cell the
   READER names, and with the caption of a random cell of the same
   photograph.

A hit means a content word of the self-report also appears in the crop
caption. The random crops of the same photograph give the control, so
the scene, the camera and the light never move.

| Outcome | What it means |
|---|---|
| the pole crop hits more | the model names what the reader finds |
| the arms tie | the model explains something else |

    python -m research.monocle.self_report --case restaurants --n-images 60 \
        --maps outputs/_monocle/gateB60_g1L/restaurants/maps.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

from research.monocle import caption_loop as cl  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_caption as pc  # noqa: E402
from research.monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/self_report"


def log(m: str) -> None:
    print(f"[self-report] {m}", flush=True)


def ask_text(proc: Any, tmpl: str, case: str, answer: str) -> str:
    """The follow-up turn, with the model's own answer put back in."""
    body = sg.prompt_text(case)
    follow = (f"You answered \"{answer}\". Name the ONE thing you can see "
              f"in the picture that most made you answer that way. "
              f"Give a short noun phrase and nothing else.")
    messages = [
        {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": body}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        {"role": "user", "content": [{"type": "text", "text": follow}]},
    ]
    return proc.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=tmpl)


def label_of(model: Any, inputs: dict, pos_id: int, neg_id: int,
             pos: str, neg: str) -> tuple:
    """The answer the model gives, and the margin behind it.

    The prompt carries the forced JSON prefix, so the next position
    emits the label itself. We read the 2 label logits there and need
    no lens.
    """
    import torch
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    z = out.logits[0, -1, :].float()
    c = float(z[pos_id] - z[neg_id])
    return (pos if c > 0 else neg), c


def run_case(
    case: str, *, proc: Any, model: Any, tmpl: str,
    n_images: int, n_random: int, seed: int, pad: int, max_new: int,
    maps: dict, out_dir: Path, device: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    rating = sg.rating(case)
    rows: list[dict] = []
    t0 = time.time()
    done = 0
    for r in sg.load_single_rows(n_images, case, seed=seed):
        key = str(r["image_id"])
        if key not in maps:
            continue
        values = maps[key]
        side = int(round(float(values.size) ** 0.5))
        img = Image.open(r["path"]).convert("RGB")

        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        answer, c0 = label_of(model, inputs, pos_id, neg_id,
                              rating["positive"], rating["negative"])

        text = ask_text(proc, tmpl, case, answer)
        rep_in = proc(text=[text], images=[[img]],
                      return_tensors="pt").to(device)
        report = cl.generate(model, proc, rep_in, max_new)

        cells, arms = pc.pick_cells(values, n_random, rng)
        for arm, cell in zip(arms, cells):
            piece, box = pc.crop_of(img, cell, side, pad, 896)
            cap_in = cl.caption_inputs(proc, tmpl, piece, device)
            caption = cl.generate(model, proc, cap_in, max_new)
            rr, cc = divmod(int(cell), side)
            rows.append({
                "case": case, "image_id": key, "arm": arm,
                "cell": int(cell), "row": rr, "col": cc,
                "answer": answer, "c0": c0,
                "report": report.strip(), "text": caption.strip()})
        done += 1
        log(f"  {done}/{n_images} {key} {answer} c0={c0:+.2f} "
            f"report={report.strip()[:48]!r} {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    d = out_dir / case
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "reports.parquet", index=False)
    log(f"wrote {d / 'reports.parquet'} with {len(df)} rows")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="restaurants")
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--pick", default="gxa")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    from research.monocle import attn_route as ar
    from research.monocle import transfer_k1 as tk

    cl.CAPTION_QUESTION = pc.CROP_QUESTION
    maps = tk.load_maps(Path(args.maps), args.pick, args.layer)
    log(f"{len(maps)} maps from {args.maps} at pick={args.pick}")
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    run_case(args.case, proc=proc, model=model,
             tmpl=tmpl, n_images=args.n_images, n_random=args.n_random,
             seed=args.seed, pad=args.pad, max_new=args.max_new, maps=maps,
             out_dir=Path(args.out_dir) / f"{args.case}_{args.pick}",
             device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
