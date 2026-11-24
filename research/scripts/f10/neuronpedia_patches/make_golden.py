#!/usr/bin/env python
"""Write the test fixture of the viewer instruments (F10.35): a small run and the output of the Python core.

The run comes from the export functions of `hop_viewer.py` (`run`, `rank_block`), and the
expected lists come from `research.monocle.instruments`. The TypeScript test
`jlens-instruments.test.ts` of the patch reads the file and compares.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/neuronpedia_patches/make_golden.py OUT.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from research.monocle.instruments import layer_loss, position_loss, ranks, salience_count, smear
from research.scripts.f10.hop_viewer import CONTRAST, RAW, rank_block, run, sequence

V, LAYERS, SEED = 120, [0, 4, 8, 12, 16, 20], 20261123
MANIFEST = {"prefix_tokens": ["<s>", "user"], "prefix_ids": [1, 2], "tail_tokens": ["</v>", "Is", " this", " place", " cool", "?"],
            "tail_ids": [3, 4, 5, 6, 7, 8], "positions": ["after_image", "first_text", "this", "place", "after_place", "final"]}


def arrays() -> tuple[np.ndarray, np.ndarray]:
    """Give the reads with smooth change across layers and tokens, thus losses, gains and smears occur."""
    rng = np.random.default_rng(SEED)
    n_read = len(MANIFEST["prefix_tokens"]) + len(MANIFEST["positions"])
    base = rng.normal(size=(n_read, 1, V))
    drift = rng.normal(scale=0.6, size=(n_read, len(LAYERS), V)).cumsum(1)
    raw = (base + drift).astype(np.float32)
    contrast = (raw - raw.mean(0, keepdims=True) + rng.normal(scale=0.3, size=raw.shape)).astype(np.float32)
    contrast[: len(MANIFEST["prefix_tokens"])] = np.nan
    return raw, contrast


def main() -> None:
    tokens, at = sequence(MANIFEST)
    positions = list(at)
    words = np.array([f" w{i:03d}" for i in range(V)])
    raw, contrast = arrays()
    full = run("test-model", tokens, at, positions, LAYERS, raw, contrast, words)
    full["instruments"] = rank_block(at, positions, LAYERS, raw, contrast, words)
    view = {i: at[p] for i, p in enumerate(positions)}  # read index -> viewer position
    cases = []
    for col, x in ((RAW, raw), (CONTRAST, contrast)):
        r = ranks(x)
        for lo, hi in ((0, 5), (1, 4)):
            rng_layers = [LAYERS[lo], LAYERS[hi]]
            sub = r[:, lo:hi + 1]
            for p in (0, 3, 5, 7):
                one = layer_loss(r, words, lo, hi, p=p)
                cases.append({"type": col, "range": rng_layers, "selected": [view[p]],
                              "state": {"mode": "layerLoss", "variant": "one"}, "ordered": True,
                              "rows": [[w, f"{s} → {e}", int(e)] for w, s, e in one.itertuples(index=False)]})
                sm = smear(sub, words, p, m=2, layers=LAYERS[lo:hi + 1])
                cases.append({"type": col, "range": rng_layers, "selected": [view[p]], "state": {"mode": "smear", "m": 2},
                              "ordered": False, "rows": [[w, f"{n}: L{a}–{b}"] for w, n, a, b in sm.itertuples(index=False)]})
            sc = salience_count(r, words, lo, over="tokens")
            cases.append({"type": col, "range": rng_layers, "selected": [], "ordered": False,
                          "state": {"mode": "layerLoss", "variant": "all", "measure": "salience"},
                          "rows": [[w, f"{a} / {b}"] for w, a, b in sc.itertuples(index=False)]})
            lg = layer_loss(r, words, lo, hi)
            cases.append({"type": col, "range": rng_layers, "selected": [], "ordered": False,
                          "state": {"mode": "layerLoss", "variant": "all", "measure": "lossGain"},
                          "rows": [[w, f"{a} / {b}"] for w, a, b in lg.itertuples(index=False)]})
            for A, B in ((3, 7), (2, 6)):
                one = position_loss(r, words, A, B, layer=hi)
                cases.append({"type": col, "range": rng_layers, "selected": [view[A], view[B]], "ordered": True,
                              "state": {"mode": "positionLoss", "variant": "one", "layer": LAYERS[hi]},
                              "rows": [[w, f"{s} → {e}", int(e)] for w, s, e in one.itertuples(index=False)]})
                rev = position_loss(r, words, B, A, layer=lo)
                cases.append({"type": col, "range": rng_layers, "selected": [view[A], view[B]], "ordered": True,
                              "state": {"mode": "positionLoss", "variant": "one", "layer": LAYERS[lo], "reverse": True},
                              "rows": [[w, f"{s} → {e}", int(e)] for w, s, e in rev.itertuples(index=False)]})
                sa = salience_count(sub, words, A, over="layers")
                cases.append({"type": col, "range": rng_layers, "selected": [view[A]], "ordered": False,
                              "state": {"mode": "positionLoss", "variant": "all", "measure": "salience"},
                              "rows": [[w, f"{a} / {b}"] for w, a, b in sa.itertuples(index=False)]})
                pg = position_loss(sub, words, A, B)
                cases.append({"type": col, "range": rng_layers, "selected": [view[A], view[B]], "ordered": False,
                              "state": {"mode": "positionLoss", "variant": "all", "measure": "lossGain"},
                              "rows": [[w, f"{a} / {b}"] for w, a, b in pg.itertuples(index=False)]})
    json.dump({"note": "Made by monocle research/scripts/f10/neuronpedia_patches/make_golden.py. Do not edit.",
               "run": full, "cases": cases}, open(sys.argv[1], "w"), ensure_ascii=False)
    print(f"{len(cases)} cases, {sum(len(c['rows']) for c in cases)} rows")


if __name__ == "__main__":
    main()
