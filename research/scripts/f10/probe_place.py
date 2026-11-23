#!/usr/bin/env python
"""F10.18, part 2, for 3 models: how often is a place probe correct, by borough? CPU only.

The input is the probe run of `probe_query.py` (`data/f10/hop_probe/{model}/probes.jsonl`). All 591
images are in New York City, thus the `in New York` row is an accuracy measure. The script also
checks each probe run against the answer store of the same prompt: the `cool` probe against
`data/f10/hop_logit/{model}/answers.parquet`, and the `urban` probe against the F10.25 `urban` store.

Output: `research/figures/f10/probe/probe_place.csv`, `research/figures/f10/probe/probe_check.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/probe_place.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10")]
from research.scripts.f10.analyze_probe import margins  # noqa: E402
from research.scripts.f10.hop_borough import BOROUGHS  # noqa: E402

__version__ = "1.0.0"
MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
PROBES = ("in New York", "in a city", "urban", "outdoors")
CHECKS = {"cool": "{model}", "urban": "{model}-urban"}
OUT = ROOT / "research/figures/f10/probe"


def place_rows(model: str, p: pd.DataFrame, images: pd.DataFrame) -> list[dict]:
    """Give the share of images with a positive margin on each probe, over all images and by borough."""
    b = images.set_index("image_idx")["borough"].reindex(p.index)
    rows = []
    for w in PROBES:
        yes = p[w] > 0
        rows.append({"model": model, "word": w, "n": int(yes.notna().sum()), "all": float(yes.mean())} | {k: float(v) for k, v in yes.groupby(b).mean().items()})
    return rows


def check_rows(model: str, p: pd.DataFrame) -> list[dict]:
    """Give r between a probe margin and the margin of the answer store that asked the same question."""
    rows = []
    for word, store in CHECKS.items():
        path = ROOT / "data/f10/hop_logit" / store.format(model=model) / "answers.parquet"
        if not path.exists() or word not in p.columns:
            continue
        a = pd.read_parquet(path).set_index("image_idx")["margin"].reindex(p.index)
        ok = a.notna() & p[word].notna()
        rows.append({"model": model, "word": word, "store": store.format(model=model), "n": int(ok.sum()), "r": float(np.corrcoef(a[ok], p.loc[ok, word])[0, 1]),
                     "yes_probe": float((p.loc[ok, word] > 0).mean()), "yes_store": float((a[ok] > 0).mean()), "same_sign": float(((a[ok] > 0) == (p.loc[ok, word] > 0)).mean())})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = pd.read_parquet(ROOT / "data/f10/hop_all/images.parquet")
    place, checks = [], []
    for m in MODELS:
        if not (ROOT / "data/f10/hop_probe" / m / "probes.jsonl").exists():
            print(f"[f10.18] {m}: no probe run", flush=True)
            continue
        p = margins(m)
        place += place_rows(m, p, images)
        checks += check_rows(m, p)
    pd.DataFrame(place).to_csv(OUT / "probe_place.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "probe_check.csv", index=False)
    print(pd.DataFrame(place).round(3).to_string(index=False))
    print(pd.DataFrame(checks).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
