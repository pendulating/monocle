"""Open the output of `hop_depth.py`. The arrays stay on disk until a cell reads them.

    from hop_store import open_store
    s = open_store("qwen3.5-9b")
    s.read("final", 20)          # [images, vocab]
    s.gap("final", 20, s.images["borough"] == "Manhattan")   # a Series of mean differences, by word
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "data/f10/hop_depth"
ALL_ROOT = ROOT.parent / "hop_all"  # F10.6: every token. `topk()` does not apply: use `top()`


@dataclass
class Store:
    path: Path

    @cached_property
    def manifest(self) -> dict:
        return json.loads((self.path / "manifest.json").read_text())

    @cached_property
    def vocab(self) -> pd.DataFrame:
        return pd.read_parquet(self.path / "vocab.parquet")

    @cached_property
    def answers(self) -> pd.DataFrame:
        return pd.concat(pd.read_parquet(p) for p in sorted((self.path / "answers").glob("answers-*.parquet"))).reset_index(drop=True)

    @cached_property
    def images(self) -> pd.DataFrame:
        """Give 1 row for each image, in the order of the arrays."""
        got = pd.read_parquet(self.path.parent / "images.parquet").merge(self.answers, on="image_idx", validate="1:1", sort=False)
        return got

    def _axis(self, position: str, layer: int) -> tuple[int, int]:
        return self.manifest["positions"].index(position), self.manifest["layers"].index(layer)

    def _take(self, folder: str, position: str, layer: int) -> np.ndarray:
        a, b = self._axis(position, layer)
        files = sorted((self.path / folder).glob("*.npy"))
        return np.concatenate([np.load(f, mmap_mode="r")[:, a, b] for f in files])

    def read(self, position: str, layer: int) -> np.ndarray:
        """Give the lens logits [images, vocab]."""
        return self._take("reads", position, layer)

    def residual(self, position: str, layer: int) -> np.ndarray:
        """Give the residuals [images, d]."""
        return self._take("residuals", position, layer)

    def text_read(self, position: str, layer: int) -> np.ndarray:
        """Give the read of the same text with no image [vocab]. `after_image` is NaN."""
        a, b = self._axis(position, layer)
        return np.load(self.path / "text_reads.npy", mmap_mode="r")[a, b]

    def gap(self, position: str, layer: int, group: pd.Series) -> pd.Series:
        """Give the mean read of a group of images minus the mean of the others, by word."""
        x, mask = self.read(position, layer), np.asarray(group, bool)
        return pd.Series(x[mask].mean(0) - x[~mask].mean(0), index=self.vocab["token"].str.strip()).sort_values()

    def top(self, position: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Give the top token ids and logits [images, 30] of a tail token. F10.6 only."""
        a, b = self._axis(position, layer)
        take = lambda name: np.concatenate([np.load(f, mmap_mode="r")[:, a, b] for f in sorted((self.path / "topk").glob(f"{name}-*.npy"))])
        return take("ids"), take("logits")

    def patch_top(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Give the top token ids and logits [images, patches, 30]. F10.6 only."""
        b = self.manifest["layers"].index(layer)
        take = lambda name: np.concatenate([np.load(f, mmap_mode="r")[:, :, b] for f in sorted((self.path / "patches").glob(f"{name}-*.npy"))])
        return take("ids"), take("logits")

    def topk(self) -> pd.DataFrame:
        return pd.concat(pd.read_parquet(p) for p in sorted((self.path / "topk").glob("topk-*.parquet")))


def open_store(model: str, root: Path = ROOT) -> Store:
    return Store(Path(root) / model)
