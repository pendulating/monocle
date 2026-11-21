"""Tests for the k=1 transfer battery."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import transfer_k1 as tk  # noqa: E402


def write_maps(tmp_path: Path, with_layers: bool) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    layers = [12, 18, 24] if with_layers else [None]
    for img in ("a", "b"):
        causal = rng.normal(size=8)
        for lay in layers:
            rows.append({"target": img, "layer": lay, "causal": causal,
                         "gxa": rng.normal(size=8)})
    p = tmp_path / "maps.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_load_truth_ignores_layers(tmp_path):
    p = write_maps(tmp_path, with_layers=True)
    maps = tk.load_maps(p, "causal", None)
    assert set(maps) == {"a", "b"}
    assert maps["a"].shape == (8,)


def test_load_instrument_at_a_layer(tmp_path):
    p = write_maps(tmp_path, with_layers=True)
    df = pd.read_parquet(p)
    want = np.asarray(df[(df.target == "a") & (df.layer == 18)].iloc[0]["gxa"])
    got = tk.load_maps(p, "gxa", 18)["a"]
    assert np.allclose(got, want)


def test_pooled_ranks_are_ranks(tmp_path):
    p = write_maps(tmp_path, with_layers=True)
    got = tk.load_maps(p, "gxa", None)["a"]
    assert got.min() >= 1.0 and got.max() <= 8.0
    assert abs(got.mean() - 4.5) < 1e-9


def test_missing_column_stops(tmp_path):
    p = write_maps(tmp_path, with_layers=True)
    with pytest.raises(SystemExit):
        tk.load_maps(p, "cheap", 18)


def test_missing_layer_stops(tmp_path):
    p = write_maps(tmp_path, with_layers=True)
    with pytest.raises(SystemExit):
        tk.load_maps(p, "gxa", 99)


def test_pixel_gap_is_zero_for_the_same_pixels():
    px = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    assert tk.pixel_gap(px, px, 2) == 0.0


def test_pixel_gap_grows_with_the_difference():
    a = torch.zeros(4, 6)
    b = torch.ones(4, 6)
    near = tk.pixel_gap(a, b, 0)
    far = tk.pixel_gap(a, b * 3.0, 0)
    assert far > near > 0


def frame() -> pd.DataFrame:
    rows = []
    for t, s, dp, dn in (("t1", "s1", 0.5, -0.4), ("t2", "s1", 0.2, 0.1)):
        rows.append({"target_id": t, "source_id": s, "arm": "pos", "dc": dp,
                     "pixel_gap": 1.0, "c_target": 0.0, "c_source": 1.0})
        rows.append({"target_id": t, "source_id": s, "arm": "neg", "dc": dn,
                     "pixel_gap": 1.0, "c_target": 0.0, "c_source": 1.0})
        for i, v in enumerate((0.05, -0.05, 0.1)):
            rows.append({"target_id": t, "source_id": s, "arm": "random",
                         "dc": v, "pixel_gap": 1.0, "c_target": 0.0,
                         "c_source": 1.0})
    return pd.DataFrame(rows)


def test_summarize_pairs_and_signs():
    s = tk.summarize(frame())
    assert len(s) == 2
    r = s[s.target_id == "t1"].iloc[0]
    assert r.separation == pytest.approx(0.9)
    assert r.separation_z > 0


def test_summarize_skips_a_pair_with_no_pole():
    df = frame()
    df = df[~((df.target_id == "t2") & (df.arm == "pos"))]
    s = tk.summarize(df)
    assert set(s.target_id) == {"t1"}


def test_report_writes_a_verdict(tmp_path):
    s = tk.summarize(frame())
    tk.report(s, "restaurants", "causal", tmp_path)
    import json
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["n_pairs"] == 2
    assert v["mean_separation"] == pytest.approx((0.9 + 0.1) / 2)


def test_poles_are_the_extremes_of_the_map():
    v = np.array([0.1, -3.0, 2.0, 0.5])
    order = np.argsort(v)
    assert int(order[-1]) == 2
    assert int(order[0]) == 1


def test_random_cells_never_hold_a_pole():
    rng = np.random.default_rng(1)
    v = rng.normal(size=32)
    order = np.argsort(v)
    pos_cell, neg_cell = int(order[-1]), int(order[0])
    rest = [c for c in range(v.size) if c not in (pos_cell, neg_cell)]
    draw = rng.choice(rest, size=8, replace=False)
    assert pos_cell not in draw and neg_cell not in draw
    assert len(set(draw.tolist())) == 8
