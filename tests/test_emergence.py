"""CPU-only unit tests for monocle.emergence (per-patch emergence-depth maps).

No model, no GPU, no cyclomedia/DuckDB access. Synthetic logit tensors with a
tiny fake vocab exercise the agreement maths, first-crossing vs sustained
ignition, the never-ignited NaN path, the PIL renderer, the corpus summary, and
the eval-image sampler's exclusion logic (over a mocked tmp parquet).

Run:
    /share/pierson/matt/mllmsci/.venv/bin/python -m pytest tests/test_emergence.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import emergence  # noqa: E402


# ---------------------------------------------------------------------------
# Per-patch primitives: hand-computable cases
# ---------------------------------------------------------------------------
class TestPrimitives:
    def test_identical_logits_jaccard_one_js_zero(self):
        logits = torch.randn(8, 50)
        jac = emergence.topk_jaccard_per_patch(logits, logits, k=10)
        js = emergence.js_div_per_patch(logits, logits)
        assert torch.allclose(jac, torch.ones(8), atol=1e-6)
        assert torch.allclose(js, torch.zeros(8), atol=1e-6)

    def test_disjoint_topk_jaccard_zero(self):
        # a: top-k on ids 0..9; b: top-k on ids 40..49 -> no overlap.
        a = torch.full((4, 50), -10.0)
        b = torch.full((4, 50), -10.0)
        a[:, 0:10] = 5.0
        b[:, 40:50] = 5.0
        jac = emergence.topk_jaccard_per_patch(a, b, k=10)
        assert torch.allclose(jac, torch.zeros(4), atol=1e-6)

    def test_half_overlap_jaccard(self):
        # a top-10 = ids 0..9, b top-10 = ids 5..14 -> |∩|=5, |∪|=15 -> 1/3.
        a = torch.full((3, 50), -10.0)
        b = torch.full((3, 50), -10.0)
        a[:, 0:10] = 5.0
        b[:, 5:15] = 5.0
        jac = emergence.topk_jaccard_per_patch(a, b, k=10)
        assert torch.allclose(jac, torch.full((3,), 1.0 / 3.0), atol=1e-6)

    def test_js_bounded_by_ln2(self):
        a = torch.full((2, 50), -10.0)
        b = torch.full((2, 50), -10.0)
        a[:, 0] = 20.0
        b[:, 49] = 20.0
        js = emergence.js_div_per_patch(a, b)
        assert torch.all(js <= np.log(2) + 1e-3)
        assert torch.all(js > 0.6)  # near the ceiling for near-one-hot disjoint

    def test_mean_wrappers_match_compare(self):
        a, b = torch.randn(6, 50), torch.randn(6, 50)
        assert emergence.topk_jaccard(a, b, 10) == pytest.approx(
            float(emergence.topk_jaccard_per_patch(a, b, 10).mean()))
        assert emergence.js_div(a, b) == pytest.approx(
            float(emergence.js_div_per_patch(a, b).mean()))


# ---------------------------------------------------------------------------
# layer_agreement
# ---------------------------------------------------------------------------
def _make_per_layer(n_patches=8, vocab=50):
    """3 fitted layers {6,12,18} + final 47 over n_patches x vocab.

    Layer 6 disjoint from final; 12 half-overlap; 18 identical to final.
    """
    torch.manual_seed(0)
    final = torch.full((n_patches, vocab), -10.0)
    for i in range(n_patches):
        final[i, 20:30] = 5.0  # top-10 = ids 20..29
    l6 = torch.full((n_patches, vocab), -10.0)
    l6[:, 0:10] = 5.0
    l12 = torch.full((n_patches, vocab), -10.0)
    l12[:, 25:35] = 5.0  # overlaps final on ids 25..29 -> |∩|=5 -> 1/3
    l18 = final.clone()
    return {6: l6, 12: l12, 18: l18, 47: final}


class TestLayerAgreement:
    def test_columns_and_length(self):
        per = _make_per_layer()
        df = emergence.layer_agreement(per, final_layer=47, k=10)
        assert set(df.columns) == {"patch_idx", "layer", "jaccard", "js"}
        assert len(df) == 3 * 8  # 3 fitted layers x 8 patches
        assert sorted(df["layer"].unique()) == [6, 12, 18]

    def test_agreement_values(self):
        per = _make_per_layer()
        df = emergence.layer_agreement(per, final_layer=47, k=10)
        j = df.groupby("layer")["jaccard"].mean()
        assert j[6] == pytest.approx(0.0, abs=1e-6)
        assert j[12] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert j[18] == pytest.approx(1.0, abs=1e-6)
        js = df.groupby("layer")["js"].mean()
        assert js[18] == pytest.approx(0.0, abs=1e-6)
        assert js[6] > js[18]

    def test_missing_final_raises(self):
        per = _make_per_layer()
        del per[47]
        with pytest.raises(ValueError):
            emergence.layer_agreement(per, final_layer=47)

    def test_shape_mismatch_raises(self):
        per = _make_per_layer()
        per[6] = per[6][:, :40]  # wrong vocab width
        with pytest.raises(ValueError):
            emergence.layer_agreement(per, final_layer=47)


# ---------------------------------------------------------------------------
# ignition: first-crossing vs sustained, never-ignited NaN
# ---------------------------------------------------------------------------
class TestIgnition:
    def _flicker_df(self):
        """One patch (0). Jaccard by layer: L6=0.5(up) L12=0.1(down)
        L18=0.4(up) L24=0.6 L30=0.7 -> first crossing tau=0.3 at L6, but the
        L12 dip breaks the sustained suffix, so sustained starts at L18."""
        rows = [
            (0, 6, 0.5), (0, 12, 0.1), (0, 18, 0.4), (0, 24, 0.6), (0, 30, 0.7),
        ]
        return pd.DataFrame(rows, columns=["patch_idx", "layer", "jaccard"]).assign(js=0.0)

    def test_first_crossing_vs_sustained_differ(self):
        df = self._flicker_df()
        ign = emergence.ignition_layers(df, tau=0.3)
        sus = emergence.ignition_layers_sustained(df, tau=0.3)
        assert ign.loc[0] == 6.0
        assert sus.loc[0] == 18.0

    def test_never_ignited_is_nan(self):
        rows = [(0, 6, 0.05), (0, 12, 0.1), (0, 18, 0.2)]  # never >= 0.3
        df = pd.DataFrame(rows, columns=["patch_idx", "layer", "jaccard"]).assign(js=0.0)
        ign = emergence.ignition_layers(df, tau=0.3)
        sus = emergence.ignition_layers_sustained(df, tau=0.3)
        assert np.isnan(ign.loc[0])
        assert np.isnan(sus.loc[0])

    def test_sustained_nan_when_last_below_tau(self):
        # crosses early then falls below at the final fitted layer.
        rows = [(0, 6, 0.9), (0, 12, 0.9), (0, 18, 0.1)]
        df = pd.DataFrame(rows, columns=["patch_idx", "layer", "jaccard"]).assign(js=0.0)
        ign = emergence.ignition_layers(df, tau=0.3)
        sus = emergence.ignition_layers_sustained(df, tau=0.3)
        assert ign.loc[0] == 6.0
        assert np.isnan(sus.loc[0])

    def test_all_sustained_from_start(self):
        rows = [(0, 6, 0.9), (0, 12, 0.9), (0, 18, 0.9)]
        df = pd.DataFrame(rows, columns=["patch_idx", "layer", "jaccard"]).assign(js=0.0)
        assert emergence.ignition_layers_sustained(df, tau=0.3).loc[0] == 6.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
class TestSummarize:
    def test_corpus_stats(self):
        ign = pd.Series([6.0, 6.0, 12.0, np.nan, 36.0])
        s = emergence.summarize(ign, fitted_layers=[6, 12, 18, 36])
        assert s["n_patches"] == 5
        assert s["n_never_ignited"] == 1
        assert s["n_ignited"] == 4
        assert s["frac_never_ignited"] == pytest.approx(0.2)
        assert s["histogram"] == {6: 2, 12: 1, 18: 0, 36: 1}
        assert s["median_ignition_layer"] == pytest.approx(9.0)
        assert s["mean_ignition_layer"] == pytest.approx(15.0)

    def test_accepts_plain_iterable(self):
        s = emergence.summarize([6.0, np.nan])
        assert s["n_patches"] == 2 and s["n_never_ignited"] == 1


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------
class _Grid:
    def __init__(self, n_rows, n_cols):
        self.n_rows, self.n_cols = n_rows, n_cols

    @property
    def n_patches(self):
        return self.n_rows * self.n_cols


class TestRenderEmergence:
    def test_returns_correct_size(self):
        img = Image.new("RGB", (64, 48), "white")
        grid = _Grid(3, 4)  # 12 patches
        ign = pd.Series({i: [6, 12, 18, 36][i % 4] for i in range(12)})
        out = emergence.render_emergence(img, ign, grid, layers=[6, 12, 18, 36])
        assert out.size == (64, 48 + emergence.LEGEND_H)
        assert out.mode == "RGB"

    def test_handles_never_ignited(self):
        img = Image.new("RGB", (40, 40), "white")
        grid = _Grid(2, 2)
        ign = pd.Series([6.0, np.nan, np.nan, 36.0])  # some never ignite
        out = emergence.render_emergence(img, ign, grid, layers=[6, 36])
        assert out.size == (40, 40 + emergence.LEGEND_H)

    def test_pool_runs(self):
        img = Image.new("RGB", (64, 64), "white")
        grid = _Grid(4, 4)
        ign = pd.Series({i: 12 for i in range(16)})
        out = emergence.render_emergence(img, ign, grid, pool=2, layers=[12])
        assert out.size == (64, 64 + emergence.LEGEND_H)

    def test_palette_ordering(self):
        pal = emergence.palette_for_layers([6, 12, 18, 24, 30, 36, 42, 46])
        # early layer is bluer (more blue than red); late layer is redder.
        early, late = pal[6], pal[46]
        assert early[2] > early[0]   # blue dominates at L6
        assert late[0] > late[2]     # red dominates at L46


# ---------------------------------------------------------------------------
# sampler exclusion logic (mocked parquet, injected exists_fn)
# ---------------------------------------------------------------------------
class TestSampler:
    def _write_index(self, tmp_path):
        # 3 datasets x several recordings; recording ids 8 chars.
        rows = []
        for ds in ("bronx_2025_1k", "brooklyn_2025_1k", "queens_2025_1k"):
            for i in range(20):
                rows.append({"dataset": ds, "recording_id": f"{ds[0].upper()}0{i:05d}"})
        p = tmp_path / "idx.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False)
        return p

    def test_excludes_fit_images_and_fills_n(self, tmp_path):
        idx = self._write_index(tmp_path)
        # First pass: everything exists, nothing excluded -> get the paths.
        kept0, _ = emergence.sample_emergence_faces(
            idx, n=6, seed=778, exclude_paths=set(),
            raw_root="/fake/root", exists_fn=lambda p: True)
        assert len(kept0) == 6
        # Now exclude the first two of those paths; the sampler must skip them
        # and none of the excluded may appear.
        excluded = {kept0[0]["path"], kept0[1]["path"]}
        kept1, stats = emergence.sample_emergence_faces(
            idx, n=6, seed=778, exclude_paths=excluded,
            raw_root="/fake/root", exists_fn=lambda p: True)
        assert len(kept1) == 6
        assert stats["excluded_fit_images"] >= 2
        assert not (excluded & {k["path"] for k in kept1})

    def test_missing_files_are_skipped(self, tmp_path):
        idx = self._write_index(tmp_path)
        # Only paths containing "B0" (bronx-ish) "exist".
        kept, stats = emergence.sample_emergence_faces(
            idx, n=5, seed=778, exclude_paths=set(),
            raw_root="/fake/root", exists_fn=lambda p: "/B0" in p)
        assert all("/B0" in k["path"] for k in kept)
        assert stats["missing"] > 0

    def test_deterministic_under_seed(self, tmp_path):
        idx = self._write_index(tmp_path)
        a, _ = emergence.sample_emergence_faces(
            idx, n=6, seed=778, exclude_paths=set(),
            raw_root="/fake/root", exists_fn=lambda p: True)
        b, _ = emergence.sample_emergence_faces(
            idx, n=6, seed=778, exclude_paths=set(),
            raw_root="/fake/root", exists_fn=lambda p: True)
        assert [x["path"] for x in a] == [x["path"] for x in b]

    def test_load_exclude_paths(self, tmp_path):
        import json
        p = tmp_path / "fit_images.json"
        p.write_text(json.dumps([
            {"path": "/a.jpg", "recording_id": "x"}, {"path": "/b.jpg"}]))
        assert emergence.load_exclude_paths(p) == {"/a.jpg", "/b.jpg"}
        assert emergence.load_exclude_paths(tmp_path / "missing.json") == set()


# ---------------------------------------------------------------------------
# CLI parser (no execution, no jlens/model)
# ---------------------------------------------------------------------------
class TestCli:
    def test_defaults(self):
        args = emergence.build_parser().parse_args([])
        assert args.sample == emergence.DEFAULT_SAMPLE
        assert args.seed == emergence.DEFAULT_SEED
        assert args.k == emergence.DEFAULT_K
        assert args.tau == emergence.DEFAULT_TAU
        assert args.smoke is False
        assert args.jlens == emergence.DEFAULT_JLENS

    def test_smoke_and_images(self):
        args = emergence.build_parser().parse_args(
            ["--images", "a.jpg", "b.jpg", "--smoke", "--k", "5", "--tau", "0.4"])
        assert args.images == ["a.jpg", "b.jpg"]
        assert args.smoke is True
        assert args.k == 5 and args.tau == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
