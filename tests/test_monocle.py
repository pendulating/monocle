"""CPU-only unit tests for the monocle logit-lens package.

No model load, no GPU, no cyclomedia/DuckDB access. Everything runs on tiny
synthetic tensors and fake tokenizers to exercise the geometry, scoring, and
token-filtering logic plus the CLI argument parser.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import extract, scoring  # noqa: E402


# ---------------------------------------------------------------------------
# Fake tokenizers (no real model / vocab)
# ---------------------------------------------------------------------------
class FakeTokenizer:
    """Minimal stand-in: convert_ids_to_tokens over a fixed vocab list.

    Handles both a single int (score_patches) and a list of ints
    (build_token_mask). ``all_special_ids`` is exposed for build_token_mask.
    """

    def __init__(self, vocab: list[str], special_ids: list[int] | None = None):
        self.vocab = vocab
        self.all_special_ids = list(special_ids or [])

    def __len__(self) -> int:
        return len(self.vocab)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, (list, tuple)):
            return [self.vocab[i] for i in ids]
        return self.vocab[ids]


# ---------------------------------------------------------------------------
# infer_grid
# ---------------------------------------------------------------------------
class TestInferGrid:
    def test_pixel_values_strategy(self):
        # 624/48 = 13 rows, 480/48 = 10 cols -> 130 patches.
        inputs = {"pixel_values": torch.zeros(1, 3, 624, 480)}
        grid = extract.infer_grid(inputs, n_patches=130, orig_size=(480, 624))
        assert (grid.n_rows, grid.n_cols) == (13, 10)
        assert grid.strategy == "pixel_values"
        assert grid.n_patches == 130
        assert (grid.resized_w, grid.resized_h) == (480, 624)

    def test_square_strategy(self):
        # 3-D tensor is not usable pixel_values; 256 is a perfect square.
        inputs = {"pixel_values": torch.zeros(3, 224, 224)}
        grid = extract.infer_grid(inputs, n_patches=256, orig_size=(100, 100))
        assert (grid.n_rows, grid.n_cols) == (16, 16)
        assert grid.strategy == "square"

    def test_aspect_factor_strategy(self):
        # 12 patches, aspect 400/300 -> the factor pair closest is 3x4.
        inputs: dict = {}
        grid = extract.infer_grid(inputs, n_patches=12, orig_size=(400, 300))
        assert (grid.n_rows, grid.n_cols) == (3, 4)
        assert grid.strategy == "aspect_factor"


# ---------------------------------------------------------------------------
# image_token_positions
# ---------------------------------------------------------------------------
class TestImageTokenPositions:
    def test_contiguous_block(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 99, 99, 99, 3]])}
        pos = extract.image_token_positions(inputs, image_token_id=99)
        assert pos.tolist() == [2, 3, 4]

    def test_non_contiguous_raises(self):
        inputs = {"input_ids": torch.tensor([[99, 2, 99]])}
        with pytest.raises(RuntimeError):
            extract.image_token_positions(inputs, image_token_id=99)

    def test_zero_tokens_raises(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        with pytest.raises(RuntimeError):
            extract.image_token_positions(inputs, image_token_id=99)


# ---------------------------------------------------------------------------
# display_form
# ---------------------------------------------------------------------------
class TestDisplayForm:
    def test_sentencepiece_underscore_stripped(self):
        assert scoring.display_form("▁dog") == "dog"

    def test_plain_token_unchanged(self):
        assert scoring.display_form("dog") == "dog"


# ---------------------------------------------------------------------------
# score_patches
# ---------------------------------------------------------------------------
class TestScorePatches:
    # vocab index -> token (all alphabetic display forms, len >= 2)
    VOCAB = [
        "▁common",  # 0 : boosted in every patch -> high p_global
        "▁Dog",     # 1 : patch-0 specific (dedupe partner of index 2)
        "dog",           # 2 : patch-0 specific (same display 'dog')
        "▁taxi",    # 3 : patch-1 specific
        "▁tree",    # 4 : patch-2 specific
        "▁road",    # 5 : patch-3 specific
        "▁sign",    # 6
        "▁car",     # 7
        "▁door",    # 8
        "▁roof",    # 9
        "▁wall",    # 10
        "▁sky",     # 11
    ]

    def _logits(self) -> torch.Tensor:
        # 4 patches x 12 vocab
        logits = torch.zeros(4, 12)
        logits[:, 0] = 5.0            # 'common' boosted everywhere
        logits[0, 1] = 8.0            # 'Dog' only in patch 0
        logits[0, 2] = 8.0            # 'dog' only in patch 0
        logits[1, 3] = 8.0            # 'taxi' only in patch 1
        logits[2, 4] = 8.0            # 'tree' only in patch 2
        logits[3, 5] = 8.0            # 'road' only in patch 3
        return logits

    def test_patch_specific_outranks_common(self):
        tok = FakeTokenizer(self.VOCAB)
        df = scoring.score_patches(self._logits(), tok, k=3, alpha=scoring.DEFAULT_ALPHA)

        p0 = df[df["patch_idx"] == 0]
        # top-1 of patch 0 is the patch-specific 'dog', not the global 'common'
        top1 = p0[p0["rank"] == 0]["token"].iloc[0]
        assert top1.lower() == "dog"

        # 'common' is present but ranked below the patch-specific token
        dog_rank = int(p0[p0["token"].str.lower() == "dog"]["rank"].min())
        common_rows = p0[p0["token"].str.lower() == "common"]
        assert not common_rows.empty
        assert dog_rank < int(common_rows["rank"].min())

    def test_dedupe_case_and_piece_variants(self):
        tok = FakeTokenizer(self.VOCAB)
        df = scoring.score_patches(self._logits(), tok, k=3, alpha=scoring.DEFAULT_ALPHA)
        p0 = df[df["patch_idx"] == 0]
        # '▁Dog' and 'dog' collapse to a single survivor in patch 0
        dog_variants = p0[p0["token"].str.lower() == "dog"]
        assert len(dog_variants) == 1

    def test_rank_and_column_structure(self):
        tok = FakeTokenizer(self.VOCAB)
        df = scoring.score_patches(self._logits(), tok, k=3, alpha=scoring.DEFAULT_ALPHA)
        expected_cols = {"patch_idx", "rank", "token", "token_id", "score",
                         "p_patch", "p_global"}
        assert expected_cols.issubset(set(df.columns))
        for patch_idx in range(4):
            ranks = df[df["patch_idx"] == patch_idx]["rank"].tolist()
            # contiguous 0..k-1, in order
            assert ranks == list(range(len(ranks)))
            assert len(ranks) <= 3


# ---------------------------------------------------------------------------
# build_token_mask
# ---------------------------------------------------------------------------
class TestBuildTokenMask:
    def test_filtering_rules(self):
        vocab = [
            "<pad>",       # 0 : special -> dropped
            "<0xE2>",      # 1 : <...> byte-fallback / control -> dropped
            "a",           # 2 : 1-char display -> dropped
            "dog",         # 3 : kept
            "▁taxi",  # 4 : kept ('taxi')
            "<eos>",       # 5 : special -> dropped
            "123",         # 6 : no alphabetic char -> dropped
            "▁a",     # 7 : display 'a', 1-char -> dropped
        ]
        tok = FakeTokenizer(vocab, special_ids=[0, 5])
        mask = scoring.build_token_mask(tok, vocab_size=len(vocab))
        assert mask.dtype == torch.bool
        assert mask[3].item() is True
        assert mask[4].item() is True
        for i in (0, 1, 2, 5, 6, 7):
            assert mask[i].item() is False


# ---------------------------------------------------------------------------
# attach_grid
# ---------------------------------------------------------------------------
class TestAttachGrid:
    def test_row_major(self):
        n_rows, n_cols = 2, 3
        df = pd.DataFrame({"patch_idx": list(range(n_rows * n_cols))})
        out = scoring.attach_grid(df, n_rows, n_cols)
        assert out["patch_row"].tolist() == [0, 0, 0, 1, 1, 1]
        assert out["patch_col"].tolist() == [0, 1, 2, 0, 1, 2]
        # spot-check: patch 4 -> row 1, col 1
        row4 = out[out["patch_idx"] == 4].iloc[0]
        assert (int(row4["patch_row"]), int(row4["patch_col"])) == (1, 1)


# ---------------------------------------------------------------------------
# CLI argument parsing (no execution)
# ---------------------------------------------------------------------------
class TestCliArgparse:
    def test_image_source_and_defaults(self):
        from monocle import cli

        args = cli.build_parser().parse_args(["--image", "x.jpg"])
        # action="append" + nargs="+" -> list of lists
        assert args.image == [["x.jpg"]]
        assert args.k == scoring.DEFAULT_K
        assert args.alpha == scoring.DEFAULT_ALPHA
        assert args.system is None
        assert args.svg is False
        assert args.no_render is False
        assert args.faces == cli.DEFAULT_FACES

    def test_recording_source_and_faces(self):
        from monocle import cli

        args = cli.build_parser().parse_args(
            ["--recording-id", "W0D0M3OU", "--dataset", "brooklyn_2025_1k",
             "--faces", "F", "B", "--k", "5", "--alpha", "0.5", "--svg"])
        assert args.recording_id == "W0D0M3OU"
        assert args.dataset == "brooklyn_2025_1k"
        assert args.faces == ["F", "B"]
        assert args.k == 5
        assert args.alpha == 0.5
        assert args.svg is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPooling:
    """pool_probs / pooled_dims / score_patches(pool=...)."""

    def test_pooled_dims(self):
        assert scoring.pooled_dims(16, 16, 2) == (8, 8)
        assert scoring.pooled_dims(16, 16, 3) == (6, 6)
        assert scoring.pooled_dims(13, 10, 4) == (4, 3)

    def test_pool_probs_uniform_block(self):
        logits = torch.zeros(16, 6)
        for idx in (0, 1, 4, 5):
            logits[idx, 2] = 4.0
        p = torch.softmax(logits, dim=-1)
        pooled = scoring.pool_probs(p, 4, 4, 2)
        assert pooled.shape == (4, 6)
        assert torch.allclose(pooled[0], p[0])
        assert pooled[0, 2] > pooled[1, 2]

    def test_pool_probs_uneven_edge(self):
        p = torch.softmax(torch.randn(16, 6), dim=-1)
        pooled = scoring.pool_probs(p, 4, 4, 3)
        assert pooled.shape == (4, 6)
        # 1x1 corner block is exactly the last patch
        assert torch.allclose(pooled[3], p[15])

    def test_score_patches_pool_requires_grid_shape(self):
        logits = torch.zeros(16, 6)
        with pytest.raises(ValueError):
            scoring.score_patches(logits, FakeTokenizer(["t%d" % i for i in range(6)]), k=1, pool=2)


# ---------------------------------------------------------------------------
# jlens_read.lens_patch_logits — the l+1 layer convention (CPU, no jlens/model)
# ---------------------------------------------------------------------------
class FakeLens:
    """Stand-in for jlens.JacobianLens: one fitted source layer (0) whose
    Jacobian is 2*I, transport(h, l) = h @ J_l.T (matches the real API)."""

    def __init__(self):
        self.source_layers = [0]
        self.jacobians = {0: torch.eye(4) * 2.0}

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        return residual @ self.jacobians[layer].T


class FakeLensModel:
    """Stand-in for jlens's HFLensModel: fixed depth + a linear unembed."""

    def __init__(self, w_u: torch.Tensor):
        self.n_layers = 3  # final layer index = n_layers - 1 = 2
        self.w_u = w_u

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return residual @ self.w_u.T


class TestLensPatchLogits:
    """Pin monocle.jlens_read.lens_patch_logits: hook-recorded activations
    keyed BY layer (no hidden_states tuple indexing — that convention failed
    stage-A validation), final layer = final_logits verbatim, fitted-layer
    read with transport, and the unfitted / unrecorded guards. No jlens
    import, no model."""

    def _setup(self):
        import monocle.jlens_read as jr

        torch.manual_seed(0)
        seq, d = 3, 4
        # hook-style activations: {layer: [1, seq, d] block output}
        activations = {
            0: torch.full((1, seq, d), 1.0) + torch.arange(d).float(),
        }
        positions = torch.tensor([0, 2])
        w_u = torch.randn(5, d)  # vocab=5
        final_logits = torch.randn(2, 5)
        return (jr, activations, positions, final_logits,
                FakeLens(), FakeLensModel(w_u), w_u)

    def test_final_layer_is_final_logits_verbatim(self):
        jr, acts, pos, fin, lens, lm, _w_u = self._setup()
        out = jr.lens_patch_logits(lens, lm, acts, pos, fin)  # None -> [0, 2]
        assert set(out.keys()) == {0, 2}
        assert out[2] is fin  # the model's own head output, no transport

    def test_fitted_layer_with_transport(self):
        jr, acts, pos, fin, lens, lm, w_u = self._setup()
        out = jr.lens_patch_logits(lens, lm, acts, pos, fin)
        h0 = acts[0][0, pos, :].float()
        expected = (h0 @ lens.jacobians[0].T) @ w_u.T
        assert torch.allclose(out[0], expected, atol=1e-5)
        # transport actually did something (J = 2I doubles the residual)
        assert not torch.allclose(out[0], h0 @ w_u.T, atol=1e-5)

    def test_unfitted_non_final_raises(self):
        jr, acts, pos, fin, lens, lm, _w_u = self._setup()
        # layer 1 is neither fitted (jacobians has only 0) nor the final layer
        with pytest.raises(ValueError):
            jr.lens_patch_logits(lens, lm, acts, pos, fin, layers=[1])

    def test_unrecorded_fitted_layer_raises(self):
        jr, _acts, pos, fin, lens, lm, _w_u = self._setup()
        with pytest.raises(ValueError):
            jr.lens_patch_logits(lens, lm, {}, pos, fin, layers=[0])


# ---------------------------------------------------------------------------
# CLI layer-mode arguments (no execution, no jlens)
# ---------------------------------------------------------------------------
class TestCliLayerArgs:
    def test_jlens_flags_parse(self):
        from monocle import cli

        args = cli.build_parser().parse_args(
            ["--image", "x.jpg", "--jlens", "x.pt", "--layers", "6,12",
             "--gif", "--scrubber"])
        assert args.jlens == "x.pt"
        assert args.layers == "6,12"
        assert args.gif is True
        assert args.scrubber is True
        # the --layers string parses to a list of ints via the CLI helper
        assert cli.parse_layers(args.layers) == [6, 12]

    def test_parse_layers_default_and_empty(self):
        from monocle import cli

        assert cli.parse_layers(None) is None
        assert cli.parse_layers("") is None
        assert cli.parse_layers(" 6 , 12 ,24 ") == [6, 12, 24]

    def test_gif_without_jlens_errors(self):
        from monocle import cli

        args = cli.build_parser().parse_args(["--image", "x.jpg", "--gif"])
        with pytest.raises(ValueError):
            cli.validate_jlens_args(args)

    def test_scrubber_without_jlens_errors(self):
        from monocle import cli

        args = cli.build_parser().parse_args(["--image", "x.jpg", "--scrubber"])
        with pytest.raises(ValueError):
            cli.validate_jlens_args(args)

    def test_jlens_without_render_flags_is_valid(self):
        from monocle import cli

        args = cli.build_parser().parse_args(["--image", "x.jpg", "--jlens", "x.pt"])
        # no --gif/--scrubber: nothing to validate, must not raise
        cli.validate_jlens_args(args)
