"""CPU tests for the open-vocabulary answer-position readout.

Covers the pure pieces: top-k extraction, the concentration measures, shard
arithmetic, and the corpus tally. Every torch / transformers import that needs
a GPU stays inside `main`, so this file imports on a login node.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import answer_tokens as at  # noqa: E402


class FakeTokenizer:
    """Ids 0..9; ids 0-1 look like control tokens, the rest like words."""
    TOKENS = ["<pad>", "<eos>", "▁Much", "▁Less", "▁Same", "▁More",
              "▁safe", "▁dark", '▁"', "▁the"]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.TOKENS[ids]
        return [self.TOKENS[i] for i in ids]


def logits_favouring(idx: int, n: int = 10, peak: float = 8.0):
    v = torch.zeros(n)
    v[idx] = peak
    return v


# ---------------------------------------------------------------------------
# top-k
# ---------------------------------------------------------------------------
class TestTopK:
    def test_returns_k_rows_ranked(self):
        rows = at.topk_tokens(torch.arange(10).float(), FakeTokenizer(), k=3)
        assert [r["rank"] for r in rows] == [0, 1, 2]
        assert [r["token_id"] for r in rows] == [9, 8, 7]
        assert rows[0]["prob"] > rows[1]["prob"] > rows[2]["prob"]

    def test_display_strips_the_sentencepiece_marker(self):
        rows = at.topk_tokens(logits_favouring(6), FakeTokenizer(), k=1)
        assert rows[0]["token"] == "▁safe"
        assert rows[0]["display"] == "safe"

    def test_probabilities_are_over_the_whole_vocabulary(self):
        """Not renormalized over the top-k — layers must stay comparable."""
        rows = at.topk_tokens(torch.zeros(10), FakeTokenizer(), k=3)
        for r in rows:
            assert r["prob"] == pytest.approx(0.1)
        assert sum(r["prob"] for r in rows) == pytest.approx(0.3)

    def test_is_word_annotates_but_never_filters(self):
        """Raw top-k must keep control tokens; the mask only labels them."""
        mask = torch.tensor([False, False] + [True] * 8)
        # id 0 (<pad>) is the argmax and must still appear at rank 0.
        rows = at.topk_tokens(logits_favouring(0), FakeTokenizer(), k=3,
                              word_mask=mask)
        assert rows[0]["token_id"] == 0
        assert rows[0]["is_word"] is False
        assert all(r["is_word"] is True for r in rows[1:])

    def test_is_word_is_none_without_a_mask(self):
        rows = at.topk_tokens(torch.zeros(10), FakeTokenizer(), k=2)
        assert all(r["is_word"] is None for r in rows)

    def test_rejects_a_batched_row(self):
        with pytest.raises(ValueError, match="1-D"):
            at.topk_tokens(torch.zeros(2, 10), FakeTokenizer(), k=2)


# ---------------------------------------------------------------------------
# concentration measures
# ---------------------------------------------------------------------------
class TestConcentration:
    def test_uniform_entropy_is_log_n(self):
        assert at.entropy(torch.zeros(10)) == pytest.approx(math.log(10), rel=1e-5)

    def test_a_peaked_row_has_lower_entropy(self):
        assert at.entropy(logits_favouring(3, peak=20.0)) < 0.01

    def test_top_mass_of_a_uniform_row(self):
        assert at.top_mass(torch.zeros(10), k=3) == pytest.approx(0.3)

    def test_top_mass_saturates_on_a_peaked_row(self):
        assert at.top_mass(logits_favouring(3, peak=20.0), k=3) > 0.99


# ---------------------------------------------------------------------------
# sharding
# ---------------------------------------------------------------------------
class TestShard:
    def test_none_is_the_whole_list(self):
        assert at.parse_shard(None) == (0, 1)
        assert at.shard_rows(list(range(5)), 0, 1) == list(range(5))

    def test_parses_and_strides(self):
        assert at.parse_shard("2/4") == (2, 4)
        assert at.shard_rows(list(range(10)), 2, 4) == [2, 6]

    def test_shards_partition_the_list_exactly_once(self):
        rows = list(range(23))
        seen = []
        for i in range(4):
            seen.extend(at.shard_rows(rows, i, 4))
        assert sorted(seen) == rows

    @pytest.mark.parametrize("bad", ["4/4", "-1/4", "1/0", "x/2", "3"])
    def test_rejects_bad_specs(self, bad):
        with pytest.raises(ValueError):
            at.parse_shard(bad)


# ---------------------------------------------------------------------------
# corpus tally
# ---------------------------------------------------------------------------
def tally_frame():
    """Two pairs; at L42 both read 'More', at L6 they disagree."""
    rows = []
    for pid, l6 in (("p0", "UFO"), ("p1", "Hoang")):
        for layer, disp in ((6, l6), (42, "More")):
            rows.append({"cond": "prod", "pos": "label", "layer": layer,
                         "rank": 0, "display": disp, "prob": 0.5,
                         "is_word": True, "pair_id": pid})
            # a rank-1 row that must never enter the top-1 tally
            rows.append({"cond": "prod", "pos": "label", "layer": layer,
                         "rank": 1, "display": "ZZZ", "prob": 0.1,
                         "is_word": True, "pair_id": pid})
    return pd.DataFrame(rows)


class TestTally:
    def test_counts_only_rank_zero(self):
        out = at.top_tokens_by_layer(tally_frame(), "prod")
        assert all(t["token"] != "ZZZ" for toks in out.values() for t in toks)

    def test_agreement_shows_as_a_fraction(self):
        out = at.top_tokens_by_layer(tally_frame(), "prod")
        assert out[42][0] == {"token": "More", "count": 2, "frac": 1.0,
                              "mean_prob": 0.5}

    def test_disagreement_splits_the_fraction(self):
        out = at.top_tokens_by_layer(tally_frame(), "prod")
        assert {t["frac"] for t in out[6]} == {0.5}

    def test_words_only_drops_non_words(self):
        df = tally_frame()
        df.loc[df["display"] == "More", "is_word"] = False
        out = at.top_tokens_by_layer(df, "prod", words_only=True)
        assert 42 not in out or all(t["token"] != "More" for t in out[42])

    def test_other_conditions_are_excluded(self):
        assert at.top_tokens_by_layer(tally_frame(), "neutral") == {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCli:
    def test_defaults_to_all_seven_cases(self):
        from monocle import canonical
        args = at.build_parser().parse_args([])
        assert args.cases == list(canonical.CASES)
        assert args.kind == "proxy"
        assert args.n_pairs == 1000
        assert args.topk == 20
        assert args.shard is None

    def test_overrides(self):
        args = at.build_parser().parse_args(
            ["--cases", "road_quality", "--n-pairs", "50",
             "--shard", "1/8", "--topk", "5", "--positions", "label"])
        assert args.cases == ["road_quality"]
        assert args.n_pairs == 50
        assert at.parse_shard(args.shard) == (1, 8)
        assert args.positions == ["label"]

    def test_rejects_an_unregistered_case(self):
        with pytest.raises(SystemExit):
            at.build_parser().parse_args(["--cases", "not_a_case"])


# ---------------------------------------------------------------------------
# token-mask cache
# ---------------------------------------------------------------------------
class TestMaskCache:
    def test_builds_then_reuses(self, tmp_path, monkeypatch):
        calls = []

        def fake_build(tokenizer, vocab):
            calls.append(vocab)
            return torch.ones(vocab, dtype=torch.bool)

        monkeypatch.setattr(at.scoring, "build_token_mask", fake_build)
        tok = FakeTokenizer()
        m1 = at.cached_token_mask(tok, 10, tmp_path)
        m2 = at.cached_token_mask(tok, 10, tmp_path)
        assert calls == [10], "second call must hit the cache"
        assert torch.equal(m1, m2)

    def test_rebuilds_on_a_vocab_mismatch(self, tmp_path, monkeypatch):
        """A stale mask of the wrong length must never be returned."""
        calls = []

        def fake_build(tokenizer, vocab):
            calls.append(vocab)
            return torch.ones(vocab, dtype=torch.bool)

        monkeypatch.setattr(at.scoring, "build_token_mask", fake_build)
        tok = FakeTokenizer()
        at.cached_token_mask(tok, 10, tmp_path)
        # Overwrite the cache file with a mask of the wrong length.
        p = next(tmp_path.glob("token_mask_*.pt"))
        torch.save(torch.ones(4, dtype=torch.bool), p)
        m = at.cached_token_mask(tok, 10, tmp_path)
        assert m.numel() == 10
        assert calls == [10, 10]


# ---------------------------------------------------------------------------
# checkpoint parts
# ---------------------------------------------------------------------------
def rows_for(pairs):
    return [{"pair_id": p, "layer": 42, "rank": 0} for p in pairs]


class TestCheckpointParts:
    def test_parts_accumulate_and_merge(self, tmp_path):
        base = tmp_path / "answer_tokens.parquet"
        at.write_part(rows_for(["p0", "p1"]), base, 0)
        at.write_part(rows_for(["p2"]), base, 1)
        assert not base.exists(), "merge has not run yet"
        df = at.merge_parts(base)
        assert sorted(df["pair_id"]) == ["p0", "p1", "p2"]
        assert base.exists()
        assert list(base.parent.glob("*.part*.parquet")) == [], "parts removed"

    def test_completed_pairs_sees_parts_before_the_merge(self, tmp_path):
        """Resume must not redo pairs that only exist in a part file."""
        base = tmp_path / "answer_tokens.parquet"
        at.write_part(rows_for(["p0", "p1"]), base, 0)
        assert at.completed_pairs(base) == {"p0", "p1"}

    def test_completed_pairs_spans_merged_and_parts(self, tmp_path):
        base = tmp_path / "answer_tokens.parquet"
        at.write_part(rows_for(["p0"]), base, 0)
        at.merge_parts(base)
        at.write_part(rows_for(["p1"]), base, 1)
        assert at.completed_pairs(base) == {"p0", "p1"}

    def test_empty_rows_write_nothing(self, tmp_path):
        base = tmp_path / "answer_tokens.parquet"
        at.write_part([], base, 0)
        assert list(tmp_path.glob("*.parquet")) == []
        assert at.completed_pairs(base) == set()

    def test_merge_is_idempotent(self, tmp_path):
        base = tmp_path / "answer_tokens.parquet"
        at.write_part(rows_for(["p0", "p1"]), base, 0)
        first = at.merge_parts(base)
        second = at.merge_parts(base)
        assert len(first) == len(second) == 2
