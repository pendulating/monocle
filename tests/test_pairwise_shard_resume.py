"""Tests for the shard split and the resume cache of the pairwise path.

A 1,000,000-pair case runs as many jobs on a preemptable partition. Two
properties must hold, or the run loses rows without a warning:

1. The shards partition the pair table exactly — no row twice, no row lost,
   and every row of a canonical pair in the same shard.
2. A resumed job reads back what it wrote, and refuses a cache that another
   set of prompts wrote.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest
from omegaconf import OmegaConf

from dagspaces.urbanpairvqa.orchestrator import _shard_pairs
from dagspaces.common.vllm_inference import (
    _read_resume_chunk,
    _records_from_outputs,
    _resume_guard,
    _resume_settings,
    _write_resume_chunk,
)


def _pairs(n_canonical: int = 97, rows_each: int = 2) -> pd.DataFrame:
    rows = []
    for c in range(n_canonical):
        for r in range(rows_each):
            rows.append({
                "canonical_pair_id": f"unit_{c:08d}",
                "pair_id": f"unit_{c:08d}_{r}",
                "is_swapped": bool(r % 2),
            })
    return pd.DataFrame(rows)


def _cfg(**pair_sampler):
    return OmegaConf.create({"pair_sampler": pair_sampler})


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------

def test_shard_count_one_returns_everything():
    df = _pairs()
    out = _shard_pairs(df, _cfg(shard_count=1, shard_index=0))
    assert len(out) == len(df)


def test_shards_partition_the_table_exactly():
    df = _pairs(n_canonical=97, rows_each=3)
    count = 8
    seen = []
    for i in range(count):
        seen.append(_shard_pairs(df, _cfg(shard_count=count, shard_index=i)))
    total = pd.concat(seen, ignore_index=True)
    assert len(total) == len(df), "the shards lost or repeated rows"
    assert set(total["pair_id"]) == set(df["pair_id"])
    assert total["pair_id"].duplicated().sum() == 0


def test_a_canonical_pair_never_splits_across_shards():
    df = _pairs(n_canonical=50, rows_each=4)
    count = 7
    owner = {}
    for i in range(count):
        part = _shard_pairs(df, _cfg(shard_count=count, shard_index=i))
        for cid in part["canonical_pair_id"].unique():
            assert cid not in owner, f"{cid} is in shard {owner.get(cid)} and {i}"
            owner[cid] = i
    assert len(owner) == 50


def test_shard_index_out_of_range_raises():
    df = _pairs()
    with pytest.raises(ValueError, match="shard_index"):
        _shard_pairs(df, _cfg(shard_count=4, shard_index=4))


def test_shard_needs_a_canonical_pair_id():
    df = _pairs().drop(columns=["canonical_pair_id"])
    with pytest.raises(ValueError, match="canonical_pair_id"):
        _shard_pairs(df, _cfg(shard_count=2, shard_index=0))


def test_a_shard_that_holds_nothing_raises():
    df = _pairs(n_canonical=3, rows_each=1)
    with pytest.raises(ValueError, match="holds no pair"):
        _shard_pairs(df, _cfg(shard_count=10, shard_index=9))


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _model_cfg(**model):
    return OmegaConf.create({"model": model})


def test_resume_is_off_by_default():
    assert _resume_settings(OmegaConf.create({"model": {}}), 64) == (None, 0)


def test_resume_chunk_snaps_to_a_batch_boundary(tmp_path):
    cfg = _model_cfg(resume_dir=str(tmp_path), resume_chunk_rows=1000)
    _, rows = _resume_settings(cfg, 256)
    assert rows == 768, "a chunk must end where a batch ends"
    assert rows % 256 == 0


def test_resume_chunk_never_falls_below_one_batch(tmp_path):
    cfg = _model_cfg(resume_dir=str(tmp_path), resume_chunk_rows=10)
    _, rows = _resume_settings(cfg, 256)
    assert rows == 256


def test_resume_chunk_round_trip(tmp_path):
    records = [
        {"raw_text": "A is better", "prompt_tokens": 12, "completion_tokens": 4},
        {"raw_text": "B is better", "prompt_tokens": 13, "completion_tokens": 5},
    ]
    _write_resume_chunk(str(tmp_path), 0, 2, records, "test")
    back = _read_resume_chunk(str(tmp_path), 0, 2, "test")
    assert back == records


def test_a_missing_chunk_gives_none(tmp_path):
    assert _read_resume_chunk(str(tmp_path), 0, 2, "test") is None


def test_a_short_chunk_is_dropped_and_runs_again(tmp_path):
    _write_resume_chunk(str(tmp_path), 0, 4, [{"raw_text": "x",
                                               "prompt_tokens": 1,
                                               "completion_tokens": 1}], "test")
    assert _read_resume_chunk(str(tmp_path), 0, 4, "test") is None
    # The damaged file must be gone, or every later job reads it again.
    assert not os.path.exists(os.path.join(
        str(tmp_path), "gen_000000000_000000004.parquet"))


def test_a_damaged_chunk_is_dropped_and_runs_again(tmp_path):
    path = tmp_path / "gen_000000000_000000002.parquet"
    path.write_bytes(b"not a parquet file")
    assert _read_resume_chunk(str(tmp_path), 0, 2, "test") is None
    assert not path.exists()


def test_the_guard_accepts_the_same_prompts(tmp_path):
    prompts = [f"prompt {i}" for i in range(300)]
    _resume_guard(str(tmp_path), prompts, "test")
    _resume_guard(str(tmp_path), prompts, "test")   # must not raise


def test_the_guard_refuses_another_set_of_prompts(tmp_path):
    _resume_guard(str(tmp_path), [f"prompt {i}" for i in range(300)], "test")
    with pytest.raises(RuntimeError, match="another run"):
        _resume_guard(str(tmp_path), [f"other {i}" for i in range(300)], "test")


def test_the_guard_refuses_another_row_count(tmp_path):
    _resume_guard(str(tmp_path), [f"prompt {i}" for i in range(300)], "test")
    with pytest.raises(RuntimeError, match="another run"):
        _resume_guard(str(tmp_path), [f"prompt {i}" for i in range(299)], "test")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class _Completion:
    def __init__(self, text, token_ids):
        self.text = text
        self.token_ids = token_ids


class _Output:
    def __init__(self, text="hi", n_prompt=5, n_completion=2, empty=False):
        self.outputs = [] if empty else [_Completion(text, list(range(n_completion)))]
        self.prompt_token_ids = list(range(n_prompt))


def test_records_carry_text_and_token_counts():
    rows = _records_from_outputs([_Output("A is better", 7, 3)])
    assert rows == [{"raw_text": "A is better",
                     "prompt_tokens": 7, "completion_tokens": 3}]


def test_an_empty_completion_gives_an_empty_text():
    rows = _records_from_outputs([_Output(empty=True)])
    assert rows[0]["raw_text"] == ""
    assert rows[0]["completion_tokens"] == 0
