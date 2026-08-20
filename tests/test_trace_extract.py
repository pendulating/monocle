"""Unit tests for the trace-extraction stage.

These tests touch no GPU and read no parquet. They cover the parts that decide
WHICH traces a job reads: a wrong shard silently drops or repeats data, and no
downstream count can see that.
"""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from dagspaces.urbanpairvqa.stages.trace_extract import (
    _shard,
    case_of_parquet,
    source_provenance,
)


def _cfg(count: int, index: int):
    return OmegaConf.create(
        {"trace_extract": {"shard_count": count, "shard_index": index}}
    )


@pytest.fixture
def traces() -> pd.DataFrame:
    return pd.DataFrame({"pair_id": [f"p{i}" for i in range(103)]})


class TestShard:
    def test_one_shard_keeps_everything(self, traces):
        assert len(_shard(traces, _cfg(1, 0))) == len(traces)

    def test_the_shards_cover_every_trace_exactly_one_time(self, traces):
        """A gap loses data. An overlap counts a trace twice."""
        seen = []
        for i in range(8):
            seen.extend(_shard(traces, _cfg(8, i)).pair_id.tolist())
        assert sorted(seen) == sorted(traces.pair_id.tolist())
        assert len(seen) == len(set(seen))

    def test_the_shards_are_nearly_equal(self, traces):
        """103 traces over 8 jobs: no job may hold 2 more than another."""
        sizes = [len(_shard(traces, _cfg(8, i))) for i in range(8)]
        assert max(sizes) - min(sizes) <= 1

    def test_a_bad_index_raises(self, traces):
        with pytest.raises(ValueError):
            _shard(traces, _cfg(4, 4))
        with pytest.raises(ValueError):
            _shard(traces, _cfg(4, -1))


class TestNames:
    @pytest.mark.parametrize(
        "name,case",
        [
            ("subway_safety_mvp_20260813_013722.parquet", "subway_safety"),
            ("road_quality_mvp_20260813_013722.parquet", "road_quality"),
            ("libraries_mvp_20260813_013722.parquet", "libraries"),
            ("street_photography_mvp_20260724_104812.parquet", "street_photography"),
            # A gemma-4-e2b run keeps the unsplit original beside the fixed one.
            ("schools_mvp_20260624_152942.presplit.parquet", "schools"),
        ],
    )
    def test_case_of_parquet(self, name, case):
        assert case_of_parquet(f"/a/b/outputs/pairwise/{name}") == case

    def test_provenance_of_an_unknown_path_is_empty(self, tmp_path):
        """No Hydra record is normal for an older sweep. It must not raise."""
        p = tmp_path / "outputs" / "pairwise" / "subway_safety_mvp.parquet"
        got = source_provenance(str(p))
        assert got["judge_model"] == ""
        assert got["stage_dir"] == str(tmp_path)
