"""Tests for the monitor-free trace-extraction submitter.

They compose configs and touch no SLURM. A composed shard must name its own
run directory, carry the engine shape of its arm, and hold NO node launcher,
because the GPU job IS the stage job.
"""

import pytest

from dagspaces.urbanpairvqa import submit_trace_shards as S


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    arm = S.ARMS["qwen_by_gemma31_v2"]
    out_root = str(tmp_path_factory.mktemp("runs") / "subway_safety__0003")
    cfg = S.compose_cfg(arm, "subway_safety", 3, 8, 250, out_root, [])
    return out_root, S.build_payload(cfg)


class TestCompose:
    def test_the_shard_owns_its_run_dir(self, payload):
        out_root, p = payload
        assert p["output_root"] == out_root
        assert p["cfg"]["trace_extract"]["shard_index"] == 3
        assert p["cfg"]["trace_extract"]["shard_count"] == 8

    def test_the_engine_shape_follows_the_arm(self, payload):
        _, p = payload
        eng = p["cfg"]["model"]["engine_kwargs"]
        assert eng["tensor_parallel_size"] == 2
        assert eng["max_num_seqs"] == 32
        assert eng["max_model_len"] == 32768
        assert p["cfg"]["model"]["model_source"].endswith("Gemma-4-31B-it")

    def test_no_node_launcher(self, payload):
        """A launcher here would submit a second job from inside the first."""
        _, p = payload
        assert p["node"]["launcher"] in (None, "null", "")

    def test_one_output_and_it_names_the_schema(self, payload):
        _, p = payload
        paths = list(p["output_paths"].values())
        assert len(paths) == 1
        assert "urban_cues_v2_s3_" in paths[0]

    def test_every_arm_names_a_registered_run(self):
        for name, arm in S.ARMS.items():
            assert set(arm["runs"]) == set(S.CASE_ORDER), name
            assert arm["tp"] in (1, 2), name
