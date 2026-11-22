"""Tests of the template dagspace. They need no GPU: a fake model answers.

scripts/new_dagspace.py copies this file for each new dagspace.
"""

from __future__ import annotations

import os
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import dagspaces._template.orchestrator as orch
from dagspaces._template.stages import describe
from dagspaces.common.config_schema import load_pipeline_graph
from dagspaces.common.stage_utils import ensure_dotenv

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(orch.__file__)), "conf")


@pytest.fixture()
def cfg():
    ensure_dotenv()  # sets IMAGE_FORWARD_PROJECT_ROOT for the example data paths
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        yield compose(config_name="config", overrides=["pipeline=describe_images"])
    GlobalHydra.instance().clear()


def test_the_pipeline_names_only_registered_stages(cfg):
    graph = load_pipeline_graph(cfg)
    registry = orch.get_stage_registry()
    assert graph.nodes, "the pipeline has no node"
    for node in graph.nodes.values():
        assert node.stage in registry
        assert node.launcher is None  # runtime.stage_launcher is null by default


def test_execute_stage_job_pickles_by_reference():
    # Submitit sends this function to each SLURM job. A reference keeps the job small.
    assert pickle.loads(pickle.dumps(orch.execute_stage_job)) is orch.execute_stage_job


def test_preprocess_gives_an_image_path_and_the_rendered_prompt(cfg):
    pre = describe.make_preprocess(cfg)
    row = pre({"sample_id": "a", "image_path": "/x/a.jpg", "prompt": "What is it?"})
    user = row["messages"][-1]
    assert user["role"] == "user"
    assert {"type": "image", "image": "/x/a.jpg"} in user["content"]
    assert {"type": "text", "text": "What is it?"} in user["content"]
    assert row["sampling_params"]["max_tokens"] == cfg.sampling_params.max_tokens


def test_postprocess_keeps_plain_columns_and_adds_the_answer():
    out = describe.postprocess(
        {"sample_id": "a", "messages": [], "sampling_params": {}, "generated_text": " A road. "}
    )
    assert out == {"sample_id": "a", "generated_text": " A road. ", "answer": "A road."}


def test_the_runner_reads_the_example_table_and_writes_the_results(cfg, tmp_path, monkeypatch):
    def fake_inference(df, cfg, preprocess, postprocess, stage_name):
        rows = [preprocess(r) for r in df.to_dict("records")]
        for r in rows:
            r["generated_text"] = f"answer for {r['sample_id']}"
        return pd.DataFrame([postprocess(r) for r in rows])

    monkeypatch.setattr("dagspaces.common.vllm_inference.run_vllm_inference", fake_inference)
    node = load_pipeline_graph(cfg).nodes["describe"]
    out_path = tmp_path / "results.parquet"
    context = SimpleNamespace(
        cfg=cfg,
        node=node,
        inputs={},
        output_paths={"results": str(out_path)},
        output_dir=str(tmp_path),
        output_root=str(tmp_path),
        logger=None,
    )
    result = orch.get_stage_registry()["describe"].run(context)

    out = pd.read_parquet(out_path)
    assert result.metadata["rows"] == len(out) == 10
    assert result.outputs["results"] == str(out_path)
    assert out["answer"].str.startswith("answer for ").all()
    # data.image_root makes each relative path of the table absolute.
    assert all(os.path.isabs(p) and os.path.isfile(p) for p in out["image_path"])
