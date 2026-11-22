"""apply_image_root and make_dagspace."""

from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from dagspaces.common.dagspace_factory import make_dagspace
from dagspaces.common.runners.dataframe import apply_image_root


def test_image_root_joins_only_relative_paths():
    cfg = OmegaConf.create({"data": {"image_root": "/root"}})
    df = pd.DataFrame({"image_path": ["a.jpg", "/abs/b.jpg", "https://x/c.jpg", "", None]})
    out = apply_image_root(df, cfg)
    assert out["image_path"].tolist() == ["/root/a.jpg", "/abs/b.jpg", "https://x/c.jpg", "", None]
    assert df["image_path"].iloc[0] == "a.jpg"  # the input does not change


def test_no_image_root_changes_nothing():
    df = pd.DataFrame({"image_path": ["a.jpg"]})
    assert apply_image_root(df, OmegaConf.create({"data": {}})) is df


def test_make_dagspace_rejects_an_unknown_setting():
    with pytest.raises(TypeError):
        make_dagspace(__name__, name="x", stages={}, conf_dir="/tmp", no_such_flag=True)


def test_run_experiment_stops_when_the_module_does_not_bind_the_names():
    ds = make_dagspace(__name__, name="x", stages={}, conf_dir="/tmp")
    with pytest.raises(RuntimeError, match="execute_stage_job"):
        ds.run_experiment(OmegaConf.create({}))
