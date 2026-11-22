"""The framework configs compose with no project roots (no site/, sv/ or research/).

A clone of the framework alone must run each dagspace default and each example.
"""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    ("vqa", []),
    ("vqa", ["pipeline=example_images"]),
    ("ocr", []),
    ("embed", []),
    ("pairwisevqa", []),
    ("speech", []),
    ("_template", ["pipeline=describe_images"]),
]


@pytest.fixture(autouse=True)
def _framework_only(monkeypatch):
    ensure_dotenv()
    monkeypatch.setenv("IMAGE_FORWARD_NO_PROJECT_ROOTS", "1")
    monkeypatch.setenv("IMAGE_FORWARD_MODEL_ZOO", "/models")  # some model configs read it
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("dagspace,overrides", CASES, ids=[f"{d}-{'+'.join(o) or 'default'}" for d, o in CASES])
def test_a_framework_config_composes_and_resolves(dagspace, overrides):
    conf = os.path.join(REPO, "dagspaces", dagspace, "conf")
    with initialize_config_dir(config_dir=conf, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    for key in ("data", "model"):
        OmegaConf.to_container(cfg[key], resolve=True)
    # An absolute data path must point into the repo (the example images).
    for value in OmegaConf.to_container(cfg.data, resolve=True).values():
        if isinstance(value, str) and value.startswith("/") and not value.startswith("/path/to/"):
            assert value.startswith(REPO), f"{dagspace} data has a site path: {value}"
            assert os.path.exists(value), f"{dagspace} data path does not exist: {value}"


def test_the_switch_removes_the_project_roots():
    from hydra._internal.utils import create_config_search_path

    conf = os.path.join(REPO, "dagspaces", "vqa", "conf")
    providers = {e.provider for e in create_config_search_path(conf).get_path()}
    assert "mllmsci-searchpath" not in providers
