"""Tests for dagspaces.common.model_registry.resolve_model_source.

The registry sends a model load to a node-local /scratch mirror, but only when
the mirror carries a ``.sync_complete`` marker that names the same source.
Each test below sets MLLMSCI_MODEL_REGISTRY to a temporary directory, so no
test touches a real mirror.
"""

import os

import pytest

from dagspaces.common.model_registry import resolve_model_source


@pytest.fixture
def zoo(tmp_path, monkeypatch):
    """Give a fake zoo, a fake registry root, and a helper to make mirrors."""
    zoo_dir = tmp_path / "zoo" / "models"
    reg_dir = tmp_path / "registry" / "models"
    zoo_dir.mkdir(parents=True)
    reg_dir.mkdir(parents=True)
    monkeypatch.setenv("MLLMSCI_MODEL_REGISTRY", str(reg_dir))

    def make(name: str, *, mirror: bool = False, marker_src: str = None):
        src = zoo_dir / name
        src.mkdir(parents=True, exist_ok=True)
        (src / "config.json").write_text("{}")
        dst = reg_dir / name
        if mirror:
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "config.json").write_text("{}")
            if marker_src is not False:
                (dst / ".sync_complete").write_text(
                    f"src={marker_src or src}\nhost=testnode\n"
                )
        return src, dst

    return make


def test_complete_mirror_is_used(zoo):
    src, dst = zoo("Qwen3.5-9B", mirror=True)
    assert resolve_model_source(str(src)) == str(dst)


def test_trailing_slash_still_resolves(zoo):
    src, dst = zoo("Qwen3.5-9B", mirror=True)
    assert resolve_model_source(str(src) + "/") == str(dst)


def test_mirror_without_marker_is_refused(zoo):
    """A partial sync must never be used — this is the main safety property."""
    src, _ = zoo("gemma-4-12B-it", mirror=True, marker_src=False)
    assert resolve_model_source(str(src)) == str(src)


def test_marker_for_a_different_source_is_refused(zoo):
    """A basename collision must not load the weights of another model."""
    src, _ = zoo("Qwen3.5-9B", mirror=True, marker_src="/some/other/zoo/Qwen3.5-9B")
    assert resolve_model_source(str(src)) == str(src)


def test_no_mirror_falls_back(zoo):
    src, _ = zoo("Qwen3.5-4B", mirror=False)
    assert resolve_model_source(str(src)) == str(src)


def test_registry_unset_is_a_no_op(zoo, monkeypatch):
    src, dst = zoo("Qwen3.5-9B", mirror=True)
    monkeypatch.delenv("MLLMSCI_MODEL_REGISTRY", raising=False)
    assert resolve_model_source(str(src)) == str(src)
    monkeypatch.setenv("MLLMSCI_MODEL_REGISTRY", "")
    assert resolve_model_source(str(src)) == str(src)


def test_source_that_does_not_exist_falls_back(zoo, tmp_path):
    """Never redirect a source we cannot see, even if a mirror name matches."""
    zoo("Qwen3.5-9B", mirror=True)
    ghost = tmp_path / "nowhere" / "Qwen3.5-9B"
    assert resolve_model_source(str(ghost)) == str(ghost)


def test_hub_ids_and_empty_values_pass_through(zoo):
    assert resolve_model_source("Qwen/Qwen3-VL-2B-Instruct") == "Qwen/Qwen3-VL-2B-Instruct"
    assert resolve_model_source("") == ""
    assert resolve_model_source(None) == ""


def test_mirror_keeps_the_basename(zoo):
    """Path tests downstream (AWQ, gemma4-unified) must give the same result."""
    src, dst = zoo("Qwen3-VL-2B-Instruct-W4A16-AutoRound-AWQ", mirror=True)
    resolved = resolve_model_source(str(src))
    assert resolved == str(dst)
    assert os.path.basename(resolved) == os.path.basename(str(src))
    assert "awq" in resolved.lower()
