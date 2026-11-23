"""The pytest setup of monocle: site settings, markers and path fixtures.

The markers (pyproject.toml registers them):
    gpu    the test needs a CUDA GPU
    slurm  the test submits a SLURM job
    data   the test reads data that is not in git (the model zoo, multirun/, data/)

CI runs:  pytest -m "not gpu and not slurm and not data"
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dagspaces.common.stage_utils import ensure_dotenv

REPO = Path(__file__).resolve().parent

# The project root: this repo, also when pytest starts in a subfolder.
os.environ.setdefault("IMAGE_FORWARD_PROJECT_ROOT", str(REPO))

# Load server.env and .env, as the CLI does: some tests read IMAGE_FORWARD_MODEL_ZOO.
ensure_dotenv()

# A model config resolves ${oc.env:IMAGE_FORWARD_MODEL_ZOO} at compose time. With no
# server.env (CI), a placeholder lets the configs resolve. The folder does not exist,
# thus a test that needs the weights still skips.
os.environ.setdefault("IMAGE_FORWARD_MODEL_ZOO", str(REPO / "no-model-zoo"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The monocle repo root."""
    return REPO


@pytest.fixture(scope="session")
def model_zoo() -> Path:
    """The model zoo folder (IMAGE_FORWARD_MODEL_ZOO). The test skips when it is not set."""
    zoo = os.environ.get("IMAGE_FORWARD_MODEL_ZOO")
    if not zoo or not os.path.isdir(zoo):
        pytest.skip("IMAGE_FORWARD_MODEL_ZOO is not set or is not a folder")
    return Path(zoo)
