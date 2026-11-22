"""scripts/new_dagspace.py makes a dagspace whose own tests pass."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("new_dagspace", os.path.join(REPO, "scripts", "new_dagspace.py"))
new_dagspace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(new_dagspace)


def test_a_new_dagspace_passes_its_copied_tests(tmp_path):
    files = new_dagspace.new_dagspace("mydemo", str(tmp_path))
    target = tmp_path / "mydemo"
    test_file = target / "test_mydemo_dagspace.py"
    assert str(test_file) in files

    text = (target / "orchestrator.py").read_text()
    assert 'name="mydemo"' in text and "_template" not in text
    assert "MYDEMO" in (target / "conf" / "config.yaml").read_text()

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), REPO]))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.parametrize("bad", ["My", "1x", "class", "a-b"])
def test_a_bad_name_stops(bad, tmp_path):
    with pytest.raises(ValueError):
        new_dagspace.new_dagspace(bad, str(tmp_path))


def test_an_existing_folder_stops(tmp_path):
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        new_dagspace.new_dagspace("taken", str(tmp_path))
