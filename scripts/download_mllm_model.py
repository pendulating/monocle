"""A shim: this script is in image-forward ($IMAGE_FORWARD_HOME/scripts/download_mllm_model.py). See README.md."""

import runpy
import sys

from dagspaces.common.paths import framework_home

path = str(framework_home() / "scripts" / "download_mllm_model.py")
sys.argv[0] = path
runpy.run_path(path, run_name="__main__")
