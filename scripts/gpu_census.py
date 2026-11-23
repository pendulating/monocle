"""A shim: this script is in image-forward ($IMAGE_FORWARD_HOME/scripts/gpu_census.py). See README.md."""

import runpy
import sys

from dagspaces.common.paths import framework_home

path = str(framework_home() / "scripts" / "gpu_census.py")
sys.argv[0] = path
runpy.run_path(path, run_name="__main__")
