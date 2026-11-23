"""A shim: this script is in image-forward ($IMAGE_FORWARD_HOME/scripts/merge_pairwise_shards.py). See README.md."""

import runpy
import sys

from dagspaces.common.paths import framework_home

path = str(framework_home() / "scripts" / "merge_pairwise_shards.py")
sys.argv[0] = path
runpy.run_path(path, run_name="__main__")
