"""A shim: this script is in image-forward ($IMAGE_FORWARD_HOME/scripts/build_image_pack.py). See README.md."""

import runpy
import sys

from dagspaces.common.paths import framework_home

path = str(framework_home() / "scripts" / "build_image_pack.py")
sys.argv[0] = path
runpy.run_path(path, run_name="__main__")
