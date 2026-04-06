"""Multiprocessing utilities for urbanembed — re-exported from common."""
from dagspaces.common.multiprocessing_utils import (  # noqa: F401
    _worker_process_setup_hook, get_suppress_child_warnings,
    _detect_slurm_cpus, ensure_ray_init,
)
