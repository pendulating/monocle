"""Shared utilities for SLURM detection and multiprocessing warnings."""

import os
from typing import Optional

from omegaconf import DictConfig

from .host_info import _parse_cpus_on_node
from .resource_tracker_patch import apply_patch as _apply_resource_tracker_patch


def _worker_process_setup_hook() -> None:
    """Setup function for worker processes to suppress multiprocessing warnings.

    Configures logging and suppresses harmless resource_tracker warnings.
    Checks the environment variable MLLMSCI_SUPPRESS_WARNINGS.
    """
    try:
        import asyncio
        import warnings

        try:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                try:
                    asyncio.get_event_loop_policy().get_event_loop()
                except Exception:
                    asyncio.set_event_loop(asyncio.new_event_loop())
        except Exception:
            pass

        suppress_warnings = os.environ.get("MLLMSCI_SUPPRESS_WARNINGS", "true").lower() in ("true", "1", "yes")
        if not suppress_warnings:
            return

        _apply_resource_tracker_patch()

        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"

        warnings.filterwarnings(
            "ignore",
            message="resource_tracker: process died unexpectedly",
            category=UserWarning,
            module="multiprocessing.resource_tracker"
        )
        warnings.filterwarnings(
            "ignore",
            message=".*resource_tracker.*",
            category=UserWarning,
            module="multiprocessing"
        )
    except Exception:
        pass


def get_suppress_child_warnings(cfg: Optional[DictConfig]) -> bool:
    """Get the suppress_child_warnings setting from config."""
    try:
        return getattr(getattr(cfg, "runtime", None), "suppress_child_warnings", True)
    except Exception:
        return True


# ---------------------------------------------------------------------------
#  SLURM helpers
# ---------------------------------------------------------------------------

# _parse_cpus_on_node moved to dagspaces.common.host_info. This module
# imports the canonical copy above.

def _detect_slurm_cpus() -> Optional[int]:
    """Detect SLURM CPU allocation from environment variables."""
    try:
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            return int(cpt)
        con = os.environ.get("SLURM_CPUS_ON_NODE")
        if con is not None and str(con).strip() != "":
            v = _parse_cpus_on_node(con)
            return v if v > 0 else None
    except Exception:
        pass
    return None
