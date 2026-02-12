"""Shared utilities for suppressing multiprocessing resource tracker warnings."""

from typing import Optional
from omegaconf import DictConfig

from .resource_tracker_patch import apply_patch as _apply_resource_tracker_patch


def _worker_process_setup_hook() -> None:
    """Setup function for Ray worker processes to suppress multiprocessing warnings.
    
    This function is called by Ray in each worker process to configure logging
    and suppress harmless resource_tracker warnings. It checks the environment
    variable URBANOCR_SUPPRESS_WARNINGS to determine if warnings should be suppressed.
    """
    try:
        import warnings
        import os
        
        # Check environment variable to determine if we should suppress warnings
        # Default to True (suppress) if not set
        suppress_warnings = os.environ.get("URBANOCR_SUPPRESS_WARNINGS", "true").lower() in ("true", "1", "yes")
        
        if not suppress_warnings:
            return

        _apply_resource_tracker_patch()
        
        # Set PYTHONWARNINGS for this process
        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"
        
        # Suppress UserWarning from multiprocessing.resource_tracker
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
    """Get the suppress_child_warnings setting from config.
    
    Args:
        cfg: Configuration object.
    
    Returns:
        True if warnings should be suppressed (default), False otherwise.
    """
    try:
        return getattr(getattr(cfg, "runtime", None), "suppress_child_warnings", True)
    except Exception:
        return True

