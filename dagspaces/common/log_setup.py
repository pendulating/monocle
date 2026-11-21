"""One stdlib logging path for in-process stage code.

Stage modules printed to stdout before this module existed. SLURM captured
the prints, but the lines carried no severity and no time, and ten ad-hoc
``basicConfig`` calls fought over the root logger. In-process stage code now
calls :func:`get_stage_logger` once per module and logs at a level.

Subprocess workers keep their prints. Their stdout is the transport to the
SLURM output file, and a logger adds nothing there.

Usage::

    from dagspaces.common.log_setup import get_stage_logger

    log = get_stage_logger(__name__)
    log.info("loaded %d rows", n)
"""

from __future__ import annotations

import logging
import os

__all__ = ["get_stage_logger"]

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_CONFIGURED = "_mllmsci_stage_configured"


def _level_from_env() -> int:
    """Read the level from ``MLLMSCI_LOG_LEVEL``. Unknown names mean INFO."""
    raw = os.environ.get("MLLMSCI_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        return logging.INFO
    return level


def get_stage_logger(name: str) -> logging.Logger:
    """Return a stdlib logger with one shared console handler.

    The handler writes to stderr with the format
    ``%(asctime)s %(name)s %(levelname)s %(message)s``.

    The call is idempotent. Repeated calls with the same name return the
    same logger and never add a second handler.

    The level comes from the ``MLLMSCI_LOG_LEVEL`` environment variable at
    first setup. An unknown level name means INFO. A later explicit
    ``setLevel`` call wins over the env var.

    Propagation is off. Records stop at this logger, so each line prints
    once even when other code configures the root logger.
    """
    logger = logging.getLogger(name)
    if getattr(logger, _CONFIGURED, False):
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(_level_from_env())
    logger.propagate = False
    setattr(logger, _CONFIGURED, True)
    return logger
