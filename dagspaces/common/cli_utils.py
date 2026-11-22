"""Shared CLI entry core for dagspaces.

The dagspace ``cli.py`` modules repeated one SLURM environment strip block and
one ``@hydra.main`` boilerplate. This module holds a single copy of each. The
``dagspace_main`` decorator selects the exact behavior variant of a dagspace:
the env strip on or off, and the ``conf/`` directory next to the module file.

Hydra stays the driver. The copies branched on the ``-m``/``--multirun``
flag inside ``@hydra.main``, together with argv override parsing, sweeper
expansion, and the ``HydraConfig`` runtime state. A manual
``hydra.initialize`` plus ``compose`` sequence cannot reproduce those, so the
decorator applies ``@hydra.main`` with the same arguments as each copy.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Callable

import hydra
from omegaconf import DictConfig


def strip_inherited_slurm_env() -> None:
    """Remove inherited SLURM variables from the process environment.

    A launch from an interactive SLURM session leaks parent ``SLURM_*`` and
    ``SBATCH_*`` variables into the submitit launcher. The leak corrupts job
    tracking and result-pickle resolution, so the strip runs before a launch.

    Inside a submitit job, the scheduler set these variables for THIS job and
    ``SUBMITIT_EXECUTOR`` is set. The strip keeps them in that case.
    """
    if not os.environ.get("SUBMITIT_EXECUTOR"):
        for _k in list(os.environ):
            if _k.startswith("SLURM") or _k.startswith("SBATCH"):
                os.environ.pop(_k)


def dagspace_main(
    *,
    config_name: str = "config",
    version_base: str = "1.3",
    strip_slurm_env: bool = True,
) -> Callable[[Callable[[DictConfig], None]], Callable[[], None]]:
    """Build the CLI entry point of one dagspace.

    Decorate the ``main(cfg)`` function of a ``cli.py`` module::

        @dagspace_main()
        def main(cfg: DictConfig) -> None:
            run_experiment(cfg)

    At import time the decorator strips the inherited SLURM variables. Pass
    ``strip_slurm_env=False`` for a copy that never stripped them (ocr).
    The decorator then applies ``hydra.main`` with the ``conf/`` directory
    next to the file of the decorated function. Hydra passes an absolute
    config path through verbatim, so the config search path matches the old
    relative ``config_path="conf"`` form of each copy.

    The returned callable parses argv, composes the config with the command
    line overrides, and branches on the ``-m``/``--multirun`` flag through
    the same Hydra machinery as the copies did.
    """

    def decorate(fn: Callable[[DictConfig], None]) -> Callable[[], None]:
        if strip_slurm_env:
            strip_inherited_slurm_env()

        # Resolve the conf directory from the module that declares fn, so the
        # search path does not depend on where this helper lives.
        module = sys.modules.get(fn.__module__)
        source = getattr(module, "__file__", None)
        if not source:
            source = inspect.getfile(fn)
        conf_dir = os.path.join(os.path.dirname(os.path.abspath(source)), "conf")

        return hydra.main(
            version_base=version_base,
            config_path=conf_dir,
            config_name=config_name,
        )(fn)

    return decorate
