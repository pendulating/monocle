"""Build the orchestrator functions of a dagspace from its stage runners.

A dagspace ``orchestrator.py`` must give 3 module functions:
``get_stage_registry``, ``execute_stage_job`` (submitit runs it in each SLURM
job) and ``run_experiment`` (the CLI calls it). ``make_dagspace`` makes them::

    from dagspaces.common.dagspace_factory import make_dagspace
    from dagspaces.common.runners.dataframe import DataFrameStageRunner
    from .stages.describe import run_describe_stage

    _dagspace = make_dagspace(
        __name__,
        name="mydagspace",
        stages={"describe": DataFrameStageRunner("describe", run_describe_stage)},
    )
    get_stage_registry = _dagspace.get_stage_registry
    execute_stage_job = _dagspace.execute_stage_job
    run_experiment = _dagspace.run_experiment

**Warning:** bind the 3 functions to module names that are the same as the
function names, as in the example. Submitit pickles ``execute_stage_job`` by
its module and name, and the SLURM job imports it again from that module.
``run_experiment`` stops with an error when the names do not agree.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from omegaconf import DictConfig

from dagspaces.common import stage_submission
from dagspaces.common.runners.base import StageRunner

# The settings of a new dagspace: a monitor job that submits one SLURM job for
# each node (or runs a node with no launcher in the monitor), with a W&B run
# for each stage and job recovery. See DagspaceSettings for each flag.
DEFAULT_SETTINGS: Dict[str, Any] = dict(
    gpu_preflight=True,
    ensure_dotenv_first=True,
    status_includes_inputs=True,
    log_rows_metric=True,
    full_wandb_summaries=True,
    registered_runner_message=True,
    recovery_mode="basic",
    pipeline_style="monitor",
    announce_wandb_group=True,
    log_orchestrator_set_config=True,
    manifest_include_inputs=True,
)


@dataclass(frozen=True)
class Dagspace:
    """The orchestrator functions and the settings of one dagspace."""

    settings: stage_submission.DagspaceSettings
    get_stage_registry: Callable[[], Dict[str, StageRunner]]
    execute_stage_job: Callable[[Dict[str, Any]], Dict[str, Any]]
    run_experiment: Callable[[DictConfig], None]


def make_dagspace(
    module: str,
    *,
    name: str,
    stages: Mapping[str, StageRunner],
    conf_dir: Optional[str] = None,
    job_prefix: Optional[str] = None,
    **settings: Any,
) -> Dagspace:
    """Make the orchestrator functions of a dagspace.

    Args:
        module: The ``__name__`` of the orchestrator module.
        name: The dagspace name (W&B run config, the default job prefix).
        stages: The stage name of each runner, as the pipeline nodes use it.
        conf_dir: The ``conf/`` folder. The default is ``conf/`` beside the module.
        job_prefix: The SLURM job name part. The default is ``name.upper()``.
        **settings: Changes to ``DEFAULT_SETTINGS`` (see ``DagspaceSettings``).
    """
    unknown = set(settings) - set(stage_submission.DagspaceSettings.__dataclass_fields__)
    if unknown:
        raise TypeError(f"make_dagspace: unknown settings {sorted(unknown)}")
    if conf_dir is None:
        source = getattr(sys.modules.get(module), "__file__", None)
        if not source:
            raise ValueError(f"make_dagspace: module {module!r} has no file; give conf_dir")
        conf_dir = os.path.join(os.path.dirname(os.path.abspath(source)), "conf")

    registry: Dict[str, StageRunner] = dict(stages)

    def get_stage_registry() -> Dict[str, StageRunner]:
        return dict(registry)

    dagspace_settings = stage_submission.DagspaceSettings(
        dagspace_name=name,
        job_prefix=job_prefix or name.upper(),
        config_dir=conf_dir,
        registry_provider=get_stage_registry,
        **{**DEFAULT_SETTINGS, **settings},
    )

    def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run one stage. Submitit calls this function in the SLURM job."""
        return stage_submission.execute_stage_job(context_data, dagspace_settings)

    def run_experiment(cfg: DictConfig) -> None:
        """Run the pipeline of the config."""
        _check_bound(module, execute_stage_job)
        stage_submission.run_pipeline_experiment(
            cfg, dagspace_settings, execute_fn=execute_stage_job
        )

    # Give the functions the module and name that the orchestrator module binds
    # them to. Then pickle sends a reference, not a copy of the closure.
    for fn in (get_stage_registry, execute_stage_job, run_experiment):
        fn.__module__ = module
        fn.__qualname__ = fn.__name__

    return Dagspace(
        settings=dagspace_settings,
        get_stage_registry=get_stage_registry,
        execute_stage_job=execute_stage_job,
        run_experiment=run_experiment,
    )


def _check_bound(module: str, execute_stage_job: Callable[..., Any]) -> None:
    """Stop when the module does not bind ``execute_stage_job`` under that name."""
    bound = getattr(sys.modules.get(module), "execute_stage_job", None)
    if bound is not execute_stage_job:
        raise RuntimeError(
            f"{module} must bind the function of make_dagspace as "
            "`execute_stage_job = _dagspace.execute_stage_job`. "
            "SLURM jobs import it from the module by that name."
        )
