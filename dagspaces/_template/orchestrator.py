"""The orchestrator of the template dagspace.

Add a stage: write a function `(df, cfg) -> DataFrame` in stages/, and add a
runner for it to `stages=` below. A pipeline node then names it with `stage:`.
"""

from dagspaces.common.dagspace_factory import make_dagspace
from dagspaces.common.runners.dataframe import DataFrameStageRunner

from .stages.describe import run_describe_stage

_dagspace = make_dagspace(
    __name__,
    name="template",
    stages={"describe": DataFrameStageRunner("describe", run_describe_stage)},
)

# Submitit imports these names in each SLURM job. Do not change the names.
get_stage_registry = _dagspace.get_stage_registry
execute_stage_job = _dagspace.execute_stage_job
run_experiment = _dagspace.run_experiment
