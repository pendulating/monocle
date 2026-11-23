"""The orchestrator of lensread: 1 stage, `readout`."""

from dagspaces.common.dagspace_factory import make_dagspace

from .runner import ReadoutRunner

_dagspace = make_dagspace(
    __name__,
    name="lensread",
    stages={"readout": ReadoutRunner()},
)

# Submitit imports these names in each SLURM job. Do not change the names.
get_stage_registry = _dagspace.get_stage_registry
execute_stage_job = _dagspace.execute_stage_job
run_experiment = _dagspace.run_experiment
