import os

# ── Clean inherited SLURM env vars ONLY on the terminal side ───────────────
# When launching from an interactive SLURM session, parent SLURM env vars
# leak into Hydra's submitit launcher and corrupt job tracking / result-pickle
# resolution.  We strip them here so the launcher submits cleanly.
#
# Inside a submitit-managed SLURM job (SUBMITIT_EXECUTOR is set by the
# submission script), the vars are *correct* (set by the scheduler for THIS
# job) and must NOT be removed.
if not os.environ.get("SUBMITIT_EXECUTOR"):
    for _k in list(os.environ):
        if _k.startswith("SLURM") or _k.startswith("SBATCH"):
            os.environ.pop(_k)

import hydra
from omegaconf import DictConfig

from .orchestrator import run_experiment


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
