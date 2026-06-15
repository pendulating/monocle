import os

# Strip inherited SLURM env vars on the submission side so Hydra's submitit
# launcher sees a clean environment. Inside a submitit-managed job the scheduler
# sets these for the real job; do not strip there.
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
