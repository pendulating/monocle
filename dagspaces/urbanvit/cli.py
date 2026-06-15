import os

import hydra
from omegaconf import DictConfig

from .orchestrator import run_experiment


if not os.environ.get("SUBMITIT_EXECUTOR"):
    for _k in list(os.environ):
        if _k.startswith("SLURM") or _k.startswith("SBATCH"):
            os.environ.pop(_k)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
