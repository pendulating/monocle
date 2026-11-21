from omegaconf import DictConfig

from dagspaces.common.cli_utils import dagspace_main
from .orchestrator import run_experiment


# This copy never stripped the inherited SLURM variables. Keep that behavior.
@dagspace_main(strip_slurm_env=False)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
