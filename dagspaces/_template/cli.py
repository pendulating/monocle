"""The CLI of the template dagspace: `dagspace _template -m pipeline=describe_images`."""

from omegaconf import DictConfig

from dagspaces.common.cli_utils import dagspace_main
from .orchestrator import run_experiment


@dagspace_main()
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
