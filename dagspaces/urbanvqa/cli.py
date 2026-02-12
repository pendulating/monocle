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
    # W&B env/login handled centrally by orchestrator and launcher; avoid toggling here
    # Ray Data context tuning (avoid noisy bars and control errored blocks)
    try:
        import ray
        ctx = ray.data.DataContext.get_current()
        ctx.enable_progress_bars = False
        # Silence Ray Data execution bars and minimize logging noise on SLURM
        try:
            if os.environ.get("RULE_TUPLES_SILENT"):
                ctx.enable_progress_bars = False
        except Exception:
            pass
        meb = int(getattr(cfg.runtime, "max_errored_blocks", 0) or 0)
        try:
            ctx.max_errored_blocks = meb
        except Exception:
            pass
        try:
            ctx.actor_idle_timeout_s = 1
        except Exception:
            pass
    except Exception:
        pass
    run_experiment(cfg)


if __name__ == "__main__":
    main()
