"""Submit the shards of a sweep straight to the GPU partition.

Why this exists
---------------
A pairwise sweep is a graph of 1 node with no dependency. The normal path puts
a MONITOR job above each stage job: the monitor composes the config, submits
the stage, and then blocks on `job.result()` until the stage ends. That is
right for a real DAG and wasteful here. The 1,000,000-pair battery holds 966
shards of about 2 hours, thus the monitors would hold about 1,900 CPU-hours of
a node only to wait.

This module does the same work in 1 process on the login node. It composes the
config for each shard, submits every shard as a SLURM ARRAY, and exits. Each
GPU job runs `run_shard_job`, which runs the stage and writes the manifest that
the monitor used to write.

It uses `_create_submitit_executor`, the same helper the orchestrator uses, thus
it inherits `slurm_use_srun=False`. The Hydra submitit launcher cannot be used
here, because it always calls srun and srun is not usable on these nodes.

Use
---
    python -m dagspaces.urbanpairvqa.submit_shards --sweep million_proxy_qwen9b
    python -m dagspaces.urbanpairvqa.submit_shards --sweep ... --dry-run
    python -m dagspaces.urbanpairvqa.submit_shards --sweep ... \\
        --cases pairwise_schools_mvp --shards 2 --max-pairs 2000

Running a shard again
---------------------
A shard that hits the walltime ends in TIMEOUT. SLURM does NOT requeue it:
`--requeue` covers a preemption and a node failure, not a walltime. But the
shard keeps its resume chunks, thus a new job continues from where it stopped.

    python -m dagspaces.urbanpairvqa.submit_shards --sweep ... \\
        --base-dir <the same base> --only-missing --timeout-min 360

`--only-missing` asks each composed config where its parquet goes and drops the
shard when the file is there. Give it the SAME `--base-dir` as the first run.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONF_DIR = os.path.join(_HERE, "conf")


# ---------------------------------------------------------------------------
# Reading the grid out of a sweep file
# ---------------------------------------------------------------------------

def _parse_range(value: str) -> List[int]:
    """Turn Hydra `range(a,b)` or `0,1,2` into a list of integers."""
    value = str(value).strip()
    m = re.fullmatch(r"range\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", value)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))))
    return [int(v) for v in value.split(",") if v.strip() != ""]


def read_grid(sweep: str) -> Tuple[str, List[str], List[int]]:
    """Read the model, the cases and the shard indexes of a sweep file."""
    path = os.path.join(_CONF_DIR, "sweep", f"{sweep}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"no sweep file at {path}")
    cfg = OmegaConf.load(path)
    params = OmegaConf.select(cfg, "hydra.sweeper.params") or {}
    model = str(params.get("model", "")).strip()
    cases = [c.strip() for c in str(params.get("pipeline", "")).split(",") if c.strip()]
    shards = _parse_range(params.get("pair_sampler.shard_index", "0"))
    if not model or not cases:
        raise SystemExit(f"{path} does not name a model and a pipeline axis")
    return model, cases, shards


# ---------------------------------------------------------------------------
# Composing one shard's config
# ---------------------------------------------------------------------------

def pairs_path_for(pairs_dir: Optional[str], case: str) -> Optional[str]:
    """The prebuilt table of a case, when a dir was given."""
    if not pairs_dir:
        return None
    short = case.replace("pairwise_", "").replace("_mvp", "")
    return os.path.join(pairs_dir, f"{short}_pairs.parquet")


def compose_cfg(sweep: str, model: str, case: str, shard: int,
                output_root: str, extra: List[str]) -> DictConfig:
    """Compose the Hydra config of 1 shard, with no Hydra run directory."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    overrides = [
        f"+sweep={sweep}",
        f"model={model}",
        f"pipeline={case}",
        f"pair_sampler.shard_index={shard}",
        f"runtime.output_root={output_root}",
        # The node launcher must be empty: this job IS the stage job, thus a
        # launcher here would submit a second job from inside the first.
        "pipeline.graph.nodes.pairwise.launcher=null",
    ] + list(extra)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides,
                      return_hydra_config=False)
    return cfg


def build_payload(cfg: DictConfig) -> Dict[str, Any]:
    """Build what `run_shard_job` needs, the same way the monitor builds it."""
    from dagspaces.common.config_schema import load_pipeline_graph, resolve_output_root
    from dagspaces.common.orchestrator import (
        ArtifactRegistry, _node_inputs, _node_output_paths, common_parent,
        prepare_node_config,
    )

    graph_spec = load_pipeline_graph(cfg)
    output_root = resolve_output_root(graph_spec, cfg)
    os.makedirs(output_root, exist_ok=True)

    registry = ArtifactRegistry()
    for source_key, source in graph_spec.sources.items():
        path = source.path
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.expanduser(path))
        registry.register_source(source_key, path)

    order = graph_spec.topological_order()
    if len(order) != 1:
        raise SystemExit(
            f"this submitter runs a graph of 1 node, and this one holds "
            f"{len(order)}: {order}. Use the normal monitor path."
        )
    node = graph_spec.nodes[order[0]]

    inputs = _node_inputs(node, registry)
    output_paths = _node_output_paths(node, registry, output_root)
    output_dir = common_parent(output_paths.values()) or os.path.join(output_root, node.key)
    os.makedirs(output_dir, exist_ok=True)
    node_cfg = prepare_node_config(cfg, node, output_dir)

    return {
        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
        "node": {
            "key": node.key, "stage": node.stage,
            "depends_on": node.depends_on, "inputs": node.inputs,
            "outputs": {k: {"path": v.path, "type": v.type, "optional": v.optional}
                        for k, v in node.outputs.items()},
            "overrides": node.overrides, "launcher": node.launcher,
            "parallel_group": node.parallel_group,
            "max_attempts": node.max_attempts,
            "retry_backoff_s": node.retry_backoff_s,
            "wandb_suffix": node.wandb_suffix,
        },
        "inputs": inputs, "output_paths": output_paths,
        "output_dir": output_dir, "output_root": output_root,
    }


_STAMP_RE = re.compile(r"^(?P<stem>.+)_\d{8}_\d{6}(?P<ext>\.[^.]+)$")


def output_exists(path: str) -> bool:
    """True when the shard wrote this output, whatever the time in its name.

    The name of a result carries the time the job started, such as
    `subway_safety_mvp_20260818_173055.parquet`. A new config thus always
    composes a NEW name, and a test of that exact path always fails. We
    compare the part before the time instead.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    m = _STAMP_RE.match(os.path.basename(path))
    if not m:
        return False
    pattern = os.path.join(os.path.dirname(path),
                           f"{m.group('stem')}_*{m.group('ext')}")
    return any(os.path.getsize(f) > 0 for f in glob.glob(pattern))


def outputs_complete(payload: Dict[str, Any]) -> bool:
    """True when the shard already wrote every output that is not optional.

    The test reads `output_paths` of the composed payload, thus it asks the
    config where the parquet goes. A guess at the file name would go stale.
    """
    specs = payload["node"]["outputs"]
    checked = False
    for key, path in payload["output_paths"].items():
        if specs.get(key, {}).get("optional"):
            continue
        checked = True
        if not output_exists(path):
            return False
    # A node with no output that is required is never "complete".
    return checked


# ---------------------------------------------------------------------------
# Submitting
# ---------------------------------------------------------------------------

def submit(sweep: str, base_dir: str, *, launcher: str, parallel: int,
           cases: Optional[List[str]] = None, shards: Optional[int] = None,
           extra: Optional[List[str]] = None, dry_run: bool = False,
           pairs_dir: Optional[str] = None, only_missing: bool = False,
           timeout_min: Optional[int] = None) -> int:
    from dagspaces.common.orchestrator import (
        _clean_slurm_env, _create_submitit_executor, _load_launcher_config,
    )
    from dagspaces.urbanpairvqa.orchestrator import run_shard_job

    model, all_cases, all_shards = read_grid(sweep)
    use_cases = cases or all_cases
    use_shards = list(range(shards)) if shards else all_shards
    extra = list(extra or [])
    if shards:
        extra.append(f"pair_sampler.shard_count={shards}")

    print(f"[submit] sweep    : {sweep}")
    print(f"[submit] model    : {model}")
    print(f"[submit] cases    : {len(use_cases)}")
    print(f"[submit] shards   : {len(use_shards)} for each case")
    print(f"[submit] jobs     : {len(use_cases) * len(use_shards)}")
    print(f"[submit] launcher : {launcher}, {parallel} at a time")
    print(f"[submit] base     : {base_dir}")
    if pairs_dir:
        print(f"[submit] pairs    : prebuilt, from {pairs_dir}")
    if timeout_min:
        print(f"[submit] walltime : {timeout_min} min")
    if only_missing:
        print(f"[submit] mode     : only the shards that hold no final output")

    payloads: List[Tuple[str, int, Dict[str, Any]]] = []
    skipped: Dict[str, int] = defaultdict(int)
    for case in use_cases:
        short = case.replace("pairwise_", "").replace("_mvp", "")
        for shard in use_shards:
            out_root = os.path.join(base_dir, "runs", f"{short}__{shard:04d}")
            case_extra = list(extra)
            pp = pairs_path_for(pairs_dir, case)
            if pp:
                if not os.path.exists(pp):
                    raise SystemExit(
                        f"no prebuilt pair table at {pp}. Run "
                        f"scripts/prebuild_pair_tables.py first."
                    )
                case_extra.append(f"pair_sampler.pairs_path={pp}")
            cfg = compose_cfg(sweep, model, case, shard, out_root, case_extra)
            payload = build_payload(cfg)
            if only_missing and outputs_complete(payload):
                skipped[short] += 1
                continue
            payloads.append((short, shard, payload))

    if only_missing:
        total = sum(skipped.values())
        print(f"[submit] finished : {total} shards already hold their output")
        for short in sorted(skipped):
            n = skipped[short]
            print(f"    {short:<20} {n:>4} done, "
                  f"{len(use_shards) - n:>4} to run again")
    print(f"[submit] composed : {len(payloads)} shard configs")
    if not payloads:
        print("[submit] nothing to do")
        return 0
    if dry_run:
        for short, shard, p in payloads[:3]:
            print(f"    {short} shard {shard} → {p['output_root']}")
        if len(payloads) > 3:
            print(f"    ... and {len(payloads) - 3} more")
        return 0

    # One executor, thus one SLURM array for the whole sweep.
    cfg0 = compose_cfg(sweep, model, use_cases[0], use_shards[0],
                       os.path.join(base_dir, "runs", "_probe"), extra)
    launcher_cfg = _load_launcher_config(cfg0, launcher, _CONF_DIR)
    merge: Dict[str, Any] = {"array_parallelism": parallel}
    if timeout_min:
        # Override the launcher default instead of editing the shared YAML.
        # `slurm_gpu_preempt` sets 180 minutes, which other dagspaces rely on
        # for backfill. A shard that needs more asks for more here.
        merge["timeout_min"] = int(timeout_min)
    launcher_cfg = OmegaConf.merge(launcher_cfg, merge)

    log_folder = os.path.join(base_dir, ".slurm_jobs")
    os.makedirs(log_folder, exist_ok=True)
    executor = _create_submitit_executor(launcher_cfg, "PAIRVQA-shard", log_folder)

    with _clean_slurm_env():
        with executor.batch():
            jobs = [executor.submit(run_shard_job, p) for _, _, p in payloads]

    print(f"[submit] submitted {len(jobs)} jobs")
    if jobs:
        print(f"[submit] array    : {jobs[0].job_id.split('_')[0]}")
    with open(os.path.join(base_dir, "submitted_jobs.txt"), "w") as fh:
        for (short, shard, _), job in zip(payloads, jobs):
            fh.write(f"{job.job_id}\t{short}\t{shard}\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--base-dir", required=True,
                    help="Where the runs and the logs go.")
    ap.add_argument("--launcher", default="slurm_gpu_preempt")
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Pipeline names. The default takes them from the sweep.")
    ap.add_argument("--shards", type=int, default=None,
                    help="Use this many shards instead of the sweep's number.")
    ap.add_argument("--set", dest="extra", action="append", default=[],
                    help="A further Hydra override. Give it more than once.")
    ap.add_argument("--pairs-dir", default=None,
                    help="Dir of prebuilt pair tables. See "
                         "scripts/prebuild_pair_tables.py.")
    ap.add_argument("--only-missing", action="store_true",
                    help="Submit only the shards that hold no final output. "
                         "Use it to run a timed-out or preempted shard again; "
                         "it resumes from its chunks.")
    ap.add_argument("--timeout-min", type=int, default=None,
                    help="Walltime in minutes. The default is the launcher's.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    base = os.path.abspath(args.base_dir)
    os.makedirs(base, exist_ok=True)
    return submit(args.sweep, base, launcher=args.launcher,
                  parallel=args.parallel, cases=args.cases, shards=args.shards,
                  extra=args.extra, dry_run=args.dry_run,
                  pairs_dir=args.pairs_dir, only_missing=args.only_missing,
                  timeout_min=args.timeout_min)


if __name__ == "__main__":
    raise SystemExit(main())
