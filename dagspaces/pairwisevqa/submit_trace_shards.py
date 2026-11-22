"""Submit the shards of a trace-extraction arm straight to the GPU partition.

Why this exists
---------------
`extract_traces` is a graph of 1 node with no dependency. The Hydra multirun
path puts a MONITOR job above each shard: the monitor composes the config,
submits the GPU stage job, and then blocks on `job.result()` until the stage
ends. That is right for a real DAG and waste here. On 2026-09-20 the 2 qwen
arms of version 2 held 112 monitors, 1 CPU and 8 GB each, for 6 hours on
lisbeth and klara while their stage jobs waited in the queue. The monitors
took 29 of the 56 CPUs of klara, and a smoke job on klara then could not start.

This module does the monitors' work in 1 process on the login node and exits.
It composes the config of each (case, shard), submits every shard as 1 SLURM
ARRAY, and writes a record of what it submitted. Each GPU job runs
`run_shard_job`, which runs the stage and writes the `pipeline_manifest.json`
that the monitor used to write. `scripts/merge_trace_extractions.py` reads
the shard parquets from disk, thus nothing waits for anything.

It follows `submit_shards.py`, the submitter of the million-pair battery, and
it imports its helpers.

Use
---
    python -m dagspaces.pairwisevqa.submit_trace_shards --arm qwen_by_gemma_v2
    python -m dagspaces.pairwisevqa.submit_trace_shards --arm qwen_by_gemma_v2 --dry-run
    python -m dagspaces.pairwisevqa.submit_trace_shards --arm qwen_self_v2 \\
        --cases subway_safety libraries --shards 4

Running a shard again
---------------------
A preempted shard is requeued by SLURM itself (`--requeue` in the launcher),
and it skips the chunks it wrote. A shard that hit the walltime is not
requeued. Run the submitter again with `--only-missing` and the SAME
`--base-dir`: it drops each shard whose final parquet exists.

Where the output goes
---------------------
    <base-dir>/runs/<case>__<shard:04d>/outputs/extractions/<schema>_s<k>_<stamp>.parquet
    <base-dir>/runs/<case>__<shard:04d>/pipeline_manifest.json
    <base-dir>/submitted_jobs.txt      one line for each shard: job id, case, shard
    <base-dir>/launch.json             the arm, the model, the schema, the launcher

The default base dir is `multirun/<date>_<experiment>`. Merge with:

    python scripts/merge_trace_extractions.py <base-dir>/runs --out <root>

See `vlm-narratives-docs/langextract-trace-extraction.md`, "Submit without a
monitor".
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from dagspaces.pairwisevqa.submit_shards import (
    _CONF_DIR,
    build_payload,
    outputs_complete,
)

__version__ = "1.0.0"

_ROOT = "/share/ju/matt/mllmsci"

# The registered trace runs of the canonical registry, as named on 2026-09-04.
# The same table lives in `scripts/launch_trace_extraction.sh`.
_QWEN_SWEEP = "multirun/2026-08-14_URBANPAIRVQA/21-28-12"
_GEMMA_SWEEP = "multirun/2026-08-14_URBANPAIRVQA/21-28-32"
QWEN_RUNS: Dict[str, str] = {
    "subway_safety": f"{_QWEN_SWEEP}/0/outputs/pairwise/subway_safety_mvp_20260814_212820.parquet",
    "libraries": f"{_QWEN_SWEEP}/1/outputs/pairwise/libraries_mvp_20260814_212820.parquet",
    "schools": f"{_QWEN_SWEEP}/2/outputs/pairwise/schools_mvp_20260814_212820.parquet",
    "road_quality": f"{_QWEN_SWEEP}/3/outputs/pairwise/road_quality_mvp_20260814_212820.parquet",
    "parks_plazas": f"{_QWEN_SWEEP}/4/outputs/pairwise/parks_plazas_mvp_20260815_183234.parquet",
    "restaurants": f"{_QWEN_SWEEP}/5/outputs/pairwise/restaurants_mvp_20260816_015821.parquet",
    "street_photography": f"{_QWEN_SWEEP}/6/outputs/pairwise/street_photography_mvp_20260816_080638.parquet",
}
GEMMA_RUNS: Dict[str, str] = {
    "subway_safety": f"{_GEMMA_SWEEP}/0/outputs/pairwise/subway_safety_mvp_20260814_212840.parquet",
    "libraries": f"{_GEMMA_SWEEP}/1/outputs/pairwise/libraries_mvp_20260814_212840.parquet",
    "schools": f"{_GEMMA_SWEEP}/2/outputs/pairwise/schools_mvp_20260814_212840.parquet",
    "road_quality": f"{_GEMMA_SWEEP}/3/outputs/pairwise/road_quality_mvp_20260814_212840.parquet",
    "parks_plazas": f"{_GEMMA_SWEEP}/4/outputs/pairwise/parks_plazas_mvp_20260815_220955.parquet",
    "restaurants": f"{_GEMMA_SWEEP}/5/outputs/pairwise/restaurants_mvp_20260816_002948.parquet",
    "street_photography": f"{_GEMMA_SWEEP}/6/outputs/pairwise/street_photography_mvp_20260816_004528.parquet",
}
CASE_ORDER = list(QWEN_RUNS)

# The arms: (judge, extractor). The corpus roots are in
# `notebooks/cvpr/_extractions.ARMS`.
#
#   launcher, tp, seqs, mem: the engine shape. A 9B or 12B model in bf16 fits
#   1 GPU of 48 GB and holds 128 sequences. A 27B or 31B model needs 2 GPUs
#   and holds fewer sequences in the KV cache that the weights leave.
ARMS: Dict[str, Dict[str, Any]] = {
    "qwen_by_gemma_v2": dict(experiment="CUESV2Q", extractor="gemma-4-12b/instruct",
                             schema="urban_cues_v2", runs=QWEN_RUNS,
                             launcher="slurm_gpu_preempt", tp=1, seqs=128, mem=64),
    "qwen_self_v2": dict(experiment="CUESV2QQ", extractor="qwen3.5-9b/instruct",
                         schema="urban_cues_v2", runs=QWEN_RUNS,
                         launcher="slurm_gpu_preempt", tp=1, seqs=128, mem=64),
    "gemma_by_qwen_v2": dict(experiment="CUESV2", extractor="qwen3.5-9b/instruct",
                             schema="urban_cues_v2", runs=GEMMA_RUNS,
                             launcher="slurm_gpu_preempt", tp=1, seqs=128, mem=64),
    "gemma_self_v2": dict(experiment="CUESV2GG", extractor="gemma-4-12b/instruct",
                          schema="urban_cues_v2", runs=GEMMA_RUNS,
                          launcher="slurm_gpu_preempt", tp=1, seqs=128, mem=64),
    # The primary extractors (2026-09-20): the strongest model of the OTHER
    # family reads each judge. The same-family 27B/31B cell is the control.
    "gemma_by_qwen27_v2": dict(experiment="CUESV2GQ27", extractor="qwen3.8-27b/instruct",
                               schema="urban_cues_v2", runs=GEMMA_RUNS,
                               launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128),
    "qwen_by_gemma31_v2": dict(experiment="CUESV2QG31", extractor="gemma-4-31b/instruct",
                               schema="urban_cues_v2", runs=QWEN_RUNS,
                               launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128),
    "gemma_by_gemma31_v2": dict(experiment="CUESV2GG31", extractor="gemma-4-31b/instruct",
                                schema="urban_cues_v2", runs=GEMMA_RUNS,
                                launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128),
    "qwen_by_qwen27_v2": dict(experiment="CUESV2QQ27", extractor="qwen3.8-27b/instruct",
                              schema="urban_cues_v2", runs=QWEN_RUNS,
                              launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128),
    # Version 3 (2026-09-28): the classes on the IC codebook, clause-length
    # spans in the annotator's style. The 4 large arms read 1,000 random
    # traces for each case (`--set runtime.sample_n=1000`, the fixed seed
    # 777, thus the 2 arms of 1 judge read the same rows), in 4 shards of 250.
    "gemma_by_qwen27_v3": dict(experiment="CUESV3GQ27", extractor="qwen3.8-27b/instruct",
                               schema="urban_cues_v3", runs=GEMMA_RUNS,
                               launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128),
    # A qwen trace runs to 35,600 characters, and version 3 writes about 190
    # spans for one. At 16,384 tokens 22% to 35% of the answers were cut and
    # the decision, in the tail, was lost (2026-09-29). Thus the qwen-trace
    # arms take a 40,000-token cap in a 65,536-token window, and shards of
    # 63 rows (`--shards 16 --shard-rows 63 --timeout-min 300`).
    # Gemma-4-31B loops on a long answer: after about 100 distinct spans it
    # writes the same cycle again until the cap (2026-09-29: 70% duplicate
    # spans at a 40k cap, and a repetition penalty changed nothing). Thus
    # this arm reads a qwen trace in chunks of 6,000 characters, under the
    # loop threshold, with the 16k cap and the 32k window. The smoke: 26
    # minutes for 63 traces, no loop, 3% cut, 98.5% verbatim.
    "qwen_by_gemma31_v3": dict(experiment="CUESV3QG31", extractor="gemma-4-31b/instruct",
                               schema="urban_cues_v3", runs=QWEN_RUNS,
                               launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128,
                               extra=["extract.max_char_buffer=6000", "extract.max_tokens=16384",
                                      "extract.max_model_len=32768"]),
    "gemma_by_gemma31_v3": dict(experiment="CUESV3GG31", extractor="gemma-4-31b/instruct",
                                schema="urban_cues_v3", runs=GEMMA_RUNS,
                                launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128),
    "qwen_by_qwen27_v3": dict(experiment="CUESV3QQ27", extractor="qwen3.8-27b/instruct",
                              schema="urban_cues_v3", runs=QWEN_RUNS,
                              launcher="slurm_gpu_preempt_2x", tp=2, seqs=32, mem=128,
                              extra=["extract.max_tokens=40000", "extract.max_model_len=65536"]),
}


def compose_cfg(arm: Dict[str, Any], case: str, shard: int, shard_count: int,
                shard_rows: int, output_root: str, extra: List[str]) -> DictConfig:
    """Compose the Hydra config of 1 shard, with no Hydra run directory."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    results_path = arm["runs"][case]
    if not os.path.isabs(results_path):
        results_path = os.path.join(_ROOT, results_path)
    overrides = [
        "pipeline=extract_traces",
        f"extract={arm['schema']}",
        f"model={arm['extractor']}",
        f"experiment.name={arm['experiment']}",
        f"trace_extract.results_path={results_path}",
        f"trace_extract.shard_count={shard_count}",
        f"trace_extract.shard_index={shard}",
        f"trace_extract.shard_rows={shard_rows}",
        f"trace_extract.tensor_parallel_size={arm['tp']}",
        f"trace_extract.max_num_seqs={arm['seqs']}",
        f"runtime.job_memory_gb={arm['mem']}",
        f"runtime.output_root={output_root}",
        # The node launcher must be empty: this job IS the stage job, thus a
        # launcher here would submit a second job from inside the first.
        "pipeline.graph.nodes.extract.launcher=null",
    ] + list(extra)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides,
                      return_hydra_config=False)
    return cfg


def write_hydra_record(cfg: DictConfig, out_root: str, arm_name: str,
                       arm: Dict[str, Any], case: str, shard: int) -> None:
    """Write the `.hydra` pair that a normal Hydra run directory holds.

    `trace_extract.source_provenance` and `check_trace_extraction.sh` read
    `overrides.yaml`, and the canonical registry reads `config.yaml`.
    """
    try:
        hydra_dir = os.path.join(out_root, ".hydra")
        os.makedirs(hydra_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(hydra_dir, "config.yaml"))
        import yaml
        results_path = arm["runs"][case]
        with open(os.path.join(hydra_dir, "overrides.yaml"), "w") as fh:
            yaml.safe_dump([
                "pipeline=extract_traces", f"extract={arm['schema']}",
                f"model={arm['extractor']}", f"experiment.name={arm['experiment']}",
                f"trace_extract.results_path={results_path}",
                f"trace_extract.shard_index={shard}", f"arm={arm_name}",
            ], fh)
    except Exception as exc:  # noqa: BLE001
        # Provenance, not data. The merge reads the shard parquet itself.
        print(f"[submit] warning : no .hydra record for {case} shard {shard} "
              f"({type(exc).__name__}: {exc})")


def default_base_dir(experiment: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d")
    return os.path.join(_ROOT, "multirun", f"{stamp}_{experiment}")


def submit(arm_name: str, *, base_dir: Optional[str] = None,
           cases: Optional[List[str]] = None, shards: int = 8,
           shard_rows: int = 250, parallel: Optional[int] = None,
           launcher: Optional[str] = None, extra: Optional[List[str]] = None,
           only_missing: bool = False, timeout_min: Optional[int] = None,
           dry_run: bool = False, only: Optional[List[str]] = None) -> int:
    from dagspaces.common.orchestrator import (
        _clean_slurm_env, _create_submitit_executor, _load_launcher_config,
    )
    from dagspaces.pairwisevqa.orchestrator import run_shard_job

    if arm_name not in ARMS:
        raise SystemExit(f"unknown arm {arm_name!r}; one of {sorted(ARMS)}")
    if only_missing and not base_dir:
        # The default base dir carries today's date. A rerun after midnight
        # would compose a NEW base dir, find nothing there, and submit every
        # shard again beside the first run. Measured 2026-09-21 at 02:30.
        raise SystemExit(
            "--only-missing needs --base-dir: name the base dir of the first "
            "run, or the rerun lands in a new dated dir and repeats every shard"
        )
    arm = ARMS[arm_name]
    # An arm may carry its own overrides (`extra`); the command line adds to
    # them. The qwen-trace version 3 arms need a wider window and cap.
    extra = list(arm.get("extra", [])) + list(extra or [])
    launcher = launcher or arm["launcher"]
    use_cases = cases or CASE_ORDER
    base_dir = base_dir or default_base_dir(arm["experiment"])
    parallel = parallel or (len(use_cases) * shards)
    extra = list(extra or [])

    print(f"[submit] arm      : {arm_name}")
    print(f"[submit] judge run: {'qwen3.5-9b' if arm['runs'] is QWEN_RUNS else 'gemma-4-12b'} traces")
    print(f"[submit] extractor: {arm['extractor']}  schema {arm['schema']}")
    print(f"[submit] cases    : {len(use_cases)}  shards {shards} each  "
          f"chunks of {shard_rows} rows")
    print(f"[submit] jobs     : {len(use_cases) * shards}")
    print(f"[submit] launcher : {launcher}, tp={arm['tp']} seqs={arm['seqs']} "
          f"mem={arm['mem']}GB, {parallel} at a time")
    print(f"[submit] base     : {base_dir}")
    if only_missing:
        print("[submit] mode     : only the shards that hold no final output")

    # `--only case:shard` names the shards to submit, for a rerun of the few
    # that FAILED while the rest of the array still waits. `--only-missing`
    # cannot tell a failed shard from a pending one, thus it would submit the
    # pending ones a second time.
    wanted = None
    if only:
        wanted = set()
        for item in only:
            case, _, shard = item.partition(":")
            wanted.add((case, int(shard)))

    payloads: List[Tuple[str, int, Dict[str, Any]]] = []
    skipped: Dict[str, int] = defaultdict(int)
    for case in use_cases:
        if case not in arm["runs"]:
            raise SystemExit(f"no registered run for case {case!r}")
        src = arm["runs"][case]
        if not os.path.exists(os.path.join(_ROOT, src)):
            raise SystemExit(f"missing results parquet: {src}")
        for shard in range(shards):
            if wanted is not None and (case, shard) not in wanted:
                continue
            out_root = os.path.join(base_dir, "runs", f"{case}__{shard:04d}")
            cfg = compose_cfg(arm, case, shard, shards, shard_rows, out_root, extra)
            payload = build_payload(cfg)
            if only_missing and outputs_complete(payload):
                skipped[case] += 1
                continue
            if not dry_run:
                write_hydra_record(cfg, out_root, arm_name, arm, case, shard)
            payloads.append((case, shard, payload))

    if only_missing:
        total = sum(skipped.values())
        print(f"[submit] finished : {total} shards already hold their output")
    print(f"[submit] composed : {len(payloads)} shard configs")
    if not payloads:
        print("[submit] nothing to do")
        return 0
    if dry_run:
        for case, shard, p in payloads[:3]:
            eng = p["cfg"]["model"]["engine_kwargs"]
            print(f"    {case} shard {shard} → {p['output_root']}")
            print(f"      max_model_len={eng.get('max_model_len')} "
                  f"tp={eng.get('tensor_parallel_size')} seqs={eng.get('max_num_seqs')} "
                  f"model={p['cfg']['model']['model_source']}")
            print(f"      outputs: {list(p['output_paths'].values())}")
        if len(payloads) > 3:
            print(f"    ... and {len(payloads) - 3} more")
        return 0

    # One executor, thus one SLURM array for the whole arm.
    cfg0 = payloads[0][2]["cfg"]
    launcher_cfg = _load_launcher_config(OmegaConf.create(cfg0), launcher, _CONF_DIR)
    merge: Dict[str, Any] = {"array_parallelism": int(parallel)}
    if timeout_min:
        merge["timeout_min"] = int(timeout_min)
    launcher_cfg = OmegaConf.merge(launcher_cfg, merge)

    log_folder = os.path.join(base_dir, ".slurm_jobs")
    os.makedirs(log_folder, exist_ok=True)
    executor = _create_submitit_executor(
        launcher_cfg, f"{arm['experiment']}-shard", log_folder
    )
    with _clean_slurm_env():
        with executor.batch():
            jobs = [executor.submit(run_shard_job, p) for _, _, p in payloads]

    print(f"[submit] submitted {len(jobs)} jobs")
    if jobs:
        print(f"[submit] array    : {jobs[0].job_id.split('_')[0]}")
    with open(os.path.join(base_dir, "submitted_jobs.txt"), "a") as fh:
        for (case, shard, _), job in zip(payloads, jobs):
            fh.write(f"{job.job_id}\t{case}\t{shard}\n")
    record = {
        "arm": arm_name, "experiment": arm["experiment"],
        "extractor": arm["extractor"], "schema": arm["schema"],
        "launcher": launcher, "tp": arm["tp"], "seqs": arm["seqs"],
        "mem_gb": arm["mem"], "shards": shards, "shard_rows": shard_rows,
        "cases": use_cases, "array": jobs[0].job_id.split("_")[0] if jobs else None,
        "submitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "extra": extra,
    }
    with open(os.path.join(base_dir, "launch.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"[submit] merge    : python scripts/merge_trace_extractions.py "
          f"{base_dir}/runs --out <root>")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--base-dir", default=None,
                    help="where the runs go; default multirun/<date>_<experiment>")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--shard-rows", type=int, default=250,
                    help="rows for each resume chunk; the unit of loss on a preemption")
    ap.add_argument("--parallel", type=int, default=None,
                    help="array_parallelism; default every shard at once")
    ap.add_argument("--launcher", default=None, help="override the arm's launcher")
    ap.add_argument("--set", dest="extra", action="append", default=[],
                    help="an extra Hydra override, may repeat")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, metavar="CASE:SHARD",
                    help="submit these shards only, e.g. restaurants:6; for a "
                         "rerun of the shards that FAILED (needs --base-dir)")
    ap.add_argument("--timeout-min", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.only and not args.base_dir:
        ap.error("--only needs --base-dir, the base dir of the first run")
    return submit(args.arm, base_dir=args.base_dir, cases=args.cases,
                  shards=args.shards, shard_rows=args.shard_rows,
                  parallel=args.parallel, launcher=args.launcher,
                  extra=args.extra, only_missing=args.only_missing,
                  timeout_min=args.timeout_min, dry_run=args.dry_run,
                  only=args.only)


if __name__ == "__main__":
    sys.exit(main())
