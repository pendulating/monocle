#!/usr/bin/env bash
# Extract the urban cues from the thinking traces of 1 judge with 1
# extractor: 7 multiruns, 1 for each case, 8 shards each.
#
# Warning: this is the OLD path (2026-09-20). It puts a monitor job above
# each shard, 112 for an arm, and each one waits 1 CPU and 8 GB for the
# stage job. Use `python -m research.traces.submit_trace_shards
# --arm <arm>` instead: 1 SLURM array, no monitor. This script stays for the
# record of how the version 1 arms ran.
#
# Usage:
#   bash research/scripts/launch_trace_extraction.sh <arm>
#
# The arms, by (judge, extractor). The gemma-judge / qwen-extractor half of
# 2026-08-28 (`CUESV1`, `data/trace_extractions/`) was launched by hand and
# is not here.
#
#   qwen_by_gemma   qwen3.5-9b traces,  gemma-4-12b extracts   CUESV1Q   (done 2026-09-05)
#   gemma_self      gemma-4-12b traces, gemma-4-12b extracts   CUESV1GG  (done 2026-09-05)
#   qwen_self       qwen3.5-9b traces,  qwen3.5-9b extracts    CUESV1QQ  (done 2026-09-05)
#
# The 4 arms make a 2 x 2 of judge and extractor, so a difference between 2
# judges can be read apart from a difference between 2 extractors. See
# research/docs/langextract-trace-extraction.md, Phases 6 and 7.
#
# The version 2 arms (schema `urban_cues_v2`, 2026-09-20) run the same 2 x 2
# with enums on the values, a qwen example, and a buffer that holds the
# longest qwen trace in 1 chunk. Version 1 split 5% to 36% of the qwen traces.
#
#   qwen_by_gemma_v2   CUESV2Q    data/trace_extractions_v2_qwen9b/
#   qwen_self_v2       CUESV2QQ   data/trace_extractions_v2_qwen9b_self/
#   gemma_by_qwen_v2   CUESV2     data/trace_extractions_v2/
#   gemma_self_v2      CUESV2GG   data/trace_extractions_v2_gemma12b_self/
#
# Warning: 2 schema versions never pool. Read a v2 corpus against a v2
# corpus only.
#
# The primary extractors (2026-09-20): each judge is read by the strongest
# model of the OTHER family, and the same-family 27B/31B cell is the control
# for the family effect. These arms run on 2 GPUs (`slurm_gpu_preempt_2x`,
# tensor parallel 2), because a 27B or 31B model in bf16 fills 54 to 62 GB.
#
#   gemma_by_qwen27_v2    gemma-4-12b traces, Qwen3.8-27B extracts   CUESV2GQ27  data/trace_extractions_v2_gemma12b_by_qwen27b/
#   qwen_by_gemma31_v2    qwen3.5-9b traces,  Gemma-4-31B extracts   CUESV2QG31  data/trace_extractions_v2_qwen9b_by_gemma31b/
#   gemma_by_gemma31_v2   gemma-4-12b traces, Gemma-4-31B extracts   CUESV2GG31  data/trace_extractions_v2_gemma12b_by_gemma31b/
#   qwen_by_qwen27_v2     qwen3.5-9b traces,  Qwen3.8-27B extracts   CUESV2QQ27  data/trace_extractions_v2_qwen9b_by_qwen27b/
#
# Each case runs as 8 shards of 1,375 traces on the preemptable `gpu`
# partition, in chunks of 250 rows. A chunk is the unit of loss on a
# preemption, and a rerun of the same shard skips a chunk that exists.
#
# Warning: sbatch forwards through ssh, and a burst of more than about 5
# monitors makes sshd refuse the extra connections. A refused monitor exits
# with success and its stage job never lands. Thus this script launches 1
# case at a time and waits 120 s between cases. Count what landed with
# research/scripts/check_trace_extraction.sh <experiment>.
#
# The script starts the 7 monitors' cli processes in the background and
# exits. Run it from the login node under nohup, so the cli processes live
# until every monitor returns:
#   nohup bash research/scripts/launch_trace_extraction.sh gemma_self > multirun/cuesv1gg_launch.log 2>&1 &
#
# The results paths are the registered trace runs of the canonical registry
# (`research/scripts/register_canonical_runs.py show`), as named on 2026-09-04.

set -u
cd /share/ju/matt/mllmsci
source .venv-mllmsci-vllm025cu129/bin/activate

ARM="${1:-}"
SLEEP_BETWEEN=120

QWEN_SWEEP=multirun/2026-08-14_URBANPAIRVQA/21-28-12
GEMMA_SWEEP=multirun/2026-08-14_URBANPAIRVQA/21-28-32

declare -A QWEN_RUNS=(
  [subway_safety]=$QWEN_SWEEP/0/outputs/pairwise/subway_safety_mvp_20260814_212820.parquet
  [libraries]=$QWEN_SWEEP/1/outputs/pairwise/libraries_mvp_20260814_212820.parquet
  [schools]=$QWEN_SWEEP/2/outputs/pairwise/schools_mvp_20260814_212820.parquet
  [road_quality]=$QWEN_SWEEP/3/outputs/pairwise/road_quality_mvp_20260814_212820.parquet
  [parks_plazas]=$QWEN_SWEEP/4/outputs/pairwise/parks_plazas_mvp_20260815_183234.parquet
  [restaurants]=$QWEN_SWEEP/5/outputs/pairwise/restaurants_mvp_20260816_015821.parquet
  [street_photography]=$QWEN_SWEEP/6/outputs/pairwise/street_photography_mvp_20260816_080638.parquet
)
declare -A GEMMA_RUNS=(
  [subway_safety]=$GEMMA_SWEEP/0/outputs/pairwise/subway_safety_mvp_20260814_212840.parquet
  [libraries]=$GEMMA_SWEEP/1/outputs/pairwise/libraries_mvp_20260814_212840.parquet
  [schools]=$GEMMA_SWEEP/2/outputs/pairwise/schools_mvp_20260814_212840.parquet
  [road_quality]=$GEMMA_SWEEP/3/outputs/pairwise/road_quality_mvp_20260814_212840.parquet
  [parks_plazas]=$GEMMA_SWEEP/4/outputs/pairwise/parks_plazas_mvp_20260815_220955.parquet
  [restaurants]=$GEMMA_SWEEP/5/outputs/pairwise/restaurants_mvp_20260816_002948.parquet
  [street_photography]=$GEMMA_SWEEP/6/outputs/pairwise/street_photography_mvp_20260816_004528.parquet
)
ORDER=(subway_safety libraries schools road_quality parks_plazas restaurants street_photography)

SCHEMA=urban_cues_v1
LAUNCHER=slurm_gpu_preempt
TP=1
SEQS=128
MEM=64
case "$ARM" in
  qwen_by_gemma) EXPERIMENT=CUESV1Q;  EXTRACTOR=gemma-4-12b/instruct; declare -n RUNS=QWEN_RUNS ;;
  gemma_self)    EXPERIMENT=CUESV1GG; EXTRACTOR=gemma-4-12b/instruct; declare -n RUNS=GEMMA_RUNS ;;
  qwen_self)     EXPERIMENT=CUESV1QQ; EXTRACTOR=qwen3.5-9b/instruct;  declare -n RUNS=QWEN_RUNS ;;
  qwen_by_gemma_v2) EXPERIMENT=CUESV2Q;  EXTRACTOR=gemma-4-12b/instruct; SCHEMA=urban_cues_v2; declare -n RUNS=QWEN_RUNS ;;
  qwen_self_v2)     EXPERIMENT=CUESV2QQ; EXTRACTOR=qwen3.5-9b/instruct;  SCHEMA=urban_cues_v2; declare -n RUNS=QWEN_RUNS ;;
  gemma_by_qwen_v2) EXPERIMENT=CUESV2;   EXTRACTOR=qwen3.5-9b/instruct;  SCHEMA=urban_cues_v2; declare -n RUNS=GEMMA_RUNS ;;
  gemma_self_v2)    EXPERIMENT=CUESV2GG; EXTRACTOR=gemma-4-12b/instruct; SCHEMA=urban_cues_v2; declare -n RUNS=GEMMA_RUNS ;;
  gemma_by_qwen27_v2)  EXPERIMENT=CUESV2GQ27; EXTRACTOR=qwen3.8-27b/instruct; SCHEMA=urban_cues_v2; LAUNCHER=slurm_gpu_preempt_2x; TP=2; SEQS=32; MEM=128; declare -n RUNS=GEMMA_RUNS ;;
  qwen_by_gemma31_v2)  EXPERIMENT=CUESV2QG31; EXTRACTOR=gemma-4-31b/instruct; SCHEMA=urban_cues_v2; LAUNCHER=slurm_gpu_preempt_2x; TP=2; SEQS=32; MEM=128; declare -n RUNS=QWEN_RUNS ;;
  gemma_by_gemma31_v2) EXPERIMENT=CUESV2GG31; EXTRACTOR=gemma-4-31b/instruct; SCHEMA=urban_cues_v2; LAUNCHER=slurm_gpu_preempt_2x; TP=2; SEQS=32; MEM=128; declare -n RUNS=GEMMA_RUNS ;;
  qwen_by_qwen27_v2)   EXPERIMENT=CUESV2QQ27; EXTRACTOR=qwen3.8-27b/instruct; SCHEMA=urban_cues_v2; LAUNCHER=slurm_gpu_preempt_2x; TP=2; SEQS=32; MEM=128; declare -n RUNS=QWEN_RUNS ;;
  *) echo "usage: $0 {qwen_by_gemma|gemma_self|qwen_self|qwen_by_gemma_v2|qwen_self_v2|gemma_by_qwen_v2|gemma_self_v2|gemma_by_qwen27_v2|qwen_by_gemma31_v2|gemma_by_gemma31_v2|qwen_by_qwen27_v2}" >&2; exit 2 ;;
esac
TAG=$(echo "$EXPERIMENT" | tr '[:upper:]' '[:lower:]')

echo "[launch] $(date '+%F %T') arm=$ARM experiment=$EXPERIMENT extractor=$EXTRACTOR schema=$SCHEMA launcher=$LAUNCHER tp=$TP seqs=$SEQS"
for case in "${ORDER[@]}"; do
  path="/share/ju/matt/mllmsci/${RUNS[$case]}"
  if [ ! -f "$path" ]; then
    echo "[launch] $(date '+%F %T') MISSING $case: $path"
    continue
  fi
  echo "[launch] $(date '+%F %T') $case"
  python -m dagspaces.pairwisevqa.cli -m \
    pipeline=extract_traces \
    extract=$SCHEMA \
    experiment.name=$EXPERIMENT \
    model=$EXTRACTOR \
    pipeline.graph.nodes.extract.launcher=$LAUNCHER \
    trace_extract.tensor_parallel_size=$TP \
    trace_extract.max_num_seqs=$SEQS \
    runtime.job_memory_gb=$MEM \
    trace_extract.results_path="$path" \
    trace_extract.shard_count=8 \
    trace_extract.shard_rows=250 \
    trace_extract.shard_index=0,1,2,3,4,5,6,7 \
    hydra.launcher.array_parallelism=8 \
    > "multirun/${TAG}_${case}.log" 2>&1 &
  echo "[launch] $(date '+%F %T') $case monitors started, pid $!"
  sleep $SLEEP_BETWEEN
done
echo "[launch] $(date '+%F %T') all 7 cases of $EXPERIMENT started; the cli processes stay until the monitors return"
