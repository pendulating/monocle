#!/usr/bin/env bash
# F8.2 and F8.3: the counterfactual cue and the R3 readout. 1 model, 1 B300.
#
# Phase A: a vLLM server generates the 2 cue arms (rich, poor) for the
# canonical images of the 2 focus cases, with the F4 sampling and thinking on.
# The job then stops the server and reads R3 with the offline engine on the
# canonical F4 answers and on the cue answers.
# Phase B: R3 on the batch-2 F4 answers (the rest of the pool).
# Rules: vlm-narratives-docs/facct-runbook.md, F8.2 and F8.3.
#
# Submit from the project root, one job for each model and phase:
#   sbatch --export=ALL,F8_MODEL=gemma-4-12b,F8_PHASE=A --job-name=f8A-gemma-4-12b research/scripts/f8/cue_job.sbatch
# A requeue resumes: query.py and r3_readout.py skip the done rows.
#
# Optional env: F8_LIMIT (first N images or items, smoke test), F8_TAG (a
# suffix for the output dirs, e.g. smoke), F8_DESIGN (`cue`, the default:
# F8.2; `dc`: the decoupled cue of F8.5, D136, phase A only, with 4 arms and
# R3 on the new answers only).
#   sbatch --export=ALL,F8_MODEL=gemma-4-12b,F8_PHASE=A,F8_DESIGN=dc --job-name=f8dc-gemma-4-12b research/scripts/f8/cue_job.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=nissenbaum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --exclude=g3-compute-17
#SBATCH --output=/share/ju/matt/mllmsci/logs/f8/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=$PWD

MODEL="${F8_MODEL:?set F8_MODEL (gemma-4-12b, gemma-4-31b, qwen3.5-9b, qwen3.8-27b)}"
PHASE="${F8_PHASE:?set F8_PHASE (A or B)}"
LIMIT="${F8_LIMIT:-0}"
TAG="${F8_TAG:-}"
DESIGN="${F8_DESIGN:-cue}"
case "$DESIGN" in
  cue) ARMS=(rich poor); RUNS="data/f8/runs${TAG:+-$TAG}/$MODEL" ;;
  dc)  ARMS=(w_poor n_match n_rich city); RUNS="data/f8/runs_dc${TAG:+-$TAG}/$MODEL" ;;
  *) echo "unknown F8_DESIGN=$DESIGN" >&2; exit 2 ;;
esac
R3="data/f8/r3${TAG:+-$TAG}/$MODEL"
ITEMS="data/f8/items${TAG:+-$TAG}"
ZOO=/share/ju/matt/zoo/models
VENV=$ROOT/.venv-mllmsci-vllm025cu129
MAX_TOKENS=8192  # F4 used 4,096 (D89); the smoke job 103145 cut 3 of 16 cue answers at 4,096

# The model table: as research/scripts/f4/serve_and_query.sbatch (D35, D40).
QWEN_MM=(--mm-processor-kwargs '{"min_pixels": 784, "max_pixels": 1048576}')
case "$MODEL" in
  gemma-4-12b) SRC=$ZOO/gemma-4-12B-it; PARSER=gemma4; SAMP=(1.0 0.95 64 0.0 0.0 1.0); MM=() ;;
  gemma-4-31b) SRC=$ZOO/Gemma-4-31B-it; PARSER=gemma4; SAMP=(1.0 0.95 64 0.0 0.0 1.0); MM=() ;;
  qwen3.5-9b)  SRC=$ZOO/Qwen3.5-9B;     PARSER=qwen3;  SAMP=(1.0 0.95 20 0.0 1.5 1.0); MM=("${QWEN_MM[@]}") ;;
  qwen3.8-27b) SRC=$ZOO/Qwen3.8-27B;    PARSER=qwen3;  SAMP=(1.0 0.95 20 0.0 0.0 1.0); MM=("${QWEN_MM[@]}") ;;
  *) echo "unknown F8_MODEL=$MODEL" >&2; exit 2 ;;
esac
MAX_LEN=$((MAX_TOKENS + 4096))

# ── the environment (as research/scripts/f4/serve_and_query.sbatch) ─────────────
export PATH=/share/ju/matt/bin:$HOME/.local/bin:$PATH
export TMPDIR=/scratch/$USER/tmp
export HF_HOME=/scratch/$USER/hf-home
export VLLM_CACHE_ROOT=/scratch/$USER/vllm-cache
export VLLM_CONFIG_ROOT=/scratch/$USER/vllm-config
export XDG_CACHE_HOME=/scratch/$USER/xdg-cache
export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" logs/f8

echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL phase=$PHASE design=$DESIGN commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
_site="$(dirname "$PY")/../lib/python3.12/site-packages"
export FLASHINFER_EXTRA_CUDAFLAGS="-I$(readlink -f "$_site/nvidia/cublas/include")"
export FLASHINFER_EXTRA_CFLAGS="$FLASHINFER_EXTRA_CUDAFLAGS"
echo "python=$PY"

eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" data/f8/images_all.parquet "$PY")"
echo "MODEL_LOCAL=$MODEL_LOCAL IMAGE_ROOT=$IMAGE_ROOT"

# The offline engine JIT-builds a flashinfer sampling kernel that needs the
# system libstdc++ (GLIBCXX_3.4.32), not the older anaconda copy (job 103145).
R3_ENV=(env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6)
R3_ARGS=(--model-key "$MODEL" --model-path "$MODEL_LOCAL" --image-root "$IMAGE_ROOT" --limit "$LIMIT")

if [ "$PHASE" = A ]; then
  # ── the server and the cue arms ──────────────────────────────────────
  PORT=$((20000 + ${SLURM_JOB_ID} % 20000))
  SERVER_LOG=logs/f8/server-$MODEL-${SLURM_JOB_ID}.log
  "$PY" -m vllm.entrypoints.cli.main serve "$MODEL_LOCAL" \
      --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 1}' \
      --allowed-local-media-path "$IMAGE_ROOT" --seed 0 \
      --reasoning-parser "$PARSER" --max-model-len "$MAX_LEN" --max-num-seqs 128 \
      --trust-remote-code "${MM[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  stop_server() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
  trap stop_server EXIT
  trap 'echo "SIGTERM: stop; a requeue resumes"; exit 143' TERM
  for i in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "server ready after $((i * 10)) s"; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; tail -50 "$SERVER_LOG"; exit 1; fi
    sleep 10
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server not ready after 1 h"; exit 1; }

  for CASE in restaurant street; do
    for ARM in "${ARMS[@]}"; do
      echo "case=$CASE arm=$ARM $(date -u +%FT%TZ)"
      "$PY" research/scripts/whatami/query.py --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
          --sample "data/f8/${DESIGN}_${CASE}_${ARM}.parquet" --prompt-column prompt \
          --image-root "$IMAGE_ROOT" --out-dir "$RUNS/${CASE}_${ARM}" --repeats 1 \
          --max-tokens "$MAX_TOKENS" --temperature "${SAMP[0]}" --top-p "${SAMP[1]}" --top-k "${SAMP[2]}" \
          --min-p "${SAMP[3]}" --presence-penalty "${SAMP[4]}" --repetition-penalty "${SAMP[5]}" \
          --chat-template-kwargs '{"enable_thinking": true}' --limit "$LIMIT" || echo "query rc=$? (continue)"
    done
  done
  stop_server
  trap - EXIT
  echo "server stopped $(date -u +%FT%TZ)"

  # ── R3 on the canonical F4 answers and on the cue answers ────────────
  if [ "$DESIGN" = dc ]; then
    # F8.2 already read R3 on the F4 answers; read the new answers only.
    "$PY" research/scripts/f8/build_r3_items.py --model "$MODEL" --source dc --runs-root "$(dirname "$RUNS")" \
        --out "$ITEMS/$MODEL-dc.parquet"
    "${R3_ENV[@]}" "$PY" research/scripts/f8/r3_readout.py "${R3_ARGS[@]}" --items "$ITEMS/$MODEL-dc.parquet" --out-dir "$R3/dc"
    echo "done $(date -u +%FT%TZ)"
    exit 0
  fi
  "$PY" research/scripts/f8/build_r3_items.py --model "$MODEL" --source f4 --out "$ITEMS/$MODEL-f4.parquet"
  "$PY" research/scripts/f8/build_r3_items.py --model "$MODEL" --source cue --runs-root "$(dirname "$RUNS")" \
      --out "$ITEMS/$MODEL-cue.parquet"
  "${R3_ENV[@]}" "$PY" research/scripts/f8/r3_readout.py "${R3_ARGS[@]}" \
      --items "$ITEMS/$MODEL-f4.parquet" "$ITEMS/$MODEL-cue.parquet" --out-dir "$R3/f4" "$R3/cue"
else
  "$PY" research/scripts/f8/build_r3_items.py --model "$MODEL" --source f4b2 --out "$ITEMS/$MODEL-f4b2.parquet"
  "${R3_ENV[@]}" "$PY" research/scripts/f8/r3_readout.py "${R3_ARGS[@]}" --items "$ITEMS/$MODEL-f4b2.parquet" --out-dir "$R3/f4b2"
fi
echo "done $(date -u +%FT%TZ)"
