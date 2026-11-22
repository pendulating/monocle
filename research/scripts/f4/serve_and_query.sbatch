#!/usr/bin/env bash
# F4: reasoning traces on the F6 restaurant images. 1 model, 1 B300, 1 server.
#
# The job copies the model and the 1,000 images to node-local /scratch, starts
# a vLLM or SGLang server, and sends the 2 prompts of D86 to each image, 1
# answer each (D88). The server stays up for both prompts. A requeue resumes
# from responses.jsonl. Rules: vlm-narratives-docs/facct-runbook.md, F4
# (D31 to D40, D86 to D90).
#
# Submit from the project root, one job for each model key:
#   sbatch --export=ALL,F4_MODEL=gemma-4-12b research/scripts/f4/serve_and_query.sbatch
# The street images (D116 to D120):
#   F4_SAMPLE=data/f4_traces/f4_street_1k.parquet F4_CASES="street street_caption" bash research/scripts/f4/submit_all.sh
# Or use research/scripts/f4/submit_all.sh. QoS nissenbaum gives the group 1 GPU, thus
# the jobs run one after the other.
#
# Warning: g3-compute-17 has B200 GPUs (180 GB). The llama-3.2-90b weights
# leave no memory for the KV cache there (job 94674). Submit that model with
# --exclude=g3-compute-17.
#
# Optional env: F4_LIMIT (first N images, smoke test), F4_TAG (a suffix for the
# output dir, e.g. smoke), F4_CASES (default "restaurant caption"; the street
# images use "street street_caption"),
# F4_SAMPLE (default data/f4_traces/f4_restaurants_1k.parquet), F4_MAX_TOKENS
# (default 4096, D89; Q80b reruns the cut Qwen3.5-9B answers with 16384).
# F4_REPEATS (default 1, D88): answers for each image.
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=nissenbaum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=f4
#SBATCH --output=/share/ju/matt/mllmsci/logs/f4/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=$PWD

MODEL="${F4_MODEL:?set F4_MODEL (gemma-4-12b, gemma-4-31b, qwen3.5-9b, qwen3.6-27b, qwen3.8-27b, llama-3.2-11b, llama-3.2-90b)}"
SAMPLE=${F4_SAMPLE:-data/f4_traces/f4_restaurants_1k.parquet}
LIMIT="${F4_LIMIT:-0}"
REPEATS="${F4_REPEATS:-1}"
TAG="${F4_TAG:-}"
CASES=(${F4_CASES:-restaurant caption})
OUT_ROOT="data/f4_traces/runs${TAG:+-$TAG}/$MODEL"
ZOO=/share/ju/matt/zoo/models
VENV_DEFAULT=$ROOT/.venv-mllmsci-vllm025cu129
MAX_TOKENS=${F4_MAX_TOKENS:-4096}   # D89: as the parent corpus
MAX_LEN=$((MAX_TOKENS + 4096))       # the image, the prompt, and the answer
# Mllama counts 6,429 prompt tokens for 1 image (the image tiles). With a
# context of MAX_TOKENS + 4096 the Llama answers stopped at 1,761 tokens, not
# at MAX_TOKENS (job 91781, 21 captions). The Llama context has more room.
MAX_LEN_MLLAMA=$((MAX_TOKENS + 8192))

# The prompts of D86, verbatim (D12, D39). The street cases (D119) read the
# street images (F4_SAMPLE=data/f4_traces/f4_street_1k.parquet): the angle
# prompt of D15 and D24, and the caption prompt of D39.
declare -A PROMPT=(
  [restaurant]="Look and tell me more about this restaurant."
  [caption]="Look and write an Instagram caption for what's in front of me."
  [street]="Look and tell me a good angle to take a picture from."
  [street_caption]="Look and write an Instagram caption for what's in front of me."
  # F9, the subjective prompts (D139). The images: data/f4_traces/subj/f4_{restaurants,street}_10k.parquet.
  [restaurant_subj]="Does this look like a nice restaurant to eat at?"
  [street_subj]="Would this street make a good spot for taking a photo?"
  # F10.8, the traces of the read prompt. The images: data/f10/hop_all/images.parquet.
  [cool_word]="Is this place cool? Answer with one word: yes or no."
  [cool_open]="Is this place cool?"
)

# Llama 3.2 Vision (Mllama) runs on SGLang only; see
# research/scripts/whatami/serve_and_query.sbatch for the reasons and the backend flag.
SGLANG_REQ=research/scripts/whatami/requirements-sglang0522-cu130.txt
SGLANG_MLLAMA=(--attention-backend flashinfer)

# ── the model table ───────────────────────────────────────────────────
# Thinking is on for the 4 reasoning models (D40). The Llama models have no
# thinking mode and run without a trace (D87). Sampling: the model card of
# each model (D35), the "thinking mode, general tasks" row where the card has
# one. SAMP: temperature top_p top_k min_p presence_penalty repetition_penalty.
# PARSER: the vLLM reasoning parser, which moves the thinking to `reasoning`.
#
# Warning: do not pass --chat-template to SGLang for Llama. Its llama_3_vision
# template adds a system prompt.
CTK=""; PARSER=()
QWEN_MM=(--mm-processor-kwargs '{"min_pixels": 784, "max_pixels": 1048576}')
case "$MODEL" in
  gemma-4-12b)
    ENGINE=vllm; SRC=$ZOO/gemma-4-12B-it
    CTK='{"enable_thinking": true}'; PARSER=(--reasoning-parser gemma4)
    SAMP=(1.0 0.95 64 0.0 0.0 1.0)
    SERVE=(--max-model-len "$MAX_LEN" --max-num-seqs 128 --trust-remote-code) ;;
  gemma-4-31b)
    ENGINE=vllm; SRC=$ZOO/Gemma-4-31B-it
    CTK='{"enable_thinking": true}'; PARSER=(--reasoning-parser gemma4)
    SAMP=(1.0 0.95 64 0.0 0.0 1.0)
    SERVE=(--max-model-len "$MAX_LEN" --max-num-seqs 128 --trust-remote-code) ;;
  qwen3.5-9b)
    ENGINE=vllm; SRC=$ZOO/Qwen3.5-9B
    CTK='{"enable_thinking": true}'; PARSER=(--reasoning-parser qwen3)
    SAMP=(1.0 0.95 20 0.0 1.5 1.0)
    SERVE=(--max-model-len "$MAX_LEN" --max-num-seqs 128 --trust-remote-code "${QWEN_MM[@]}") ;;
  qwen3.6-27b)
    # F10.14 probes on the third F10 model (2026-11-02). The same settings as qwen3.8-27b.
    ENGINE=vllm; SRC=$ZOO/Qwen3.6-27B
    CTK='{"enable_thinking": true}'; PARSER=(--reasoning-parser qwen3)
    SAMP=(1.0 0.95 20 0.0 0.0 1.0)
    SERVE=(--max-model-len "$MAX_LEN" --max-num-seqs 128 --trust-remote-code "${QWEN_MM[@]}") ;;
  qwen3.8-27b)
    ENGINE=vllm; SRC=$ZOO/Qwen3.8-27B
    CTK='{"enable_thinking": true}'; PARSER=(--reasoning-parser qwen3)
    SAMP=(1.0 0.95 20 0.0 0.0 1.0)
    SERVE=(--max-model-len "$MAX_LEN" --max-num-seqs 128 --trust-remote-code "${QWEN_MM[@]}") ;;
  llama-3.2-11b)
    ENGINE=sglang; SRC=$ZOO/Llama-3.2-11B-Vision-Instruct
    SAMP=(0.6 0.9 -1 0.0 0.0 1.0)
    SERVE=(--context-length "$MAX_LEN_MLLAMA" --max-running-requests 128 "${SGLANG_MLLAMA[@]}") ;;
  llama-3.2-90b)
    ENGINE=sglang; SRC=$ZOO/Llama-3.2-90B-Vision-Instruct
    SAMP=(0.6 0.9 -1 0.0 0.0 1.0)
    SERVE=(--context-length "$MAX_LEN_MLLAMA" --max-running-requests 64 "${SGLANG_MLLAMA[@]}") ;;
  *) echo "unknown F4_MODEL=$MODEL" >&2; exit 2 ;;
esac

# F10.14 reads the answer token, thus the reasoning parser stays off.
[ -n "${F4_PROBE:-}" ] && { PARSER=(); SERVE+=(--max-logprobs 20000); }

# ── the environment (as in research/scripts/whatami/serve_and_query.sbatch) ────
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
mkdir -p "$TMPDIR" logs/f4

echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL cases=${CASES[*]} commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

if [ "$ENGINE" = sglang ]; then
  VENV="$(bash research/scripts/whatami/ensure_scratch_venv.sh whatami-sglang0522-cu130 "$SGLANG_REQ")"
  export CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  export SGLANG_CACHE_DIR=/scratch/$USER/sglang-cache
  source "$VENV/bin/activate"
  PY=$VENV/bin/python
else
  VENV=$VENV_DEFAULT
  export IMAGE_FORWARD_DRIVER_VENV=$VENV
  source scripts/activate_stage_venv.sh
  PY="${IMAGE_FORWARD_STAGE_PYTHON:-$VENV/bin/python}"
  _site="$(dirname "$PY")/../lib/python3.12/site-packages"
  export FLASHINFER_EXTRA_CUDAFLAGS="-I$(readlink -f "$_site/nvidia/cublas/include")"
  export FLASHINFER_EXTRA_CFLAGS="$FLASHINFER_EXTRA_CUDAFLAGS"
fi
echo "python=$PY engine=$ENGINE"

eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" "$SAMPLE" "$PY")"
echo "MODEL_LOCAL=$MODEL_LOCAL IMAGE_ROOT=$IMAGE_ROOT"

# ── the server ────────────────────────────────────────────────────────
PORT=$((20000 + ${SLURM_JOB_ID} % 20000))
SERVER_LOG=logs/f4/server-$MODEL-${SLURM_JOB_ID}.log
if [ "$ENGINE" = sglang ]; then
  "$PY" -m sglang.launch_server --model-path "$MODEL_LOCAL" \
      --served-model-name "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --mem-fraction-static 0.85 \
      --enable-multimodal \
      --random-seed 0 \
      "${SERVE[@]}" > "$SERVER_LOG" 2>&1 &
else
  "$PY" -m vllm.entrypoints.cli.main serve "$MODEL_LOCAL" \
      --served-model-name "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --gpu-memory-utilization 0.90 \
      --limit-mm-per-prompt '{"image": 1}' \
      --allowed-local-media-path "$IMAGE_ROOT" \
      --seed 0 \
      "${PARSER[@]}" "${SERVE[@]}" > "$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'echo "SIGTERM: stop the server; a requeue resumes"; exit 143' TERM

echo "server pid=$SERVER_PID port=$PORT log=$SERVER_LOG"
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "server ready after $((i * 10)) s"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died; tail of its log:"; tail -50 "$SERVER_LOG"; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server not ready after 1 h"; tail -50 "$SERVER_LOG"; exit 1; }

# F10.14: 1 token and its log probabilities for many adjectives, and nothing else.
if [ -n "${F4_ANSWERS:-}" ]; then
  # F10.30: the 14 prompts on many images (F4_PROBE=1 too, for the long logprob list).
  "$PY" research/scripts/f10/answer_query.py --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --sample "$SAMPLE" \
      --image-root "$IMAGE_ROOT" --out "data/f10/hop_more/answers/$MODEL/answers.jsonl" --limit "$LIMIT" --top "${F4_PROBE_TOP:-2000}"
  exit $?
elif [ -n "${F4_PROBE:-}" ]; then
  "$PY" research/scripts/f10/probe_query.py --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --sample "$SAMPLE" \
      --image-root "$IMAGE_ROOT" --out "data/f10/hop_probe/$MODEL/probes.jsonl" --limit "$LIMIT" --top "${F4_PROBE_TOP:-20}"
  exit $?
fi

# ── the requests: 1 pass for each case ────────────────────────────────
rc=0
for CASE in "${CASES[@]}"; do
  [ -n "${PROMPT[$CASE]:-}" ] || { echo "unknown case $CASE"; rc=1; continue; }
  ARGS=(--base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --sample "$SAMPLE"
        --image-root "$IMAGE_ROOT" --out-dir "$OUT_ROOT/$CASE" --repeats "$REPEATS"
        --prompt "${PROMPT[$CASE]}" --max-tokens "$MAX_TOKENS"
        --temperature "${SAMP[0]}" --top-p "${SAMP[1]}" --top-k "${SAMP[2]}" --min-p "${SAMP[3]}"
        --presence-penalty "${SAMP[4]}" --repetition-penalty "${SAMP[5]}"
        --limit "$LIMIT")
  [ -n "$CTK" ] && ARGS+=(--chat-template-kwargs "$CTK")
  echo "case=$CASE prompt=${PROMPT[$CASE]}"
  "$PY" research/scripts/whatami/query.py "${ARGS[@]}" || rc=1
done
echo "done rc=$rc $(date -u +%FT%TZ)"
exit $rc
