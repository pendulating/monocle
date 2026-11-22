#!/usr/bin/env bash
# "What am I looking at?" probe: one model, one B300, a vLLM or SGLang server.
#
# The job copies the model and the images to node-local /scratch, starts
# the server, waits for /health, and then sends 1,000 images x 10 repeats
# (research/scripts/whatami/query.py). The server stays up for all 10,000 requests.
# A requeue resumes from responses.jsonl.
#
# Submit from the project root, one job for each model key:
#   sbatch --export=ALL,WHATAMI_MODEL=gemma-4-12b research/scripts/whatami/serve_and_query.sbatch
# Or use research/scripts/whatami/submit_all.sh. QoS nissenbaum gives the group 1 GPU,
# thus the jobs run one after the other.
#
# Optional env: WHATAMI_SAMPLE (default data/whatami/sample_1k.parquet),
# WHATAMI_LIMIT (first N images, smoke test), WHATAMI_REPEATS (default 10),
# WHATAMI_TAG (a suffix for the output dir, e.g. smoke),
# WHATAMI_OUT (default multirun/whatami/<sample>[-<tag>]/<model>).
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
#SBATCH --job-name=whatami
#SBATCH --output=/share/ju/matt/mllmsci/logs/whatami/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=$PWD

MODEL="${WHATAMI_MODEL:?set WHATAMI_MODEL (gemma-4-12b, qwen3.5-9b, llama-3.2-11b, llama-3.2-90b)}"
SAMPLE="${WHATAMI_SAMPLE:-data/whatami/sample_1k.parquet}"
REPEATS="${WHATAMI_REPEATS:-10}"
LIMIT="${WHATAMI_LIMIT:-0}"
TAG="${WHATAMI_TAG:-}"
OUT="${WHATAMI_OUT:-multirun/whatami/$(basename "$SAMPLE" .parquet)${TAG:+-$TAG}/$MODEL}"
ZOO=/share/ju/matt/zoo/models
VENV_DEFAULT=$ROOT/.venv-mllmsci-vllm025cu129
# Llama 3.2 Vision (Mllama) runs on SGLang. vLLM removed Mllama in 0.10.2, and
# 0.10.1.1 needs torch 2.7.1, which has no B300 (sm_103) kernels (job 70187:
# "no kernel image is available"). The SGLang venv is built on /scratch in the
# job (research/scripts/whatami/ensure_scratch_venv.sh).
SGLANG_REQ=research/scripts/whatami/requirements-sglang0522-cu130.txt
# On Blackwell, SGLang selects the TRT-LLM attention backend, which fails on
# the Mllama cross-attention layers in the first prefill (job 70604: "Tensor
# match failed" in kvcache.cuh). The FlashInfer backend handles encoder_lens.
SGLANG_MLLAMA=(--attention-backend flashinfer)

# ── the model table ───────────────────────────────────────────────────
# ENGINE: vllm or sglang. CTK: chat_template_kwargs. Thinking is off for both
# reasoning models, because a glasses answer is spoken at once. SERVE: extra
# server flags.
#
# Warning: do not pass --chat-template to SGLang for Llama. Its llama_3_vision
# template adds a system prompt. Without the flag, SGLang uses the model's own
# chat template, which adds no system prompt when an image is present.
CTK=""
case "$MODEL" in
  gemma-4-12b)
    ENGINE=vllm; SRC=$ZOO/gemma-4-12B-it
    CTK='{"enable_thinking": false}'
    SERVE=(--max-model-len 4096 --max-num-seqs 128 --trust-remote-code) ;;
  qwen3.5-9b)
    ENGINE=vllm; SRC=$ZOO/Qwen3.5-9B
    CTK='{"enable_thinking": false}'
    # The same pixel cap as the CVPR battery (qwen3.5-9b/instruct.yaml).
    SERVE=(--max-model-len 4096 --max-num-seqs 128 --trust-remote-code
           --mm-processor-kwargs '{"min_pixels": 784, "max_pixels": 1048576}') ;;
  llama-3.2-11b)
    ENGINE=sglang; SRC=$ZOO/Llama-3.2-11B-Vision-Instruct
    SERVE=(--context-length 8192 --max-running-requests 128 "${SGLANG_MLLAMA[@]}") ;;
  llama-3.2-90b)
    ENGINE=sglang; SRC=$ZOO/Llama-3.2-90B-Vision-Instruct
    SERVE=(--context-length 8192 --max-running-requests 64 "${SGLANG_MLLAMA[@]}") ;;
  *) echo "unknown WHATAMI_MODEL=$MODEL" >&2; exit 2 ;;
esac

# ── the environment (as in /share/ju/matt/lfpw/scripts/slurm/*.sbatch) ──
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
mkdir -p "$TMPDIR" logs/whatami

echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

# The venv on /scratch. For vLLM: the mirror, CUDA 12.9 and the FFmpeg libs
# (see activate_stage_venv.sh). For SGLang: a CUDA 13 stack, thus the CUDA 13.3
# toolkit for its FlashInfer JIT.
if [ "$ENGINE" = sglang ]; then
  VENV="$(bash research/scripts/whatami/ensure_scratch_venv.sh whatami-sglang0522-cu130 "$SGLANG_REQ")"
  export CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3
  export PATH=$CUDA_HOME/bin:$PATH
  # The venv python is anaconda's, which loads its own old libstdc++ first. The
  # SGLang JIT kernels need GLIBCXX_3.4.31 (job 70482). The repo launchers use
  # the same preload.
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  # The default is ~/.cache/sglang, on the home quota.
  export SGLANG_CACHE_DIR=/scratch/$USER/sglang-cache
  source "$VENV/bin/activate"
  PY=$VENV/bin/python
else
  VENV=$VENV_DEFAULT
  export MLLMSCI_DRIVER_VENV=$VENV
  source scripts/activate_stage_venv.sh
  PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
  # On the B300 (sm_103a), qwen gets the FlashInfer attention backend, and its
  # JIT build includes cublasLt.h. /share/ju/matt/cuda-12.9 has no cuBLAS
  # headers (job 70186). The cuBLAS wheel in the venv has them, at the same
  # 12.9 version as torch. The module needs only the header, not a link flag.
  _site="$(dirname "$PY")/../lib/python3.12/site-packages"
  export FLASHINFER_EXTRA_CUDAFLAGS="-I$(readlink -f "$_site/nvidia/cublas/include")"
  export FLASHINFER_EXTRA_CFLAGS="$FLASHINFER_EXTRA_CUDAFLAGS"
fi
echo "python=$PY engine=$ENGINE"

eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" "$SAMPLE" "$PY")"
echo "MODEL_LOCAL=$MODEL_LOCAL IMAGE_ROOT=$IMAGE_ROOT"

# ── the server ────────────────────────────────────────────────────────
PORT=$((20000 + ${SLURM_JOB_ID} % 20000))
SERVER_LOG=logs/whatami/server-$MODEL-${SLURM_JOB_ID}.log
# Start each server with "$PY -m". Do not call the mirror's bin/vllm: its
# shebang names the NFS python, thus each process then imports torch and vllm
# over NFS (job 70086: the server banner came about 15 min after the start).
if [ "$ENGINE" = sglang ]; then
  # SGLang reads any local path; it has no media allowlist.
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
      "${SERVE[@]}" > "$SERVER_LOG" 2>&1 &
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
curl -s "http://127.0.0.1:$PORT/version" || curl -s "http://127.0.0.1:$PORT/get_server_info" | head -c 400 || true; echo

# ── the requests ──────────────────────────────────────────────────────
ARGS=(--base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --sample "$SAMPLE"
      --image-root "$IMAGE_ROOT" --out-dir "$OUT" --repeats "$REPEATS" --temperature 0.6
      --limit "$LIMIT")
[ -n "$CTK" ] && ARGS+=(--chat-template-kwargs "$CTK")
"$PY" research/scripts/whatami/query.py "${ARGS[@]}" &
wait $!
echo "done $(date -u +%FT%TZ)"
