#!/usr/bin/env bash
# Launch the 100,000-pair TEXT-FIRST battery.
#
# The same 7 cases, the same 2 raters, the same CANONICAL prompts and the same
# 100,000 pairs as the no-city battery. One thing changes: the prompt text
# comes BEFORE the images.
#
#   million_textfirst_qwen9b    10 shards for each case, 70 jobs
#   million_textfirst_gemma12b   5 shards for each case, 35 jobs
#
# ┌─ THE 2 RATERS TAKE 2 DIFFERENT LAYOUTS, ON PURPOSE ────────────────────────┐
# │                                                                            │
# │ qwen3.5-9b    production  images_then_text   image A, image B, prompt      │
# │               this arm    text_first         prompt, image A, image B      │
# │                                                                            │
# │ gemma-4-12b   production  images_then_labels                               │
# │                             image A, "Image A.", image B, "Image B.",      │
# │                             prompt                                         │
# │               this arm    text_first_labels                                │
# │                             prompt, image A, "Image A.", image B,          │
# │                             "Image B."                                     │
# │                                                                            │
# │ Each rater makes ONE change against ITS production layout: the prompt      │
# │ moves to the front.                                                        │
# │                                                                            │
# │ Warning: gemma-4-12b does NOT take plain `text_first`. That layout puts    │
# │ the 2 images next to each other, and this arch then reads them as 1 image  │
# │ (55 of 70 traces, NotSure 81.4%, measured 2026-08-20). The anchors of      │
# │ `text_first_labels` keep the images apart, thus the binding holds and the  │
# │ arm measures the prompt position and nothing else.                        │
# └────────────────────────────────────────────────────────────────────────────┘
#
# ┌─ THE PAIRS ARE THE SAME PAIRS ─────────────────────────────────────────────┐
# │ `pair_seed` is 777 and `max_pairs` is 100,000, thus this script reads the  │
# │ SAME `pair_tables_100k` directory as the no-city battery. The draw at that │
# │ seed is NESTED in the 1,000,000-pair draw, thus a row here joins a row of  │
# │ the registered battery.                                                    │
# │                                                                            │
# │ Warning: join on `pair_id`, not on `canonical_pair_id`. `pair_id` carries  │
# │ the presentation order.                                                    │
# └────────────────────────────────────────────────────────────────────────────┘
#
# ┌─ WALLTIME ─────────────────────────────────────────────────────────────────┐
# │ The launcher default is 180 minutes and the first battery lost 291 shards  │
# │ to it: SLURM does NOT requeue a walltime expiry. `--requeue` covers a      │
# │ preemption and a node failure only. The defaults below come from the       │
# │ measured shard times of that battery.                                      │
# └────────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash research/scripts/launch_textfirst_battery.sh --dry-run
#   bash research/scripts/launch_textfirst_battery.sh --smoke        # 1 case, 2 shards
#   bash research/scripts/launch_textfirst_battery.sh                # both raters
#   bash research/scripts/launch_textfirst_battery.sh --rater qwen
#
# When it lands:
#   python scripts/merge_pairwise_shards.py --sweep-dir <base>/qwen/runs \
#       --sweep million_textfirst_qwen9b --out-dir multirun/textfirst_merged
#
# See `vlm-narratives-docs/million-pair-battery.md`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"

DRY=0
SMOKE=0
LAUNCHER="slurm_gpu_preempt"
RATERS=()
P_QWEN=160
P_GEMMA=80
# Measured on the 2026-08-18 battery: a qwen shard needs 3.2 to 5.3 hours and a
# gemma shard 3.2 to 6.4 hours. Ask for more than the longest, because the wall
# is terminal and a resumed shard still pays the cold import.
T_QWEN=360
T_GEMMA=480
PREBUILD=1
# A SEPARATE directory from `pair_tables_1m`, on purpose. This battery draws
# 100,000 pairs and that one drew 1,000,000, thus the fingerprints differ. If
# both wrote to 1 directory the second run would REDRAW over the first, and it
# would destroy the tables a running battery reads.
PAIRS_DIR="${TEXTFIRST_PAIRS_DIR:-${ROOT}/multirun/pair_tables_100k}"

while [ $# -gt 0 ]; do
  case "$1" in
    --rater)          RATERS+=("$2"); shift 2 ;;
    --parallel-qwen)  P_QWEN="$2"; shift 2 ;;
    --parallel-gemma) P_GEMMA="$2"; shift 2 ;;
    --parallel)       P_QWEN="$2"; P_GEMMA="$2"; shift 2 ;;
    --timeout-qwen)   T_QWEN="$2"; shift 2 ;;
    --timeout-gemma)  T_GEMMA="$2"; shift 2 ;;
    --pierson)        LAUNCHER="slurm_gpu_1x"; shift ;;
    --pairs-dir)      PAIRS_DIR="$2"; shift 2 ;;
    --no-prebuild)    PREBUILD=0; shift ;;
    --smoke)          SMOKE=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    -h|--help)        sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ ${#RATERS[@]} -eq 0 ] && RATERS=(qwen gemma)

cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${ROOT}/multirun/textfirst_battery_${STAMP}"
mkdir -p "${BASE}"

# A smoke run overrides max_pairs, thus the prebuilt tables do not match it.
[ "${SMOKE}" -eq 1 ] && PREBUILD=0

if [ "${PREBUILD}" -eq 1 ]; then
  echo "[textfirst] pairs   : ${PAIRS_DIR}"
  # Idempotent. The FIRST run builds all 7 tables at 100,000 pairs; a later run
  # reports `kept`. The draw matches `looks_proxy_*`, which took the same seed
  # and the same `max_pairs`.
  "${PY}" scripts/prebuild_pair_tables.py \
      --sweep million_textfirst_qwen9b --out-dir "${PAIRS_DIR}" \
      | grep -E "^\[prebuild\] (built|kept|done|FAILED)"
  echo
fi

echo "[textfirst] launcher: ${LAUNCHER}"
echo "[textfirst] base    : ${BASE}"
[ "${SMOKE}" -eq 1 ] && echo "[textfirst] SMOKE   : 1 case, 2 shards, 2,000 pairs"
echo

launch_one() {
  local rater="$1" sweep="$2" parallel="$3" timeout="$4"
  local -a cmd=("${PY}" -m dagspaces.pairwisevqa.submit_shards
                "--sweep=${sweep}"
                "--base-dir=${BASE}/${rater}"
                "--launcher=${LAUNCHER}"
                "--parallel=${parallel}"
                "--timeout-min=${timeout}")

  [ "${PREBUILD}" -eq 1 ] && cmd+=("--pairs-dir=${PAIRS_DIR}")

  if [ "${SMOKE}" -eq 1 ]; then
    cmd+=("--cases" "pairwise_schools_mvp" "--shards" "2"
          "--set" "pair_sampler.max_pairs=2000")
  fi
  [ "${DRY}" -eq 1 ] && cmd+=("--dry-run")

  echo "=== ${rater} (${sweep}) ==="
  "${cmd[@]}"
  echo
}

for rater in "${RATERS[@]}"; do
  case "${rater}" in
    qwen)  launch_one qwen  million_textfirst_qwen9b   "${P_QWEN}"  "${T_QWEN}" ;;
    gemma) launch_one gemma million_textfirst_gemma12b "${P_GEMMA}" "${T_GEMMA}" ;;
    *) echo "unknown rater: ${rater} (use qwen or gemma)" >&2; exit 1 ;;
  esac
done

[ "${DRY}" -eq 1 ] && exit 0

cat <<TXT
To watch:
  squeue -u \$USER -r -h -t RUNNING -n matt-PAIRVQA-shard | wc -l

Check that each rater took ITS layout, and that nothing else moved:
  ${PY} - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from dagspaces.pairwisevqa.submit_shards import compose_cfg
for sweep, model, want in [
    ("million_textfirst_qwen9b", "qwen3.5-9b/instruct", "text_first"),
    ("million_textfirst_gemma12b", "gemma-4-12b/instruct", "text_first_labels"),
]:
    c = compose_cfg(sweep, model, "pairwise_schools_mvp", 0, "/tmp/x", [])
    got = c.prompt.image_layout
    print(f"{model:22s} layout={got:18s} {'OK' if got == want else 'WRONG'}")
    print(f"{'':22s} names the city : {'New York' in c.prompt.user_template}")
    print(f"{'':22s} may abstain    : {c.prompt.structured_output.allow_not_sure}")
PYEOF

Finished shards:
  ls -d ${BASE}/*/runs/*/outputs/pairwise/*_mvp_*.parquet 2>/dev/null | wc -l

Run the missing shards again (the wall is terminal, SLURM does not requeue it):
  ${PY} -m dagspaces.pairwisevqa.submit_shards --sweep million_textfirst_qwen9b \\
      --base-dir ${BASE}/qwen --pairs-dir ${PAIRS_DIR} --only-missing --timeout-min 360

When it lands:
  ${PY} scripts/merge_pairwise_shards.py --sweep-dir ${BASE}/qwen/runs \\
      --sweep million_textfirst_qwen9b --out-dir ${ROOT}/multirun/textfirst_merged
TXT
