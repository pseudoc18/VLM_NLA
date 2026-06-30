#!/usr/bin/env bash
set -euo pipefail

# Run the H4 absent-object hallucination shortcut pilot.
#
# Required environment:
#   RUN_ID
#   AV_PARQUET
#   ADAPTER
#   ACTIVATION_ADAPTER
#
# Optional environment:
#   GPU                     default 0
#   COCO_ROOT               default data/coco2017
#   MAX_IMAGES              default 16
#   ABSENT_PER_IMAGE        default 3
#   MAX_OBJECTS_PER_IMAGE   default 3
#   MAX_HALLUCINATIONS      default 24
#   LAYER_INDEX             default 15
#   MAX_BBOX_TOKENS         default 8
#   NUM_INJECTION_TOKENS    default 8
#   INJECTION_SCALE         default 57.75
#   SEED                    default 4701

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID is required" >&2
  exit 2
fi
if [[ -z "${AV_PARQUET:-}" ]]; then
  echo "AV_PARQUET is required" >&2
  exit 2
fi
if [[ -z "${ADAPTER:-}" ]]; then
  echo "ADAPTER is required" >&2
  exit 2
fi
if [[ -z "${ACTIVATION_ADAPTER:-}" ]]; then
  echo "ACTIVATION_ADAPTER is required" >&2
  exit 2
fi

GPU="${GPU:-0}"
COCO_ROOT="${COCO_ROOT:-data/coco2017}"
MAX_IMAGES="${MAX_IMAGES:-16}"
ABSENT_PER_IMAGE="${ABSENT_PER_IMAGE:-3}"
MAX_OBJECTS_PER_IMAGE="${MAX_OBJECTS_PER_IMAGE:-3}"
MAX_HALLUCINATIONS="${MAX_HALLUCINATIONS:-24}"
LAYER_INDEX="${LAYER_INDEX:-15}"
MAX_BBOX_TOKENS="${MAX_BBOX_TOKENS:-8}"
NUM_INJECTION_TOKENS="${NUM_INJECTION_TOKENS:-8}"
INJECTION_SCALE="${INJECTION_SCALE:-57.75}"
SEED="${SEED:-4701}"

RUN_DIR="experiments/runs/${RUN_ID}"

python tools/init_experiment_run.py --run-id "${RUN_ID}" --hypothesis H4 --study D --status running --force

{
  echo "# H4 hallucination shortcut probe start $(date -Is)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "RUN_ID=${RUN_ID}"
  echo "AV_PARQUET=${AV_PARQUET}"
  echo "ADAPTER=${ADAPTER}"
  echo "ACTIVATION_ADAPTER=${ACTIVATION_ADAPTER}"
  echo "GPU=${GPU}"
  echo "COCO_ROOT=${COCO_ROOT}"
  echo "MAX_IMAGES=${MAX_IMAGES}"
  echo "ABSENT_PER_IMAGE=${ABSENT_PER_IMAGE}"
  echo "MAX_OBJECTS_PER_IMAGE=${MAX_OBJECTS_PER_IMAGE}"
  echo "MAX_HALLUCINATIONS=${MAX_HALLUCINATIONS}"
  echo "LAYER_INDEX=${LAYER_INDEX}"
  echo "MAX_BBOX_TOKENS=${MAX_BBOX_TOKENS}"
  echo "NUM_INJECTION_TOKENS=${NUM_INJECTION_TOKENS}"
  echo "INJECTION_SCALE=${INJECTION_SCALE}"
  echo "SEED=${SEED}"
} >> "${RUN_DIR}/command_log.txt"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/eval_qwen3vl_hallucination_shortcut_probe.py \
  --av-parquet "${AV_PARQUET}" \
  --adapter "${ADAPTER}" \
  --activation-adapter "${ACTIVATION_ADAPTER}" \
  --coco-root "${COCO_ROOT}" \
  --out "${RUN_DIR}/hallucination_shortcut_probe.json" \
  --max-images "${MAX_IMAGES}" \
  --absent-per-image "${ABSENT_PER_IMAGE}" \
  --max-objects-per-image "${MAX_OBJECTS_PER_IMAGE}" \
  --max-hallucinations "${MAX_HALLUCINATIONS}" \
  --layer-index "${LAYER_INDEX}" \
  --max-bbox-tokens "${MAX_BBOX_TOKENS}" \
  --batch-size 2 \
  --score-batch-size 8 \
  --injection-scale "${INJECTION_SCALE}" \
  --num-injection-tokens "${NUM_INJECTION_TOKENS}" \
  --seed "${SEED}"

cp "${RUN_DIR}/hallucination_shortcut_probe.json" "${RUN_DIR}/semantic_eval.json"
echo "# H4 hallucination shortcut probe finished $(date -Is)" >> "${RUN_DIR}/command_log.txt"
