#!/usr/bin/env bash
set -euo pipefail

# Run one H1 token-granularity condition end to end.
#
# Required environment:
#   RUN_ID
#   TARGET_TOKEN            object_center or object_bbox_mean
#
# Optional environment:
#   MAX_BBOX_TOKENS         default 8, ignored by object_center
#   GPU                     default 0
#   TRAIN_SIZE              default 128
#   VAL_SIZE                default 32
#   TEST_SIZE               default 64
#   SPLIT_SEED              default 4200
#   TRAIN_SEED              default 4201
#   TEST_SEED               default 4203
#   LAYER_INDEX             default 15
#   EPOCHS                  default 2
#   GRAD_ACCUM              default 8
#   COCO_ROOT               default data/coco2017
#   NUM_INJECTION_TOKENS    default 8
#   INJECTION_SCALE         default 57.75

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID is required" >&2
  exit 2
fi
if [[ -z "${TARGET_TOKEN:-}" ]]; then
  echo "TARGET_TOKEN is required" >&2
  exit 2
fi

MAX_BBOX_TOKENS="${MAX_BBOX_TOKENS:-8}"
GPU="${GPU:-0}"
TRAIN_SIZE="${TRAIN_SIZE:-128}"
VAL_SIZE="${VAL_SIZE:-32}"
TEST_SIZE="${TEST_SIZE:-64}"
SPLIT_SEED="${SPLIT_SEED:-4200}"
TRAIN_SEED="${TRAIN_SEED:-4201}"
TEST_SEED="${TEST_SEED:-4203}"
LAYER_INDEX="${LAYER_INDEX:-15}"
EPOCHS="${EPOCHS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
COCO_ROOT="${COCO_ROOT:-data/coco2017}"
NUM_INJECTION_TOKENS="${NUM_INJECTION_TOKENS:-8}"
INJECTION_SCALE="${INJECTION_SCALE:-57.75}"

RUN_DIR="experiments/runs/${RUN_ID}"
OUT_DIR="outputs/${RUN_ID}"
PARQUET_NAME="qwen3vl_coco_L${LAYER_INDEX}_${TARGET_TOKEN}_av_sft.parquet"
SHORT_PARQUET_NAME="qwen3vl_coco_L${LAYER_INDEX}_${TARGET_TOKEN}_short_av_sft.parquet"

mkdir -p "${OUT_DIR}"
python tools/init_experiment_run.py --run-id "${RUN_ID}" --hypothesis H1 --study A --status running --force

{
  echo "# H1 condition start $(date -Is)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "RUN_ID=${RUN_ID}"
  echo "TARGET_TOKEN=${TARGET_TOKEN}"
  echo "MAX_BBOX_TOKENS=${MAX_BBOX_TOKENS}"
  echo "GPU=${GPU}"
  echo "TRAIN_SIZE=${TRAIN_SIZE}"
  echo "VAL_SIZE=${VAL_SIZE}"
  echo "TEST_SIZE=${TEST_SIZE}"
  echo "SPLIT_SEED=${SPLIT_SEED}"
  echo "TRAIN_SEED=${TRAIN_SEED}"
  echo "TEST_SEED=${TEST_SEED}"
  echo "COCO_ROOT=${COCO_ROOT}"
} >> "${RUN_DIR}/command_log.txt"

python scripts/qwen3vl/build_coco_object_split_manifest.py \
  --coco-root "${COCO_ROOT}" \
  --out "${OUT_DIR}/split_manifest.json" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --test-size "${TEST_SIZE}" \
  --seed "${SPLIT_SEED}" \
  --min-area-frac 0.015

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/extract_qwen3vl_coco_object_tokens.py \
  --coco-root "${COCO_ROOT}" \
  --out-dir "${OUT_DIR}/train_extract" \
  --num-samples "${TRAIN_SIZE}" \
  --seed "${TRAIN_SEED}" \
  --batch-size 2 \
  --layer-index "${LAYER_INDEX}" \
  --target-token "${TARGET_TOKEN}" \
  --max-bbox-tokens "${MAX_BBOX_TOKENS}" \
  --image-ids-json "${OUT_DIR}/split_manifest.json" \
  --image-ids-key train

python scripts/qwen3vl/make_qwen3vl_coco_short_label_parquet.py \
  --src "${OUT_DIR}/train_extract/${PARQUET_NAME}" \
  --out-dir "${OUT_DIR}/train_short_labels"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/extract_qwen3vl_coco_object_tokens.py \
  --coco-root "${COCO_ROOT}" \
  --out-dir "${OUT_DIR}/test_extract" \
  --num-samples "${TEST_SIZE}" \
  --seed "${TEST_SEED}" \
  --batch-size 2 \
  --layer-index "${LAYER_INDEX}" \
  --target-token "${TARGET_TOKEN}" \
  --max-bbox-tokens "${MAX_BBOX_TOKENS}" \
  --image-ids-json "${OUT_DIR}/split_manifest.json" \
  --image-ids-key test

python scripts/qwen3vl/make_qwen3vl_coco_short_label_parquet.py \
  --src "${OUT_DIR}/test_extract/${PARQUET_NAME}" \
  --out-dir "${OUT_DIR}/test_short_labels"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/train_qwen3vl_av_lora_tiny.py \
  --av-parquet "${OUT_DIR}/train_short_labels/${SHORT_PARQUET_NAME}" \
  --out-dir "${OUT_DIR}/adapter" \
  --max-rows "${TRAIN_SIZE}" \
  --epochs "${EPOCHS}" \
  --grad-accum "${GRAD_ACCUM}" \
  --lr 3e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --injection-scale "${INJECTION_SCALE}" \
  --num-injection-tokens "${NUM_INJECTION_TOKENS}" \
  --train-activation-adapter \
  --activation-adapter-lr 1e-4 \
  --contrastive-shuffle-weight 1.0 \
  --contrastive-margin 0.02 \
  --response-contrastive-weight 1.0 \
  --response-contrastive-margin 0.02 \
  --seed "${TRAIN_SEED}"

cp "${OUT_DIR}/adapter/summary.json" "${RUN_DIR}/train_summary.json"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/eval_qwen3vl_av_activation_sensitivity.py \
  --av-parquet "${OUT_DIR}/test_short_labels/${SHORT_PARQUET_NAME}" \
  --adapter "${OUT_DIR}/adapter/adapter" \
  --activation-adapter "${OUT_DIR}/adapter/activation_adapter.pt" \
  --out "${RUN_DIR}/sensitivity.json" \
  --max-rows "${TEST_SIZE}" \
  --batch-size 8 \
  --injection-scale "${INJECTION_SCALE}" \
  --num-injection-tokens "${NUM_INJECTION_TOKENS}" \
  --seed "${TEST_SEED}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/eval_qwen3vl_av_candidate_ranking.py \
  --av-parquet "${OUT_DIR}/test_short_labels/${SHORT_PARQUET_NAME}" \
  --adapter "${OUT_DIR}/adapter/adapter" \
  --activation-adapter "${OUT_DIR}/adapter/activation_adapter.pt" \
  --out "${RUN_DIR}/ranking_nll.json" \
  --max-rows "${TEST_SIZE}" \
  --max-queries "${TEST_SIZE}" \
  --candidate-mode all \
  --score-mode nll \
  --batch-size 8 \
  --injection-scale "${INJECTION_SCALE}" \
  --num-injection-tokens "${NUM_INJECTION_TOKENS}" \
  --seed "${TEST_SEED}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/eval_qwen3vl_av_candidate_ranking.py \
  --av-parquet "${OUT_DIR}/test_short_labels/${SHORT_PARQUET_NAME}" \
  --adapter "${OUT_DIR}/adapter/adapter" \
  --activation-adapter "${OUT_DIR}/adapter/activation_adapter.pt" \
  --out "${RUN_DIR}/ranking_gain.json" \
  --max-rows "${TEST_SIZE}" \
  --max-queries "${TEST_SIZE}" \
  --candidate-mode all \
  --score-mode activation_gain \
  --batch-size 8 \
  --injection-scale "${INJECTION_SCALE}" \
  --num-injection-tokens "${NUM_INJECTION_TOKENS}" \
  --seed "${TEST_SEED}"

cp "${RUN_DIR}/ranking_nll.json" "${RUN_DIR}/ranking.json"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
s = json.loads((run / "sensitivity.json").read_text())
r = json.loads((run / "ranking_nll.json").read_text())
g = json.loads((run / "ranking_gain.json").read_text())
summary = {
    "run_id": "${RUN_ID}",
    "target_token": "${TARGET_TOKEN}",
    "max_bbox_tokens": int("${MAX_BBOX_TOKENS}"),
    "train_size": int("${TRAIN_SIZE}"),
    "test_size": int("${TEST_SIZE}"),
    "matched_mean_nll": s["matched_mean_nll"],
    "shuffled_mean_nll": s["shuffled_mean_nll"],
    "sensitivity_delta": s["shuffled_minus_matched_mean"],
    "matched_better_fraction": s["shuffled_match_better_fraction"],
    "nll_top1": r["top1_accuracy"],
    "nll_top5": r["top5_accuracy"],
    "nll_unique_top1": r["unique_label_top1_accuracy"],
    "nll_unique_top5": r["unique_label_top5_accuracy"],
    "gain_top1": g["top1_accuracy"],
    "gain_top5": g["top5_accuracy"],
    "gain_unique_top1": g["unique_label_top1_accuracy"],
    "gain_unique_top5": g["unique_label_top5_accuracy"],
}
(run / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "# H1 condition finished $(date -Is)" >> "${RUN_DIR}/command_log.txt"
