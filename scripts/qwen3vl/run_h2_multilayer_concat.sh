#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate a true multi-layer concat AV condition for H2.
#
# Required environment:
#   RUN_ID
#   TRAIN_PARQUETS    space-separated layer parquets, same sample ids/order not required
#   TEST_PARQUETS     space-separated layer parquets, same sample ids/order not required
#
# Optional environment:
#   GPU                     default 0
#   TRAIN_SIZE              default 128
#   TEST_SIZE               default 64
#   EPOCHS                  default 2
#   GRAD_ACCUM              default 8
#   NUM_INJECTION_TOKENS    default 8
#   INJECTION_SCALE         default 57.75
#   LAYER_TAG               default L10-L15-L20
#   SEED                    default 4801

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID is required" >&2
  exit 2
fi
if [[ -z "${TRAIN_PARQUETS:-}" ]]; then
  echo "TRAIN_PARQUETS is required" >&2
  exit 2
fi
if [[ -z "${TEST_PARQUETS:-}" ]]; then
  echo "TEST_PARQUETS is required" >&2
  exit 2
fi

GPU="${GPU:-0}"
TRAIN_SIZE="${TRAIN_SIZE:-128}"
TEST_SIZE="${TEST_SIZE:-64}"
EPOCHS="${EPOCHS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_INJECTION_TOKENS="${NUM_INJECTION_TOKENS:-8}"
INJECTION_SCALE="${INJECTION_SCALE:-57.75}"
LAYER_TAG="${LAYER_TAG:-L10-L15-L20}"
SEED="${SEED:-4801}"

RUN_DIR="experiments/runs/${RUN_ID}"
OUT_DIR="outputs/${RUN_ID}"
PARQUET_NAME="qwen3vl_coco_${LAYER_TAG}_object_bbox_mean_multilayer_short_av_sft.parquet"

read -r -a TRAIN_PARQUET_ARRAY <<< "${TRAIN_PARQUETS}"
read -r -a TEST_PARQUET_ARRAY <<< "${TEST_PARQUETS}"

mkdir -p "${OUT_DIR}"
python tools/init_experiment_run.py --run-id "${RUN_ID}" --hypothesis H2 --study B --status running --force

{
  echo "# H2 multi-layer concat start $(date -Is)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "RUN_ID=${RUN_ID}"
  echo "TRAIN_PARQUETS=${TRAIN_PARQUETS}"
  echo "TEST_PARQUETS=${TEST_PARQUETS}"
  echo "GPU=${GPU}"
  echo "TRAIN_SIZE=${TRAIN_SIZE}"
  echo "TEST_SIZE=${TEST_SIZE}"
  echo "EPOCHS=${EPOCHS}"
  echo "GRAD_ACCUM=${GRAD_ACCUM}"
  echo "NUM_INJECTION_TOKENS=${NUM_INJECTION_TOKENS}"
  echo "INJECTION_SCALE=${INJECTION_SCALE}"
  echo "LAYER_TAG=${LAYER_TAG}"
  echo "SEED=${SEED}"
} >> "${RUN_DIR}/command_log.txt"

python scripts/qwen3vl/make_qwen3vl_multilayer_parquet.py \
  --src "${TRAIN_PARQUET_ARRAY[@]}" \
  --out-dir "${OUT_DIR}/train_multilayer" \
  --out-name "${PARQUET_NAME}"

python scripts/qwen3vl/make_qwen3vl_multilayer_parquet.py \
  --src "${TEST_PARQUET_ARRAY[@]}" \
  --out-dir "${OUT_DIR}/test_multilayer" \
  --out-name "${PARQUET_NAME}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/train_qwen3vl_av_lora_tiny.py \
  --av-parquet "${OUT_DIR}/train_multilayer/${PARQUET_NAME}" \
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
  --seed "${SEED}"

cp "${OUT_DIR}/adapter/summary.json" "${RUN_DIR}/train_summary.json"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/eval_qwen3vl_av_activation_sensitivity.py \
  --av-parquet "${OUT_DIR}/test_multilayer/${PARQUET_NAME}" \
  --adapter "${OUT_DIR}/adapter/adapter" \
  --activation-adapter "${OUT_DIR}/adapter/activation_adapter.pt" \
  --out "${RUN_DIR}/sensitivity.json" \
  --max-rows "${TEST_SIZE}" \
  --batch-size 8 \
  --injection-scale "${INJECTION_SCALE}" \
  --num-injection-tokens "${NUM_INJECTION_TOKENS}" \
  --seed "${SEED}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/eval_qwen3vl_av_candidate_ranking.py \
  --av-parquet "${OUT_DIR}/test_multilayer/${PARQUET_NAME}" \
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
  --seed "${SEED}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/qwen3vl/eval_qwen3vl_av_candidate_ranking.py \
  --av-parquet "${OUT_DIR}/test_multilayer/${PARQUET_NAME}" \
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
  --seed "${SEED}"

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
    "layer_tag": "${LAYER_TAG}",
    "train_size": int("${TRAIN_SIZE}"),
    "test_size": int("${TEST_SIZE}"),
    "num_injection_tokens": int("${NUM_INJECTION_TOKENS}"),
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

echo "# H2 multi-layer concat finished $(date -Is)" >> "${RUN_DIR}/command_log.txt"
