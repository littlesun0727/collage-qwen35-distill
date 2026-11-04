#!/usr/bin/env bash
# Qwen3.5-4B official non-thinking general-task sampling, for comparison experiments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Set these environment variables before running; use your actual local paths.
: "${MODEL_DIR:?Set MODEL_DIR to the original Qwen3.5-4B directory}"
: "${DATA_DIR:?Set DATA_DIR to the dataset directory containing test.jsonl and images}"
GPU="${GPU:-6}"
LIMIT="${LIMIT:-137}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/preview-official-$(date +%Y%m%d-%H%M%S)}"

args=(
  --model "$MODEL_DIR" --data "$DATA_DIR" --gpu "$GPU"
  --infer-backend vllm --preview-split test --limit "$LIMIT"
  --infer-batch-size "$INFER_BATCH_SIZE"
  --max-length 12288 --image-tokens 1024 --max-new-tokens "$MAX_NEW_TOKENS"
  --vllm-gpu-memory-utilization 0.7
  --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0
  --presence-penalty 1.5 --repetition-penalty 1.0
  --output "$OUTPUT_DIR"
)
if [[ -n "${ADAPTER_DIR:-}" ]]; then
  args+=(--adapter "$ADAPTER_DIR")
else
  args+=(--preview)
fi
if [[ -n "${BASELINE_DIR:-}" ]]; then
  args+=(--compare-to "$BASELINE_DIR")
fi
python -u training/run_lora.py "${args[@]}" "$@"
