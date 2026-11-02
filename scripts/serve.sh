#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
model="${MODEL_NAME:-Qwen3.5-9B}"
test -f "models/$model/download-manifest.json" || { echo 'Run download_model.py first.' >&2; exit 1; }
exec .venv/bin/vllm serve "$PWD/models/$model" \
  --served-model-name "$model" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --limit-mm-per-prompt '{"image":1}' \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
