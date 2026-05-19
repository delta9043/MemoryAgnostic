#!/bin/bash
set -e

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate vllm

MODEL_PATH="${VLLM_MODEL_PATH:?VLLM_MODEL_PATH must be set}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "[start_vllm] Model: $MODEL_PATH"
echo "[start_vllm] Port: $PORT"
echo "[start_vllm] Max model len: $MAX_MODEL_LEN"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$(basename $MODEL_PATH)" \
    --trust-remote-code \
    --generation-config vllm