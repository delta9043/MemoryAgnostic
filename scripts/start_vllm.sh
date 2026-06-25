#!/bin/bash
set -e

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate vllm

MODEL_PATH="${VLLM_MODEL_PATH:?VLLM_MODEL_PATH must be set}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
TP_SIZE="${VLLM_TP_SIZE:-1}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

echo "[start_vllm] Model: $MODEL_PATH"
echo "[start_vllm] Port: $PORT"
echo "[start_vllm] Max model len: $MAX_MODEL_LEN"
echo "[start_vllm] Tensor parallel size: $TP_SIZE"
echo "[start_vllm] GPU memory utilization: $GPU_UTIL"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --served-model-name "$MODEL_PATH" \
    --trust-remote-code \
    --generation-config vllm \
    --override-generation-config '{"enable_thinking": false}'
