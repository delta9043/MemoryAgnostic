#!/bin/bash
set -e

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate simplemem

CONFIG="${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG must be set}"
BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"

echo "[run_experiment] Config: $CONFIG"
echo "[run_experiment] vLLM URL: $BASE_URL"

# vLLM 서버 준비 대기 (최대 600초)
echo "[run_experiment] Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "$BASE_URL/models" > /dev/null 2>&1; then
        echo "[run_experiment] vLLM server ready (after $((i * 5))s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "[run_experiment] vLLM server not ready after 600s. Aborting."
        exit 1
    fi
    sleep 5
done

cd /data/delta9043/repos/MemoryAgnostic
exec python main.py --config "$CONFIG"