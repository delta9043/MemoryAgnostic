#!/bin/bash
set -e

source /data/delta9043/anaconda3/etc/profile.d/conda.sh

CONDA_ENV="${CONDA_ENV:-simplemem}"
CONFIG="${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG must be set}"
BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
# output 로그 디렉토리. 호출 스크립트가 backend별 경로(.../logs/{backend}/output)를
# 넘겨주면 그걸 쓰고, 없으면 기존 평면 logs/로 폴백한다.
LOG_DIR="${LOG_DIR:-${REPO:-/data/delta9043/repos/MemoryAgnostic}/logs}"

conda activate "$CONDA_ENV"
mkdir -p "$LOG_DIR"

echo "[run_experiment] Conda env: $CONDA_ENV"
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
echo "[run_experiment] Starting python main.py..."
echo "[run_experiment] Python log: ${LOG_DIR}/output-${SLURM_JOB_ID}.out"

PYTHONUNBUFFERED=1 python main.py --config "$CONFIG" \
    > "${LOG_DIR}/output-${SLURM_JOB_ID}.out" 2>&1

echo "[run_experiment] python main.py finished"