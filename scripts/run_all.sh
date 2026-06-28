#!/bin/bash
#SBATCH -J memoryagnostic
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 6-0
#SBATCH -o /data/delta9043/repos/MemoryAgnostic/logs/simplemem/slurm/slurm-%A.out
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]

set -e

echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

source /data/delta9043/anaconda3/etc/profile.d/conda.sh

# 설정
REPO=/data/delta9043/repos/MemoryAgnostic
MODEL_PATH="/data/delta9043/models/Qwen3-14B"
VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))
VLLM_MAX_MODEL_LEN=8192
VLLM_TP_SIZE=2
EXPERIMENT_CONFIG="configs/test_simplemem.yaml"
CONDA_ENV="simplemem"

mkdir -p ${REPO}/logs/simplemem/{slurm,vllm,output} ${REPO}/results

echo "vLLM Port: $VLLM_PORT"

# vLLM 서버 시작 (할당된 GPU 전부 사용, util=0.7로 embedding 공간 확보)
echo "Starting vLLM server (tp=$VLLM_TP_SIZE, util=0.7)..."
VLLM_MODEL_PATH="$MODEL_PATH" \
VLLM_PORT="$VLLM_PORT" \
VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
VLLM_TP_SIZE="$VLLM_TP_SIZE" \
    bash ${REPO}/scripts/common/start_vllm.sh > ${REPO}/logs/simplemem/vllm/vllm-${SLURM_JOB_ID}.out 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up vLLM (PID $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

# 실험 실행 (CUDA_VISIBLE_DEVICES 그대로 → embedding이 남은 공간 사용)
echo "Starting experiment..."
CONDA_ENV="$CONDA_ENV" \
EXPERIMENT_CONFIG="${REPO}/$EXPERIMENT_CONFIG" \
VLLM_BASE_URL="http://localhost:$VLLM_PORT/v1" \
LOG_DIR="${REPO}/logs/simplemem/output" \
    bash -x ${REPO}/scripts/common/run_experiment.sh

echo "Experiment finished: $(date)"
exit 0