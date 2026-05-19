#!/bin/bash
#SBATCH -J memoryagnostic
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out

set -e

echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

source /data/delta9043/anaconda3/etc/profile.d/conda.sh

# 설정
MODEL_PATH="/data/delta9043/models/Qwen3-8B"
VLLM_PORT=8000
VLLM_GPU_COUNT=1
VLLM_MAX_MODEL_LEN=8192
EXPERIMENT_CONFIG="configs/baseline_fixed_locomo10.yaml"

# GPU 분배
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
if [ ${#GPU_ARRAY[@]} -lt $((VLLM_GPU_COUNT + 1)) ]; then
    echo "ERROR: Need at least $((VLLM_GPU_COUNT + 1)) GPUs, got ${#GPU_ARRAY[@]}"
    exit 1
fi

VLLM_GPUS=$(IFS=','; echo "${GPU_ARRAY[*]:0:$VLLM_GPU_COUNT}")
EXP_GPUS=$(IFS=','; echo "${GPU_ARRAY[*]:$VLLM_GPU_COUNT}")

echo "vLLM GPUs: $VLLM_GPUS"
echo "Experiment GPUs: $EXP_GPUS"

mkdir -p logs results

# vLLM 서버 시작 (백그라운드)
echo "Starting vLLM server..."
VLLM_MODEL_PATH="$MODEL_PATH" \
VLLM_PORT="$VLLM_PORT" \
VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    bash scripts/start_vllm.sh > logs/vllm-${SLURM_JOB_ID}.out 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up vLLM (PID $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

# 실험 실행
echo "Starting experiment..."
EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
VLLM_BASE_URL="http://localhost:$VLLM_PORT/v1" \
CUDA_VISIBLE_DEVICES="$EXP_GPUS" \
    bash scripts/run_experiment.sh

echo "Experiment finished: $(date)"
exit 0