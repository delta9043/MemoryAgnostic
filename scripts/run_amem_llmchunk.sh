#!/bin/bash
#SBATCH -J am_llmchunk
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 6-0
#SBATCH -o /data/delta9043/repos/MemoryAgnostic/logs/slurm-%A.out
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]

# A-Mem + Qwen3-14B + LLMChunker(PrecomputedChunker). chunks.json 필요(먼저 청킹).
# 독립 재실행 가능. chunks 파일이 없으면 PrecomputedChunker가 명확히 에러낸다.

set -e

echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

source /data/delta9043/anaconda3/etc/profile.d/conda.sh

# ============ 설정 ============
REPO=/data/delta9043/repos/MemoryAgnostic
MODEL_PATH="/data/delta9043/models/Qwen3-14B"
VLLM_PORT=$((8000 + (SLURM_JOB_ID % 100) * 10))
VLLM_MAX_MODEL_LEN=8192

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS=${#GPU_ARRAY[@]}
VLLM_GPU_COUNT=$((N_GPUS - 1))
VLLM_GPUS=$(IFS=','; echo "${GPU_ARRAY[*]:0:$VLLM_GPU_COUNT}")
EXP_GPU="${GPU_ARRAY[$VLLM_GPU_COUNT]}"
VLLM_TP_SIZE=$VLLM_GPU_COUNT

echo "vLLM GPUs: $VLLM_GPUS (tp=$VLLM_TP_SIZE)"
echo "Experiment GPU: $EXP_GPU"
echo "vLLM Port: $VLLM_PORT"
# =============================

mkdir -p ${REPO}/logs ${REPO}/results

echo "Starting vLLM server with $VLLM_TP_SIZE GPUs..."
VLLM_MODEL_PATH="$MODEL_PATH" \
VLLM_PORT="$VLLM_PORT" \
VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
VLLM_TP_SIZE="$VLLM_TP_SIZE" \
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    bash ${REPO}/scripts/start_vllm.sh > ${REPO}/logs/vllm-${SLURM_JOB_ID}.out 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up vLLM (PID $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

echo ""
echo "========================================"
echo "Running: env=a-mem config=amem_llmchunker.yaml"
echo "========================================"

CONDA_ENV="a-mem" \
EXPERIMENT_CONFIG="${REPO}/configs/amem_llmchunker.yaml" \
VLLM_BASE_URL="http://localhost:$VLLM_PORT/v1" \
CUDA_VISIBLE_DEVICES="$EXP_GPU" \
    bash ${REPO}/scripts/run_experiment.sh

echo "[done] amem_llmchunker.yaml: $(date)"
exit 0
