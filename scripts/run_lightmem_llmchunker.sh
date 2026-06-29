#!/bin/bash
#SBATCH -J lm_llmchunk
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 6-0
#SBATCH -o /data/delta9043/repos/MemoryAgnostic/logs/lightmem/slurm/slurm-%A.out
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]

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

# GPU 분배: 마지막 1장은 실험 코드(임베딩; NoOp이라 LLMLingua 미사용), 나머지는 vLLM
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

mkdir -p ${REPO}/logs/lightmem/{slurm,vllm,output} ${REPO}/results

# vLLM 서버 시작
echo "Starting vLLM server with $VLLM_TP_SIZE GPUs..."
VLLM_MODEL_PATH="$MODEL_PATH" \
VLLM_PORT="$VLLM_PORT" \
VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
VLLM_TP_SIZE="$VLLM_TP_SIZE" \
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
    bash ${REPO}/scripts/common/start_vllm.sh > ${REPO}/logs/lightmem/vllm/vllm-${SLURM_JOB_ID}.out 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up vLLM (PID $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

# 실험 실행 (chunks_qwen32b.json이 미리 생성돼 있어야 함: sbatch scripts/run_llmchunker.sh)
echo ""
echo "========================================"
echo "Running: env=lightmem config=lightmem_llmchunker.yaml"
echo "========================================"

CONDA_ENV="lightmem" \
EXPERIMENT_CONFIG="${REPO}/configs/lightmem_llmchunker.yaml" \
VLLM_BASE_URL="http://localhost:$VLLM_PORT/v1" \
CUDA_VISIBLE_DEVICES="$EXP_GPU" \
LOG_DIR="${REPO}/logs/lightmem/output" \
    bash ${REPO}/scripts/common/run_experiment.sh

echo "[done] lightmem_llmchunker.yaml: $(date)"
echo ""
echo "All experiments finished: $(date)"
exit 0
