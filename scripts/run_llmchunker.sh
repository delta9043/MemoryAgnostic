#!/bin/bash
#SBATCH --job-name=chunk
#SBATCH --partition=batch_ce_ugrad
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=6-00:00:00
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]
#SBATCH --output=/data/delta9043/repos/MemoryAgnostic/logs/precompute/slurm/slurm-%j.out

MODEL_PATH="${1:-/data/delta9043/models/Qwen3-32B}"

echo "=============================="
echo "JOB: chunk | MODEL: $MODEL_PATH"
echo "STARTED: $(date)"
echo "=============================="

cd /data/delta9043/repos/MemoryAgnostic
mkdir -p logs/precompute/slurm data/chunked_data

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate magno

python run_chunker.py \
    --model_path "$MODEL_PATH" \
    --output data/chunked_data/chunks_qwen32b.json

echo "=============================="
echo "FINISHED: $(date)"
echo "=============================="
