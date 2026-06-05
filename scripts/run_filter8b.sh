#!/bin/bash
#SBATCH --job-name=filter-8b
#SBATCH --partition=batch_ce_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=6-00:00:00
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]
#SBATCH --output=/data/delta9043/repos/MemoryAgnostic/logs/slurm-%j.out

echo "=============================="
echo "JOB: filter-8b | MODEL: Qwen3-8B"
echo "STARTED: $(date)"
echo "=============================="

cd /data/delta9043/repos/MemoryAgnostic
mkdir -p logs data/filtered_data

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate magno

python run_filter.py \
    --model_path /data/delta9043/models/Qwen3-8B \
    --output data/filtered_data/filtered_8b.json

echo "=============================="
echo "FINISHED: $(date)"
echo "=============================="