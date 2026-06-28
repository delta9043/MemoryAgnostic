#!/bin/bash
#SBATCH --job-name=filter-32b
#SBATCH --partition=batch_ce_ugrad
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=6-00:00:00
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]
#SBATCH --output=/data/delta9043/repos/MemoryAgnostic/logs/precompute/slurm/slurm-%j.out

echo "=============================="
echo "JOB: filter-32b | MODEL: Qwen3-32B"
echo "STARTED: $(date)"
echo "=============================="

cd /data/delta9043/repos/MemoryAgnostic
mkdir -p logs/precompute/slurm data/filtered_data

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate magno

python run_filter.py \
    --model_path /data/delta9043/models/Qwen3-32B \
    --output data/filtered_data/filtered_32b.json

echo "=============================="
echo "FINISHED: $(date)"
echo "=============================="