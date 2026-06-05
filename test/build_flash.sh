#!/bin/bash
#SBATCH -J flash-build
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 6-0
#SBATCH -o /data/delta9043/repos/MemoryAgnostic/logs/slurm-%A.out

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate magno_fix
pip uninstall flash-attn -y
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-binary=:all: --no-cache-dir