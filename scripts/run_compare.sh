#!/bin/bash
#SBATCH -J llmchunk_compare
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 0-1
#SBATCH -o /data/delta9043/repos/MemoryAgnostic/logs/slurm-%A.out
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]

# 4조건 결과(results/*.json)를 모아 NoChunker vs LLMChunker 비교표를 출력.
# GPU 불필요. 4조건이 모두 끝난 뒤 한 번 실행한다(개별 실행 스크립트와 분리).
# 로그인 노드에서 그냥 `python compare_results.py`로 돌려도 동일.

set -e

source /data/delta9043/anaconda3/etc/profile.d/conda.sh

REPO=/data/delta9043/repos/MemoryAgnostic
COMPARE_ENV="${COMPARE_ENV:-simplemem}"

conda activate "$COMPARE_ENV"
cd ${REPO}
python compare_results.py --results_dir results
