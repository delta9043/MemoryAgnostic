#!/bin/bash
# LLMChunker 실험 총괄 제출 스크립트 (로그인 노드에서 실행):
#   청킹 1회 → 4조건 → 표 를 SLURM 의존성(--dependency)으로 순서 보장하며 제출한다.
#
#   사용법:
#       bash scripts/run_all_llmchunk.sh                       # 청킹 모델 기본(Qwen3-32B)
#       bash scripts/run_all_llmchunk.sh /data/.../Qwen3-32B   # 청킹 모델 지정
#
# 각 잡은 자기 vLLM을 띄우는 독립 SLURM 잡이라, 개별 스크립트로 따로 제출해도 된다:
#       sbatch scripts/run_llmchunker.sh
#       sbatch scripts/run_amem_nochunk.sh   ...
#
# 의존성 구조:
#   chunk ──▶ amem_llmchunk ─┐
#         └─▶ simplemem_llmchunk ─┤
#   amem_nochunk ───────────────┤   (nochunk은 chunk 불필요 → 병렬 실행)
#   simplemem_nochunk ──────────┤
#                               └─▶ compare (4조건 모두 afterok)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHUNK_MODEL="${1:-}"   # 비어있으면 run_llmchunker.sh 기본(Qwen3-32B) 사용

echo "[orchestrator] Submitting jobs..."

# 1) 청킹 (가장 무거움, 1회)
if [ -n "$CHUNK_MODEL" ]; then
    JID_CHUNK=$(sbatch --parsable "${SCRIPT_DIR}/run_llmchunker.sh" "$CHUNK_MODEL")
else
    JID_CHUNK=$(sbatch --parsable "${SCRIPT_DIR}/run_llmchunker.sh")
fi
echo "[orchestrator] chunk job:            $JID_CHUNK"

# 2) nochunk 2조건 (chunks.json 불필요 → 의존성 없이 바로 제출, 병렬)
JID_AN=$(sbatch --parsable "${SCRIPT_DIR}/run_amem_nochunk.sh")
echo "[orchestrator] amem_nochunk:         $JID_AN"
JID_SN=$(sbatch --parsable "${SCRIPT_DIR}/run_simplemem_nochunk.sh")
echo "[orchestrator] simplemem_nochunk:    $JID_SN"

# 3) llmchunk 2조건 (chunk 완료 후)
JID_AL=$(sbatch --parsable --dependency=afterok:${JID_CHUNK} "${SCRIPT_DIR}/run_amem_llmchunk.sh")
echo "[orchestrator] amem_llmchunk:        $JID_AL  (after chunk)"
JID_SL=$(sbatch --parsable --dependency=afterok:${JID_CHUNK} "${SCRIPT_DIR}/run_simplemem_llmchunk.sh")
echo "[orchestrator] simplemem_llmchunk:   $JID_SL  (after chunk)"

# 4) 표 (4조건 모두 완료 후)
JID_CMP=$(sbatch --parsable \
    --dependency=afterok:${JID_AN}:${JID_SN}:${JID_AL}:${JID_SL} \
    "${SCRIPT_DIR}/run_compare.sh")
echo "[orchestrator] compare table:        $JID_CMP  (after all 4 conditions)"

echo "[orchestrator] Done. Watch with: squeue -u \$USER"
