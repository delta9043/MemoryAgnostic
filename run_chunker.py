"""
run_chunker.py — LLMChunker를 전체 LoCoMo10 샘플에 한 번만 적용하고 결과를 JSON으로 저장.

청크를 backend마다 다시 만들지 않기 위한 precompute 단계
저장된 JSON은 PrecomputedChunker가 로드하여 A-Mem/SimpleMem 양쪽에서 동일하게 사용한다.

모델은 LLMChunker가 transformers로 프로세스 안에 직접 로드한다.
run_filter.py와 동일한 단순 구조.

사용법:
    python run_chunker.py \
        --model_path /data/delta9043/models/Qwen3-32B \
        --output data/chunked_data/chunks_qwen32b.json

출력 형식:
    [
        {
            "sample_id": "conv-26",
            "chunks": [
                {
                    "chunk_id": 0,
                    "text": "...",
                    "metadata": {...},
                    "turns": [
                        {"turn_id": "...", "speaker": "...", "content": "...",
                         "timestamp": "...", "session_id": "...", "metadata": {...}},
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
"""

import argparse
import json
import os

from data.locomo_loader import load_locomo10_all
from core.chunker.llm_chunker import LLMChunker


LOCOMO_PATH = "/data/delta9043/datasets/locomo/locomo10.json"


def _turn_to_dict(turn) -> dict:
    return {
        "turn_id": turn.turn_id,
        "speaker": turn.speaker,
        "content": turn.content,
        "timestamp": turn.timestamp,
        "session_id": turn.session_id,
        "metadata": dict(turn.metadata),
    }


def run_chunker(model_path: str, output_path: str) -> None:
    # 1. 데이터 로드
    print(f"[run_chunker] 데이터 로드 중: {LOCOMO_PATH}", flush=True)
    samples = load_locomo10_all(LOCOMO_PATH)
    print(f"[run_chunker] {len(samples)}개 샘플 로드 완료", flush=True)

    # 2. LLMChunker 초기화 (모델을 transformers로 in-process 로드)
    #    세션 단위 batch 방식
    print(f"[run_chunker] 모델 로드 중: {model_path}", flush=True)
    chunker = LLMChunker(model_path=model_path)
    print("[run_chunker] 모델 로드 완료", flush=True)

    # 3. 전체 샘플 청킹
    results = []
    total = len(samples)
    for idx, sample in enumerate(samples):
        print(f"[run_chunker] Sample {idx+1}/{total} ({sample.sample_id}) | turns={len(sample.turns)}", flush=True)
        chunks = chunker.chunk(sample.turns)
        results.append({
            "sample_id": sample.sample_id,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "metadata": c.metadata,
                    "turns": [_turn_to_dict(t) for t in c.turns],
                }
                for c in chunks
            ],
        })
        report = chunker.get_failure_report()
        print(
            f"[run_chunker] {sample.sample_id} 완료 | chunks={len(chunks)} | "
            f"session_failures(누적)={report['failure_count']}",
            flush=True,
        )

    # 4. 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[run_chunker] 저장 완료: {output_path}", flush=True)

    # 5. 최종 통계
    report = chunker.get_failure_report()
    total_chunks = sum(len(r["chunks"]) for r in results)
    print(f"[run_chunker] 총 chunk 수: {total_chunks}", flush=True)
    print(f"[run_chunker] 세션 파싱/호출 실패: {report['failure_count']}건", flush=True)
    if report["failed_sessions"]:
        print(f"[run_chunker] 실패 세션(session_id): {report['failed_sessions']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="청킹 모델 경로 (예: /data/delta9043/models/Qwen3-32B)")
    parser.add_argument("--output", type=str, default="data/chunked_data/chunks_qwen32b.json",
                        help="청킹 결과 저장 경로")
    args = parser.parse_args()
    run_chunker(args.model_path, args.output)


if __name__ == "__main__":
    main()
