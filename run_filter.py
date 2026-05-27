"""
run_filter.py — LLMFilter를 전체 LoCoMo10 샘플에 적용하고 결과를 JSON으로 저장.

사용법:
    python run_filter.py \
        --model_path /data/delta9043/models/Qwen3-32B \
        --output data/filtered_data/filtered_32b.json

출력 형식:
    [
        {
            "sample_id": "conv-26",
            "turns": [
                {"turn_id": "D1:1", "speaker": "A", "content": "filtered..."},
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
from core.filter.llm_filter import LLMFilter


LOCOMO_PATH = "/data/delta9043/datasets/locomo/locomo10.json"


def run_filter(model_path: str, output_path: str) -> None:
    # 1. 데이터 로드
    print(f"[run_filter] 데이터 로드 중: {LOCOMO_PATH}", flush=True)
    samples = load_locomo10_all(LOCOMO_PATH)
    print(f"[run_filter] {len(samples)}개 샘플 로드 완료", flush=True)

    # 2. LLMFilter 초기화 (모델 로드)
    print(f"[run_filter] 모델 로드 중: {model_path}", flush=True)
    llm_filter = LLMFilter(
        model_path=model_path,
        use_flash_attention=False,
        attn_implementation="sdpa",
    )
    print("[run_filter] 모델 로드 완료", flush=True)

    # 3. 전체 샘플에 필터링 적용
    results = []
    total_samples = len(samples)
    for idx, sample in enumerate(samples):
        print(f"[run_filter] Sample {idx+1}/{total_samples} ({sample.sample_id}) 시작", flush=True)
        filtered_turns = llm_filter.run(sample.turns)
        results.append({
            "sample_id": sample.sample_id,
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "speaker": turn.speaker,
                    "content": turn.content,
                    "timestamp": turn.timestamp,
                    "session_id": turn.session_id,
                }
                for turn in filtered_turns
            ]
        })
        print(
            f"[run_filter] {sample.sample_id} 완료 | "
            f"turns={len(filtered_turns)} | "
            f"failures={llm_filter.failure_count}",
            flush=True,
        )

    # 4. 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[run_filter] 저장 완료: {output_path}", flush=True)

    # 5. 최종 실패 통계
    report = llm_filter.get_failure_report()
    print(f"[run_filter] 전체 실패: {report['failure_count']}건", flush=True)
    if report["failed_turn_ids"]:
        print(f"[run_filter] 실패 turn_ids: {report['failed_turn_ids']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="LLMFilter 모델 경로 (예: /data/delta9043/models/Qwen3-32B)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="필터링 결과 저장 경로 (예: data/filtered_data/filtered_32b.json)"
    )
    args = parser.parse_args()
    run_filter(args.model_path, args.output)


if __name__ == "__main__":
    main()