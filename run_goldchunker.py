"""
run_goldchunker.py — GoldChunker(연속 turn 흐름 + carryover + LLM 경계)를 전체 LoCoMo10에
1회 적용해 chunk JSON으로 저장한다. PrecomputedChunker가 로드해 각 backend에서 공유한다.

gold(upper-bound) 청킹: run_chunker.py(LLMChunker, 세션 경계 무조건 컷)와 달리 세션 파티션을
쓰지 않고, gpt-5.6-sol이 내용만 보고 모든 경계를 판단한다. baseline은 수정하지 않는다.

GPU/로컬 모델 의존성이 없으므로(OpenAI API만 사용) 서버가 아닌 로컬에서도 실행 가능:
    python run_goldchunker.py --data <로컬 locomo10.json> --limit 1     # 스모크
    python run_goldchunker.py                                           # 전체 (서버 기본 경로)
API key는 OPENAI_API_KEY 환경변수에서 읽는다. 윈도우 응답은 디스크 캐시되어 재실행 시 재개.

출력 형식은 run_chunker.py와 동일(PrecomputedChunker 호환).

실행 코드 : python run_goldchunker.py \
    --data /data/delta9043/datasets/locomo/locomo10.json \
    --model gpt-5.6-sol \
    --output data/chunked_data/chunks_gpt-5_6-sol.json

    model은 .env에 기입되어 있으면 안 써도 됨.
    data도 아래 코드 내 경로가 맞으면 상관 없음.
    limit 1 도 원하면 제외 가능
    
"""

import argparse
import json
import os
from pathlib import Path

# MemoryAgnostic/.env 로드 (OPENAI_API_KEY). dotenv 미설치 환경이면 OS 환경변수만 사용.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from data.locomo_loader import load_locomo10_all
from core.chunker.gold_chunker import GoldChunker


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


def run(args) -> None:
    print(f"[goldchunker] 데이터 로드 중: {args.data}", flush=True)
    samples = load_locomo10_all(args.data)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[goldchunker] {len(samples)}개 샘플", flush=True)

    chunker = GoldChunker(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        window_turns=args.window_turns,
        carryover_max_turns=args.carryover_max_turns,
    )

    results = []
    total = len(samples)
    n_cross = 0
    for idx, sample in enumerate(samples):
        print(f"[goldchunker] Sample {idx+1}/{total} ({sample.sample_id}) | turns={len(sample.turns)}", flush=True)
        chunks = chunker.chunk(sample.turns)
        n_cross += sum(1 for c in chunks if c.metadata.get("crosses_session"))
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
        print(f"[goldchunker] {sample.sample_id} 완료 | chunks={len(chunks)}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[goldchunker] 저장 완료: {args.output}", flush=True)

    # 통계: cross-session 청크 수가 '세션 공짜 이점 제거'의 관측 지표
    sizes = [len(c["turns"]) for r in results for c in r["chunks"]]
    report = chunker.get_report()
    print(f"[goldchunker] 총 chunk: {len(sizes)} | cross-session chunk: {n_cross} | "
          f"평균 크기: {sum(sizes)/len(sizes):.1f} turns", flush=True)
    print(f"[goldchunker] LLM 호출: {report['usage']['api_calls']} | 캐시 히트: {report['usage']['cache_hits']} | "
          f"실패 윈도우: {report['failure_count']}", flush=True)
    print(f"[goldchunker] 토큰: prompt={report['usage']['prompt_tokens']:,} "
          f"completion={report['usage']['completion_tokens']:,}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="LLM stream gold chunking (upper-bound)")
    parser.add_argument("--data", type=str, default=LOCOMO_PATH,
                        help="locomo10.json 경로 (로컬 실행 시 로컬 경로 지정)")
    parser.add_argument("--model", type=str, default=os.environ.get("GOLD_MODEL"),
                        help="LLM 모델명. 미지정 시 .env의 GOLD_MODEL 사용")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--window_turns", type=int, default=100, help="W: 윈도우 트리거 turn 수")
    parser.add_argument("--carryover_max_turns", type=int, default=50, help="C: carryover 상한")
    parser.add_argument("--limit", type=int, default=0, help="앞 N개 샘플만(스모크). 0=전체")
    parser.add_argument("--output", type=str, default="data/chunked_data/chunks_goldchunker.json")
    args = parser.parse_args()
    if not args.model:
        parser.error("모델명이 필요합니다. MemoryAgnostic/.env에 GOLD_MODEL=... 을 설정하거나 --model 로 전달하세요.")
    run(args)


if __name__ == "__main__":
    main()
