"""Gold chunking 진입점 (1차 세그먼트 패스 + 2차 leading-split 교정 패스).

옵션
  --dataset {locomo,longmemeval}   필수. 데이터셋 종류
  --data <path>                    필수. 원본 데이터 JSON 경로
  --model <name>                   OpenAI 모델명 (기본: .env의 GOLD_MODEL)
  --temperature <float>            미지정 시 .env의 GOLD_TEMPERATURE, 그것도 없으면 API 기본값
  --concurrency <int>              동시 API 요청 수 (기본: .env의 GOLD_CONCURRENCY)
  --limit <int>                    스모크 테스트용: locomo는 샘플 수, longmemeval은 세션 수 제한
  --out <path>                     결과 JSON 경로 (기본: results/{dataset}_gold.json)
  --fix-leading-splits             1차 패스 대신 2차 교정 패스 실행 (locomo 전용)

실행 예시
  # 1차 패스
  python run_gold.py --dataset locomo --data <locomo10.json 경로>
  python run_gold.py --dataset longmemeval --data <longmemeval_s.json 경로>
  python run_gold.py --dataset locomo --data ... --limit 1        # 스모크 테스트

  # 2차 패스 (1차 결과 파일을 제자리 수정)
  python run_gold.py --dataset locomo --data <locomo10.json 경로> --fix-leading-splits

출력: results/{dataset}_gold.json (청크에 turn 전문 포함 → 단독 소비 가능)
캐시: cache/ 에 세션별 응답 저장 → 중단 후 재실행 시 이어서 진행.
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from core import (
    CONCURRENCY,
    MODEL,
    TEMPERATURE,
    BoundaryClient,
    load_locomo_samples,
    load_longmemeval_sessions,
)
from prompts import PROMPT_VERSION

HERE = Path(__file__).parent


# ── segments → 청크 (turn 전문 포함) ─────────────────────────────────
def build_chunks(session: dict, result: dict, next_chunk_id: int) -> list:
    """result가 None(실패)이면 세션 전체를 한 청크로 fallback."""
    n = len(session["turns"])
    segments = result["segments"] if result else [{"start": 1, "end": n, "topic": ""}]

    chunks = []
    for seg in segments:
        seg_turns = session["turns"][seg["start"] - 1 : seg["end"]]
        chunks.append({
            "chunk_id": next_chunk_id + len(chunks),
            "session_id": session["session_id"],
            "topic": seg.get("topic", ""),
            "turns": [
                {
                    "turn_id": t["turn_id"],
                    "speaker": t["speaker"],
                    "content": t["content"],
                    "timestamp": session["date"],
                    "session_id": session["session_id"],
                }
                for t in seg_turns
            ],
        })
    return chunks


def check_invariant(session: dict, chunks: list) -> None:
    """청크들을 이으면 원본 세션과 turn_id 시퀀스가 정확히 같아야 한다."""
    original = [t["turn_id"] for t in session["turns"]]
    rebuilt = [t["turn_id"] for c in chunks for t in c["turns"]]
    assert original == rebuilt, f"invariant violated in session {session['session_id']}"


# ── 세션 병렬 처리 ────────────────────────────────────────────────────
async def segment_all(client: BoundaryClient, dataset: str, sessions: list) -> dict:
    """{id(session): result or None}. 진행 상황을 주기적으로 출력.

    LoCoMo는 session_id(session_1 등)가 샘플 간 중복되므로 객체 정체성으로 키를 잡는다.
    """
    total = len(sessions)
    done = 0
    results = {}

    async def one(session):
        nonlocal done
        results[id(session)] = await client.segment_session(
            dataset=dataset,
            session_id=session["session_id"],
            date=session["date"],
            turns=session["turns"],
        )
        done += 1
        if done % 10 == 0 or done == total:
            print(f"[gold] {done}/{total} sessions "
                  f"(api {client.usage['api_calls']}, cache {client.usage['cache_hits']}, "
                  f"fail {len(client.failed_sessions)})", flush=True)

    await asyncio.gather(*(one(s) for s in sessions))
    return results


# ── 통계 ─────────────────────────────────────────────────────────────
def chunk_size_stats(sizes: list) -> dict:
    return {
        "min": min(sizes), "median": statistics.median(sizes), "max": max(sizes),
        "pct_single_turn": round(100 * sum(s == 1 for s in sizes) / len(sizes), 1),
        "pct_over_20": round(100 * sum(s > 20 for s in sizes) / len(sizes), 1),
    }


def compute_stats(all_chunks: list, n_sessions: int, client: BoundaryClient) -> dict:
    sizes = [len(c["turns"]) for c in all_chunks]
    return {
        "n_sessions": n_sessions,
        "n_chunks": len(all_chunks),
        "chunks_per_session_avg": round(len(all_chunks) / n_sessions, 2) if n_sessions else 0,
        "chunk_size": chunk_size_stats(sizes) if sizes else {},
        "failed_sessions": list(client.failed_sessions),
        "usage": dict(client.usage),
    }


# ── 1차 패스 ──────────────────────────────────────────────────────────
async def run_locomo(client: BoundaryClient, data_path: str, limit: int) -> dict:
    samples = load_locomo_samples(data_path)
    if limit:
        samples = samples[:limit]

    flat_sessions = [s for sample in samples for s in sample["sessions"]]
    print(f"[gold] locomo: {len(samples)} samples, {len(flat_sessions)} sessions", flush=True)
    results = await segment_all(client, "locomo", flat_sessions)

    out_samples = []
    all_chunks = []
    for sample in samples:
        chunks = []
        for session in sample["sessions"]:
            session_chunks = build_chunks(session, results[id(session)], next_chunk_id=len(chunks))
            check_invariant(session, session_chunks)
            chunks.extend(session_chunks)
        out_samples.append({"sample_id": sample["sample_id"], "chunks": chunks})
        all_chunks.extend(chunks)

    return {
        "dataset": "locomo10",
        "samples": out_samples,
        "stats": compute_stats(all_chunks, len(flat_sessions), client),
    }


async def run_longmemeval(client: BoundaryClient, data_path: str, limit: int) -> dict:
    sessions = load_longmemeval_sessions(data_path)
    if limit:
        sessions = sessions[:limit]

    print(f"[gold] longmemeval_s: {len(sessions)} unique sessions", flush=True)
    results = await segment_all(client, "longmemeval", sessions)

    out_sessions = []
    all_chunks = []
    for session in sessions:
        chunks = build_chunks(session, results[id(session)], next_chunk_id=0)
        check_invariant(session, chunks)
        out_sessions.append({
            "session_id": session["session_id"],
            "date": session["date"],
            "chunks": chunks,
        })
        all_chunks.extend(chunks)

    return {
        "dataset": "longmemeval_s",
        "sessions": out_sessions,
        "stats": compute_stats(all_chunks, len(sessions), client),
    }


# ── 2차 패스: 세션 첫 청크가 1턴인 over-split을 타깃 merge로 교정 ─────
#   - 대상: 세션의 첫 청크가 1턴이고 그 세션이 2청크 이상인 경우만.
#   - (첫 청크, 둘째 청크)를 LLM에 보여주고 '한 에피소드인가?' 질문.
#     merge=True면 첫 청크를 둘째로 병합(둘째 topic 유지).
#   - 나머지 청크는 전혀 건드리지 않는다. 교정 후 불변식 재검증 + 통계 갱신.
def group_by_session(chunks: list) -> dict:
    """청크는 세션 순서대로 flat하게 저장돼 있으므로 삽입 순서 = 세션 순서."""
    by_sess = {}
    for c in chunks:
        by_sess.setdefault(c["session_id"], []).append(c)
    return by_sess


def check_sample_invariant(sample_id: str, chunks: list, orig_ids: list) -> None:
    rebuilt = [t["turn_id"] for c in chunks for t in c["turns"]]
    assert orig_ids == rebuilt, f"invariant violated in sample {sample_id}"


async def fix_leading_splits(client: BoundaryClient, gold_path: Path, data_path: str) -> None:
    out = json.load(open(gold_path, encoding="utf-8"))
    orig = {s["sample_id"]: [t["turn_id"] for sess in s["sessions"] for t in sess["turns"]]
            for s in load_locomo_samples(data_path)}

    # 1) 대상 수집: (sample_idx, session_id) — 첫 청크 1턴 & 세션 2청크+
    targets = []
    for si, sample in enumerate(out["samples"]):
        for sid, cs in group_by_session(sample["chunks"]).items():
            if len(cs) > 1 and len(cs[0]["turns"]) == 1:
                targets.append((si, sid, cs[0], cs[1]))
    print(f"[merge] 대상 세션: {len(targets)}개")

    # 2) 병렬 merge 질문
    async def decide(t):
        si, sid, a, b = t
        date = a["turns"][0].get("timestamp")
        merge = await client.check_merge(date, a, b)
        return (si, sid, merge)

    decisions = {(si, sid): m for si, sid, m in await asyncio.gather(*(decide(t) for t in targets))}
    n_merged = sum(1 for v in decisions.values() if v is True)
    n_fail = sum(1 for v in decisions.values() if v is None)
    print(f"[merge] merge=True: {n_merged}, keep: {sum(1 for v in decisions.values() if v is False)}, fail: {n_fail}")

    # 3) 적용: 세션 순서대로 재조립하며 첫 청크 병합, chunk_id 재부여
    for si, sample in enumerate(out["samples"]):
        new_chunks = []
        for sid, cs in group_by_session(sample["chunks"]).items():
            if decisions.get((si, sid)) is True:
                merged = dict(cs[1])
                merged["turns"] = cs[0]["turns"] + cs[1]["turns"]  # 여는 턴을 앞에 붙임
                cs = [merged] + cs[2:]
            for c in cs:
                c["chunk_id"] = len(new_chunks)
                new_chunks.append(c)
        check_sample_invariant(sample["sample_id"], new_chunks, orig[sample["sample_id"]])
        sample["chunks"] = new_chunks

    # 4) 통계 갱신 (분포만 재계산; merge_pass 요약 추가)
    all_chunks = [c for s in out["samples"] for c in s["chunks"]]
    sizes = [len(c["turns"]) for c in all_chunks]
    st = out["stats"]
    st["n_chunks"] = len(all_chunks)
    st["chunks_per_session_avg"] = round(len(all_chunks) / st["n_sessions"], 2)
    st["chunk_size"] = chunk_size_stats(sizes)
    st["merge_pass"] = {"checked": len(targets), "merged": n_merged, "failed": n_fail}

    json.dump(out, open(gold_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[merge] saved: {gold_path}")
    print(json.dumps(st, ensure_ascii=False, indent=2))


# ── main ─────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="LLM gold chunking (topic boundary detection)")
    parser.add_argument("--dataset", required=True, choices=["locomo", "longmemeval"])
    parser.add_argument("--data", required=True, help="원본 데이터 JSON 경로")
    parser.add_argument("--model", default=MODEL, help="OpenAI 모델명 (기본: .env의 GOLD_MODEL)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="미지정 시 .env의 GOLD_TEMPERATURE, 그것도 없으면 API 기본값")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help="동시 API 요청 수 (기본: .env의 GOLD_CONCURRENCY)")
    parser.add_argument("--limit", type=int, default=0,
                        help="스모크 테스트용: locomo는 샘플 수, longmemeval은 세션 수 제한")
    parser.add_argument("--out", default=None, help="결과 JSON 경로 (기본: results/{dataset}_gold.json)")
    parser.add_argument("--fix-leading-splits", action="store_true",
                        help="1차 패스 대신 2차 leading-split 교정 패스 실행 (locomo 전용)")
    args = parser.parse_args()

    client = BoundaryClient(
        model=args.model,
        cache_dir=str(HERE / "cache"),
        temperature=args.temperature,
        concurrency=args.concurrency,
    )

    # 2차 패스: 1차 결과 파일을 제자리 수정하고 종료
    if args.fix_leading_splits:
        if args.dataset != "locomo":
            parser.error("--fix-leading-splits는 locomo 전용입니다")
        gold_path = Path(args.out) if args.out else HERE / "results" / "locomo10_gold.json"
        await fix_leading_splits(client, gold_path, args.data)
        return

    if args.dataset == "locomo":
        output = await run_locomo(client, args.data, args.limit)
    else:
        output = await run_longmemeval(client, args.data, args.limit)

    output["model"] = args.model
    output["prompt_version"] = PROMPT_VERSION

    out_path = Path(args.out) if args.out else HERE / "results" / f"{output['dataset']}_gold.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)

    print(f"\n[gold] saved: {out_path}")
    print(json.dumps(output["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
