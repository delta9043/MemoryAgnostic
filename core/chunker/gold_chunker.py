"""
GoldChunker — 연속 turn 흐름 위에서 LLM이 경계를 찾는 carryover 방식 gold(upper-bound) 청커.

설계 (2026-07-17 결정)
  - 모든 세션의 turn을 이어붙인 '하나의 연속 stream'을 W-turn 버퍼로 흘린다.
    세션 경계에서 끊지 않고, session_id/timestamp를 경계 판단에 쓰지 않는다(내용만).
    → 세션 파티션을 공짜로 받는 confound 제거. chunk가 세션을 넘을 수 있음(의도된 동작).
  - 버퍼가 W turn에 도달하면 LLM(gpt-5.6-sol)이 버퍼 전체를 에피소드로 타일링한다.
  - carryover: 마지막 경계는 윈도우 끝에 가까워 근거(미래 context)가 얇다. 그래서 '마지막
    두 세그먼트'(마지막 완성 청크 + 꼬리)는 commit하지 않고 버퍼에 남겨, 다음 윈도우에서
    백지 상태로 재판단한다(이전 판단 힌트는 주지 않음 — 앵커링 방지).
    단, 진행 보장을 위해 carryover가 C turn 이상이면 강등: 마지막 한 세그먼트만 남기고,
    그것도 C 이상이면 전부 commit. → 호출당 최소 W−C turn 진행 = 호출 수 ≤ N/(W−C).
  - baseline(AttentionSimilarityChunker 등)은 수정하지 않는다: 이 청커는 논문 방법의 재현이
    아니라 'LLM 청킹의 상한(upper bound)'을 재는 용도다.

출력 불변식
  - 반환 Chunk들의 turns를 이으면 입력 turns와 정확히 동일(누락/중복/재정렬 없음).
  - 원본 Turn 객체를 그대로 담는다 → timestamp/session_id 보존(다운스트림·관측용.
    프롬프트에는 들어가지 않음).

실패(호출/파싱) 윈도우는 '분할 없음(한 세그먼트)'로 처리돼 통째로 commit + 통계 기록.
윈도우 응답은 버퍼 내용 해시로 디스크 캐시 → 재실행 시 같은 윈도우 시퀀스가 결정적으로
리플레이되어 중단 지점부터 무료로 재개된다.
"""

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from core.chunker.base import BaseChunker
from data.schema import Chunk, Turn


# 캐시 키에 포함됨. 프롬프트를 바꾸면 반드시 올려서 캐시를 무효화할 것.
PROMPT_VERSION = "v1"

# 세션/날짜 전제 없음: 한 윈도우가 여러 세션·여러 시점을 걸칠 수 있으므로 내용만으로 판단시킨다.
SYSTEM_PROMPT = """You are an episodic memory boundary detection expert. You are given a window of consecutive conversation turns (they may span different topics, and possibly different occasions). Your task is to find the natural "episode boundaries" within this window and split it into meaningful, independently memorable segments. Your core principle is **"default to merging, split cautiously"**.

### When to split

Add a boundary (by turn number) only when a **clear signal** appears:
- **Substantive topic change:** the conversation shifts from one concrete topic to a completely unrelated one (e.g., a health concern → weekend travel plans).
- **Task/thread completion + new topic:** a closing turn ("sounds good, thanks!") belongs to its current episode; split only when the **next** turn opens a genuinely unrelated topic.

**Do NOT split for:**
- Greetings, farewells ("hi", "bye", "thanks") — keep them with the episode they serve.
- Transition phrases ("by the way", "oh also", "speaking of") — these usually CONTINUE the current episode unless they introduce a major, unrelated topic.
- Follow-up questions, clarifications, or brief reactions on the same topic.

### Decision Principles
- **Merge by default:** when in doubt, do not split; only split on clear signals.
- **Content over form:** greetings and farewells belong to the episode they serve, not their own segment.
- **Process continuity:** consecutive turns working toward the same goal (e.g., describe a problem → discuss a fix) form one episode.
- The first turn of the window can never be a boundary (it already starts the first segment).

### Examples
(input format: "N. speaker: content"; a boundary is a turn number AFTER which to split)

**Example 1 — one boundary:**
1. Alice: Can you help me debug the login issue?
2. Bob: Sure, let me check the logs.
3. Bob: Found it — a null pointer in AuthService.
4. Alice: Fixed, thanks!
5. Alice: By the way, are you free for lunch today?
6. Bob: Sure, 12:30?
Output:
{"reasoning": "Turns 1-4 are a complete bug-fix episode; turn 5 opens an unrelated lunch topic.", "boundaries": [4]}

**Example 2 — no boundary:**
1. Alice: What's the status of the Q2 roadmap?
2. Bob: About 60% done. Need to finalize the API specs.
3. Alice: OK, let's review the specs tomorrow.
Output:
{"reasoning": "All turns are part of the same Q2 roadmap discussion with no topic change.", "boundaries": []}"""

USER_PROMPT_TEMPLATE = """Here is a window of consecutive conversation turns. Split it into topic-coherent episodes.

{turns_block}

Return STRICT JSON only, no other text:
{{"reasoning": "<one sentence explaining all boundary decisions>", "boundaries": [<turn numbers AFTER which to split>]}}

- Numbers are 1-based and refer to the list above.
- A number b means: split AFTER turn b — turns up to b end one episode, and turn b+1 starts the next.
- Valid range is 1..{n_minus_1} (you cannot split after the last turn).
- `"boundaries": []` means the whole window is a single episode (no split)."""

# OpenAI Structured Outputs용 strict 스키마
BOUNDARIES_SCHEMA = {
    "name": "window_boundaries",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "boundaries": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["reasoning", "boundaries"],
        "additionalProperties": False,
    },
}


class GoldChunker(BaseChunker):
    """연속 stream + carryover 윈도우에서 LLM으로 경계를 찾는 청커. 인터페이스는 BaseChunker.chunk()."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        window_turns: int = 100,
        carryover_max_turns: int = 50,
        max_retries: int = 6,
        cache_dir: Optional[str] = "data/chunked_data/.cache_goldchunker",
    ):
        """
        Args:
            model:               OpenAI 호환 모델명 (필수). runner가 .env의 GOLD_MODEL에서 읽어 전달. 예: gpt-5.6-sol.
            api_key/base_url:    미지정 시 OPENAI_API_KEY 환경변수 / OpenAI 기본 엔드포인트.
            temperature:         None이면 API 기본값(reasoning 모델은 미지원일 수 있음).
            window_turns:        W. 버퍼가 이 turn 수에 도달하면 LLM 호출.
            carryover_max_turns: C. carryover 상한(진행 보장). W−C가 호출당 최소 진행량.
            cache_dir:           윈도우 응답 디스크 캐시(재개용). None이면 캐시 안 함.
        """
        if window_turns < 2:
            raise ValueError(f"window_turns must be >= 2, got {window_turns}")
        if not (0 < carryover_max_turns < window_turns):
            raise ValueError(
                f"carryover_max_turns must satisfy 0 < C < window_turns, "
                f"got C={carryover_max_turns}, W={window_turns}"
            )

        self.model = model
        self.temperature = temperature
        self.window_turns = window_turns
        self.carryover_max_turns = carryover_max_turns
        self.max_retries = max_retries

        # 클라이언트는 lazy load (실행 없이 인스턴스만 만드는 경우 API key 불필요하게).
        self._client = None
        self._api_key = api_key
        self._base_url = base_url

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 관측 통계. reset_stats()로 초기화.
        self.usage = {"api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.failure_count: int = 0

    def _load(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
            )

    def reset_stats(self) -> None:
        self.usage = {"api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.failure_count = 0

    def get_report(self) -> dict:
        return {"failure_count": self.failure_count, "usage": dict(self.usage)}

    # ──────────────────────────────────────────────────────────────────────
    # 1) LLM 호출 (캐시 + 재시도)
    # ──────────────────────────────────────────────────────────────────────
    def _cache_path(self, buffer_texts: List[str]) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key_src = json.dumps(
            {"model": self.model, "prompt_version": PROMPT_VERSION, "texts": buffer_texts},
            ensure_ascii=False, sort_keys=True,
        )
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _build_messages(self, buffer_texts: List[str]) -> list:
        turns_block = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(buffer_texts))
        user = USER_PROMPT_TEMPLATE.format(turns_block=turns_block, n_minus_1=len(buffer_texts) - 1)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _call_llm(self, messages: list) -> Optional[str]:
        kwargs = dict(
            model=self.model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": BOUNDARIES_SCHEMA},
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                self.usage["api_calls"] += 1
                if getattr(resp, "usage", None):
                    self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                    self.usage["completion_tokens"] += resp.usage.completion_tokens
                return resp.choices[0].message.content
            except (RateLimitError, APIConnectionError, APITimeoutError):
                pass  # 재시도
            except APIStatusError as e:
                if e.status_code < 500:
                    raise  # 4xx는 설정 오류일 가능성 → 즉시 중단
            time.sleep(min(2 ** attempt, 30) + random.random())
        return None

    # ──────────────────────────────────────────────────────────────────────
    # 2) 응답 파싱
    # ──────────────────────────────────────────────────────────────────────
    def _parse_boundaries(self, raw: Optional[str], n_turns: int) -> Optional[List[int]]:
        """LLM 응답 → 1-based 'after which to split' 경계 리스트(유효 1..n-1, 정렬/중복제거).
        실패(호출/JSON/형식) 시 None. 빈 [](= 경계 없음, 성공)과 구분된다."""
        if raw is None:
            return None
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("boundaries"), list):
            return None
        valid = [
            b for b in parsed["boundaries"]
            if isinstance(b, int) and not isinstance(b, bool) and 1 <= b <= n_turns - 1
        ]
        return sorted(set(valid))

    # ──────────────────────────────────────────────────────────────────────
    # 3) 버퍼 1개 → 세그먼트 타일링
    # ──────────────────────────────────────────────────────────────────────
    def _segment_buffer(self, buffer: List[Turn]) -> List[List[Turn]]:
        """버퍼를 LLM 경계로 타일링. 실패 시 [버퍼 전체 1세그먼트] + 통계."""
        n = len(buffer)
        if n < 2:
            return [list(buffer)]

        buffer_texts = [f"{t.speaker}: {t.content}" for t in buffer]

        cache_path = self._cache_path(buffer_texts)
        if cache_path and cache_path.exists():
            self.usage["cache_hits"] += 1
            bounds = json.loads(cache_path.read_text(encoding="utf-8"))["boundaries"]
        else:
            raw = self._call_llm(self._build_messages(buffer_texts))
            bounds = self._parse_boundaries(raw, n)
            if bounds is None:
                self.failure_count += 1
                bounds = []  # 실패 → 분할 없음(한 세그먼트로 통째 commit됨)
            elif cache_path:
                # 검증 통과한 응답만 캐시 (실패는 재실행 시 다시 시도)
                cache_path.write_text(
                    json.dumps({"model": self.model, "prompt_version": PROMPT_VERSION,
                                "boundaries": bounds}, ensure_ascii=False),
                    encoding="utf-8",
                )

        cut_points = [0] + bounds + [n]
        return [buffer[a:b] for a, b in zip(cut_points[:-1], cut_points[1:]) if b > a]

    # ──────────────────────────────────────────────────────────────────────
    # 4) commit 규칙 (carryover)
    # ──────────────────────────────────────────────────────────────────────
    def _split_commit_carryover(self, segments: List[List[Turn]]):
        """(commit할 세그먼트들, carryover turn들)로 분리.

        마지막 경계는 윈도우 끝에 가까워 근거가 얇다 → 마지막 두 세그먼트를 잠정으로 남겨
        다음 윈도우에서 재판단. 단 carryover가 C 이상이면 강등(진행 보장):
        마지막 한 세그먼트만 → 그것도 C 이상이면 전부 commit.
        """
        C = self.carryover_max_turns

        if len(segments) >= 2:
            tail2 = segments[-2] + segments[-1]
            if len(tail2) < C:
                return segments[:-2], tail2
            if len(segments[-1]) < C:
                return segments[:-1], list(segments[-1])
        # 세그먼트 1개(윈도우 전체 한 주제/실패)이거나 꼬리가 너무 큼 → 전부 commit
        return segments, []

    # ──────────────────────────────────────────────────────────────────────
    # 5) 진입점
    # ──────────────────────────────────────────────────────────────────────
    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        if not turns:
            return []

        self._load()

        sample_id = turns[0].metadata.get("sample_id", "?")
        buffer: List[Turn] = []
        all_segments: List[List[Turn]] = []
        window_idx = 0  # 이 샘플 내 윈도우 번호(1,2,3…). usage는 전역 누적이라 로그엔 안 씀.
        n_calls_expected = max(1, len(turns) // (self.window_turns - self.carryover_max_turns))

        for turn in turns:
            buffer.append(turn)
            if len(buffer) >= self.window_turns:
                segments = self._segment_buffer(buffer)
                committed, buffer = self._split_commit_carryover(segments)
                all_segments.extend(committed)
                window_idx += 1
                print(f"[GoldChunker] {sample_id} window {window_idx}/~{n_calls_expected} "
                      f"(committed {len(all_segments)} segs, carryover {len(buffer)} turns)", flush=True)

        # stream 끝: 남은 버퍼는 마지막 1회 분할 후 전부 commit (carryover 없음)
        if buffer:
            all_segments.extend(self._segment_buffer(buffer))

        # 세그먼트 → Chunk. session_ids/crosses_session은 confound 제거 관측용 메타데이터.
        chunks = []
        for chunk_id, seg_turns in enumerate(all_segments):
            seg_text = "\n".join(f"{t.speaker}: {t.content}" for t in seg_turns)
            session_ids = list(dict.fromkeys(t.session_id for t in seg_turns))
            chunks.append(Chunk(
                chunk_id=chunk_id,
                turns=list(seg_turns),  # 원본 Turn 객체 보존 → timestamp 유지
                text=seg_text,
                metadata={
                    "start_turn_id": seg_turns[0].turn_id,
                    "end_turn_id": seg_turns[-1].turn_id,
                    "start_timestamp": seg_turns[0].timestamp,
                    "end_timestamp": seg_turns[-1].timestamp,
                    "session_ids": session_ids,
                    "crosses_session": len(session_ids) > 1,
                    "chunker": "GoldChunker",
                },
            ))
        return chunks
