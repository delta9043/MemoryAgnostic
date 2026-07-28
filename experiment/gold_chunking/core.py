"""Gold chunking 공통 모듈 — 설정 / 데이터 로더 / OpenAI 클라이언트.

구 GoldChunking의 config.py + loaders/{locomo,longmemeval}.py + boundary_client.py를
합친 것. 동작은 동일하다.

설정은 같은 디렉토리의 .env에서 읽는다 (.env.example을 복사해 값을 채울 것).
CLI 인자가 있으면 CLI 우선, 없으면 .env 값이 기본값.
"""

import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from prompts import (
    CORRECTION_TEMPLATE,
    MERGE_SCHEMA,
    PROMPT_VERSION,
    SEGMENTS_SCHEMA,
    build_merge_messages,
    build_messages,
)

# ── 설정 ─────────────────────────────────────────────────────────────
# .env를 명시적으로 로드 (실행 위치와 무관하게 동작)
load_dotenv(Path(__file__).parent / ".env")


def _get_temperature():
    val = os.environ.get("GOLD_TEMPERATURE", "").strip()
    return float(val) if val else None


MODEL = os.environ.get("GOLD_MODEL", "gpt-5.6-sol")
TEMPERATURE = _get_temperature()
CONCURRENCY = int(os.environ.get("GOLD_CONCURRENCY", "8"))
# OPENAI_API_KEY는 openai SDK가 환경변수에서 직접 읽으므로 여기선 로드만 하면 된다.


# ── 데이터 로더 ───────────────────────────────────────────────────────
def _get_session_keys(conversation: dict) -> List[str]:
    # session_N 키만 번호 순서대로
    pattern = re.compile(r"^session_(\d+)$")
    keys = [(int(m.group(1)), k) for k in conversation if (m := pattern.match(k))]
    keys.sort()
    return [k for _, k in keys]


def load_locomo_samples(path: str) -> List[dict]:
    """LoCoMo10 → [{"sample_id", "sessions": [{"session_id", "date", "turns": [...]}]}].

    MemoryAgnostic locomo_loader와 동일한 파싱 규칙.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for raw in data:
        sample_id = str(raw["sample_id"])
        conversation = raw["conversation"]
        sessions = []
        for session_key in _get_session_keys(conversation):
            turns = [
                {"turn_id": t["dia_id"], "speaker": t["speaker"], "content": t["text"]}
                for t in conversation[session_key]
            ]
            if not turns:
                continue
            sessions.append({
                "session_id": session_key,
                "date": conversation.get(f"{session_key}_date_time"),
                "turns": turns,
            })
        samples.append({"sample_id": sample_id, "sessions": sessions})
    return samples


def load_longmemeval_sessions(path: str) -> List[dict]:
    """LongMemEval_s → 유니크 세션 리스트 (첫 등장 순서).

    - 세션 키 = haystack_session_ids[i]. 같은 세션은 1회만 청킹.
    - turn_id는 원본에 없으므로 "{session_id}:{idx}"로 부여.
    - has_answer 등 정답 위치 힌트는 프롬프트에 노출하지 않는다 (role/content만 사용).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    sessions = {}
    for item in data:
        for sid, date, raw_turns in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
        ):
            if sid in sessions or not raw_turns:
                continue
            sessions[sid] = {
                "session_id": sid,
                "date": date,
                "turns": [
                    {"turn_id": f"{sid}:{i}", "speaker": t["role"], "content": t["content"]}
                    for i, t in enumerate(raw_turns)
                ],
            }
    return list(sessions.values())


# ── OpenAI 클라이언트 ─────────────────────────────────────────────────
def validate_segments(segments: list, n: int) -> bool:
    """segments가 turn 1..N을 순서대로 빈틈/겹침 없이 타일링하는지 검사."""
    if not isinstance(segments, list) or not segments:
        return False
    for s in segments:
        if not isinstance(s.get("start"), int) or not isinstance(s.get("end"), int):
            return False
        if s["start"] > s["end"]:
            return False
    if segments[0]["start"] != 1 or segments[-1]["end"] != n:
        return False
    for prev, cur in zip(segments, segments[1:]):
        if cur["start"] != prev["end"] + 1:
            return False
    return True


class BoundaryClient:
    """Structured Outputs(strict json_schema)로 형식을 강제하고, 타일링만 코드에서 검증한다.

    검증 통과한 응답만 디스크 캐시 → 재실행 시 무료로 재개.
    실패(재시도 소진/타일링 정정 실패) 세션은 None 반환 → 상위에서 세션=한 청크 fallback.
    """

    def __init__(
        self,
        model: str,
        cache_dir: str,
        temperature: Optional[float] = None,
        concurrency: int = 8,
        max_retries: int = 6,
    ):
        self.client = AsyncOpenAI()  # API key는 OPENAI_API_KEY 환경변수에서 읽음
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sem = asyncio.Semaphore(concurrency)

        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0, "cache_hits": 0}
        self.failed_sessions: list = []

    # ── 캐시 ─────────────────────────────────────────────────────────
    def _cache_path(self, messages: list) -> Path:
        key_src = json.dumps(
            {"model": self.model, "prompt_version": PROMPT_VERSION, "messages": messages},
            ensure_ascii=False, sort_keys=True,
        )
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    # ── API 호출 (재시도 포함) ────────────────────────────────────────
    async def _create(self, messages: list, schema: dict = SEGMENTS_SCHEMA) -> Optional[str]:
        kwargs = dict(
            model=self.model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": schema},
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        for attempt in range(self.max_retries):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                if resp.usage:
                    self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                    self.usage["completion_tokens"] += resp.usage.completion_tokens
                self.usage["api_calls"] += 1
                return resp.choices[0].message.content
            except (RateLimitError, APIConnectionError, APITimeoutError):
                pass  # 재시도
            except APIStatusError as e:
                if e.status_code < 500:
                    raise  # 4xx는 설정 오류일 가능성 → 즉시 중단
            await asyncio.sleep(min(2 ** attempt, 30) + random.random())
        return None

    @staticmethod
    def _parse(raw: Optional[str]) -> Optional[dict]:
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or "segments" not in parsed:
            return None
        return parsed

    # ── 세션 1개 → segments ──────────────────────────────────────────
    async def segment_session(self, dataset: str, session_id: str, date: str, turns: list) -> Optional[dict]:
        """성공: {"reasoning", "segments"} / 실패: None (상위에서 fallback)."""
        n = len(turns)
        if n == 1:
            return {"reasoning": "single-turn session", "segments": [{"start": 1, "end": 1, "topic": ""}]}

        messages = build_messages(dataset, date, turns)
        cache_path = self._cache_path(messages)
        if cache_path.exists():
            self.usage["cache_hits"] += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))["response"]

        async with self.sem:
            raw = await self._create(messages)
            parsed = self._parse(raw)

            # 타일링 위반 시 1회 정정 요청
            if parsed is not None and not validate_segments(parsed["segments"], n):
                retry_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": CORRECTION_TEMPLATE.format(n=n)},
                ]
                parsed = self._parse(await self._create(retry_messages))

        if parsed is None or not validate_segments(parsed["segments"], n):
            self.failed_sessions.append(session_id)
            return None

        cache_path.write_text(
            json.dumps({"model": self.model, "prompt_version": PROMPT_VERSION,
                        "session_id": session_id, "response": parsed}, ensure_ascii=False),
            encoding="utf-8",
        )
        return parsed

    # ── 인접 세그먼트 merge 검토 (2차 타깃 패스) ──────────────────────
    async def check_merge(self, date: str, seg_a: dict, seg_b: dict) -> Optional[bool]:
        """A와 B가 한 에피소드인가(경계 제거) → True/False. 실패 시 None."""
        messages = build_merge_messages(date, seg_a, seg_b)
        cache_path = self._cache_path(messages)
        if cache_path.exists():
            self.usage["cache_hits"] += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))["response"]["merge"]

        async with self.sem:
            raw = await self._create(messages, schema=MERGE_SCHEMA)
        parsed = self._parse_merge(raw)
        if parsed is None:
            return None

        cache_path.write_text(
            json.dumps({"model": self.model, "prompt_version": PROMPT_VERSION,
                        "response": parsed}, ensure_ascii=False),
            encoding="utf-8",
        )
        return parsed["merge"]

    @staticmethod
    def _parse_merge(raw: Optional[str]) -> Optional[dict]:
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("merge"), bool):
            return None
        return parsed
