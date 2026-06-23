"""
LLMChunker — 세션 단위 batch 방식의 LLM topic-boundary 청커.

설계 규칙
  - loader가 각 Turn에 session_id와 그 세션의 날짜 timestamp를 붙여 둔다.
  - 청킹 경계는 두 종류로 나뉜다:
      1) 세션 경계  → 날짜가 바뀌므로 '무조건' 자른다 (LLM에게 묻지 않는다).
      2) 세션 내부  → 한 세션 안에서 주제가 바뀌는 지점을 LLM이 찾는다.

  이 방식은 EverMemOS의 contextual segmentation 아이디어(주제 전환 시 경계)를 따르되, batch 단위를 '세션'으로 두었다.

출력 불변식:
  - 반환된 Chunk들의 turns를 순서대로 이으면 입력 turns와 정확히 동일하다(누락/중복/재정렬 없음).
  - 어떤 Chunk도 세션 경계를 넘지 않는다(한 Chunk의 모든 turn은 같은 session_id).
  - 원본 Turn 객체를 그대로 담는다 → timestamp 등 메타데이터가 경계에서 손실되지 않는다.

모델은 transformers로 이 프로세스 안에 직접 로드해 generate한다.
실패(생성 오류/JSON 파싱) 세션은 내부 분할 없이 '세션=한 청크'로 두고 실패 통계에 기록한다.
"""

import json
import re
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.chunker.base import BaseChunker
from data.schema import Chunk, Turn


# LLM에게 주는 시스템 프롬프트
SYSTEM_PROMPT = """You are an episodic memory boundary detection expert. You are given the turns of a SINGLE conversation session (all messages share the same day). Your task is to find the natural "episode boundaries" within this session and split it into meaningful, independently memorable segments. Your core principle is **"default to merging, split cautiously"**.

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
- The first turn of the session can never be a boundary (it already starts the first segment).

### Examples
(input format: "N. speaker: content"; a boundary is a turn number AFTER which to split)

**Example 1 — one boundary:**
[1] Alice: Can you help me debug the login issue?
[2] Bob: Sure, let me check the logs.
[3] Bob: Found it — a null pointer in AuthService.
[4] Alice: Fixed, thanks!
[5] Alice: By the way, are you free for lunch today?
[6] Bob: Sure, 12:30?
Output:
{"reasoning": "Turns 1-4 are a complete bug-fix episode; turn 5 opens an unrelated lunch topic.", "boundaries": [4]}

**Example 2 — no boundary:**
[1] Alice: What's the status of the Q2 roadmap?
[2] Bob: About 60% done. Need to finalize the API specs.
[3] Alice: OK, let's review the specs tomorrow.
Output:
{"reasoning": "All turns are part of the same Q2 roadmap discussion with no topic change.", "boundaries": []}"""

# 사용자 프롬프트 템플릿.
# turns_block에는 "1. speaker: content" 형태로 1-based 번호를 붙여 넣는다.
# 출력은 JSON 한 줄: '그 뒤에서 자를' turn 번호 리스트(after which to split).
USER_PROMPT_TEMPLATE = """Here are the turns of one conversation session (date: {session_date}). Split it into topic-coherent episodes.

{turns_block}

Return STRICT JSON only, no other text:
{{"reasoning": "<one sentence explaining all boundary decisions>", "boundaries": [<turn numbers AFTER which to split>]}}

- Numbers are 1-based and refer to the list above.
- A number b means: split AFTER turn b — turns up to b end one episode, and turn b+1 starts the next.
- Valid range is 1..{n_minus_1} (you cannot split after the last turn).
- `"boundaries": []` means the whole session is a single episode (no split)."""


class LLMChunker(BaseChunker):
    """세션 단위 batch LLM 청커. 인터페이스는 BaseChunker.chunk(turns) -> List[Chunk]."""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
        enable_thinking: bool = False,
    ):
        """
        Args:
            model_path:  로컬 모델 경로 또는 HF repo id (예: /data/delta9043/models/Qwen3-32B).
            device_map:  transformers 디바이스 배치. "auto"면 가용 GPU에 자동 분산.
            torch_dtype: 가중치 dtype. "auto"면 모델 config의 dtype 사용
            temperature
            seed:        재현성용 시드. greedy면 사실상 항상 같은 출력이지만 안전하게 고정한다.
            max_tokens:  생성할 최대 새 토큰 수
            enable_thinking: 기본 False
        """
        self.model_path = model_path
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

        # 모델/토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

        # greedy 재현성을 위해 시드 고정
        torch.manual_seed(self.seed)

        # 실패 통계(관측용). 한 run 동안 누적된다. reset_failure_stats()로 초기화.
        self.failure_count: int = 0
        self.failed_sessions: List[str] = []

    def reset_failure_stats(self) -> None:
        self.failure_count = 0
        self.failed_sessions = []

    def get_failure_report(self) -> dict:
        return {
            "failure_count": self.failure_count,
            "failed_sessions": list(self.failed_sessions),
        }

    # ──────────────────────────────────────────────────────────────────────
    # 1) 세션 그룹핑 (turn을 세션별로 그룹을 만든다.)
    # ──────────────────────────────────────────────────────────────────────
    def _group_by_session(self, turns: List[Turn]) -> List[List[Turn]]:
        """
        turns를 '연속된 같은 session_id' 단위로 묶는다.

        loader가 session_1, session_2 … 순서대로 flat하게 이어 붙이므로,
        session_id가 바뀌는 지점이 곧 세션 경계다. 정렬을 다시 하지 않고
        '연속 런(run)'으로 끊어 입력 순서를 그대로 보존한다.

        """
        groups: List[List[Turn]] = []
        current: List[Turn] = []
        current_sid = object()  # 첫 비교에서 반드시 다르도록 object()로 초기화

        for t in turns:
            if t.session_id != current_sid:
                if current: # 새로운 그룹 등장 시 이전 그룹 처리 / 초기화
                    groups.append(current)
                current = [t]
                current_sid = t.session_id
            else: # 기존 그룹에 추가
                current.append(t)
        if current: # 마지막 그룹 처리
            groups.append(current)
        return groups

    # ──────────────────────────────────────────────────────────────────────
    # 2) 프롬프트 렌더링
    # ──────────────────────────────────────────────────────────────────────
    def _render_turn(self, n: int, turn: Turn) -> str:
        # "n. speaker: content" 형식으로 렌더링
        return f"{n}. {turn.speaker}: {turn.content or ''}"

    def _build_messages(self, session_turns: List[Turn]) -> list: # system message + user message로 만든다.
        turns_block = "\n".join(
            self._render_turn(i + 1, t) for i, t in enumerate(session_turns)
        )
        # 세션 내 turn들은 같은 날짜 timestamp를 공유하므로 날짜는 맥락용으로 1번만 보여준다.
        session_date = session_turns[0].timestamp or "unknown"
        # 경계는 'after which to split'이라 마지막 turn 뒤는 경계가 될 수 없다 
        # → 유효 상한은 n-1.
        user_prompt = USER_PROMPT_TEMPLATE.format(
            session_date=session_date,
            turns_block=turns_block,
            n_minus_1=len(session_turns) - 1,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    # ──────────────────────────────────────────────────────────────────────
    # 3) LLM 호출
    # ──────────────────────────────────────────────────────────────────────
    def _call_llm(self, messages: list) -> Optional[str]:
        """
        messages(system+user)를 chat template로 펴서 generate하고,
        '새로 생성된 토큰'만 디코드해 문자열로 돌려준다. 실패 시 None(상위에서 fallback).

        반환 문자열의 후처리(<think> 제거, JSON 추출)는 _parse_boundaries가 담당한다.
        """
        # 1) chat template 적용 → 모델 입력 문자열.
        #    enable_thinking은 Qwen3 chat template에 그대로 전달(미검증: 비-Qwen 모델은 무시할 수 있음).
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        # 2) 토크나이즈 후 모델 디바이스로 이동
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # 3) 생성
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        try:
            with torch.no_grad():
                generated = self.model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            print(f"[LLMChunker] generate failed: {e}", flush=True)
            return None

        # 4) 입력 길이 이후의 '새로 생성된' 부분만 디코드(프롬프트 에코 제거)
        new_tokens = generated[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ──────────────────────────────────────────────────────────────────────
    # 4) 응답 파싱
    # ──────────────────────────────────────────────────────────────────────
    def _parse_boundaries(self, raw: Optional[str], n_turns: int) -> Optional[List[int]]:
        """
        LLM 응답에서 {"boundaries": [...]}를 robust하게 추출한다.

        반환:
          - 성공: 세션-로컬 1-based 경계 리스트(유효 범위 1..num_turns-1만, 정렬/중복제거).
                  경계 b의 의미 = "turn b '뒤에서' 자른다(after which to split)".
          - 실패(호출 실패/JSON 깨짐/형식 불일치): None  → 상위에서 '내부 분할 없음'으로 처리.

        주의: 빈 리스트 []는 '성공했고 경계가 없음'을 뜻한다. None(실패)과 구분된다.
        """
        # 호출 실패 방어
        if raw is None:
            return None

        # Qwen3 thinking 블록과 코드펜스 마커 제거
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?", "", text)

        # 가장 바깥 중괄호 한 쌍만 추출.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try: # JSON 파싱
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        # 형식 검증
        if not isinstance(parsed, dict) or "boundaries" not in parsed:
            return None
        raw_bounds = parsed["boundaries"]
        if not isinstance(raw_bounds, list):
            return None

        # 유효한 정수 경계만 채택: 1..n_turns-1. (마지막 turn의 뒤는 자를 수 없음.)
        # bool은 사전에 배제
        valid = []
        for b in raw_bounds:
            if isinstance(b, bool):
                continue
            if isinstance(b, int) and 1 <= b <= n_turns - 1:
                valid.append(b)
        return sorted(set(valid))

    # ──────────────────────────────────────────────────────────────────────
    # 5) 한 세션 → 청크들
    # ──────────────────────────────────────────────────────────────────────
    def _chunk_one_session(self, session_turns: List[Turn], next_chunk_id: int) -> List[Chunk]:
        """
        한 세션을 LLM 경계로 잘라 Chunk 리스트로 만든다.
        next_chunk_id부터 chunk_id를 연속 부여한다.
        """
        n = len(session_turns)

        # 1-turn 세션이면 분할할 게 없다.
        if n == 1:
            local_bounds: Optional[List[int]] = []
        else:
            messages = self._build_messages(session_turns)
            raw = self._call_llm(messages)
            local_bounds = self._parse_boundaries(raw, n)

        if local_bounds is None:
            # 실패 → 내부 분할 없이 세션 전체를 한 청크로. 통계에 기록.
            self.failure_count += 1
            self.failed_sessions.append(session_turns[0].session_id or "?")
            local_bounds = []

        # cut_points 예: 세션 6턴, boundaries=[4] 
        #                → [0, 4, 6] → turns[0:4], turns[4:6]
        cut_points = [0] + list(local_bounds) + [n]

        chunks: List[Chunk] = []
        for seg_idx, (a, b) in enumerate(zip(cut_points[:-1], cut_points[1:])):
            seg_turns = session_turns[a:b]
            if not seg_turns:
                continue
            seg_text = "\n".join(f"{t.speaker}: {t.content}" for t in seg_turns)
            chunks.append(Chunk(
                chunk_id=next_chunk_id + len(chunks),
                turns=list(seg_turns),  # 원본 Turn 객체 보존 → timestamp 유지
                text=seg_text,
                metadata={
                    "start_turn_id": seg_turns[0].turn_id,
                    "end_turn_id": seg_turns[-1].turn_id,
                    "start_timestamp": seg_turns[0].timestamp,
                    "end_timestamp": seg_turns[-1].timestamp,
                    "session_id": seg_turns[0].session_id,
                    "chunker": "LLMChunker",
                },
            ))
        return chunks

    # ──────────────────────────────────────────────────────────────────────
    # 6) 진입점
    # ──────────────────────────────────────────────────────────────────────
    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        if not turns:
            return []

        sessions = self._group_by_session(turns)
        total = len(sessions)          # = 이 샘플의 세션 수(= LLM 호출 횟수). 보통 19~32.
        bar_width = 20
        # 진행바 라벨용 sample_id (loader가 turn metadata에 넣어줌). 없으면 "?".
        sample_id = turns[0].metadata.get("sample_id", "?")

        chunks: List[Chunk] = []
        for i, session_turns in enumerate(sessions):
            # 세션 1개 = LLM 호출 1번(느린 부분). 그래서 진행바 단위도 '세션'으로 둔다.
            chunks.extend(self._chunk_one_session(session_turns, next_chunk_id=len(chunks)))

            if (i + 1) % 2 == 0 or (i + 1) == total:
                pct = (i + 1) / total
                filled = int(bar_width * pct)
                bar = "█" * filled + " " * (bar_width - filled)
                print(f"[LLMChunker] {sample_id} {i+1:>3}/{total} sessions [{bar}] {pct*100:5.1f}%", flush=True)

        return chunks
