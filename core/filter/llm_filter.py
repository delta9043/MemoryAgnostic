import json
import re
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.filter.base import BaseFilter
from data.schema import Turn


# SimpleMem 논문 Instruction 1 (Information Filtering) 기반 프롬프트
# 형식은 SimpleMem 구현체(_build_extraction_prompt) 스타일을 따름
PROMPT_TEMPLATE = """Your task is to filter a single utterance by removing parts that carry no informative content.

[Input Utterance]
Speaker: {speaker}
Content: {content}

[Requirements]
1. **Discard social filler**: Remove acknowledgements and conversational routines that introduce no new factual or semantic information.
2. **Discard redundant confirmations**: Discard redundant confirmations unless they modify or finalize a decision.
3. **Keep informative content unchanged**: Do NOT paraphrase, summarize, or add any new information. Only remove the non-informative parts.
4. **If no informative content is present**: Output an empty string.

[Output Format]
Return a JSON object:

```json
{{"filtered_content": "remaining informative content, or empty string if none"}}
```

[Example]
Input:
Speaker: Alice
Content: Oh yeah, sure! That sounds great. Let's meet at Central Park at 3pm on Saturday.

Output:
```json
{{"filtered_content": "Let's meet at Central Park at 3pm on Saturday."}}
```

[Example]
Input:
Speaker: Bob
Content: Haha yeah, totally! I know right?

Output:
```json
{{"filtered_content": ""}}
```

Now process the above utterance. Return ONLY the JSON object, no other explanations.
"""


class LLMFilter(BaseFilter):
    # Turn 내부 content에서 informative하지 않은 부분만 제거하는 Filter
    # 입력 Turn 개수와 출력 Turn 개수는 동일 (content만 수정, 빈 문자열 가능)
    # JSON 파싱 실패 시 원본 content 유지하고 failure_count/failed_turn_ids 누적

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = False,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        # 모델/토크나이저 1회 로드 후 재사용 (생성자에서 로드)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # device_map="auto": 다중 GPU 자동 분배 (Qwen3-32B bf16 ≈ 64GB, RTX 3090 단일 24GB 불가)
        model_kwargs = {
            "dtype": dtype,
            "device_map": "auto",
        }
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        # 실패 통계 (누적)
        self.failure_count: int = 0
        self.failed_turn_ids: List[str] = []

    def reset_failure_stats(self) -> None:
        # 누적된 실패 통계 초기화
        self.failure_count = 0
        self.failed_turn_ids = []

    def get_failure_report(self) -> dict:
        # 현재까지 누적된 실패 통계 반환
        return {
            "failure_count": self.failure_count,
            "failed_turn_ids": list(self.failed_turn_ids),
        }

    def _build_prompt(self, turn: Turn) -> str:
        return PROMPT_TEMPLATE.format(speaker=turn.speaker, content=turn.content)

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        # Qwen3 chat template 사용, thinking 비활성화
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # 입력 부분 제외하고 새로 생성된 토큰만 디코딩
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _extract_json(self, raw: str) -> Optional[dict]:
        # robust JSON 추출
        # 1. <think> 블록 제거 (Qwen3 thinking 모드 대비)
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

        # 2. 마크다운 코드 펜스 제거
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)

        # 3. 첫 '{'부터 마지막 '}'까지 추출
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _filter_one(self, turn: Turn) -> str:
        # 단일 turn에 대해 필터링된 content 반환
        # 실패 시 원본 content 반환 + 실패 통계 누적
        prompt = self._build_prompt(turn)
        raw = self._generate(prompt)
        parsed = self._extract_json(raw)

        if parsed is None or "filtered_content" not in parsed:
            self.failure_count += 1
            self.failed_turn_ids.append(turn.turn_id)
            return turn.content

        filtered = parsed["filtered_content"]
        if not isinstance(filtered, str):
            self.failure_count += 1
            self.failed_turn_ids.append(turn.turn_id)
            return turn.content

        return filtered

    def run(self, turns: List[Turn]) -> List[Turn]:
        result: List[Turn] = []
        total = len(turns)
        bar_width = 20

        for i, turn in enumerate(turns):
            new_content = self._filter_one(turn)
            new_turn = Turn(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                content=new_content,
                timestamp=turn.timestamp,
                session_id=turn.session_id,
                metadata=dict(turn.metadata),
            )
            result.append(new_turn)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                pct = (i + 1) / total
                filled = int(bar_width * pct)
                bar = "█" * filled + " " * (bar_width - filled)
                print(f"[LLMFilter] {i+1:>4}/{total} turns [{bar}] {pct*100:5.1f}%", flush=True)

        return result