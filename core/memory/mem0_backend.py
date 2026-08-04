import json
import sys
from typing import List, Optional

from jinja2 import Template

from core.memory.base import BaseMemoryBackend, normalize_prediction
from data.schema import Chunk

# mem0 repo(OSS)를 import 가능하도록 sys.path에 추가.
MEM0_PATH = "/data/delta9043/repos/mem0"
if MEM0_PATH not in sys.path:
    sys.path.insert(0, MEM0_PATH)

from mem0 import Memory  # noqa: E402  (OSS 무료 경로. MemoryClient=유료 클라우드)


# mem0 공식 LoCoMo 평가 프롬프트(evaluation/prompts.py의 non-graph ANSWER_PROMPT와 동일).
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """


class Mem0Backend(BaseMemoryBackend):
    """
    Mem0를 wrapping하는 MemoryBackend.

    native mem0 LoCoMo 평가(evaluation/src/memzero)를 그대로 재현
    1. build(chunks): 화자 2명 각각의 메모리 뱅크를 만든다. 
                      한 chunk를 화자별로 role을 뒤집어(뱅크 주인=user, 상대=assistant) mem.add() 1회씩 호출한다.
                      → chunk 1개 = add() 2회
    2. query(question): 두 뱅크를 각각 search(top_k) → 공식 ANSWER_PROMPT로 답변 생성.
    3. reset(): 벡터스토어 컬렉션을 mem.reset()

    LLM은 외부 vLLM(OpenAI 호환 API), 임베더는 HuggingFace, 벡터스토어는 로컬 Qdrant를 쓴다.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        embedding_path: str,
        embedding_dims: int = 384,
        qdrant_path: str = "./qdrant_data/mem0",
        top_k: int = 10,               # native 기본값 10 (evaluation/src/memzero/search.py)
        max_tokens: int = 8192,
        api_key: str = "dummy",
        clear_on_init: bool = True,
    ):
        self.base_url = base_url
        self.model = model
        self.embedding_path = embedding_path
        self.embedding_dims = embedding_dims
        self.qdrant_path = qdrant_path
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.api_key = api_key

        self.mem = Memory.from_config(self._build_config())
        self._patch_thinking_off()
        self._speakers: List[str] = []   # build에서 채워짐(첫 등장 순 화자 2명)

        # Intialization
        if clear_on_init:
            self.mem.reset()

    def _build_config(self) -> dict:
        return {
            "llm": {
                "provider": "vllm",
                "config": {
                    "model": self.model,
                    "vllm_base_url": self.base_url,
                    "api_key": self.api_key,
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self.embedding_path,
                    "embedding_dims": self.embedding_dims,
                    "model_kwargs": {"device": "cuda"},
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "embedding_model_dims": self.embedding_dims,
                    "path": self.qdrant_path,
                    "on_disk": True,
                },
            },
        }

    def _patch_thinking_off(self) -> None:
        # Qwen3 thinking off
        _orig = self.mem.llm.generate_response

        def _wrapped(**kwargs):
            kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
            return _orig(**kwargs)

        self.mem.llm.generate_response = _wrapped

    def build(self, chunks: List[Chunk]) -> None:
        """chunk를 timestamp(=세션) 연속 구간으로 나눠 구간마다 add()한다 (뱅크당 1회씩).

        mem0는 add() 1회에 timestamp를 1개만 받는다. 
        cross-session 청크(gold/llmchunk)를 통째로 넣으면 후행 세션 사실이 첫 세션 날짜로 저장돼 temporal 답이 틀어진다.
        """
        self._speakers = self._collect_speakers(chunks) # 현재 대화 세션의 화자 추출
        n_add = 0
        for chunk in chunks:
            for group in self._split_by_timestamp(chunk.turns): # Mem0는 timestamp가 달라지는 메모리를 못 만들기에 이를 분할
                timestamp = group[0].timestamp
                for owner in self._speakers:
                    messages = [
                        {
                            "role": "user" if t.speaker == owner else "assistant",
                            "content": f"{t.speaker}: {t.content}",   # native 형식 그대로
                        }
                        for t in group # group 안의 turn들을 t라는 이름으로 하나씩 호출
                    ]
                    self.mem.add(
                        messages,
                        user_id=owner,
                        metadata={"timestamp": timestamp},
                        infer=True,
                    )
                    n_add += 1
        print(f"[mem0] build 완료: 화자 {len(self._speakers)}명 / add() 호출 {n_add}회 "
              f"(chunk {len(chunks)}개, 세션 경계로 분할 저장)", flush=True)

    @staticmethod
    def _split_by_timestamp(turns: List) -> List[List]:
        # 연속된 같은 timestamp(=세션)끼리 묶는다. cross-session 청크만 실제로 나뉜다.
        groups: List[List] = []
        cur: List = []
        cur_ts = object()  # 첫 비교에서 반드시 다르도록
        for t in turns:
            if t.timestamp != cur_ts:
                if cur:
                    groups.append(cur)
                cur = [t]
                cur_ts = t.timestamp
            else:
                cur.append(t)
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def _collect_speakers(chunks: List[Chunk]) -> List[str]:
        # 첫 등장 순으로 고유 화자를 모은다.
        seen: List[str] = []
        for chunk in chunks:
            for t in chunk.turns:
                if t.speaker not in seen:
                    seen.append(t.speaker)
        return seen

    def query(self, question: str, category: Optional[str] = None,
              answer: Optional[str] = None) -> str:
        """두 뱅크를 각각 search → 공식 ANSWER_PROMPT로 답변 생성.
        category/answer는 인터페이스 통일용(공식 mem0 답변은 미사용)."""
        # 화자가 <2명이면 부족한 자리를 빈 뱅크로 채운다(프롬프트 자리 유지).
        s1 = self._speakers[0] if len(self._speakers) > 0 else "speaker_1"
        s2 = self._speakers[1] if len(self._speakers) > 1 else "speaker_2"

        mem1 = self._format_memories(self._search(s1, question))
        mem2 = self._format_memories(self._search(s2, question))

        prompt = Template(ANSWER_PROMPT).render(
            speaker_1_user_id=s1,
            speaker_2_user_id=s2,
            speaker_1_memories=mem1,
            speaker_2_memories=mem2,
            question=question,
        )
        # 답변 생성도 감싼 generate_response를 재사용(thinking off 동일 적용).
        resp = self.mem.llm.generate_response(
            messages=[{"role": "system", "content": prompt}],
        )
        return normalize_prediction(resp)

    def _search(self, user_id: str, question: str) -> List[dict]:
        res = self.mem.search(question, user_id=user_id, limit=self.top_k)
        return res.get("results", [])

    @staticmethod
    def _format_memories(memories: List[dict]) -> str:
        # native와 동일: "{timestamp}: {memory}" 리스트를 JSON으로 변환 (search.py:106-107).
        # 빈 memory의 경우 "[]"
        lines = [
            f"{m.get('metadata', {}).get('timestamp', '')}: {m.get('memory', '')}"
            for m in memories
        ]
        return json.dumps(lines, indent=4)

    def reset(self) -> None:
        """다음 샘플 전 초기화: 벡터스토어 컬렉션 drop+recreate."""
        self.mem.reset()
        self._speakers = []
