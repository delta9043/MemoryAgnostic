import sys
from typing import Dict, List, Optional, Union

from core.memory.base import BaseMemoryBackend, strip_think
from data.schema import Chunk

# A-Mem repo를 import 가능하도록 sys.path에 추가
AMEM_PATH = "/data/delta9043/repos/A-mem"
if AMEM_PATH not in sys.path:
    sys.path.insert(0, AMEM_PATH)

# A-Mem 내부 모듈 import
from test_advanced_robust import RobustAdvancedMemAgent  # noqa: E402
from llm_text_parsers import parse_plain_text_answer  # noqa: E402

# LoCoMo category 번호. data/locomo_loader.py의 LOCOMO_CATEGORY와 반대 방향 맵이다.
CATEGORY_MAP = {
    "multi_hop": 1,
    "temporal": 2,
    "open_domain": 3,
    "single_hop": 4,
    "adversarial": 5,
}

# 논문 §4.2: "we primarily employ k=10 for top-k memory selection".
DEFAULT_RETRIEVE_K = 10

# 원본 test_advanced.py가 쓰던 json_schema. structured output이면 예측이
# {"answer": "..."} 한 필드로 와서 "Short answer:" 에코 같은 잡토큰이 안 섞인다.
ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

KEYWORDS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {"keywords": {"type": "string"}},
            "required": ["keywords"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _resolve_retrieve_k(spec: Union[int, Dict[str, int], None]):
    """retrieve_k 설정을 (카테고리별 맵, 기본값)으로 푼다.

    스칼라(10) 또는 dict({default: 10, open_domain: 50}) 둘 다 받는다.
    dict 형태는 논문 Appendix A.5 Table 8(모델별/카테고리별 다른 값의 k)을 표현하기 위한 것.
    """
    if spec is None:
        return {}, DEFAULT_RETRIEVE_K

    if isinstance(spec, int):
        return {}, spec

    if not isinstance(spec, dict):
        raise ValueError(f"retrieve_k must be int or dict, got {spec!r}")

    per_category = {k: v for k, v in spec.items() if k != "default"}
    unknown = set(per_category) - set(CATEGORY_MAP)
    if unknown:
        raise ValueError(
            f"retrieve_k에 모르는 카테고리 키: {sorted(unknown)}. "
            f"허용: {sorted(CATEGORY_MAP)} + 'default'"
        )

    return per_category, spec.get("default", DEFAULT_RETRIEVE_K)


class AMemBackend(BaseMemoryBackend):
    """
    A-Mem(RobustAdvancedMemAgent)을 wrapping하는 MemoryBackend.

    동작 방식:
    1. build(chunks): 각 chunk의 turn들을 원본과 같은 문자열로 add_memory 호출
    2. query(question, category): agent.answer_question() 호출, prediction 반환
    3. reset(): agent 인스턴스를 새로 생성하여 메모리 초기화

    LLM은 외부 vLLM 서버(OpenAI 호환 API)에 요청하는 방식으로 호출된다.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "dummy",
        retrieve_k: Union[int, Dict[str, int], None] = None,
        temperature_c5: float = 0.5,
        structured_output: bool = True,
    ):
        """
        Args:
            base_url: vLLM 서버 주소 (예: "http://localhost:8000/v1")
            model: 사용할 모델 이름 (예: "Qwen/Qwen3-14B")
            api_key: API 키 (vLLM은 dummy 가능)
            retrieve_k: 검색할 메모리 수. int 또는 {default, <category>: k} dict.
            temperature_c5: category 5 질문의 temperature (=0.7), 나머지 category는 0.5로 A-Mem에서 지정함.
            structured_output: 답변·키워드 생성을 json_schema로 받을지
                (논문 설정. vLLM guided decoding 필요)
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.structured_output = structured_output

        self._k_by_category, self._k_default = _resolve_retrieve_k(retrieve_k)
        print(f"[amem] retrieve_k default={self._k_default} "
              f"per_category={self._k_by_category or '없음'}")

        # base_url에서 host/port 파싱
        # 예: "http://localhost:8000/v1" -> host="http://localhost", port=8000
        self._host, self._port = self._parse_base_url(base_url)

        self.agent = self._create_agent()

    def _parse_base_url(self, base_url: str):
        # "http://localhost:8000/v1" → ("http://localhost", 8000)
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        host = f"{parsed.scheme}://{parsed.hostname}"
        port = parsed.port if parsed.port else 8000
        return host, port

    def _create_agent(self) -> RobustAdvancedMemAgent:
        return RobustAdvancedMemAgent(
            model=self.model,
            backend="vllm",  # vLLM OpenAI 호환 API 사용
            retrieve_k=self._k_default,
            temperature_c5=self.temperature_c5,
            sglang_host=self._host,
            sglang_port=self._port,
        )

    def build(self, chunks: List[Chunk]) -> None:
        """chunk 하나를 메모리 노트 하나로 추가."""
        for chunk in chunks:
            if not chunk.turns:
                continue

            content = "\n".join(
                "Speaker " + turn.speaker + "says : " + turn.content
                for turn in chunk.turns
            )
            # 첫 번째 turn의 timestamp 사용
            timestamp = chunk.turns[0].timestamp
            self.agent.add_memory(content, time=timestamp)

    def query(self, question: str, category: Optional[str] = None,
              answer: Optional[str] = None) -> str:
        """질문에 대한 답변을 반환."""
        # category가 없으면 기본값 1 사용
        cat = CATEGORY_MAP.get(category, 1) if category is not None else 1
        ans = answer if cat == 5 else ""

        # category별 k 값 설정
        self.agent.retrieve_k = self._k_by_category.get(category, self._k_default)

        schemas = (
            {"answer_format": ANSWER_SCHEMA, "keywords_format": KEYWORDS_SCHEMA}
            if self.structured_output else {}
        )
        prediction, _, _ = self.agent.answer_question(
            question=question,
            category=cat,
            answer=ans or "",
            **schemas,
        )
        # thinking 블록 제거 코드
        # "Short answer:" prefix 제거는 진행하지 않음
        ## 만일 점수가 너무 낮다면 해당 부분에 prefix 제거 추가하기
        return parse_plain_text_answer(strip_think(prediction))

    def reset(self) -> None:
        """메모리 초기화. 다음 샘플 처리 전에 호출."""
        self.agent = self._create_agent()
