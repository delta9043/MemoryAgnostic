import os
import sys
from typing import List, Optional

from core.memory.base import BaseMemoryBackend, normalize_prediction
from data.schema import Chunk

# A-Mem repo를 import 가능하도록 sys.path에 추가
AMEM_PATH = "/data/delta9043/repos/A-mem"
if AMEM_PATH not in sys.path:
    sys.path.insert(0, AMEM_PATH)

# A-Mem 내부 모듈 import
from test_advanced_robust import RobustAdvancedMemAgent  # noqa: E402

CATEGORY_MAP = {
    "single_hop": 1,
    "temporal": 2,
    "open_domain": 3,
    "multi_hop": 4,
    "adversarial": 5,
}

# ── 디버깅 토글 (환경변수로 제어; query()에서 사용) ──────────────────────────
# AMEM_NO_THINK=0  → /no_think 비활성화(=thinking 켜짐). 빈 답 재현/비교용.
# AMEM_DEBUG=1     → query마다 raw 출력/think 포함 여부/길이를 stdout에 찍음.
_FALSEY = ("", "0", "false", "False")
NO_THINK = os.environ.get("AMEM_NO_THINK", "1") not in _FALSEY
DEBUG = os.environ.get("AMEM_DEBUG", "") not in _FALSEY

class AMemBackend(BaseMemoryBackend):
    """
    A-Mem(RobustAdvancedMemAgent)을 wrapping하는 MemoryBackend.

    동작 방식:
    1. build(chunks): 각 chunk의 turn들을 "Speaker X says: ..." 형식으로 add_memory 호출
    2. query(question, category): agent.answer_question() 호출, prediction 반환
    3. reset(): agent 인스턴스를 새로 생성하여 메모리 초기화

    LLM은 외부 vLLM 서버(OpenAI 호환 API)에 요청하는 방식으로 호출된다.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "dummy",
        retrieve_k: int = 5,    # default : 10 (Qwen의 한정된 context 때문에 5로 줄임)
        temperature_c5: float = 0.5,
    ):
        """
        Args:
            base_url: vLLM 서버 주소 (예: "http://localhost:8000/v1")
            model: 사용할 모델 이름 (예: "Qwen/Qwen3-14B")
            api_key: API 키 (vLLM은 dummy 가능)
            retrieve_k: 검색할 메모리 수
            temperature_c5: category 5 질문의 temperature
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

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
            retrieve_k=self.retrieve_k,
            temperature_c5=self.temperature_c5,
            sglang_host=self._host,
            sglang_port=self._port,
        )

    def build(self, chunks: List[Chunk]) -> None:
        """chunk 하나를 메모리 노트 하나로 추가."""
        for chunk in chunks:
            if not chunk.turns:
                continue
            # chunk 안의 turn들을 하나의 텍스트로 합쳐 노트 1개 생성
            content = "\n".join(
                f"Speaker {turn.speaker} says: {turn.content}"
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
        # adversarial(category 5)는 ['Not mentioned', answer] 2지선다 프롬프트를
        # 구성하므로 gold answer가 필요하다. answer=""면 선택지가 빈칸으로 붕괴해
        # 모델이 항상 'Not mentioned'를 고른다. 그 외 카테고리(1~4)는 answer를
        # 프롬프트에 쓰므로, gold가 새어들어가지 않도록 cat 5일 때만 넘긴다.
        ans = answer if cat == 5 else ""
        # Qwen3 thinking 비활성화: 서버의 --override-generation-config
        # '{"enable_thinking": false}'는 generation_config가 아니라 chat-template
        # 인자라 무시된다(빈 <think></think>가 안 붙는 걸로 확인). thinking이 켜진
        # 채로 긴 프롬프트(특히 chunk 사용 시)에서 추론이 max_tokens를 다 먹고
        # 잘리면, strip_think이 닫힘 없는 <think>를 통째로 지워 답이 빈칸이 된다.
        # 프롬프트에 /no_think 소프트 스위치를 넣어 답변 생성 단계의 thinking을 끈다.
        # (chat 호출에선 템플릿이 /no_think를 제거하지만, retrieve용 임베딩에는
        #  접미사가 남아 약간의 노이즈가 됨 — 빈 답 손실 대비 허용 가능한 trade-off.)
        q = f"{question} /no_think" if NO_THINK else question
        prediction, _, _ = self.agent.answer_question(
            question=q,
            category=cat,
            answer=ans or "",
        )
        pred = normalize_prediction(prediction)
        if DEBUG:
            has_think = "<think>" in prediction
            closed = "</think>" in prediction
            print(
                f"[amem][debug] cat={cat} no_think={NO_THINK} "
                f"think={has_think}/closed={closed} "
                f"raw_len={len(prediction)} pred_len={len(pred)}\n"
                f"  Q   : {question[:80]}\n"
                f"  raw : {prediction[:300]!r}\n"
                f"  pred: {pred[:200]!r}",
                flush=True,
            )
        if not pred:
            # /no_think 이후에도 빈 답이면 원인 추적용으로 raw를 남긴다.
            print(f"[amem][empty] cat={cat} raw={prediction[:200]!r}", flush=True)
        return pred

    def reset(self) -> None:
        """메모리 초기화. 다음 샘플 처리 전에 호출."""
        self.agent = self._create_agent()