import re
from abc import ABC, abstractmethod
from typing import List

from data.schema import Chunk


def strip_think(text: str) -> str:
    """LLM 응답에서 Qwen3 등 reasoning 모델의 <think> 블록을 제거한다.

    - 닫힌 <think>...</think> 블록 제거.
    - 닫힘 태그 없이 열린 <think>(추론이 max_tokens에 잘린 경우)는 그 이후를 모두 제거.
    """
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def strip_answer_prefix(text: str) -> str:
    """프롬프트 꼬리("Short answer:" 등)를 모델이 에코한 prefix를 제거한다.

    A-Mem robust 프롬프트가 "... Question: {q} Short answer:" 로 끝나 모델이
    "Short answer: <답>" 형태로 답을 뱉으면, F1/BLEU 채점 시 불필요한 토큰이
    섞여 점수가 깎인다. 원본 A-Mem은 JSON schema로 answer만 추출해 이 문제가
    없었으나, robust 변형은 그 장치가 없어 wrapper에서 보정한다.
    """
    if not text:
        return text
    return re.sub(r"^\s*(?:short\s+answer|answer)\s*:\s*", "", text,
                  flags=re.IGNORECASE).strip()


def normalize_prediction(text: str) -> str:
    """채점 전 예측 답변 정규화: <think> 제거 후 answer prefix 제거."""
    return strip_answer_prefix(strip_think(text))


class BaseMemoryBackend(ABC):
    """
    모든 MemoryBackend가 구현해야 하는 인터페이스.

    역할:
    - 우리 파이프라인에서 생성된 List[Chunk]를 받아 메모리를 구축한다.
    - 자연어 질문에 대한 답변을 반환한다.
    - 샘플 사이 메모리를 초기화한다.

    구현체는 SimpleMem, A-Mem, Mem0 등 각 Memory 시스템을 wrapping한다.
    """

    @abstractmethod
    def build(self, chunks: List[Chunk]) -> None:
        """
        chunks를 메모리에 저장한다.

        Args:
            chunks: List[Chunk] - 우리 파이프라인이 생성한 chunk 리스트.
                각 chunk는 turn들의 묶음이며 chunk 경계가 메모리 단위가 된다.
        """
        pass

    @abstractmethod
    def query(self, question: str, category=None) -> str:
        """
        질문에 대한 답변을 반환한다.
s
        Args:
            question: str - 자연어 질문

        Returns:
            str - LLM이 생성한 답변
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        메모리를 초기화한다. 다음 샘플 처리 전에 호출한다.
        이전 샘플의 메모리가 다음 샘플에 섞이지 않도록 한다.
        """
        pass