from abc import ABC, abstractmethod
from typing import List

from data.schema import Chunk


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
    def query(self, question: str) -> str:
        """
        질문에 대한 답변을 반환한다.

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