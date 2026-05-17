from abc import ABC, abstractmethod
from typing import List

from data.schema import Chunk, Turn


class BaseChunker(ABC):
    # 모든 Chunker가 구현해야 하는 인터페이스
    # 입력: List[Turn], 출력: List[Chunk]

    @abstractmethod
    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        pass