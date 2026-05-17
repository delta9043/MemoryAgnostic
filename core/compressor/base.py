from abc import ABC, abstractmethod
from typing import List

from data.schema import Turn


class BaseCompressor(ABC):
    # 모든 Compressor가 구현해야 하는 인터페이스
    # 입력: List[Turn], 출력: List[Turn]
    # 압축 전 원본 텍스트는 turn.metadata["original_content"]에 백업한다

    @abstractmethod
    def run(self, turns: List[Turn]) -> List[Turn]:
        pass