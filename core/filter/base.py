from abc import ABC, abstractmethod
from typing import List

from data.schema import Turn


class BaseFilter(ABC):
    # 모든 Filter가 구현해야 하는 인터페이스
    # 입력: List[Turn], 출력: List[Turn]

    @abstractmethod
    def run(self, turns: List[Turn]) -> List[Turn]:
        pass