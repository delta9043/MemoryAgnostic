from typing import List

from core.filter.base import BaseFilter
from data.schema import Turn


class NoFilter(BaseFilter):
    # 입력 turns를 그대로 반환 (필터링 없음)
    # 파이프라인 테스트 및 Filter 비교 실험의 baseline으로 사용

    def run(self, turns: List[Turn]) -> List[Turn]:
        return turns