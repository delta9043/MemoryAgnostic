from typing import List

from core.compressor.base import BaseCompressor
from data.schema import Turn


class NoCompressor(BaseCompressor):
    # 입력 turns를 그대로 반환 (압축 없음)
    # 파이프라인 테스트 및 Compressor 비교 실험의 baseline으로 사용

    def run(self, turns: List[Turn]) -> List[Turn]:
        return turns