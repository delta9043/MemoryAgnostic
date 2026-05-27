from typing import List

from core.chunker.base import BaseChunker
from data.schema import Chunk, Turn


class NoChunker(BaseChunker):
    """
    turn 1개 = chunk 1개로 반환하는 chunker.
    """

    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        return [
            Chunk(
                chunk_id=i,
                turns=[turn],
                text=f"{turn.speaker}: {turn.content}",
            )
            for i, turn in enumerate(turns)
        ]