from typing import List

from core.chunker.base import BaseChunker
from data.schema import Chunk, Turn


class FixedSizeChunker(BaseChunker):
    # turns를 고정 크기 window로 묶어서 Chunk 리스트로 반환
    # overlap: 인접 chunk 간 겹치는 turn 수

    def __init__(self, window: int = 5, overlap: int = 0):
        # 오류 검증
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if not (0 <= overlap < window):
            raise ValueError(f"overlap must be >= 0 and < window, got overlap={overlap}, window={window}")
        
        self.window = window
        self.overlap = overlap

    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        chunks = []
        step = self.window - self.overlap
        chunk_id = 0

        for start in range(0, len(turns), step):
            chunk_turns = turns[start:start + self.window]
            if not chunk_turns:
                break

            # turns의 content를 "speaker: content" 형식으로 이어붙여 text 생성
            text = "\n".join(f"{t.speaker}: {t.content}" for t in chunk_turns)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                turns=chunk_turns,
                text=text,
                metadata={
                    "start_turn_id": chunk_turns[0].turn_id,
                    "end_turn_id": chunk_turns[-1].turn_id,
                },
            ))
            chunk_id += 1

        return chunks