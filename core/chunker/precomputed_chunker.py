import json
from typing import Dict, List, Optional

from core.chunker.base import BaseChunker
from data.schema import Chunk, Turn


class PrecomputedChunker(BaseChunker):
    """
    미리 생성해 디스크에 저장한 chunk(JSON)를 로드해서 그대로 반환하는 chunker.

    run_chunker.py로 한 번만 청킹해 JSON에 저장한 뒤, A-Mem/SimpleMem 두 backend가 동일한 청크 파일을 입력으로 쓰게 한다\.

    매칭 키 = sample_id:
        loader가 모든 Turn에 metadata={"source": ..., "sample_id": ...}를 붙인다.

    입력 JSON 형식 (run_chunker.py 출력과 동일):
        [
            {
                "sample_id": "conv-26",
                "chunks": [
                    {
                        "chunk_id": 0,
                        "text": "...",
                        "metadata": {...},
                        "turns": [
                            {"turn_id": "...", "speaker": "...", "content": "...",
                             "timestamp": "...", "session_id": "...", "metadata": {...}},
                            ...
                        ]
                    },
                    ...
                ]
            },
            ...
        ]
    """

    def __init__(self, chunks_path: str):
        self.chunks_path = chunks_path
        # sample_id -> List[Chunk]. _load()에서 한 번만 채운다.
        self._by_sample: Optional[Dict[str, List[Chunk]]] = None

    def _load(self) -> None:
        # 파일 파싱 + dict/list → Turn/Chunk 객체 복원 + sample_id 색인을 수행.
        # 이미 로딩됐으면 즉시 반환(chunk()가 샘플마다 불려도 재작업 없음)
        if self._by_sample is not None:
            return

        with open(self.chunks_path, encoding="utf-8") as f:
            data = json.load(f)

        index: Dict[str, List[Chunk]] = {}
        for entry in data:
            sample_id = entry["sample_id"]
            chunks = [
                Chunk(
                    chunk_id=rc["chunk_id"],
                    turns=[Turn(**t) for t in rc["turns"]],  # dict → Turn 객체 복원
                    text=rc["text"],
                    metadata=rc.get("metadata", {}),
                )
                for rc in entry.get("chunks", [])
            ]
            index[sample_id] = chunks

        self._by_sample = index

    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        if not turns:
            return []
        self._load()

        # 1) 입력 turns가 속한 샘플 식별 (loader가 넣어준 metadata 사용)
        sample_id = turns[0].metadata.get("sample_id")
        chunks = self._by_sample.get(sample_id)
        if chunks is None:
            raise KeyError(
                f"PrecomputedChunker: no precomputed chunks for sample_id "
                f"{sample_id!r} (file: {self.chunks_path}). "
                f"Available sample_ids: {list(self._by_sample.keys())}. "
                f"sample_id가 없거나 다르면 청크 파일과 현재 데이터가 어긋난 것입니다."
            )

        # 2) 무결성 검증: 청크 turn 수 합 == 입력 turn 수.
        #    precompute는 NoFilter 기준이라, 현재 입력에 필터가 끼면 수가 달라져 여기서 걸린다.
        total_chunk_turns = sum(len(c.turns) for c in chunks)
        if total_chunk_turns != len(turns):
            raise ValueError(
                f"PrecomputedChunker: turn count mismatch for sample {sample_id!r} "
                f"(precomputed={total_chunk_turns}, input={len(turns)}). "
                f"청킹에 쓴 turn 시퀀스와 현재 입력이 다릅니다."
            )
        return chunks
