import sys
from typing import List, Optional

from core.memory.base import BaseMemoryBackend
from data.schema import Chunk

# SimpleMem repo를 import 가능하도록 sys.path에 추가
SIMPLEMEM_PATH = "/data/delta9043/repos/SimpleMem"
if SIMPLEMEM_PATH not in sys.path:
    sys.path.insert(0, SIMPLEMEM_PATH)

# SimpleMem 내부 모듈 import (sys.path 추가 후 가능)
from main import SimpleMemSystem  # noqa: E402
from models.memory_entry import Dialogue  # noqa: E402


class SimpleMemBackend(BaseMemoryBackend):
    """
    SimpleMem(https://github.com/aiming-lab/SimpleMem)을 wrapping하는 MemoryBackend.

    동작 방식:
    1. build(chunks): 각 chunk의 turn들을 하나의 텍스트로 합쳐 Dialogue 하나로 변환.
       window_size=1이므로 Dialogue 하나가 SimpleMem의 메모리 단위 하나가 된다.
    2. query(question): system.ask(question)을 그대로 호출한다.
    3. reset(): VectorStore를 비워 다음 샘플 처리에 대비한다.

    LLM은 외부 vLLM 서버(OpenAI 호환 API)에 요청하는 방식으로 호출된다.
    """
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "dummy",
        db_path: str = "./lancedb_data",
        table_name: Optional[str] = None,
        clear_db_on_init: bool = True,
    ):
        """
        Args:
            base_url: vLLM 서버 주소 (예: "http://localhost:8000/v1")
            model: 사용할 모델 이름 (예: "Qwen/Qwen3-14B")
            api_key: API 키 (vLLM은 dummy 가능)
            db_path: LanceDB 저장 경로
            table_name: 메모리 테이블 이름. None이면 SimpleMem 기본값 사용.
            clear_db_on_init: 인스턴스 생성 시 DB를 비울지 여부
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.db_path = db_path
        self.table_name = table_name
        self.clear_db_on_init = clear_db_on_init

        # 누적 dialogue_id 카운터 (chunk 사이에서도 unique해야 함)
        self._dialogue_id_counter = 0

        # SimpleMemSystem 인스턴스 생성
        self.system = self._create_system(clear_db=clear_db_on_init)

    def _create_system(self, clear_db: bool) -> SimpleMemSystem:
        """SimpleMemSystem 인스턴스를 생성한다."""
        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "db_path": self.db_path,
            "clear_db": clear_db,
            "enable_parallel_processing": False,
        }
        if self.table_name is not None:
            kwargs["table_name"] = self.table_name
        return SimpleMemSystem(**kwargs)

    def build(self, chunks: List[Chunk]) -> None:
        dialogues = []
        for chunk in chunks:
            self._dialogue_id_counter += 1
            # chunk 안의 turn들을 하나의 텍스트로 합침
            content = "\n".join(
                f"{turn.speaker}: {turn.content}" for turn in chunk.turns
            )
            # 첫 번째 turn의 timestamp 사용
            timestamp = chunk.turns[0].timestamp if chunk.turns else None

            dialogues.append(Dialogue(
                dialogue_id=self._dialogue_id_counter,
                speaker="chunk",
                content=content,
                timestamp=timestamp,
            ))

        # window_size=1이므로 Dialogue 하나 = 메모리 단위 하나
        self.system.add_dialogues(dialogues)
        self.system.finalize()

    def query(self, question: str) -> str:
        """질문에 대한 답변을 반환한다."""
        return self.system.ask(question)

    def reset(self) -> None:
        """
        메모리를 초기화한다. 다음 샘플 처리 전에 호출.
        - VectorStore를 비운다.
        - dialogue_id 카운터를 초기화한다.
        """
        self.system.vector_store.clear()
        self._dialogue_id_counter = 0