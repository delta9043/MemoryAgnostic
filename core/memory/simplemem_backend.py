import sys
import importlib
from typing import List, Optional

from core.memory.base import BaseMemoryBackend, normalize_prediction
from data.schema import Chunk

# SimpleMem repo를 import 가능하도록 sys.path에 추가
SIMPLEMEM_PATH = "/data/delta9043/repos/SimpleMem"


def _load_simplemem():
    """
    SimpleMem 모듈을 로드한다.
    """
    if SIMPLEMEM_PATH not in sys.path:
        sys.path.insert(0, SIMPLEMEM_PATH)

    # MemoryAgnostic의 core.* 를 임시 보관 후 제거
    saved = {k: v for k, v in sys.modules.items()
             if k == "core" or k.startswith("core.")}
    for k in saved:
        del sys.modules[k]

    try:
        SimpleMemSystem = importlib.import_module("main").SimpleMemSystem
        Dialogue = importlib.import_module("models.memory_entry").Dialogue
    finally:
        # 항상 MemoryAgnostic의 core.* 복원
        sys.modules.update(saved)

    return SimpleMemSystem, Dialogue


SimpleMemSystem, Dialogue = _load_simplemem()


class SimpleMemBackend(BaseMemoryBackend):
    """
    SimpleMem(https://github.com/aiming-lab/SimpleMem)을 wrapping하는 MemoryBackend.

    동작 방식:
    1. build(chunks): turn 1개 = Dialogue 1개로 만들고, chunk 하나를 통째로
       add_dialogue_group에 넘긴다 → chunk 1개 = 메모리 추출 1회.
       SimpleMem 내부 슬라이딩 윈도우(dialogue_buffer/WINDOW_SIZE)는 거치지 않는다.
    2. query(question): system.ask(question)을 그대로 호출한다.
    3. reset(): VectorStore와 builder 상태를 비워 다음 샘플에 대비한다.

    LLM 입력 양식은 SimpleMem이 만든다(Dialogue.__str__ + 추출 프롬프트). 우리가
    문자열을 조립하지 않으므로 native와 동일한 양식이 유지되고, 청킹 조건 간에
    달라지는 것은 '한 번에 몇 turn이 들어가는가'뿐이다.

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

    def _create_system(self, clear_db: bool) -> "SimpleMemSystem":
        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "db_path": self.db_path,
            "clear_db": clear_db,
            # build는 add_dialogue_group(순차)만 쓰므로 미사용. 실수로 add_dialogues가
            # 불려도 병렬 경로(previous_entries 공유)로 새지 않도록 False.
            "enable_parallel_processing": False,
        }
        if self.table_name is not None:
            kwargs["table_name"] = self.table_name
        return SimpleMemSystem(**kwargs)

    def build(self, chunks: List[Chunk]) -> None:
        """chunk 1개 = 메모리 추출 1회.

        turn을 합치지 않고 Dialogue 1개씩 넘긴다. 합치면 speaker 자리에 "chunk"가
        들어가고 turn별 시각이 사라져 native와 양식이 달라진다.
        """
        n_groups = n_turns = 0
        for chunk in chunks:
            if not chunk.turns:
                continue
            dialogues = []
            for turn in chunk.turns:
                self._dialogue_id_counter += 1
                dialogues.append(Dialogue(
                    dialogue_id=self._dialogue_id_counter,
                    speaker=turn.speaker,
                    content=turn.content,
                    timestamp=turn.timestamp,
                ))
            self.system.add_dialogue_group(dialogues)
            n_groups += 1
            n_turns += len(dialogues)

        # Tantivy FTS 인덱스는 증분이 아니다. 그룹마다 나눠 넣으므로 첫 그룹만
        # 색인된 채로 남는다 → 전체 위에 한 번 다시 만든다. (원본은 add_dialogues로
        # 한 번에 넣어 이 문제가 없었다.)
        self.system.vector_store.rebuild_fts_index()

        print(f"[simplemem] build 완료: 추출 호출 {n_groups}회 / turn {n_turns}개 "
              f"(내부 윈도잉 미사용, FTS 인덱스 재생성)", flush=True)

    def query(self, question: str, category: str = None,
              answer: str = None) -> str:
        """질문에 대한 답변을 반환한다.

        answer는 인터페이스 통일을 위해 받지만 SimpleMem.ask()는 사용하지 않는다.
        """
        return normalize_prediction(self.system.ask(question))

    def reset(self) -> None:
        """
        메모리를 초기화한다. 다음 샘플 처리 전에 호출.
        - VectorStore를 비운다.
        - MemoryBuilder 상태를 비운다. previous_entries가 남으면 다음 샘플의
          첫 추출 프롬프트에 이전 샘플의 사실이 섞여 들어간다.
        - dialogue_id 카운터를 초기화한다.
        """
        self.system.vector_store.clear()
        builder = self.system.memory_builder
        builder.previous_entries = []
        builder.dialogue_buffer = []
        builder.processed_count = 0
        self._dialogue_id_counter = 0
