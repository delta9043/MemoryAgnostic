import os
import sys
import importlib
from typing import List, Optional

from core.memory.base import BaseMemoryBackend, normalize_prediction
from data.schema import Chunk

# SimpleMem repo를 import 가능하도록 sys.path에 추가
SIMPLEMEM_PATH = "/data/delta9043/repos/SimpleMem"

# SimpleMem 메모리 윈도우 기본값(=원본 config.py 값)
# NoChunker일 때 사용
_NATIVE_WINDOW, _NATIVE_OVERLAP = 20, 5

# Chunker 사용 시 해당 값으로 오버라이딩
_CHUNKED_WINDOW, _CHUNKED_OVERLAP = 1, 0


def _read_chunker_type():
    """
    실행 중인 config(main.py의 --config)의 pipeline.chunker.type을 읽는다.
    못 읽으면 None 반환.
    """
    path = None
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
            break
        if a.startswith("--config="):
            path = a.split("=", 1)[1]
            break
    if not path or not os.path.exists(path):
        return None
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("pipeline", {}).get("chunker", {}).get("type")
    except Exception:
        return None


def _resolve_window_overlap():
    """SimpleMem WINDOW_SIZE/OVERLAP_SIZE 결정. (window, overlap, 출처) 반환.

    우선순위:
    1. 환경변수 SIMPLEMEM_WINDOW_SIZE/SIMPLEMEM_OVERLAP_SIZE (수동 강제).
    2. config의 chunker 타입: NoChunker → 원본(20/5), 그 외 → 1/0.
    """
    if ("SIMPLEMEM_WINDOW_SIZE" in os.environ
            or "SIMPLEMEM_OVERLAP_SIZE" in os.environ):
        win = int(os.environ.get("SIMPLEMEM_WINDOW_SIZE", _NATIVE_WINDOW))
        ovl = int(os.environ.get("SIMPLEMEM_OVERLAP_SIZE", _NATIVE_OVERLAP))
        return win, ovl, "env"
    chunker = _read_chunker_type()
    if chunker and chunker != "NoChunker":
        return _CHUNKED_WINDOW, _CHUNKED_OVERLAP, f"chunker={chunker}"
    return _NATIVE_WINDOW, _NATIVE_OVERLAP, f"chunker={chunker or 'unknown'}"


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
        win, ovl, src = _resolve_window_overlap()
        try:
            cfg = importlib.import_module("config")
            cfg.WINDOW_SIZE = win
            cfg.OVERLAP_SIZE = ovl
            print(f"[simplemem] config override: WINDOW_SIZE={win} "
                  f"OVERLAP_SIZE={ovl} (src={src})", flush=True)
        except Exception as e:
            # config 모듈명이 다르면 덮어쓰기 실패 → 크래시 대신 경고만.
            # 이 경우 SimpleMem config.py의 파일 값이 그대로 쓰이므로 확인 필요.
            print(f"[simplemem][WARN] WINDOW_SIZE/OVERLAP_SIZE override 실패 "
                  f"({type(e).__name__}: {e}). config.py 파일 값이 사용됨.",
                  flush=True)

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
            "enable_parallel_processing": True,
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
        - dialogue_id 카운터를 초기화한다.
        """
        self.system.vector_store.clear()
        self._dialogue_id_counter = 0