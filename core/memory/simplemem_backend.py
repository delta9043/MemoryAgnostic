import sys
import importlib
import random
from typing import List, Optional

from core.memory.base import BaseMemoryBackend, normalize_prediction
from data.schema import Chunk

# category 5(adversarial) 정답. native SimpleMem과 동일(대화에 없으면 이걸 골라야 정답).
CATEGORY5_GROUND_TRUTH = "Not mentioned in the conversation"

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
        semantic_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        structured_top_k: Optional[int] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
    ):
        """
        Args:
            base_url: vLLM 서버 주소 (예: "http://localhost:8000/v1")
            model: 사용할 모델 이름 (예: "Qwen/Qwen3-14B")
            api_key: API 키 (vLLM은 dummy 가능)
            db_path: LanceDB 저장 경로
            table_name: 메모리 테이블 이름. None이면 SimpleMem 기본값 사용.

        검색 파라미터(전부 None이면 SimpleMem config.py 기본값 = native 동작):
            semantic_top_k/keyword_top_k/structured_top_k: 검색 층별 entry 상한.
            enable_reflection: 컨텍스트가 불충분하다고 판단되면 추가 검색을 반복하는 기능.
            max_reflection_rounds: 그 반복 상한. 라운드마다 쿼리 1~3개 × semantic_top_k가
                누적되므로(제거 없음) 답변 프롬프트 크기를 좌우한다.
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.db_path = db_path
        self.table_name = table_name
        self.clear_db_on_init = clear_db_on_init
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k
        self.structured_top_k = structured_top_k
        self.enable_reflection = enable_reflection
        self.max_reflection_rounds = max_reflection_rounds

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

            # build는 add_dialogue_group(순차)만 쓰므로 실제로는 미사용.
            # 병렬 처리 Flag
            "enable_parallel_processing": False,

            # None이면 SimpleMem이 config.py 기본값을 사용.
            "enable_reflection": self.enable_reflection,
            "max_reflection_rounds": self.max_reflection_rounds,
        }
        if self.table_name is not None:
            kwargs["table_name"] = self.table_name
        system = SimpleMemSystem(**kwargs)

        # 층top_k는 k값 설정 코드 (자동 전달이 불가능하여 추가 작업 필요)
        for attr in ("semantic_top_k", "keyword_top_k", "structured_top_k"):
            value = getattr(self, attr)
            if value is not None:
                setattr(system.hybrid_retriever, attr, value)
        return system

    def build(self, chunks: List[Chunk]) -> None:
        """
        chunk 1개 = 메모리 추출 1회.
        turn마다 Dialogue 객체로 만들고, chunk마다 dialogues를 넘겨서 LLM을 호출하여 처리한다. 
        """
        # n_groups : 이번 build에서 추출 호출 수 (= chunk 수)
        # n_turns : 이번 build에서 처리한 turn 수
        # _dialogue_id_counter : turn마다의 일련번호

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

        # Tantivy FTS 인덱스 재생성
        self.system.vector_store.rebuild_fts_index()

        print(f"[simplemem] build 완료: 추출 호출 {n_groups}회 / turn {n_turns}개 "
              f"FTS 인덱스 생성 완료", flush=True)

    def query(self, question: str, category: str = None,
              answer: str = None) -> str:
        """
        질문에 대한 답변을 반환한다.

        category 5(adversarial)는 native SimpleMem처럼 특수 처리한다(2지 선다 처리). 그 외는 ask()로 자유형 생성.
        answer는 adversarial일 때 오답 후보(adversarial_answer)로만 쓴다.
        """
        if category == "adversarial":
            return self._answer_category5(question, answer)
        return normalize_prediction(self.system.ask(question))

    def _answer_category5(self, question: str, adversarial_answer: Optional[str]) -> str:
        # native SimpleMem generate_category5_answer 재현
        # "Not mentioned" vs 오답 후보 중 하나를 선택하도록 한다. (순서 셔플)

        contexts = self.system.hybrid_retriever.retrieve(question, enable_reflection=False)
        adv = adversarial_answer or "Unknown answer"
        options = [CATEGORY5_GROUND_TRUTH, adv]
        if random.random() < 0.5:
            options = [options[1], options[0]]

        context_str = self.system.answer_generator._format_contexts(contexts)

        # Build special prompt for category 5
        prompt = f"""
Based on the context below, answer the following question.

Context:
{context_str}

Question: {question}

Select the correct answer from the following two options. If the given answer is wrong or not answerable based on the context, you should choose "{CATEGORY5_GROUND_TRUTH}".

Option A: {options[0]}
Option B: {options[1]}

Requirements:
1. Choose the option that best matches the context
2. If neither answer is supported by the context, or if the provided specific answer is incorrect, choose "{CATEGORY5_GROUND_TRUTH}"
3. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your choice",
  "answer": "Your selected answer (either '{options[0]}' or '{options[1]}')"
}}
```

Return ONLY the JSON, no other text.
"""
        messages = [
            {"role": "system", "content": "You are a professional Q&A assistant. You must output valid JSON format."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.system.llm_client.chat_completion(messages, temperature=0.5, max_retries=3)
            result = self.system.llm_client.extract_json(response)
            return normalize_prediction(result.get("answer", response))
        except Exception:
            return CATEGORY5_GROUND_TRUTH  # 실패 시 안전한 기본값

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
        builder.dialogue_buffer = []    # 실제 사용 X
        builder.processed_count = 0
        self._dialogue_id_counter = 0
