import datetime
import os
import sys
from typing import List, Optional

from core.memory.base import BaseMemoryBackend, normalize_prediction
from data.schema import Chunk

# LightMem repo(src 레이아웃)를 import 가능하도록 sys.path에 추가.
# 서버 경로는 환경변수 LIGHTMEM_PATH로 오버라이드 가능.
LIGHTMEM_PATH = os.environ.get("LIGHTMEM_PATH", "/data/delta9043/repos/LightMem/src")
if LIGHTMEM_PATH not in sys.path:
    sys.path.insert(0, LIGHTMEM_PATH)

from lightmem.memory.lightmem import LightMemory  # noqa: E402
import lightmem.memory.lightmem as _lm_module  # noqa: E402  (전역 상태 reset용)
from openai import OpenAI  # noqa: E402


# LightMem LoCoMo 공식 추출 프롬프트(experiments/locomo/prompts.py의
# METADATA_GENERATE_PROMPT_locomo 전문을 그대로 복사 — few-shot 예시·이미지 처리 포함).
# experiments/는 패키지가 아니므로 import 대신 상수로 둔다.
LOCOMO_EXTRACTION_PROMPT = """
You are a Personal Information Extractor.
Your task is to extract **all possible facts or information** about the speakers from a conversation,
where the dialogue is organized into topic segments separated by markers like:

--- Topic X ---
[timestamp, weekday] <source_id>.<SpeakerName>: <message>
...

**Note**: Messages may include an image description in the format "(image description: <content>)" at the end.
This represents visual context captured when the message was sent. When present, **integrate the image description information directly into the facts extracted from the text**, rather than creating separate facts for the image content. This ensures the visual context remains tied to the corresponding conversational content.

Important Instructions:
0. You MUST process messages **strictly in ascending source_id order** (lowest → highest).
   For each message, stop and **carefully** evaluate its content before moving to the next.
   Do NOT reorder, batch-skip, or skip ahead — treat messages one-by-one.
1. You MUST process every user message in order, one by one.
   For each message, decide whether it contains any factual information.
   - If yes → extract it and rephrase into a standalone sentence.
   - **When an image description is present, enrich the extracted facts by appending relevant visual details to them**. Do NOT create separate facts solely for the image content.
   - Do NOT skip just because the information looks minor, trivial, or unimportant.
     Extract ALL meaningful information including:
     * Past events and current states
     * Future plans and intentions
     * Thoughts, opinions, and attitudes
     * Wants, hopes, desires, and preferences
2. **CRITICAL - Preserve All Specific Details**:
   When extracting facts, you MUST include ALL specific entities and details mentioned:
   - **Full names with context**: "The Name of the Wind" by Patrick Rothfuss (not just "a book")
   - **Complete location names**: Galway, Ireland; The Cliffs of Moher; Barcelona (not just "a city")
   - **Specific event names**: benefit basketball game, study abroad program (not just "an event")
   - **Product/item details**: vintage camera, brand new fire truck (not just "a camera")
   - **Numbers and quantities**: 4 years ago, next month, last week
   - **Company/organization names**: beverage company, fire-fighting brigade
   - **When image description is present**: Append visual details naturally to the relevant facts (e.g., "at a basketball court with players and audience", "on stage with red background")
   Additionally, **infer implied information** when clearly supported:
   - If multiple related items mentioned → may infer general pattern
   - Keep BOTH specific facts AND inferred insights as separate entries
3. Perform light contextual completion so that each fact is a clear standalone statement.
4. **Time Handling**:
   Note: Distinguish mention time (when said) vs event time (when happened).
   - For events with relative time (yesterday, last week, X ago, next month):
     Preserve the relative time and reference the message timestamp (YYYY-MM-DD).
     Format: "<fact with ALL details> <relative time> <timestamp>."
   - For ongoing/timeless facts: No time annotation needed.
5. Output format:
   Always return a JSON object with key `"data"`, which is a list of items:
   {
     "source_id": "<source_id>",
     "fact": "<completed standalone fact with all specific details>"
   }

Examples:
--- Topic 1 ---
[2024-01-07T17:24:00.000, Sun] 0.Tim: Hey John! Next month I'm off to Ireland for a semester in Galway
[2024-01-07T17:24:01.000, Sun] 1.John: That's awesome! Where will you stay?
[2024-01-07T17:24:02.000, Sun] 2.Tim: In Galway. I also want to visit The Cliffs of Moher
[2024-01-07T17:24:03.000, Sun] 3.John: Nice! By the way, I held a benefit basketball game last week (image description: basketball court with players and audience)
[2024-01-07T17:24:04.000, Sun] 4.Tim: Cool! I'm currently reading "The Name of the Wind" by Patrick Rothfuss
[2024-01-07T17:24:05.000, Sun] 5.John: That sounds interesting!
--- Topic 2 ---
[2024-01-12T13:41:00.000, Fri] 6.John: Got great news! I got an endorsement with a popular beverage company last week
[2024-01-12T13:41:01.000, Fri] 7.Tim: Congrats! That's amazing
[2024-01-12T13:41:02.000, Fri] 8.John: Thanks! By the way, Barcelona is a must-visit city
[2024-01-12T13:41:03.000, Fri] 9.Tim: I'll add it to my list!

{"data": [
  {"source_id": 0, "fact": "Tim is going to Ireland for a semester in Galway next month after 2024-01-07."},
  {"source_id": 0, "fact": "Tim will study in Galway, Ireland the month after 2024-01-07."},
  {"source_id": 2, "fact": "Tim will stay in Galway."},
  {"source_id": 2, "fact": "Tim wants to visit The Cliffs of Moher."},
  {"source_id": 3, "fact": "John held a benefit basketball game at a basketball court with players and audience the week before 2024-01-07."},
  {"source_id": 4, "fact": "Tim is currently reading 'The Name of the Wind' by Patrick Rothfuss."},
  {"source_id": 4, "fact": "Tim is reading a fantasy novel."},
  {"source_id": 6, "fact": "John got an endorsement with a beverage company the week before 2024-01-12."},
  {"source_id": 8, "fact": "John recommends Barcelona as a must-visit city."},
  {"source_id": 9, "fact": "Tim has a travel list and plans to add Barcelona to it."}
]}

Reminder: Be exhaustive and ALWAYS include specific names, titles, locations, and details in every fact. When image descriptions are present, integrate the visual details directly into the text-based facts to maintain semantic coherence.
"""


# flat retrieve() 결과(타임스탬프+메모리 문자열)를 컨텍스트로 쓰는 답변 프롬프트.
# LightMem에는 답변생성 API가 없어 retrieve + 이 프롬프트로 직접 생성한다.
LIGHTMEM_ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to the timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. If the memories contain contradictory information, prioritize the most recent memory.
5. For relative time references (e.g., "last year", "two months ago"), calculate the actual date from the memory timestamp and answer with the specific date/month/year.
6. Focus only on the content of the memories. Do not confuse character names mentioned in memories with the actual speakers.
7. The answer should be less than 5-6 words.

Memories:
{memories}

Question: {question}
Answer:"""


def _to_lm_timestamp(ts: Optional[str]) -> Optional[str]:
    """LoCoMo 타임스탬프("7:24 pm on 7 January, 2024")를 LightMem이 파싱 가능한
    ISO 형식("2024-01-07 19:24:00")으로 변환. 알 수 없는 포맷은 그대로 둔다
    (MessageNormalizer가 fromisoformat fallback으로 시도)."""
    if not ts:
        return ts
    s = ts.strip().strip("()")
    try:
        dt = datetime.datetime.strptime(s, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s


class LightMemBackend(BaseMemoryBackend):
    """
    LightMem(https://github.com/zjunlp/LightMem)을 wrapping하는 MemoryBackend.

    동작 방식:
    1. build(chunks): 각 turn을 user/assistant 쌍으로 add_memory에 주입한 뒤
       offline_update(sleep-time consolidation)까지 수행한다.
       - native_chunking=True: 모든 turn을 한 스트림으로 흘려보내고 마지막에만
         force → LightMem 자체 segmenter(llmlingua-2 attention+similarity)가 청킹.
       - native_chunking=False: chunk마다 force_segment → 우리 chunk 경계가 메모리
         단위가 됨(LightMem 청킹은 NoOpSegmenter로 무력화되어 있어야 함).
    2. query(question): lm.retrieve()로 메모리를 받아 답변 LLM을 직접 호출한다.
    3. reset(): Qdrant 컬렉션 + 모듈 전역 + 버퍼 상태를 비운다(인스턴스는 유지).

    manager LLM과 답변 LLM은 외부 vLLM(OpenAI 호환 API)에 요청한다. LightMem은
    추가로 임베딩 모델과(native일 때) LLMLingua-2를 in-process로 로드한다.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        embedding_path: str,
        native_chunking: bool,
        llmlingua_path: Optional[str] = None,   # native_chunking=True일 때만 필요
        embedding_dims: int = 384,
        qdrant_path: str = "./qdrant_data/lightmem",
        retrieve_k: int = 30,
        api_key: str = "dummy",
    ):
        if native_chunking and not llmlingua_path:
            raise ValueError("native_chunking=True이면 llmlingua_path가 필요합니다.")

        self.base_url = base_url
        self.model = model
        self.embedding_path = embedding_path
        self.native_chunking = native_chunking
        self.llmlingua_path = llmlingua_path
        self.embedding_dims = embedding_dims
        self.qdrant_path = qdrant_path
        self.retrieve_k = retrieve_k
        self.api_key = api_key

        self.lm = LightMemory.from_config(self._build_lm_config())
        # 답변 생성용 클라이언트(manager LLM과 동일 vLLM)
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _build_lm_config(self) -> dict:
        # native면 llmlingua-2 segmenter(자체 모델 로드), 아니면 NoOpSegmenter.
        if self.native_chunking:
            segmenter = {
                "model_name": "llmlingua-2",
                "configs": {
                    "model_name": self.llmlingua_path,
                    "device_map": "cuda",
                    "buffer_len": 512,
                },
            }
        else:
            segmenter = {"model_name": "noop"}

        return {
            "pre_compress": False,           # 우리 Filter/Compress 축과 겹침 방지
            "topic_segment": True,           # False면 저장 안 되는 stub → 항상 True
            "precomp_topic_shared": False,   # 공유할 compressor 없음
            "topic_segmenter": segmenter,
            "messages_use": "user_only",
            "metadata_generate": True,
            "text_summary": True,
            "extract_threshold": 0.1,
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": self.model,
                    "api_key": self.api_key,
                    "openai_base_url": self.base_url,
                    "max_tokens": 16000,
                },
            },
            "index_strategy": "embedding",
            "retrieve_strategy": "embedding",
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": self.embedding_path,
                    "embedding_dims": self.embedding_dims,
                    "model_kwargs": {"device": "cuda"},
                },
            },
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": "lightmem",
                    "embedding_model_dims": self.embedding_dims,
                    "path": self.qdrant_path,
                    "on_disk": True,
                },
            },
            "update": "offline",
            "extraction_mode": "flat",
        }

    def _add_turn(self, turn, force: bool) -> None:
        # 실제 발화 1개 → user(content) + assistant("") 쌍 (LightMem buffer 단위).
        ts = _to_lm_timestamp(turn.timestamp)
        messages = [
            {"role": "user", "content": turn.content,
             "speaker_name": turn.speaker, "time_stamp": ts},
            {"role": "assistant", "content": "",
             "speaker_name": turn.speaker, "time_stamp": ts},
        ]
        self.lm.add_memory(
            messages,
            METADATA_GENERATE_PROMPT=LOCOMO_EXTRACTION_PROMPT,
            force_segment=force,
            force_extract=force,
        )

    def build(self, chunks: List[Chunk]) -> None:
        if self.native_chunking:
            # chunk 경계 무시, 전체 turn 스트림 → LightMem segmenter가 청킹.
            turns = [t for c in chunks for t in c.turns]
            n = len(turns)
            for i, t in enumerate(turns):
                self._add_turn(t, force=(i == n - 1))
        else:
            # chunk 경계 = 메모리 단위. chunk 끝에서 force_segment로 강제 컷.
            for c in chunks:
                m = len(c.turns)
                for j, t in enumerate(c.turns):
                    self._add_turn(t, force=(j == m - 1))

        # sleep-time consolidation (LightMem 핵심 단계 — 생략 불가)
        self.lm.construct_update_queue_all_entries()
        self.lm.offline_update_all_entries(score_threshold=0.9)

    def query(self, question: str, category: Optional[str] = None,
              answer: Optional[str] = None) -> str:
        """retrieve + 답변 LLM 직접 호출. answer는 인터페이스 통일용(미사용)."""
        memories = self.lm.retrieve(question, limit=self.retrieve_k)
        if not memories:
            memories = "No relevant memories found."
        prompt = LIGHTMEM_ANSWER_PROMPT.format(memories=memories, question=question)
        # Qwen3 thinking off: chat template 인자로 꺼야 실제로 꺼진다(서버 플래그와
        # 최상위 enable_thinking은 vLLM이 무시). /no_think는 이중 방어로 유지.
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt + " /no_think"}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return normalize_prediction(resp.choices[0].message.content)

    def reset(self) -> None:
        """다음 샘플 전 초기화: Qdrant 컬렉션 + 모듈 전역 + 버퍼 상태."""
        self.lm.embedding_retriever.reset()      # delete_col + create_col
        _lm_module.GLOBAL_TOPIC_IDX = 0
        _lm_module.GLOBAL_LAST_SUMMARY_TIME = None

        bm = self.lm.senmem_buffer_manager
        bm.buffer.clear()
        bm.big_buffer.clear()
        bm.token_count = 0

        sm = self.lm.shortmem_buffer_manager
        sm.buffer.clear()
        sm.token_count = 0
