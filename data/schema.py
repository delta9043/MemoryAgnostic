from dataclasses import dataclass, field
from typing import List, Optional


# 대화의 단일 발화 단위
@dataclass
class Turn:
    turn_id: str           # 발화 고유 ID
    speaker: str           # 화자 (예: "user", "assistant")
    content: str           # 발화 내용
    timestamp: Optional[str] = None    # 발화 시각
    session_id: Optional[str] = None   # 소속 세션 ID
    metadata: dict = field(default_factory=dict)


# 연속된 Turn들을 묶은 청크 단위 (텍스트 표현 포함)
@dataclass
class Chunk:
    chunk_id: int          # 청크 고유 ID
    turns: List[Turn]      # 청크에 포함된 발화 목록
    text: str              # 청크 전체를 하나의 문자열로 이어붙인 텍스트
    metadata: dict = field(default_factory=dict)


# 질문-답변 쌍
@dataclass
class QA:
    qa_id: str             # QA 고유 ID
    question: str          # 질문
    answer: str            # 정답
    category: Optional[str] = None     # QA 유형 또는 카테고리
    metadata: dict = field(default_factory=dict)


# 전처리 이전의 원본 대화 샘플
@dataclass
class RawSample:
    sample_id: str         # 샘플 고유 ID
    turns: List[Turn]      # 원본 발화 목록
    qa: List[QA]           # 해당 샘플에 연관된 QA 쌍 목록
    metadata: dict = field(default_factory=dict)


# 청킹 등 전처리가 완료된 샘플 (Turn 대신 Chunk 단위로 관리)
@dataclass
class ProcessedSample:
    sample_id: str         # 원본 RawSample과 동일한 ID
    chunks: List[Chunk]    # 전처리된 청크 목록
    qa: List[QA]           # 해당 샘플에 연관된 QA 쌍 목록
    metadata: dict = field(default_factory=dict)
