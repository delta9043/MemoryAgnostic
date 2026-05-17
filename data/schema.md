# schema.py 설명

파이프라인 전체에서 사용하는 공통 데이터 타입을 정의한다.
모든 모듈(Filter, Compressor, Chunker, Memory)은 이 타입을 기준으로 입출력을 맞춘다.

---

## 타입 목록

### Turn
대화의 단일 발화 단위.

두 데이터셋의 원본 필드를 공통 형식으로 정규화한 것이다.

| 필드 | 타입 | LoCoMo10 원본 | longmemeval 원본 |
|------|------|---------------|-----------------|
| turn_id | str | dia_id (예: "D1:1") | "{session_id}_{turn_idx}" 로 생성 |
| speaker | str | speaker (예: "Caroline") | role (예: "user", "assistant") |
| content | str | text | content |
| timestamp | Optional[str] | session_N_date_time (예: "1:56 pm on 8 May, 2023") | haystack_dates[session_idx] (예: "2023/05/20 (Sat) 02:21") |
| session_id | Optional[str] | "session_1", "session_2", ... | haystack_session_ids[session_idx] |
| metadata | dict | 부가 정보 | 부가 정보 |

**주의사항**
- timestamp는 turn마다 별도로 존재하지 않는다. 세션 단위 날짜를 해당 세션의 모든 turn에 복사해서 넣는다.
- content는 Compressor를 거치면 압축된 텍스트로 교체된다. 원본을 보존할 경우 압축 전에 `metadata["original_content"]`에 백업한다.

---

### Chunk
Chunker의 출력 단위. 연속된 Turn들의 묶음이다.

| 필드 | 타입 | 설명 |
|------|------|------|
| chunk_id | int | 샘플 내 chunk 순서 (0부터 시작) |
| turns | List[Turn] | 원본 Turn 리스트 보존 |
| text | str | turns의 content를 이어붙인 텍스트. Memory 시스템 입력으로 사용 |
| metadata | dict | 부가 정보 (예: boundary_type, score 등) |

**text 형식 예시**
```
Caroline: Hey Mel! Good to see you!
Melanie: Hey Caroline! I'm swamped with the kids & work.
```

**주의사항**
- timestamp, session_id 등은 Chunk에 별도로 두지 않는다. `chunk.turns[i].timestamp` 로 접근한다.
- turns를 보존하는 이유: 나중에 Temporal 분석이나 Cost 측정(압축 전후 토큰 수 비교) 시 원본이 필요하기 때문이다.

---

### QA
평가용 질문-답변 쌍.

| 필드 | 타입 | LoCoMo10 원본 | longmemeval 원본 |
|------|------|---------------|-----------------|
| qa_id | str | "{sample_id}_{idx}" 로 생성 | question_id |
| question | str | question | question |
| answer | str | answer | answer |
| category | Optional[str] | category | question_type |
| metadata | dict | 부가 정보 | 부가 정보 |

**category 값 예시**
- LoCoMo10: "single_hop", "multi_hop", "temporal", "open_domain", "adversarial"
- longmemeval: 확인 필요

---

### RawSample
DataLoader의 출력 단위. 파이프라인의 입력이다.

| 필드 | 타입 | 설명 |
|------|------|------|
| sample_id | str | 샘플 고유 ID |
| turns | List[Turn] | 전체 대화를 시간 순서대로 flat하게 펼친 Turn 리스트 |
| qa | List[QA] | 이 샘플에 대한 QA 쌍 리스트 |
| metadata | dict | 데이터셋 이름, 원본 파일 경로 등 |

**turns가 flat한 이유**
LoCoMo10은 session_1, session_2... 로 분리되어 있고, longmemeval은 haystack_sessions 리스트로 되어 있다. DataLoader에서 전부 이어붙여 단일 리스트로 만든다. session 구분은 각 Turn의 session_id로 추적할 수 있다.

---

### ProcessedSample
파이프라인의 출력 단위.

| 필드 | 타입 | 설명 |
|------|------|------|
| sample_id | str | RawSample의 sample_id와 동일 |
| chunks | List[Chunk] | Chunker 출력. Memory 시스템의 입력으로 사용 |
| qa | List[QA] | RawSample의 qa와 동일. 평가 시 사용 |
| metadata | dict | 파이프라인 실행 정보 등 |

---

## 파이프라인 데이터 흐름

```
DataLoader
    └─ RawSample (turns: List[Turn])
          │
          ▼
    Filter.run(turns) → List[Turn]       # 불필요한 turn 제거
          │
          ▼
    Compressor.run(turns) → List[Turn]   # turn 내부 텍스트 압축
          │
          ▼
    Chunker.chunk(turns) → List[Chunk]   # turn 묶음으로 분할
          │
          ▼
    ProcessedSample (chunks: List[Chunk])
          │
          ▼
    MemoryBackend (chunks를 메모리에 저장 후 QA 수행)
```

**Filter와 Compressor 순서는 config로 변경 가능하다.**
Chunker는 항상 마지막에 실행된다.

---

## metadata 활용 예시

| 위치 | 키 | 값 예시 | 용도 |
|------|----|---------|------|
| Turn.metadata | original_content | "압축 전 원본 텍스트" | Compressor 적용 전 백업 |
| Turn.metadata | source | "locomo10" | 데이터셋 추적 |
| Chunk.metadata | boundary_type | "topic" | Chunker 종류 기록 |
| Chunk.metadata | score | 0.85 | topic boundary 신뢰도 |
| RawSample.metadata | dataset | "locomo10" | 데이터셋 이름 |
| RawSample.metadata | source_path | "/data/.../locomo10.json" | 원본 파일 경로 |
| ProcessedSample.metadata | pipeline | ["NoFilter", "NoCompressor", "FixedSizeChunker"] | 실행된 파이프라인 기록 |
| QA.metadata | question_date | "2023/05/20" | 질문 날짜 |
| QA.metadata | answer_session_ids | ["session_3"] | 정답이 있는 세션 |
