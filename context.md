# Context Handoff: MemoryAgnostic Pipeline Framework (Phase 0~2)

생성일: 2026-05-17
원본 대화 한 줄 요약: 메모리 증강 LLM 연구를 위한 전처리 파이프라인 프레임워크(Phase 0~2)를 설계하고 구현했다.

---

## 1. 목표와 현재 단계

- 최종 목표: Conversation 데이터에 Filter → Compressor → Chunker를 적용하는 모듈형 파이프라인을 구축하고, Oracle(Qwen 32B 기반 LLM 모듈) vs Baseline(FixedSize, LightMem, LLMLingua-2)의 성능을 비교한다.
- 현재까지 진행된 것:
  - Phase 0: 데이터 타입(Turn, Chunk, QA, RawSample, ProcessedSample) 확정, 파이프라인 순서 결정
  - Phase 1: LoCoMo10 DataLoader, BaseChunker + FixedSizeChunker, BaseFilter + NoFilter, BaseCompressor + NoCompressor, main.py end-to-end 동작 확인
  - Phase 2: config(yaml) 기반 모듈 선택, factory.py로 인스턴스 생성, main.py argparse 리팩터링, GitHub push 완료
- 지금 멈춰 있는 지점: Phase 2까지 완료. Phase 3(LightMemChunker 래핑) 시작 전.

---

## 2. 확정된 결정사항

- 파이프라인 순서: Filter → Compressor → Chunker (고정) — 이유: Filter로 불필요한 turn을 먼저 제거하고 남은 것을 압축하는 흐름이 Problem Definition과 일치.
- Filter/Compressor 순서는 config로 변경 가능, Chunker는 항상 마지막 — 이유: Chunker만 출력 타입이 List[Turn] → List[Chunk]로 바뀌므로 분리.
- 데이터 타입에 metadata dict 포함 — 이유: 데이터셋별 원본 필드, 압축 전 텍스트 등 부가 정보를 보존하기 위해.
- Compressor가 content를 수정할 경우 원본을 turn.metadata["original_content"]에 백업 — 이유: 압축 전후 토큰 수 비교 등 Cost 측정에 필요.
- LoCoMo10 category 숫자를 문자열로 매핑: {1: "single_hop", 2: "temporal", 3: "open_domain", 4: "multi_hop", 5: "adversarial"} — 이유: Evaluator에서 category별 분석 시 숫자보다 문자열이 명확.
- adversarial QA는 answer 키 없이 adversarial_answer 키를 사용 — 이유: LoCoMo10 데이터셋 원본 구조.
- conda 환경 이름: magno (Python 3.10) — 이유: MemoryAgnostic 줄임말, lightmem 환경과 분리.
- 클러스터: moana-y1, 프로젝트 경로: /data/delta9043/repos/MemoryAgnostic

---

## 3. 폐기된 대안

- Turn에 timestamps: List[str] 포함 — 폐기 이유: 어떤 timestamp가 어떤 turn에 속하는지 모호해짐. turn.timestamp로 접근하는 것으로 대체.
- Chunk에 session_ids, timestamps 별도 포함 — 폐기 이유: 중복. chunk.turns[i]를 통해 접근 가능.
- Turn.original_content를 top-level 필드로 — 폐기 이유: 항상 필요하지 않은 필드가 타입 정의를 지저분하게 만듦. metadata["original_content"]로 대체.
- vLLM 로컬 서빙 — 폐기 이유: ugrad QoS GPU 1장 제한으로 vLLM + LLMLingua-2 동시 구동 불가.
- 파이프라인 순서 완전 자유화(config로 Chunker 위치도 변경 가능) — 폐기 이유: Chunker 전후로 타입이 달라지므로 복잡도만 증가. Chunker는 고정으로 단순화.

---

## 4. 미해결 질문 / 다음 액션

- [ ] Phase 3: see_chunks.py를 LightMemChunker 클래스로 래핑 (core/chunker/lightmem.py)
- [ ] Phase 4: Evaluator 구현 (eval/cost.py, eval/qa.py). eval/redundancy.py는 지표 정의 후 구현
- [ ] Phase 5: LLMClient 구현 후 LLMChunker, LLMFilter, LLMCompressor 구현 (Oracle)
- [ ] Phase 6: LLMLinguaCompressor 래핑 (core/compressor/llmlingua.py)
- [ ] Phase 7: ablation 실험 실행 (config 파일 여러 개 + sbatch)
- [ ] Redundancy 지표 정의 미확정 (chunk 간 임베딩 유사도? 중복 사실 비율?)
- [ ] Qwen 32B를 GPU 1장에 띄울 수 있는지 확인 (양자화/offload/HF Inference API 검토 필요)
- [ ] longmemeval DataLoader 미구현 (Phase 1은 LoCoMo10만 구현)
- [ ] MemoryBackend 미구현 (SimpleMem, A-Mem wrapping)

---

## 5. 산출물

프로젝트 루트: `/data/delta9043/repos/MemoryAgnostic/`
GitHub: `https://github.com/delta9043/MemoryAgnostic` (master 브랜치)

### 디렉토리 구조

```
MemoryAgnostic/
├── main.py                         # argparse + yaml config 로드 후 파이프라인 실행
├── factory.py                      # config dict → 모듈 인스턴스 생성
├── configs/
│   └── baseline_fixed_locomo10.yaml
├── data/
│   ├── schema.py                   # Turn, Chunk, QA, RawSample, ProcessedSample dataclass
│   ├── schema.md                   # 타입 설명 및 필드 매핑 문서
│   └── loader.py                   # load_locomo10(), load_locomo10_all()
└── core/
    ├── chunker/
    │   ├── base.py                 # BaseChunker(ABC): chunk(List[Turn]) -> List[Chunk]
    │   └── fixed_size.py           # FixedSizeChunker(window, overlap)
    ├── filter/
    │   ├── base.py                 # BaseFilter(ABC): run(List[Turn]) -> List[Turn]
    │   └── no_filter.py            # NoFilter: passthrough
    └── compressor/
        ├── base.py                 # BaseCompressor(ABC): run(List[Turn]) -> List[Turn]
        └── no_compressor.py        # NoCompressor: passthrough
```

### 핵심 타입 (data/schema.py)

```python
Turn: turn_id, speaker, content, timestamp?, session_id?, metadata{}
Chunk: chunk_id, turns: List[Turn], text: str, metadata{}
QA: qa_id, question, answer, category?, metadata{}
RawSample: sample_id, turns: List[Turn], qa: List[QA], metadata{}
ProcessedSample: sample_id, chunks: List[Chunk], qa: List[QA], metadata{}
```

### 파이프라인 흐름 (main.py + factory.py)

```python
raw = load_locomo10(path, sample_idx)
turns = raw.turns
for module in pre_chunking_modules:   # NoFilter, NoCompressor 순서대로
    turns = module.run(turns)         # List[Turn] → List[Turn]
chunks = chunker.chunk(turns)         # List[Turn] → List[Chunk]
processed = ProcessedSample(sample_id, chunks, raw.qa, metadata)
```

### config 예시 (configs/baseline_fixed_locomo10.yaml)

```yaml
dataset:
  name: locomo10
  path: /data/delta9043/datasets/locomo/locomo10.json
  sample_idx: 0
pipeline:
  pre_chunking:
    - type: NoFilter
    - type: NoCompressor
  chunker:
    type: FixedSizeChunker
    window: 5
    overlap: 0
```

### 실행 확인된 결과

```
[loader] sample_id: conv-26 / num turns: 419 / num qa: 199
[pipeline] turns after pre-chunking: 419
[chunker] num chunks: 84
[processed] first qa: QA(qa_id='conv-26_0', question='When did Caroline go to the LGBTQ support group?', answer='7 May 2023', category='temporal', ...)
```

---

## 6. 참고자료

- LoCoMo10 경로: `/data/delta9043/datasets/locomo/locomo10.json` (10 samples, 각 sample당 400~500 turns, 약 200 QA)
- longmemeval 경로: `/data/delta9043/datasets/longmemeval/` (longmemeval_s_cleaned.json, longmemeval_m_cleaned.json, longmemeval_oracle.json, 각 500 samples)
- longmemeval turn 구조: `{"role": "user"/"assistant", "content": str}`, haystack_dates로 timestamp 제공
- LightMem 기존 코드: `/data/delta9043/repos/LightMem/` — see_chunks.py가 Phase 3의 LightMemChunker 원본
- conda 환경: `magno` (Python 3.10), 설치된 패키지: pyyaml

---

## 7. 새 대화에서 시작하는 법

이 문서를 첨부한 뒤 다음과 같이 요청하면 된다:
"이 문서를 읽고 Phase 3(LightMemChunker 구현, core/chunker/lightmem.py)부터 시작해줘."

코드 파일이 필요하면 GitHub에서 클론하거나 아래 파일을 직접 첨부한다:
- `data/schema.py`
- `core/chunker/base.py`
- `core/chunker/fixed_size.py`
- `factory.py`
- `main.py`
