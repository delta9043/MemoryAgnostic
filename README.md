# MemoryAgnostic

LLM 메모리 시스템의 **입력 전처리(필터링·청킹)** 가 장기 대화 QA 성능에 미치는 영향을,
여러 메모리 백엔드에 대해 **동일한 파이프라인·동일한 채점 척도**로 측정하는 실험 프레임워크.

- **벤치마크**: LoCoMo10 (대화 10개, QA ~1,986개, 5개 카테고리 — single_hop / temporal / open_domain / multi_hop / adversarial)
- **메모리 백엔드**: SimpleMem · A-Mem · LightMem · mem0 (각 원본 repo를 wrapping)
- **전처리 축**: Filter(정보 필터링) / Chunker(청킹 전략)
- **채점**: F1 / Exact Match(원본 SimpleMem·A-Mem과 동일) + ROUGE / BLEU / METEOR / BERTScore

> 이 README는 프레임워크의 **구조와 실행법** 위주입니다. 세부 연구 동기·가설은 프로젝트 문서(CLAUDE.md 등)를 참고하세요.

---

## 파이프라인

```
LoCoMo10 원본 turns
      │
      ▼  pre_chunking (순서대로)
  ┌─ Filter        (turn별 정보 필터링; 개수 보존)
  └─ Compressor    (현재 NoCompressor = 통과)
      │
      ▼  Chunker    (turn들을 chunk로 묶는다)
      │
      ▼  MemoryBackend.build(chunks)   ← 각 메모리 시스템이 저장/색인
      │
      ▼  MemoryBackend.query(question) ← QA마다 검색 + 답변 생성
      │
      ▼  eval/metrics.py               ← 모든 백엔드를 동일 척도로 채점
      │
      ▼  results/{backend}/{variant}/result.json
```

핵심은 **백엔드마다 다른 native 채점(LLM-judge 등)을 쓰지 않고**, 우리 `eval/metrics.py`로
예측을 통일 채점한다는 점입니다. 그래야 SimpleMem·A-Mem·LightMem·mem0를 같은 자로 비교할 수 있습니다.

---

## 지원 모듈

**Filter** (`core/filter/`)
| type | 설명 |
|------|------|
| `NoFilter` | 통과 (baseline) |
| `LLMFilter` | turn별 informative하지 않은 내용 제거 (Qwen3-32B, transformers in-process) |

**Compressor** (`core/compressor/`)
| type | 설명 |
|------|------|
| `NoCompressor` | 통과 (전처리 슬롯 자리표시자) |

**Chunker** (`core/chunker/`)
| type | 설명 |
|------|------|
| `NoChunker` | turn 1개 = chunk 1개 (turn-level baseline) |
| `FixedSizeChunker` | 고정 크기 슬라이딩 윈도우 (`window`, `overlap`) |
| `AttentionSimilarityChunker` | LightMem 방식 topic-boundary (attention peak ∩ embedding cosine) |
| `LLMChunker` | 연속 stream + carryover 방식 LLM topic-boundary 청킹 (OpenAI 호환 API 모델) |
| `PrecomputedChunker` | 미리 만든 chunk JSON을 로드 (`sample_id` 매칭) |

**MemoryBackend** (`core/memory/`) — 모두 Qwen3-14B를 vLLM(OpenAI 호환)으로 서빙해 사용
| type | 원본 | 특징 |
|------|------|------|
| `SimpleMemBackend` | SimpleMem | 슬라이딩 윈도우 대신 chunk를 추출 단위로 주입, cat5 2지선다 |
| `AMemBackend` | A-Mem | chunk를 노트로 저장(RobustAdvancedMemAgent) |
| `LightMemBackend` | LightMem | sleep-time consolidation + native/NoOp segmenter |
| `Mem0Backend` | mem0(OSS) | 화자별 2뱅크, 세션 timestamp 보존, 로컬 Qdrant |

청킹은 **4조건과 분리해 1회만 precompute**한 뒤 `PrecomputedChunker`로 공유합니다(재현성·공정 비교).

---

## 셋업

**필수 전제**
- 각 백엔드 config의 `base_url`(기본 `http://localhost:8000/v1`)에 **Qwen3-14B를 서빙하는
  vLLM(OpenAI 호환) 서버**가 떠 있어야 합니다. `scripts/run_*.sh`(SLURM)는 vLLM 기동과 실행을
  함께 처리합니다. 직접 `python main.py`를 돌릴 땐 서버를 먼저 띄우세요.
- conda 환경(백엔드별): `simplemem`, `a-mem`, `lightmem`, `mem0`. 필터/청킹용 환경은 스크립트 참고.
- 원본 백엔드 repo가 sibling 경로에 있어야 합니다 (예: `/data/delta9043/repos/{SimpleMem,A-mem,LightMem,mem0}`).
- 모델: Qwen3-14B(백엔드), Qwen3-32B(LLMFilter), llmlingua-2·all-MiniLM-L6-v2(AttentionSimilarity/LightMem/mem0).
- 데이터: `locomo10.json`.

---

## 실험 실행

### 1) (선택) 전처리 산출물 precompute

전처리를 config 실행 중에 매번 다시 하지 않도록, 비싼 단계는 한 번만 만들어 둡니다.

```bash
# LLM 청킹 (연속 stream, OpenAI 호환 API 모델). 결과 JSON을 PrecomputedChunker가 로드.
python precompute/run_goldchunker.py --model <model> --output data/chunked_data/chunks_<model>.json

# LLM 필터링 (Qwen3-32B). filtered_data JSON 생성.
python precompute/run_filter.py ...
```

### 2) 조건별 실험 실행

각 조건은 자립 SLURM 잡(vLLM 기동 포함)입니다.

```bash
sbatch scripts/run_simplemem_default.sh       # baseline (NoChunker)
sbatch scripts/run_simplemem_llmchunk.sh      # LLM 청킹 (PrecomputedChunker)
sbatch scripts/run_simplemem_aschunker.sh     # AttentionSimilarity 청킹
sbatch scripts/run_amem_default.sh
sbatch scripts/run_lightmem_default.sh
sbatch scripts/run_mem0_default.sh
# ...
```

직접 실행(서버가 이미 떠 있을 때):
```bash
python main.py --config configs/simplemem_default.yaml
```

config 이름 규칙: **`{backend}_{variant}.yaml`** — variant = `default`(무변환) / `llmchunker` /
`aschunker` / `fixedsize` / `8bfilter` / `32bfilter`.

---

## 평가 & 결과

- `main.py`가 QA를 돌려 예측을 만들고 `eval/metrics.py`로 채점합니다.
- 전체 집계는 **질문 단위 micro-average**(모든 샘플 질문 풀링) — 원본 SimpleMem·A-Mem과 동일.
- adversarial(cat5) 정답은 `"Not mentioned in the conversation"`(오답 후보를 GT로 쓰지 않음).
- 결과 위치는 config의 `evaluation.results_dir`가 정합니다. 그 아래로
  `result_sample{N}.json`(샘플 1개) / `result.json`(10샘플 집계)이 떨어집니다.
  **1샘플 실행은 집계 파일명을 쓰지 않습니다.** *`results/`는 gitignore.*
- 조건 취합 비교: `python eval/compare_results.py`.

---

## 저장소 구조

```
main.py                 # 파이프라인 진입점 (config 1개 실행)
factory.py              # type 문자열 → 모듈 인스턴스 디스패처

precompute/             # main.py 이전에 도는 입력 생성기
  run_filter.py         # 필터 precompute        → data/filtered_data/
  run_goldchunker.py    # LLM 청킹 precompute    → data/chunked_data/

core/
  filter/               # NoFilter, LLMFilter
  compressor/           # NoCompressor
  chunker/              # No/FixedSize/AttentionSimilarity/LLM/Precomputed
  memory/               # SimpleMem/AMem/LightMem/Mem0 backend (+ base)
data/
  schema.py             # Turn / Chunk / QA / RawSample / ProcessedSample
  locomo_loader.py      # LoCoMo10 로더
  chunked_data/, filtered_data/   # precompute 산출물 (gitignore)
eval/
  metrics.py            # 통일 채점 (F1/EM/ROUGE/BLEU/METEOR/BERTScore)
  rescore.py            # LLM 재실행 없는 오프라인 재채점
  compare_results.py    # 조건 비교표
configs/                # _base.yaml(원본) + {backend}_{variant}.yaml
scripts/                # SLURM 실행 스크립트 (+ common/{start_vllm,run_experiment}.sh)
experiment/             # 메인 파이프라인과 분리된 연구용 코드 (별도 브랜치 위주)
```

---

## 참고 (구현 노트)

- **Qwen3 thinking off**: vLLM은 `enable_thinking`을 무시하므로 각 백엔드가
  `chat_template_kwargs`(또는 프롬프트 `/no_think`)로 끕니다. 서버 플래그로는 안 꺼집니다.
- **컨텍스트 길이**: `VLLM_MAX_MODEL_LEN=32768`로 통일(예전 8192에서 답변 유실 발생).
- **백엔드 재생성**: 샘플마다 백엔드를 새로 만들고, mem0/LightMem은 `clear_on_init`으로
  저장소를 비우고 시작합니다(직전 실행 잔여 오염 방지).
