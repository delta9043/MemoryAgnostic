export const meta = {
  name: 'memagnostic-code-review',
  description: 'MemoryAgnostic 전체 정확성 우선 코드 리뷰 + 백엔드 native 충실도 검증',
  phases: [
    { title: 'Review', detail: '컴포넌트별 정확성/간결성/주석/충실도 리뷰' },
    { title: 'Verify', detail: '정확성 지적을 회의적 검증자로 adversarial 검증' },
  ],
}

const MA = 'C:/Program Code/memory_agnostic_project/MemoryAgnostic'
const ROOT = 'C:/Program Code/memory_agnostic_project'

const SHARED = `
[저장소]
- 메인: ${MA}
- 형제 백엔드(원본 fork, 원본코드 수정 최소): ${ROOT}/A-mem, ${ROOT}/SimpleMem, ${ROOT}/LightMem, ${ROOT}/mem0
경로는 Bash/Grep/Glob에 forward-slash로 쓰면 된다.

[우선순위] (1) 정확성  (2) 간결성=읽기 쉬움("짧게"가 아니라 "쉽게 읽힘").

[주석 규칙 — 사용자 기준]
- 짧은 signpost. 코드가 자명하면 주석 없음. 비자명(트릭·파라미터 의미·왜 이 방식)만 한 줄로.
- 코드 중간 에세이 금지. 형식/언어 혼용, 과한 주석, 사실과 다른(틀린) 주석을 잡아라.

[보고 원칙]
- 반드시 코드를 실제로 읽고 확인. 추측 금지. 없으면 findings 비워도 된다(허위 지적이 더 나쁨).
- '정확성' 지적은 반드시 구체 실패 시나리오(입력/상태 → 잘못된 출력/크래시)를 detail에 적어라.
- 실제 수정은 하지 마라(리뷰만). suggested_action에 무엇을 어떻게 고칠지만 적어라.
severity: correctness(동작 결함) | simplicity(간결·가독) | comment(주석) | unused(미사용/dead) | overlap(중복 컴포넌트) | question(설계 의문) | other

[이미 확인된 사실 — 재확인에 시간 쓰지 말고 검증에 집중]
- 현재 브랜치 master. experiment/의 추적 소스 0개(소스는 feat/attn-boundary 브랜치, 아직 main 미merge). master엔 gitignore된 산출물만.
- backend는 Chunk.turns만 사용(Chunk.text 미참조). text는 precomputed 왕복+chunk_evaluator에서만 쓰임.
- load_locomo10_all은 run_chunker/run_filter/run_goldchunker가 사용(=dead 아님).
- 불필요 후보(확인됨): configs/config.yaml(참조0), configs/test_amem.yaml(참조0), configs/test_simplemem.yaml+scripts/run_all.sh(옛 포트공식·TP=2 잔재 한 쌍). __pycache__는 미추적. results/에 추적 JSON 26개.
- NoChunker ≡ FixedSizeChunker(window=1,overlap=0)(기능 동일).
`

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    component: { type: 'string' },
    summary: { type: 'string', description: '이 컴포넌트 전반 평가 1-2문장' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          severity: { type: 'string', enum: ['correctness','simplicity','comment','unused','overlap','question','other'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          title: { type: 'string' },
          detail: { type: 'string' },
          suggested_action: { type: 'string' },
          confidence: { type: 'string', enum: ['high','medium','low'] },
        },
        required: ['severity','file','line','title','detail','suggested_action','confidence'],
      },
    },
  },
  required: ['component','summary','findings'],
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['confirmed','refuted','uncertain'] },
    reasoning: { type: 'string' },
  },
  required: ['verdict','reasoning'],
}

const UNITS = [
  { key: 'chunker-basic', prompt: `core/chunker/{base,no_chunker,fixed_size,precomputed_chunker}.py 검토.
명세 대조: no_chunker=turn 1개→chunk 1개; fixed_size=W/overlap 슬라이딩 윈도우(step=W-overlap, 경계/마지막 처리·overlap 검증 정확?); precomputed=JSON 로드+sample_id 매칭+turn수 무결성 검사. Chunk.text 생성 형식이 청커마다 일관적인지(단일 turn vs join). 정확성 버그·dead code·주석 품질.` },

  { key: 'attn-similarity', prompt: `core/chunker/attention_similarity.py가 LightMem 원본 청킹을 정확히 재현하는지 DEEP 검증.
LightMem 원본 소스: ${ROOT}/LightMem/src/lightmem — grep으로 LlmLingua2Segmenter.propose_cut(attention peak/boundary)와 SenMemBufferManager.cut_with_segmenter(embedding cosine)를 찾아 우리 구현과 라인 단위로 대조하라.
특히: (a) attention 집계 방향 — 우리 L172 'sub.sum(axis=1).mean()'가 LightMem의 A-side/B-side 집계와 같은 방향인가? (b) boundary 위치가 'peak turn 앞'인가 — LightMem 최근 커밋 e5fb813 "place attention boundary before the peak turn per paper (App. C.1)"; 우리는 local maxima index를 그대로 boundary로 쓴다(off-by-one 가능성). (c) 기본 attention_layers [8,9,10,11]가 LightMem 기본과 일치? (d) threshold 0.2→0.5 step 0.05 상승 로직. (e) B1∩B2 결합(거리 이내 채택, 매칭 없으면 B2, 둘 다 없으면 전체 1청크). 불일치는 correctness로, 실패 시나리오 명시.` },

  { key: 'llm-gold-chunker', prompt: `core/chunker/llm_chunker.py 와 gold_chunker.py(+ run_goldchunker.py) 검토.
llm_chunker 정확성: _group_by_session 연속 세션 런 분리 정확?; 출력 불변식(모든 chunk.turns 이으면 입력과 정확히 동일-누락/중복/재정렬 없음, chunk가 세션경계 안 넘음); _parse_boundaries(유효범위 1..n-1, bool 배제, None(실패) vs [](경계없음) 구분, cut_points 계산); thinking off(apply_chat_template enable_thinking).
gold_chunker 정확성: 윈도우 버퍼링/carryover(마지막 2세그 재판단)/C-cap 강등/캐시 리플레이/실패 fallback이 논리적으로 맞는지, 출력 불변식 동일. dead code·주석.
★ 추가 분석(overlap severity): llm_chunker와 gold_chunker의 코드 중복을 구체적으로 대조하라 — SYSTEM_PROMPT/USER_PROMPT_TEMPLATE, _parse_boundaries, 경계→Chunk 조립, 실패처리(None vs [])가 거의 동일한지. 둘은 실험 역할(세션-aware 실용 vs 세션-agnostic 상한)은 다르나 코드가 겹친다. 공통 base(_LLMBoundaryChunkerBase: 프롬프트/파싱/조립 공유)로 두 전략 유지하며 중복만 제거하는 안이 타당한지, 위험(동작 변화 없이 가능한지)을 suggested_action에.` },

  { key: 'filter-compressor', prompt: `core/filter/{base,no_filter,llm_filter}.py, core/compressor/{base,no_compressor}.py 검토.
llm_filter 정확성: 입력 turn 수 == 출력 turn 수(content만 수정, 빈 문자열 허용); _extract_json robust 파싱(<think> 제거, 코드펜스, 첫{~마지막}); 실패 시 원본 content 유지 + failure_count/failed_turn_ids 누적; thinking off는 transformers apply_chat_template(enable_thinking=False)로 처리됨(vLLM 아님) 확인. no_compressor/no_filter는 항등(정상). BaseCompressor docstring이 original_content 백업을 언급하나 실제 구현은 없음(주석-코드 불일치?). 주석·dead code.` },

  { key: 'backend-simplemem', prompt: `core/memory/simplemem_backend.py가 native SimpleMem 파이프라인과 정확히 일치하는지 검증(입력형식+파이프라인+출력 일치).
native 원본: ${ROOT}/SimpleMem — main.py의 SimpleMemSystem(add_dialogue_group/add_dialogues/ask), models/memory_entry.py의 Dialogue(__str__ 양식), core/memory_builder.py의 process_group vs process_window.
검증: (1) 입력형식 — Dialogue 1개=turn 1개(speaker=실제화자, timestamp), 우리가 문자열 조립 안 함(native 양식 유지). (2) 파이프라인 — add_dialogue_group→process_group가 process_window와 추출/저장/previous_entries(H) 갱신이 동일한지 원본 대조; FTS rebuild_fts_index 필요성/정확성. (3) 출력 — ask()→normalize_prediction. (4) reset 누수(previous_entries/dialogue_buffer/processed_count/카운터). sys.modules로 core.* 격리하는 부분의 위험. 커스텀 추가분(process_group, rebuild_fts_index)만 원본 대비 검토. correctness/comment.` },

  { key: 'backend-amem', prompt: `core/memory/amem_backend.py가 native A-Mem과 일치하는지 검증.
native 원본: ${ROOT}/A-mem — test_advanced_robust.py의 RobustAdvancedMemAgent(add_memory, answer_question 시그니처/기대 입력형식), memory_layer_robust.py.
검증: (1) 입력형식 — 우리는 chunk turns를 "Speaker X says: ..."로 합쳐 add_memory한다. native A-Mem이 실제로 이 형식/이 방식(노트당 텍스트)을 쓰는가? 우리가 임의 형식을 만든 것은 아닌가? timestamp=chunk 첫 turn. (2) category 1~5 매핑, cat5에 answer 전달, 그 외 answer 무시. (3) query에 "/no_think" 추가 + native의 <think> strip. (4) retrieve_k=10(native 기본과 일치?). (5) reset=agent 재생성. 입력형식이 native와 다르면 correctness로. comment.` },

  { key: 'backend-lightmem', prompt: `core/memory/lightmem_backend.py가 native LightMem과 일치하는지 검증.
native 원본: ${ROOT}/LightMem/src/lightmem — LightMemory.from_config/add_memory/retrieve, offline consolidation(construct_update_queue_all_entries, offline_update_all_entries), experiments/locomo의 add 흐름(참고).
검증: (1) turn→{user:content}+{assistant:""} pair 합성이 native(add_locomo)와 동일한가. (2) native_chunking True(llmlingua-2 segmenter)/False(NoOpSegmenter+우리 chunk 경계) 분기 정확. (3) sleep-time offline consolidation 호출(생략 시 LightMem 핵심 미검증). (4) retrieve(문자열 반환)+LIGHTMEM_ANSWER_PROMPT 답변, thinking off(chat_template_kwargs + /no_think). (5) reset(GLOBAL_TOPIC_IDX/GLOBAL_LAST_SUMMARY_TIME/버퍼/Qdrant). (6) _to_lm_timestamp ISO 변환. correctness/comment.` },

  { key: 'backend-mem0', prompt: `core/memory/mem0_backend.py가 native mem0 LoCoMo 평가와 일치하는지 엄밀 검증(방금 구현됨).
native 참조: ${ROOT}/mem0/evaluation/src/memzero/{add,search}.py + ${ROOT}/mem0/mem0/memory/main.py(Memory.add/search/reset, VllmLLM.generate_response, base._is_reasoning_model).
검증: (1) 화자 2-pass — 뱅크 주인=user/상대=assistant, user_id 2개, first-seen 순. (2) chunk→add() 뱅크당 1회(=chunk×2), content "speaker: content"(native 형식), metadata timestamp. (3) search 두 뱅크 top_k=10 → 공식 non-graph ANSWER_PROMPT(speaker_1/2 블록), "{ts}: {mem}" 형식. (4) thinking off — generate_response 래핑(setdefault extra_body chat_template_kwargs)이 fact추출·ADD/UPDATE·답변 3경로 전부에 도달하는가; Qwen3-14B가 _is_reasoning_model에 안 걸려 extra_body 보존되는가. (5) reset=mem.reset(). (6) search 결과 timestamp 경로 item["metadata"]["timestamp"] 정확? correctness/comment.` },

  { key: 'main-factory-runners', prompt: `main.py, factory.py, run_chunker.py, run_filter.py, run_goldchunker.py 검토.
정확성 중점: (a) 파이프라인 순서(pre_chunking filter/compress → chunk → build → QA → eval → save → reset). (b) ★ run()이 샘플마다 build_memory_backend로 backend를 새로 생성하는데, 각 backend.reset()은 "인스턴스 재사용" 전제로 설계됨 → reset()이 현재 흐름에서 dead인지, 그리고 lightmem/mem0 무거운 모델(임베더/LLMLingua)이 샘플마다 재로드되어 낭비인지 판단하고 실패/비효율 시나리오 기술. (c) 전체집계(main.py main(): 샘플별 category metric을 단순 평균 — 질문 수 가중 아님)의 방법론 타당성. (d) load_filtered_json 경로/qa 소스. factory: 모든 type 분기 존재 및 kwargs 전달. dead code·주석.` },

  { key: 'eval', prompt: `eval/metrics.py, compare_results.py 검토.
정확성: calculate_f1(token-level, set 교집합-중복토큰 미반영은 SQuAD 표준과 다름? 확인), bleu/rouge/meteor/bertscore 계산, evaluate_results 집계(카테고리별+overall, ×100), NLTK 다운로드 가드. compare_results: 파일 경로(results/{backend}/{variant})/포맷/delta. dead code(안 쓰는 metric 함수?)·주석.
추가로 '왜 각 backend 자체 eval(LLM-judge 등) 대신 이 통일 F1/BLEU eval이 필요한가'를 코드 근거로 1-2문장 정리해 summary에 포함(사용자 설명용).` },

  { key: 'data-loader', prompt: `data/locomo_loader.py, data/schema.py 검토.
정확성: 세션 키 정렬(session_N 번호순), timestamp/session_id 부여, adversarial answer 정규화(answer 없으면 adversarial_answer, list→첫 원소), sample_id metadata(PrecomputedChunker/LLMChunker가 turns[0].metadata에 의존). dead code: load_locomo10_all이 repo 어디서든 쓰이는지 grep(안 쓰이면 unused). load_filtered_json의 Turn(**t)(필드 불일치 위험)·RawSample metadata 누락. 주석.` },

  { key: 'scripts', prompt: `scripts/ 모든 *.sh + scripts/common/{start_vllm.sh,run_experiment.sh} 검토(정확성 중점, logs는 스킵).
검증: (a) thinking off — 각 실험 스크립트/공통 스크립트가 무엇을 설정하나. vLLM 자체로는 안 꺼지고 backend가 chat_template_kwargs로 끈다. 스크립트의 --override-generation-config 등 오해소지 설정이 남아있나. (b) VLLM_MAX_MODEL_LEN=32768 전 스크립트 일관. (c) start_vllm.sh의 tp_size/gpu 분배/포트(VLLM_PORT 충돌)/served-model-name. (d) --exclude 노드 목록 유효. (e) mem0 MEM0_TELEMETRY, conda env 이름 정확. 하드웨어 전제(GPU 수, VRAM)에 주석이 있으면 좋은 곳 지적. 스크립트 간 불일치·불필요/중복 스크립트(run_all.sh 등) 목록.` },

  { key: 'configs', prompt: `configs/ 모든 *.yaml 검토.
각 yaml의 memory_backend/chunker/pre_chunking 필드가 해당 클래스 생성자 인자와 정확히 맞는지 factory.py _build_module 및 각 클래스 __init__와 대조(오타/누락/여분 키는 TypeError 유발). db_path/qdrant_path가 backend/variant 간 충돌 없는지. 불필요·중복 config 후보(test_amem.yaml/test_simplemem.yaml/config.yaml 사용처, 옛 aschunker/8bfilter/32bfilter가 현 실험 계획과 맞나) 목록. 값 오류(경로, embedding_dims, chunks_path 존재 가정).` },

  { key: 'readme', prompt: `README.md를 현재 파이프라인과 대조.
현 구조: pre_chunking(filter/compress) → chunker 6종(no/fixed_size/attention_similarity/llm/gold/precomputed) → memory backend 4종(SimpleMem/A-Mem/LightMem/mem0) → 통일 eval(metrics.py). README가 옛 구조·옛 명령어·존재하지 않는 파일·틀린 실행법을 참조하는 부분을 전부 찾아, 각 항목을 '현재 무엇으로 고쳐야 하는지' 구체적으로 제시(불일치 목록). 실제 수정은 하지 말 것. 모든 finding severity=other 또는 comment로, file=README.md.` },

  { key: 'crosscut', prompt: `MemoryAgnostic 전체 횡단 스캔(감지·목록만, 삭제/수정 금지).
(1) 주석 형식 통일성: 파일별 docstring 스타일(""" vs #), 한/영 혼용, 과한 주석·틀린 주석 패턴을 파일 목록으로. (2) 불필요 파일 후보: test/ 내용, configs/config.yaml, configs/test_*.yaml, 추적되는 __pycache__, results/ 내용(사용자가 지우겠다 함), experiment/의 .pyc-only 잔재, .env.example 등 — 근거와 함께 목록(unused severity). (3) 겹치는 컴포넌트: 청커 6종의 역할 경계가 명확한가(fixed_size vs no_chunker(=window1), precomputed vs llm/gold 산출물 소비 관계, attention_similarity vs llm/gold), filter/compressor 축 등 중복·혼란 소지(overlap severity)를 '설명'과 함께. Bash로 grep/find 활용.` },

  { key: 'experiment-chunkeval', prompt: `chunk_evaluator/(evaluate.py, report.md, chunk/gpt-5.6-sol.json) 와 experiment/ 검토.
사실: 현재 브랜치 master에는 experiment/의 추적 소스가 0개다(소스는 feat/attn-boundary 브랜치; master엔 gitignore된 산출물 pyc/npz/png/cache/html/results만 남음). git ls-files experiment/ 로 확인하라.
(a) chunk_evaluator를 experiment/ 하위로 옮기는 것이 타당한가 + 옮길 때 문제(evaluate.py는 HERE=Path(__file__).parent 기반이라 경로 자립적인지, .gitignore 영향, import 없음 확인). 이동 절차를 suggested_action에. (b) experiment/의 현재 상태(master 기준)만 사실 보고 — 무엇이 gitignore 산출물이고 소스가 어느 브랜치인지. evaluate.py 자체의 정확성/주석도 검토.` },
]

function verifyPrompt(u, f) {
  return `${SHARED}
당신은 회의적 검증자다. 아래 '정확성(correctness)' 지적이 실제 결함인지 코드를 직접 읽어 독립 검증하라. 기본 입장은 회의적 — 반박을 시도하고, 코드로 확증 못 하면 uncertain.

[컴포넌트] ${u.key}
[파일] ${f.file} (line ~${f.line})
[주장] ${f.title}
[상세] ${f.detail}
[실패 시나리오가 실제로 성립하는가?] 해당 파일(필요시 native 원본)을 읽고 판단하라.

confirmed(실제 결함) / refuted(오탐) / uncertain(불확정) + 근거(reasoning).`
}

log(`코드 리뷰 시작: ${UNITS.length}개 컴포넌트 병렬 리뷰 → 정확성 지적 adversarial 검증`)

const reviewed = await pipeline(
  UNITS,
  u => agent(`${SHARED}\n\n[검토 대상: ${u.key}]\n${u.prompt}`,
             { label: `review:${u.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }),
  (review, u) => {
    if (!review) return null
    const corr = (review.findings || []).filter(f => f.severity === 'correctness')
    if (!corr.length) return { ...review, key: u.key }
    return parallel(corr.map(f => () =>
      agent(verifyPrompt(u, f), { label: `verify:${u.key}`, phase: 'Verify', schema: VERDICT_SCHEMA })
        .then(v => ({ f, v }))
    )).then(pairs => {
      const vmap = new Map(pairs.filter(Boolean).map(x => [x.f, x.v]))
      const findings = review.findings.map(f =>
        f.severity === 'correctness' ? { ...f, verdict: vmap.get(f) || null } : f)
      return { ...review, findings, key: u.key }
    })
  }
)

return reviewed.filter(Boolean)
