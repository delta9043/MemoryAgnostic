# gold_chunking

LLM(OpenAI)으로 대화 세션을 주제 단위로 분할해 **gold chunk**를 만든다.
구 `GoldChunking/`를 파일 수만 줄여 옮긴 것으로, 동작은 동일하다.

## 파일

| 파일 | 역할 |
|---|---|
| `prompts.py` | 시스템 프롬프트 · few-shot · JSON 스키마. **캐시 키에 포함되므로 수정 시 `PROMPT_VERSION`을 올릴 것** |
| `core.py` | 설정(.env) + 데이터 로더(locomo, longmemeval) + `BoundaryClient` |
| `run_gold.py` | 진입점 — 1차 세그먼트 패스, `--fix-leading-splits`로 2차 교정 패스 |
| `merge_gold_html.py` | LightMem Boundary Explorer HTML에 gold 레일/통계/필터를 얹어 `html/`에 저장 |

`cache/`(API 응답, 재실행 시 무료 재개) · `results/`(gold JSON) · `html/`(뷰어)는 git에서 제외된다.

## 준비

```bash
cp .env.example .env    # OPENAI_API_KEY 등 채우기
pip install openai python-dotenv
```

## 실행

```bash
# 1차 패스
python run_gold.py --dataset locomo --data <locomo10.json>
python run_gold.py --dataset longmemeval --data <longmemeval_s.json>
python run_gold.py --dataset locomo --data <...> --limit 1    # 스모크 테스트

# 2차 패스: 세션 첫 청크가 1턴인 over-split을 타깃 merge로 교정 (locomo 전용, 결과 파일 제자리 수정)
python run_gold.py --dataset locomo --data <locomo10.json> --fix-leading-splits

# HTML 뷰어에 gold 병합
python merge_gold_html.py
```

출력은 `results/{dataset}_gold.json`. 청크에 turn 전문이 들어 있어 단독 소비가 가능하다.

## 불변식

청크를 순서대로 이으면 원본 세션의 turn_id 시퀀스와 정확히 같아야 한다.
1차·2차 패스 모두 매 샘플에서 이를 assert로 검증한다.

## 구 GoldChunking 대비 변경

파일 배치만 바뀌었다 (8개 모듈 → 4개). 검증 결과:

- `prompts.py` sha256 동일 → **기존 `cache/` 272개 세션 전부 그대로 적중**
- 로더 출력(10 샘플 / 272 세션), `validate_segments`, `build_chunks`, `compute_stats`,
  2차 패스 대상 선정 모두 구버전과 일치
- `merge_gold_html.py` 산출 HTML이 기존 결과와 바이트 단위 동일

호출 방식 중 바뀐 것은 하나뿐: `python fix_leading_splits.py` → `python run_gold.py
--dataset locomo --data <locomo10.json> --fix-leading-splits`.
구버전이 하드코딩하던 locomo10.json 경로가 `--data`로 올라왔다.
