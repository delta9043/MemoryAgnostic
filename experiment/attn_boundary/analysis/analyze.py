"""
analyze.py — Phase B 재설계: 어텐션 원재료 → 경계 분석 figure + summary.

extract_attn.py가 만든 stats_*.npz를 읽어, gold chunking 경계와 어텐션 지표의
관계를 figure/summary.md로 정리한다. 레이블은 2개뿐:
  경계   = gold JSON의 모든 chunk 시작 turn (세션/세션내부 구분 없음, 빨간 실선)
  비경계 = 나머지 turn

실행 예시 (MemoryAgnostic 루트에서):
  python experiment/attn_boundary/analysis/analyze.py \
      --gold data/chunked_data/chunks_fable-5.json

  # conv 하나만 맵/막대 + 층 축소 + 태그
  python experiment/attn_boundary/analysis/analyze.py \
      --gold data/chunked_data/chunks_fable-5.json \
      --layers 8,avg --map_layers 8 --convs conv-26 --tag smoke

옵션:
  --gold PATH        (필수) 경계 JSON (chunk JSON: sample_id + chunks)
  --data DIR         stats_*.npz 위치 (기본: ../data)
  --layers SPEC      hist·boundary-avg·attnbar·budget-split용 층.
                     예 "8,9,10,11,avg" — 층마다 그림 생성, avg=나열된 숫자 층 평균
  --map_layers L,..  attnmap용 층. 주어진 층들의 평균 1버전만 생성 (기본 8,9,10,11)
  --convs A,B        attnmap·attnbar 그릴 conv 제한 (기본: 전체)
  --tile N           attnmap-diag 타일 크기(턴, 겹침 10) (기본 60)
  --k N              boundary-avg 반경 ±N턴 (기본 5)
  --tag NAME         출력 디렉토리(타임스탬프) 뒤 태그
  --output DIR       산출물 루트 (기본: analysis/output)

산출물 (output/타임스탬프[_tag]/):
  attnmap-diag_{층}_{conv}_tAAA-BBB.png  turn×turn 정사각 맵 타일 (자기자신 마스킹, 로그)
  attnmap-band_{층}_{conv}.png           띠 펴기: x=turn, y=몇 턴 앞(1~8), 색=mass
  attnmap-band-corr_{층}_{conv}.png      거리 보정판 (같은 거리 평균 대비, 빨강=강함/파랑=약함)
  attnbar_{층}_{conv}.png                turn별 mass_prev 막대 (여러 줄 분할)
  hist_{지표}_{층}.png                   경계 vs 비경계 값 분포
  boundary-avg_{지표}_{층}.png           경계 정렬 ±k턴 평균 곡선 (오차밴드+기준선)
  budget-split.png                       경계/비경계 평균 예산 구성 (층별 subfigure)
  layer-gap.png                          층별 경계/비경계 평균 (층 선택 근거)
  summary.md                             ①실험 개요 ②figure 목록 ③핵심 결과 ④세부 통계 ⑤해석
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MAX_DIST = 10         # attnmap-band의 y축 범위 (몇 턴 앞까지)
TILE_OVERLAP = 10     # attnmap-diag 타일 겹침
BAND_PER_ROW = 140    # attnmap-band 한 행의 turn 수
BAR_PER_ROW = 110     # attnbar 한 행의 turn 수
PRIOR_LAYERS = [8, 9, 10, 11]  # 선행연구(LightMem segmenter)와 동일한 층 평균

# 기본 지표 3종 (hist / boundary-avg / layer-gap 공유)
METRIC_DESC = {
    "mass_prev": "직전 turn(t-1)에 쓴 예산 비율",
    "mass_self": "자기 turn 내부에 쓴 예산 비율",
    "mass_cls": "CLS(싱크)로 흘린 예산 비율",
}
# extract_attn 재실행 후 추가되는 CLS 확장 필드 (npz에 있으면 자동 포함)
OPTIONAL_METRICS = {
    "cls_first": "turn 첫 토큰이 CLS로 흘린 양",
    "cls_max": "CLS로 가장 많이 흘린 토큰의 양",
}
# budget-split 구성 — 다섯 몫의 합 = 예산 전체(1.0)
BUDGET_COMPS = ["mass_prev", "mass_far", "mass_self", "mass_cls", "mass_sep"]


# ══════════════════════════════════════════════════════════════════════
# 로드 · 파생
# ══════════════════════════════════════════════════════════════════════

def load_stats(stats_dir):
    """stats_*.npz 전부 로드 → {sample_id: dict}"""
    out = {}
    for f in sorted(Path(stats_dir).glob("stats_*.npz")):
        d = np.load(f, allow_pickle=True)
        out[f.stem[len("stats_"):]] = {k: d[k] for k in d.files}
    return out


def load_gold(path):
    """chunk JSON → {sample_id: 경계 turn 집합} (각 chunk 시작 turn, 첫 chunk 제외)"""
    out = {}
    for rec in json.load(open(path, encoding="utf-8")):
        acc, starts = 0, set()
        for c in rec["chunks"][:-1]:
            acc += len(c["turns"])
            starts.add(acc)
        out[rec["sample_id"]] = starts
    return out


def derive(d, metric_keys):
    """원재료 1개 conv → {지표: [N, L]} (NaN = 미채점)"""
    owner, prev = d["pair_owner"], d["pair_prev"]
    psum, n_tokens = d["pair_sum"], d["n_tokens"]
    N = len(d["scored"])
    L = psum.shape[1] if len(psum) else d["cls_mass"].shape[1]

    Q = {q: np.full((N, L), np.nan, np.float32)
         for q in ["mass_prev", "mass_self", "mass_far", "ratio_prev"]}
    Q["mass_cls"] = d["cls_mass"].astype(np.float32).copy()
    Q["mass_sep"] = d["sep_mass"].astype(np.float32).copy()
    for opt in OPTIONAL_METRICS:
        if opt in metric_keys:
            Q[opt] = d[opt].astype(np.float32).copy()
    for q in ["mass_cls", "mass_sep"] + [o for o in OPTIONAL_METRICS if o in Q]:
        Q[q][~d["scored"]] = np.nan

    for t in np.where(d["scored"])[0]:
        idx = np.where(owner == t)[0]
        if not len(idx):
            continue
        u = prev[idx]
        mass = psum[idx] / float(n_tokens[t])  # 토큰당 예산 비율
        ctx = u != t
        self_r = np.where(u == t)[0]
        if len(self_r):
            Q["mass_self"][t] = mass[self_r[0]]
        pr = np.where(u == t - 1)[0]
        if len(pr):
            m_prev = mass[pr[0]]
            Q["mass_prev"][t] = m_prev
            ctx_total = mass[ctx].sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                Q["ratio_prev"][t] = np.where(ctx_total > 0, m_prev / ctx_total, np.nan)
            Q["mass_far"][t] = ctx_total - m_prev
    return Q


def build_mass_matrix(d, idxs):
    """turn×turn mass 행렬 (지정 층 평균). M[t,u] = t가 u에 쓴 토큰당 예산."""
    N = len(d["scored"])
    M = np.full((N, N), np.nan, np.float32)
    psum = d["pair_sum"][:, idxs].mean(axis=1)
    nt = d["n_tokens"].astype(np.float32)
    for p in range(len(d["pair_owner"])):
        t, u = int(d["pair_owner"][p]), int(d["pair_prev"][p])
        M[t, u] = psum[p] / nt[t]
    return M


def reduce_layers(arr, idxs):
    """[N, L] → [N] (지정 층 평균)"""
    return arr[:, idxs].mean(axis=1)


# ══════════════════════════════════════════════════════════════════════
# 층 스펙 · 통계 유틸
# ══════════════════════════════════════════════════════════════════════

def parse_layer_specs(spec, L):
    """'8,9,avg' → [("L8",[8]), ("L9",[9]), ("Lavg8-9",[8,9])]"""
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    nums = [int(t) for t in tokens if t != "avg"]
    bad = [n for n in nums if not 0 <= n < L]
    if bad:
        sys.exit(f"[analyze] 층 범위 벗어남 {bad} (0~{L - 1})")
    out = []
    for t in tokens:
        if t == "avg":
            if not nums:
                sys.exit("[analyze] avg는 숫자 층과 함께 줘야 함 (예: 8,9,avg)")
            out.append((f"Lavg{'-'.join(map(str, nums))}", nums))
        else:
            out.append((f"L{t}", [int(t)]))
    return out


def map_layer_spec(spec, L):
    """--map_layers '8,9' → ('Lavg8-9', [8,9]) — 항상 평균 1버전"""
    nums = [int(t) for t in spec.split(",") if t.strip()]
    bad = [n for n in nums if not 0 <= n < L]
    if bad or not nums:
        sys.exit(f"[analyze] --map_layers 잘못됨: {spec}")
    label = f"L{nums[0]}" if len(nums) == 1 else f"Lavg{'-'.join(map(str, nums))}"
    return label, nums


def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def med_iqr(x):
    x = x[~np.isnan(x)]
    if not len(x):
        return "-"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.4f} [{q1:.4f},{q3:.4f}]"


def prior_rule_score(conv, layer_idxs, tol):
    """선행연구 규칙(ratio_prev 극대점=경계 제안)을 gold와 대조.
    반환: (제안 수, 적중 수, gold 수, 포착 수)"""
    n_prop = n_hit = n_gold = n_cov = 0
    for c in conv.values():
        r = reduce_layers(c["Q"]["ratio_prev"], layer_idxs)
        props = [t for t in range(1, len(r) - 1)
                 if np.isfinite(r[t - 1:t + 2]).all()
                 and r[t - 1] < r[t] > r[t + 1]]
        gold = c["gold"]
        n_prop += len(props)
        n_gold += len(gold)
        n_hit += sum(1 for p in props if any(abs(p - g) <= tol for g in gold))
        n_cov += sum(1 for g in gold if any(abs(p - g) <= tol for p in props))
    return n_prop, n_hit, n_gold, n_cov


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="Phase B 재설계: 경계 분석 figure + summary")
    base = Path(__file__).resolve()
    ap.add_argument("--gold", required=True, help="경계 JSON 위치")
    ap.add_argument("--data", default=str(base.parents[1] / "data"), help="stats_*.npz 위치")
    ap.add_argument("--layers", default="8,9,10,11,avg")
    ap.add_argument("--map_layers", default="8,9,10,11")
    ap.add_argument("--convs", default="", help="맵·막대 그릴 conv 제한 (콤마 구분)")
    ap.add_argument("--tile", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", default=str(base.parent / "output"))
    return ap.parse_args()


def main():
    args = parse_args()
    stats = load_stats(args.data)
    if not stats:
        sys.exit(f"[analyze] {args.data} 에 stats_*.npz 없음")
    gold = load_gold(args.gold)

    run_name = time.strftime("%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.output) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] 산출물 디렉토리: {out_dir}", flush=True)

    # CLS 확장 필드는 모든 conv에 있을 때만 사용
    first = next(iter(stats.values()))
    L = first["pair_sum"].shape[1] if len(first["pair_sum"]) else first["cls_mass"].shape[1]
    extra = [k for k in OPTIONAL_METRICS
             if all(k in d for d in stats.values())]
    metrics = list(METRIC_DESC) + extra

    layer_specs = parse_layer_specs(args.layers, L)
    map_label, map_idxs = map_layer_spec(args.map_layers, L)
    draw_convs = (set(x.strip() for x in args.convs.split(",") if x.strip())
                  if args.convs else set(stats))

    # conv별 파생값 + 경계 라벨
    conv = {}
    for sid, d in stats.items():
        if sid not in gold:
            print(f"[analyze] 경고: {sid} gold 없음 → 전부 비경계 취급", flush=True)
        conv[sid] = {"Q": derive(d, set(d)), "N": len(d["scored"]),
                     "gold": gold.get(sid, set()), "n_tokens": d["n_tokens"]}

    # 전 conv 풀링 (hist/boundary-avg/budget-split/layer-gap/summary용)
    P = {q: np.concatenate([c["Q"][q] for c in conv.values()])
         for q in metrics + ["mass_far", "mass_sep", "ratio_prev"]}
    is_b = np.concatenate([[i in c["gold"] for i in range(c["N"])] for c in conv.values()])
    n_tok = np.concatenate([c["n_tokens"] for c in conv.values()]).astype(float)
    rel_pos = np.concatenate([np.arange(c["N"]) / max(1, c["N"] - 1) for c in conv.values()])
    print(f"[analyze] conv {len(conv)}개, turns {len(is_b)}, "
          f"경계 {int(is_b.sum())}, 층 {L}, 지표 {metrics}", flush=True)

    write_summary(out_dir, run_name, args, conv, P, is_b, n_tok, rel_pos,
                  layer_specs, metrics, L, extra)
    make_figures(out_dir, args, stats, conv, P, is_b, layer_specs, metrics, L,
                 map_label, map_idxs, draw_convs)


# ══════════════════════════════════════════════════════════════════════
# summary.md
# ══════════════════════════════════════════════════════════════════════

FIGURE_GUIDE = """\
## 2. Figure 목록

모든 그림에서 빨간 실선은 gold chunking 경계를 나타낸다. 예외적으로
attnmap-band-corr에서는 셀 색(빨강/파랑)과의 구분을 위해 초록 실선을 사용한다.

### 2.1 attnmap-diag_{layer}_{conv}_tAAA-BBB.png — turn×turn 어텐션 맵

- 전체 turn×turn 어텐션 맵을 «tile»-turn 타일로 분할한다. 인접 타일은 «overlap»
  turns만큼 겹친다.
- y축: 어텐션을 보내는 query turn t / x축: 어텐션을 받는 key turn u.
- 색상: 할당된 어텐션 mass의 로그값. 자기 자신 칸과 분석 윈도우 밖은 회색.
- 해석: 경계선으로 구분된 동일 chunk 내부가 밝고 경계를 넘어선 영역이 어두우면,
  어텐션 패턴이 gold chunk 구조를 반영한다고 해석할 수 있다.

### 2.2 attnmap-band_{layer}_{conv}.png / attnmap-band-corr_... — 대각선 band 맵

- turn×turn 맵에서 대각선 주변만 추출하여 전체 대화를 펼친 그림이다. 한 행에
  «bandrow» turns를 표시한다.
- x축: 현재 turn / y축: 몇 turns 이전을 바라보는지 나타내는 거리 (1~«maxdist»).
- band: 어텐션 mass의 로그값. band-corr: 동일 거리(행)의 평균 대비 상대 비율 —
  빨간색은 평소보다 강한 어텐션, 파란색은 약한 어텐션 (경계선은 초록).
- 경계는 계단형 실선으로 표시된다. 계단선 아래(오른쪽 아래) 영역이 경계를 넘어
  과거를 보는 칸들이다 (turn b가 새 chunk 시작이면, 열 t에서 거리 t−b+1 이상인
  칸이 경계를 넘는다).
- 해석: 계단선 아래 영역이 어둡거나 파랗게 나타나면, 새 chunk가 시작될 때 이전
  turn들로 향하는 어텐션이 감소한다고 해석할 수 있다.

### 2.3 attnbar_{layer}_{conv}.png — 직전 turn 어텐션 막대그래프

- 각 turn의 mass_prev(직전 turn에 할당한 어텐션 예산)를 막대로 표시한다. 한 행에
  최대 «barrow» turns를 표시한다.
- 해석: 경계선 위치에서 막대가 주변보다 일관되게 낮거나 높으면 mass_prev를 경계
  탐지 신호로 사용할 가능성이 있다.

### 2.4 hist_{metric}_{layer}.png — 경계·비경계 분포 비교

- 경계 turn과 비경계 turn의 지표 값 분포를 겹쳐 그린다.
- 해석: 두 분포의 겹침이 작을수록 해당 지표와 레이어가 경계 구분에 유리하다.

### 2.5 boundary-avg_{metric}_{layer}.png — 경계 정렬 평균 곡선

- 모든 gold 경계를 x=0에 정렬한 뒤, 경계 전후 ±«k» turns의 지표 값을 평균낸다.
- 실선(boundary mean): 상대 위치별 경계 turn 평균 / 음영: 95% 신뢰구간 /
  회색 수평 점선(non boundary mean): 전체 비경계 turn의 평균값.
- 해석: x=0에서 곡선이 기준선에서 크게 벗어나면 해당 지표에 경계 신호가 존재한다.
  벗어남이 유지되는 turns 범위로 경계 판정에 필요한 문맥 크기를 추정할 수 있다.

### 2.6 budget-split.png — 어텐션 예산 분배

- 경계·비경계 turn이 평균적으로 어텐션 예산을 어디(직전 turn / 더 먼 과거 turn /
  자기 자신 / CLS / SEP)에 사용했는지 비교한다. 다섯 몫의 합은 예산 전체(1.0)이다.
  --layers 설정과 무관하게 모든 레이어(0~«lmax»)를 패널로 그린다.
- 해석: 경계에서 직전 turn의 몫이 줄고 자기 자신·CLS의 몫이 늘면 어텐션 싱크
  가설(경계에서 갈 곳을 잃은 어텐션이 싱크로 재분배된다)과 부합한다.

### 2.7 layer-gap.png — 레이어별 그룹 차이

- 지표별로 레이어 0~«lmax»의 경계 그룹·비경계 그룹 평균값을 비교한다.
- 해석: 두 선이 가장 크게 벌어지는 레이어일수록 경계 구분에 유리하다. 분석 대상
  레이어(--layers) 선택의 근거로 사용한다.
"""


def write_summary(out_dir, run_name, args, conv, P, is_b, n_tok, rel_pos,
                  layer_specs, metrics, L, extra):
    lines = ["# Attention Boundary Phase B 분석 요약\n",
             "## 1. 실험 개요\n",
             "| 항목 | 설정 |", "|---|---|",
             f"| 실행 ID | {run_name} |",
             f"| 데이터 경로 | {args.data} |",
             f"| Gold chunking | {args.gold} |",
             f"| 분석 레이어 | {args.layers} |",
             f"| 어텐션 맵 레이어 | {args.map_layers} |",
             f"| 타일 크기 | {args.tile} turns |",
             f"| 경계 주변 분석 범위 | ±{args.k} turns |",
             f"| 대화 수 | {len(conv)} |",
             f"| 전체 turn 수 | {len(is_b)} |",
             f"| 경계 turn 수 | {int(is_b.sum())} |",
             f"| 비경계 turn 수 | {int((~is_b).sum())} |",
             f"| CLS 확장 필드 | {extra if extra else '없음 — extract_attn 재실행 전'} |",
             "",
             "**Gold 경계의 정의** — 새로운 chunk가 시작되는 turn을 경계로 정의한다. "
             "세션 경계와 세션 내부의 의미적 경계를 별도로 구분하지 않는다.",
             "",
             "**mass의 정의** — 특정 turn의 토큰들이 가진 전체 어텐션 예산 중 특정 "
             "대상에 할당한 비율이다. 각 토큰의 어텐션 예산 합은 1이다.",
             "",
             FIGURE_GUIDE.replace("«tile»", str(args.tile))
                 .replace("«overlap»", str(TILE_OVERLAP))
                 .replace("«bandrow»", str(BAND_PER_ROW))
                 .replace("«barrow»", str(BAR_PER_ROW))
                 .replace("«maxdist»", str(MAX_DIST))
                 .replace("«k»", str(args.k))
                 .replace("«lmax»", str(L - 1))]

    # ── 3) 핵심 결과: 선행연구 규칙 재현 ───────────────────────────
    pl = [l for l in PRIOR_LAYERS if l < L]
    lines += ["## 3. 핵심 결과\n",
              "### 3.1 LightMem 경계 규칙 재현\n",
              "LightMem segmenter의 규칙을 그대로 적용하였다.\n",
              "- 지표: ratio_prev — 과거 turn에 할당한 어텐션 중 직전 turn에 할당한 비율",
              f"- 사용 레이어: {', '.join(map(str, pl))}의 평균",
              "- 경계 제안 규칙: ratio_prev가 양옆 turn보다 큰 국소 극대점(local "
              "maximum)을 경계 후보로 선택\n",
              "| 허용 오차 | 경계 제안 수 | Precision | Gold 경계 수 | Recall |",
              "|---|---|---|---|---|"]
    scores = {}
    for tol in (0, 1):
        n_prop, n_hit, n_gold, n_cov = prior_rule_score(conv, pl, tol)
        scores[tol] = (n_prop, n_hit, n_gold, n_cov)
        prec = f"{n_hit} / {n_prop} = {n_hit / n_prop:.1%}" if n_prop else "-"
        rec = f"{n_cov} / {n_gold} = {n_cov / n_gold:.1%}" if n_gold else "-"
        lines.append(f"| ±{tol} turn | {n_prop} | {prec} | {n_gold} | {rec} |")
    n_prop, _, n_gold, _ = scores[1]
    r0 = scores[0][3] / n_gold if n_gold else 0
    r1 = scores[1][3] / n_gold if n_gold else 0
    p1 = scores[1][1] / n_prop if n_prop else 0
    lines += ["", "### 3.2 요약\n",
              f"- 정확히 같은 turn에서 경계를 맞히는 성능은 낮다 (recall {r0:.1%}).",
              f"- 허용 오차를 ±1 turn으로 확장하면 recall이 {r0:.1%}에서 {r1:.1%}로 "
              "증가한다. 신호가 경계 turn 자체보다 인접 turn에 나타나는 경향이 있다.",
              f"- 제안 수({n_prop})가 gold 경계 수({n_gold})의 {n_prop / max(n_gold, 1):.1f}"
              f"배라 precision은 {p1:.1%}에 그친다. 후보를 걸러내는 조건이 개선 지점이다.",
              ""]

    # ── 4) 세부 통계 ────────────────────────────────────────────────
    lines += ["## 4. 세부 통계\n",
              "| 지표 | 의미 |", "|---|---|"]
    for q in metrics:
        desc = METRIC_DESC.get(q) or OPTIONAL_METRICS.get(q, "")
        lines.append(f"| {q} | {desc} |")
    lines += ["| ratio_prev | 과거 turn에 할당한 어텐션 중 직전 turn에 할당한 비율 |",
              "",
              "길이 상관·위치 상관은 지표와 turn 길이(토큰 수)·대화 내 상대 위치 "
              "사이의 Spearman 순위상관계수이다. 절댓값이 0에 가까울수록 해당 편향이 작다.\n"]
    for label, idxs in layer_specs:
        lines += [f"### {label}\n",
                  "| 지표 | 경계 중앙값 [IQR] | 비경계 중앙값 [IQR] | 길이 상관 | 위치 상관 |",
                  "|---|---|---|---|---|"]
        for q in metrics:
            x = reduce_layers(P[q], idxs)
            lines.append(f"| {q} | {med_iqr(x[is_b])} | {med_iqr(x[~is_b])} "
                         f"| {spearman(x, n_tok):+.3f} | {spearman(x, rel_pos):+.3f} |")
        lines.append("")

    # ── 5) 해석 (중앙값·상관 기반 자동 요약) ────────────────────────
    lines.append("## 5. 해석\n")
    max_pos = 0.0
    for q in metrics:
        dirs, lcorrs = [], []
        for _, idxs in layer_specs:
            x = reduce_layers(P[q], idxs)
            bm, nm = np.nanmedian(x[is_b]), np.nanmedian(x[~is_b])
            dirs.append(np.sign(bm - nm))
            lcorrs.append(spearman(x, n_tok))
            max_pos = max(max_pos, abs(spearman(x, rel_pos)))
        if all(d < 0 for d in dirs):
            msg = f"- {q}: 모든 분석 레이어에서 경계 turn의 중앙값이 비경계보다 낮다."
        elif all(d > 0 for d in dirs):
            msg = f"- {q}: 모든 분석 레이어에서 경계 turn의 중앙값이 비경계보다 높다."
        else:
            msg = (f"- {q}: 경계·비경계 차이의 방향이 레이어에 따라 일관되지 않아 "
                   "단독 경계 지표로 쓰기 어렵다.")
        worst = max(lcorrs, key=abs)
        if abs(worst) >= 0.3:
            msg += (f" 다만 turn 길이와의 상관(최대 {worst:+.3f})이 커서 길이 통제 "
                    "후에도 차이가 유지되는지 검증이 필요하다.")
        lines.append(msg)
    if max_pos < 0.1:
        lines.append("- 위치 편향: 모든 지표의 위치 상관이 0에 가까워, 경계·비경계 "
                     "차이가 대화 내 위치 효과일 가능성은 크지 않다.")
    else:
        lines.append(f"- 위치 편향: 위치 상관 절댓값 최대 {max_pos:.3f} — 위치 효과 "
                     "가능성을 확인할 필요가 있다.")
    lines += [f"- LightMem 극대점 규칙은 ±1 turn 기준 recall {r1:.1%}로 경계 근처를 "
              f"포착하지만, 과잉 제안으로 precision이 {p1:.1%}에 그친다.",
              "",
              "위 서술은 중앙값과 상관계수에 기반한 자동 요약이다. 최종 판단은 figure와 "
              "함께 내릴 것."]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("[analyze] summary.md 저장", flush=True)


# ══════════════════════════════════════════════════════════════════════
# figures
# ══════════════════════════════════════════════════════════════════════

def make_figures(out_dir, args, stats, conv, P, is_b, layer_specs, metrics, L,
                 map_label, map_idxs, draw_convs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.colors import LogNorm, TwoSlopeNorm
        # 한글 라벨 폰트 (Windows/Linux 순서로 폴백)
        plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("[analyze] matplotlib 없음 - figure 생략 (summary.md만 생성)", flush=True)
        return

    BOUND = dict(color="red", lw=1.2)             # 경계 = 빨간 실선 (전 figure 공통)
    GRAY = "#f0f0f0"

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)

    def robust_lognorm(a):
        pos = a[np.isfinite(a) & (a > 0)]
        if not len(pos):
            return None
        return LogNorm(np.percentile(pos, 5), np.percentile(pos, 99.5))

    # ── 1) attnmap: diag 타일 + band + band-corr ───────────────────
    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad(GRAY)
    rdbu = plt.get_cmap("RdBu_r").copy()
    rdbu.set_bad(GRAY)

    for sid in sorted(draw_convs & set(conv)):
        c, N = conv[sid], conv[sid]["N"]
        M = build_mass_matrix(stats[sid], map_idxs)
        np.fill_diagonal(M, np.nan)               # 자기자신 마스킹 (색 범위 독점 방지)
        norm = robust_lognorm(M)
        if norm is None:
            continue

        # diag 타일
        stride = max(args.tile - TILE_OVERLAP, 1)
        w0 = 0
        while w0 < N:
            w1 = min(w0 + args.tile, N)
            if w1 - w0 >= 10:
                T = w1 - w0
                size = max(5.0, 0.13 * T)
                fig, ax = plt.subplots(figsize=(size * 1.1, size))
                ax.imshow(np.ma.masked_invalid(M[w0:w1, w0:w1]),
                          cmap=viridis, norm=norm, interpolation="nearest")
                for b in c["gold"]:
                    if w0 < b < w1:
                        ax.axvline(b - w0 - 0.5, **BOUND)  # key 축 위치만 (가로선 없음)
                ticks = list(range(0, T, max(1, T // 15)))
                ax.set_xticks(ticks, [str(w0 + x) for x in ticks], fontsize=7, rotation=90)
                ax.set_yticks(ticks, [str(w0 + x) for x in ticks], fontsize=7)
                ax.set_xlabel("key turn u")
                ax.set_ylabel("query turn t")
                ax.set_title(f"{sid} {map_label} turns {w0}-{w1 - 1} | mass(log), "
                             "red=gold boundary", fontsize=9)
                cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, format="{x:.3g}")
                # 보조 눈금 라벨을 mathtext 대신 일반 숫자로 (한글 폰트에 U+2212 없음)
                cb.ax.yaxis.set_minor_formatter(
                    mticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(1, 0.4)))
                save(fig, f"attnmap-diag_{map_label}_{sid}_t{w0:03d}-{w1:03d}.png")
            if w1 == N:
                break
            w0 += stride

        # band: B[k-1, t] = t가 k턴 앞에 쓴 mass
        B = np.full((MAX_DIST, N), np.nan, np.float32)
        for k in range(1, MAX_DIST + 1):
            B[k - 1, k:] = M[np.arange(k, N), np.arange(0, N - k)]

        def band_fig(mat, cmap, nrm, cbar_label, fname, line_color="red"):
            # 여러 줄로 분할 (한 줄 통짜는 너무 빽빽해 판독 불가)
            per_row = BAND_PER_ROW
            nrows = int(np.ceil(N / per_row))
            fig, axes = plt.subplots(nrows, 1, figsize=(14, 2.1 * nrows), squeeze=False)
            im = None
            for r in range(nrows):
                ax = axes[r][0]
                r0, r1 = r * per_row, min((r + 1) * per_row, N)
                im = ax.imshow(np.ma.masked_invalid(mat[:, r0:r1]), cmap=cmap, norm=nrm,
                               aspect="auto", interpolation="nearest",
                               extent=(r0 - 0.5, r1 - 0.5, MAX_DIST - 0.5, -0.5))
                for b in c["gold"]:
                    if b - 0.5 <= r1 and b + MAX_DIST - 0.5 >= r0:
                        # 계단선: 선 아래(오른쪽 아래)가 경계를 넘어 과거를 보는 칸들
                        xs, ys = [b - 0.5], [-0.5]
                        for i in range(MAX_DIST):
                            xs += [b + i + 0.5, b + i + 0.5]
                            ys += [i - 0.5, i + 0.5]
                        ax.plot(xs, ys, color=line_color, lw=1.2,
                                solid_capstyle="butt")
                ax.set_xlim(r0 - 0.5, r0 + per_row - 0.5)
                ax.set_yticks(range(MAX_DIST), [str(k) for k in range(1, MAX_DIST + 1)],
                              fontsize=7)
                ax.set_ylabel("몇 턴 앞", fontsize=8)
            axes[0][0].set_title(f"{sid} {map_label} | {cbar_label}, "
                                 f"{line_color}=gold boundary", fontsize=9)
            axes[-1][0].set_xlabel("turn t")
            cb = fig.colorbar(im, ax=[a[0] for a in axes], fraction=0.02, pad=0.01,
                              format="{x:.3g}")
            if isinstance(nrm, LogNorm):
                cb.ax.yaxis.set_minor_formatter(
                    mticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(1, 0.4)))
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        band_fig(B, viridis, robust_lognorm(B), "mass(log)",
                 f"attnmap-band_{map_label}_{sid}.png")

        # band-corr: 같은 거리(행) 평균 대비 비율 → 거리빨 제거
        # 경계선은 초록 — RdBu 셀 색(빨강/파랑)과 겹치지 않는 유일한 색
        with np.errstate(invalid="ignore"):
            ratio = B / np.nanmean(B, axis=1, keepdims=True)
        lo = min(float(np.nanpercentile(ratio, 2)), 0.9)
        hi = max(float(np.nanpercentile(ratio, 98)), 1.1)
        band_fig(ratio, rdbu, TwoSlopeNorm(vmin=lo, vcenter=1.0, vmax=hi),
                 "mass / 같은 거리 평균 (빨강=강함, 파랑=약함)",
                 f"attnmap-band-corr_{map_label}_{sid}.png", line_color="limegreen")

    # ── 2) attnbar: turn별 mass_prev 막대 ──────────────────────────
    for label, idxs in layer_specs:
        for sid in sorted(draw_convs & set(conv)):
            c, N = conv[sid], conv[sid]["N"]
            vals = reduce_layers(c["Q"]["mass_prev"], idxs)
            nrows = int(np.ceil(N / BAR_PER_ROW))
            ymax = float(np.nanpercentile(vals, 99.5)) * 1.1
            fig, axes = plt.subplots(nrows, 1, figsize=(14, 1.7 * nrows),
                                     squeeze=False, sharey=True)
            for r in range(nrows):
                ax = axes[r][0]
                r0, r1 = r * BAR_PER_ROW, min((r + 1) * BAR_PER_ROW, N)
                ax.bar(range(r0, r1), np.nan_to_num(vals[r0:r1]),
                       width=0.85, color="steelblue")
                for b in c["gold"]:
                    if r0 <= b < r1:
                        ax.axvline(b - 0.5, **BOUND)  # 경계는 turn 사이 → 막대 사이에
                ax.set_xlim(r0 - 0.5, r0 + BAR_PER_ROW - 0.5)
                ax.set_ylim(0, ymax)
                ax.set_ylabel("mass_prev", fontsize=7)
            axes[0][0].set_title(f"{sid} {label} mass_prev | red=gold boundary", fontsize=9)
            axes[-1][0].set_xlabel("turn t")
            save(fig, f"attnbar_{label}_{sid}.png")

    # ── 3) hist: 경계 vs 비경계 분포 ───────────────────────────────
    for label, idxs in layer_specs:
        for q in metrics:
            x = reduce_layers(P[q], idxs)
            v = ~np.isnan(x)
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            ax.hist(x[~is_b & v], bins=40, density=True, alpha=0.55,
                    color="steelblue", label="non boundary")
            ax.hist(x[is_b & v], bins=40, density=True, alpha=0.55,
                    color="red", label="gold boundary")
            ax.set_xlabel(q); ax.set_ylabel("density")
            ax.set_title(f"{q} {label}", fontsize=10)
            ax.legend(fontsize=8)
            save(fig, f"hist_{q}_{label}.png")

    # ── 4) boundary-avg: 경계 정렬 ±k턴 평균 곡선 ──────────────────
    k = args.k
    offsets = np.arange(-k, k + 1)
    for label, idxs in layer_specs:
        for q in metrics:
            cols = [[] for _ in offsets]          # offset별 값 수집
            for c in conv.values():
                s = reduce_layers(c["Q"][q], idxs)
                for b in c["gold"]:
                    for oi, off in enumerate(offsets):
                        t = b + off
                        if 0 <= t < c["N"] and not np.isnan(s[t]):
                            cols[oi].append(s[t])
            mean = np.array([np.mean(v) if v else np.nan for v in cols])
            ci = np.array([1.96 * np.std(v) / np.sqrt(len(v)) if len(v) > 1 else np.nan
                           for v in cols])
            base = np.nanmean(reduce_layers(P[q], idxs)[~is_b])  # 비경계 평균 기준선

            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(offsets, mean, marker="o", ms=4, color="steelblue",
                    label="boundary mean")
            ax.fill_between(offsets, mean - ci, mean + ci, alpha=0.25,
                            color="steelblue", label="95% CI")
            ax.axhline(base, color="gray", ls="--", lw=1, label="non boundary mean")
            ax.axvline(0, **BOUND)
            ax.set_xlabel("경계로부터의 상대 turn (0 = 경계 turn)")
            ax.set_ylabel(q)
            ax.set_title(f"{q} {label} — 경계 주변 평균", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")  # loc 고정 (그림마다 이동 방지)
            save(fig, f"boundary-avg_{q}_{label}.png")

    # ── 5) budget-split: 평균 예산 구성 (--layers와 무관하게 전 층) ─
    ncols = 6
    nrows_b = int(np.ceil(L / ncols))
    fig, axes = plt.subplots(nrows_b, ncols, figsize=(2.3 * ncols, 3.4 * nrows_b),
                             squeeze=False, sharey=True)
    for l in range(L):
        ax = axes[l // ncols][l % ncols]
        bottoms = np.zeros(2)
        for comp in BUDGET_COMPS:
            x = P[comp][:, l]
            vals = np.array([np.nanmean(x[is_b]), np.nanmean(x[~is_b])])
            ax.bar(["boundary", "non boundary"], vals, bottom=bottoms, label=comp)
            bottoms += vals
        ax.set_title(f"L{l}", fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    for j in range(L, nrows_b * ncols):          # 남는 칸 숨김
        axes[j // ncols][j % ncols].axis("off")
    for r in range(nrows_b):
        axes[r][0].set_ylabel("mean mass")
    fig.legend(*axes[0][0].get_legend_handles_labels(), fontsize=8,
               loc="upper right", bbox_to_anchor=(0.995, 0.99))
    fig.suptitle("어텐션 예산 분배 — 경계 vs 비경계 (전 층)", fontsize=11)
    save(fig, "budget-split.png")

    # ── 6) layer-gap: 층별 그룹 평균 (층 선택 근거) ─────────────────
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4),
                             squeeze=False)
    for j, q in enumerate(metrics):
        ax = axes[0][j]
        ax.plot(range(L), [np.nanmean(P[q][:, l][is_b]) for l in range(L)],
                marker="o", ms=4, color="red", label="gold boundary")
        ax.plot(range(L), [np.nanmean(P[q][:, l][~is_b]) for l in range(L)],
                marker="s", ms=4, color="steelblue", label="non boundary")
        ax.set_xlabel("layer"); ax.set_title(q, fontsize=10)
        if j == 0:
            ax.set_ylabel("mean"); ax.legend(fontsize=8)
    save(fig, "layer-gap.png")

    print(f"[analyze] figure 저장 완료 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
