"""
pick_threshold.py — 확정된 신호 위에서 임계값 tau를 고르는 도구 (Phase B, 청커 설정용).

`pick_signal.py`가 "어떤 대표값을 쓸지"(A·B·층)를 골랐다면, 이 스크립트는 그 대표값 위에서
"얼마나 낮아야 자를지"를 고른다. 규칙은 청커(`core/chunker/attn_chunker.py`)와 동일하다:

    자른다 = 골짜기(s[t-1] > s[t] < s[t+1])  AND  s[t] < tau

보조 규칙(최소 간격·최대 청크 크기)은 없다 — 결과를 보고 만든 규칙이 되므로 넣지 않았다.
계산은 `pick_signal`의 prepare/token_values/turn_values/propose를 그대로 재사용한다.
GPU 불필요 — npz만 읽는다. 몇 초.

실행 (MemoryAgnostic 루트에서):
  python experiment/attn_boundary/analysis/pick_threshold.py \
      --gold data/chunked_data/chunks_fable-5_v2.json
옵션: --tau 0.0020,0.0030,0.00002 (min,max,step)  --layers 7,8,9  --a top5  --b r85
      --out <표.md>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from analyze import load_gold, load_stats  # 같은 폴더
from pick_signal import prepare, propose, token_values, turn_values

DEFAULT_LAYERS = [7, 8, 9]   # FINDINGS §8: 층 부분집합 4,095개 전수 탐색에서 4위(상위권은 오차 내)


# ══════════════════════════════════════════════════════════════════════
# 신호 · 규칙
# ══════════════════════════════════════════════════════════════════════

def build_scores(convs, a_spec, b_spec, layers):
    """{sid: s[N]} — A(토큰 접기) → B(turn 접기) → 층 평균. 청커 _score()와 같은 순서."""
    out = {}
    for sid, c in convs.items():
        S = turn_values(c, token_values(c, a_spec), b_spec)   # [turn, 층]
        out[sid] = S[:, layers].mean(axis=1)
    return out


def cuts_for(s, tau):
    """골짜기 중 tau 미만인 곳 = 경계."""
    return [int(t) for t in propose(s, "valley") if s[t] < tau]


def evaluate(scores, gold, tau):
    n_prop = n_hit = n_gold = n_cov = 0
    sizes = []
    for sid, s in scores.items():
        cuts = cuts_for(s, tau)
        g = gold.get(sid, set())
        n_prop += len(cuts)
        n_gold += len(g)
        n_hit += sum(1 for x in cuts if x in g)
        n_cov += sum(1 for x in g if x in cuts)
        edges = [0] + cuts + [len(s)]
        sizes += [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    p = n_hit / n_prop if n_prop else 0.0
    r = n_cov / n_gold if n_gold else 0.0
    return dict(tau=tau, n=n_prop, P=p, R=r,
                F1=(2 * p * r / (p + r) if p + r else 0.0),
                **size_stats(sizes))


def size_stats(sizes):
    a = np.array(sizes, dtype=float)
    return {"n_chunk": len(a), "mean": a.mean(), "med": np.median(a),
            "p95": np.percentile(a, 95), "max": a.max()}


def gold_row(scores, gold):
    """비교 기준선: gold 자체의 청크 크기 분포."""
    sizes = []
    for sid, s in scores.items():
        edges = [0] + sorted(gold.get(sid, set())) + [len(s)]
        sizes += [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    return {"tau": None, "n": sum(len(gold.get(sid, set())) for sid in scores),
            "P": None, "R": None, "F1": None, **size_stats(sizes)}


# ══════════════════════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════════════════════

HEAD = ("| tau | 컷 | 정확도 | 재현율 | F1 | 청크 | 평균 | 중앙 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def fmt(r, mark=""):
    tau = "**gold**" if r["tau"] is None else f"{r['tau']:.5f}{mark}"
    pct = lambda v: "—" if v is None else f"{v:.1%}"
    f1 = "—" if r["F1"] is None else f"**{r['F1']:.3f}**"
    return (f"| {tau} | {r['n']} | {pct(r['P'])} | {pct(r['R'])} | {f1} | "
            f"{r['n_chunk']} | {r['mean']:.1f} | {r['med']:.0f} | "
            f"{r['p95']:.0f} | {r['max']:.0f} |")


def parse_args():
    base = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description="어텐션 골짜기 청커의 임계값 tau 고르기")
    ap.add_argument("--gold", required=True, help="경계 JSON (chunk JSON)")
    ap.add_argument("--data", default=str(base.parents[1] / "data"), help="stats_*.npz 위치")
    ap.add_argument("--tau", default="0.0018,0.0030,0.00002", help="min,max,step")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--a", default="top5", help="토큰 접기 (pick_signal의 A)")
    ap.add_argument("--b", default="r85", help="turn 접기 (pick_signal의 B)")
    ap.add_argument("--rows", type=int, default=25, help="화면에 뿌릴 tau 행 수 (균등 추출)")
    ap.add_argument("--out", default="", help="전체 표를 쓸 .md 경로")
    return ap.parse_args()


def main():
    args = parse_args()
    stats = load_stats(args.data)
    if not stats:
        sys.exit(f"[pick_threshold] {args.data} 에 stats_*.npz 없음")

    convs = {sid: prepare(d) for sid, d in stats.items()}
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    L = convs[next(iter(convs))]["L"]
    bad = [l for l in layers if not 0 <= l < L]
    if bad:
        sys.exit(f"[pick_threshold] 층 범위 벗어남 {bad} (0~{L - 1})")

    scores = build_scores(convs, args.a, args.b, layers)
    gold = load_gold(args.gold)
    lo, hi, step = (float(x) for x in args.tau.split(","))
    taus = np.round(np.arange(lo, hi + step / 2, step), 8)

    n_gold = sum(len(gold.get(sid, set())) for sid in scores)
    n_scored = sum(int(c["scored"].sum()) for c in convs.values())
    n_valley = sum(len(propose(s, "valley")) for s in scores.values())
    print(f"[pick_threshold] conv {len(scores)}개 · 층 {layers} · A={args.a} B={args.b}")
    print(f"[pick_threshold] gold 경계 {n_gold}개 · 채점 turn {n_scored}개 · "
          f"골짜기 후보 {n_valley}개")
    print(f"[pick_threshold] 무작위로 찍었을 때 기대 정확도: {n_gold / n_scored:.1%}\n")

    rows = [evaluate(scores, gold, float(t)) for t in taus]
    best = max(rows, key=lambda r: r["F1"])
    near = min(rows, key=lambda r: abs(r["n"] - n_gold))   # gold와 컷 수가 가장 비슷한 tau

    # 화면에는 균등 간격으로 --rows 개만 (전체는 --out)
    stride = max(1, len(rows) // args.rows)
    shown = rows[::stride]
    for r in (best, near):
        if r not in shown:
            shown.append(r)
    shown.sort(key=lambda r: r["tau"])

    print("\n".join(HEAD))
    print(fmt(gold_row(scores, gold)))
    for r in shown:
        mark = " ★" if r is best else (" ○" if r is near else "")
        print(fmt(r, mark))
    print(f"\n★ F1 최대: tau={best['tau']:.5f} F1={best['F1']:.3f} 컷={best['n']} "
          f"평균={best['mean']:.1f}턴")
    print(f"○ gold와 컷 수가 가장 비슷: tau={near['tau']:.5f} 컷={near['n']} "
          f"(gold {n_gold}) F1={near['F1']:.3f}")
    print("\n※ F1 곡선이 평평한 구간에서는 tau 하나를 고른 게 아니라 '그 대역'을 고른 것이다.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = ["# 임계값 tau 스윕 (골짜기 ∧ s < tau)\n",
                f"conv {len(scores)}개 · 층 {layers} · A={args.a} · B={args.b} · "
                f"gold {args.gold} (경계 {n_gold}개) · 골짜기 후보 {n_valley}개 · "
                f"무작위 기대 정확도 {n_gold / n_scored:.1%}\n",
                *HEAD, fmt(gold_row(scores, gold))]
        body += [fmt(r, " ★" if r is best else "") for r in rows]
        out.write_text("\n".join(body), encoding="utf-8")
        print(f"\n[pick_threshold] 전체 표 저장: {out}")


if __name__ == "__main__":
    main()
