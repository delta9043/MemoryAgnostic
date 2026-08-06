"""
pick_signal.py — turn 대표값 후보를 gold 경계로 채점해 고르는 도구 (Phase B, 청커 설계용).

extract_attn.py가 저장한 토큰별 값(tok_sum / tok_topk)에서 turn 대표값을 만드는 방식을
전부 조합해 보고, "경계 = 골짜기(극소)" 규칙이 gold 경계를 얼마나 재현하는지 표로 낸다.
GPU 불필요 — npz만 읽는다.

  (A) 토큰 i 하나의 값 : A[i, :]  (직전 turn의 토큰들)을 하나로
      sum      전부 더함              (직전 turn 길이에 민감)
      mean     sum / 직전 turn 토큰 수
      top{k}   큰 것 k개 평균         (k=1이면 최대)
  (B) turn 하나의 값   : (A)로 나온 토큰별 값들을 하나로
      r{pct}   상위 pct%만 평균       (r100=전부 평균, r85=하위 15% 절사)
      max      가장 큰 값 하나
      median   중앙값
      first    첫 토큰 값

  규칙: valley  s[t-1] > s[t] < s[t+1]  → turn t가 새 chunk 시작 (우리 가설)
        peak    s[t-1] < s[t] > s[t+1]  → 선행연구(LightMem) 규칙, 비교용

실행 (MemoryAgnostic 루트에서):
  python experiment/attn_boundary/analysis/pick_signal.py \
      --gold data/chunked_data/chunks_fable-5.json
옵션: --k 1,2,3,5,10  --r 100,85,50,25  --tol 0,1  --top 20  --out <표.md>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from analyze import derive, load_gold, load_stats, reduce_layers  # 같은 폴더

PRIOR_LAYERS = [8, 9, 10, 11]  # 선행연구(LightMem segmenter)가 쓰는 층 평균


# ══════════════════════════════════════════════════════════════════════
# conv 1개 준비 (토큰 배열을 turn 순서로 정렬 + 파생값)
# ══════════════════════════════════════════════════════════════════════

def prepare(d):
    """npz 1개 → 채점에 필요한 것만 담은 dict."""
    N = len(d["scored"])
    L = d["tok_sum"].shape[1]

    # 토큰 배열을 owner(turn) 순으로. stable이라 turn 내부의 토큰 순서는 보존된다(first용).
    order = np.argsort(d["tok_owner"], kind="stable")
    owner = d["tok_owner"][order]
    tok_sum = d["tok_sum"][order]
    tok_topk = d["tok_topk"][order]

    # turn별 토큰 구간 [(t, slice), ...]
    uniq, start, count = np.unique(owner, return_index=True, return_counts=True)
    groups = [(int(t), slice(int(s), int(s + c))) for t, s, c in zip(uniq, start, count)]

    # 직전 turn 쌍의 nk (= 윈도우에 보인 t-1의 토큰 수). A의 mean에 쓴다.
    nk_prev = np.full(N, np.nan, np.float32)
    is_prev = d["pair_prev"] == d["pair_owner"] - 1
    nk_prev[d["pair_owner"][is_prev]] = d["pair_nk"][is_prev]

    Q = derive(d, set(d))  # 기존 파생값 (mass_prev·ratio_prev = 검증/선행규칙용)
    return {
        "N": N, "L": L, "groups": groups, "owner": owner,
        "tok_sum": tok_sum, "tok_topk": tok_topk, "nk_prev": nk_prev,
        "n_tokens": d["n_tokens"].astype(np.float32), "scored": d["scored"],
        "mass_prev": Q["mass_prev"], "ratio_prev": Q["ratio_prev"],
        "pair_meanmax": d["pair_meanmax"][is_prev], "pair_max": d["pair_max"][is_prev],
        "prev_owner": d["pair_owner"][is_prev],
        "partial_prev": int((d["pair_nk"][is_prev] < d["n_tokens"][d["pair_owner"][is_prev] - 1]).sum()),
    }


# ══════════════════════════════════════════════════════════════════════
# (A) 토큰 하나의 값 → [토큰, 층]
# ══════════════════════════════════════════════════════════════════════

def token_values(c, spec):
    if spec == "sum":
        return c["tok_sum"]
    if spec == "mean":
        return c["tok_sum"] / c["nk_prev"][c["owner"]][:, None]
    k = int(spec[3:])                                  # "top3" → 3
    return np.nanmean(c["tok_topk"][:, :, :k], axis=2)


# ══════════════════════════════════════════════════════════════════════
# (B) turn 하나의 값 → [turn, 층]
# ══════════════════════════════════════════════════════════════════════

def turn_values(c, v, spec):
    out = np.full((c["N"], c["L"]), np.nan, np.float32)
    for t, sl in c["groups"]:
        g = v[sl]                                      # [그 turn의 토큰 수, 층]
        if spec == "first":
            out[t] = g[0]
        elif spec == "median":
            out[t] = np.median(g, axis=0)
        else:
            gs = -np.sort(-g, axis=0)                  # 층마다 내림차순
            m = 1 if spec == "max" else max(1, int(np.ceil(len(g) * int(spec[1:]) / 100)))
            out[t] = gs[:m].mean(axis=0)
    return out


# ══════════════════════════════════════════════════════════════════════
# 규칙 · 채점
# ══════════════════════════════════════════════════════════════════════

def propose(s, mode):
    """골짜기/봉우리 turn 인덱스. 양옆 중 하나라도 값이 없으면 판정하지 않는다."""
    ok = np.isfinite(s)
    mid, left, right = s[1:-1], s[:-2], s[2:]
    both = ok[1:-1] & ok[:-2] & ok[2:]
    hit = ((mid < left) & (mid < right)) if mode == "valley" else ((mid > left) & (mid > right))
    return np.where(both & hit)[0] + 1


def score(props_per_conv, gold_per_conv, tol):
    """(제안 수, 적중 수, gold 수, 포착 수)"""
    n_prop = n_hit = n_gold = n_cov = 0
    for props, gold in zip(props_per_conv, gold_per_conv):
        g = np.fromiter(gold, dtype=int, count=len(gold))
        n_prop += len(props)
        n_gold += len(g)
        if len(props) and len(g):
            near = np.abs(props[:, None] - g[None, :]) <= tol
            n_hit += int(near.any(axis=1).sum())
            n_cov += int(near.any(axis=0).sum())
    return n_prop, n_hit, n_gold, n_cov


def prf(n_prop, n_hit, n_gold, n_cov):
    p = n_hit / n_prop if n_prop else 0.0
    r = n_cov / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def run_rule(S_per_conv, layer, gold_per_conv, tols, mode="valley"):
    """한 조합(지표+층)을 허용 오차별로 채점 → 행 하나."""
    props = [propose(S[:, layer], mode) for S in S_per_conv]
    row = {"n_prop": sum(len(p) for p in props)}
    for tol in tols:
        p, r, f = prf(*score(props, gold_per_conv, tol))
        row[tol] = (p, r, f)
    return row


# ══════════════════════════════════════════════════════════════════════
# 진단 (저장이 제대로 됐는지 + 데이터 상태)
# ══════════════════════════════════════════════════════════════════════

def diagnose(convs):
    """토큰별 저장이 기존 집계와 일치하는지 3칸 확인 + 데이터 상태."""
    lines = []
    ok3 = [True, True, True]
    n_unscored = n_partial = n_turn = 0
    for sid, c in convs.items():
        a1 = turn_values(c, token_values(c, "sum"), "r100")     # == mass_prev
        t1 = token_values(c, "top1")
        m1 = turn_values(c, t1, "r100")                        # == pair_meanmax
        x1 = turn_values(c, t1, "max")                         # == pair_max
        owner = c["prev_owner"]
        for i, (ours, ref) in enumerate((
            (a1[owner], c["mass_prev"][owner]),
            (m1[owner], c["pair_meanmax"]),
            (x1[owner], c["pair_max"]),
        )):
            if not np.allclose(ours, ref, rtol=1e-3, atol=1e-6, equal_nan=True):
                ok3[i] = False
        n_unscored += int((~c["scored"]).sum())
        n_partial += c["partial_prev"]
        n_turn += c["N"]
    names = ["A1+B1(r100) == mass_prev", "top1+B1(r100) == pair_meanmax", "top1+B1(max) == pair_max"]
    for name, ok in zip(names, ok3):
        lines.append(f"- 저장 검증 {'OK ' if ok else '**불일치**'} : {name}")
    lines.append(f"- 채점 안 된 turn: {n_unscored} / {n_turn} "
                 f"(t=0 이거나 문맥이 안 잡힌 turn — 주변 판정도 같이 빠진다)")
    lines.append(f"- 직전 turn이 잘려 보인 turn: {n_partial} "
                 f"(창 예산 부족 — 많으면 신호가 부분 노출에 오염된다)")
    return lines


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    base = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description="turn 대표값 후보를 gold 경계로 채점")
    ap.add_argument("--gold", required=True, help="경계 JSON (chunk JSON)")
    ap.add_argument("--data", default=str(base.parents[1] / "data"), help="stats_*.npz 위치")
    ap.add_argument("--k", default="1,2,3,5,10", help="top-k의 k 목록 (k <= 저장된 K)")
    ap.add_argument("--r", default="100,85,50,25", help="상위 r%% 목록")
    ap.add_argument("--tol", default="0,1", help="허용 오차(turn) 목록")
    ap.add_argument("--top", type=int, default=20, help="화면에 뿌릴 상위 조합 수")
    ap.add_argument("--out", default="", help="전체 표를 쓸 .md 경로 (미지정 시 저장 안 함)")
    return ap.parse_args()


def main():
    args = parse_args()
    stats = load_stats(args.data)
    if not stats:
        sys.exit(f"[pick_signal] {args.data} 에 stats_*.npz 없음")
    if "tok_topk" not in next(iter(stats.values())):
        sys.exit("[pick_signal] npz에 tok_topk가 없다 — extract_attn.py를 다시 돌려야 한다")

    gold = load_gold(args.gold)
    convs = {sid: prepare(d) for sid, d in stats.items()}
    sids = list(convs)
    gold_per_conv = [gold.get(sid, set()) for sid in sids]
    L = convs[sids[0]]["L"]
    K = convs[sids[0]]["tok_topk"].shape[2]

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    bad = [k for k in ks if k > K]
    if bad:
        sys.exit(f"[pick_signal] 저장된 K={K}보다 큰 k={bad} — 재추출이 필요하다")
    A_SPECS = ["sum", "mean"] + [f"top{k}" for k in ks]
    B_SPECS = [f"r{r.strip()}" for r in args.r.split(",") if r.strip()] + ["max", "median", "first"]
    tols = [int(x) for x in args.tol.split(",") if x.strip()]

    n_gold = sum(len(g) for g in gold_per_conv)
    n_scored = sum(int(c["scored"].sum()) for c in convs.values())
    print(f"[pick_signal] conv {len(sids)}개 · 층 {L} · K {K} · gold 경계 {n_gold}개")
    print(f"[pick_signal] 조합 {len(A_SPECS)} × {len(B_SPECS)} × {L} = "
          f"{len(A_SPECS) * len(B_SPECS) * L}개\n")
    diag = diagnose(convs)
    print("\n".join(diag))

    # ── 전 조합 채점 ────────────────────────────────────────────────
    rows = []
    for a in A_SPECS:
        vs = {sid: token_values(c, a) for sid, c in convs.items()}
        for b in B_SPECS:
            S = [turn_values(convs[sid], vs[sid], b) for sid in sids]
            for layer in range(L):
                row = run_rule(S, layer, gold_per_conv, tols)
                row.update({"A": a, "B": b, "L": layer})
                rows.append(row)

    # ── 기준선 ──────────────────────────────────────────────────────
    base_rows = []
    n_tok = [np.where(convs[sid]["scored"], convs[sid]["n_tokens"], np.nan) for sid in sids]
    for mode in ("valley", "peak"):
        r = run_rule([s[:, None] for s in n_tok], 0, gold_per_conv, tols, mode)
        r.update({"A": "[기준] turn 길이", "B": "-", "L": "-", "mode": mode})
        base_rows.append(r)
    pl = [l for l in PRIOR_LAYERS if l < L]
    prior = [reduce_layers(convs[sid]["ratio_prev"], pl)[:, None] for sid in sids]
    for mode in ("peak", "valley"):
        r = run_rule(prior, 0, gold_per_conv, tols, mode)
        r.update({"A": "[기준] 선행 ratio_prev", "B": f"avg{pl}", "L": "-", "mode": mode})
        base_rows.append(r)

    # ── 출력 ────────────────────────────────────────────────────────
    head = "| A (토큰) | B (turn) | 층 | 제안 | " + " | ".join(
        f"P±{t} | R±{t} | F1±{t}" for t in tols) + " |"
    sep = "|---|---|---|---:|" + "---:|" * (3 * len(tols))

    def fmt(r, mode=""):
        cells = "".join(f" {r[t][0]:.1%} | {r[t][1]:.1%} | **{r[t][2]:.3f}** |" for t in tols)
        name = f"{r['A']}{f' ({mode})' if mode else ''}"
        return f"| {name} | {r['B']} | {r['L']} | {r['n_prop']} |{cells}"

    key = tols[0]
    rows.sort(key=lambda r: r[key][2], reverse=True)
    print(f"\n## 상위 {args.top}개 (valley 규칙, F1±{key} 기준)\n")
    print(head)
    print(sep)
    for r in rows[: args.top]:
        print(fmt(r))

    print("\n## 기준선\n")
    print(head)
    print(sep)
    for r in base_rows:
        print(fmt(r, r["mode"]))
    exp_p = {t: (2 * t + 1) * n_gold / n_scored for t in tols}
    print("\n무작위로 찍었을 때 기대 정확도: " +
          " / ".join(f"±{t} {exp_p[t]:.1%}" for t in tols) +
          "  ← 이보다 못하면 신호가 아니다")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = ["# turn 대표값 후보 채점 (valley 규칙)\n",
                f"conv {len(sids)}개 · 층 {L} · K {K} · gold 경계 {n_gold}개 · "
                f"채점된 turn {n_scored}개\n", *diag, "", head, sep]
        body += [fmt(r) for r in rows]
        body += ["", "## 기준선", "", head, sep] + [fmt(r, r["mode"]) for r in base_rows]
        out.write_text("\n".join(body), encoding="utf-8")
        print(f"\n[pick_signal] 전체 표 저장: {out}")


if __name__ == "__main__":
    main()
