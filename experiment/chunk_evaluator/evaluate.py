"""chunk_evaluator — 여러 청킹 결과(gpt, opus 등)를 쉬운 지표로 비교/검증한다.

목적(T3): gpt-5.6-sol이 찍은 경계가 '실제 대화 구조'인지 'gpt 혼자만의 변덕'인지 검증.
논리: 변덕이면 다른 모델은 다른 데를 자른다 → 독립적으로 같은 자리를 자르면 실제 구조다.

입력: chunk/ 안의 *.json (PrecomputedChunker 포맷: [{sample_id, chunks:[{turns:[...]}]}]).
      각 turn에 session_id가 있어 '진짜 세션 경계'를 파일에서 바로 유도한다(HTML 불필요).

지표(어려운 분할 통계 안 씀):
  1. 경계 겹침율   — 두 모델이 같은 자리를 잘랐나 (파일 2개 이상일 때)
  2. 세션 재현율   — gpt가 못 본 '진짜 세션 경계'를 맞혔나 (+ random 기준선)
  3. 청크 결       — 청크 개수/평균 크기 (granularity sanity)

파일 1개만 있으면 2·3만, 2개 이상이면 1까지 모두 출력한다.
실행:  python evaluate.py            # chunk/ 스캔
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
CHUNK_DIR = HERE / "chunk"
TOLS = [0, 1]            # 경계 일치 허용 오차(turn): 0=정확, 1=한 칸 차이도 같은 자리로 인정
RANDOM_TRIALS = 300
SEED = 0


# ---------- 로드 & 경계 유도 ----------
def load_file(path: Path) -> dict:
    """sample_id -> {sizes:[청크별 turn수], sess:[진짜 세션경계], n:turn수}"""
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for rec in data:
        sizes = [len(c["turns"]) for c in rec["chunks"]]
        sess_ids = [t["session_id"] for c in rec["chunks"] for t in c["turns"]]
        out[rec["sample_id"]] = {
            "sizes": sizes,
            "sess": _session_boundaries(sess_ids),
            "n": len(sess_ids),
        }
    return out


def _boundaries(sizes: list) -> list:
    """청크 크기 누적합 = cut-after 경계. 첫 시작(0)·마지막 끝은 제외."""
    b, acc = [], 0
    for s in sizes[:-1]:
        acc += s
        b.append(acc)
    return b


def _session_boundaries(sess_ids: list) -> list:
    """session_id가 바뀌는 지점 = 진짜 세션 시작(첫 turn 0은 제외)."""
    return [i for i in range(1, len(sess_ids)) if sess_ids[i] != sess_ids[i - 1]]


# ---------- 지표 ----------
def _greedy_match(A: list, B: list, tol: int) -> int:
    """A와 B를 1:1로 최대한 짝지어 맞은 개수(같은 B를 중복 사용하지 않음)."""
    B = sorted(B)
    used = [False] * len(B)
    matched = 0
    for a in sorted(A):
        best, best_d = -1, tol + 1
        for j, b in enumerate(B):
            if used[j]:
                continue
            d = abs(a - b)
            if d <= tol and d < best_d:
                best, best_d = j, d
        if best >= 0:
            used[best] = True
            matched += 1
    return matched


def _recall(pred: list, truth: list, tol: int) -> tuple:
    """truth 중 pred 근처(±tol)에 있는 비율. (맞은수, truth수) 반환(마이크로 집계용)."""
    ps = sorted(pred)
    hit = sum(1 for t in truth if any(abs(t - p) <= tol for p in ps))
    return hit, len(truth)


def _random_recall(n_pred: int, n: int, truth: list, tol: int, rng: random.Random) -> float:
    if not truth or n < 2:
        return 0.0
    k = min(n_pred, n - 1)
    vals = []
    for _ in range(RANDOM_TRIALS):
        pred = rng.sample(range(1, n), k)
        hit, tot = _recall(pred, truth, tol)
        vals.append(hit / tot)
    return sum(vals) / len(vals)


# ---------- 리포트 ----------
def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def per_file_report(name: str, F: dict, rng: random.Random) -> list:
    """지표 2·3: 세션 재현율(+random) + 청크 결. 표 라인들을 반환."""
    lines = [f"\n### [{name}] 세션 재현율 (진짜 세션경계를 맞힌 비율)",
             f"{'conv':10}{'#gold':>7}{'#sess':>7}   recall@0 / @1      random@1"]
    micro = {t: [0, 0] for t in TOLS}
    rnd_hit = rnd_tot = 0.0
    for sid in sorted(F):
        d = F[sid]
        gold = _boundaries(d["sizes"])
        rec = {}
        for t in TOLS:
            h, tot = _recall(gold, d["sess"], t)
            micro[t][0] += h
            micro[t][1] += tot
            rec[t] = h / tot if tot else 0.0
        rr = _random_recall(len(gold), d["n"], d["sess"], 1, rng)
        rnd_hit += rr * len(d["sess"])
        rnd_tot += len(d["sess"])
        lines.append(f"{sid:10}{len(gold):7}{len(d['sess']):7}   "
                     f"{fmt_pct(rec[0])} / {fmt_pct(rec[1])}     {fmt_pct(rr)}")
    micro_r = {t: (micro[t][0] / micro[t][1] if micro[t][1] else 0.0) for t in TOLS}
    lines.append(f"{'전체(micro)':10}{'':7}{'':7}   "
                 f"{fmt_pct(micro_r[0])} / {fmt_pct(micro_r[1])}     "
                 f"{fmt_pct(rnd_hit / rnd_tot if rnd_tot else 0)}")

    n_chunks = sum(len(d["sizes"]) for d in F.values())
    n_turns = sum(d["n"] for d in F.values())
    lines.append(f"\n### [{name}] 청크 결")
    lines.append(f"총 청크 {n_chunks} | 평균 {n_turns / n_chunks:.1f} turn/청크 | "
                 f"샘플 {len(F)}개 | 총 turn {n_turns}")
    return lines


def pair_report(nameA: str, A: dict, nameB: str, B: dict) -> list:
    """지표 1: 경계 겹침율(마이크로 집계). A→B, B→A, Jaccard."""
    lines = [f"\n### 경계 겹침: [{nameA}] vs [{nameB}]",
             f"{'':16}{'exact(0칸)':>12}{'±1칸':>10}"]
    agg = {t: [0, 0, 0] for t in TOLS}  # matched, |A|, |B|
    for sid in sorted(set(A) & set(B)):
        a = _boundaries(A[sid]["sizes"])
        b = _boundaries(B[sid]["sizes"])
        for t in TOLS:
            agg[t][0] += _greedy_match(a, b, t)
            agg[t][1] += len(a)
            agg[t][2] += len(b)

    def rate(num, den):
        return num / den if den else 0.0

    a2b = {t: rate(agg[t][0], agg[t][1]) for t in TOLS}
    b2a = {t: rate(agg[t][0], agg[t][2]) for t in TOLS}
    jac = {t: rate(agg[t][0], agg[t][1] + agg[t][2] - agg[t][0]) for t in TOLS}
    lines.append(f"{nameA + '→' + nameB + ' 동의':16}{fmt_pct(a2b[0]):>12}{fmt_pct(a2b[1]):>10}")
    lines.append(f"{nameB + '→' + nameA + ' 동의':16}{fmt_pct(b2a[0]):>12}{fmt_pct(b2a[1]):>10}")
    lines.append(f"{'겹침율(Jaccard)':16}{fmt_pct(jac[0]):>12}{fmt_pct(jac[1]):>10}")
    return lines


def main():
    files = sorted(CHUNK_DIR.glob("*.json"))
    if not files:
        print(f"[chunk_evaluator] {CHUNK_DIR}/ 에 청킹 json이 없습니다.")
        return
    rng = random.Random(SEED)
    loaded = {p.stem: load_file(p) for p in files}
    print(f"[chunk_evaluator] 파일 {len(files)}개: {', '.join(loaded)}")

    out = []
    for name, F in loaded.items():
        out += per_file_report(name, F, rng)

    names = list(loaded)
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                out += pair_report(names[i], loaded[names[i]], names[j], loaded[names[j]])
    else:
        out.append("\n(경계 겹침 지표는 파일 2개 이상일 때 활성화됩니다. opus 파일을 chunk/ 에 넣으세요.)")

    text = "\n".join(out)
    print(text)
    (HERE / "report.md").write_text(text, encoding="utf-8")
    print(f"\n[chunk_evaluator] 리포트 저장: {HERE / 'report.md'}")


if __name__ == "__main__":
    main()
