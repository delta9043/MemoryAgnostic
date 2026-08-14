"""AttnChunker 검증 — 모드 2개.

① 로직 (기본, GPU/모델 불필요)
   `experiment/attn_boundary/data`의 stats_*.npz에서 뽑은 turn 점수 s[t]를 `_score()`에
   **주입**해서, 같은 점수를 줬을 때 청커가 분석 스크립트(`pick_signal.propose`)와 같은
   곳을 자르는지 본다. "신호가 맞나"가 아니라 "신호가 같으면 결과가 같나"를 본다.

       python test/test_attn_chunker.py

   torch/transformers가 없으면 가짜 모듈로 대체한다(repo 관행). 주입 때문에 실제로는
   쓰이지 않으며, 진짜 torch가 있으면 그쪽을 그대로 쓴다.

② 신호 parity (서버, 모델 필요)  ★ 이게 tau의 유효성을 결정한다
   청커가 실제로 forward해서 만든 s[t]가 npz에서 나온 값과 같은지 본다. 창 구성·토큰화·
   fp16·접는 순서 중 하나만 어긋나면 tau가 통째로 무효가 되므로, 여기서 잡는다.

       python test/test_attn_chunker.py --parity \
           --model_path /data/delta9043/models/llmlingua-2 \
           --data /data/delta9043/datasets/locomo/locomo10.json

   npz는 배치 4로 뽑았고 청커는 배치 1이라 fp16 끝자리는 다를 수 있다 → 값은 rtol로 보고
   **판정은 컷 위치 일치**로 한다.
"""

import argparse
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATS_DIR = os.path.join(ROOT, "experiment", "attn_boundary", "data")
GOLD_PATH = os.path.join(ROOT, "data", "chunked_data", "chunks_fable-5_v2.json")
DATA_PATH = os.path.join(ROOT, "data", "locomo10.json")

# AttnChunker 기본값과 같아야 한다 (어긋나면 아래 기대값이 무의미해진다)
TAU, LAYERS, TOP_K, RATIO = 0.00234, [7, 8, 9], 5, 85

# 계획서 §1 tau 표(v2 gold, conv 10개)에서 나온 기대값
EXPECT = {"cuts": 433, "f1": 0.458, "mean_size": 13.3}


def _install_fakes():
    """torch/transformers가 없는 환경용 최소 스텁 (import만 통과시키면 된다)."""
    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: _NoGrad()
    torch.long = "int64"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch

    tf = types.ModuleType("transformers")
    tf.AutoModel = tf.AutoTokenizer = object
    sys.modules["transformers"] = tf


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    _install_fakes()
    HAS_TORCH = False

from core.chunker.attn_chunker import AttnChunker  # noqa: E402
from data.locomo_loader import load_locomo10_all  # noqa: E402


# ══════════════════════════════════════════════════════════════════
# npz → turn 점수 (pick_signal의 token_values/turn_values와 같은 계산)
# ══════════════════════════════════════════════════════════════════

def turn_scores(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    order = np.argsort(d["tok_owner"], kind="stable")
    owner, topk = d["tok_owner"][order], d["tok_topk"][order]
    n_turns, n_layers = len(d["scored"]), topk.shape[1]

    per_token = np.nanmean(topk[:, :, :TOP_K], axis=2)              # A = top5
    out = np.full((n_turns, n_layers), np.nan, np.float32)
    uniq, start, count = np.unique(owner, return_index=True, return_counts=True)
    for t, s0, cnt in zip(uniq, start, count):
        g = per_token[s0:s0 + cnt]
        gs = -np.sort(-g, axis=0)                                   # B = r85
        out[int(t)] = gs[:max(1, int(np.ceil(cnt * RATIO / 100)))].mean(axis=0)
    return out[:, LAYERS].mean(axis=1)


def load_gold(path):
    """chunk JSON → {sample_id: 경계 turn 집합} (analyze.load_gold와 동일)"""
    out = {}
    for rec in json.load(open(path, encoding="utf-8")):
        acc, starts = 0, set()
        for c in rec["chunks"][:-1]:
            acc += len(c["turns"])
            starts.add(acc)
        out[rec["sample_id"]] = starts
    return out


def expected_boundaries(s):
    """pick_signal.propose(valley) + s < tau — 청커가 재현해야 할 규칙."""
    ok = np.isfinite(s)
    mid, left, right = s[1:-1], s[:-2], s[2:]
    idx = np.where(ok[1:-1] & ok[:-2] & ok[2:] & (mid < left) & (mid < right))[0] + 1
    return [int(t) for t in idx if s[t] < TAU]


def bounds_of(chunks):
    """chunk 리스트 → 경계 turn 인덱스 (각 chunk 시작, 첫 chunk 제외)"""
    acc, out = 0, []
    for c in chunks[:-1]:
        acc += len(c.turns)
        out.append(acc)
    return out


def stat_files():
    for name in sorted(os.listdir(STATS_DIR)):
        if name.startswith("stats_") and name.endswith(".npz"):
            yield name[len("stats_"):-len(".npz")], os.path.join(STATS_DIR, name)


# ══════════════════════════════════════════════════════════════════
# ① 로직 — 점수를 주입한 청커
# ══════════════════════════════════════════════════════════════════

class _FakeTok:
    cls_token_id, sep_token_id = 101, 102

    def encode(self, text, add_special_tokens=False):
        return [0]        # 내용은 안 쓴다 — 점수를 직접 주입하므로


def injected_chunker(scores):
    """s[t]를 순서대로 돌려주는 _score()를 가진 AttnChunker."""
    ch = AttnChunker(model_path="(unused)", tau=TAU, layers=LAYERS,
                     top_k=TOP_K, ratio=RATIO)
    ch._load_model = lambda: None
    ch._tokenizer = _FakeTok()
    ch._win_len = 510
    box = {"i": 0}

    def _score(tokens):
        v = float(scores[box["i"]])
        box["i"] += 1
        return v

    ch._score = _score
    return ch


def run_logic(samples, check):
    gold = load_gold(GOLD_PATH)
    n_prop = n_hit = n_gold = n_cov = 0
    sizes = []

    for sid, path in stat_files():
        s = turn_scores(path)
        turns = samples[sid].turns
        print(f"\n[{sid}] turns={len(turns)}")
        if not check(len(turns) == len(s),
                     f"{sid}: npz turn 수({len(s)}) == 데이터 turn 수({len(turns)})"):
            continue

        chunks = injected_chunker(s).chunk(turns)

        # 분석 규칙과 같은 경계를 내는가
        check(bounds_of(chunks) == expected_boundaries(s),
              f"{sid}: 경계가 propose(valley) ∧ s<tau 와 일치 ({len(chunks) - 1} cuts)")

        # 스트리밍 동치 — 하나씩 넣어도 결과가 같은가
        stream = injected_chunker(s)
        one_by_one = []
        for t in turns:
            one_by_one += stream.push(t)
        one_by_one += stream.flush()
        check([[t.turn_id for t in c.turns] for c in one_by_one]
              == [[t.turn_id for t in c.turns] for c in chunks],
              f"{sid}: chunk() == push() 반복 + flush()")

        # 무결성 — turn 수·순서 보존, 빈 chunk 없음, chunk_id 연속
        flat = [t for c in chunks for t in c.turns]
        check([t.turn_id for t in flat] == [t.turn_id for t in turns],
              f"{sid}: turn 시퀀스 완전 보존 ({len(flat)}턴)")
        check(all(c.turns for c in chunks), f"{sid}: 빈 chunk 없음")
        check([c.chunk_id for c in chunks] == list(range(len(chunks))),
              f"{sid}: chunk_id 0..{len(chunks) - 1}")

        cuts, g = bounds_of(chunks), gold[sid]
        n_prop += len(cuts)
        n_gold += len(g)
        n_hit += sum(1 for x in cuts if x in g)
        n_cov += sum(1 for x in g if x in cuts)
        sizes += [len(c.turns) for c in chunks]

    p, r = n_hit / n_prop, n_cov / n_gold
    f1 = 2 * p * r / (p + r)
    mean_size = sum(sizes) / len(sizes)
    print(f"\n[집계] cuts={n_prop} chunks={len(sizes)} P={p:.1%} R={r:.1%} "
          f"F1={f1:.3f} mean={mean_size:.1f} max={max(sizes)} (gold 경계 {n_gold})")
    check(n_prop == EXPECT["cuts"], f"컷 수 {n_prop} == {EXPECT['cuts']}")
    check(abs(f1 - EXPECT["f1"]) < 5e-4, f"F1 {f1:.3f} == {EXPECT['f1']}")
    check(abs(mean_size - EXPECT["mean_size"]) < 0.05,
          f"평균 청크 {mean_size:.1f} == {EXPECT['mean_size']}")


# ══════════════════════════════════════════════════════════════════
# ② 신호 parity — 실제 forward vs npz
# ══════════════════════════════════════════════════════════════════

def run_parity(samples, check, args):
    for sid, path in stat_files():
        if args.conv and sid != args.conv:
            continue
        ref = turn_scores(path)
        turns = samples[sid].turns
        print(f"\n[{sid}] turns={len(turns)} — forward {len(turns)}회")
        if not check(len(turns) == len(ref),
                     f"{sid}: npz turn 수({len(ref)}) == 데이터 turn 수({len(turns)})"):
            continue

        ch = AttnChunker(model_path=args.model_path, tau=TAU, layers=LAYERS,
                         top_k=TOP_K, ratio=RATIO, device=args.device)
        real_score, got = ch._score, []

        def rec(tokens):
            v = real_score(tokens)
            got.append(v)
            return v

        ch._score = rec
        chunks = ch.chunk(turns)
        mine = np.array(got, dtype=np.float64)

        # 판정 불가(nan) 자리가 같은가
        check(np.array_equal(np.isnan(mine), np.isnan(ref)),
              f"{sid}: 점수 없는 turn 위치 일치 ({int(np.isnan(ref).sum())}개)")

        both = ~np.isnan(mine) & ~np.isnan(ref)
        rel = np.abs(mine[both] - ref[both]) / np.abs(ref[both])
        print(f"  상대오차: 중앙 {np.median(rel):.2e} · 최대 {rel.max():.2e}")
        check(rel.max() < 1e-2, f"{sid}: s[t] 상대오차 최대 {rel.max():.2e} < 1e-2")

        # ★ 최종 판정 — 값이 조금 달라도 자르는 위치가 같아야 한다
        check(bounds_of(chunks) == expected_boundaries(ref),
              f"{sid}: 컷 위치가 npz 기준과 완전 일치 ({len(chunks) - 1} cuts)")


# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AttnChunker 검증")
    ap.add_argument("--parity", action="store_true",
                    help="실제 forward로 신호 parity 확인 (모델 필요)")
    ap.add_argument("--model_path", default="/data/delta9043/models/llmlingua-2")
    ap.add_argument("--data", default=DATA_PATH, help="locomo10.json 경로")
    ap.add_argument("--conv", default="", help="parity를 conv 하나만 (기본: 전체)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not os.path.isdir(STATS_DIR):
        sys.exit(f"[test] stats 디렉토리 없음: {STATS_DIR}")
    if args.parity and not HAS_TORCH:
        sys.exit("[test] --parity 는 진짜 torch/transformers가 필요하다 (서버에서 실행)")

    samples = {s.sample_id: s for s in load_locomo10_all(args.data)}
    fails = []

    def check(cond, msg):
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            fails.append(msg)
        return cond

    if args.parity:
        run_parity(samples, check, args)
    else:
        run_logic(samples, check)

    print(f"\n{'[test] 전부 통과' if not fails else f'[test] 실패 {len(fails)}건'}")
    for m in fails:
        print(f"  - {m}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
