"""
extract_attn.py — Phase A: 무편향 attention 충분통계 추출 (attn_boundary 연구).

동작 원리
  모든 turn t(t>=1)를 '윈도우 맨 끝'에 놓고 채점한다:
      [CLS][ ...과거 문맥(가변, 총길이를 max_len으로 고정)... | turn t ][SEP]
  인코더(LLMLingua-2, bidirectional)를 1회 forward하여 attention을 뽑고,
  turn t의 쿼리 행 × 윈도우 내 각 과거 turn u의 키 열 서브행렬을 즉시
  '충분통계'로 접어 저장한다. 점수 함수(v0~v4)·층·극성은 여기서 정하지 않고
  Phase B(analyze_signal.py)가 통계로부터 파생/결정한다.

저장 통계 (conv별 stats_{sample_id}.npz, CSR형 flat 배열)
  pair_sum[P, L]     : 서브행렬 원소 합   → v0(쌍평균)=sum/(n_q*n_k), v4(질량)=sum/n_q
  pair_max[P, L]     : 서브행렬 최대값    → v3(EpiCache식)
  pair_meanmax[P, L] : 쿼리 토큰별 최대값의 평균 (mean-of-max, 길이 강건 연결 강도)
  pair_owner[P]      : 이 pair의 채점 대상 turn t 인덱스
  pair_prev[P]       : 상대 turn u 인덱스 (u==t 이면 자기자신 = intra-turn)
  pair_nk[P]         : 윈도우에 보이는 u의 토큰 수 (u 전체보다 작으면 부분 노출)
  n_tokens[N]        : turn별 전체 토큰 수 (길이 편향 검증용)

  직전 turn(u==t-1) 쌍만 토큰별로도 남긴다 (CSR형, T = 채점된 turn들의 토큰 총합).
  turn 대표값(top-k 평균 / 상위 r% 평균 / 중앙값 …)을 재추출 없이 파생하기 위한 원재료다.
  tok_owner[T]       : 이 토큰이 속한 turn t
  tok_sum[T, L]      : 토큰 i가 직전 turn 전체에 준 합
  tok_topk[T, L, K]  : 토큰 i가 직전 turn 토큰들에 준 값 중 상위 K개 (내림차순, 모자라면 NaN)
  cls_mass/sep_mass[N, L]   : turn t 쿼리가 CLS/SEP에 주는 평균 질량 (sink 지표)
  cls_first/sep_first[N, L] : turn t 첫 토큰이 CLS/SEP에 주는 질량 (sink 가설: 새 주제 첫 토큰)
  cls_max/sep_max[N, L]     : CLS/SEP에 가장 많이 준 토큰의 질량
  attn_entropy[N, L] : 쿼리 토큰별 전체 분포 엔트로피 평균 (log(seq_len) 정규화, 0=몰림 1=균등)
  first_ctx_mass[N, L]  : 첫 토큰이 과거 문맥 전체에 준 질량
  first_prev_mass[N, L] : 첫 토큰이 직전 turn(t-1)에 준 질량
  scored[N]          : 채점 성립 여부 (t=0, 문맥 부족 등은 False)
  turn_ids/session_ids[N] : 라벨 유도용 메타

실행 (서버 GPU):
  python experiment/attn_boundary/extract_attn.py \
      --data /data/delta9043/datasets/locomo/locomo10.json \
      --model_path <llmlingua-2 경로>
  conv 단위 재개: 기존 stats_*.npz가 있으면 스킵.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # MemoryAgnostic/
sys.path.insert(0, str(ROOT))

from data.locomo_loader import load_locomo10_all  # noqa: E402


def turn_text(turn) -> str:
    # baseline(attention_similarity.py)과 동일 포맷
    return f"{turn.speaker}: {turn.content}"


def build_window(turn_tokens, t, max_len):
    """turn t를 끝에 둔 윈도우 구성. 반환: (ctx_ids, ctx_turn_map, q_len) 또는 None."""
    q_ids = turn_tokens[t]
    ctx_budget = max_len - 2 - len(q_ids)  # CLS/SEP 제외
    if ctx_budget < 1 or t == 0:
        return None
    # 과거 turn들을 뒤에서부터 채움 (가장 최근이 turn t 바로 앞)
    ctx_ids, ctx_map = [], []  # ctx_map[i] = 해당 토큰이 속한 turn 인덱스
    for u in range(t - 1, -1, -1):
        ids = turn_tokens[u]
        take = min(len(ids), ctx_budget - len(ctx_ids))
        if take <= 0:
            break
        ctx_ids = ids[-take:] + ctx_ids  # u의 꼬리 take개
        ctx_map = [u] * take + ctx_map
        if len(ctx_ids) >= ctx_budget:
            break
    if not ctx_ids:
        return None
    return ctx_ids, ctx_map, len(q_ids)


class StatsAccum:
    """conv 1개 분량의 CSR형 통계 누적."""

    def __init__(self, n_turns, n_layers, topk):
        self.owner, self.prev, self.nk = [], [], []
        self.psum, self.pmax, self.pmm = [], [], []  # 각 원소 [L]
        # 직전 turn 쌍의 토큰별 값 (원소는 [n_q, L] / [n_q, L, K])
        self.tok_owner, self.tok_sum, self.tok_topk = [], [], []
        self.n_layers = n_layers
        self.topk = topk
        z = lambda: np.zeros((n_turns, n_layers), dtype=np.float32)
        self.cls_mass, self.sep_mass = z(), z()
        self.cls_first, self.cls_max = z(), z()
        self.sep_first, self.sep_max = z(), z()
        self.attn_entropy = z()
        self.first_ctx, self.first_prev = z(), z()
        self.scored = np.zeros(n_turns, dtype=bool)

    def add_pair(self, t, u, nk, psum_l, pmax_l, pmm_l):
        self.owner.append(t)
        self.prev.append(u)
        self.nk.append(nk)
        self.psum.append(psum_l)
        self.pmax.append(pmax_l)
        self.pmm.append(pmm_l)

    def add_prev_tokens(self, t, sum_rows, topk_rows):
        # 직전 turn(t-1)에 대한 토큰별 값. sum_rows [n_q, L] / topk_rows [n_q, L, K]
        self.tok_owner.extend([t] * len(sum_rows))
        self.tok_sum.append(sum_rows)
        self.tok_topk.append(topk_rows)

    def to_npz_dict(self):
        stack = lambda rows: (np.stack(rows).astype(np.float32) if rows
                              else np.zeros((0, self.n_layers), np.float32))
        cat = lambda rows, empty: (np.concatenate(rows).astype(np.float32) if rows
                                   else np.zeros(empty, np.float32))
        return {
            "pair_owner": np.array(self.owner, dtype=np.int32),
            "pair_prev": np.array(self.prev, dtype=np.int32),
            "pair_nk": np.array(self.nk, dtype=np.int32),
            "pair_sum": stack(self.psum),
            "pair_max": stack(self.pmax),
            "pair_meanmax": stack(self.pmm),
            "tok_owner": np.array(self.tok_owner, dtype=np.int32),
            "tok_sum": cat(self.tok_sum, (0, self.n_layers)),
            "tok_topk": cat(self.tok_topk, (0, self.n_layers, self.topk)),
            "cls_mass": self.cls_mass,
            "sep_mass": self.sep_mass,
            "cls_first": self.cls_first,
            "cls_max": self.cls_max,
            "sep_first": self.sep_first,
            "sep_max": self.sep_max,
            "attn_entropy": self.attn_entropy,
            "first_ctx_mass": self.first_ctx,
            "first_prev_mass": self.first_prev,
            "scored": self.scored,
        }


@torch.no_grad()
def process_batch(model, tokenizer, batch, accum):
    """batch = [(t, ctx_ids, ctx_map, q_ids), ...] → forward 1회 → 통계로 접기."""
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    device = next(model.parameters()).device

    # 입력 조립
    input_rows, attn_rows, metas = [], [], []
    max_seq = 0
    for t, ctx_ids, ctx_map, q_ids in batch:
        ids = [cls_id] + ctx_ids + q_ids + [sep_id]
        metas.append({
            "t": t,
            "ctx_len": len(ctx_ids),
            "ctx_map": ctx_map,
            "q_start": 1 + len(ctx_ids),          # turn t 시작 위치
            "q_end": 1 + len(ctx_ids) + len(q_ids),  # exclusive
            "seq_len": len(ids),
        })
        input_rows.append(ids)
        max_seq = max(max_seq, len(ids))
    for ids in input_rows:
        attn_rows.append([1] * len(ids) + [0] * (max_seq - len(ids)))
        ids.extend([pad_id] * (max_seq - len(ids)))

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn_rows, dtype=torch.long, device=device)
    out = model(input_ids, attention_mask=attention_mask, output_attentions=True)

    # 층별 head 평균 → [B, N, N] 리스트 (fp32로 승격해 합산 오차 방지)
    layer_maps = [a.float().mean(dim=1) for a in out.attentions]  # L × [B,N,N]
    n_layers = len(layer_maps)

    for bi, m in enumerate(metas):
        t, ctx_map = m["t"], m["ctx_map"]
        qs, qe, seq_len = m["q_start"], m["q_end"], m["seq_len"]
        n_q = qe - qs

        # [L, n_q, seq_len] : turn t 쿼리 행만
        q_block = torch.stack([lm[bi, qs:qe, :seq_len] for lm in layer_maps])

        # CLS/SEP 싱크 통계: 쿼리 평균 + 첫 토큰 + 최대 토큰 (sink 가설)
        cls_col = q_block[:, :, 0]                     # [L, n_q]
        sep_col = q_block[:, :, seq_len - 1]
        accum.cls_mass[t] = cls_col.mean(dim=1).cpu().numpy()
        accum.cls_first[t] = cls_col[:, 0].cpu().numpy()
        accum.cls_max[t] = cls_col.amax(dim=1).cpu().numpy()
        accum.sep_mass[t] = sep_col.mean(dim=1).cpu().numpy()
        accum.sep_first[t] = sep_col[:, 0].cpu().numpy()
        accum.sep_max[t] = sep_col.amax(dim=1).cpu().numpy()

        # 토큰별 전체 분포 엔트로피 (log(seq_len) 정규화 → 0=몰림, 1=균등)
        p = q_block
        ent = -torch.where(p > 0, p * p.log(), torch.zeros_like(p)).sum(dim=2)  # [L, n_q]
        accum.attn_entropy[t] = (ent.mean(dim=1) / np.log(seq_len)).cpu().numpy()

        # 첫 토큰의 과거 예산 (sink의 반대면: 새 주제 첫 토큰은 과거를 안 보나)
        first_row = q_block[:, 0, :]                   # [L, seq_len]
        ctx_len = m["ctx_len"]
        accum.first_ctx[t] = first_row[:, 1:1 + ctx_len].sum(dim=1).cpu().numpy()

        # 문맥의 각 과거 turn u + 자기자신(intra) 서브행렬 통계
        ctx_arr = np.array(ctx_map, dtype=np.int32)
        for u in np.unique(ctx_arr):
            cols = torch.as_tensor(np.where(ctx_arr == u)[0] + 1,  # +1: CLS offset
                                   device=q_block.device)
            sub = q_block[:, :, cols]
            accum.add_pair(
                t, int(u), len(cols),
                sub.sum(dim=(1, 2)).cpu().numpy(),
                sub.amax(dim=(1, 2)).cpu().numpy(),
                sub.amax(dim=2).mean(dim=1).cpu().numpy(),  # mean-of-max
            )
            if u == t - 1:
                accum.first_prev[t] = sub[:, 0, :].sum(dim=1).cpu().numpy()
                # 토큰별 값 보존: 집계 방식(top-k, 상위 r%, 중앙값…)을 분석 단계에서 고르려면
                # 여기서 접으면 안 된다. sub = [L, n_q, n_k].
                k_eff = min(accum.topk, sub.shape[2])
                top = sub.topk(k_eff, dim=2).values.permute(1, 0, 2).cpu().numpy()
                if k_eff < accum.topk:  # 직전 turn이 K보다 짧으면 뒤는 NaN
                    padded = np.full((top.shape[0], top.shape[1], accum.topk),
                                     np.nan, np.float32)
                    padded[:, :, :k_eff] = top
                    top = padded
                accum.add_prev_tokens(t, sub.sum(dim=2).cpu().numpy().T, top)
        # intra-turn (u == t)
        sub_self = q_block[:, :, qs:qe]
        accum.add_pair(t, t, n_q, sub_self.sum(dim=(1, 2)).cpu().numpy(),
                       sub_self.amax(dim=(1, 2)).cpu().numpy(),
                       sub_self.amax(dim=2).mean(dim=1).cpu().numpy())
        accum.scored[t] = True
    return n_layers


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract_attn] 데이터 로드: {args.data}", flush=True)
    samples = load_locomo10_all(args.data)
    if args.limit:
        samples = samples[: args.limit]

    print(f"[extract_attn] 모델 로드: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, attn_implementation="eager")
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = model.to(device).eval()
    if device.startswith("cuda"):
        model = model.half()

    max_pos = getattr(model.config, "max_position_embeddings", 512)
    max_len = min(args.max_len, max_pos - 2)  # 위치 임베딩 여유
    print(f"[extract_attn] device={device} max_len={max_len} "
          f"batch={args.batch_size}", flush=True)

    for si, sample in enumerate(samples):
        out_file = out_dir / f"stats_{sample.sample_id}.npz"
        if out_file.exists():
            print(f"[extract_attn] {sample.sample_id} 스킵 (기존 파일)", flush=True)
            continue

        turns = sample.turns
        n = len(turns)
        turn_tokens = [
            tokenizer.encode(turn_text(t), add_special_tokens=False) for t in turns
        ]
        sess_ids = [t.session_id for t in turns]

        # 윈도우 목록 (t>=1, 문맥 성립하는 것만)
        windows = []
        for t in range(1, n):
            w = build_window(turn_tokens, t, max_len)
            if w is not None:
                ctx_ids, ctx_map, _ = w
                windows.append((t, ctx_ids, ctx_map, turn_tokens[t]))

        # 층 수는 첫 배치에서 확정되므로 임시로 config 값 사용
        n_layers_cfg = getattr(model.config, "num_hidden_layers", 12)
        accum = StatsAccum(n, n_layers_cfg, args.topk)

        t0 = time.time()
        for b0 in range(0, len(windows), args.batch_size):
            batch = windows[b0 : b0 + args.batch_size]
            process_batch(model, tokenizer, batch, accum)
            if (b0 // args.batch_size) % 20 == 0:
                done = min(b0 + args.batch_size, len(windows))
                print(f"[extract_attn] {sample.sample_id} {done}/{len(windows)} windows "
                      f"({time.time() - t0:.0f}s)", flush=True)

        data = accum.to_npz_dict()
        data.update({
            "n_tokens": np.array([len(x) for x in turn_tokens], dtype=np.int32),
            "turn_ids": np.array([t.turn_id for t in turns]),
            "session_ids": np.array([str(s) for s in sess_ids]),
            "meta_model": np.array(args.model_path),
            "meta_max_len": np.array(max_len),
            "meta_topk": np.array(args.topk),
        })
        np.savez_compressed(out_file, **data)

        # sanity: v4 질량 합 <= 1 (softmax 행합 1이므로 전체 질량 초과 불가)
        d = data
        if len(d["pair_owner"]):
            t_ex = int(d["pair_owner"][0])
            mask = d["pair_owner"] == t_ex
            n_q = d["n_tokens"][t_ex]
            mass = d["pair_sum"][mask].sum(axis=0) / n_q + d["cls_mass"][t_ex] + d["sep_mass"][t_ex]
            assert mass.max() <= 1.01, f"mass sanity 실패: {mass.max()}"

            # sanity: 토큰별 저장 == 기존 집계 (같은 값을 두 경로로 계산한 것이라 일치해야 한다)
            prev = mask & (d["pair_prev"] == t_ex - 1)
            rows = d["tok_owner"] == t_ex
            if prev.any() and rows.any():
                assert np.allclose(d["tok_sum"][rows].sum(axis=0),
                                   d["pair_sum"][prev][0], rtol=1e-3), "tok_sum 불일치"
                assert np.allclose(np.nanmean(d["tok_topk"][rows][:, :, 0], axis=0),
                                   d["pair_meanmax"][prev][0], rtol=1e-3), "tok_topk 불일치"

        print(f"[extract_attn] {sample.sample_id} 완료 | turns={n} windows={len(windows)} "
              f"pairs={len(d['pair_owner'])} tokens={len(d['tok_owner'])} "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"[extract_attn] 전체 완료 → {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Phase A: attention 충분통계 추출")
    p.add_argument("--data", type=str, default="/data/delta9043/datasets/locomo/locomo10.json")
    p.add_argument("--model_path", type=str, required=True, help="LLMLingua-2 모델 경로")
    p.add_argument("--max_len", type=int, default=512, help="윈도우 총 토큰 수 (CLS/SEP 포함)")
    p.add_argument("--topk", type=int, default=10,
                   help="K: 토큰별로 남길 상위 attention 개수. 분석 때 k<=K를 자유 선택")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0, help="앞 N개 conv만 (스모크). 0=전체")
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "data"))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
