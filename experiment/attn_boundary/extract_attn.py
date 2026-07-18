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
  pair_sum[P, L]  : 서브행렬 원소 합   → v0(쌍평균)=sum/(n_q*n_k), v4(질량)=sum/n_q
  pair_max[P, L]  : 서브행렬 최대값    → v3(EpiCache식)
  pair_owner[P]   : 이 pair의 채점 대상 turn t 인덱스
  pair_prev[P]    : 상대 turn u 인덱스 (u==t 이면 자기자신 = intra-turn)
  pair_nk[P]      : 윈도우에 보이는 u의 토큰 수 (u 전체보다 작으면 부분 노출)
  n_tokens[N]     : turn별 전체 토큰 수 (길이 편향 검증용)
  cls_mass/sep_mass[N, L] : turn t 쿼리가 CLS/SEP에 주는 평균 질량 (sink 지표)
  scored[N]       : 채점 성립 여부 (t=0, 문맥 부족 등은 False)
  turn_ids/session_ids[N] : 라벨 유도용 메타

시각화 예시 (maps_{sample_id}.npz)
  세션 시작 turn 몇 개 + 주제 중간 turn 1개의 토큰×토큰 attention map
  (지정 층만, f16) + token→turn 매핑. Phase B에서 히트맵 육안 판단용.

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

    def __init__(self, n_turns, n_layers):
        self.owner, self.prev, self.nk = [], [], []
        self.psum, self.pmax = [], []  # 각 원소 [L]
        self.n_layers = n_layers
        self.cls_mass = np.zeros((n_turns, n_layers), dtype=np.float32)
        self.sep_mass = np.zeros((n_turns, n_layers), dtype=np.float32)
        self.scored = np.zeros(n_turns, dtype=bool)

    def add_pair(self, t, u, nk, psum_l, pmax_l):
        self.owner.append(t)
        self.prev.append(u)
        self.nk.append(nk)
        self.psum.append(psum_l)
        self.pmax.append(pmax_l)

    def to_npz_dict(self):
        return {
            "pair_owner": np.array(self.owner, dtype=np.int32),
            "pair_prev": np.array(self.prev, dtype=np.int32),
            "pair_nk": np.array(self.nk, dtype=np.int32),
            "pair_sum": np.stack(self.psum).astype(np.float32) if self.psum else np.zeros((0, self.n_layers), np.float32),
            "pair_max": np.stack(self.pmax).astype(np.float32) if self.pmax else np.zeros((0, self.n_layers), np.float32),
            "cls_mass": self.cls_mass,
            "sep_mass": self.sep_mass,
            "scored": self.scored,
        }


@torch.no_grad()
def process_batch(model, tokenizer, batch, accum, example_store):
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

        # CLS/SEP 질량 (쿼리 평균)
        accum.cls_mass[t] = q_block[:, :, 0].mean(dim=1).cpu().numpy()
        accum.sep_mass[t] = q_block[:, :, seq_len - 1].mean(dim=1).cpu().numpy()

        # 문맥의 각 과거 turn u + 자기자신(intra) 서브행렬 통계
        ctx_arr = np.array(ctx_map, dtype=np.int32)
        for u in np.unique(ctx_arr):
            cols = np.where(ctx_arr == u)[0] + 1  # +1: CLS offset
            sub = q_block[:, :, torch.as_tensor(cols, device=q_block.device)]
            accum.add_pair(
                t, int(u), len(cols),
                sub.sum(dim=(1, 2)).cpu().numpy(),
                sub.amax(dim=(1, 2)).cpu().numpy(),
            )
        # intra-turn (u == t)
        sub_self = q_block[:, :, qs:qe]
        accum.add_pair(t, t, n_q, sub_self.sum(dim=(1, 2)).cpu().numpy(),
                       sub_self.amax(dim=(1, 2)).cpu().numpy())
        accum.scored[t] = True

        # 시각화 예시 저장 (지정 turn만, 지정 층만, f16)
        if t in example_store["want"]:
            full = torch.stack(
                [layer_maps[li][bi, :seq_len, :seq_len] for li in example_store["layers"]]
            )
            example_store["maps"][f"map_t{t}"] = full.half().cpu().numpy()
            example_store["maps"][f"tok2turn_t{t}"] = np.array(
                [-1] + ctx_map + [t] * n_q + [-2], dtype=np.int32  # -1=CLS, -2=SEP
            )
    return n_layers


def pick_example_turns(sess_ids, k_boundary=2):
    """세션 시작 turn 앞쪽 k개 + 주제 중간 turn 1개 (시각화용)."""
    starts = [i for i in range(1, len(sess_ids)) if sess_ids[i] != sess_ids[i - 1]]
    want = starts[:k_boundary]
    if starts:
        mid = starts[0] + max(2, (starts[1] - starts[0]) // 2) if len(starts) > 1 else starts[0] + 3
        if mid < len(sess_ids) and mid not in want:
            want.append(mid)
    return set(want)


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
    example_layers = [int(x) for x in args.example_layers.split(",")]
    print(f"[extract_attn] device={device} max_len={max_len} "
          f"batch={args.batch_size} example_layers={example_layers}", flush=True)

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

        example_store = {
            "want": pick_example_turns(sess_ids),
            "layers": example_layers,
            "maps": {},
        }
        # 층 수는 첫 배치에서 확정되므로 임시로 config 값 사용
        n_layers_cfg = getattr(model.config, "num_hidden_layers", 12)
        accum = StatsAccum(n, n_layers_cfg)

        t0 = time.time()
        for b0 in range(0, len(windows), args.batch_size):
            batch = windows[b0 : b0 + args.batch_size]
            process_batch(model, tokenizer, batch, accum, example_store)
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
        })
        np.savez_compressed(out_file, **data)

        if example_store["maps"]:
            example_store["maps"]["example_layers"] = np.array(example_layers, dtype=np.int32)
            np.savez_compressed(out_dir / f"maps_{sample.sample_id}.npz", **example_store["maps"])

        # sanity: v4 질량 합 <= 1 (softmax 행합 1이므로 전체 질량 초과 불가)
        d = data
        if len(d["pair_owner"]):
            t_ex = int(d["pair_owner"][0])
            mask = d["pair_owner"] == t_ex
            n_q = d["n_tokens"][t_ex]
            mass = d["pair_sum"][mask].sum(axis=0) / n_q + d["cls_mass"][t_ex] + d["sep_mass"][t_ex]
            assert mass.max() <= 1.01, f"mass sanity 실패: {mass.max()}"

        print(f"[extract_attn] {sample.sample_id} 완료 | turns={n} windows={len(windows)} "
              f"pairs={len(d['pair_owner'])} examples={len(example_store['maps'])} "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"[extract_attn] 전체 완료 → {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Phase A: attention 충분통계 추출")
    p.add_argument("--data", type=str, default="/data/delta9043/datasets/locomo/locomo10.json")
    p.add_argument("--model_path", type=str, required=True, help="LLMLingua-2 모델 경로")
    p.add_argument("--max_len", type=int, default=512, help="윈도우 총 토큰 수 (CLS/SEP 포함)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0, help="앞 N개 conv만 (스모크). 0=전체")
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "data"))
    p.add_argument("--example_layers", type=str, default="0,5,11",
                   help="시각화 map 저장할 층 (쉼표 구분)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
