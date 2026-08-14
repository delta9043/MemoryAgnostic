"""
attn_chunker.py — 어텐션 골짜기(local minimum) 기반 스트리밍 청커.

가설: 새 주제를 꺼내는 turn은 직전 turn을 잘 안 쳐다본다 → 어텐션이 골짜기를 만든다.
turn이 하나 도착할 때마다 인코더에 1회 통과시켜 "이번 turn이 직전 turn에 준 어텐션"을
숫자 하나(s)로 접고, s가 양옆보다 낮으면서(골짜기) tau보다 작으면 그 turn 앞에서 자른다.

    turn 도착 → 창 구성 → forward 1회 → s로 접기 → 직전 turn이 골짜기였나? → 자름/보류

s[t]는 turn t가 도착해야 생기므로 "t-1이 골짜기였다"는 판정은 t에서 난다 = 결정 지연 1턴.
창에는 t 이하만 들어가므로 미래를 보지 않는다 — 스트리밍이 성립하는 이유다.

설정값 근거는 FINDINGS.md §9 (A=top5 · B=r85 · 층 7·8·9 평균 · 세기=값 자체),
tau는 gold 경계 기준 스윕 결과(experiment/attn_boundary/analysis/pick_threshold.py).

⚠ tau는 **절대** 어텐션 값이라, 아래가 하나라도 어긋나면 무효가 된다:
   창 구성 · 토큰화 · fp16 · 접는 순서(토큰 top-k → turn 상위 r% → 층 평균).
   전부 experiment/attn_boundary/extract_attn.py와 동일하게 맞춰져 있다.
"""

import math
from collections import deque
from typing import List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from core.chunker.base import BaseChunker
from data.schema import Chunk, Turn


class AttnChunker(BaseChunker):
    """
    자른다 = ① s[t-1] > s[t] < s[t+1] (골짜기)  ∧  ② s[t] < tau

    최소 간격·최대 청크 크기 같은 보조 규칙은 두지 않는다(결과를 보고 만든 규칙이 되므로).

    파라미터
      model_path : LLMLingua-2(BERT backbone) 경로
      tau        : 임계값. 낮을수록 덜 자른다. 데이터/모델이 바뀌면 재보정 대상
      layers     : 쓸 층. 각 층에서 값을 만든 뒤 평균한다
      top_k      : (A) 토큰 하나가 직전 turn에 준 값 중 상위 몇 개를 평균할지
      ratio      : (B) turn 안에서 상위 몇 %의 토큰 값만 평균할지 (85 = 하위 15% 절사)
      max_len    : 창 총 토큰 수 상한. 모델 위치 임베딩 한도(−2)로 한 번 더 깎인다
    """

    def __init__(
        self,
        model_path: str,
        tau: float = 0.00234,
        layers: Sequence[int] = (7, 8, 9),
        top_k: int = 5,
        ratio: int = 85,
        max_len: int = 512,
        device: str = "cuda",
    ):
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not 0 < ratio <= 100:
            raise ValueError(f"ratio must be in (0, 100], got {ratio}")
        if not layers:
            raise ValueError("layers must not be empty")
        if max_len <= 4:
            raise ValueError(f"max_len must be > 4, got {max_len}")

        self.tau = tau
        self.layers = list(layers)
        self.top_k = top_k
        self.ratio = ratio
        self.max_len = max_len
        self.device = device

        # 모델은 lazy load: 첫 push()에서 로드 (attention_similarity.py와 같은 방식)
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._runtime_device = None
        self._win_len = None      # 실제 창 길이 (모델 위치 임베딩 한도 반영)

        self.reset()

    # ========== 스트림 상태 ==========

    def reset(self) -> None:
        """스트림 상태를 비운다(대화 하나가 끝나고 다음 대화를 받을 때)."""
        self._ctx = deque()          # 최근 turn들의 토큰 — 창을 채우는 재료
        self._ctx_len = 0            # 그 토큰들의 합계
        self._scores = deque(maxlen=3)   # [s(t-2), s(t-1), s(t)]
        self._buf: List[Turn] = []   # 아직 chunk로 안 나간 turn
        self._next_id = 0
        self._n_unscored = 0

    def _load_model(self) -> None:
        if self._model is not None:
            return
        device = self.device if (torch.cuda.is_available() or self.device == "cpu") else "cpu"
        model = AutoModel.from_pretrained(
            self._model_path, attn_implementation="eager",
        ).to(device).eval()
        if device.startswith("cuda"):
            model = model.half()

        n_layers = model.config.num_hidden_layers
        bad = [l for l in self.layers if not 0 <= l < n_layers]
        if bad:
            raise ValueError(f"layers out of range {bad} (model has {n_layers} layers)")

        # 위치 임베딩 여유 2칸 — extract_attn.run과 동일 계산 (512 모델이면 창은 510)
        self._win_len = min(self.max_len, model.config.max_position_embeddings - 2)
        self._runtime_device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = model

    # ========== 창 구성 + 어텐션 → 숫자 1개 ==========

    @torch.no_grad()
    def _score(self, tokens: List[int]) -> float:
        """이번 turn의 대표값 s. 창을 못 만들면 nan(= 경계 후보에서 제외)."""
        if not tokens or not self._ctx:
            return math.nan
        budget = self._win_len - 2 - len(tokens)   # CLS/SEP 뺀 과거 자리
        if budget < 1:
            return math.nan

        # 과거를 최신 turn부터 거꾸로 채운다. 자리가 모자라면 그 turn의 뒷부분만.
        # take<=0이면 더 채우지 않고 중단 — extract_attn.build_window와 동일하다.
        ctx_ids: List[int] = []
        n_prev = 0                                  # 직전 turn에서 가져온 토큰 수
        for i in range(len(self._ctx) - 1, -1, -1):
            ids = self._ctx[i]
            take = min(len(ids), budget - len(ctx_ids))
            if take <= 0:
                break
            ctx_ids = ids[-take:] + ctx_ids
            if i == len(self._ctx) - 1:
                n_prev = take
            if len(ctx_ids) >= budget:
                break
        if not ctx_ids:
            return math.nan

        tok = self._tokenizer
        input_ids = [tok.cls_token_id] + ctx_ids + tokens + [tok.sep_token_id]
        q0 = 1 + len(ctx_ids)          # 이번 turn 시작 위치
        p0 = q0 - n_prev               # 직전 turn 토큰은 창의 [p0, q0) 구간

        ids_t = torch.tensor([input_ids], dtype=torch.long, device=self._runtime_device)
        out = self._model(ids_t, attention_mask=torch.ones_like(ids_t),
                          output_attentions=True)

        # 행 = 이번 turn 토큰, 열 = 직전 turn 토큰. head 평균은 fp32로(합산 오차 방지).
        sub = torch.stack([
            out.attentions[l][0].float().mean(dim=0)[q0:q0 + len(tokens), p0:q0]
            for l in self.layers
        ])                                          # [층, 이번 turn 토큰, 직전 turn 토큰]

        # (A) 가로 접기: 토큰마다 상위 top_k 평균 (직전 turn이 짧으면 있는 만큼)
        k = min(self.top_k, sub.shape[2])
        per_token = sub.topk(k, dim=2).values.mean(dim=2)            # [층, 토큰]
        # (B) 세로 접기: 큰 것부터 상위 ratio%만 평균
        keep = max(1, math.ceil(per_token.shape[1] * self.ratio / 100))
        per_turn = per_token.sort(dim=1, descending=True).values[:, :keep].mean(dim=1)
        return float(per_turn.mean())                                # 층 평균

    # ========== 스트리밍 코어 ==========

    def push(self, turn: Turn) -> List[Chunk]:
        """turn 1개 투입 → 이번에 완성된 chunk (아직 없으면 빈 리스트)."""
        self._load_model()
        tokens = self._tokenizer.encode(f"{turn.speaker}: {turn.content}",
                                        add_special_tokens=False)
        s = self._score(tokens)
        if math.isnan(s):
            self._n_unscored += 1

        self._buf.append(turn)
        self._scores.append(s)
        self._ctx.append(tokens)
        self._ctx_len += len(tokens)
        # 창이 실제로 쓰는 만큼만 남긴다 — 결과는 그대로이고 메모리는 대화 길이와 무관해진다
        while len(self._ctx) > 1 and self._ctx_len - len(self._ctx[0]) >= self._win_len:
            self._ctx_len -= len(self._ctx.popleft())

        if not self._is_boundary():
            return []
        # 경계 turn = 버퍼의 뒤에서 두 번째. 골짜기는 연속으로 나올 수 없으므로 cut >= 1이다
        # (t-1이 골짜기면 s[t-1] < s[t]인데, t가 골짜기이려면 s[t-1] > s[t]여야 해서 모순).
        cut = len(self._buf) - 2
        done, self._buf = self._buf[:cut], self._buf[cut:]
        return [self._make_chunk(done)]

    def flush(self) -> List[Chunk]:
        """스트림 끝 — 버퍼에 남은 turn을 마지막 chunk로 내보낸다."""
        if not self._buf:
            return []
        done, self._buf = self._buf, []
        return [self._make_chunk(done)]

    def _is_boundary(self) -> bool:
        """직전 turn이 골짜기이고 tau보다 낮은가. 셋 중 하나라도 값이 없으면 판정하지 않는다."""
        if len(self._scores) < 3:
            return False
        left, mid, right = self._scores
        if math.isnan(left) or math.isnan(mid) or math.isnan(right):
            return False
        return mid < left and mid < right and mid < self.tau

    def _make_chunk(self, turns: List[Turn]) -> Chunk:
        chunk = Chunk(
            chunk_id=self._next_id,
            turns=turns,
            text="\n".join(f"{t.speaker}: {t.content}" for t in turns),
            metadata={"start_turn_id": turns[0].turn_id,
                      "end_turn_id": turns[-1].turn_id},
        )
        self._next_id += 1
        return chunk

    # ========== BaseChunker 인터페이스 ==========

    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        # 알고리즘은 push()가 전부다. 여기서는 스트림을 그대로 흉내낼 뿐이라,
        # 한꺼번에 넣든 하나씩 넣든 결과가 같다.
        self.reset()
        chunks: List[Chunk] = []
        for turn in turns:
            chunks += self.push(turn)
        chunks += self.flush()

        if chunks:
            sizes = [len(c.turns) for c in chunks]
            print(f"[attn_chunker] tau={self.tau} chunks={len(chunks)} "
                  f"avg={sum(sizes) / len(sizes):.1f} max={max(sizes)} "
                  f"unscored_turns={self._n_unscored}", flush=True)
        return chunks
