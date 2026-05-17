from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from core.chunker.base import BaseChunker
from data.schema import Chunk, Turn


class AttentionSimilarityChunker(BaseChunker):
    """
    LightMem 방식의 topic-boundary chunker (B1 ∩ B2).

    토큰 수가 buffer_max_tokens를 초과하지 않도록 turn을 순차적으로 버퍼에 쌓고,
    초과 직전에 현재 버퍼에 대해 B1+B2 계산 후 chunk를 생성한다.
    이후 버퍼를 비우고 다음 turn들을 계속 처리한다 (LightMem SenMemBufferManager 방식).

    B1 (Attention boundary):
        LLMLingua-2 BERT backbone의 sentence-level attention 행렬에서
        인접 sentence 간 attention 값의 local maxima 위치를 boundary로 사용.

    B2 (Embedding boundary):
        all-MiniLM-L6-v2로 각 turn을 임베딩한 뒤
        인접 turn 간 cosine similarity < threshold 위치를 boundary로 사용.
        threshold는 threshold_start부터 threshold_step씩 올려가며
        boundary가 발견될 때까지 반복 (최대 threshold_end).

    최종 boundary:
        B2 boundary 중 B1 boundary와 거리 boundary_match_distance 이내인 것만 채택.
        매칭이 하나도 없으면 B2를 그대로 사용.
        B1, B2 모두 boundary가 없으면 버퍼 전체를 하나의 chunk로.
    """

    def __init__(
        self,
        llmlingua_model_path: str,
        embedder_model_path: str,
        threshold_start: float = 0.2,
        threshold_end: float = 0.5,
        threshold_step: float = 0.05,
        boundary_match_distance: int = 3,
        attention_layers: Optional[List[int]] = None,
        buffer_max_tokens: int = 512,
        device: str = "cuda",
    ):
        if not (0 < threshold_start <= threshold_end <= 1):
            raise ValueError(
                f"threshold_start({threshold_start}) and threshold_end({threshold_end}) "
                f"must satisfy 0 < start <= end <= 1"
            )
        if threshold_step <= 0:
            raise ValueError(f"threshold_step must be > 0, got {threshold_step}")
        if boundary_match_distance < 0:
            raise ValueError(f"boundary_match_distance must be >= 0, got {boundary_match_distance}")
        if buffer_max_tokens <= 0:
            raise ValueError(f"buffer_max_tokens must be > 0, got {buffer_max_tokens}")

        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.threshold_step = threshold_step
        self.boundary_match_distance = boundary_match_distance
        # LightMem 기본값: 상위 4개 layer [8, 9, 10, 11]
        self.attention_layers = attention_layers if attention_layers is not None else [8, 9, 10, 11]
        self.buffer_max_tokens = buffer_max_tokens
        self.device = device

        # 모델은 lazy load: 첫 chunk() 호출 시 로드하여 불필요한 메모리 점유를 방지
        self._llmlingua_model = None
        self._llmlingua_tokenizer = None
        self._embedder = None
        self._llmlingua_model_path = llmlingua_model_path
        self._embedder_model_path = embedder_model_path

    def _load_models(self):
        # LLMLingua-2 BERT backbone 로드 (compression 헤드 없이 attention 추출 전용)
        if self._llmlingua_model is None:
            self._llmlingua_model = AutoModel.from_pretrained(
                self._llmlingua_model_path,
                attn_implementation="eager",
            ).to(self.device).eval()
            self._llmlingua_tokenizer = AutoTokenizer.from_pretrained(
                self._llmlingua_model_path
            )

        # all-MiniLM-L6-v2 sentence embedder 로드
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                self._embedder_model_path,
                device=self.device,
            )

    # ========== B1: Attention boundary ==========

    def _compute_attention_boundaries(self, texts: List[str]) -> List[int]:
        """
        texts를 LLMLingua-2에 통과시켜 sentence-level attention 행렬을 추출하고,
        인접 sentence 간 attention 값(outer)의 local maxima 위치를 boundary로 반환.

        outer[i] = M[i, i-1]: i번째 sentence가 직전 sentence에 주는 attention 값.
        outer[k-1] < outer[k] > outer[k+1] 인 k가 boundary.
        """
        n = len(texts)
        if n < 3:
            # 3개 미만이면 local maxima 계산 불가
            return []

        model = self._llmlingua_model
        tokenizer = self._llmlingua_tokenizer
        device = next(model.parameters()).device

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id

        # [CLS] sent1 sent2 ... sentN [SEP] 형태로 이어붙이고 각 sentence의 span 기록
        per_sent_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in texts]
        input_ids = [cls_id]
        spans = []
        cur = 1
        for ids in per_sent_tokens:
            spans.append((cur, cur + len(ids)))
            input_ids.extend(ids)
            cur += len(ids)
        input_ids.append(sep_id)
        seq_len = len(input_ids)

        # 모델 max position 초과 시 처리 불가 → boundary 없이 반환
        # 호출 측(_process_buffer)에서 토큰 수를 제어하므로 일반적으로 발생하지 않음
        max_pos = getattr(model.config, "max_position_embeddings", 512)
        if seq_len > max_pos:
            return []

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor, device=device)

        with torch.no_grad():
            outputs = model(
                input_tensor,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        # 지정한 layer들의 attention을 head 평균 → [seq_len, seq_len]
        selected = [outputs.attentions[i] for i in self.attention_layers]
        att_mean = torch.stack(selected, dim=0).mean(dim=(0, 2))[0].cpu().numpy()

        # 양 끝 k=3 토큰은 노이즈로 간주하여 유효 범위에서 제외 (LightMem 원본 로직)
        k = 3
        valid = np.ones(seq_len, dtype=bool)
        if seq_len >= 2 * k:
            valid[:k] = False
            valid[-k:] = False

        # M[i, j]: i번째 sentence가 j번째 sentence에 주는 attention 평균 (j < i)
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            i_pos = np.arange(*spans[i])
            i_pos = i_pos[valid[i_pos]]
            if i_pos.size == 0:
                continue

            row_vals = []
            for j in range(i):
                j_pos = np.arange(*spans[j])
                j_pos = j_pos[valid[j_pos]]
                if j_pos.size == 0:
                    row_vals.append(0.0)
                    continue
                sub = att_mean[np.ix_(i_pos, j_pos)]
                row_vals.append(float(sub.sum(axis=1).mean()))

            if row_vals:
                row_vals = np.array(row_vals, dtype=float)
                s = row_vals.sum()
                if s > 0:
                    row_vals /= s
                M[i, :i] = row_vals

        # local maxima 위치를 boundary로
        outer = [M[i, i - 1] for i in range(1, n)]
        return [
            k_idx for k_idx in range(1, len(outer) - 1)
            if outer[k_idx - 1] < outer[k_idx] > outer[k_idx + 1]
        ]

    # ========== B2: Embedding boundary ==========

    def _compute_embedding_boundaries(self, texts: List[str]) -> List[int]:
        """
        all-MiniLM-L6-v2로 turn별 임베딩 후 인접 cosine similarity를 계산.
        threshold_start부터 threshold_step씩 올려가며 boundary가 발견될 때까지 반복.
        """
        n = len(texts)
        if n < 2:
            return []

        embeddings = self._embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

        # 인접 turn 간 cosine similarity
        sims = []
        for i in range(n - 1):
            v1, v2 = embeddings[i], embeddings[i + 1]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            sims.append(float(np.dot(v1, v2) / denom) if denom > 0 else 0.0)

        # threshold를 단계적으로 올려가며 boundary 탐색 (엄격 → 느슨)
        threshold = self.threshold_start
        while threshold <= self.threshold_end:
            boundaries = [i + 1 for i, sim in enumerate(sims) if sim < threshold]
            if boundaries:
                return boundaries
            threshold += self.threshold_step

        return []

    # ========== Boundary 결합 ==========

    def _combine_boundaries(self, b1: List[int], b2: List[int]) -> List[int]:
        """
        B2 boundary 중 B1 boundary와 거리 boundary_match_distance 이내인 것만 채택.
        매칭이 하나도 없으면 B2를 그대로 사용 (LightMem 원본 로직).
        """
        if not b1:
            return b2
        if not b2:
            return []

        matched = [
            fb for fb in b2
            if any(abs(fb - cb) <= self.boundary_match_distance for cb in b1)
        ]
        return sorted(set(matched)) if matched else b2

    # ========== 버퍼 처리 ==========

    def _count_tokens(self, text: str) -> int:
        # 토큰 수 측정용 (special token 제외)
        return len(self._llmlingua_tokenizer.encode(text, add_special_tokens=False))

    def _process_buffer(self, buffer_turns: List[Turn], buffer_texts: List[str]) -> List[List[Turn]]:
        """
        버퍼에 쌓인 turns에 대해 B1+B2를 계산하고 boundary 기준으로 분할.
        분할된 각 구간(List[Turn])의 리스트를 반환한다.
        """
        if not buffer_turns:
            return []

        b1 = self._compute_attention_boundaries(buffer_texts)
        b2 = self._compute_embedding_boundaries(buffer_texts)
        boundaries = self._combine_boundaries(b1, b2)

        # boundary가 없으면 버퍼 전체를 하나의 segment로
        if not boundaries:
            return [list(buffer_turns)]

        segments = []
        start = 0
        for b in boundaries + [len(buffer_turns)]:
            seg = buffer_turns[start:b]
            if seg:
                segments.append(seg)
            start = b
        return segments

    # ========== chunk(): 최종 진입점 ==========

    def chunk(self, turns: List[Turn]) -> List[Chunk]:
        if not turns:
            return []

        self._load_models()

        # CLS, SEP를 위한 여유분 (2개) 확보
        max_content_tokens = self.buffer_max_tokens - 2

        buffer_turns: List[Turn] = []
        buffer_texts: List[str] = []
        buffer_token_count = 0
        all_segments: List[List[Turn]] = []

        for turn in turns:
            text = f"{turn.speaker}: {turn.content}"
            turn_tokens = self._count_tokens(text)

            # 단일 turn이 max_content_tokens 초과하면 단독 segment로 처리
            # (BERT 모델에 넣으면 truncation 됨)
            if turn_tokens > max_content_tokens:
                # 기존 버퍼를 먼저 flush
                if buffer_turns:
                    all_segments.extend(self._process_buffer(buffer_turns, buffer_texts))
                    buffer_turns = []
                    buffer_texts = []
                    buffer_token_count = 0
                # 이 turn은 단독으로 segment
                all_segments.append([turn])
                continue

            # 버퍼에 추가 시 max_content_tokens 초과하면 먼저 flush
            if buffer_token_count + turn_tokens > max_content_tokens:
                all_segments.extend(self._process_buffer(buffer_turns, buffer_texts))
                buffer_turns = []
                buffer_texts = []
                buffer_token_count = 0

            buffer_turns.append(turn)
            buffer_texts.append(text)
            buffer_token_count += turn_tokens

        # 마지막 버퍼 처리
        if buffer_turns:
            all_segments.extend(self._process_buffer(buffer_turns, buffer_texts))

        # segments를 Chunk로 변환
        chunks = []
        for chunk_id, seg_turns in enumerate(all_segments):
            seg_texts = [f"{t.speaker}: {t.content}" for t in seg_turns]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                turns=list(seg_turns),
                text="\n".join(seg_texts),
                metadata={
                    "start_turn_id": seg_turns[0].turn_id,
                    "end_turn_id": seg_turns[-1].turn_id,
                },
            ))
        return chunks