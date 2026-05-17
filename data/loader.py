import json
import re
from typing import List

from data.schema import QA, RawSample, Turn


# LoCoMo10 category 숫자 → 문자열 매핑
LOCOMO_CATEGORY = {
    1: "single_hop",
    2: "temporal",
    3: "open_domain",
    4: "multi_hop",
    5: "adversarial",
}


def _get_session_keys(conversation: dict) -> List[str]:
    # conversation dict에서 session_N 키만 번호 순서대로 추출
    pattern = re.compile(r"^session_(\d+)$")
    keys = [(int(m.group(1)), k) for k in conversation if (m := pattern.match(k))]
    keys.sort()
    return [k for _, k in keys]


def _parse_turns(sample_id: str, conversation: dict) -> List[Turn]:
    # session_1, session_2, ... 순서대로 flat한 Turn 리스트로 변환
    # 각 turn에 해당 session의 날짜(timestamp)와 session_id를 붙인다
    turns = []
    for session_key in _get_session_keys(conversation):
        timestamp = conversation.get(f"{session_key}_date_time")
        for raw_turn in conversation[session_key]:
            turns.append(Turn(
                turn_id=raw_turn["dia_id"],
                speaker=raw_turn["speaker"],
                content=raw_turn["text"],
                timestamp=timestamp,
                session_id=session_key,
                metadata={"source": "locomo10", "sample_id": sample_id},
            ))
    return turns


def _parse_qa(sample_id: str, raw_qa_list: list) -> List[QA]:
    # qa 리스트를 QA 리스트로 변환
    # adversarial 타입은 answer 키가 없고 adversarial_answer 키를 사용한다
    return [
        QA(
            qa_id=f"{sample_id}_{idx}",
            question=raw["question"],
            answer=raw.get("answer") or raw.get("adversarial_answer", ""),
            category=LOCOMO_CATEGORY.get(raw["category"]) if raw.get("category") is not None else None,
            metadata={"evidence": raw.get("evidence")},
        )
        for idx, raw in enumerate(raw_qa_list)
    ]


def load_locomo10(path: str, sample_idx: int = 0) -> RawSample:
    # LoCoMo10 JSON에서 단일 샘플을 읽어 RawSample로 반환
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    raw = data[sample_idx]
    sample_id = str(raw["sample_id"])

    return RawSample(
        sample_id=sample_id,
        turns=_parse_turns(sample_id, raw["conversation"]),
        qa=_parse_qa(sample_id, raw["qa"]),
        metadata={"dataset": "locomo10", "source_path": path, "sample_idx": sample_idx},
    )


def load_locomo10_all(path: str) -> List[RawSample]:
    # LoCoMo10 JSON의 모든 샘플을 RawSample 리스트로 반환
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [load_locomo10(path, idx) for idx in range(len(data))]