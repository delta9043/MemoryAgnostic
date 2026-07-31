"""eval/metrics.py가 A-Mem 원본 채점(`A-mem/utils.py`)과 같은 수를 내는지 확인한다.

원본과 채점이 어긋나면 논문 대조가 통째로 무효가 된다. 실제로 BLEU 토크나이저 한 줄
(`.split()` vs `nltk.word_tokenize`) 때문에 temporal BLEU-1이 9.7%p 낮게 나오고 있었다.

A-mem repo(utils.py)와 nltk 데이터가 필요하므로 **서버의 a-mem conda env에서** 돌린다:
    AMEM_PATH=/data/delta9043/repos/A-mem python test/test_metrics_parity.py
    python test/test_metrics_parity.py --results results/amem/default/result.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import calculate_pair_metrics  # noqa: E402

AMEM_PATH = os.environ.get("AMEM_PATH", "/data/delta9043/repos/A-mem")

# 두 구현이 갈릴 수 있는 지점을 고루 덮는 쌍: 구두점, 대소문자, 숫자/날짜,
# 부분일치, 완전일치, 빈 예측, 다어절.
PAIRS = [
    ("8 May 2023.", "7 May 2023"),
    ("mental health", "mental health"),
    ("Not mentioned in the conversation", "necklace"),
    ("Short answer: counseling", "Psychology, counseling certification"),
    ("She went to the LGBTQ support group.", "LGBTQ support group"),
    ("2022", "2022"),
    ("Caroline's sister, Mel!", "Mel"),
    ("", "mental health"),
    ("yes", "Yes"),
    ("a Ferrari", "Ferrari"),
    ("He raised awareness for mental health issues among youth",
     "mental health"),
    ("photography, hiking", "hiking and photography"),
]

COMPARE_KEYS = [
    "exact_match", "f1",
    "rouge1_f", "rouge2_f", "rougeL_f",
    "bleu1", "bleu2", "bleu3", "bleu4",
    "meteor",
]

TOLERANCE = 1e-4


def load_pairs_from_results(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        (row.get("answer_pred") or "", row.get("answer_gt") or "")
        for row in data.get("results", [])
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="result.json에서 (예측, 정답) 쌍을 뽑아 함께 검사")
    parser.add_argument("--limit", type=int, default=200,
                        help="--results에서 검사할 최대 쌍 수")
    args = parser.parse_args()

    if not os.path.isdir(AMEM_PATH):
        print(f"[parity] A-mem repo를 찾지 못했다: {AMEM_PATH}")
        print("[parity] AMEM_PATH 환경변수로 경로를 지정할 것.")
        return 2

    sys.path.insert(0, AMEM_PATH)
    from utils import calculate_metrics as amem_calculate_metrics  # noqa: E402

    pairs = list(PAIRS)
    if args.results:
        extra = load_pairs_from_results(args.results)[: args.limit]
        print(f"[parity] {args.results}에서 {len(extra)}쌍 추가")
        pairs.extend(extra)

    mismatches = []

    for prediction, reference in pairs:
        ours = calculate_pair_metrics(prediction, reference)
        theirs = amem_calculate_metrics(prediction, reference)

        for key in COMPARE_KEYS:
            a, b = ours[key], theirs[key]
            if abs(a - b) > TOLERANCE:
                mismatches.append((prediction, reference, key, a, b))

    print(f"[parity] {len(pairs)}쌍 × {len(COMPARE_KEYS)}지표 검사")

    if not mismatches:
        print("[parity] PASS — 원본과 모든 지표가 일치")
        return 0

    print(f"[parity] FAIL — 불일치 {len(mismatches)}건")
    for prediction, reference, key, ours_val, theirs_val in mismatches[:30]:
        print(f"  {key:<10} ours={ours_val:.6f} amem={theirs_val:.6f}  "
              f"pred={prediction[:40]!r} ref={reference[:40]!r}")
    if len(mismatches) > 30:
        print(f"  ... (총 {len(mismatches)}건)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
