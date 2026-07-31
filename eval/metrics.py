"""
QA 평가 메트릭 계산 모듈.

채점 로직은 A-Mem(`A-mem/utils.py:calculate_metrics`)과 SimpleMem
(`SimpleMem/test_locomo10.py:calculate_metrics`)에 맞춘다. 두 원본은 채점 함수가
서로 동일하므로(BLEU=word_tokenize, METEOR=소문자화 없음, 빈 예측이면 전 메트릭 0)
하나의 구현으로 양쪽과 대조 가능하다.

adversarial(cat5)만 두 원본의 **채점 정답이 다르다**:
- A-Mem  : GT = `adversarial_answer`(함정 오답 후보)  — `test_advanced_robust.py:261`
- SimpleMem: GT = "Not mentioned in the conversation" — `test_locomo10.py:871-874`
어느 쪽으로 통일해도 한쪽 논문과 비교가 깨지므로 **둘 다** 낸다
(`adversarial_amem` / `adversarial_simplemem`).
"""

from collections import defaultdict
from typing import Dict, List, Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# NLTK 데이터 다운로드 (한 번만)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


# 원본 두 곳과 동일한 BLEU weight 목록(0.33 반복도 원본 그대로)
BLEU_WEIGHTS = [
    (1, 0, 0, 0),
    (0.5, 0.5, 0, 0),
    (0.33, 0.33, 0.33, 0),
    (0.25, 0.25, 0.25, 0.25),
]

# 전 메트릭 0 (예측 또는 정답이 빈 경우). 원본 `calculate_metrics` 앞부분과 동일.
ZERO_METRICS = {
    "exact_match": 0.0,
    "f1": 0.0,
    "rouge1_f": 0.0,
    "rouge2_f": 0.0,
    "rougeL_f": 0.0,
    "bleu1": 0.0,
    "bleu2": 0.0,
    "bleu3": 0.0,
    "bleu4": 0.0,
    "meteor": 0.0,
}


def calculate_rouge(prediction: str, reference: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, prediction)

    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def calculate_bleu(prediction: str, reference: str) -> Dict[str, float]:
    # 토크나이저는 반드시 word_tokenize. `.split()`이면 구두점이 토큰에 붙어
    # ("2023." vs "2023") 원본보다 BLEU가 크게 낮게 나온다(temporal −9.7%p 실측).
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    smooth = SmoothingFunction().method1

    results = {}

    for n, weights in enumerate(BLEU_WEIGHTS, start=1):
        try:
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=weights,
                smoothing_function=smooth,
            )
        except Exception:
            score = 0.0

        results[f"bleu{n}"] = score

    return results


def calculate_meteor(prediction: str, reference: str) -> float:
    # 원본은 소문자화하지 않는다(`utils.py:88`, `test_locomo10.py:320`).
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception:
        return 0.0


def simple_tokenize(text: str) -> List[str]:
    # 원본 SimpleMem/A-Mem과 동일: 소문자화 + 구두점(. , ! ?)을 공백으로 치환 후 split.
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_f1(prediction: str, reference: str) -> float:
    """Token-level F1. SimpleMem/A-Mem 원본과 동일하게 set 기반(중복 토큰 무시)."""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def calculate_pair_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """예측-정답 한 쌍의 메트릭. 원본 `calculate_metrics`와 같은 순서·같은 규칙."""
    if not prediction or not reference:
        return dict(ZERO_METRICS)

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    return {
        "exact_match": calculate_exact_match(prediction, reference),
        "f1": calculate_f1(prediction, reference),
        **calculate_rouge(prediction, reference),
        **calculate_bleu(prediction, reference),
        "meteor": calculate_meteor(prediction, reference),
    }


def calculate_bert_score(
    predictions: List[str],
    references: List[str],
) -> List[float]:
    """
    BERTScore는 배치로 계산하는 것이 효율적이라 별도 함수.
    """
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        verbose=False,
    )

    return f1.tolist()


def calculate_sbert_similarity(
    predictions: List[str],
    references: List[str],
) -> List[float]:
    """SBERT 코사인 유사도(논문 Appendix A.3 지표). 원본과 같은 all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import pytorch_cos_sim

    model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_emb = model.encode(predictions, convert_to_tensor=True)
    ref_emb = model.encode(references, convert_to_tensor=True)

    return [
        float(pytorch_cos_sim(pred_emb[i], ref_emb[i]).item())
        for i in range(len(predictions))
    ]


# adversarial 채점 기준별 카테고리 라벨
ADV_LABEL_AMEM = "adversarial_amem"
ADV_LABEL_SIMPLEMEM = "adversarial_simplemem"


def _scoring_units(results: List[dict]) -> List[dict]:
    """결과 항목을 '채점 단위'로 펼친다.

    adversarial은 기준이 두 개라 한 예측이 두 단위가 된다(라벨이 다름).
    나머지 카테고리는 1:1.
    """
    units = []

    for result in results:
        prediction = result.get("answer_pred") or ""
        category = result.get("category", "unknown")
        reference = result.get("answer_gt") or ""

        if category == "adversarial":
            gt_simplemem = result.get("answer_gt_simplemem")

            units.append({"label": ADV_LABEL_AMEM, "pred": prediction, "ref": reference})
            if gt_simplemem is not None:
                units.append({
                    "label": ADV_LABEL_SIMPLEMEM,
                    "pred": prediction,
                    "ref": gt_simplemem,
                })
        else:
            units.append({"label": category, "pred": prediction, "ref": reference})

    return units


def evaluate_results(
    results: List[dict],
    use_bertscore: bool = True,
    use_sbert: bool = True,
) -> dict:
    """
    Args:
        results:
            [
                {
                    "answer_pred": "...",
                    "answer_gt": "...",             # A-Mem 방식 GT (cat5 = adversarial_answer)
                    "answer_gt_simplemem": "...",   # adversarial만. 없으면 해당 라벨 생략
                    "category": "..."
                },
                ...
            ]

        use_bertscore / use_sbert:
            무거운 임베딩 지표 계산 여부.

    Returns:
        카테고리별 + overall 메트릭 dict. 값은 ×100.

        카테고리 라벨:
        - single_hop / multi_hop / temporal / open_domain
        - adversarial_amem      : GT = adversarial_answer (A-Mem·LoCoMo 방식)
        - adversarial_simplemem : GT = "Not mentioned in the conversation" (SimpleMem 방식)
        - overall               : 비-adversarial + adversarial_amem
        - overall_simplemem     : 비-adversarial + adversarial_simplemem
        - overall_no_adv        : 비-adversarial만
    """

    if not results:
        return {}

    units = _scoring_units(results)

    # 임베딩 지표는 전 단위를 한 번에 배치 계산
    bert_f1_list = None
    sbert_list = None

    if use_bertscore:
        try:
            bert_f1_list = calculate_bert_score(
                [u["pred"] for u in units],
                [u["ref"] for u in units],
            )
        except Exception as e:
            print(f"[metrics] BERTScore failed: {e}")
            bert_f1_list = [0.0] * len(units)

    if use_sbert:
        try:
            sbert_list = calculate_sbert_similarity(
                [u["pred"] for u in units],
                [u["ref"] for u in units],
            )
        except Exception as e:
            print(f"[metrics] SBERT similarity failed: {e}")
            sbert_list = [0.0] * len(units)

    per_unit = []

    for i, unit in enumerate(units):
        item = calculate_pair_metrics(unit["pred"], unit["ref"])

        # 빈 예측/정답이면 임베딩 지표도 0 (원본과 동일)
        empty = not unit["pred"] or not unit["ref"]

        if bert_f1_list is not None:
            item["bert_f1"] = 0.0 if empty else bert_f1_list[i]
        if sbert_list is not None:
            item["sbert_similarity"] = 0.0 if empty else sbert_list[i]

        per_unit.append({"label": unit["label"], **item})

    # 카테고리별 + overall 집계 (질문 단위 micro-average)
    by_category = defaultdict(list)

    for item in per_unit:
        label = item["label"]
        by_category[label].append(item)

        if label == ADV_LABEL_AMEM:
            by_category["overall"].append(item)
        elif label == ADV_LABEL_SIMPLEMEM:
            by_category["overall_simplemem"].append(item)
        else:
            by_category["overall"].append(item)
            by_category["overall_simplemem"].append(item)
            by_category["overall_no_adv"].append(item)

    # adversarial_simplemem이 없는 결과(구 파일)면 overall_simplemem은 의미가 없다
    if ADV_LABEL_SIMPLEMEM not in by_category:
        by_category.pop("overall_simplemem", None)

    metric_keys = [key for key in per_unit[0].keys() if key != "label"]

    aggregated = {}

    for category, items in by_category.items():
        aggregated[category] = {
            "count": len(items),
        }

        for key in metric_keys:
            avg = sum(item[key] for item in items) / len(items)
            aggregated[category][key] = round(avg * 100, 2)

    return aggregated


CATEGORY_ORDER = [
    "single_hop",
    "temporal",
    "open_domain",
    "multi_hop",
    ADV_LABEL_AMEM,
    ADV_LABEL_SIMPLEMEM,
    "overall",
    "overall_simplemem",
    "overall_no_adv",
]


def print_metrics(metrics: dict) -> None:
    """
    metrics를 보기 좋게 출력.
    """
    keys = [
        "exact_match",
        "f1",
        "rouge1_f",
        "rougeL_f",
        "bleu1",
        "bleu4",
        "meteor",
        "bert_f1",
        "sbert_similarity",
    ]

    print("\n" + "=" * 120)
    print("[metrics] Evaluation Results")
    print("=" * 120)

    header = (
        f"{'category':<22} "
        + " ".join(f"{key:>10}" for key in keys)
        + f"  {'count':>6}"
    )

    print(header)
    print("-" * len(header))

    for category in CATEGORY_ORDER:
        if category not in metrics:
            continue

        metric = metrics[category]

        row = (
            f"{category:<22} "
            + " ".join(f"{metric.get(key, 0):>10.2f}" for key in keys)
            + f"  {metric['count']:>6}"
        )

        print(row)

    print("=" * 120)
    print("adversarial_amem      = GT: adversarial_answer (A-Mem/LoCoMo 방식)")
    print("adversarial_simplemem = GT: \"Not mentioned in the conversation\" (SimpleMem 방식)")
